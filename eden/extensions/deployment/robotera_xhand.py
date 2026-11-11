"""RobotEra XHand real-robot deployment backend."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import time

import numpy as np

import eden as en
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    bind_deployment,
    resolve_dof_index_map,
)
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.options.robots.robotera_xhand import _XHand1

try:
    from xhand_controller import xhand_control
except ImportError:
    xhand_control = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions


# Maps each URDF joint name (right or left) to its 0-based finger command index in the
# xhand_control API.  Both hands share the same ordinal scheme (0-11); the hand side is
# selected at the device level via ``hand_id``.
XHAND_FINGER_IDX: dict[str, int] = {
    # Right hand
    "right_hand_thumb_bend_joint": 0,
    "right_hand_thumb_rota_joint1": 1,
    "right_hand_thumb_rota_joint2": 2,
    "right_hand_index_bend_joint": 3,
    "right_hand_index_joint1": 4,
    "right_hand_index_joint2": 5,
    "right_hand_mid_joint1": 6,
    "right_hand_mid_joint2": 7,
    "right_hand_ring_joint1": 8,
    "right_hand_ring_joint2": 9,
    "right_hand_pinky_joint1": 10,
    "right_hand_pinky_joint2": 11,
    # Left hand
    "left_hand_thumb_bend_joint": 0,
    "left_hand_thumb_rota_joint1": 1,
    "left_hand_thumb_rota_joint2": 2,
    "left_hand_index_bend_joint": 3,
    "left_hand_index_joint1": 4,
    "left_hand_index_joint2": 5,
    "left_hand_mid_joint1": 6,
    "left_hand_mid_joint2": 7,
    "left_hand_ring_joint1": 8,
    "left_hand_ring_joint2": 9,
    "left_hand_pinky_joint1": 10,
    "left_hand_pinky_joint2": 11,
}

_NUM_JOINTS = 12
# Fingers {2, 5, 7, 9, 11} carry tip sensors; sensor_data is ordered
# by their ascending finger-ID index (sensor_data[0] -> finger 2, etc.).
_SENSOR_FINGERS = (2, 5, 7, 9, 11)


@DEPLOYMENT_REGISTRY.register()
class RoboTeraXHandDeployment(DeploymentBase):
    """Deployment backend for the RoboTera XHand1 dexterous hand.

    Supports both EtherCAT and RS485 communication protocols.  For RS485 you
    must also set ``serial_port`` and ``baud_rate`` to match your hardware.

    The hand is treated as a fixed-base robot: ``read_state`` returns an
    identity quaternion and zero angular velocity because the hand has no IMU.

    **Joint velocity is a finite difference.** The SDK's ``FingerState_t`` has a position and
    no velocity, so ``read_state`` differences consecutive positions over the wall-clock time
    between the two reads (monotonic clock). The first read after ``connect`` has no previous
    sample and reports zero. It is a control-period average, not the instantaneous value the
    simulator reports, so expect it to be noisier and a period late.

    **The hand runs its own PD, with the gains below -- not the command's.** ``finger_command``
    carries ``kp`` / ``ki`` / ``kd`` as ``int16`` values in firmware units, which have no known
    mapping to the N·m/rad of :attr:`RobotCommand.dofs_kp` (the ``xhand1`` sysid example fits the
    effective sim-side gains for that reason). So ``send_payload`` forwards positions only:
    ``RobotCommand.dofs_kp`` / ``dofs_kd`` / ``dofs_vel`` / ``dofs_torque`` are not sent, and the
    firmware gains are set once at ``connect`` from ``dofs_kp`` / ``dofs_ki`` / ``dofs_kd``.
    A payload with ``kp = 0`` therefore changes nothing on the hand. The two base options that
    work through command gains, ``deploy_kp_kd_scale`` and ``disable_joints``, are refused at
    construction rather than accepted and ignored. Change these options instead.
    ``connect`` logs the gains it programs.

    Parameters
    ----------
    protocol : {"RS485", "EtherCAT"}
        Bus to open. RS485 also needs ``serial_port`` and ``baud_rate``.
    serial_port : str
        RS485 device path.
    baud_rate : int
        RS485 baud rate.
    hand_id : int
        Device ID of the hand to drive. ``0`` selects the first enumerated hand.
    dofs_kp, dofs_ki, dofs_kd : int
        Firmware PD(I) gains programmed on every finger, in firmware units.
    tor_max : int
        Firmware torque limit on every finger, in firmware units.
    finger_mode : int
        Firmware finger mode: ``0`` powerless, ``3`` position (default), ``5`` powerful.
    crc_retries : int
        Attempts per ``read_state`` / ``send_command`` before a CRC failure raises.
    """

    # Firmware PD on gains programmed at connect: a command's kp/kd never reach the hand.
    sends_command_gains: ClassVar[bool] = False

    protocol: Literal["RS485", "EtherCAT"] = "RS485"
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 3_000_000
    hand_id: int = 0
    dofs_kp: int = 50
    dofs_ki: int = 0
    dofs_kd: int = 25
    tor_max: int = 300
    # 0: powerless  3: position (default)  5: powerful
    finger_mode: int = 3
    crc_retries: int = 3

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)

        self._device: Any | None = None
        self._hand_cmd: Any | None = None
        self._active_hand_id: int = self.hand_id
        # Previous position sample and its monotonic stamp, for the finite-difference velocity.
        self._prev_dofs_pos: np.ndarray | None = None
        self._prev_read_t: float | None = None

    def _resolve_options(self) -> None:
        """Resolve each deployment DOF's finger index (see :meth:`DeploymentBase._resolve_options`)."""
        super()._resolve_options()
        self._dof_index = resolve_dof_index_map(
            self.dofs_name, XHAND_FINGER_IDX, owner=type(self).__name__, what="finger index"
        )
        self._reject_unforwarded_gain_options()

    def _reject_unforwarded_gain_options(self) -> None:
        """Refuse the base options that act through ``RobotCommand`` gains, which never reach the hand.

        Both are safety knobs an operator reaches for on a first run, and both would otherwise be
        accepted and do nothing: the fingers would servo at full firmware gain while the operator
        believes them softened, or limp.
        """
        owner = type(self).__name__
        firmware = f"dofs_kp={self.dofs_kp}, dofs_ki={self.dofs_ki}, dofs_kd={self.dofs_kd}"
        if self.deploy_kp_kd_scale != 1.0:
            soft_kp = max(round(self.dofs_kp * self.deploy_kp_kd_scale), 0)
            soft_kd = max(round(self.dofs_kd * self.deploy_kp_kd_scale), 0)
            raise ValueError(
                f"{owner}: deploy_kp_kd_scale={self.deploy_kp_kd_scale:g} cannot reach the hand. The XHand runs "
                f"its own PD on firmware gains programmed once at connect ({firmware}, int16 firmware units with "
                f"no known mapping from N·m/rad), and send_payload forwards joint positions only. To soften the "
                f"hand, lower the firmware gains instead: {owner}.configure(dofs_kp={soft_kp}, dofs_kd={soft_kd}). "
                f"Leave deploy_kp_kd_scale at 1.0; setting it would only change gains in a command the hand never "
                f"reads."
            )
        if self.disabled_dofs:
            raise ValueError(
                f"{owner}: disable_joints matched {list(self.disabled_dofs)}, but the XHand cannot leave a single "
                f"finger limp. disable_joints works by sending kp=kd=0, and the hand ignores command gains, so it "
                f"would keep servoing those fingers to the policy's targets at its firmware gains ({firmware}). "
                f"Remove disable_joints; this backend has no per-finger unpowered mode."
            )

    # ------------------------------------------------------------------
    # DeploymentBase interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if xhand_control is None:
            raise RuntimeError(
                "xhand_controller package is not installed. Install it before using RoboTeraXHandDeployment."
            )

        self._device = xhand_control.XHandControl()

        if self.protocol == "EtherCAT":
            ports = self._device.enumerate_devices("EtherCAT")
            if not ports:
                raise RuntimeError("No EtherCAT XHand devices found.")
            rsp = self._device.open_ethercat(ports[0])
        elif self.protocol == "RS485":
            rsp = self._device.open_serial(self.serial_port, self.baud_rate)
        else:
            raise ValueError(f"Unknown protocol '{self.protocol}'. Expected 'RS485' or 'EtherCAT'.")

        if rsp.error_code != 0:
            raise RuntimeError(f"Failed to open XHand device: {rsp.error_message}")

        # Wait until the device enumerates at least one hand, then verify that the
        # configured hand_id is present.  When hand_id is 0 (default) we auto-select
        # the first discovered hand for single-hand setups.
        start_time = time.time()
        while True:
            ids = self._device.list_hands_id()
            if ids:
                if self.hand_id in ids:
                    self._active_hand_id = self.hand_id
                    break
                if self.hand_id == 0:
                    self._active_hand_id = ids[0]
                    break
                raise RuntimeError(f"Configured hand_id {self.hand_id} not found. Available hand IDs: {ids}")
            if time.time() - start_time > self.connect_timeout_s:
                raise RuntimeError("Timed out waiting for XHand device to enumerate hand IDs.")
            time.sleep(0.05)

        self._hand_cmd = xhand_control.HandCommand_t()
        for i in range(_NUM_JOINTS):
            fc = self._hand_cmd.finger_command[i]
            fc.id = i
            fc.kp = self.dofs_kp
            fc.ki = self.dofs_ki
            fc.kd = self.dofs_kd
            fc.position = 0.0
            fc.tor_max = self.tor_max
            fc.mode = self.finger_mode

        # A fresh link has no previous sample: the first read reports zero velocity.
        self._prev_dofs_pos = None
        self._prev_read_t = None
        en.logger.info(
            f"[deploy] XHand firmware PD: kp={self.dofs_kp} ki={self.dofs_ki} kd={self.dofs_kd} "
            f"tor_max={self.tor_max} (firmware units, on every finger). RobotCommand gains are not "
            "forwarded; set the dofs_kp / dofs_ki / dofs_kd deployment options to change these."
        )

    def close(self) -> None:
        self._teardown_common()
        if self._device is not None:
            self._set_finger_mode(0)  # release to powerless before disconnect
        self._device = None
        self._hand_cmd = None

    def read_state(self) -> RobotState:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")

        for attempt in range(self.crc_retries):
            err, state = self._device.read_state(self._active_hand_id, True)
            if err.error_code == 0:
                break
            en.logger.warning(f"read_state CRC error (attempt {attempt + 1}/{self.crc_retries}): {err.error_message}")
        else:
            raise RuntimeError(f"Failed to read XHand state after {self.crc_retries} attempts: {err.error_message}")
        read_t = time.perf_counter()

        dofs_pos = np.zeros(self.num_dofs, dtype=np.float64)
        dofs_torque = np.zeros(self.num_dofs, dtype=np.float64)

        joint_diagnostics: list[dict] = []
        for i, finger_idx in enumerate(self._dof_index):
            finger = state.finger_state[finger_idx]
            dofs_pos[i] = finger.position
            dofs_torque[i] = finger.torque
            joint_diagnostics.append(
                {
                    "id": finger.id,
                    "raw_position": finger.raw_position,
                    "sensor_id": finger.sensor_id,
                    "temperature": finger.temperature,
                    "commboard_err": finger.commboard_err,
                    "jonitboard_err": finger.jonitboard_err,
                    "tipboard_err": finger.tipboard_err,
                }
            )

        # Collect fingertip tactile data for the five sensor-equipped fingers.
        fingertip_sensors: dict[int, dict] = {}
        for sensor_idx, finger_id in enumerate(_SENSOR_FINGERS):
            sensor_data = state.sensor_data[sensor_idx]
            fingertip_sensors[finger_id] = {
                "calc_pressure": np.array(
                    [
                        sensor_data.calc_force.fx,
                        sensor_data.calc_force.fy,
                        sensor_data.calc_force.fz,
                    ]
                ),
                "raw_pressure": np.array([[force.fx, force.fy, force.fz] for force in sensor_data.raw_force]),
                "sensor_temperature": sensor_data.calc_temperature,
            }

        dofs_vel = self._finite_difference_dofs_vel(dofs_pos, read_t)

        return RobotState(
            stamp=time.time(),
            # The XHand1 is a fixed-base end-effector with no on-board IMU.
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # identity (wxyz)
            base_ang_vel=np.zeros(3),
            base_lin_acc=np.zeros(3),
            dofs_pos=dofs_pos,
            dofs_vel=dofs_vel,
            dofs_torque=dofs_torque,
            extra={
                "fingertip_sensors": fingertip_sensors,
                "joint_diagnostics": joint_diagnostics,
            },
        )

    def send_payload(self, payload: RobotCommand) -> None:
        if self._device is None or self._hand_cmd is None:
            raise RuntimeError("XHand deployment is not connected.")

        # Low-pass every path (hold / INIT / policy); the payload carries what is sent.
        payload.dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))
        for i, finger_idx in enumerate(self._dof_index):
            fc = self._hand_cmd.finger_command[finger_idx]
            fc.id = finger_idx
            fc.position = float(payload.dofs_pos[i])

        for attempt in range(self.crc_retries):
            err = self._device.send_command(self._active_hand_id, self._hand_cmd)
            if err.error_code == 0:
                break
            en.logger.warning(f"send_command CRC error (attempt {attempt + 1}/{self.crc_retries}): {err.error_message}")
        else:
            raise RuntimeError(f"Failed to send XHand command after {self.crc_retries} attempts: {err.error_message}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finite_difference_dofs_vel(self, dofs_pos: np.ndarray, read_t: float) -> np.ndarray:
        """Joint velocity from this read and the previous one (the SDK reports none).

        Zero when there is no previous sample -- the first read after ``connect`` -- or when the
        clock has not advanced, rather than a division by zero.

        Parameters
        ----------
        dofs_pos : np.ndarray
            Joint positions just read, in deployment DOF order.
        read_t : float
            ``time.perf_counter()`` when the read returned.

        Returns
        -------
        np.ndarray
            Joint velocities in rad/s, in deployment DOF order.
        """
        prev_pos, prev_t = self._prev_dofs_pos, self._prev_read_t
        self._prev_dofs_pos, self._prev_read_t = dofs_pos.copy(), read_t
        if prev_pos is None or prev_t is None or read_t <= prev_t:
            return np.zeros_like(dofs_pos)
        return (dofs_pos - prev_pos) / (read_t - prev_t)

    def _set_finger_mode(self, mode: int) -> None:
        """Broadcast a mode change to all fingers and send immediately."""
        if self._device is None or self._hand_cmd is None:
            return
        for i in range(_NUM_JOINTS):
            self._hand_cmd.finger_command[i].mode = mode
        err = self._device.send_command(self._active_hand_id, self._hand_cmd)
        if err.error_code != 0:
            # Called on the way down from ``close``, so warn rather than raise -- but say so: a
            # failed release to powerless leaves the fingers holding their last target.
            en.logger.warning(f"[deploy] XHand finger mode change to {mode} failed: {err.error_message}")
        # Restore the configured operating mode so the next send_payload is correct.
        for i in range(_NUM_JOINTS):
            self._hand_cmd.finger_command[i].mode = self.finger_mode

    def reset_sensor(self, sensor_id: int) -> None:
        """Reset a fingertip sensor by its ID."""
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err = self._device.reset_sensor(self._active_hand_id, sensor_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to reset sensor {sensor_id}: {err.error_message}")

    def set_hand_id(self, new_id: int) -> None:
        """Change the hand ID on the device."""
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err = self._device.set_hand_id(self._active_hand_id, new_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to set hand ID from {self._active_hand_id} to {new_id}: {err.error_message}")
        self._active_hand_id = new_id

    def _sdk_version(self) -> str:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        return self._device.get_sdk_version()

    def _serial_number(self) -> str:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err, serial_number = self._device.get_serial_number(self._active_hand_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to get serial number: {err.error_message}")
        return serial_number

    def _hand_type(self) -> str:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err, hand_type = self._device.get_hand_type(self._active_hand_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to get hand type: {err.error_message}")
        return hand_type


# Both hands (XHand1_L / XHand1_R), alone or as the hand of a composite rig (FrankaXHand, ...).
bind_deployment(_XHand1, "robo_tera_x_hand_deployment")
