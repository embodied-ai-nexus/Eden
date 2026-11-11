"""Unitree Go1 real-robot deployment backend (legacy ``robot_interface`` UDP SDK).

Mirrors the structure of :mod:`eden.extensions.deployment.unitree_go2` — same
registry/binding, ``_dof_index`` Eden→SDK map, and Eden-order
``RobotState``/``RobotCommand`` contract — but the transport is the *legacy*
``unitree_legged_sdk`` (synchronous ``UDP.Recv``/``Send`` + ``Safety``) rather
than the DDS ``unitree_sdk2py`` stack Go2/G1 use.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import time

import numpy as np

import eden as en

from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    SoftShutdownMixin,
    bind_deployment,
    resolve_dof_index_map,
)
from eden.options.robots.unitree_go1 import Go1
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.extensions.deployment.utils.unitree_go1_sdk import (
    LOWLEVEL,
    ensure_go1_sdk,
    robot_interface as sdk,
)

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions


# UDP command / link-priming period (s) — the legacy SDK's 500 Hz low-level cadence.
_PRIME_DT_S = 0.002

# Legacy SDK ``MotorState.mode`` value when a motor is thermally derating itself.
_OVERHEAT_MODE = 0x08

# Go1 legacy-SDK motor index order: leg-major FR, FL, RR, RL, each [hip, thigh, calf].
# The base resolves ``self.dofs_name`` in Eden order; ``self._dof_index`` scatters through
# this map, so it is the ONLY place Go1's SDK order appears.
#
# ``GO2_JOINT_IDX`` currently holds the same twelve entries. That is kept deliberately
# separate rather than shared: a robot's SDK motor layout is a per-robot fact tied to its
# firmware and SDK, and folding the two together would let a future Go2 change silently
# rewrite Go1's mapping. Equal today is not the same as guaranteed equal.
GO1_JOINT_IDX: dict[str, int] = {
    "FR_hip_joint": 0,
    "FR_thigh_joint": 1,
    "FR_calf_joint": 2,
    "FL_hip_joint": 3,
    "FL_thigh_joint": 4,
    "FL_calf_joint": 5,
    "RR_hip_joint": 6,
    "RR_thigh_joint": 7,
    "RR_calf_joint": 8,
    "RL_hip_joint": 9,
    "RL_thigh_joint": 10,
    "RL_calf_joint": 11,
}


@DEPLOYMENT_REGISTRY.register()
class UnitreeGo1Deployment(SoftShutdownMixin, DeploymentBase):
    # Robot control board endpoint + local UDP port (SDK example defaults).
    robot_ip: str = "192.168.123.10"
    robot_port: int = 8007
    local_port: int = 8080
    # MotorCmd servo mode (``0x0A`` == Servo in the legacy SDK's comm.h).
    motor_mode: int = 0x0A
    # ``Safety.PowerProtect`` factor: 1..10 == 10%..100% of rated power. Default 10
    # (100%, no derating): lower factors clamp the calf torque a locomotion gait
    # needs — factor 7 froze the move_forward Go1 mid-stance (PD demand ~30 N·m
    # derated to ~5), and the SDK examples' value of 1 (10%) starves it entirely
    # and trips "Error: Power Protection". Lower it via --power_protect_factor for
    # cautious bring-up with the robot supported.
    power_protect_factor: int = 10
    # The INIT stand-up ramp's options are DeploymentBase's, and close()'s soft shutdown's are
    # SoftShutdownMixin's.
    _SOFT_SHUTDOWN_LABEL = "Go1 soft shutdown"
    # Warn (once per joint, until it recovers) when a motor reports the SDK's Overheat
    # mode or reaches this temperature (°C). A derating motor makes its leg go weak —
    # a common cause of the robot freezing/sagging mid-deploy (see "Monitoring and diagnostics"
    # in ``docs/guides/deployment.md``). Also recorded
    # into --debug_record as ``x_motor_temp`` / ``x_motor_mode``.
    motor_temp_warn_c: int = 80

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)

        self._udp = None
        self._safe = None
        self._cmd = None
        self._state = None

        # Per-joint thermal-warning dedupe (warn once until the motor recovers).
        self._thermal_warned: set[str] = set()

    def _resolve_options(self) -> None:
        """Resolve each deployment DOF's SDK motor index (see :meth:`DeploymentBase._resolve_options`)."""
        super()._resolve_options()
        self._dof_index = resolve_dof_index_map(self.dofs_name, GO1_JOINT_IDX, owner=type(self).__name__)
        self._num_full_dof = len(GO1_JOINT_IDX)

    @property
    def _connected(self) -> bool:
        """Whether :meth:`connect` has opened the UDP link and :meth:`close` has not dropped it."""
        return self._udp is not None and self._safe is not None and self._cmd is not None and self._state is not None

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Unitree Go1 deployment is not connected.")

    def _refresh_state(self) -> None:
        """Pull the latest low-state datagram into ``self._state`` (UDP is synchronous)."""
        self._udp.Recv()
        self._udp.GetRecv(self._state)

    def connect(self) -> None:
        ensure_go1_sdk()

        self._udp = sdk.UDP(LOWLEVEL, self.local_port, self.robot_ip, self.robot_port)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
        self._cmd = sdk.LowCmd()
        self._state = sdk.LowState()
        self._udp.InitCmdData(self._cmd)
        self._init_low_cmd()

        # Prime the bidirectional UDP link and wait for the first valid low-state
        # packet. The robot only streams state back to a client it has already
        # heard from, so we must SEND while polling: the ``_init_low_cmd`` command
        # is limp (kp=kd=tau=0, so no torque / no motion) and safe to send before
        # we have any state. A Recv-only wait hangs forever whenever the robot
        # isn't already streaming (e.g. after a previous client closed).
        # ``tick`` (ms) stays 0 until the control board is heard from.
        start_time = time.time()
        while True:
            self._udp.SetSend(self._cmd)
            self._udp.Send()
            self._refresh_state()
            if self._state.tick != 0:
                break
            if time.time() - start_time > self.connect_timeout_s:
                raise RuntimeError(
                    "Timed out waiting for Unitree Go1 low state. Check the robot is powered, "
                    "in low-level mode (L2+A → L2+B → L1+L2+start on the remote), and reachable "
                    f"at {self.robot_ip}:{self.robot_port} from local UDP port {self.local_port}."
                )
            time.sleep(_PRIME_DT_S)

    def close(self) -> None:
        # Flush the debug recording and release the operator input device, then settle to a
        # damping state before dropping the handles, so the robot doesn't hold its last stiff
        # target until the onboard watchdog trips. The legacy SDK has no explicit teardown;
        # nulling the handles frees the UDP socket for a clean reconnect.
        self._teardown_common()
        self._soft_shutdown()
        self._udp = None
        self._safe = None
        self._cmd = None
        self._state = None

    def _shutdown_hold_pos(self) -> np.ndarray | None:
        """Hold where the robot is now (see :meth:`SoftShutdownMixin._shutdown_hold_pos`); nothing once disconnected."""
        if not self._connected:
            return None
        return self.read_state().dofs_pos

    def read_state(self) -> RobotState:
        self._require_connected()
        self._refresh_state()
        state = self._state

        dofs_pos = np.zeros(self.num_dofs, dtype=np.float64)
        dofs_vel = np.zeros(self.num_dofs, dtype=np.float64)
        dofs_torque = np.zeros(self.num_dofs, dtype=np.float64)
        # Motor health (Eden dof order): recorded to --debug_record and used to warn on
        # thermal derating (a frozen/sagging leg is often an overheated motor cutting its
        # own torque, not a controller/policy issue).
        temps = np.zeros(self.num_dofs, dtype=np.float64)
        modes = np.zeros(self.num_dofs, dtype=np.float64)
        full_dofs_pos = np.zeros(self._num_full_dof, dtype=np.float64)

        for i, motor_idx in enumerate(self._dof_index):
            motor = state.motorState[motor_idx]
            dofs_pos[i] = motor.q
            dofs_vel[i] = motor.dq
            dofs_torque[i] = motor.tauEst
            temps[i] = motor.temperature
            modes[i] = motor.mode

        for i in range(self._num_full_dof):
            full_dofs_pos[i] = state.motorState[i].q

        self._warn_thermal(temps, modes)

        return RobotState(
            stamp=time.time(),
            base_quat=np.array(state.imu.quaternion, dtype=np.float64),  # wxyz
            base_ang_vel=np.array(state.imu.gyroscope, dtype=np.float64),  # body frame
            base_lin_acc=np.array(state.imu.accelerometer, dtype=np.float64),
            dofs_pos=dofs_pos,
            dofs_vel=dofs_vel,
            dofs_torque=dofs_torque,
            full_dofs_pos=full_dofs_pos,
            extra={"motor_temp": temps, "motor_mode": modes, **self._battery_extra()},
        )

    def _battery_extra(self) -> dict[str, np.ndarray]:
        """Battery voltage / SOC from the SDK BMS, best-effort (fields vary by firmware)."""
        bms = getattr(self._state, "bms", None)
        soc = float(getattr(bms, "SOC", np.nan)) if bms is not None else np.nan
        try:
            volt = sum(bms.cell_vol) / 1000.0  # mV -> V
        except Exception:
            volt = np.nan
        return {"battery_v": np.array([volt]), "battery_soc": np.array([soc])}

    def _warn_thermal(self, temps: np.ndarray, modes: np.ndarray) -> None:
        """Warn once per joint when a motor overheats / derates (recovery clears it)."""
        for i, name in enumerate(self.dofs_name):
            overheat = int(modes[i]) == _OVERHEAT_MODE
            hot = temps[i] >= self.motor_temp_warn_c
            if overheat or hot:
                if name not in self._thermal_warned:
                    self._thermal_warned.add(name)
                    why = "Overheat mode" if overheat else f"{temps[i]:.0f}°C"
                    en.logger.warning(
                        f"[deploy] Go1 motor '{name}' {why} — it derates its own torque here, so this "
                        "leg goes weak (a common cause of the robot freezing/sagging). Let it cool."
                    )
            else:
                self._thermal_warned.discard(name)

    def send_payload(self, payload: RobotCommand) -> None:
        self._require_connected()

        dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))

        for i, motor_idx in enumerate(self._dof_index):
            motor_cmd = self._cmd.motorCmd[motor_idx]
            motor_cmd.q = float(dofs_pos[i])
            motor_cmd.dq = float(payload.dofs_vel[i])
            motor_cmd.tau = float(payload.dofs_torque[i])
            if payload.dofs_kp is not None:
                motor_cmd.Kp = float(payload.dofs_kp[i])
            if payload.dofs_kd is not None:
                motor_cmd.Kd = float(payload.dofs_kd[i])

        # Refresh state first so PowerProtect compares against the current measurement,
        # not the one read at the end of the previous tick -- or none at all: the soft
        # shutdown ramp sends without reading.
        # CRC is computed inside SetSend (no manual CRC, unlike the DDS backends).
        self._refresh_state()
        self._safe.PowerProtect(self._cmd, self._state, self.power_protect_factor)
        self._udp.SetSend(self._cmd)
        self._udp.Send()

    def _init_low_cmd(self) -> None:
        for i in range(self._num_full_dof):
            motor_cmd = self._cmd.motorCmd[i]
            motor_cmd.mode = self.motor_mode
            motor_cmd.q = 0.0
            motor_cmd.dq = 0.0
            motor_cmd.Kp = 0.0
            motor_cmd.Kd = 0.0
            motor_cmd.tau = 0.0


bind_deployment(Go1, "unitree_go1_deployment")
