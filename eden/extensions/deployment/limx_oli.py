"""LimX Dynamics Oli (HU_D04) real-robot deployment backend.

Talks to the robot through the LimX low-level SDK (``limxsdk``, the Python
wheel shipped in https://github.com/limxdynamics/limxsdk-lowlevel): joint
state and IMU arrive on subscription callbacks, commands go out via
``publishRobotCmd``. Command/state arrays are indexed in the robot's motor
order, which the backend maps to/from Eden's ``dofs_name`` order by motor
*name* read live from the robot (``getMotorNames()``) — LimX's own docs and
shipped controller YAMLs disagree on the head-joint order, so a positional
map would be silently wrong.

Commands are sent in PR (pitch/roll, serial-joint) space with the SDK's
``parallel_solve_required`` flag set, so the SDK converts to the physical
A/B motors of the parallel ankle/waist linkages. This matches Eden's
:class:`~eden.options.robots.limx_oli.Oli` model (the serial
``HU_D04_01.urdf``) and LimX's own RL deployment stack; state read back in
this mode is PR-space as well.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import os
import threading
import time
from dataclasses import dataclass

import numpy as np

import eden as en
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    SoftShutdownMixin,
    bind_deployment,
    resolve_dofs_values,
)
from eden.extensions.deployment.utils.operator_input import (
    CMD_INIT,
    CMD_KILL,
    CMD_NEXT_POLICY,
    CMD_START,
    CMD_STOP,
)
from eden.extensions.deployment.utils.republisher import CommandRepublisherMixin
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.options.robots.limx_oli import OLI_DOFS_NAME, Oli
from eden.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.entities import RobotOptions
    from eden.options.extensions.deployment import DeploymentOptions


IS_LIMX_SDK_AVAILABLE = True
_LIMX_SDK_IMPORT_ERROR: Exception | None = None
try:
    from limxsdk.datatypes import RobotCmd as LimxRobotCmd
    from limxsdk.robot import Robot as LimxRobot
    from limxsdk.robot import RobotType as LimxRobotType
except Exception as _exc:
    IS_LIMX_SDK_AVAILABLE = False
    _LIMX_SDK_IMPORT_ERROR = _exc
    LimxRobotCmd = None
    LimxRobot = None
    LimxRobotType = None


def ensure_limx_sdk() -> None:
    """Raise a descriptive error if the LimX SDK is not importable."""
    if not IS_LIMX_SDK_AVAILABLE:
        raise RuntimeError(
            "LimX SDK (limxsdk) failed to import. Install the platform wheel shipped in "
            "https://github.com/limxdynamics/limxsdk-lowlevel under python3/<arch>/ "
            "(e.g. `uv pip install python3/amd64/limxsdk-*.whl`). "
            f"Underlying error: {_LIMX_SDK_IMPORT_ERROR!r}"
        )


# Robot IP the LimX humanoid answers on when reached over its own network
# (the LimX MuJoCo simulator listens on 127.0.0.1 instead). Overridable via
# the ``robot_ip`` option or the ROBOT_IP environment variable (the same
# variable LimX's own deploy tooling honors).
_DEFAULT_ROBOT_IP = "10.192.1.2"

# LimX remote button indices in ``SensorJoy.buttons`` (from the chord handling
# in limxdynamics/humanoid-rl-deploy-python's main.py). The remote is
# PS4-style, so buttons are named by their physical symbols; LimX's own code
# comments name the same indices with Xbox-style letters
# (cross=A, circle=B, square=X, triangle=Y).
_JOY_BUTTON_IDX: dict[str, int] = {"cross": 0, "circle": 1, "square": 2, "triangle": 3, "L1": 4, "R1": 7}

# Operator chords → commands, checked in order (KILL first so an L1+R1+square
# mash shuts down instead of starting the policy). The chords follow LimX's
# own deploy tooling where a counterpart exists: L1+triangle starts their
# stand controller (→ INIT), R1+square their walk policy (→ START), L1+cross
# their damping controller (→ STOP/hold), L1+square exits (→ KILL);
# L1+circle (their mimic switch) cycles to the next loaded policy slot.
_REMOTE_CHORDS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"L1", "square"}), CMD_KILL),
    (frozenset({"L1", "triangle"}), CMD_INIT),
    (frozenset({"L1", "cross"}), CMD_STOP),
    (frozenset({"L1", "circle"}), CMD_NEXT_POLICY),
    (frozenset({"R1", "square"}), CMD_START),
)


def _canonical_motor_name(name: str) -> str:
    """Strip a trailing ``_joint`` so SDK motor names and URDF joint names compare suffix-independently."""
    return name.removesuffix("_joint")


# LimX's deploy-time torque budget where it differs from Eden's ``default_dofs_force_limits``
# (``walk_param.yaml`` ``user_torque_limit`` in humanoid-rl-deploy-python). The six parallel PR
# joints are each driven by two 42 N·m A/B motors through the linkage, so LimX budgets them at
# 80 N·m on the robot while its RL model -- like Eden's training force limits and the action
# scale derived from them -- keeps the per-joint 42. ``waist_yaw`` is a single motor and stays at 42.
LIMX_OLI_CLAMP_TORQUE_LIMITS: dict[str, float] = {
    "*_ankle_*_joint": 80.0,
    "waist_roll_joint": 80.0,
    "waist_pitch_joint": 80.0,
}


def resolve_clamp_torque_limits(
    robot_options: RobotOptions, dofs_name: list[str], overrides: dict[str, float]
) -> np.ndarray:
    """Per-DOF torque budget for the deploy clamp: the robot's force limits with ``overrides`` on top.

    ``overrides`` maps DOF-name patterns to N·m. It is resolved on its own and scattered onto the
    force-limit array instead of being merged into one dict, because
    :func:`~eden.utils.string.resolve_matching_names_values` refuses a DOF matched by two patterns
    (``waist_*_joint`` from the robot options and ``waist_roll_joint`` from the overrides would
    collide). A pattern matching no DOF raises, so a typo cannot silently leave a joint at the
    robot's limit.
    """
    limits = resolve_dofs_values(robot_options.default_dofs_force_limits, dofs_name)
    if overrides:
        idx, _, values = resolve_matching_names_values(overrides, dofs_name)
        limits[idx] = values
    return limits


# Defensive snapshots of the SDK subscription messages. LimX's own controllers
# ``copy.deepcopy`` the cached message before reading it, which suggests the
# native bridge may reuse/mutate message objects in place; copying the payload
# in the callback makes each rebind an atomic, immutable snapshot so
# ``read_state`` / the remote poll can never observe a torn message.


@dataclass
class _StateSnapshot:
    """Joint state in the robot's motor order (copied from a limxsdk RobotState)."""

    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray


@dataclass
class _ImuSnapshot:
    """IMU sample (copied from a limxsdk ImuData; ``quat`` is wxyz)."""

    quat: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray


@dataclass
class _JoySnapshot:
    """Remote sample (copied from a limxsdk SensorJoy)."""

    axes: list[float]
    buttons: list[int]


class _RemoteInput:
    """Reads the LimX remote from the deployment's latest ``SensorJoy`` message.

    No extra subscription state is needed: the deployment already subscribes
    to ``SensorJoy`` and keeps the latest message. A chord fires on its rising
    edge — the poll where it first becomes fully held — so holding it maps to
    a single intent.
    """

    def __init__(self, deployment: LimxOliDeployment) -> None:
        self._dep = deployment
        self._prev_pressed: frozenset[str] = frozenset()

    def start(self) -> bool:
        return True

    def poll(self) -> list[str]:
        joy = self._dep._latest_joy
        if joy is None:
            return []
        buttons = joy.buttons
        pressed = frozenset(name for name, idx in _JOY_BUTTON_IDX.items() if idx < len(buttons) and buttons[idx])
        prev = self._prev_pressed
        self._prev_pressed = pressed
        return [cmd for chord, cmd in _REMOTE_CHORDS if chord <= pressed and not chord <= prev]


@DEPLOYMENT_REGISTRY.register()
class LimxOliDeployment(CommandRepublisherMixin, SoftShutdownMixin, DeploymentBase):
    """LimX Oli deployment backend on top of :class:`DeploymentBase`.

    Adds, on top of the base options:

    - ``robot_ip``: IP the LimX SDK connects to. ``None`` (default) resolves
      to the ``ROBOT_IP`` environment variable, then to the real robot's
      ``10.192.1.2``; point it at ``127.0.0.1`` for the LimX MuJoCo simulator.
    - ``shutdown_damping_steps`` / ``shutdown_damping_kd_scale``: soft
      shutdown — cycles to hold a damping (kp=0, kd) state before releasing.
    - ``soft_torque_limit``: fraction of each joint's torque budget the
      *position* term of the implied PD torque may use; every commanded target
      is clamped to ``q ± soft_torque_limit * budget / kp`` around the measured
      pose. Same 0.95 default as LimX's own walk controller, but without the
      velocity term theirs includes -- see :meth:`_clamp_torque_feasible` for
      the hardware measurements behind that. ``0`` disables the clamp.
    - ``clamp_torque_limits``: DOF-name pattern -> N·m overrides on top of the
      robot's ``default_dofs_force_limits`` for that budget. Defaults to LimX's
      deploy-time ``user_torque_limit`` where it differs from Eden's training
      limits: 80 N·m on the six parallel ankle/waist PR joints (two 42 N·m A/B
      motors each). Training keeps 42 -- the shipped action scale derives from
      it -- so this is a deploy-only budget.
    - ``publish_rate``: rate at which the current wire command is re-sent
      between control ticks (default 1000 Hz, LimX's own rate). **Load-bearing
      on this robot**: in PR mode the robot evaluates the PD law of the six
      parallel-linkage joints (ankle pitch/roll, waist roll/pitch) once per
      *received packet*, on that packet's state -- measured on hardware, where
      their torque fits ``kp*(q_des-q) - kd*dq[previous packet]`` at R^2 0.92-0.99
      while the serial joints fit the same-instant velocity. At 50 Hz that
      damping term acts on a 20 ms-old velocity, which is anti-damping at 25 Hz:
      the waist-roll linkage rings at exactly that frequency once a foot strike
      kicks it. ``None`` sends one packet per control tick, for A/B comparison
      only.
    - ``motor_name_map``: Eden DOF name → SDK motor name overrides for robots
      whose ``getMotorNames()`` vocabulary doesn't match the URDF joint names
      (unmapped DOFs still match by suffix-normalized name).
    - ``verbose_diagnostics``: also log warning/OK robot diagnostics as they
      change. By default only error-level (2) diagnostics are logged — the
      robot's diagnostic stream is chatty and the OK/warning transitions are
      commissioning detail, but a fault must never be silent.
    """

    robot_ip: str | None = None
    soft_torque_limit: float = 0.95
    clamp_torque_limits: dict = dict(LIMX_OLI_CLAMP_TORQUE_LIMITS)
    motor_name_map: dict = {}
    verbose_diagnostics: bool = False
    publish_rate: float | None = 1000.0

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)

        self._robot: Any | None = None
        self._init_republisher()
        self._latest_state: Any | None = None
        self._latest_imu: Any | None = None
        self._latest_joy: Any | None = None
        # Last (code, level) logged per diagnostic name, so the continuous
        # diagnostic stream is logged only on change.
        self._last_diagnostic: dict[str, tuple[int, int]] = {}
        self._state_event = threading.Event()
        self._imu_event = threading.Event()
        self._dof_index: list[int] | None = None
        self._num_motors: int | None = None
        self._cmd: Any | None = None

    def _resolve_options(self) -> None:
        """Check the DOF→motor mapping and the torque clamp (see :meth:`DeploymentBase._resolve_options`)."""
        super()._resolve_options()
        self._validate_oli_dofs()
        self.dof_torque_limits = resolve_clamp_torque_limits(
            self.robot_options, self.dofs_name, self.clamp_torque_limits
        )
        self._validate_torque_clamp()
        self._validate_publish_rate()

    def _validate_oli_dofs(self) -> None:
        """Fail fast at construction on a robot config whose DOFs are not HU_D04 motors.

        The authoritative per-index map is built from the live ``getMotorNames()``
        in :meth:`connect` (motor *order* is firmware-owned, only names are
        stable); this check only catches renamed/extra DOFs before hardware
        contact. A DOF with a ``motor_name_map`` entry is exempt (the mapping is
        an explicit declaration of its SDK motor), and map keys that are not
        deployment DOFs are a typo — rejected here.
        """
        stray = [name for name in self.motor_name_map if name not in self.dofs_name]
        if stray:
            raise ValueError(f"motor_name_map keys {stray} are not deployment DOFs. DOFs: {self.dofs_name}")
        known = {_canonical_motor_name(name) for name in OLI_DOFS_NAME}
        unknown = [
            name
            for name in self.dofs_name
            if name not in self.motor_name_map and _canonical_motor_name(name) not in known
        ]
        if unknown:
            raise ValueError(f"DOFs {unknown} are not LimX Oli (HU_D04) motors. Known motors: {sorted(OLI_DOFS_NAME)}")

    def _validate_torque_clamp(self) -> None:
        """Fail fast on an unusable torque-clamp configuration (see :meth:`_clamp_torque_feasible`)."""
        if not 0.0 <= self.soft_torque_limit <= 1.0:
            raise ValueError(f"soft_torque_limit must be in [0, 1] (0 disables), got {self.soft_torque_limit}")
        if self.soft_torque_limit == 0.0:
            return
        unlimited = [name for name, lim in zip(self.dofs_name, self.dof_torque_limits) if lim <= 0.0]
        if unlimited:
            raise ValueError(
                f"soft_torque_limit={self.soft_torque_limit} needs a positive default_dofs_force_limits "
                f"entry for every DOF, but {unlimited} resolved to <= 0. Set the robot's force limits "
                "or disable the clamp with soft_torque_limit=0."
            )

    # ------------------------------------------------------------------
    # SDK connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        ensure_limx_sdk()
        self._latest_state = None
        self._latest_imu = None
        self._latest_joy = None
        self._state_event.clear()
        self._imu_event.clear()

        ip = self.robot_ip or os.environ.get("ROBOT_IP") or _DEFAULT_ROBOT_IP
        robot = LimxRobot(LimxRobotType.Humanoid)
        if not robot.init(ip):
            raise RuntimeError(
                f"Failed to connect to LimX Oli at {ip}. Check the robot is powered and reachable "
                "(real robot: 10.192.1.2 over the robot's network; LimX MuJoCo sim: 127.0.0.1), "
                "or override the target with the ROBOT_IP environment variable "
                "(or the robot_ip deployment option)."
            )

        motor_names = list(robot.getMotorNames())
        num_motors = int(robot.getMotorNumber())
        if len(motor_names) != num_motors:
            raise RuntimeError(
                f"LimX SDK reports {num_motors} motors but {len(motor_names)} motor names: {motor_names}"
            )
        self._dof_index = self._build_dof_index(motor_names)
        self._num_motors = num_motors
        driven = set(self._dof_index)
        idle = [name for i, name in enumerate(motor_names) if i not in driven]
        if idle:
            en.logger.warning(
                f"[deploy] {len(idle)} robot motor(s) are not in the task's DOFs and will be held "
                f"limp (kp=kd=0): {idle}"
            )

        self._cmd = self._init_robot_cmd(motor_names)
        self._last_diagnostic.clear()
        robot.subscribeRobotState(self._on_robot_state)
        robot.subscribeImuData(self._on_imu_data)
        robot.subscribeSensorJoy(self._on_sensor_joy)
        robot.subscribeDiagnosticValue(self._on_diagnostic)
        self._robot = robot

        start_time = time.time()
        while not (self._state_event.is_set() and self._imu_event.is_set()):
            if time.time() - start_time > self.connect_timeout_s:
                missing = [
                    name
                    for name, event in (("robot state", self._state_event), ("IMU", self._imu_event))
                    if not event.is_set()
                ]
                raise RuntimeError(f"Timed out waiting for LimX Oli {' and '.join(missing)} stream(s).")
            time.sleep(0.05)

        self._start_publisher()

    def close(self) -> None:
        # Flush the debug recording and release the operator input device before tearing
        # down (both are in-memory, independent of the SDK link).
        self._teardown_common()
        # Ramp to a damping state before dropping the handle so the robot
        # settles softly instead of losing all commands at once.
        self._soft_shutdown()
        self._stop_publisher()
        # The SDK has no unsubscribe/disconnect API; dropping the handle stops
        # our publishing and the daemon-side link dies with the process.
        self._robot = None
        self._cmd = None
        self._latest_state = None
        self._latest_imu = None
        self._latest_joy = None
        self._state_event.clear()
        self._imu_event.clear()

    def _build_dof_index(self, motor_names: list[str]) -> list[int]:
        """Map each Eden DOF to its slot in the robot's motor arrays, by (suffix-normalized) name.

        A DOF with a ``motor_name_map`` entry looks up that SDK motor name
        instead of its own.
        """
        canon_to_idx: dict[str, int] = {}
        for idx, name in enumerate(motor_names):
            canon = _canonical_motor_name(name)
            if canon in canon_to_idx:
                raise RuntimeError(
                    f"LimX SDK motor name '{name}' collides with '{motor_names[canon_to_idx[canon]]}' "
                    f"after suffix normalization; cannot map DOFs by name. Motor names: {motor_names}"
                )
            canon_to_idx[canon] = idx

        def slot(dof_name: str) -> int | None:
            return canon_to_idx.get(_canonical_motor_name(self.motor_name_map.get(dof_name, dof_name)))

        missing = [name for name in self.dofs_name if slot(name) is None]
        if missing:
            raise RuntimeError(
                f"Robot does not expose motors for DOFs {missing}. Motor names reported by the robot: "
                f"{motor_names}. If the robot uses a different naming scheme, provide motor_name_map "
                "(Eden DOF name -> SDK motor name)."
            )
        return [slot(name) for name in self.dofs_name]

    def _init_robot_cmd(self, motor_names: list[str]) -> Any:
        """Build the reusable full-motor command message (all gains zero until filled per send)."""
        cmd = LimxRobotCmd()
        n = len(motor_names)
        cmd.motor_names = list(motor_names)
        cmd.mode = [0] * n  # torque-position hybrid control
        cmd.q = [0.0] * n
        cmd.dq = [0.0] * n
        cmd.tau = [0.0] * n
        cmd.Kp = [0.0] * n
        cmd.Kd = [0.0] * n
        # PR mode: the SDK solves the parallel ankle/waist linkages, so q/Kp/Kd
        # stay in the serial-joint space the policy (and Eden's Oli URDF) uses.
        cmd.parallel_solve_required = [True] * n
        return cmd

    # ------------------------------------------------------------------
    # Operator remote and shutdown
    # ------------------------------------------------------------------

    def _remote_inputs(self) -> list[Any]:
        return [_RemoteInput(self)]

    def _shutdown_hold_pos(self) -> np.ndarray | None:
        """Hold the last pose the loop read (see :meth:`SoftShutdownMixin._shutdown_hold_pos`); nothing before it."""
        if self._robot is None or self._cmd is None or self.state is None:
            return None
        return self.state.dofs_pos.copy()

    def _print_control_guide(self) -> None:
        en.logger.info("=" * 60)
        en.logger.info("LIMX OLI DEPLOY CONTROLS  (PS4-style remote chord / keyboard)")
        en.logger.info("  R1+□ / s+Enter  START  (hand control to the policy)")
        en.logger.info("  L1+✕ / x+Enter  STOP   (hold current pose with PD)")
        en.logger.info("  L1+△ / i+Enter  INIT   (interpolate to default pose)")
        en.logger.info("  L1+□ / k+Enter  KILL   (soft-damp and exit)")
        en.logger.info("  L1+○ / n+Enter  SWITCH to next policy (if >1 loaded)")
        en.logger.info("  buttons: □ square(X)  △ triangle(Y)  ✕ cross(A)  ○ circle(B)")
        en.logger.info("  1-9 + Enter     SWITCH to that policy slot")
        en.logger.info("Robot boots in STOP/hold. Press INIT, then START.")
        en.logger.info("The robot should feel STIFF (holding) from this point. If it is limp, the")
        en.logger.info("motors are not activated (remote: CLICK THE RIGHT JOYSTICK after power-on).")
        en.logger.info("If motors buzz/fight, the robot is not in developer mode (remote: R1+DOWN;")
        en.logger.info("exit R1+LEFT) and the onboard controller is contending with SDK commands.")
        en.logger.info("=" * 60)

    # ------------------------------------------------------------------
    # High-rate republisher (see the ``publish_rate`` option)
    # ------------------------------------------------------------------

    def _publish_held_command(self) -> None:
        """Re-send the held ``RobotCmd`` (see :class:`CommandRepublisherMixin`).

        ``stamp`` is refreshed on every send: it is the packet's own timestamp, not the
        command's, and the robot reads it to age-check what it receives.
        """
        robot, cmd = self._robot, self._cmd
        if robot is None or cmd is None:
            return
        cmd.stamp = time.time_ns()
        robot.publishRobotCmd(cmd)
        self._pub_count += 1

    # ------------------------------------------------------------------
    # Wire I/O
    # ------------------------------------------------------------------

    def read_state(self) -> RobotState:
        if self._latest_state is None or self._latest_imu is None:
            raise RuntimeError("No LimX Oli robot state / IMU data received yet.")

        msg = self._latest_state
        imu = self._latest_imu
        q = np.asarray(msg.q, dtype=np.float64)
        dq = np.asarray(msg.dq, dtype=np.float64)
        tau = np.asarray(msg.tau, dtype=np.float64)
        if q.shape[0] != self._num_motors:
            raise RuntimeError(
                f"LimX robot state carries {q.shape[0]} motors, expected {self._num_motors} "
                "(robot type / firmware mismatch?)."
            )

        return RobotState(
            stamp=time.time(),
            base_quat=np.asarray(imu.quat, dtype=np.float64),  # wxyz, same as Eden
            base_ang_vel=np.asarray(imu.gyro, dtype=np.float64),
            base_lin_acc=np.asarray(imu.acc, dtype=np.float64),
            dofs_pos=q[self._dof_index],
            dofs_vel=dq[self._dof_index],
            dofs_torque=tau[self._dof_index],
            full_dofs_pos=q,
        )

    def _clamp_torque_feasible(self, dofs_pos: np.ndarray, dofs_kp: np.ndarray | None) -> np.ndarray:
        """Clamp position targets so the position-implied PD torque stays within the per-joint budget.

        The wire PD torque is ``kp*(q_des - q) - kd*dq`` (commanded velocity and feedforward
        torque are always 0). Only the position term is budgeted: ``q_des`` is clipped to
        ``q ± soft_torque_limit * budget / kp`` around the pose measured at the previous control
        tick. Runs on the wire values, *after* the target low-pass, so every command path
        (init/hold/policy) is covered.

        LimX's walk controller clamps the full expression, i.e. centres the window on
        ``q + kd*dq/kp``. That velocity term is deliberately left out here, for two measured
        reasons:

        - This clamp runs once per 20 ms tick on a state up to one tick old; LimX re-evaluates
          theirs every 1 ms with fresh state. ``q`` moves a few hundredths of a radian in a tick,
          but ``kd*dq/kp`` (0.127*dq on the waist) moved the window centre by 0.13-0.23 rad on
          average and 0.75 rad at peak on hardware -- past the whole half-window.
        - While the waist-roll linkage rang (see ``publish_rate``), the PR-space velocity the SDK
          reported for it was 8-11x the derivative of the reported position and uncorrelated with
          it. Feeding that into the target pinned the wire command to the window edge on 11-28 %
          of walking ticks (waist_roll) and 6-17 % (ankle_pitch), each pin a ~40 N·m torque step
          that amplified the oscillation. A position-only window is inert while walking
          (|q_des - q| p95 ~ 0.15 rad against a >= 0.43 rad half-window) and still catches a
          runaway target or a far INIT pose.

        Joints commanded with ``kp == 0`` produce no position-dependent torque and pass through
        unclamped; ``kp`` defaults to the values currently in the command template when the
        payload doesn't carry them (matching what :meth:`send_payload` sends).
        """
        if self.soft_torque_limit == 0.0:
            return dofs_pos
        state = self.state if self.state is not None else self.read_state()
        if dofs_kp is None:
            dofs_kp = [self._cmd.Kp[m] for m in self._dof_index]
        kp = np.asarray(dofs_kp, dtype=np.float64)
        half_window = self.soft_torque_limit * self.dof_torque_limits / np.where(kp > 0.0, kp, 1.0)
        clamped = np.clip(dofs_pos, state.dofs_pos - half_window, state.dofs_pos + half_window)
        return np.where(kp > 0.0, clamped, dofs_pos)

    def send_payload(self, payload: RobotCommand) -> None:
        if self._robot is None or self._cmd is None:
            raise RuntimeError("LimX Oli deployment is not connected.")

        # Low-pass the commanded targets (no-op unless --target_lpf < 1) before
        # they hit the wire, so every path (init/hold/policy) is smoothed, then
        # clamp them to the position-torque budget (the clamp runs last so the
        # wire command is guaranteed within it).
        payload.dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))
        payload.dofs_pos = self._clamp_torque_feasible(payload.dofs_pos, payload.dofs_kp)

        with self._cmd_lock:
            cmd = self._cmd
            cmd.stamp = time.time_ns()
            for i, motor_idx in enumerate(self._dof_index):
                cmd.q[motor_idx] = float(payload.dofs_pos[i])
                cmd.dq[motor_idx] = float(payload.dofs_vel[i])
                cmd.tau[motor_idx] = float(payload.dofs_torque[i])
                if payload.dofs_kp is not None:
                    cmd.Kp[motor_idx] = float(payload.dofs_kp[i])
                if payload.dofs_kd is not None:
                    cmd.Kd[motor_idx] = float(payload.dofs_kd[i])
            self._robot.publishRobotCmd(cmd)
            self._note_command_published()

    # ------------------------------------------------------------------
    # SDK subscription callbacks. Each runs on an SDK dispatch thread; the
    # payload is copied into an immutable snapshot before the (atomic)
    # attribute rebind, so readers never observe a torn message even if the
    # native bridge reuses/mutates its message objects in place.
    # ------------------------------------------------------------------

    def _on_robot_state(self, msg: Any) -> None:
        self._latest_state = _StateSnapshot(
            q=np.asarray(msg.q, dtype=np.float64),
            dq=np.asarray(msg.dq, dtype=np.float64),
            tau=np.asarray(msg.tau, dtype=np.float64),
        )
        self._state_event.set()

    def _on_imu_data(self, msg: Any) -> None:
        self._latest_imu = _ImuSnapshot(
            quat=np.asarray(msg.quat, dtype=np.float64),  # wxyz
            gyro=np.asarray(msg.gyro, dtype=np.float64),
            acc=np.asarray(msg.acc, dtype=np.float64),
        )
        self._imu_event.set()

    def _on_sensor_joy(self, msg: Any) -> None:
        self._latest_joy = _JoySnapshot(axes=list(msg.axes), buttons=list(msg.buttons))

    def _on_diagnostic(self, msg: Any) -> None:
        """Log robot-side diagnostics on change (SDK levels: 0 OK, 1 warning, 2 error).

        The robot streams diagnostics continuously (motor faults, overheat,
        communication health, ...); a component is logged only when its
        ``(code, level)`` changes, and — unless ``verbose_diagnostics`` — only
        at error level: the OK/warning chatter is commissioning detail, but a
        fault must never be silent.
        """
        name, code, level = str(msg.name), int(msg.code), int(msg.level)
        if self._last_diagnostic.get(name) == (code, level):
            return
        self._last_diagnostic[name] = (code, level)
        text = f"[deploy] robot diagnostic '{name}': level={level} code={code} {msg.message}"
        if level >= 2:
            en.logger.error(text)
        elif self.verbose_diagnostics:
            if level == 1:
                en.logger.warning(text)
            else:
                en.logger.info(text)


bind_deployment(Oli, "limx_oli_deployment")
