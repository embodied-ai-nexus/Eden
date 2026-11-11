"""Franka Panda real-robot deployment backend."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar
from collections.abc import Mapping
from types import MappingProxyType

import threading
import time

import numpy as np

import eden as en
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY, DeploymentBase, bind_deployment
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda, FrankaEmikaPandaNoHand, FrankaResearch3

try:
    import panda_py
    from panda_py import controllers as _panda_controllers
    from panda_py import libfranka as _libfranka
except ImportError:
    panda_py = None  # type: ignore[assignment]
    _panda_controllers = None  # type: ignore[assignment]
    _libfranka = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions


# Eden DOF name -> 0-based index into the 7-element arm state ``q`` returned by panda-py.
PANDA_ARM_JOINT_IDX: dict[str, int] = {
    "joint1": 0,
    "joint2": 1,
    "joint3": 2,
    "joint4": 3,
    "joint5": 4,
    "joint6": 5,
    "joint7": 6,
}
# Finger joint names that map to the Franka Hand width.
PANDA_FINGER_JOINTS: tuple[str, ...] = ("finger_joint1", "finger_joint2")

# Default Franka Hand maximum jaw opening (m); read from device after homing.
_DEFAULT_MAX_FINGER_WIDTH = 0.08
# Minimum target-width change before a new ``gripper.move`` is dispatched (m).
_GRIPPER_MOVE_THRESHOLD = 0.003

# Panda per-joint velocity limits (rad/s) from the manufacturer datasheet.
# Used by ``send_payload`` to clamp the commanded delta per step so the
# libfranka reflex (``joint_velocity_violation``) doesn't trip.
_PANDA_JOINT_VELOCITY_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=np.float64)


@DEPLOYMENT_REGISTRY.register()
class FrankaPandaDeployment(DeploymentBase):
    """Deployment backend for Franka Emika Panda / FR3 via panda-py.

    The 7-DOF arm is driven through panda-py's real-time
    :class:`controllers.JointPosition` controller running at 1 kHz inside
    libfranka. The Franka Hand gripper is driven through libfranka's
    blocking ``Gripper`` API, dispatched from a worker thread so the
    per-step :meth:`send_payload` call never blocks the control loop.

    The arm is fixed-base with no IMU, so :meth:`read_state` returns an
    identity ``base_quat``, zero base angular velocity and zero linear acceleration.

    Requirements
    ------------
    - ``panda-python`` must be installed with a libfranka build that matches
      the controller's reported server version. See
      https://github.com/JeanElsner/panda-py/releases for per-firmware wheels.
    - FCI must be activated in Desk and the joint brakes released.
    - A PREEMPT_RT kernel is strongly recommended for stable 1 kHz control;
      without it expect occasional ``communication_constraints_violation``.

    Parameters
    ----------
    robot_ip:
        IP address of the Franka Control box. Defaults to the factory IP.
    arm_stiffness, arm_damping:
        Optional 7-element per-joint gains for the JointPosition controller.
        If ``None``, panda-py's defaults are used.
    arm_filter_coeff:
        Exponential low-pass coefficient applied to the JointPosition
        setpoint inside panda-py's 1 kHz control loop. ``1.0`` = no
        filtering (raw step inputs every ``set_control`` call); lower
        values smooth the trajectory. For policy/teleop at 30-100 Hz the
        controller sees abrupt setpoint changes between updates which
        cause the arm to buzz/shake under PD, so the default here is
        ``0.05`` (panda-py's value is ``1.0``). Drop further if you still
        see oscillation; raise toward ``1.0`` if you need maximum
        tracking responsiveness at the cost of smoothness.
    velocity_safety_factor:
        Fraction of Panda's per-joint velocity limits that the deployment
        will allow per control step. ``send_payload`` clamps the
        commanded delta from ``current_q`` so the implied joint velocity
        is bounded by ``vel_limit * velocity_safety_factor``. Direction
        is preserved by uniformly scaling the delta (rather than clipping
        per joint, which would distort EE motion). Without this clamp
        Differential IK near a high-Jacobian configuration can emit
        deltas that trigger libfranka's ``joint_velocity_violation``
        reflex. Default ``0.5`` is conservative; raise toward 1.0 for
        more aggressive teleop on a PREEMPT_RT kernel.
    use_grasp_on_close:
        When ``True``, the gripper worker uses ``libfranka.Gripper.grasp``
        (force-controlled) for *closing* commands and reserves
        ``Gripper.move`` for *opening*. ``grasp`` is more responsive on
        held objects (it returns as soon as firmware detects finger
        contact, instead of waiting out ``move``'s stall timeout) and
        applies continuous holding force, so objects don't slip during
        arm motion. The classification is by absolute target width:
        a command below ``close_width_threshold`` is treated as a
        close, anything at or above as an open. Disable to fall back
        to ``move``-only behavior (e.g. if your task is purely
        position-mode finger control).
    close_width_threshold:
        Target width (m) below which a command is treated as a close
        (uses ``grasp`` when ``use_grasp_on_close`` is set). Default
        ``0.04`` is roughly half the Franka Hand's max opening.
    gripper_speed:
        Default gripper move speed (m/s).
    gripper_force:
        Default grasp force (N) for the :meth:`grasp` helper.
    gripper_epsilon_inner, gripper_epsilon_outer:
        Tolerance (m) for :meth:`grasp` success on the inner/outer side.
    skip_gripper_homing:
        If ``True``, do not run ``gripper.homing()`` even when the encoder
        reports uncalibrated. Useful when an object is already held.

    Like every backend, the arm boots in STOP holding the pose it is measured at: the first
    command starts the JointPosition controller with that pose as its setpoint, so nothing moves
    until the operator's INIT ramps it to ``init_target`` (over ``init_steps``, with the per-step
    velocity clamp below) and START hands it to the policy. The gripper is the policy's alone:
    the hold, INIT and the START tick leave it where the policy last commanded it (at boot, where
    it is), so a STOP or INIT never opens a grasp (:attr:`DeploymentBase.sending_hold`). And
    ``target_lpf`` smooths the arm's targets only: the gripper takes the commanded width as is,
    since each change of it is a new gripper command that stops the one in flight.
    """

    # panda-py's JointPosition controller runs on its own stiffness/damping, not the command's.
    sends_command_gains: ClassVar[bool] = False

    _retired_params: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "speed_factor": "the arm no longer makes a blocking move_to_joint_position home on reset; the "
            "operator's INIT ramps it to init_target over init_steps control ticks instead. Set init_steps.",
            "match_current_pose_on_init": "every deployment now boots in STOP holding the measured pose, which "
            "is what True did; INIT then ramps it to init_target. Drop the option.",
        }
    )

    robot_ip: str = "192.168.1.1"
    arm_stiffness: tuple[float, ...] | None = None
    arm_damping: tuple[float, ...] | None = None
    arm_filter_coeff: float = 0.05
    velocity_safety_factor: float = 0.5
    use_grasp_on_close: bool = True
    close_width_threshold: float = 0.04
    gripper_speed: float = 0.1
    gripper_force: float = 20.0
    gripper_epsilon_inner: float = 0.02
    gripper_epsilon_outer: float = 0.02
    skip_gripper_homing: bool = False

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)

        # _finger_dof_indices comes from _resolve_options, which super().__init__ ran.
        self._has_gripper: bool = len(self._finger_dof_indices) > 0
        self._gripper_max_width: float = _DEFAULT_MAX_FINGER_WIDTH

        self._panda: Any | None = None
        self._gripper: Any | None = None
        self._controller: Any | None = None
        self._controller_started: bool = False

        # Background gripper command dispatch.
        self._gripper_target_width: float | None = None
        self._gripper_command_lock = threading.Lock()
        self._gripper_thread: threading.Thread | None = None
        self._gripper_stop = threading.Event()

    def _resolve_options(self) -> None:
        """Reject torque-mode action terms and partition the DOFs (see :meth:`DeploymentBase._resolve_options`).

        Reading the action manager here is fine — the swap replaces an *entity*, and the
        managers are untouched either way.
        """
        super()._resolve_options()

        # FrankaPandaDeployment drives the arm through panda-py's
        # ``JointPosition`` controller, so action terms that emit joint
        # *torques* (like OperationalSpaceController) cannot be wired
        # through ``send_payload``: their torques would be sent to the
        # JointPosition controller as if they were joint angles, which is
        # unsafe. Reject such terms up front rather than at first command.
        for term_name, term in self._env.action_manager._terms.items():
            if isinstance(term, OperationalSpaceController):
                raise ValueError(
                    f"FrankaPandaDeployment cannot drive action term '{term_name}' "
                    f"({type(term).__name__}): it produces joint torques, but this "
                    f"deployment only consumes joint positions via panda-py's "
                    f"JointPosition controller. Use DifferentialIKController or "
                    f"another position-mode action term instead, or implement a "
                    f"torque-mode deployment via panda_py.controllers.AppliedTorque."
                )

        # Partition deployment DOFs into arm joints and finger joints.
        self._arm_dof_indices: list[int] = []
        self._arm_joint_indices: list[int] = []
        self._finger_dof_indices: list[int] = []
        for i, name in enumerate(self.dofs_name):
            if name in PANDA_ARM_JOINT_IDX:
                self._arm_dof_indices.append(i)
                self._arm_joint_indices.append(PANDA_ARM_JOINT_IDX[name])
            elif name in PANDA_FINGER_JOINTS:
                self._finger_dof_indices.append(i)
            else:
                raise ValueError(
                    f"Unknown Franka Panda DOF '{name}'. Expected one of "
                    f"{list(PANDA_ARM_JOINT_IDX)} or {list(PANDA_FINGER_JOINTS)}."
                )

        if not self._arm_dof_indices:
            raise ValueError("FrankaPandaDeployment requires at least one arm joint in dofs_name.")

    # ------------------------------------------------------------------
    # DeploymentBase interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if panda_py is None:
            raise RuntimeError(
                "panda-python package is not installed. Install the libfranka-matched "
                "wheel from https://github.com/JeanElsner/panda-py/releases ."
            )

        self._panda = panda_py.Panda(self.robot_ip)
        # Clear any latched controller error from a prior session.
        self._panda.recover()

        if self._has_gripper:
            self._gripper = _libfranka.Gripper(self.robot_ip)
            state = self._gripper.read_once()
            # ``max_width`` reads as ~0 when the encoder hasn't been homed
            # this power cycle. Homing fully opens then closes the fingers
            # to calibrate, so it requires nothing be held.
            if not self.skip_gripper_homing and state.max_width <= 0.01:
                en.logger.info("[franka_panda] Homing gripper to calibrate encoder...")
                self._gripper.homing()
                state = self._gripper.read_once()
            if state.max_width > 0.0:
                self._gripper_max_width = float(state.max_width)

            self._gripper_stop.clear()
            self._gripper_thread = threading.Thread(target=self._gripper_worker, daemon=True)
            self._gripper_thread.start()

        kwargs: dict[str, Any] = {}
        if self.arm_stiffness is not None:
            kwargs["stiffness"] = np.asarray(self.arm_stiffness, dtype=np.float64)
        if self.arm_damping is not None:
            kwargs["damping"] = np.asarray(self.arm_damping, dtype=np.float64)
        # Setpoint low-pass (see ``arm_filter_coeff``). Passed by keyword so a panda-py
        # build that names it differently raises instead of silently ignoring it.
        kwargs["filter_coeff"] = float(self.arm_filter_coeff)
        self._controller = _panda_controllers.JointPosition(**kwargs)

    def close(self) -> None:
        self._teardown_common()
        if self._gripper_thread is not None:
            self._gripper_stop.set()
            self._gripper_thread.join(timeout=2.0)
        self._gripper_thread = None

        if self._panda is not None and self._controller_started:
            try:
                self._panda.stop_controller()
            except Exception as e:
                en.logger.warning(f"[franka_panda] stop_controller failed: {e}")
        self._controller_started = False

        if self._gripper is not None:
            try:
                self._gripper.stop()
            except Exception:
                pass

        self._controller = None
        self._gripper = None
        self._panda = None

    def read_state(self) -> RobotState:
        if self._panda is None:
            raise RuntimeError("Franka Panda deployment is not connected.")

        s = self._panda.get_state()
        dofs_pos = np.zeros(self.num_dofs, dtype=np.float64)
        dofs_vel = np.zeros(self.num_dofs, dtype=np.float64)
        dofs_torque = np.zeros(self.num_dofs, dtype=np.float64)

        q = np.asarray(s.q, dtype=np.float64)
        dq = np.asarray(s.dq, dtype=np.float64)
        tau = np.asarray(s.tau_J, dtype=np.float64)
        for dof_i, joint_i in zip(self._arm_dof_indices, self._arm_joint_indices):
            dofs_pos[dof_i] = q[joint_i]
            dofs_vel[dof_i] = dq[joint_i]
            dofs_torque[dof_i] = tau[joint_i]

        extra: dict[str, Any] = {
            # O_T_EE is column-major (Fortran) in libfranka.
            "O_T_EE": np.asarray(s.O_T_EE, dtype=np.float64).reshape(4, 4, order="F"),
            "robot_mode": str(s.robot_mode),
        }

        if self._has_gripper and self._gripper is not None:
            g = self._gripper.read_once()
            width = float(g.width)
            # Symmetric split between the two finger joints (the Franka Hand
            # constraint ties them together at 2x the per-finger displacement).
            per_finger = 0.5 * width
            for dof_i in self._finger_dof_indices:
                dofs_pos[dof_i] = per_finger
            extra["gripper_width"] = width
            extra["gripper_is_grasped"] = bool(g.is_grasped)
            extra["gripper_max_width"] = float(g.max_width)

        return RobotState(
            stamp=time.time(),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # identity (wxyz) - fixed-base arm
            base_ang_vel=np.zeros(3),
            base_lin_acc=np.zeros(3),
            dofs_pos=dofs_pos,
            dofs_vel=dofs_vel,
            dofs_torque=dofs_torque,
            extra=extra,
        )

    def send_payload(self, payload: RobotCommand) -> None:
        if self._panda is None or self._controller is None:
            raise RuntimeError("Franka Panda deployment is not connected.")
        if not self._controller_started:
            # The first payload is the boot STOP hold of the measured pose, so the controller
            # starts with the arm's own pose as its setpoint: no jerk.
            self._panda.start_controller(self._controller)
            self._controller_started = True

        # Low-pass the arm on every path (hold / INIT / policy); the payload carries what is sent. Not
        # the fingers: a filtered width ramps over many ticks, the gripper worker dispatches each step
        # of it as a new command, and every new command stops the one in flight -- a closing grasp
        # would be stopped and re-issued at the control rate (:meth:`_gripper_worker`).
        commanded = np.asarray(payload.dofs_pos, dtype=np.float64)
        sent = self._apply_target_lpf(commanded)
        sent[self._finger_dof_indices] = commanded[self._finger_dof_indices]
        payload.dofs_pos = sent
        target_q = self._extract_arm_target(payload.dofs_pos)
        target_q = self._clamp_arm_velocity(target_q)
        self._controller.set_control(target_q)

        # The hold, the INIT ramp and the START tick leave the gripper at the policy's last
        # command: a target from the measured width would stop a force-controlled grasp and
        # re-send it as a plain move, and INIT's default width would open it on what it holds.
        if self._has_gripper and not self.sending_hold:
            self._set_gripper_target(self._extract_target_width(payload.dofs_pos))

    def _before_step(self) -> None:
        """Restore the sim entity's qpos from cached real state before the action manager runs.

        ``DeploymentBase.step`` reads state at the *end* of each step, so on
        the next call the wrapper buffer holds the most-recent real
        measurement — but a caller that also runs ``env.step`` between
        iterations (the keyboard teleop recipe does, to drive the viewer)
        will have mutated the sim entity's qpos by integrating sim physics.
        Action terms (DiffIK, etc.) cache the sim entity at build time and
        read ``self.entity.get_jacobian`` / ``self.entity.get_dofs_pos``
        from it, so they would otherwise see post-physics drift instead of
        real arm pose.

        Re-running ``state_entity.update(self.state)`` writes the wrapper's
        cached values back into the sim entity (``set_dofs_pos`` /
        ``set_dofs_vel``) without touching the network — no extra
        ``panda.get_state`` or ``gripper.read_once`` calls, which is
        important because the gripper worker thread is blocking on
        ``Gripper.move`` and a contending ``read_once`` here would stall
        the control loop.
        """
        if getattr(self, "state", None) is not None:
            self._state_entity.update(self.state)

    # ------------------------------------------------------------------
    # Public helpers usable from manipulation policies / scripts
    # ------------------------------------------------------------------

    def grasp(self, width: float, force: float | None = None) -> bool:
        """Grasp (synchronously) an object of approximately ``width`` meters.

        Returns ``True`` when the firmware reports the fingers clamped on
        something within the configured tolerance. Sets the background
        gripper worker's target to ``width`` first, so the worker does not
        countermand the grasp with a ``move`` of its own.
        """
        if self._gripper is None:
            raise RuntimeError("Gripper is not configured for this deployment.")
        with self._gripper_command_lock:
            self._gripper_target_width = float(width)
        return bool(
            self._gripper.grasp(
                width=float(width),
                speed=self.gripper_speed,
                force=float(force if force is not None else self.gripper_force),
                epsilon_inner=self.gripper_epsilon_inner,
                epsilon_outer=self.gripper_epsilon_outer,
            )
        )

    def recover(self) -> None:
        """Clear an error state on the Franka controller (e.g. after a reflex)."""
        if self._panda is None:
            raise RuntimeError("Franka Panda deployment is not connected.")
        self._panda.recover()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_arm_q(self) -> np.ndarray:
        """Return the panda's 7-element q vector from cached state.

        Falls back to a fresh ``panda.get_state().q`` read if nothing has
        been cached yet (only true before the first ``read_state``).
        """
        if self.state is not None:
            q = np.empty(7, dtype=np.float64)
            for dof_i, joint_i in zip(self._arm_dof_indices, self._arm_joint_indices):
                q[joint_i] = float(self.state.dofs_pos[dof_i])
            return q
        return np.asarray(self._panda.get_state().q, dtype=np.float64).copy()

    def _clamp_arm_velocity(self, target_q: np.ndarray) -> np.ndarray:
        """Clamp the per-step joint delta so the implied velocity stays within safe limits.

        Clamps ``target_q - current_q`` so the implied per-step velocity stays
        within ``velocity_safety_factor * PANDA_JOINT_VELOCITY_LIMITS``.

        Uses a single direction-preserving scale (the most-violating
        joint's ratio) instead of per-joint clipping, so EE motion direction
        is preserved when the IK output exceeds limits — the arm just moves
        toward the same target at a safe rate.
        """
        current_q = self._current_arm_q()
        delta = target_q - current_q
        dt = self.control_dt if self.control_dt and self.control_dt > 0 else (1.0 / 30.0)
        max_step = _PANDA_JOINT_VELOCITY_LIMITS * dt * float(self.velocity_safety_factor)

        abs_delta = np.abs(delta)
        # Avoid divide-by-zero: where abs_delta is tiny, the per-joint
        # ratio is infinite (no constraint). ``np.minimum`` over the
        # resulting array gives the smallest non-trivial scale.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(abs_delta > 1e-9, max_step / abs_delta, np.inf)
        scale = float(np.minimum(1.0, ratios.min()))
        if scale < 1.0:
            delta = delta * scale
        return current_q + delta

    def _extract_arm_target(self, dofs_pos: np.ndarray) -> np.ndarray:
        """Return a length-7 arm target in joint1..joint7 order.

        Any arm joints not present in ``dofs_name`` are held at their last
        measured value so the controller doesn't try to drive them to zero.
        """
        if len(self._arm_dof_indices) == 7:
            target = np.empty(7, dtype=np.float64)
            for dof_i, joint_i in zip(self._arm_dof_indices, self._arm_joint_indices):
                target[joint_i] = float(dofs_pos[dof_i])
            return target

        current = np.asarray(self._panda.get_state().q, dtype=np.float64).copy()
        for dof_i, joint_i in zip(self._arm_dof_indices, self._arm_joint_indices):
            current[joint_i] = float(dofs_pos[dof_i])
        return current

    def _extract_target_width(self, dofs_pos: np.ndarray) -> float:
        """Sum finger-joint targets into a gripper width (m), clipped to range."""
        if len(self._finger_dof_indices) == 1:
            # The two fingers move symmetrically; one commanded joint doubles to width.
            w = 2.0 * float(dofs_pos[self._finger_dof_indices[0]])
        else:
            w = float(sum(float(dofs_pos[i]) for i in self._finger_dof_indices))
        return float(np.clip(w, 0.0, self._gripper_max_width))

    def _set_gripper_target(self, width: float) -> None:
        with self._gripper_command_lock:
            self._gripper_target_width = width

    def _gripper_worker(self) -> None:
        """Dispatch gripper commands as the policy's target width changes.

        Why this is non-trivial:

        ``libfranka.Gripper.move`` is blocking — it returns only when the
        fingers reach the commanded width or libfranka's internal stall
        timeout fires. With an object between the fingers ``move`` can
        block for hundreds of ms to several seconds before returning,
        which would freeze our responsiveness if the worker called it
        synchronously: e.g. closing on a held object then commanding
        "open" would wait out the prior ``move(0.0)``'s timeout first.

        So each command is dispatched into its own daemon sub-thread;
        the poller stays alive in this outer loop. When a new target
        arrives while a previous command is still in flight, we call
        ``gripper.stop()`` (safe from a separate thread on the same
        Gripper instance) to abort it, join briefly, and immediately
        dispatch the new one.

        For "close" commands (target below ``close_width_threshold``)
        we use ``Gripper.grasp`` instead of ``Gripper.move`` when
        ``use_grasp_on_close`` is enabled. ``grasp`` is force-controlled
        and returns as soon as the firmware detects finger contact with
        an object, which is typically much faster than ``move``'s stall
        timeout, and it maintains a hold force so the object doesn't
        slip during arm motion. The grasp's ``epsilon_outer`` is sized
        dynamically to the available finger travel so any object up to
        ``max_width`` registers as a successful grasp.
        """
        in_flight: threading.Thread | None = None
        last_sent: float | None = None

        while not self._gripper_stop.is_set():
            with self._gripper_command_lock:
                target = self._gripper_target_width

            # Idle when target hasn't moved meaningfully since our last
            # dispatch (avoids spamming the gripper at the policy rate).
            if target is None or (last_sent is not None and abs(target - last_sent) <= _GRIPPER_MOVE_THRESHOLD):
                time.sleep(0.002)
                continue

            # New target: pre-empt any in-flight command so the next one
            # isn't queued behind libfranka's timeout.
            if in_flight is not None and in_flight.is_alive():
                try:
                    self._gripper.stop()
                except Exception as e:
                    en.logger.debug(f"[franka_panda] gripper.stop() during preemption: {e}")
                # Short bound: stop() should let the in-flight call
                # return within ~100 ms. If it doesn't, dispatch anyway
                # rather than starving the user; libfranka will serialize.
                in_flight.join(timeout=0.2)

            new_target = float(target)
            is_close = self.use_grasp_on_close and new_target < self.close_width_threshold

            if is_close:
                # Size epsilon_outer to span the remaining travel so the
                # firmware registers a successful grasp for any object
                # whose width is between ``new_target`` and ``max_width``.
                eps_outer = max(
                    float(self.gripper_epsilon_outer),
                    float(self._gripper_max_width - new_target),
                )
                eps_inner = float(self.gripper_epsilon_inner)
                force = float(self.gripper_force)

                def _dispatch(
                    width: float = new_target,
                    speed: float = float(self.gripper_speed),
                    f: float = force,
                    ei: float = eps_inner,
                    eo: float = eps_outer,
                ) -> None:
                    try:
                        self._gripper.grasp(
                            width=width,
                            speed=speed,
                            force=f,
                            epsilon_inner=ei,
                            epsilon_outer=eo,
                        )
                    except Exception as e:
                        en.logger.warning(f"[franka_panda] gripper.grasp({width:.3f}, force={f:.1f}) failed: {e}")

            else:

                def _dispatch(
                    width: float = new_target,
                    speed: float = float(self.gripper_speed),
                ) -> None:
                    try:
                        self._gripper.move(width=width, speed=speed)
                    except Exception as e:
                        en.logger.warning(f"[franka_panda] gripper.move({width:.3f}) failed: {e}")

            in_flight = threading.Thread(target=_dispatch, daemon=True)
            in_flight.start()
            last_sent = new_target
            time.sleep(0.002)


# Same backend drives both Panda and FR3; eden deploy picks via
# ``resolve_deployment_for(robot_options)``.
bind_deployment(FrankaEmikaPanda, "franka_panda_deployment")
bind_deployment(FrankaResearch3, "franka_panda_deployment")
# The bare arm of a composite rig (FrankaXHand, ...): the same backend with no gripper DOFs.
bind_deployment(FrankaEmikaPandaNoHand, "franka_panda_deployment")
