"""Manipulation action adapters for remote policy clients.

An adapter converts a raw action array returned by a policy server into the
``{controller_name: torch.Tensor}`` dict that Eden's action manager expects.
It is the mirror image of the server-specific obs packer: the packer serializes
Eden's world state for the server; the adapter deserializes the server's action
chunk back into Eden's controller contract.

The adapters here are transport-agnostic on purpose — they are composed into
per-setup strategies (OpenPI ``pi/``, GR00T, RLDX-1, Cosmos3, GalaxeaVLA) but nothing
about them is wire-specific, so any client policy that receives a numpy action vector
can reuse them. They are the *manipulation* family's decoders; other families add
their own module beside this one. The module also carries the manipulation-side
composition helpers shared by the setups (image observation groups, the WidowX EE-delta
action pair, robot DOF layouts read from the robot option classes) and the construction
checks every client runs against the env it is given (``check_*``: the control rate, the
gripper decision term, the plain proprio group, joint terms that pass commands through,
the DROID arm term and joint state, LIBERO's camera resolution).

Four adapter shapes cover the setups that use them:

* :class:`EEDeltaAdapter` — 7-D ``[ee_delta(6), gripper(1)]`` for the WidowX/Bridge and
  Google-Robot setups (OpenPI, GR00T, Octo). Euler or axis-angle rotation deltas; both reach
  the IK controller as a rotation vector.
* :class:`JointVelocityAdapter` — ``(n_arm + 1)``-D joint velocity + gripper for
  OpenPI DROID. Emits true velocity commands by default, with an optional
  position-integration fallback for arms without a usable velocity controller.
* :class:`JointPositionAdapter` — ``(n_arm + 1)``-D absolute joint position
  targets + gripper for DROID joint-position checkpoints (OpenPI, Cosmos3, GalaxeaVLA).
* :class:`BimanualJointPositionAdapter` — 14-D joint positions split into two
  arm controllers for Aloha.

None of them decides the gripper: each forwards the server's gripper scalar to the
env's ``GripperDecisionController``, which applies the setup's rule (threshold,
hysteresis, sticky latch). See :func:`check_env_gripper_term`.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

import eden as en
from eden.utils.geom import (
    R_to_quat,
    axis_angle_from_quat,
    euler_xyz_from_quat,
    inv_quat,
    quat_from_euler_xyz,
    quat_mul,
    transform_by_quat,
)
from eden.utils.image import to_uint8_hwc
from eden.utils.torch import broadcast_to_envs

from eden.options.robots.franka_emika_panda import FrankaResearch3
from eden.options.robots.robotiq_2f85 import Robotiq2f85
from eden.options.robots.trossen_widowx250 import WidowX250

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


# ---------------------------------------------------------------------------
# Adapter base classes
# ---------------------------------------------------------------------------


def _per_env(values, env: "EnvBase") -> torch.Tensor:
    """``(D,)`` values -> ``(num_envs, D)`` float32 on the env's device.

    Copied, never aliasing the caller's action chunk (``torch.as_tensor`` would share a CPU
    numpy row's memory), then expanded over the envs without a further copy.
    """
    return broadcast_to_envs(torch.tensor(values, dtype=torch.float32, device=env.device), env.num_envs)


class ActionAdapter(ABC):
    """A server action row -> ``{action_term: tensor}`` decoder.

    ``bind(env)`` runs once, when the setup that owns the adapter is built: it resolves and
    validates everything the adapter reads from the env (entity, DOF names), so a bad handle
    raises at construction instead of on the first control step. ``reset()`` clears
    per-episode state. Both default to no-ops.
    """

    #: Width ``D`` of the ``(D,)`` row :meth:`__call__` accepts.
    row_width: int

    def bind(self, env: "EnvBase") -> None:
        """Resolve and validate what this adapter reads from ``env`` (default: nothing)."""

    def reset(self) -> None:
        """Clear per-episode state (default: nothing)."""

    @abstractmethod
    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        """Decode one ``(row_width,)`` server row into ``{action_term: (num_envs, ...)}``."""

    def _check_row(self, raw_action: np.ndarray) -> np.ndarray:
        row = np.asarray(raw_action, dtype=np.float32)
        if row.shape != (self.row_width,):
            raise ValueError(
                f"{type(self).__name__} expects an action row of shape ({self.row_width},), got {row.shape}."
            )
        return row


class _JointAdapterBase(ActionAdapter):
    """Shared pieces of the joint-space adapters: ``(n_arm + 1)``-D rows, arm then gripper.

    The gripper: the server's scalar is forwarded to the env's ``GripperDecisionController``, which
    makes the open/close decision (threshold, hysteresis, latch), so the adapter only reshapes it to
    ``(num_envs, 1)`` and records the gripper diagnostics evaluators plot. ``gripper_threshold`` here
    only labels the intent in those diagnostics.
    """

    _debug_label = "joint"

    def __init__(
        self,
        *,
        n_arm: int = 7,
        gripper_threshold: float = 0.5,
        debug_gripper: bool = False,
        arm_key: str = "ee_controller",
        gripper_key: str = "gripper_controller",
    ):
        if not isinstance(n_arm, int) or isinstance(n_arm, bool) or n_arm <= 0:
            raise ValueError(f"n_arm must be a positive integer, got {n_arm!r}")
        self.n_arm = n_arm
        self.row_width = n_arm + 1
        self.gripper_threshold = gripper_threshold
        self.debug_gripper = debug_gripper
        self.arm_key = arm_key
        self.gripper_key = gripper_key
        self._debug_step = 0
        self.last_diagnostics: dict[str, np.ndarray | float | str] | None = None

    def reset(self) -> None:
        self._debug_step = 0
        self.last_diagnostics = None

    def _forward_gripper(self, g_raw: float, env: "EnvBase") -> tuple[torch.Tensor, dict[str, float | str]]:
        intent = "close" if g_raw > self.gripper_threshold else "open"
        if self.debug_gripper:
            en.logger.info(
                f"[{self._debug_label} gripper] step={self._debug_step:04d} raw={g_raw:.4f} "
                f"threshold={self.gripper_threshold:.4f} intent={intent}"
            )
            self._debug_step += 1
        diagnostics = {
            "raw_gripper_action": g_raw,
            "gripper_threshold": float(self.gripper_threshold),
            "gripper_command": g_raw,
            "gripper_intent": intent,
        }
        return _per_env([g_raw], env), diagnostics


# ---------------------------------------------------------------------------
# EE-delta adapter (widowx_bridge, google_robot)
# ---------------------------------------------------------------------------


class EEDeltaAdapter(ActionAdapter):
    """Adapter for 7-D end-effector delta actions.

    Input shape: ``(7,)`` — ``[dx, dy, dz, drx, dry, drz, gripper]``.

    Output dict:
      * ``ee_controller``  — ``(num_envs, 6)``  ``[dxyz, rotation vector]``: the task-space delta
        :class:`DifferentialIKController` solves ``J @ dq = delta`` for, whose angular rows are an
        angular displacement (axis * angle), not Euler angles.
      * ``gripper_controller`` — ``(num_envs, 1)``  the server's gripper scalar, unchanged; the env's
        ``GripperDecisionController`` interprets it (openness / sign / sticky latch, per setup).

    Parameters
    ----------
    rotation : {"euler", "axis_angle"}
        Rotation convention of the input delta slice ``raw[3:6]``. ``"axis_angle"`` is already a
        rotation vector and passes through unchanged. ``"euler"`` (extrinsic xyz roll / pitch / yaw)
        is converted to the rotation vector of the same rotation, as SimplerEnv does
        (``euler2axangle``) before it steps the arm. WidowX / Google-Robot servers output Euler
        (LIBERO's axis-angle deltas go to robosuite ``OSC_POSE``, not through this adapter).
    """

    row_width = 7

    def __init__(self, *, rotation: Literal["euler", "axis_angle"] = "euler"):
        if rotation not in ("euler", "axis_angle"):
            raise ValueError(f"rotation must be 'euler' or 'axis_angle', got {rotation!r}")
        self.rotation = rotation

    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        row = self._check_row(raw_action)
        delta = torch.tensor(row[:6], dtype=torch.float32, device=env.device)
        if self.rotation == "euler":
            delta[3:6] = axis_angle_from_quat(quat_from_euler_xyz(*delta[3:6]))
        delta = broadcast_to_envs(delta, env.num_envs)
        manager = getattr(env, "action_manager", None)
        if manager is not None and "ee_controller" in manager.terms:
            delta = ee_delta_to_world(delta, manager.terms["ee_controller"].entity.get_quat())
        return {
            "ee_controller": delta,
            "gripper_controller": _per_env(row[6:], env),
        }


def ee_delta_to_world(delta_b: torch.Tensor, base_quat_w: torch.Tensor) -> torch.Tensor:
    """A robot-base-frame ``(num_envs, 6)`` ``[dxyz, rotation vector]`` delta in the world frame.

    EE-delta servers (Bridge, Fractal, RobotArena) compose their deltas in the robot base frame, while
    :class:`DifferentialIKController` solves against the world-frame Jacobian: the two agree only for a
    base with identity rotation. Both parts are displacements, so the base's world orientation
    ``base_quat_w`` (``(num_envs, 4)`` wxyz, the controlled entity's ``get_quat()``) alone maps them.
    """
    return torch.cat(
        [transform_by_quat(delta_b[:, :3], base_quat_w), transform_by_quat(delta_b[:, 3:6], base_quat_w)], dim=-1
    )


# ---------------------------------------------------------------------------
# Joint-velocity adapter (droid)
# ---------------------------------------------------------------------------


def _first_order_poles(tau_s: np.ndarray, dc_gain: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """``H(z) = b / (1 - a z^-1)`` with ``a = exp(-dt / tau)``, ``b = dc * (1 - a)``."""
    a = np.exp(-dt / tau_s)
    return a, dc_gain * (1.0 - a)


def _per_joint_params(params: dict, keys: tuple[str, ...], n_arm: int, what: str) -> dict[str, np.ndarray]:
    """``params[key]`` as finite ``(n_arm,)`` float64 arrays; a missing key or a bad shape raises."""
    missing = sorted(set(keys) - set(params))
    if missing:
        raise ValueError(f"{what} must provide {list(keys)}; missing {missing}")
    out = {}
    for key in keys:
        values = np.asarray(params[key], dtype=np.float64)
        if values.shape != (n_arm,):
            raise ValueError(f"{what}['{key}'] must have shape ({n_arm},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{what} values must all be finite")
        out[key] = values
    return out


class _FirstOrderLag:
    r"""First-order lag on the *commanded* velocity (``velocity_lag``).

    Reproduces the real DROID controller's finite bandwidth, which Genesis's
    velocity actuator does not have: measured time constant is ~210-320 ms on
    the real robot versus ~11-48 ms in sim, while the DC gains agree (~0.94
    vs ~0.99). The sim/real gap in velocity control is *bandwidth*, not gain,
    which is why a scalar ``action_scale`` is not the right correction.

    Because Genesis tracks commanded velocity essentially deadbeat over one
    policy period, ``v_achieved ~= v_sent``; shaping the command with the real
    transfer function therefore makes the *achieved* profile match:

    .. math::
        y[t] = a\, y[t-1] + b\, u[t], \quad
        a = e^{-dt/\tau}, \quad b = \mathrm{dc}\,(1 - a)

    ``a`` and ``b`` are derived from ``dt`` on every call so the filter stays
    rate-correct for whatever period separates two adapter calls (the constants
    were fit at 15 Hz). For a :class:`~eden.extensions.policies.client.ClientMixin`
    client that period is ``env.dt``, since it executes one action row per env step;
    an evaluator that holds one action across several env steps (e.g. 0.064 s on a
    20 ms env) passes its own, longer period.
    """

    def __init__(self, params: dict, n_arm: int):
        p = _per_joint_params(params, ("tau_s", "dc_gain"), n_arm, "velocity_lag")
        if not np.all(p["tau_s"] > 0.0):
            raise ValueError("velocity_lag['tau_s'] must be strictly positive (seconds)")
        self.tau_s, self.dc_gain = p["tau_s"], p["dc_gain"]
        self.reset()

    def reset(self) -> None:
        self._output = np.zeros_like(self.tau_s)

    def __call__(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        a, b = _first_order_poles(self.tau_s, self.dc_gain, dt)
        self._output = a * self._output + b * velocity.astype(np.float64)
        return self._output.astype(np.float32)


class _RealOverSimCompensation:
    r"""``H_real / H_sim`` for fitted first-order plant models (``velocity_compensation``).

    For ``H = b / (1 - a z^-1)``, the causal command filter is
    ``(b_real / b_sim) * (1 - a_sim z^-1) / (1 - a_real z^-1)``.
    """

    def __init__(self, params: dict, n_arm: int):
        p = _per_joint_params(
            params, ("real_tau_s", "real_dc_gain", "sim_tau_s", "sim_dc_gain"), n_arm, "velocity_compensation"
        )
        if not np.all(p["real_tau_s"] > 0.0) or not np.all(p["sim_tau_s"] > 0.0):
            raise ValueError("velocity_compensation time constants must be positive")
        if not np.all(p["sim_dc_gain"] > 0.0) or not np.all(p["real_dc_gain"] > 0.0):
            raise ValueError("velocity_compensation DC gains must be positive")
        self.params = p
        self.reset()

    def reset(self) -> None:
        self._previous_input = np.zeros_like(self.params["real_tau_s"])
        self._previous_output = np.zeros_like(self.params["real_tau_s"])

    def __call__(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        p = self.params
        real_a, real_b = _first_order_poles(p["real_tau_s"], p["real_dc_gain"], dt)
        sim_a, sim_b = _first_order_poles(p["sim_tau_s"], p["sim_dc_gain"], dt)
        current_input = velocity.astype(np.float64)
        self._previous_output = real_a * self._previous_output + (real_b / sim_b) * (
            current_input - sim_a * self._previous_input
        )
        self._previous_input = current_input
        return self._previous_output.astype(np.float32)


class JointVelocityAdapter(_JointAdapterBase):
    """Adapter for joint-velocity + gripper actions (DROID).

    Pi0 DROID returns ``(T, 8)`` chunks where the first 7 dims are joint velocities in
    **DROID's unit action space**, not rad/s, and the last dim is a gripper command
    binarized at 0.5 on the robot side (here by the env's ``GripperDecisionController``).
    openpi's ``examples/droid/main.py`` clips every dim element-wise to ``[-1, 1]`` before
    ``RobotEnv.step`` (which leaves DROID's own ``v / max|v|`` rescale nothing to do), and
    DROID's ``RobotIKSolver.joint_velocity_to_delta`` turns a unit velocity into
    ``max_joint_delta`` = 0.2 rad per 15 Hz control step, i.e. 3 rad/s per unit. That
    conversion is the caller's ``action_scale`` (``DroidSetup`` passes 3.0 with ``clip=1.0``
    and ``dt=1/15`` s); the default ``action_scale=1.0`` is a pass-through and is not DROID's.

    In ``mode="velocity"`` (default), the arm output is ``action_scale * clip(v)``, the rad/s
    velocity command consumed by :class:`ImplicitVelocityController`.

    In ``mode="velocity_integrate"``, this
    adapter integrates velocities to joint-position targets on the client side::

        target_pos = current_pos + dt * action_scale * clip(v, -1, 1)

    and the integrated target is consumed by a standard
    :class:`ImplicitPDController`.

    Limitation: client-side integration assumes the downstream position
    controller tracks the reference cleanly. Stiff Franka-like gains usually
    do, but over long horizons small integration errors accumulate; prefer
    ``mode="velocity"`` wherever the arm has an ``ImplicitVelocityController``.

    Parameters
    ----------
    n_arm : int
        Number of arm joints (length of the velocity slice in ``raw_action``).
    entity_name : str
        Eden entity name used to read current joint positions for integration.
    arm_dofs_name : list[str] | None
        Ordered DOF names under ``entity_name``, one per arm joint. Must match the
        ``dofs_name`` list on the downstream arm controller. Required by the integration
        modes (resolved on the env in :meth:`bind`); unused in ``mode="velocity"``.
    dt : float | None
        Integration timestep (seconds). If ``None``, uses ``env.dt``.
        Only used by integration modes.
    mode : {"velocity", "velocity_integrate"}
        ``velocity`` emits rad/s commands directly. ``velocity_integrate`` integrates them
        into position targets (the fallback above).
    action_scale : float | list[float]
        Rad/s per unit of the clipped arm output, applied before it is sent to the
        downstream controller (scalar, or one value per arm joint). DROID's is 3.0:
        0.2 rad per 1/15 s control step (``DROID_RAD_PER_S_PER_UNIT`` in the pi setups).
        The default ``1.0`` passes the clipped value through, which is right only for a
        server whose arm output is already rad/s.
    clip : float
        Each arm dim is clipped element-wise to ``[-clip, clip]`` *before* scaling, in the
        server's units (DROID: openpi's ``[-1, 1]`` unit-space clip, so at most 3 rad/s
        per joint after the 3.0 scale).
    velocity_lag : dict | None
        Optional ``{"tau_s": (n_arm,), "dc_gain": (n_arm,)}`` first-order lag
        applied to the commanded velocity in ``mode="velocity"``, modelling the
        real robot's controller bandwidth (Genesis's velocity actuator is ~10x
        faster than DROID's). ``None`` (default) is an exact no-op. See
        :class:`_FirstOrderLag`.
    velocity_compensation : dict | None
        Optional ``{"real_tau_s", "real_dc_gain", "sim_tau_s", "sim_dc_gain"}`` (each
        ``(n_arm,)``): the ``H_real / H_sim`` command filter, in ``mode="velocity"``.
        Mutually exclusive with ``velocity_lag``.
    velocity_lag_dt : float | None
        The period between two adapter calls in seconds, required when ``velocity_lag``
        or ``velocity_compensation`` is set. A :class:`~eden.extensions.policies.client.ClientMixin`
        client executes one action row per env step, so for it this is ``env.dt`` (``DroidSetup``
        pins that to DROID's 1/15 s and passes it). Only an evaluator that holds one action
        across several env steps has a longer one (e.g. 0.064 s on a 20 ms env), which is
        why it is passed rather than read from the env.
    gripper_threshold : float
        Closedness above which the diagnostics label the gripper intent "close". The scalar
        itself is forwarded unchanged; the env's ``GripperDecisionController`` decides.
    debug_gripper : bool
        Log each gripper scalar and its intent.
    arm_key / gripper_key : str
        Keys in the returned dict — must match the controller names
        registered in the env's action manager.
    """

    _debug_label = "joint velocity"

    def __init__(
        self,
        *,
        n_arm: int = 7,
        entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        dt: float | None = None,
        mode: Literal["velocity", "velocity_integrate"] = "velocity",
        action_scale: float | list[float] = 1.0,
        clip: float = 1.0,
        velocity_lag: dict[str, "Sequence[float]"] | None = None,
        velocity_compensation: dict[str, "Sequence[float]"] | None = None,
        velocity_lag_dt: float | None = None,
        gripper_threshold: float = 0.5,
        debug_gripper: bool = False,
        arm_key: str = "ee_controller",
        gripper_key: str = "gripper_controller",
    ):
        super().__init__(
            n_arm=n_arm,
            gripper_threshold=gripper_threshold,
            debug_gripper=debug_gripper,
            arm_key=arm_key,
            gripper_key=gripper_key,
        )
        if mode not in ("velocity", "velocity_integrate"):
            raise ValueError(f"JointVelocityAdapter.mode must be 'velocity' or 'velocity_integrate', got {mode!r}")
        self.mode = mode
        self.entity_name = entity_name
        self.arm_dofs_name = None if arm_dofs_name is None else list(arm_dofs_name)
        if mode != "velocity":
            if self.arm_dofs_name is None:
                raise ValueError(
                    f"JointVelocityAdapter(mode={mode!r}) integrates from the arm's current joint positions, so "
                    f"it needs arm_dofs_name: the {n_arm} arm joint names on {entity_name!r}, in the arm "
                    "controller's dofs_name order."
                )
            if len(self.arm_dofs_name) != n_arm:
                raise ValueError(
                    f"JointVelocityAdapter: arm_dofs_name names {len(self.arm_dofs_name)} joints "
                    f"({self.arm_dofs_name}), but n_arm={n_arm}. Fix: pass exactly one name per arm joint."
                )
        if dt is not None and (not np.isfinite(dt) or dt <= 0.0):
            raise ValueError(f"dt must be a positive finite action period, got {dt!r}")
        self.dt = None if dt is None else float(dt)

        # Shape () or (n_arm,): broadcasts over the arm velocities either way.
        self.action_scale = np.array(action_scale, dtype=np.float32)
        if self.action_scale.shape not in ((), (n_arm,)):
            raise ValueError(
                "action_scale must be a scalar or have one value per arm joint; "
                f"expected shape ({n_arm},), got {self.action_scale.shape}"
            )
        if not np.all(np.isfinite(self.action_scale)):
            raise ValueError("action_scale must contain only finite values")
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError(f"clip must be positive and finite, got {clip!r}")
        self.clip = float(clip)

        self._velocity_filter = self._make_velocity_filter(velocity_lag, velocity_compensation, velocity_lag_dt)
        self.velocity_lag_dt = None if velocity_lag_dt is None else float(velocity_lag_dt)
        self._arm_dofs_idx_local: torch.Tensor | None = None

    def _make_velocity_filter(
        self, velocity_lag, velocity_compensation, velocity_lag_dt
    ) -> _FirstOrderLag | _RealOverSimCompensation | None:
        """The optional command filter; malformed parameters fail here rather than mid-rollout."""
        if velocity_lag is None and velocity_compensation is None:
            return None
        if velocity_lag is not None and velocity_compensation is not None:
            raise ValueError("velocity_lag and velocity_compensation are mutually exclusive")
        velocity_filter = (
            _FirstOrderLag(velocity_lag, self.n_arm)
            if velocity_lag is not None
            else _RealOverSimCompensation(velocity_compensation, self.n_arm)
        )
        if self.mode != "velocity":
            raise ValueError(
                f"velocity filtering models the robot's velocity-controller bandwidth and only "
                f"applies to mode='velocity'; got mode={self.mode!r}"
            )
        if velocity_lag_dt is None or not np.isfinite(velocity_lag_dt) or velocity_lag_dt <= 0.0:
            # Not defaulted to env.dt: that is the period between calls only for a client that
            # executes one action per env step; an evaluator holding each action for several
            # env steps (e.g. 0.064 s on a 20 ms env) would get a silently mis-tuned lag.
            raise ValueError(
                "velocity filtering requires velocity_lag_dt = the policy action period in "
                "seconds between adapter calls (env.dt for a ClientMixin client, e.g. 1/15 for DROID)"
            )
        return velocity_filter

    def bind(self, env: "EnvBase") -> None:
        """Resolve the arm joints the integration modes read, once, against ``env``.

        Only the derived indices are cached; the entity is looked up on every call, since
        deployment can swap ``env.entities[entity_name]``.
        """
        if self.mode == "velocity":
            return
        if self.entity_name not in env.entities:
            raise ValueError(
                f"JointVelocityAdapter.entity_name={self.entity_name!r} is not an entity of the env. "
                f"Available: {sorted(env.entities)}"
            )
        selection = env.entities[self.entity_name].find_named_dofs(self.arm_dofs_name)
        if len(selection.names) != self.n_arm:
            raise ValueError(
                f"JointVelocityAdapter: arm_dofs_name {self.arm_dofs_name} selects {len(selection.names)} joints "
                f"on {self.entity_name!r} ({list(selection.names)}), but n_arm={self.n_arm}. Fix: name each arm "
                "joint exactly (a glob can match more joints than intended)."
            )
        self._arm_dofs_idx_local = selection.dofs_idx_local

    def reset(self) -> None:
        super().reset()
        # Stateful filter over the episode's command history - must not leak
        # across episode boundaries.
        if self._velocity_filter is not None:
            self._velocity_filter.reset()

    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        row = self._check_row(raw_action)
        if not np.all(np.isfinite(row)):
            raise ValueError("raw_action must contain only finite values")

        clipped = np.clip(row[: self.n_arm], -self.clip, self.clip)
        prefilter_velocity = clipped * self.action_scale  # target controller units, rad/s in velocity mode
        if self.mode == "velocity":
            velocity = prefilter_velocity
            if self._velocity_filter is not None:
                velocity = self._velocity_filter(prefilter_velocity, self.velocity_lag_dt)
            arm_action = _per_env(velocity, env)
        else:
            if self._arm_dofs_idx_local is None:
                raise RuntimeError(
                    f"JointVelocityAdapter(mode={self.mode!r}) was called before bind(env). The owning setup binds "
                    "its adapter when it is built; a hand-built adapter must call adapter.bind(env) once first."
                )
            velocity = prefilter_velocity
            dt = self.dt if self.dt is not None else env.dt
            current = env.entities[self.entity_name].get_dofs_pos(dofs_idx_local=self._arm_dofs_idx_local)
            arm_action = current + _per_env(velocity * dt, env)  # (num_envs, n_arm)

        gripper, gripper_diagnostics = self._forward_gripper(float(row[self.n_arm]), env)
        self.last_diagnostics = {
            "mode": self.mode,
            "raw_arm_action": row[: self.n_arm].copy(),
            "clipped_arm_action": clipped,
            "prefilter_target_velocity": prefilter_velocity,
            "target_velocity": velocity,
            "velocity_lag_enabled": isinstance(self._velocity_filter, _FirstOrderLag),
            "velocity_compensation_enabled": isinstance(self._velocity_filter, _RealOverSimCompensation),
            "action_scale": self.action_scale.copy(),
            "clip": self.clip,
            **gripper_diagnostics,
        }
        return {self.arm_key: arm_action, self.gripper_key: gripper}


# ---------------------------------------------------------------------------
# Joint-position adapter (droid_jointpos)
# ---------------------------------------------------------------------------


class JointPositionAdapter(_JointAdapterBase):
    """Adapter for absolute joint-position + gripper actions.

    Pi0.5 DROID joint-position checkpoints are trained in delta joint-position
    space, but OpenPI's output transform reconstructs absolute joint targets by
    adding the current state back before the client receives the chunk. This
    adapter therefore passes the first ``n_arm`` dimensions directly to an
    ``ImplicitPDController`` configured with ``reference_source=ZERO``, and the
    gripper scalar to the env's ``GripperDecisionController``. Shared by OpenPI's
    ``droid_jointpos``, Cosmos3 and GalaxeaVLA's DROID-Franka setup.
    """

    _debug_label = "joint position"

    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        row = self._check_row(raw_action)
        # Not checked for finiteness: a NaN gripper is a deliberate "no command" (GalaxeaVLA omits the key).
        target_pos = row[: self.n_arm]
        gripper, gripper_diagnostics = self._forward_gripper(float(row[self.n_arm]), env)
        self.last_diagnostics = {
            "mode": "joint_position",
            "raw_arm_action": target_pos.copy(),
            "target_position": target_pos.copy(),
            **gripper_diagnostics,
        }
        return {self.arm_key: _per_env(target_pos, env), self.gripper_key: gripper}


# ---------------------------------------------------------------------------
# Bimanual joint-position adapter (aloha)
# ---------------------------------------------------------------------------


class BimanualJointPositionAdapter(ActionAdapter):
    """Adapter for 14-D bimanual joint-position actions (Aloha).

    Splits a ``(left_n + right_n)``-D action vector into its two arms' slices and emits
    them under separate controller keys.
    """

    def __init__(
        self,
        *,
        left_n: int = 7,
        right_n: int = 7,
        left_key: str = "left_arm_controller",
        right_key: str = "right_arm_controller",
    ):
        self.left_n = left_n
        self.row_width = left_n + right_n
        self.left_key = left_key
        self.right_key = right_key

    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        row = self._check_row(raw_action)
        return {
            self.left_key: _per_env(row[: self.left_n], env),
            self.right_key: _per_env(row[self.left_n :], env),
        }


# ---------------------------------------------------------------------------
# Manipulation-side composition helpers shared by the setups
# ---------------------------------------------------------------------------

#: WidowX-250s arm DOFs in Eden's MJCF declaration order (from the robot options class).
WIDOWX_DOFS: tuple[str, ...] = tuple(WidowX250.arm_dofs_name)

#: Franka / DROID 7-DOF arm joints in MuJoCo XML declaration order (from the robot options class).
DROID_ARM_DOFS: tuple[str, ...] = tuple(FrankaResearch3.arm_dofs_name)

#: DROID's control rate (``RobotEnv.control_hz``): every DROID checkpoint's action row is one 1/15 s step.
DROID_CONTROL_HZ: float = 15.0
#: ``(sim_dt as typed, decimation)`` for a 15 Hz DROID env, as PolaRiS's ``droid_cfg.py`` sets it
#: (``sim.dt = 1/120``, ``decimation = 8``): the example timing of a DROID client's :func:`check_control_rate`.
DROID_CONTROL_TIMING: tuple[str, int] = ("1/120", 8)
#: Appended to :data:`DROID_CONTROL_TIMING` in that refusal.
DROID_CONTROL_TIMING_NOTE: str = (
    " as PolaRiS does, with sim_substeps=9 to keep the Robotiq 2F85's substep under 1 ms (Eden's PolaRiS tasks "
    "step every 20 ms)"
)

#: Robotiq 2F-85 driver position at the close target (both drivers close to the same value).
ROBOTIQ_CLOSE_POS: float = Robotiq2f85.close_dofs_pos["left_driver_joint"]

#: Tool offset from Eden's FR3 ``attachment`` link to DROID's ``cartesian_position`` frame (Euler
#: xyz, right-multiplied). ``attachment`` sits on the flange origin rotated +135 deg about z
#: (``fr3.xml``), and DROID's ``cartesian_position`` is the flange (``panda_link8``) pose: Panda FK
#: of ``joint_position`` over the ``droid_100`` episodes reproduces it to 0.000 mm / 0.000 deg, and
#: is exactly 45 deg off ``panda_hand``. GR00T and RLDX-1 (``droid_deploy.py`` feeds
#: ``cartesian_position`` through unchanged) both expect this frame.
DROID_FLANGE_FROM_FR3_ATTACHMENT_EULER: tuple[float, float, float] = (0.0, 0.0, -0.75 * np.pi)

#: Bridge/WidowX top-down EE convention. EE poses in Bridge data are expressed relative to
#: a top-down pose, not the robot base. See
#: https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ec/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
BRIDGE_DEFAULT_ROT = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


#: Inverse of :data:`BRIDGE_DEFAULT_ROT` as a wxyz quaternion.
_BRIDGE_DEFAULT_QUAT_INV = inv_quat(R_to_quat(torch.as_tensor(BRIDGE_DEFAULT_ROT)))


def bridge_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Bridge / SimplerEnv WidowX ``rpy``: the EE rotation relative to :data:`BRIDGE_DEFAULT_ROT`, xyz Euler.

    ``R @ BRIDGE_DEFAULT_ROT.T`` as quaternions, decomposed into extrinsic xyz Euler angles.
    """
    quat = torch.as_tensor(quat_wxyz, dtype=torch.float64)
    return torch.stack(euler_xyz_from_quat(quat_mul(quat, _BRIDGE_DEFAULT_QUAT_INV))).numpy()


def robosuite_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Robosuite's ``quat2axisangle`` for a wxyz quaternion, as LIBERO's official clients compute EE state.

    Both OpenPI (``examples/libero/main.py``) and Isaac-GR00T (``gr00t/eval/sim/LIBERO/libero_env.py``)
    copy it. Kept robosuite-exact -- no sign canonicalisation (the angle may exceed pi when w < 0), hard
    zero at the singularity -- rather than routed through :func:`eden.utils.geom.axis_angle_from_quat` or
    ``Rotation.as_rotvec``, because the servers' normalisation statistics were fit on robosuite's output.
    """
    w, x, y, z = quat_wxyz
    w = float(np.clip(w, -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    angle = 2.0 * math.acos(w)
    return np.array([x, y, z]) * angle / den


def libero_task_action_options() -> dict[str, "ActionTermOptions"]:
    """The stock ``libero`` task's action terms, from the task's own builder (no restated constants).

    ``osc_pose`` (robosuite ``OSC_POSE`` on the grip site) and ``gripper`` (the stateful
    ``LiberoPandaGripperController``, ``+1`` closes / ``-1`` opens): the controllers LIBERO's
    demonstrations, and so every LIBERO checkpoint (OpenPI, GR00T), were recorded with.
    """
    from eden.tasks.libero.config import _make_action_options

    options = _make_action_options()
    # term_keys(), not model_extra: the extras also carry the _option_module_ / _option_class_ markers.
    return {name: getattr(options, name) for name in options.term_keys()}


#: ``(width, height)`` both official LIBERO clients render the cameras at: openpi ``examples/libero/main.py``
#: ``LIBERO_ENV_RESOLUTION = 256`` ("resolution used to render training data") and Isaac-GR00T
#: ``gr00t/eval/sim/LIBERO/libero_env.py`` ``OffScreenRenderEnv(camera_heights=256, camera_widths=256)``. The stock
#: ``libero`` task renders 128x128, so an env for either client overrides its cameras' ``cam_resolution``.
LIBERO_CAMERA_RESOLUTION: tuple[int, int] = (256, 256)


def check_libero_camera_resolution(env, *, subject: str, trained_on: str, wrong_fix: str) -> None:
    """Refuse an env whose ``image`` / ``wrist_image`` groups render at another size than LIBERO's 256x256.

    The camera is the one each group's term names in ``params["camera_name"]`` (:func:`make_image_groups`);
    a camera term without one is judged by the image its group publishes. ``subject`` names the client and
    setup, ``trained_on`` the upstream render the checkpoints saw, and ``wrong_fix`` is the closing "Do not
    ..." sentence about the client's own image-size knob. Envs without an observation manager skip.
    """
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return
    width, height = LIBERO_CAMERA_RESOLUTION
    for group, name, term in manager.iter_terms():
        if group not in ("image", "wrist_image"):
            continue
        camera = (getattr(term, "params", None) or {}).get("camera_name")
        if camera is None:
            rows, cols = manager.group_obs_dim[group][:2]
            resolution, where, camera = (
                (int(cols), int(rows)),
                f"the camera behind observation term {group}/{name}",
                "<camera>",
            )
        else:
            resolution = tuple(int(v) for v in env.cameras[camera].cam_resolution)
            where = f"camera {camera!r} (observation term {group}/{name})"
        if resolution == LIBERO_CAMERA_RESOLUTION:
            continue
        raise ValueError(
            f"{subject} sends {where}, which renders at cam_resolution={resolution}, but {trained_on}, so the frame "
            f"would be upsampled off that distribution. Fix: in the env's cameras_options set the {camera!r} camera's "
            f"cam_resolution=({width}, {height}), e.g. cfg.cameras_options.{camera}.cam_resolution = ({width}, "
            f"{height}). {wrong_fix}"
        )


def libero_training_view(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """An upright Eden render as LIBERO's training images show it: mirrored left-right, ``(H, W, 3)`` uint8.

    LIBERO's official clients send ``obs["agentview_image"][::-1, ::-1]`` and the same for
    ``robot0_eye_in_hand_image`` (OpenPI ``examples/libero/main.py`` lines 115-116, Isaac-GR00T
    ``gr00t/eval/sim/LIBERO/libero_env.py`` lines 148-149), as the LIBERO datasets were recorded. robosuite
    1.4's raw frame is OpenGL bottom-up (the upright view flipped vertically), so that pair of flips leaves the
    upright view mirrored left-right. Eden renders upright, so only the left-right mirror reproduces it;
    porting ``[::-1, ::-1]`` literally would turn the image upside down.
    """
    return np.ascontiguousarray(to_uint8_hwc(image)[:, ::-1])


def dofs_pos_joint_names(env: "EnvBase", term) -> list[str]:
    """The joints a built ``DofsPos`` term reads, in its column order."""
    entity = env.entities[term.entity_name]
    if term.dofs_name is not None:
        return list(entity.find_named_dofs(list(term.dofs_name)).names)
    if term.terms_idx is not None:
        return [entity.dofs_name[idx] for idx in term.terms_idx]
    return list(entity.dofs_name)


def gripper_dofs_from_name(gripper_dof_name: str) -> list[str]:
    """Expand a single Robotiq 2F-85 driver name to both actuated drivers; other names pass through."""
    if gripper_dof_name in Robotiq2f85.actuated_dofs_name:
        return list(Robotiq2f85.actuated_dofs_name)
    return [gripper_dof_name]


def make_image_groups(
    camera_name: str,
    use_wrist_image: bool,
    wrist_camera_name: str,
) -> dict[str, ObservationGroupOptions]:
    """Build ``image`` (and optionally ``wrist_image``) observation groups, one ``rgb`` term each."""
    from eden.managers.terms.observations.camera import rgb_image
    from eden.options.managers.observations import ObservationGroupOptions
    from eden.terms import ObsTerm

    groups: dict[str, ObservationGroupOptions] = {
        "image": ObservationGroupOptions(
            rgb=ObsTerm.configure(func=rgb_image, params={"camera_name": camera_name}),
        ),
    }
    if use_wrist_image:
        groups["wrist_image"] = ObservationGroupOptions(
            rgb=ObsTerm.configure(func=rgb_image, params={"camera_name": wrist_camera_name}),
        )
    return groups


def make_ee_pose_group(
    entity_name: str,
    ee_link_name: str,
    *,
    gripper_entity_name: str | None,
    gripper_dof_name: str,
    open_gripper_pos: float = WidowX250.open_finger_pos,
    gripper_close_pos: float = ROBOTIQ_CLOSE_POS,
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ObservationGroupOptions:
    """``[ee_pos(3), ee_quat_wxyz(4), gripper_openness(1)]`` proprio group (Bridge / Google-robot / RobotArena).

    With the gripper on the arm entity, openness is ``pos / open_gripper_pos`` (WidowX fingers:
    fully open at ``open_gripper_pos``, closed at 0). With a separately mounted gripper it is
    ``1 - pos / gripper_close_pos`` (Robotiq driver: open at 0, closed at ``gripper_close_pos``).
    """
    from eden.managers.terms.observations.manipulation import EndEffectorPose, GripperState
    from eden.options.managers.observations import ObservationGroupOptions

    if gripper_entity_name is None:  # WidowX fingers on the arm: open at open_gripper_pos, closed at 0
        gripper_entity_name, open_pos, close_pos = entity_name, open_gripper_pos, 0.0
    else:  # a mounted Robotiq: open at 0, closed at gripper_close_pos
        open_pos, close_pos = 0.0, gripper_close_pos
    return ObservationGroupOptions(
        ee_pose=EndEffectorPose.configure(
            entity_name=entity_name, ee_link_name=ee_link_name, rotation="quat", offset_pos=tuple(offset_pos)
        ),
        gripper=GripperState.configure(
            entity_name=gripper_entity_name,
            dofs_name=[gripper_dof_name],
            normalize="openness",
            open_pos=open_pos,
            close_pos=close_pos,
        ),
    )


def make_droid_joint_state_group(
    arm_entity_name: str,
    arm_dofs_name: list[str] | tuple[str, ...],
    *,
    gripper_entity_name: str | None = None,
    gripper_dof_name: str,
    gripper_close_pos: float = ROBOTIQ_CLOSE_POS,
) -> ObservationGroupOptions:
    """``[joint_pos(N), gripper_closedness(1)]`` proprio group for DROID-style joint-state servers.

    The last value is DROID's ``robot_state["gripper_position"]``: closedness ``pos / gripper_close_pos``
    clipped to ``[0, 1]`` (a Robotiq driver can travel past its close target), read from
    ``gripper_dof_name`` on ``gripper_entity_name`` -- the arm entity when that is ``None``.
    """
    from eden.managers.terms.observations.manipulation import GripperState
    from eden.managers.terms.observations.proprio import DofsPos
    from eden.options.managers.observations import ObservationGroupOptions

    return ObservationGroupOptions(
        joint_pos=DofsPos.configure(entity_name=arm_entity_name, dofs_name=list(arm_dofs_name)),
        gripper=GripperState.configure(
            entity_name=gripper_entity_name if gripper_entity_name is not None else arm_entity_name,
            dofs_name=[gripper_dof_name],
            normalize="closedness",
            open_pos=0.0,
            close_pos=gripper_close_pos,
            clip=(0.0, 1.0),
        ),
    )


def make_widowx_ee_delta_actions(
    entity_name: str,
    ee_link_name: str,
    dofs_name: list[str],
    scale: float,
    *,
    gripper_input: str = "openness",
    gripper_threshold: float = 0.5,
    gripper_sticky_repeats: int = 0,
    gripper_zero_holds: bool = False,
) -> dict[str, ActionTermOptions]:
    """``ee_controller`` (differential IK) + ``gripper_controller`` for WidowX EE-delta policies.

    The gripper term is a :class:`~eden.managers.terms.actions.binary_actions.GripperDecisionController`
    that receives the server's raw gripper scalar (Bridge servers emit *openness*, ``> 0.5`` =
    open) and decides open/close itself; ``gripper_sticky_repeats`` enables the fractal latch
    the Google-robot checkpoints were trained with.
    """
    from eden.managers.terms.actions.binary_actions import GripperDecisionController
    from eden.managers.terms.actions.task_space_actions import DifferentialIKController

    return dict(
        ee_controller=DifferentialIKController.configure(
            entity_name=entity_name,
            ee_link_name=ee_link_name,
            dofs_name=dofs_name,
            position_scale=scale,
            orientation_scale=scale,
        ),
        gripper_controller=GripperDecisionController.configure(
            entity_name=entity_name,
            dofs_name=list(WidowX250.gripper_dofs_name),
            open_action={"left_finger": WidowX250.open_finger_pos, "right_finger": -WidowX250.open_finger_pos},
            close_action=0.0,
            input=gripper_input,
            threshold=gripper_threshold,
            sticky_repeats=gripper_sticky_repeats,
            zero_holds=gripper_zero_holds,
        ),
    )


def make_droid_joint_position_actions(
    arm_entity_name: str = "robot",
    arm_dofs_name: list[str] | None = None,
    *,
    gripper_entity_name: str | None = None,
    gripper_dof_name: str = "left_driver_joint",
    arm_key: str = "ee_controller",
    **gripper_options,
) -> dict[str, ActionTermOptions]:
    """Absolute DROID arm joint targets + the Robotiq gripper term, as joint-position DROID servers need.

    The servers return absolute joint targets, so the arm is an ``ImplicitPDController`` with a zero
    reference. The gripper is :func:`make_droid_gripper_action` on ``gripper_entity_name`` (the arm
    entity when the Robotiq is part of it), configured by ``gripper_options``.
    """
    from eden.constants import ReferenceSource
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    arm = ImplicitPDController.configure(
        entity_name=arm_entity_name,
        dofs_name=list(DROID_ARM_DOFS) if arm_dofs_name is None else list(arm_dofs_name),
        reference_source=ReferenceSource.ZERO,
    )
    gripper = make_droid_gripper_action(
        gripper_entity_name if gripper_entity_name is not None else arm_entity_name,
        gripper_dof_name,
        **gripper_options,
    )
    return {arm_key: arm, "gripper_controller": gripper}


def make_droid_gripper_action(
    gripper_entity_name: str,
    gripper_dof_name: str,
    *,
    gripper_close_pos: float = ROBOTIQ_CLOSE_POS,
    input: str = "closedness",
    threshold: float = 0.5,
    open_threshold: float | None = None,
    close_threshold: float | None = None,
    hold_steps: int = 0,
    continuous: bool = False,
    snap_threshold: float | None = None,
    on_missing: str = "hold",
) -> ActionTermOptions:
    """``gripper_controller`` for DROID-style servers: closedness in, ``[0, gripper_close_pos]`` targets out.

    The open/close decision (threshold, hysteresis, hold, continuous) is made by the term, not
    the client. The single Robotiq driver name expands to both actuated drivers.
    """
    from eden.managers.terms.actions.binary_actions import GripperDecisionController

    return GripperDecisionController.configure(
        entity_name=gripper_entity_name,
        dofs_name=gripper_dofs_from_name(gripper_dof_name),
        open_action=0.0,
        close_action=gripper_close_pos,
        input=input,
        threshold=threshold,
        open_threshold=open_threshold,
        close_threshold=close_threshold,
        hold_steps=hold_steps,
        continuous=continuous,
        snap_threshold=snap_threshold,
        on_missing=on_missing,
    )


def check_control_rate(
    env: "EnvBase",
    *,
    control_hz: float,
    subject: str,
    source: str,
    timing: tuple[str, int] | None = None,
    timing_note: str = "",
    wrong_fix: str | None = None,
) -> None:
    """Refuse an env whose control period ``env.dt`` is not the upstream client's ``1 / control_hz`` s.

    A remote checkpoint's rows are one upstream control step each (an EE or joint delta, the next waypoint
    of a trajectory), and ``exec_horizon`` / a video window's stride count env steps, so at another rate
    every chunk executes at the wrong speed, silently. ``subject`` names the client and setup
    (``"PiPolicy.policy_setup='droid'"``) and ``source`` where upstream sets the rate. ``timing`` is the
    ``(sim_dt as typed, decimation)`` the fix suggests, with ``timing_note`` appended (e.g. the
    ``sim_substeps`` a Robotiq 2F85 needs); ``None`` derives one from the env's ``sim_dt``. ``wrong_fix`` replaces
    the closing "do not rescale or resample the actions" sentence for a client with a more plausible wrong fix.
    """
    env_dt = float(env.dt)
    period = 1.0 / control_hz
    if math.isclose(env_dt, period, rel_tol=1e-6):
        return
    speed = f"{period / env_dt:.2f}x too fast" if env_dt < period else f"{env_dt / period:.2f}x too slow"
    if timing is None:
        decimation = max(1, round(period / env.sim_dt))
        exact = math.isclose(env.sim_dt * decimation, period, rel_tol=1e-6)
        timing = (f"{env.sim_dt!r}" if exact else f"1/{control_hz * decimation:g}", decimation)
    sim_dt, decimation = timing
    if wrong_fix is None:
        wrong_fix = (
            f"Do not rescale or resample the actions instead: the checkpoint's chunk is a trajectory at "
            f"{control_hz:g} Hz, and a faster env would also re-query the server that much more often."
        )
    raise ValueError(
        f"{subject} executes one action row per {period:.6f} s ({control_hz:g} Hz) control step, as {source}, but "
        f"the env steps every env.dt={env_dt:.6f} s (sim_dt * decimation), which would run each chunk {speed}. Fix: "
        f"build the env with sim_dt * decimation = 1/{control_hz:g} s, e.g. EnvOptions(sim_dt={sim_dt}, "
        f"decimation={decimation}){timing_note}. {wrong_fix}"
    )


GRIPPER_DECISION_FIELDS = (
    "input",
    "scale",
    "threshold",
    "open_threshold",
    "close_threshold",
    "hold_steps",
    "sticky_repeats",
    "zero_holds",
    "continuous",
    "snap_threshold",
)
"""The :class:`GripperDecisionController` fields that decide what a gripper command does."""


def check_env_gripper_term(env, term_name: str, **expected) -> None:
    """Fail fast when the env's built gripper term disagrees with settings a policy also carries.

    Some client policies expose gripper knobs (thresholds, hold, decode mode) that the env's
    :class:`GripperDecisionController` also carries. When the env is built (it always is before a
    policy is constructed in the drivers), the two are compared so a mismatch is reported at
    setup time rather than silently ignored. Envs without an action manager (unit tests) skip.

    Every field of :data:`GRIPPER_DECISION_FIELDS` is compared: one the caller does not pass in
    ``expected`` must be at the controller's default, because hysteresis, a hold, a sticky latch or
    continuous mode each replace the plain threshold rule the caller relies on.

    Clients forward the server's raw gripper scalar, so any other gripper term (e.g. a task
    config's own ``BinaryJointController``) would silently misread it: a missing term, a term of
    another type, or an expected field the term does not have all raise.
    """
    from eden.managers.terms.actions.binary_actions import GripperDecisionController

    manager = getattr(env, "action_manager", None)
    if manager is None:
        return
    term = manager.terms.get(term_name)
    if term is None:
        raise ValueError(
            f"The env has no action term {term_name!r}; available: {sorted(manager.terms)}. Build the env "
            "with the policy's action_options() so its GripperDecisionController receives the gripper command."
        )
    if not isinstance(term, GripperDecisionController):
        raise TypeError(
            f"Action term {term_name!r} is a {type(term).__name__}, but this policy forwards the server's raw "
            "gripper scalar and needs a GripperDecisionController to interpret it. Build the env's gripper "
            "term from the policy's action_options()."
        )
    unknown = sorted(name for name in expected if not hasattr(term, name))
    if unknown:
        raise AttributeError(f"Gripper term {term_name!r} has no field(s) {unknown} to check against the policy.")
    undeclared = {
        name: getattr(GripperDecisionController, name) for name in GRIPPER_DECISION_FIELDS if name not in expected
    }
    mismatched = {
        name: (getattr(term, name), value)
        for name, value in {**expected, **undeclared}.items()
        if getattr(term, name) != value
    }
    if mismatched:
        details = ", ".join(
            f"{k}: env term has {a!r}, policy {'has' if k in expected else 'does not use it (default)'} {b!r}"
            for k, (a, b) in mismatched.items()
        )
        raise ValueError(
            f"Gripper term {term_name!r} was configured differently from the policy ({details}). "
            "The decision now lives in the env's GripperDecisionController: build the env's gripper term from "
            "the policy's action_options(), not from another client's or a task's own gripper term."
        )


def check_plain_observation_group(
    env, group: str, *, width: int | None = None, scales: dict[str, Sequence[float]] | None = None, fix: str
) -> None:
    """Fail fast when an observation group a client slices by position is not the plain current reading.

    Clients pack a group by column (``proprio[:7]`` is the arm), so the group must be one flat
    ``(num_envs, width)`` tensor of this step's values: no history window (it puts older frames
    first), no ``scale``/``clip``/``modifiers`` (a delay or a rescale reaches the server unannounced),
    and no noise when the group corrupts. ``scales`` names the terms whose post-compute ``scale`` is
    part of the wire contract (term -> the exact per-column scale the factory sets, e.g. robosuite's
    second-finger sign flip): such a term must exist and carry exactly that scale, every other term
    a scale of 1. ``fix`` is appended to the message and should name the factory that builds the
    group. Envs without an observation manager (unit tests) skip.
    """
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return
    groups = env.config.observation_options
    if group not in groups.term_keys():
        raise ValueError(
            f"The env has no observation group {group!r}; available: {sorted(groups.term_keys())}. Fix: {fix}"
        )
    group_options = getattr(groups, group)
    problems = []
    if not group_options.concatenate_terms:
        problems.append("concatenate_terms=False (the client needs one tensor)")
    if group_options.history_length is not None:
        problems.append(f"history_length={group_options.history_length}")
    from eden.managers.terms.observations.manipulation import GripperState

    built = {name: obs_term for g, name, obs_term in manager.iter_terms() if g == group}
    scales = scales or {}
    term_names = manager.group_term_names[group]
    problems.extend(
        f"no term {name!r} (it carries scale={tuple(scales[name])!r})" for name in scales if name not in term_names
    )
    for name in term_names:
        term = getattr(group_options, name)
        if term.history_length:
            problems.append(f"{name}.history_length={term.history_length}")
        scale = np.asarray(term.scale, dtype=float)
        if name in scales:
            if not np.array_equal(scale, np.asarray(scales[name], dtype=float)):
                problems.append(f"{name}.scale={term.scale!r} (expected {tuple(scales[name])!r})")
        elif not bool(np.all(scale == 1.0)):
            problems.append(f"{name}.scale={term.scale!r}")
        # A [0, 1] clip on a normalised gripper reading only bounds an overshoot past the stroke ends
        # (make_droid_joint_state_group sets it); any other clip changes values the server reads.
        unit_gripper_clip = (
            isinstance(built.get(name), GripperState)
            and built[name].normalize != "raw"
            and tuple(term.clip or ()) == (0.0, 1.0)
        )
        if term.clip is not None and not unit_gripper_clip:
            problems.append(f"{name}.clip={term.clip!r}")
        if term.modifiers:
            problems.append(f"{name}.modifiers={[type(m).__name__ for m in term.modifiers]}")
        if term.noise is not None and group_options.enable_corruption:
            problems.append(f"{name}.noise={type(term.noise).__name__}")
    actual = manager.group_obs_dim[group]
    if width is not None and not problems and tuple(actual) != (width,):
        problems.append(f"its terms {manager.group_term_names[group]} are {tuple(actual)} wide, not ({width},)")
    if problems:
        raise ValueError(
            f"Observation group {group!r} is not the plain current reading the client packs by column: "
            f"{'; '.join(problems)}. A history or a post-compute transform shifts or rescales what the server "
            f"reads. Fix: {fix}"
        )


def check_gripper_state_matches_action(env, *, group: str, term: str, action_term: str) -> None:
    """Fail fast when a gripper state term and the gripper action term describe different grippers.

    A client sends the state normalised over ``[open_pos, close_pos]`` and the env maps the server's
    command onto ``[open_action, close_action]``; the two come from separate ``observation_options``
    and ``action_options`` arguments, so a stroke set in one call only would skew every request. The
    state must read a DOF of the action term's entity, and its open / close positions must equal the
    action term's (per-DOF dict targets are compared for that DOF). Envs without managers skip.
    """
    obs_manager = getattr(env, "observation_manager", None)
    action_manager = getattr(env, "action_manager", None)
    if obs_manager is None or action_manager is None:
        return
    from eden.managers.terms.observations.manipulation import GripperState

    state = next((t for g, n, t in obs_manager.iter_terms() if (g, n) == (group, term)), None)
    action = action_manager.terms.get(action_term)
    if not isinstance(state, GripperState) or action is None:
        return  # The callers' own layout and gripper-term checks name a missing or foreign term.
    problems = []
    if state.entity_name != action.entity_name or not set(state.dofs_name) <= set(action.dofs_name):
        problems.append(
            f"it reads {state.entity_name!r} {list(state.dofs_name)} but {action_term!r} drives "
            f"{action.entity_name!r} {list(action.dofs_name)}"
        )
    dof = state.dofs_name[0] if state.dofs_name else None
    for side, state_pos, target in (
        ("open", state.open_pos, action.open_action),
        ("close", state.close_pos, action.close_action),
    ):
        target = target.get(dof) if isinstance(target, dict) else target
        if state_pos is not None and target is not None and not math.isclose(state_pos, target, abs_tol=1e-6):
            problems.append(f"{side}_pos={state_pos!r} but {action_term!r} has {side}_action={target!r}")
    if problems:
        raise ValueError(
            f"Gripper state {group}/{term} and action term {action_term!r} describe different grippers: "
            f"{'; '.join(problems)}. Fix: pass the same gripper entity and the same open/close positions to "
            "the policy's observation_options() and action_options()."
        )


def _pass_through_problems(term) -> list[str]:
    """What keeps joint action ``term`` from executing the raw command as is, or ``[]``.

    The term's command is ``raw * scale + offset + reference``, and it has to be ``raw`` itself: a zero reference
    (``ReferenceSource.ZERO``), unit effective scale, a zero configured offset (an all-zero per-joint dict is the
    identity) and no ``set_dofs_pos_offset()`` offset on top.
    """
    from eden.constants import ReferenceSource

    problems = []
    # str(): the term keeps a plain str when configured with one or reloaded from JSON, the enum otherwise.
    reference_source = str(term.reference_source)
    if reference_source != ReferenceSource.ZERO:
        problems.append(
            f"reference_source={reference_source!r}, which adds a reference to every command (needs "
            "ReferenceSource.ZERO)"
        )
    scale = torch.as_tensor(term.effective_scale, dtype=torch.float32)
    if not torch.all(scale == 1.0):
        problems.append(
            f"effective_scale={scale.flatten().tolist()} (from scale={term.scale!r}, scale_ratio={term.scale_ratio!r}, "
            "needs 1)"
        )
    # The configured offset, not effective_offset: a 'default' reference folds default_dofs_pos into the latter,
    # which the reference_source problem above already names.
    configured_offset = term.offset.values() if isinstance(term.offset, dict) else (term.offset,)
    if any(value != 0.0 for value in configured_offset):
        problems.append(f"offset={term.offset!r} (needs 0)")
    elif reference_source == ReferenceSource.ZERO and torch.any(term.effective_offset != 0.0):
        # Per-env after domain randomization: summarized, not listed (num_envs x num_dofs values).
        offset = term.effective_offset
        problems.append(
            f"effective_offset up to {offset.abs().max().item():.4g} in magnitude, shape {tuple(offset.shape)} (row 0: "
            f"{[round(v, 6) for v in offset[0].tolist()]}), from a set_dofs_pos_offset() call (needs 0)"
        )
    return problems


def _refuse_term_problems(
    term, problems: list[str], *, term_name: str, subject: str, fix: str, commands: str, wrong_fix: str
) -> None:
    if problems:
        raise ValueError(
            f"{subject} sends {commands} to {term_name!r} to be executed as they are, but that "
            f"{type(term).__name__} has {'; '.join(problems)}. Fix: build the env's action_options from {fix}. "
            f"{wrong_fix}"
        )


#: The closing sentence of a pass-through refusal unless the caller names a more specific wrong fix.
_PASS_THROUGH_WRONG_FIX = (
    "Do not undo the difference in the client (subtracting the reference or dividing by the scale): the term must "
    "execute the reply as is."
)
#: :data:`_PASS_THROUGH_WRONG_FIX` for a DROID arm term, which can also be refused for its joint order.
_DROID_ARM_WRONG_FIX = (
    "Do not undo the difference in the client (subtracting the reference, dividing by the scale or reordering the "
    "rows): the term must execute the reply as is."
)


def check_joint_pass_through(
    term,
    *,
    term_name: str,
    subject: str,
    fix: str,
    commands: str = "the server's absolute joint targets",
    wrong_fix: str = _PASS_THROUGH_WRONG_FIX,
) -> None:
    """Raise unless joint action ``term`` executes the client's commands untouched (see :func:`_pass_through_problems`).

    ``subject`` names the client and setup (``"PiPolicy.policy_setup='aloha'"``); ``commands`` what it sends
    (absolute targets by default); ``fix`` the ``action_options(...)`` call that builds the right term; and
    ``wrong_fix`` the client-side workaround to rule out. The caller has already checked the term's type. Every
    problem is listed at once.
    """
    _refuse_term_problems(
        term,
        _pass_through_problems(term),
        term_name=term_name,
        subject=subject,
        fix=fix,
        commands=commands,
        wrong_fix=wrong_fix,
    )


def check_droid_arm_term(
    arm,
    *,
    term_name: str,
    subject: str,
    fix: str,
    commands: str = "the server's absolute joint targets",
    wrong_fix: str = _DROID_ARM_WRONG_FIX,
) -> None:
    """:func:`check_joint_pass_through` for a DROID arm term, which must also take DROID's joints in DROID's order.

    That is one column per DROID arm joint, and DROID's own joint names in DROID's order (a rig with other names is
    checked by width only): the server reads row ``i`` as DROID's joint ``i``.
    """
    problems = _pass_through_problems(arm)
    if arm.action_dim != len(DROID_ARM_DOFS):
        problems.append(f"width {arm.action_dim} (DROID's arm has {len(DROID_ARM_DOFS)} joints)")
    dofs = list(arm.dofs_name)
    if set(dofs) <= set(DROID_ARM_DOFS) and dofs != [name for name in DROID_ARM_DOFS if name in dofs]:
        problems.append(f"dofs_name={dofs} (needs {list(DROID_ARM_DOFS)}, in that order)")
    _refuse_term_problems(
        arm, problems, term_name=term_name, subject=subject, fix=fix, commands=commands, wrong_fix=wrong_fix
    )


def check_droid_joint_state(env: "EnvBase", arm, *, group: str, subject: str, fix: str) -> None:
    """Raise unless ``group`` is ``[DofsPos of arm's joints in its order, GripperState closedness]``.

    A DROID client sends ``group[:7]`` as ``joint_position`` and ``group[7]`` as ``gripper_position``, and the
    server replies commands for those joints in that order, which the adapter hands column by column to ``arm``'s
    ``dofs_name``. ``observation_options()`` and ``action_options()`` take the arm entity and DOF names separately,
    so the two can differ without either failing to build. The joint angles must be absolute (no
    ``offset_from_default``); ``fix`` names the factory calls that build matching terms.
    """
    from eden.managers.terms.observations.manipulation import GripperState
    from eden.managers.terms.observations.proprio import DofsPos

    terms = {name: term for g, name, term in env.observation_manager.iter_terms() if g == group}
    arm_dofs = list(arm.dofs_name)
    problems = []
    if list(terms) != ["joint_pos", "gripper"]:
        problems.append(f"it holds the terms {list(terms)}, not ['joint_pos', 'gripper'] in that order")
    else:
        joints, gripper = terms["joint_pos"], terms["gripper"]
        if not isinstance(joints, DofsPos):
            problems.append(f"'joint_pos' is a {type(joints).__name__}, not a DofsPos")
        else:
            read = dofs_pos_joint_names(env, joints)
            if (joints.entity_name, read) != (arm.entity_name, arm_dofs):
                problems.append(
                    f"'joint_pos' reads {joints.entity_name!r} {read} but the arm term drives {arm.entity_name!r} "
                    f"{arm_dofs}, so each joint would be sent another joint's command"
                )
            if joints.offset_from_default:
                problems.append("'joint_pos' has offset_from_default=True (the server needs absolute joint angles)")
        if not isinstance(gripper, GripperState) or gripper.normalize != "closedness":
            problems.append(
                f"'gripper' is a {type(gripper).__name__} with normalize={getattr(gripper, 'normalize', None)!r}, "
                "not a GripperState normalized to 'closedness' (DROID's gripper_position, 1 = closed)"
            )
    if problems:
        raise ValueError(
            f"{subject} sends {group}[:{len(arm_dofs)}] as DROID's joint_position and {group}[{len(arm_dofs)}] as its "
            "gripper_position, and the server replies commands for those joints in that order, but the env's "
            f"{group!r} group does not match: {'; '.join(problems)}. Fix: {fix}. Do not reorder the server's rows in "
            "the client instead: the state it is sent and the commands it returns share one joint order."
        )
