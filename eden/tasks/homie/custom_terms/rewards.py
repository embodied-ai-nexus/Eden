"""Reward terms of the HOMIE task with no Eden built-in equivalent.

Ports of OpenHomie's G1 reward functions (``HomieRL/legged_gym/legged_gym/envs/base/legged_robot.py``,
``_reward_*``). Weights live in :mod:`eden.tasks.homie.config` and are OpenHomie's ``rewards.scales``
verbatim — the reward manager multiplies the weighted sum by ``dt``, as legged_gym does. A term
whose plain name a built-in already owns (``feet_air_time``, ``feet_slip``, …) carries a ``Homie``
prefix: the term registry is global, and the semantics differ.

DOF ordering: ``get_dofs_*`` / ``default_dofs_pos`` / ``get_dofs_limit`` are in declaration order
(``robot.dofs_name`` == ``G1_DOFS29_NAME``), so a joint subset is selected by its position in
``dofs_name`` (:func:`_dofs_name_idx`), never by ``find_named_dofs_idx_local``'s kinematic-tree
indices. Link quantities use :func:`resolve_link_idx`.
"""

from __future__ import annotations

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.geom import inv_quat, transform_by_quat

# command[:, 3] (the absolute base-height target, m) at or above this counts as standing —
# OpenHomie's ``commands[:, 4] >= 0.735`` gate on the squat-relaxed penalties.
HEIGHT_STAND_GATE: float = 0.735

# Sole sample points in the ``*_ankle_roll_link`` frame, from the G1 MJCF foot collision capsules
# (sole plane z ≈ -0.025, x ∈ [-0.05, 0.13], y ∈ ±0.026). OpenHomie samples several ``*_foot``
# marker bodies per foot; Eden's G1 foot is one link, so these corners stand in for them in the
# foot-flatness and feet-parallel rewards. Order: front-left, front-right, back-left, back-right.
_FOOT_SOLE_CORNERS: list[tuple[float, float, float]] = [
    (0.13, 0.026, -0.025),
    (0.13, -0.026, -0.025),
    (-0.05, 0.026, -0.025),
    (-0.05, -0.026, -0.025),
]


def _dofs_name_idx(robot, substrings: list[str], device) -> torch.Tensor:
    """Positions in ``robot.dofs_name`` of the DOFs whose name contains any of ``substrings``."""
    idx = [i for i, n in enumerate(robot.dofs_name) if any(s in n for s in substrings)]
    if not idx:
        raise ValueError(f"No DOF of the robot matches any of {substrings!r}; its DOFs are {list(robot.dofs_name)!r}.")
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def _to_body_frame(robot, vecs: torch.Tensor) -> torch.Tensor:
    """Rotate per-link world vectors ``(N, F, 3)`` into the base frame."""
    n, f, _ = vecs.shape
    q_inv = inv_quat(robot.get_quat()).unsqueeze(1).expand(n, f, 4).reshape(-1, 4)
    return transform_by_quat(vecs.reshape(-1, 3), q_inv).reshape(n, f, 3)


def _links_in_body_frame(robot, links_idx: torch.Tensor) -> torch.Tensor:
    """Link positions relative to the base, in the base frame. ``(N, F, 3)``."""
    return _to_body_frame(robot, robot.get_links_pos(ls_idx_local=links_idx) - robot.get_pos().unsqueeze(1))


def _foot_corners_world(robot, feet_idx: torch.Tensor, corners_local: torch.Tensor) -> torch.Tensor:
    """World positions of each foot's sole corners. ``corners_local`` ``(C, 3)`` → ``(N, F, C, 3)``."""
    link_pos = robot.get_links_pos(ls_idx_local=feet_idx)
    link_quat = robot.get_links_quat(ls_idx_local=feet_idx)
    n, f, _ = link_pos.shape
    c = corners_local.shape[0]
    q = link_quat.unsqueeze(2).expand(n, f, c, 4).reshape(-1, 4)
    off = corners_local.view(1, 1, c, 3).expand(n, f, c, 3).reshape(-1, 3)
    return link_pos.unsqueeze(2) + transform_by_quat(off, q).reshape(n, f, c, 3)


def _standing(command: torch.Tensor) -> torch.Tensor:
    return (command[:, 3] >= HEIGHT_STAND_GATE).float()


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class _AxisVelTracking(RewardTerm):
    """``exp(-(cmd - v)^2 / sigma)`` on one body-frame linear-velocity axis."""

    command_name: str = ""
    entity_name: str = "robot"
    sigma: float = 0.25
    _axis: int = 0

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self.resolve_entity(self.entity_name)

    def compute(self) -> torch.Tensor:
        vel = self.resolve_entity(self.entity_name).get_vel(frame="body")[:, self._axis]
        return torch.exp(-torch.square(self._command.command[:, self._axis] - vel) / self.sigma)


@REWARD_TERM_REGISTRY.register()
class TrackingXVel(_AxisVelTracking):
    """Track the commanded forward velocity (OpenHomie ``tracking_x_vel``)."""

    _axis: int = 0


@REWARD_TERM_REGISTRY.register()
class TrackingYVel(_AxisVelTracking):
    """Track the commanded lateral velocity (OpenHomie ``tracking_y_vel``)."""

    _axis: int = 1


@REWARD_TERM_REGISTRY.register()
class TrackingBaseHeight(RewardTerm):
    """Track the commanded base height above the lower foot (OpenHomie ``tracking_base_height``).

    ``base_h = max(root_z - foot_z)`` over the feet; reward ``exp(-4 |base_h - cmd_h + ankle_sole_distance|)``.
    """

    command_name: str = ""
    entity_name: str = "robot"
    feet_link_names: list[str] = []
    ankle_sole_distance: float = 0.02

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self._feet_idx = resolve_link_idx(self.resolve_entity(self.entity_name), self.feet_link_names, self.device)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        foot_z = robot.get_links_pos(ls_idx_local=self._feet_idx)[:, :, 2]
        base_h = torch.max(robot.get_pos()[:, 2:3] - foot_z, dim=1).values
        return torch.exp(-4.0 * torch.abs(base_h - self._command.command[:, 3] + self.ankle_sole_distance))


# ---------------------------------------------------------------------------
# Velocity / orientation
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
class LinVelZGated(RewardTerm):
    """Vertical base velocity, penalized only while standing — a squat needs it (OpenHomie ``lin_vel_z``)."""

    command_name: str = ""
    entity_name: str = "robot"

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self.resolve_entity(self.entity_name)

    def compute(self) -> torch.Tensor:
        vz = self.resolve_entity(self.entity_name).get_vel(frame="body")[:, 2]
        return torch.square(vz) * _standing(self._command.command)


@REWARD_TERM_REGISTRY.register()
def ang_vel_xy(env, *, entity_name: str) -> torch.Tensor:
    """Base roll/pitch rate, ``sum(ang_vel_b[:2]^2)`` (OpenHomie ``ang_vel_xy``)."""
    return torch.sum(torch.square(env.entities[entity_name].get_ang(frame="body")[:, :2]), dim=1)


# ---------------------------------------------------------------------------
# Effort / smoothness
# ---------------------------------------------------------------------------


class _FullActionReward(RewardTerm):
    """OpenHomie's full action vector: the 12 policy leg actions and the upper-body curriculum's.

    OpenHomie's ``self.actions`` holds both (the curriculum's in normalized-action units too), and
    its ``action_rate`` / ``smoothness`` penalties span the whole vector. The curriculum term has
    ``action_dim = 0`` and exposes its applied pose as ``normalized_action``.
    """

    leg_action_term: str = "dofs_pos_controller"
    upper_action_term: str = "upper_body"

    def build(self) -> None:
        self._leg = self._env.action_manager.get_term(self.leg_action_term)
        self._upper = self._env.action_manager.get_term(self.upper_action_term)
        width = self._leg.action.shape[-1] + self._upper.normalized_action.shape[-1]
        self._prev = torch.zeros(self.num_envs, width, device=self.device)
        self._prev_prev = torch.zeros(self.num_envs, width, device=self.device)

    def _full_action(self) -> torch.Tensor:
        return torch.cat([self._leg.action, self._upper.normalized_action], dim=1)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        self._prev[idx] = 0.0
        self._prev_prev[idx] = 0.0


@REWARD_TERM_REGISTRY.register()
class HomieActionRate(_FullActionReward):
    """``sum((a_prev - a)^2)`` over the legs + upper body (OpenHomie ``action_rate``)."""

    def compute(self) -> torch.Tensor:
        full = self._full_action()
        out = torch.sum(torch.square(self._prev - full), dim=1)
        self._prev.copy_(full)
        return out


@REWARD_TERM_REGISTRY.register()
class HomieActionSmoothness(_FullActionReward):
    """``sum((a - 2 a_prev + a_prev_prev)^2)`` over the legs + upper body (OpenHomie ``smoothness``)."""

    def compute(self) -> torch.Tensor:
        full = self._full_action()
        out = torch.sum(torch.square(full - 2.0 * self._prev + self._prev_prev), dim=1)
        self._prev_prev.copy_(self._prev)
        self._prev.copy_(full)
        return out


@REWARD_TERM_REGISTRY.register()
class HomieDofsAcc(RewardTerm):
    """``sum(((v_prev - v) / dt)^2)`` over all DOFs (OpenHomie ``dof_acc``).

    The built-in ``DofsAccL2`` leaves out the ``1 / dt^2``; keeping it here keeps the weight at
    OpenHomie's -2.5e-7.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        robot = self.resolve_entity(self.entity_name)
        self._prev_vel = torch.zeros(self.num_envs, robot.num_dofs, device=self.device)

    def compute(self) -> torch.Tensor:
        vel = self.resolve_entity(self.entity_name).get_dofs_vel()
        out = torch.sum(torch.square((self._prev_vel - vel) / self._env.dt), dim=1)
        self._prev_vel.copy_(vel)
        return out

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self._prev_vel[slice(None) if envs_idx is None else envs_idx] = 0.0


@REWARD_TERM_REGISTRY.register()
class HomieDofsPosLimits(RewardTerm):
    """Joint positions past a ``soft_factor`` band of the hard limits, summed (OpenHomie ``dof_pos_limits``).

    OpenHomie's ``soft_dof_pos_limit`` is 0.975; the entity's own soft limits are fixed at 0.9.
    """

    entity_name: str = "robot"
    soft_factor: float = 0.975

    def build(self) -> None:
        lower, upper = self.resolve_entity(self.entity_name).get_dofs_limit()
        mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower) * self.soft_factor
        self._soft_lower, self._soft_upper = mid - half, mid + half

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self.entity_name).get_dofs_pos()
        return torch.sum((self._soft_lower - q).clamp(min=0.0) + (q - self._soft_upper).clamp(min=0.0), dim=1)


class _LegDofsReward(RewardTerm):
    """An effort penalty over the 12 leg DOFs.

    The waist and arms follow the upper-body curriculum, not the policy, so their effort stays
    out of these — OpenHomie's ``[:, :num_lower_dof]``.
    """

    entity_name: str = "robot"
    leg_substrings: list[str] = ["hip", "knee", "ankle"]

    def build(self) -> None:
        self._leg_idx = _dofs_name_idx(self.resolve_entity(self.entity_name), self.leg_substrings, self.device)


@REWARD_TERM_REGISTRY.register()
class HomieDofsVel(_LegDofsReward):
    """``sum(dof_vel^2)`` over the legs (OpenHomie ``dof_vel``)."""

    def compute(self) -> torch.Tensor:
        return torch.sum(torch.square(self.resolve_entity(self.entity_name).get_dofs_vel()[:, self._leg_idx]), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieDofsVelLimits(_LegDofsReward):
    """Leg joint speed past ``soft_vel_limit · max_vel``, summed (OpenHomie ``dof_vel_limits``).

    OpenHomie reads per-joint URDF velocity limits; this applies one representative limit (the
    weight is -2e-3, so the approximation is second order).
    """

    max_vel: float = 20.0  # rad/s
    soft_vel_limit: float = 0.8

    def compute(self) -> torch.Tensor:
        v = self.resolve_entity(self.entity_name).get_dofs_vel()[:, self._leg_idx].abs()
        return torch.sum((v - self.max_vel * self.soft_vel_limit).clamp(min=0.0), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieTorques(_LegDofsReward):
    """``sum((tau / kp)^2)`` over the legs, with the nominal kp (OpenHomie ``torques``)."""

    def build(self) -> None:
        super().build()
        # Read before any reset-mode kp DR has fired: OpenHomie normalizes by the nominal p_gains.
        self._kp = self.resolve_entity(self.entity_name).get_dofs_kp()[..., self._leg_idx].clone()

    def compute(self) -> torch.Tensor:
        tau = self.resolve_entity(self.entity_name).get_dofs_control_force()[:, self._leg_idx]
        return torch.sum(torch.square(tau / self._kp), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieTorqueLimits(_LegDofsReward):
    """Leg torque past ``soft_torque_limit`` of the effort limit, summed (OpenHomie ``torque_limits``)."""

    soft_torque_limit: float = 0.95

    def build(self) -> None:
        super().build()
        _, upper = self.resolve_entity(self.entity_name).get_dofs_force_range()
        self._limit = upper.abs()[..., self._leg_idx] * self.soft_torque_limit

    def compute(self) -> torch.Tensor:
        tau = self.resolve_entity(self.entity_name).get_dofs_control_force()[:, self._leg_idx]
        return torch.sum((tau.abs() - self._limit).clamp(min=0.0), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieJointPower(RewardTerm):
    """``sum(|v| |tau|) / clip(vx^2 + vy^2 + 0.2 wz^2, min=0.1)`` over all DOFs (OpenHomie ``joint_power``)."""

    command_name: str = ""
    entity_name: str = "robot"

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self.resolve_entity(self.entity_name)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        power = torch.sum(robot.get_dofs_vel().abs() * robot.get_dofs_control_force().abs(), dim=1)
        cmd = self._command.command
        return power / torch.clip(cmd[:, 0] ** 2 + cmd[:, 1] ** 2 + 0.2 * cmd[:, 2] ** 2, min=0.1)


@REWARD_TERM_REGISTRY.register()
class HomieJointTrackingError(RewardTerm):
    """``sum((target - q)^2)`` over the leg controller's DOFs (OpenHomie ``joint_tracking_error``).

    Reads the controller's own joint target, so it is exact whatever the action scale.
    """

    entity_name: str = "robot"
    action_term_name: str = "dofs_pos_controller"

    def build(self) -> None:
        self._action_term = self._env.action_manager.get_term(self.action_term_name)
        robot = self.resolve_entity(self.entity_name)
        # Columns: ``compute`` slices the full-width ``get_dofs_pos()`` read with them.
        self._idx = robot.find_named_dofs(self._action_term.dofs_name).columns

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self.entity_name).get_dofs_pos()[:, self._idx]
        return torch.sum(torch.square(self._action_term.processed_action - q), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieActionVanish(RewardTerm):
    """Raw leg actions that would drive a joint past its hard limit, summed (OpenHomie ``action_vanish``).

    The bounds are ``(hard_limit - default) / scale``, read off the controller's own scale.
    """

    entity_name: str = "robot"
    action_term_name: str = "dofs_pos_controller"

    def build(self) -> None:
        self._action_term = self._env.action_manager.get_term(self.action_term_name)
        robot = self.resolve_entity(self.entity_name)
        idx = robot.find_named_dofs(self._action_term.dofs_name).columns
        lower, upper = robot.get_dofs_limit()
        default = robot.default_dofs_pos[:, idx]
        scale = self._action_term.effective_scale
        self._action_min = (lower[..., idx] - default) / scale
        self._action_max = (upper[..., idx] - default) / scale

    def compute(self) -> torch.Tensor:
        raw = self._action_term.action
        return torch.sum((raw - self._action_max).clamp(min=0.0) + (self._action_min - raw).clamp(min=0.0), dim=1)


# ---------------------------------------------------------------------------
# Posture (gated on standing)
# ---------------------------------------------------------------------------


class _GatedDeviation(RewardTerm):
    """``sum((q - default)^2)`` over a joint subset, only while standing."""

    command_name: str = ""
    entity_name: str = "robot"
    joint_substrings: list[str] = []

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self._idx = _dofs_name_idx(self.resolve_entity(self.entity_name), self.joint_substrings, self.device)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        dev = torch.square(robot.get_dofs_pos() - robot.default_dofs_pos)[:, self._idx]
        return torch.sum(dev, dim=1) * _standing(self._command.command)


@REWARD_TERM_REGISTRY.register()
class DeviationHipJoint(_GatedDeviation):
    """Hip deviation from the default pose while standing (OpenHomie ``deviation_hip_joint``)."""

    joint_substrings: list[str] = ["hip"]


@REWARD_TERM_REGISTRY.register()
class DeviationAnkleJoint(_GatedDeviation):
    """Ankle-*roll* deviation while standing (OpenHomie ``deviation_ankle_joint``; pitch does the squat)."""

    joint_substrings: list[str] = ["ankle_roll"]


@REWARD_TERM_REGISTRY.register()
class DeviationKneeJoint(RewardTerm):
    """Knee flexion consistent with the commanded height (OpenHomie ``deviation_knee_joint``).

    Each knee is normalized to ``[0, 1]`` over its hard range, centered, and weighted by the height
    error ``root_z - cmd_h``; the absolute values are summed.
    """

    command_name: str = ""
    entity_name: str = "robot"

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        robot = self.resolve_entity(self.entity_name)
        self._idx = _dofs_name_idx(robot, ["knee"], self.device)
        lower, upper = robot.get_dofs_limit()
        self._lower = lower[..., self._idx]
        self._span = (upper[..., self._idx] - self._lower).clamp(min=1e-6)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        norm = (robot.get_dofs_pos()[:, self._idx] - self._lower) / self._span
        height_err = (robot.get_pos()[:, 2] - self._command.command[:, 3]).unsqueeze(1)
        return torch.sum(torch.abs((norm - 0.5) * height_err), dim=1)


# ---------------------------------------------------------------------------
# Feet / contact
# ---------------------------------------------------------------------------


class _FeetReward(RewardTerm):
    """Resolves the feet (and, when named, the command) for the feet rewards."""

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    command_name: str = ""

    def build(self) -> None:
        self._feet_idx = resolve_link_idx(self.resolve_entity(self.entity_name), self.feet_link_names, self.device)
        self._command = self._env.command_manager.get_term(self.command_name) if self.command_name else None

    def _feet_contact_force(self) -> torch.Tensor:
        """Per-foot net contact force, ``(N, F, 3)``."""
        return self.resolve_entity(self.entity_name).get_links_net_contact_force()[:, self._feet_idx, :]


class _ContactTrackedFeetReward(_FeetReward):
    """A feet reward on OpenHomie's per-step contact bookkeeping (``legged_robot.py`` ``post_physics_step``).

    Contact is ``‖F‖ > 1``, debounced as ``contact | last_contact``; ``first_contact`` reads the air
    time before it is incremented, and the air time of a foot in contact is zeroed after the reward
    reads it — deferred here to the start of the next step's update, which nothing reads in between.
    """

    contact_force_threshold: float = 1.0

    def build(self) -> None:
        super().build()
        n = int(self._feet_idx.numel())
        self._air_time = torch.zeros(self.num_envs, n, device=self.device)
        self._last_contact = torch.zeros(self.num_envs, n, dtype=torch.bool, device=self.device)
        self._prev_contact_filt = torch.zeros(self.num_envs, n, dtype=torch.bool, device=self.device)

    def _update_contact(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the trackers one step; return ``(first_contact, continue_contact)``."""
        dt = self._env.dt
        self._air_time.mul_((~self._prev_contact_filt).float())
        contact = torch.norm(self._feet_contact_force(), dim=-1) > self.contact_force_threshold
        contact_filt = contact | self._last_contact
        self._last_contact.copy_(contact)
        first_contact = (self._air_time >= dt) & contact_filt
        self._air_time.add_(dt)
        continue_contact = (self._air_time >= 3.0 * dt) & contact_filt
        self._prev_contact_filt.copy_(contact_filt)
        return first_contact, continue_contact

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        self._air_time[idx] = 0.0
        self._last_contact[idx] = False
        self._prev_contact_filt[idx] = False


@REWARD_TERM_REGISTRY.register()
class HomieFeetAirTime(_ContactTrackedFeetReward):
    """``sum((air_time - 0.5) · first_contact)`` under a nonzero command (OpenHomie ``feet_air_time``)."""

    command_threshold: float = 0.1

    def compute(self) -> torch.Tensor:
        first_contact, _ = self._update_contact()
        reward = torch.sum((self._air_time - 0.5) * first_contact.float(), dim=1)
        return reward * (torch.norm(self._command.command[:, :3], dim=1) > self.command_threshold).float()


@REWARD_TERM_REGISTRY.register()
class FeetGroundParallel(_ContactTrackedFeetReward):
    """Variance of each landed foot's sole-corner heights (OpenHomie ``feet_ground_parallel``).

    Counted for a foot airborne at least ``3 dt`` and now in (debounced) contact; a flat foot has
    equal corner heights.
    """

    def build(self) -> None:
        super().build()
        self._corners = torch.tensor(_FOOT_SOLE_CORNERS, device=self.device)

    def compute(self) -> torch.Tensor:
        _, continue_contact = self._update_contact()
        robot = self.resolve_entity(self.entity_name)
        var = _foot_corners_world(robot, self._feet_idx, self._corners)[..., 2].var(dim=-1)
        return torch.sum(var * continue_contact.float(), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieFeetSlip(_FeetReward):
    """``sum(‖v_foot,xy‖ · (F_z > 1))`` — no command gate (OpenHomie ``feet_slip``)."""

    contact_force_threshold: float = 1.0

    def compute(self) -> torch.Tensor:
        contact = (self._feet_contact_force()[..., 2] > self.contact_force_threshold).float()
        foot_vel = self.resolve_entity(self.entity_name).get_links_vel(ls_idx_local=self._feet_idx)
        return torch.sum(torch.norm(foot_vel[:, :, :2], dim=-1) * contact, dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieFeetClearance(_FeetReward):
    """Swing-height error weighted by lateral foot speed, near full height (OpenHomie ``feet_clearance``).

    ``sum((foot_h - target)^2 · ‖v_foot,xy‖)`` with ``foot_h`` the mean sole-corner height (clipped at
    0) and the foot velocity taken relative to the base, in the base frame; gated on ``cmd_h >= 0.71``.
    """

    clearance_target: float = 0.14
    height_gate: float = 0.71

    def build(self) -> None:
        super().build()
        self._corners = torch.tensor(_FOOT_SOLE_CORNERS, device=self.device)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        foot_h = _foot_corners_world(robot, self._feet_idx, self._corners)[..., 2].mean(dim=-1).clamp(min=0.0)
        rel_vel = robot.get_links_vel(ls_idx_local=self._feet_idx) - robot.get_vel().unsqueeze(1)
        vel_lat = torch.norm(_to_body_frame(robot, rel_vel)[:, :, :2], dim=-1)
        cost = torch.sum(torch.square(foot_h - self.clearance_target) * vel_lat, dim=1)
        return cost * (self._command.command[:, 3] >= self.height_gate).float()


@REWARD_TERM_REGISTRY.register()
class HomieFeetContactForces(_FeetReward):
    """Foot contact force past ``max_contact_force``, summed (OpenHomie ``feet_contact_forces``)."""

    max_contact_force: float = 400.0

    def compute(self) -> torch.Tensor:
        excess = torch.norm(self._feet_contact_force(), dim=-1) - self.max_contact_force
        return torch.sum(excess.clamp(min=0.0), dim=1)


@REWARD_TERM_REGISTRY.register()
class HomieContactMomentum(_FeetReward):
    """``sum(clip(v_foot,z, max=0) · clip(F_z - 50, min=0))`` — soft landings (OpenHomie ``contact_momentum``)."""

    force_offset: float = 50.0

    def compute(self) -> torch.Tensor:
        foot_vz = self.resolve_entity(self.entity_name).get_links_vel(ls_idx_local=self._feet_idx)[:, :, 2]
        fz = self._feet_contact_force()[..., 2]
        return torch.sum(torch.clip(foot_vz, max=0.0) * torch.clip(fz - self.force_offset, min=0.0), dim=1)


@REWARD_TERM_REGISTRY.register()
class FeetDistanceLateral(_FeetReward):
    """Lateral foot separation in ``[least, most]``, base frame (OpenHomie ``feet_distance_lateral``)."""

    least: float = 0.2
    most: float = 0.35

    def compute(self) -> torch.Tensor:
        body = _links_in_body_frame(self.resolve_entity(self.entity_name), self._feet_idx)
        dist = torch.abs(body[:, 0, 1] - body[:, 1, 1])
        too_close = torch.clamp(dist - self.least, max=0.0)
        too_wide = torch.clamp(self.most - dist, max=0.0) * _standing(self._command.command)
        return too_close + too_wide


@REWARD_TERM_REGISTRY.register()
class KneeDistanceLateral(RewardTerm):
    """Lateral knee + hip-yaw separation in ``[2 least, 2 most]`` (OpenHomie ``knee_distance_lateral``)."""

    entity_name: str = "robot"
    command_name: str = ""
    # Pairs (0, 2) and (1, 3) are compared.
    knee_link_names: list[str] = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
    least: float = 0.2
    most: float = 0.35

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self._knee_idx = resolve_link_idx(self.resolve_entity(self.entity_name), self.knee_link_names, self.device)

    def compute(self) -> torch.Tensor:
        y = _links_in_body_frame(self.resolve_entity(self.entity_name), self._knee_idx)[:, :, 1]
        dist = torch.abs(y[:, 0] - y[:, 2]) + torch.abs(y[:, 1] - y[:, 3])
        too_close = torch.clamp(dist - 2.0 * self.least, max=0.0)
        too_wide = torch.clamp(2.0 * self.most - dist, max=0.0) * _standing(self._command.command)
        return too_close + too_wide


@REWARD_TERM_REGISTRY.register()
class FeetParallel(_FeetReward):
    """Variance of three paired left/right sole-corner distances, while standing (OpenHomie ``feet_parallel``).

    Both feet use the same local sample points, so the paired distances are equal when the feet
    share an orientation and spread apart as one tilts against the other.
    """

    n_points: int = 3

    def build(self) -> None:
        super().build()
        self._corners = torch.tensor(_FOOT_SOLE_CORNERS[: self.n_points], device=self.device)

    def compute(self) -> torch.Tensor:
        corners = _foot_corners_world(self.resolve_entity(self.entity_name), self._feet_idx, self._corners)
        var = torch.norm(corners[:, 0] - corners[:, 1], dim=-1).var(dim=-1)
        return var * _standing(self._command.command)


@REWARD_TERM_REGISTRY.register()
class HomieNoFly(_FeetReward):
    """1 for exactly one foot down, or for any stance under a near-zero command (OpenHomie ``no_fly``)."""

    command_threshold: float = 0.1
    contact_force_threshold: float = 0.5

    def compute(self) -> torch.Tensor:
        contacts = self._feet_contact_force()[..., 2] > self.contact_force_threshold
        single = (contacts.float().sum(dim=1) == 1).float()
        near_zero_cmd = (torch.norm(self._command.command[:, :3], dim=1) < self.command_threshold).float()
        return torch.maximum(single, near_zero_cmd)


@REWARD_TERM_REGISTRY.register()
class HomieStandStill(_FeetReward):
    """Feet off the ground while told to stand still at full height (OpenHomie ``stand_still``)."""

    command_threshold: float = 0.1
    contact_force_threshold: float = 0.1

    def compute(self) -> torch.Tensor:
        lifted = (self._feet_contact_force()[..., 2] < self.contact_force_threshold).float().sum(dim=1)
        near_zero_cmd = (torch.norm(self._command.command[:, :3], dim=1) < self.command_threshold).float()
        return lifted * _standing(self._command.command) * near_zero_cmd
