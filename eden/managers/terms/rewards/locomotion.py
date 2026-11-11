"""Reward terms for locomotion and velocity tracking tasks.

Most terms here are stateful class terms: they resolve their entity/command refs, link indices, and constant
tensors (gravity, per-joint std, pose weights) once in :meth:`build` and reuse them every step, never stashing
caches on the ``env`` object. Trivial penalties with nothing to cache are plain functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import quat_to_xyz

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import (
    base_height_above_ground,
    gravity_vec,
    projected_gravity,
    resolve_link_idx,
    resolve_per_dof_std,
    resolve_terrain,
)
from eden.utils.misc import as_genesis_index, resolve_named_entity
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity
    from eden.managers.command_manager import CommandTerm
    from eden.managers.terms.commands.gait import GaitPhaseCommand


# ---------------------------------------------------------------------------
# Basic locomotion tracking rewards
# ---------------------------------------------------------------------------


def _compute_tracking_lin_vel(command_xy: torch.Tensor, base_lin_vel_xy: torch.Tensor, sigma: float) -> torch.Tensor:
    lin_vel_error = torch.sum(torch.square(command_xy - base_lin_vel_xy), dim=1)
    return torch.exp(-lin_vel_error / sigma)


class _VelocityTracking(RewardTerm):
    """Shared config + ``build()`` for the exp-shaped velocity-tracking rewards.

    Subclasses only implement :meth:`compute`. Not registered (abstract).
    """

    command_name: str = ""
    entity_name: str = "robot"
    sigma: float = 0.25

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self._robot = self.resolve_entity(self.entity_name)


@REWARD_TERM_REGISTRY.register()
class TrackingLinVel(_VelocityTracking):
    """Tracking of linear velocity commands (xy axes).

    Body frame: ``exp(-||cmd_xy - v_xy||^2 / sigma)``; ``sigma`` divides the squared error directly, so it
    is a variance, not a std.
    """

    def compute(self) -> torch.Tensor:
        base_lin_vel = self._robot.get_vel(frame="body")
        return _compute_tracking_lin_vel(self._command.command[:, :2], base_lin_vel[:, :2], self.sigma)


def _compute_tracking_ang_vel(command_wz: torch.Tensor, base_ang_vel_wz: torch.Tensor, sigma: float) -> torch.Tensor:
    ang_vel_error = torch.square(command_wz - base_ang_vel_wz)
    return torch.exp(-ang_vel_error / sigma)


@REWARD_TERM_REGISTRY.register()
class TrackingAngVel(_VelocityTracking):
    """Tracking of angular velocity commands (yaw).

    Body frame: ``exp(-(cmd_wz - w_z)^2 / sigma)``.
    """

    def compute(self) -> torch.Tensor:
        base_ang_vel = self._robot.get_ang(frame="body")
        return _compute_tracking_ang_vel(self._command.command[:, 2], base_ang_vel[:, 2], self.sigma)


@REWARD_TERM_REGISTRY.register()
def constant_base_height(
    env: EnvBase,
    *,
    entity_name: str,
    target_height: float,
    terrain_entity_name: str | None = None,
    relative: bool = False,
) -> torch.Tensor:
    """Penalize base height away from target — :func:`base_height_l2` without velocity gating.

    ``terrain_entity_name`` and ``relative`` mean what they do there, and are forwarded unchanged.
    """
    return base_height_l2(
        env,
        entity_name=entity_name,
        target_height=target_height,
        terrain_entity_name=terrain_entity_name,
        relative=relative,
    )


@REWARD_TERM_REGISTRY.register()
def lin_vel_z(env: EnvBase, *, entity_name: str) -> torch.Tensor:
    """Penalize z axis base linear velocity."""
    robot: Entity = env.get_entity(entity_name)
    base_lin_vel_b = robot.get_vel(frame="body")
    return torch.square(base_lin_vel_b[:, 2])


# ---------------------------------------------------------------------------
# Velocity tracking rewards
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def track_lin_vel_with_z_penalty(
    env: EnvBase,
    *,
    command_name: str,
    entity_name: str,
    std: float,
) -> torch.Tensor:
    """Track commanded linear velocity with additional z-velocity penalty.

    reward = exp(-(xy_error + 2 * z_vel^2) / std^2)
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    robot: Entity = env.get_entity(entity_name)
    actual = robot.get_vel(frame="body")
    xy_error = torch.sum(torch.square(command.command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + 2.0 * z_error
    return torch.exp(-lin_vel_error / (std**2))


@REWARD_TERM_REGISTRY.register()
def track_ang_vel_with_xy_penalty(
    env: EnvBase,
    *,
    command_name: str,
    entity_name: str,
    std: float,
) -> torch.Tensor:
    """Track commanded angular velocity with additional xy angular velocity penalty.

    reward = exp(-(z_error + 0.05 * xy_error) / std^2)
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    robot: Entity = env.get_entity(entity_name)
    actual = robot.get_ang(frame="body")
    z_error = torch.square(command.command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + 0.05 * xy_error
    return torch.exp(-ang_vel_error / (std**2))


@REWARD_TERM_REGISTRY.register()
class FlatOrientationL2(RewardTerm):
    """Penalize non-flat base orientation (projected gravity deviation in xy)."""

    entity_name: str = "robot"

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._grav = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        grav_in_base = projected_gravity(self._robot.get_quat(), self._grav)
        return torch.sum(torch.square(grav_in_base[:, :2]), dim=1)


@REWARD_TERM_REGISTRY.register()
class Upright(RewardTerm):
    """Reward for keeping a body link upright.

    Projects gravity into the link's frame and returns ``exp(-sum(gravity_xy^2) / std^2)`` (matches mjlab's
    ``upright``). Shared by the stand and velocity-tracking tasks. The link index and the constant world-frame
    gravity vector are resolved once in :meth:`build`.
    """

    entity_name: str = "robot"
    link_name: str = ""
    std: float = 1.0

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        # A one-link `range`, not a one-element tensor: Genesis reads the latter back with `.item()`,
        # a sync per call.
        self._ls_idx = as_genesis_index(resolve_link_idx(self._robot, [self.link_name], self.device))
        self._grav = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        link_quat = self._robot.get_links_quat(ls_idx_local=self._ls_idx).squeeze(1)  # (num_envs, 4)
        grav_in_link = projected_gravity(link_quat, self._grav)
        xy_squared = torch.sum(torch.square(grav_in_link[:, :2]), dim=1)
        return torch.exp(-xy_squared / (self.std**2))


@REWARD_TERM_REGISTRY.register()
class BodyAngVelXYPenalty(RewardTerm):
    """Penalize body angular velocity in xy (allow z rotation)."""

    entity_name: str = "robot"
    link_name: str = ""

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        # A one-link `range`, not a one-element tensor: Genesis reads the latter back with `.item()`,
        # a sync per call.
        self._ls_idx = as_genesis_index(resolve_link_idx(self._robot, [self.link_name], self.device))

    def compute(self) -> torch.Tensor:
        ang_vel = self._robot.get_links_ang(ls_idx_local=self._ls_idx)  # (num_envs, 1, 3)
        ang_vel = ang_vel.squeeze(1)  # (num_envs, 3)
        return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


@REWARD_TERM_REGISTRY.register()
def is_terminated_penalty(env: EnvBase) -> torch.Tensor:
    """Return 1 for terminated environments."""
    return env.termination_manager.terminated.float()


@REWARD_TERM_REGISTRY.register()
class VariablePosture(RewardTerm):
    """Speed-dependent posture reward.

    Uses per-joint standard deviations to control how much each joint can deviate from default pose. The
    reward is ``exp(-mean(error^2 / std^2))``. Three speed regimes select the std set:

    - standing (speed < walking_threshold)
    - walking (walking_threshold <= speed < running_threshold)
    - running (speed >= running_threshold)

    Each std dict must cover every DOF of the robot (a ``".*"`` entry is the explicit "everything else"
    catch-all); DOFs no pattern matches raise at :meth:`build` instead of silently training against a
    default tolerance.
    """

    entity_name: str = "robot"
    command_name: str = ""
    std_standing: dict[str, float] = {}
    std_walking: dict[str, float] = {}
    std_running: dict[str, float] = {}
    walking_threshold: float = 0.1
    running_threshold: float = 1.5

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._command = self._env.command_manager.get_term(self.command_name)
        term = type(self).__name__
        self._std_standing = resolve_per_dof_std(
            self.std_standing, self._robot, self.device, field=f"{term}.std_standing"
        )
        self._std_walking = resolve_per_dof_std(self.std_walking, self._robot, self.device, field=f"{term}.std_walking")
        self._std_running = resolve_per_dof_std(self.std_running, self._robot, self.device, field=f"{term}.std_running")

    def compute(self) -> torch.Tensor:
        command = self._command.command
        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self.walking_threshold).float()
        walking_mask = ((total_speed >= self.walking_threshold) & (total_speed < self.running_threshold)).float()
        running_mask = (total_speed >= self.running_threshold).float()

        std = (
            self._std_standing * standing_mask.unsqueeze(1)
            + self._std_walking * walking_mask.unsqueeze(1)
            + self._std_running * running_mask.unsqueeze(1)
        )

        current_dofs_pos = self._robot.get_dofs_pos()
        error_squared = torch.square(current_dofs_pos - self._robot.default_dofs_pos)
        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


@REWARD_TERM_REGISTRY.register()
class FeetClearance(RewardTerm):
    """Penalize deviation from target clearance height, weighted by foot xy velocity."""

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    target_height: float = 0.0
    command_name: str = ""
    command_threshold: float = 0.1

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._command = self._env.command_manager.get_term(self.command_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )

    def compute(self) -> torch.Tensor:
        foot_pos = self._robot.get_links_pos(ls_idx_local=self._ls_idx)  # (num_envs, N, 3)
        foot_vel = self._robot.get_links_vel(ls_idx_local=self._ls_idx)  # (num_envs, N, 3)

        foot_z = foot_pos[:, :, 2]  # (num_envs, N)
        vel_norm = torch.norm(foot_vel[:, :, :2], dim=-1)  # (num_envs, N)
        delta = torch.abs(foot_z - self.target_height)
        cost = torch.sum(delta * vel_norm, dim=1)

        command = self._command.command
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        active = (total_command > self.command_threshold).float()
        return torch.mul(cost, active)


@REWARD_TERM_REGISTRY.register()
class FeetSlip(RewardTerm):
    """Penalize foot sliding (xy velocity while in contact with ground)."""

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    command_name: str = ""
    command_threshold: float = 0.1
    contact_height_threshold: float = 0.05

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._command = self._env.command_manager.get_term(self.command_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )

    def compute(self) -> torch.Tensor:
        foot_pos = self._robot.get_links_pos(ls_idx_local=self._ls_idx)  # (num_envs, N, 3)
        foot_vel = self._robot.get_links_vel(ls_idx_local=self._ls_idx)  # (num_envs, N, 3)

        # Simple ground contact check: foot z below threshold
        in_contact = (foot_pos[:, :, 2] < self.contact_height_threshold).float()  # (num_envs, N)
        vel_xy_norm_sq = torch.sum(torch.square(foot_vel[:, :, :2]), dim=-1)  # (num_envs, N)
        cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1)

        command = self._command.command
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        active = (total_command > self.command_threshold).float()
        return torch.mul(cost, active)


@REWARD_TERM_REGISTRY.register()
class FeetAirTime(RewardTerm):
    """Legged-gym style feet air-time reward for aperiodic gaits.

    Tracks per-foot air time; on the control step a foot *first* makes contact it
    credits ``(air_time - target_s)`` (summed over feet), gated to when a movement
    command is active. Rewards proper swing cycles, so it discourages a leg that
    stays permanently planted or permanently airborne ("dragging" limps). Use a
    POSITIVE weight.
    """

    entity_name: str = "robot"
    command_name: str = ""
    feet_link_names: list[str] = []
    target_s: float = 0.35
    contact_force_threshold: float = 2.0
    command_threshold: float = 0.1

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._command = self._env.command_manager.get_term(self.command_name)
        self._feet_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        n = self._feet_idx.numel()
        self._air_time = torch.zeros(self.num_envs, n, device=self.device)
        self._last_contacts = torch.zeros(self.num_envs, n, dtype=torch.bool, device=self.device)
        self._dt = float(self._env.dt)

    def compute(self) -> torch.Tensor:
        forces = self.resolve_entity(self.entity_name).get_links_net_contact_force()[:, self._feet_idx]
        contact = torch.norm(forces, dim=-1) > self.contact_force_threshold
        contact_filt = contact | self._last_contacts
        self._last_contacts.copy_(contact)  # copy_ (not rebind): compute may run under inference_mode
        first_contact = (self._air_time > 0.0) & contact_filt
        self._air_time.add_(self._dt)
        reward = torch.sum((self._air_time - self.target_s) * first_contact.float(), dim=1)
        command = self._command.command
        total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        active = (total_command > self.command_threshold).float()
        self._air_time.mul_((~contact_filt).float())  # reset air time on (filtered) contact
        return reward * active

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._air_time.zero_()
            self._last_contacts.zero_()
        else:
            self._air_time[envs_idx] = 0.0
            self._last_contacts[envs_idx] = False


@REWARD_TERM_REGISTRY.register()
class FeetContactBalancePenalty(RewardTerm):
    """Penalize uneven foot usage (a limp) via the variance of per-foot contact duty.

    Keeps an EMA of each foot's contact state and returns the variance across the
    feet: a healthy gait uses all legs (~equal duty) -> low variance; a leg that is
    always planted or always in the air spikes the variance. Complements a mirror /
    symmetry loss by also catching front/rear imbalance. Use a NEGATIVE weight.
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    contact_force_threshold: float = 2.0
    ema_alpha: float = 0.02  # ~1 s time constant at 50 Hz control

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._feet_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        self._ema = torch.full((self.num_envs, self._feet_idx.numel()), 0.5, device=self.device)

    def compute(self) -> torch.Tensor:
        forces = self.resolve_entity(self.entity_name).get_links_net_contact_force()[:, self._feet_idx]
        contact = (torch.norm(forces, dim=-1) > self.contact_force_threshold).float()
        self._ema.mul_(1.0 - self.ema_alpha).add_(contact, alpha=self.ema_alpha)
        return self._ema.var(dim=1)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._ema.fill_(0.5)
        else:
            self._ema[envs_idx] = 0.5


# ---------------------------------------------------------------------------
# Gait-phase-aware locomotion rewards
# ---------------------------------------------------------------------------


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Compute the expected foot height from gait phase using a smoothstep profile.

    Maps phase from [-pi, pi] to a foot height trajectory:
    - First half: 0 → swing_height via cubic smoothstep (foot rising)
    - Second half: swing_height → 0 via cubic smoothstep (foot falling)
    """
    x = (phi + torch.pi) / (2.0 * torch.pi)  # Normalize to [0, 1]

    def _smoothstep(t: torch.Tensor) -> torch.Tensor:
        # Hermite smoothstep: 3t^2 - 2t^3
        return t * t * (3.0 - 2.0 * t)

    # Rising half: normalize x from [0, 0.5] to [0, 1], then 0 → swing_height
    x1 = 2.0 * x
    stance = swing_height * _smoothstep(x1)
    # Falling half: normalize x from [0.5, 1] to [0, 1], then swing_height → 0
    x2 = 2.0 * x - 1.0
    swing = swing_height * (1.0 - _smoothstep(x2))
    return torch.where(x <= 0.5, stance, swing)


@REWARD_TERM_REGISTRY.register()
class FeetPhase(RewardTerm):
    """Reward for tracking desired foot height based on gait phase.

    Uses a smoothstep profile to define the expected swing trajectory.
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    gait_command_name: str = ""
    swing_height: float = 0.08
    tracking_sigma: float = 0.25

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        self._gait_cmd: GaitPhaseCommand = self._env.command_manager.get_term(self.gait_command_name)

    def compute(self) -> torch.Tensor:
        foot_pos = self._robot.get_links_pos(ls_idx_local=self._ls_idx)  # (num_envs, N, 3)
        phase = self._gait_cmd.command  # (num_envs, num_feet)

        if phase.shape[1] != foot_pos.shape[1]:
            raise ValueError(
                f"feet_phase: GaitPhaseCommand num_feet ({phase.shape[1]}) != "
                f"feet_link_names count ({foot_pos.shape[1]}). Check config."
            )
        expected_z = _expected_foot_height(phase, self.swing_height)  # (num_envs, num_feet)
        actual_z = foot_pos[:, :, 2]  # (num_envs, num_feet)
        total_error = torch.sum(torch.square(actual_z - expected_z), dim=1)

        return torch.exp(-total_error / self.tracking_sigma)


@REWARD_TERM_REGISTRY.register()
class CloseFeetPenalty(RewardTerm):
    """Penalize when feet are too close together in the xy plane.

    Returns 1.0 (penalty active) when the smallest pairwise xy distance between feet is < threshold, else 0.0.
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    threshold: float = 0.15

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        n_feet = self._ls_idx.numel()
        # Diagonal mask (self-distance) precomputed once; None when <2 feet (penalty is a no-op).
        self._diag_mask = torch.eye(n_feet, device=self.device, dtype=torch.bool).unsqueeze(0) if n_feet >= 2 else None

    def compute(self) -> torch.Tensor:
        if self._diag_mask is None:
            return torch.zeros(self.num_envs, device=self.device)
        foot_pos = self._robot.get_links_pos(ls_idx_local=self._ls_idx)  # (num_envs, N, 3)
        foot_xy = foot_pos[:, :, :2]  # (num_envs, N, 2)
        # Pairwise distances: (num_envs, N, N)
        diff = foot_xy.unsqueeze(2) - foot_xy.unsqueeze(1)
        pairwise_dist = torch.norm(diff, dim=-1)
        pairwise_dist = pairwise_dist.masked_fill(self._diag_mask, float("inf"))
        min_dist = pairwise_dist.min(dim=-1).values.min(dim=-1).values  # (num_envs,)
        return (min_dist < self.threshold).float()


@REWARD_TERM_REGISTRY.register()
class FeetOrientationPenalty(RewardTerm):
    """Penalize feet orientation deviation from flat (projected gravity in foot frame)."""

    entity_name: str = "robot"
    feet_link_names: list[str] = []

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        # One gravity row per (env, foot). Size by the resolved link count, not len(feet_link_names):
        # a glob/regex pattern can resolve to a different number of links.
        self._grav = gravity_vec(self.num_envs * len(self._ls_idx), self.device)

    def compute(self) -> torch.Tensor:
        foot_quat = self._robot.get_links_quat(ls_idx_local=self._ls_idx)  # (num_envs, N, 4)
        n_feet = foot_quat.shape[1]

        # Flatten (num_envs, N, 4) → (num_envs*N, 4) for batched quat ops
        flat_quat = foot_quat.reshape(-1, 4)
        projected = projected_gravity(flat_quat, self._grav)  # (num_envs*N, 3)
        # Deviation from vertical per foot, then sum across feet
        deviation = torch.sum(torch.square(projected[:, :2]), dim=1).sqrt()  # (num_envs*N,)
        return torch.sum(deviation.reshape(self.num_envs, n_feet), dim=1)


@REWARD_TERM_REGISTRY.register()
class WeightedPosePenalty(RewardTerm):
    """Penalize deviation from default pose with per-joint weights."""

    entity_name: str = "robot"
    pose_weights: list[float] = []

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        expected_dof_count = self._robot.get_dofs_pos().shape[1]
        if len(self.pose_weights) != expected_dof_count:
            raise ValueError(
                f"{term_subject(self)} expected {expected_dof_count} pose weights, got {len(self.pose_weights)}."
            )
        self._weights = torch.tensor(self.pose_weights, device=self.device, dtype=torch.float32).unsqueeze(0)

    def compute(self) -> torch.Tensor:
        error = torch.square(self._robot.get_dofs_pos() - self._robot.default_dofs_pos)
        return torch.sum(error * self._weights, dim=1)


# ---------------------------------------------------------------------------
# Holosoma-style locomotion penalties
# (ported from holosoma's .../reward/terms/locomotion.py)
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
class StumblePenalty(RewardTerm):
    """Compute an indicator penalty for feet hitting vertical surfaces (e.g. stairs riser, walls).

    Returns 1.0 when any foot's lateral contact force exceeds
    ``horizontal_to_vertical_ratio * |z contact|``, else 0.0. Mirrors
    ``holosoma/.../reward/terms/locomotion.py:316-329``.
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    horizontal_to_vertical_ratio: float = 4.0

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )

    def compute(self) -> torch.Tensor:
        forces = self._robot.get_links_net_contact_force()  # (num_envs, n_links, 3)
        foot_forces = forces[:, self._ls_idx, :]  # (num_envs, n_feet, 3)
        horiz = torch.norm(foot_forces[:, :, :2], dim=-1)
        vert = foot_forces[:, :, 2].abs()
        return (horiz > self.horizontal_to_vertical_ratio * vert).any(dim=1).float()


@REWARD_TERM_REGISTRY.register()
class CloseFeetYawProjectedPenalty(RewardTerm):
    """Penalize feet closer than a lateral distance in the base's yaw frame.

    Unlike the naive xy distance in :class:`CloseFeetPenalty`, this rotates the foot-pair offset into the
    robot's yaw-aligned frame and measures only the lateral component. This is more robust during in-place
    turns where the naive metric flips sign as the body yaws. Requires exactly two feet — humanoid
    convention. Mirrors ``holosoma/.../reward/terms/locomotion.py:191-215``.
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    threshold: float = 0.05

    def build(self) -> None:
        self._robot = self.resolve_entity(self.entity_name)
        self._ls_idx = resolve_link_idx(
            self._robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        if self._ls_idx.numel() != 2:
            raise ValueError(
                f"close_feet_yaw_projected_penalty expects exactly 2 feet, got {self._ls_idx.numel()}. "
                "Use close_feet_penalty for >2 feet."
            )

    def compute(self) -> torch.Tensor:
        foot_pos = self._robot.get_links_pos(ls_idx_local=self._ls_idx)  # (num_envs, 2, 3)
        left_xy = foot_pos[:, 0, :2]
        right_xy = foot_pos[:, 1, :2]
        yaw = quat_to_xyz(self._robot.get_quat(), rpy=True)[:, 2]
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        # Lateral component of the foot-pair offset in the body's yaw-aligned frame:
        # rotating world (dx, dy) by -yaw gives lateral = -sin(yaw)*dx + cos(yaw)*dy.
        # The outer abs() makes the (left, right) pair ordering immaterial.
        lateral = torch.abs(cos_y * (left_xy[:, 1] - right_xy[:, 1]) - sin_y * (left_xy[:, 0] - right_xy[:, 0]))
        return (lateral < self.threshold).float()


@REWARD_TERM_REGISTRY.register()
def base_height_l2(
    env: EnvBase,
    *,
    entity_name: str,
    target_height: float,
    command_name: str | None = None,
    zero_vel_penalty_scale: float = 1.0,
    zero_vel_threshold: float = 0.1,
    terrain_entity_name: str | None = None,
    relative: bool = False,
) -> torch.Tensor:
    """L2 base-height penalty with optional zero-velocity amplification.

    Squared deviation of the base height from ``target_height``. When ``command_name``
    is provided and the commanded planar velocity is below ``zero_vel_threshold``,
    the penalty is multiplied by ``zero_vel_penalty_scale`` so the policy is held
    more strictly to the target height while standing. Mirrors
    ``holosoma/.../reward/terms/locomotion.py:218-250``.

    Height is read through :func:`~eden.managers.terms.utils.base_height_above_ground`, the same
    helper as the :func:`~eden.managers.terms.observations.locomotion.base_height` observation and
    the :func:`~eden.managers.terms.terminations.locomotion.root_height_below_minimum` termination,
    and it takes their two parameters for the same reasons. ``terrain_entity_name`` names a
    :class:`~eden.entities.terrain.Terrain` to measure against instead of the world-frame ``z``:
    without it a target height on a heightfield is unreachable in a valley and undershot on a hill,
    and — the reason it belongs on *this* term rather than only on those two — a config that sets it
    on the observation and the termination but not here trains a policy whose reward pulls toward a
    height its own feature and its episode boundary do not measure. ``relative`` selects the frame
    the base pose is read in (see
    :func:`~eden.managers.terms.observations.proprio.base_pos`); the default ``False`` is the solver
    frame.

    Equivalent to :func:`constant_base_height` when ``command_name`` is ``None``.
    The stance-mode amplification from holosoma's variant is intentionally not
    ported here — that path reaches into env-set boolean attributes with no clear
    producer contract. If you need it, add a dedicated term that consumes a typed
    stance source (e.g. a phase-aware command term).
    """
    robot: Entity = resolve_named_entity(env.entities, entity_name)
    terrain = resolve_terrain(env.entities, terrain_entity_name)
    err = torch.square(base_height_above_ground(robot, terrain, relative=relative) - target_height)
    if command_name is not None:
        command: CommandTerm = env.command_manager.get_term(command_name)
        zero_vel_mask = torch.norm(command.command[:, :2], dim=1) < zero_vel_threshold
        err = torch.where(zero_vel_mask, err * zero_vel_penalty_scale, err)
    return err
