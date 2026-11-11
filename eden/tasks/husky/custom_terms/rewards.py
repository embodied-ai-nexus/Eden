"""Reward terms for the HUSKY skateboarding task.

Ports of ``mjlab_husky.tasks.skater.mdp.rewards``. The source multiplies each
reward *group* by its gait-phase gate (push / steer / transition) after the
manager sums it; here every term applies its own gate, which is numerically
identical (the gates are per-env indicators) and keeps the logged episode sums
gated as a bonus. Shared phase/contact state comes from the
:class:`~eden.tasks.husky.custom_terms.commands.SkateCommand` term.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.utils.geom import euler_xyz_from_quat, quat_error_magnitude, wrap_to_pi

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.tasks.husky.custom_terms.commands import SkateCommand

# Steer-stance joint targets in MJCF joint order (mjlab_husky ``steer_init_pos``):
# both feet on the deck, knees bent, arms spread for balance.
STEER_DOFS_POS: tuple[float, ...] = (
    -0.15,
    0.1,
    0.05,
    0.6,
    -0.42,
    0.0,  # left leg (hip p/r/y, knee, ankle p/r)
    -0.15,
    -0.1,
    0.05,
    0.6,
    -0.42,
    0.0,  # right leg
    0.0,
    0.0,
    0.1,  # waist y/r/p
    0.0,
    0.55,
    -0.25,
    0.55,  # left arm (shoulder p/r/y, elbow)
    0.0,
    -0.55,
    -0.25,
    0.55,  # right arm
)

_RAKE_ANGLE_RAD = math.radians(60.0)


def _skate(env: EnvBase, command_name: str) -> SkateCommand:
    return env.command_manager.get_term(command_name)


@functools.lru_cache(maxsize=4)
def _steer_target(device: torch.device) -> torch.Tensor:
    """The steer-stance joint targets on ``device`` (cached: no per-step H2D copy)."""
    return torch.tensor(STEER_DOFS_POS, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Push-phase rewards
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def push_skateboard_lin_vel(
    env: EnvBase, *, command_name: str = "skate", skateboard_name: str = "skateboard", std: float = 0.5
) -> torch.Tensor:
    """Track the commanded board speed (body-frame x), penalizing lateral/vertical drift."""
    skate = _skate(env, command_name)
    board = env.entities[skateboard_name]
    vel_b = inv_transform_by_quat(board.get_vel(), board.get_quat(relative=True))
    error = torch.square(skate.command[:, 0] - vel_b[:, 0]) + torch.square(vel_b[:, 1]) + torch.square(vel_b[:, 2])
    return torch.exp(-error / std**2) * skate.contact_phase[:, 0]


@REWARD_TERM_REGISTRY.register()
def push_yaw_align(
    env: EnvBase,
    *,
    command_name: str = "skate",
    entity_name: str = "robot",
    skateboard_name: str = "skateboard",
    std: float = 0.5,
) -> torch.Tensor:
    """Keep the robot's yaw aligned with the board's while pushing."""
    skate = _skate(env, command_name)
    _, _, robot_yaw = euler_xyz_from_quat(env.entities[entity_name].get_quat(relative=True))
    _, _, board_yaw = euler_xyz_from_quat(env.entities[skateboard_name].get_quat(relative=True))
    yaw_diff = wrap_to_pi(robot_yaw - board_yaw).abs()
    return torch.exp(-yaw_diff / std**2) * skate.contact_phase[:, 0]


@REWARD_TERM_REGISTRY.register()
def push_left_foot_air_time(
    env: EnvBase,
    *,
    command_name: str = "skate",
    threshold_min: float = 0.1,
    threshold_max: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward push-foot (left) swings whose ground air time is in the useful range."""
    skate = _skate(env, command_name)
    air = skate.left_foot_air_time
    in_range = ((air > threshold_min) & (air < threshold_max)).float()
    moving = (skate.command[:, 0] > command_threshold).float()
    return in_range * moving * skate.contact_phase[:, 0]


@REWARD_TERM_REGISTRY.register()
def push_contact_ground_parallel(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Reward flat left-foot ground contact (low variance of the foot probe-site heights)."""
    skate = _skate(env, command_name)
    flat = (skate.left_foot_site_z * 10.0).var(dim=1) < 0.05
    left_on_ground = skate.feet_contact_g[:, 0]
    return flat.float() * left_on_ground.float() * skate.contact_phase[:, 0]


# ---------------------------------------------------------------------------
# Steer-phase rewards
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def steer_contact_num(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Reward both feet on the deck, penalize the push foot dragging on the ground."""
    skate = _skate(env, command_name)
    both_on_board = (skate.feet_contact_b.sum(dim=-1) == 2).float()
    left_on_ground = skate.feet_contact_g[:, 0].float()
    return (2.0 * both_on_board - left_on_ground) * skate.contact_phase[:, 1]


@REWARD_TERM_REGISTRY.register()
def steer_joint_pos(
    env: EnvBase, *, command_name: str = "skate", entity_name: str = "robot", std: float = 0.45
) -> torch.Tensor:
    """Hold the steer stance (both feet on deck, arms spread)."""
    skate = _skate(env, command_name)
    robot = env.entities[entity_name]
    error = torch.mean(torch.square(robot.get_dofs_pos() - _steer_target(env.device)), dim=1)
    return torch.exp(-error / std**2) * skate.contact_phase[:, 1]


@REWARD_TERM_REGISTRY.register()
def steer_feet_dis(env: EnvBase, *, command_name: str = "skate", std: float = 0.32) -> torch.Tensor:
    """Pull the feet toward their deck markers (left rear, right front)."""
    skate = _skate(env, command_name)
    dis = skate.feet_marker_dis.norm(dim=-1).mean(dim=-1)
    return torch.exp(-dis / std**2) * skate.contact_phase[:, 1]


@REWARD_TERM_REGISTRY.register()
def steer_track_heading(env: EnvBase, *, command_name: str = "skate", std: float = 0.14) -> torch.Tensor:
    """Track the commanded heading with the board while steering."""
    skate = _skate(env, command_name)
    error = wrap_to_pi(skate.board_heading_w() - skate.target_heading_w).abs()
    return torch.exp(-error / std**2) * skate.contact_phase[:, 1]


@REWARD_TERM_REGISTRY.register()
def steer_tilt_guide(
    env: EnvBase, *, command_name: str = "skate", skateboard_name: str = "skateboard", std: float = 0.14
) -> torch.Tensor:
    """Guide the deck tilt toward the analytic value that closes the heading error.

    Physics-aware steering shaping: with truck rake angle lambda, a deck tilt gamma
    yields the turn rate that would close the heading error over the remaining steer
    window at the current board speed (HUSKY paper eq. for ``gamma_ref``).
    """
    skate = _skate(env, command_name)
    board = env.entities[skateboard_name]
    gamma = board.get_dofs_pos()[:, 0]
    delta_theta = wrap_to_pi(skate.target_heading_w - skate.board_heading_w())
    vx = inv_transform_by_quat(board.get_vel(), board.get_quat(relative=True))[:, 0]
    delta_t = skate.steer_remaining_time_s().clamp(min=0.5)
    tan_sigma = (0.4 * delta_theta) / (vx * delta_t + 1e-6)
    sin_gamma = torch.clamp(tan_sigma / math.tan(_RAKE_ANGLE_RAD), -0.99, 0.99)
    gamma_ref = torch.clip(torch.asin(sin_gamma), -0.2, 0.2)
    return torch.exp(-(gamma - gamma_ref).abs() / std**2) * skate.contact_phase[:, 1]


# ---------------------------------------------------------------------------
# Transition rewards
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def transition_body_pos_tracking(env: EnvBase, *, command_name: str = "skate", std: float = 0.22) -> torch.Tensor:
    """Track the bezier blend targets (positions, board frame) through transitions."""
    skate = _skate(env, command_name)
    error = torch.square(skate.tracked_pos_b - skate.transition_target_pos_b).sum(dim=-1).mean(dim=-1)
    return torch.exp(-error / std**2) * skate.in_transition.float()


@REWARD_TERM_REGISTRY.register()
def transition_body_rot_tracking(env: EnvBase, *, command_name: str = "skate", std: float = 0.32) -> torch.Tensor:
    """Track the slerp blend targets (orientations) through transitions."""
    skate = _skate(env, command_name)
    error = torch.square(quat_error_magnitude(skate.transition_target_quat_b, skate.tracked_quat_b)).mean(dim=-1)
    return torch.exp(-error / std**2) * skate.in_transition.float()


@REWARD_TERM_REGISTRY.register()
def transition_penalty_contact(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Penalty (negative weight) for planting the push foot mid-transition."""
    skate = _skate(env, command_name)
    contact = (skate.foot_ground_force[:, 0].norm(dim=-1) > 1e-3).float()
    transition = torch.logical_or(skate.contact_phase[:, 2] > 0.5, skate.contact_phase[:, 3] > 0.5).float()
    return contact * transition


# ---------------------------------------------------------------------------
# Regularization rewards
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def reg_wheel_contact_number(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Reward keeping all four wheels grounded."""
    skate = _skate(env, command_name)
    return (skate.wheel_contact_filt.sum(dim=1) == 4).float()


@REWARD_TERM_REGISTRY.register()
def dofs_vel_l2(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """L2-squared joint velocities."""
    return torch.sum(torch.square(env.entities[entity_name].get_dofs_vel()), dim=1)


@REWARD_TERM_REGISTRY.register()
def dofs_torques_l2(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """L2-squared actuator torques (the PD controller's applied force)."""
    return torch.sum(torch.square(env.entities[entity_name].get_dofs_control_force()), dim=1)


@REWARD_TERM_REGISTRY.register()
def self_collision_flag(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """1 if any robot self-collision is active (the source sensor's boolean ``found``)."""
    return _skate(env, command_name).self_collision.float()


@REWARD_TERM_REGISTRY.register()
def board_flat(
    env: EnvBase, *, command_name: str = "skate", skateboard_name: str = "skateboard", std: float = 0.22
) -> torch.Tensor:
    """Keep the deck untilted outside the steer phase."""
    skate = _skate(env, command_name)
    gamma = env.entities[skateboard_name].get_dofs_pos()[:, 0]
    non_steer = (skate.contact_phase[:, 1] != 1.0).float()
    return torch.exp(-gamma.abs() / std**2) * non_steer


@REWARD_TERM_REGISTRY.register()
def stand_still(
    env: EnvBase, *, command_name: str = "skate", entity_name: str = "robot", std: float = 0.32
) -> torch.Tensor:
    """Hold the default (push-stance) pose while the speed command is ~zero."""
    skate = _skate(env, command_name)
    robot = env.entities[entity_name]
    default = robot.default_dofs_pos
    default = default[0] if default.dim() == 2 else default
    error = torch.mean(torch.square(robot.get_dofs_pos() - default), dim=1)
    return torch.exp(-error / std**2) * skate.still.float()
