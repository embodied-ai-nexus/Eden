"""Anchor-relative motion/IMU observation terms for BeyondMimic-Moves."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from genesis.utils.geom import quat_to_R

from eden.utils.geom import subtract_frame_transforms

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import RLEnvBase
    from eden.managers.terms.commands.mimic import MotionCommand


# The `g1_moves_` prefix keeps these `MotionCommand`-based terms apart from the core anchor terms
# (`motion_links_pos_b` / `motion_links_ori_b` in `eden/managers/terms/observations/tracking.py`,
# which the beyondmimic task configures as `motion_anchor_*_b`): those take entity_name +
# anchor_link_name and read the MimicRLEnvBase motion_manager.

# G1 IMU site offset in the pelvis link's local frame, copied verbatim from the
# `imu_in_pelvis` site declaration in the G1 MJCF (`g1_29dof_rev_1_0.xml` line 69
# and the upstream mjlab `g1.xml`). mjlab trains against MuJoCo's `velocimeter`
# at this offset; the velocity at the offset point differs from the pelvis-origin
# velocity by ω × r. Without it the policy's input shifts in proportion to angular
# velocity — significant for energetic dance/karate.
G1_IMU_OFFSET_BODY: tuple[float, float, float] = (0.04525, 0.0, -0.08339)

_IMU_OFFSET_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _imu_offset_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _IMU_OFFSET_CACHE.get(key)
    if t is None:
        t = torch.tensor(G1_IMU_OFFSET_BODY, device=device, dtype=dtype)
        _IMU_OFFSET_CACHE[key] = t
    return t


def _anchor_relative(env: RLEnvBase, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Position and orientation of the motion anchor in the robot anchor's body frame."""
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return subtract_frame_transforms(
        cmd.robot_anchor_pos_w,
        cmd.robot_anchor_quat_w,
        cmd.anchor_pos_w,
        cmd.anchor_quat_w,
    )


@OBSERVATION_TERM_REGISTRY.register()
def g1_moves_motion_anchor_pos_b(env: RLEnvBase, *, command_name: str = "motion") -> torch.Tensor:
    pos, _ = _anchor_relative(env, command_name)
    return pos.view(env.num_envs, -1)


@OBSERVATION_TERM_REGISTRY.register()
def g1_moves_motion_anchor_ori_b(env: RLEnvBase, *, command_name: str = "motion") -> torch.Tensor:
    _, ori = _anchor_relative(env, command_name)
    mat = quat_to_R(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


@OBSERVATION_TERM_REGISTRY.register()
def g1_moves_imu_lin_vel_b(env: RLEnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Linear velocity at the pelvis IMU site, in the pelvis body frame.

    Mirrors mjlab's `builtin_sensor("robot/imu_lin_vel")` which is a MuJoCo
    `velocimeter` at the `imu_in_pelvis` site. For a rigid body,
    ``v_imu_body = v_pelvis_body + ω_pelvis_body × r_imu_body``.
    """
    entity: Entity = env.entities[entity_name]
    v_pelvis_b = entity.get_vel(frame="body")
    omega_pelvis_b = entity.get_ang(frame="body")
    r_imu = _imu_offset_tensor(v_pelvis_b.device, v_pelvis_b.dtype)
    return v_pelvis_b + torch.linalg.cross(omega_pelvis_b, r_imu.expand_as(omega_pelvis_b))
