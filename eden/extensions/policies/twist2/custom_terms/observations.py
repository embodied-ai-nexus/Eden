"""Observation terms for the TWIST2 actor: the reference root state it tracks, and the robot's IMU tilt.

Both read roll and pitch through :func:`~eden.utils.geom.euler_xyz_from_quat`, the decomposition the
actor was trained on, rather than the core ``base_rpy`` term's: the two agree in value but not in float.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from eden.utils.geom import euler_xyz_from_quat

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.terms.commands.mimic import MotionCommand


@OBSERVATION_TERM_REGISTRY.register()
def twist2_motion_root(env: RLEnvBase, *, command_name: str = "motion") -> torch.Tensor:
    """Reference root state, ``(num_envs, 6)``: xy velocity and yaw rate in its own frame, height, roll, pitch.

    Laid out ``[vel_x, vel_y, z, roll, pitch, yaw_rate]``, the head of TWIST2's 35-dim mimic block.
    Heading- and position-invariant, so the reference needs no alignment to the robot.
    """
    motion: MotionCommand = env.command_manager.get_term(command_name)
    root_quat = motion.base_quat_w
    root_lin_vel_b = inv_transform_by_quat(motion.base_lin_vel_w, root_quat)
    root_ang_vel_b = inv_transform_by_quat(motion.base_ang_vel_w, root_quat)
    roll, pitch, _ = euler_xyz_from_quat(root_quat)
    return torch.cat(
        [root_lin_vel_b[:, :2], motion.base_pos_w[:, 2:3], torch.stack([roll, pitch], dim=-1), root_ang_vel_b[:, 2:3]],
        dim=-1,
    )


@OBSERVATION_TERM_REGISTRY.register()
def twist2_motion_dofs_pos(env: RLEnvBase, *, command_name: str = "motion") -> torch.Tensor:
    """Reference joint positions, ``(num_envs, num_dofs)`` in the motion's DOF order: the tail of the mimic block."""
    motion: MotionCommand = env.command_manager.get_term(command_name)
    return motion.dofs_pos


@OBSERVATION_TERM_REGISTRY.register()
def twist2_imu_roll_pitch(env: RLEnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Roll and pitch of the robot's base, ``(num_envs, 2)``."""
    roll, pitch, _ = euler_xyz_from_quat(env.get_entity(entity_name).get_quat())
    return torch.stack([roll, pitch], dim=-1)
