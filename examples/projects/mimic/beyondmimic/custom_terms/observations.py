from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from genesis.utils.geom import quat_to_R

from eden.utils.geom import subtract_frame_transforms

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.terms.commands.mimic import MotionCommand


@OBSERVATION_TERM_REGISTRY.register()
def robot_anchor_ori_w(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = quat_to_R(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


@OBSERVATION_TERM_REGISTRY.register()
def robot_anchor_lin_vel_w(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_lin_vel_w[:, :3].view(env.num_envs, -1)


@OBSERVATION_TERM_REGISTRY.register()
def robot_anchor_ang_vel_w(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_ang_vel_w[:, 3:6].view(env.num_envs, -1)


@OBSERVATION_TERM_REGISTRY.register()
def beyondmimic_robot_body_pos_b(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.links_name)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_links_pos_w,
        command.robot_links_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


@OBSERVATION_TERM_REGISTRY.register()
def beyondmimic_robot_body_ori_b(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.links_name)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        command.robot_links_pos_w,
        command.robot_links_quat_w,
    )
    mat = quat_to_R(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


@OBSERVATION_TERM_REGISTRY.register()
def motion_anchor_pos_b(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


@OBSERVATION_TERM_REGISTRY.register()
def motion_anchor_ori_b(
    env: RLEnvBase,
    *,
    command_name: str = "motion_command",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = quat_to_R(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
