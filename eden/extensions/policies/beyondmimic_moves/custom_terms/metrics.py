"""Motion-tracking error metrics keyed off `MotionCommand`.

The MotionManager-based equivalents (``motion_error_*``) live in
``eden/tasks/beyondmimic/custom_terms/metrics.py``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.utils.geom import quat_error_magnitude

from eden.managers.terms.commands.mimic import MotionCommand

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


def _tracking_metric(success_threshold: float):
    """Register a motion-tracking metric with the canonical settings (decorator)."""
    return METRIC_TERM_REGISTRY.register(
        success_threshold=success_threshold,
        is_cumulative=True,
        direction=MetricDirection.LIB,
        metric_mode=MetricMode.RESET,
    )


@_tracking_metric(0.2)
def error_anchor_pos(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1)


@_tracking_metric(0.2)
def error_anchor_rot(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w)


@_tracking_metric(0.6)
def error_anchor_lin_vel(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.anchor_lin_vel_w - cmd.robot_anchor_lin_vel_w, dim=-1)


@_tracking_metric(3.0)
def error_anchor_ang_vel(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.anchor_ang_vel_w - cmd.robot_anchor_ang_vel_w, dim=-1)


@_tracking_metric(0.1)
def error_body_pos(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.links_pos_relative_w - cmd.robot_links_pos_w, dim=-1).mean(dim=-1)


@_tracking_metric(0.4)
def error_body_rot(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return quat_error_magnitude(cmd.links_quat_relative_w, cmd.robot_links_quat_w).mean(dim=-1)


@_tracking_metric(0.7)
def error_body_lin_vel(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.links_lin_vel_w - cmd.robot_links_lin_vel_w, dim=-1).mean(dim=-1)


@_tracking_metric(4.0)
def error_body_ang_vel(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.links_ang_vel_w - cmd.robot_links_ang_vel_w, dim=-1).mean(dim=-1)


@_tracking_metric(1.8)
def error_dofs_pos(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.dofs_pos - cmd.robot_dofs_pos, dim=-1)


@_tracking_metric(15.0)
def error_dofs_vel(env: RLEnvBase, *, command_name: str = "motion"):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(cmd.dofs_vel - cmd.robot_dofs_vel, dim=-1)
