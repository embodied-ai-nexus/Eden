"""Motion-tracking error metrics keyed off ``env.motion_manager``.

The MotionManager-based counterparts of the ``MotionCommand``-keyed metrics in
``eden.extensions.policies.beyondmimic_moves.custom_terms.metrics``. Use these
from any task built on :class:`~eden.envs.base.MimicRLEnvBase` (i.e. an
``EdenMimicConfig``-based task).

Names are prefixed with ``motion_`` so that both this module and the
extension-side metrics module can be imported in the same process without
colliding in ``METRIC_TERM_REGISTRY``. The success thresholds and metric mode
match the extension-side versions one-for-one.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.utils.geom import quat_error_magnitude

from eden.utils.motion_reference.lookup import (
    anchor_relative_links_once,
    anchor_state_once,
    get_motion_link_idx,
    get_robot_link_idx,
)

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


def _anchor_relative_links(env: MimicRLEnvBase, entity_name: str, anchor_link_name: str, links_name: list[str]):
    """:func:`~eden.utils.motion_reference.lookup.compute_anchor_relative_links`, shared with the body-tracking
    rewards of the same step.

    Read off the current frame (index 0) of :func:`anchor_relative_links_once`, which the
    ``MotionRelativeBody*ErrorExp`` rewards ask for too: within a step's graded pass the warp is computed once
    for both, and frame 0 of it is bit-identical to ``compute_anchor_relative_links``.
    """
    pos_rel, quat_rel, robot_pos, robot_quat, _ = anchor_relative_links_once(
        env, entity_name, anchor_link_name, links_name
    )
    return pos_rel[:, 0], quat_rel[:, 0], robot_pos, robot_quat


def _tracking_metric(success_threshold: float):
    """Register a motion-tracking metric with the canonical settings (decorator)."""
    return METRIC_TERM_REGISTRY.register(
        success_threshold=success_threshold,
        is_cumulative=True,
        direction=MetricDirection.LIB,
        metric_mode=MetricMode.RESET,
    )


# ---------------------------------------------------------------------------
# Anchor errors (function-style)
# ---------------------------------------------------------------------------


@_tracking_metric(0.2)
def motion_error_anchor_pos(env: MimicRLEnvBase, *, entity_name: str = "robot", anchor_link_name: str = "torso_link"):
    robot_pos, _, motion_pos, _ = anchor_state_once(env, entity_name, anchor_link_name)
    return torch.norm(motion_pos - robot_pos, dim=-1)


@_tracking_metric(0.2)
def motion_error_anchor_rot(env: MimicRLEnvBase, *, entity_name: str = "robot", anchor_link_name: str = "torso_link"):
    _, robot_quat, _, motion_quat = anchor_state_once(env, entity_name, anchor_link_name)
    return quat_error_magnitude(motion_quat, robot_quat)


def _anchor_velocity_error(
    env: MimicRLEnvBase,
    entity_name: str,
    anchor_link_name: str,
    motion_attr: str,
    robot_method: str,
) -> torch.Tensor:
    motion_idx = get_motion_link_idx(env, entity_name, [anchor_link_name])
    robot_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])
    ref = env.motion_manager.data[entity_name]
    motion_vel = getattr(ref, motion_attr)[:, 0].index_select(1, motion_idx)[:, 0]  # not `[..., idx[0]]`: a sync
    robot_vel = getattr(env.entities[entity_name], robot_method)(ls_idx_local=robot_idx)[:, 0]
    return torch.norm(motion_vel - robot_vel, dim=-1)


@_tracking_metric(0.6)
def motion_error_anchor_lin_vel(
    env: MimicRLEnvBase, *, entity_name: str = "robot", anchor_link_name: str = "torso_link"
):
    return _anchor_velocity_error(env, entity_name, anchor_link_name, "links_lin_vel_w", "get_links_vel")


@_tracking_metric(3.0)
def motion_error_anchor_ang_vel(
    env: MimicRLEnvBase, *, entity_name: str = "robot", anchor_link_name: str = "torso_link"
):
    return _anchor_velocity_error(env, entity_name, anchor_link_name, "links_ang_vel_w", "get_links_ang")


# ---------------------------------------------------------------------------
# Body errors (class-style — need a list of tracked link names)
# ---------------------------------------------------------------------------


@_tracking_metric(0.1)
class MotionErrorBodyPos(MetricTerm):
    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"
    links_name: list[str] = []

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        links_pos_rel, _, robot_links_pos, _ = _anchor_relative_links(
            self._env, self.entity_name, self.anchor_link_name, self.links_name
        )
        return torch.norm(links_pos_rel - robot_links_pos, dim=-1).mean(dim=-1)


@_tracking_metric(0.4)
class MotionErrorBodyRot(MetricTerm):
    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"
    links_name: list[str] = []

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        _, links_quat_rel, _, robot_links_quat = _anchor_relative_links(
            self._env, self.entity_name, self.anchor_link_name, self.links_name
        )
        return quat_error_magnitude(links_quat_rel, robot_links_quat).mean(dim=-1)


class _MotionErrorBodyVelocity(MetricTerm):
    """Mean per-body world-frame velocity error magnitude across `links_name`."""

    entity_name: str = "robot"
    links_name: list[str] = []

    _motion_idx: list[int] | None = None
    _robot_idx: torch.Tensor | None = None

    # Subclasses set these.
    _motion_vel_attr: str
    _robot_vel_method: str

    def build(self) -> None:
        self._motion_idx = get_motion_link_idx(self._env, self.entity_name, self.links_name)
        self._robot_idx = get_robot_link_idx(self._env, self.entity_name, self.links_name)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        ref = self._env.motion_manager.data[self.entity_name]
        motion_vel = getattr(ref, self._motion_vel_attr)[:, 0, self._motion_idx]
        robot_vel = getattr(self._env.entities[self.entity_name], self._robot_vel_method)(ls_idx_local=self._robot_idx)
        return torch.norm(motion_vel - robot_vel, dim=-1).mean(dim=-1)


@_tracking_metric(0.7)
class MotionErrorBodyLinVel(_MotionErrorBodyVelocity):
    _motion_vel_attr: str = "links_lin_vel_w"
    _robot_vel_method: str = "get_links_vel"


@_tracking_metric(4.0)
class MotionErrorBodyAngVel(_MotionErrorBodyVelocity):
    _motion_vel_attr: str = "links_ang_vel_w"
    _robot_vel_method: str = "get_links_ang"


# ---------------------------------------------------------------------------
# DOF errors (function-style)
# ---------------------------------------------------------------------------


@_tracking_metric(1.8)
def motion_error_dofs_pos(env: MimicRLEnvBase, *, entity_name: str = "robot"):
    ref = env.motion_manager.data[entity_name]
    return torch.norm(ref.dofs_pos[:, 0] - env.entities[entity_name].get_dofs_pos(), dim=-1)


@_tracking_metric(15.0)
def motion_error_dofs_vel(env: MimicRLEnvBase, *, entity_name: str = "robot"):
    ref = env.motion_manager.data[entity_name]
    return torch.norm(ref.dofs_vel[:, 0] - env.entities[entity_name].get_dofs_vel(), dim=-1)
