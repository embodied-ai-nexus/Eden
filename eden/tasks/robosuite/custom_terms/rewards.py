"""Reward terms for the robosuite ports.

robosuite shapes every reaching phase as ``1 - tanh(coef * distance)`` between the ``grip_site``
and a target; the grasp bonuses are pulled from the :class:`PadsGrasping` metric through
``rewards.MetricReward`` rather than recomputed here.
"""

from __future__ import annotations

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.tasks.robosuite.custom_terms.common import GraspFrameMixin, ObjectPointMixin


@REWARD_TERM_REGISTRY.register()
class ReachTanh(ObjectPointMixin, GraspFrameMixin, RewardTerm):
    """``1 - tanh(coef * ||object_point - grasp_point||)`` — robosuite's reaching reward. ``(N,)`` in ``[0, 1]``.

    The object point is the root by default; ``object_link_name`` / ``object_offset`` pick a link or a
    site-like fixed point instead (see :class:`~eden.tasks.robosuite.custom_terms.common.ObjectPointMixin`).
    """

    rig: str = "robot"
    object_name: str = "object"
    object_link_name: str | None = None
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coef: float = 10.0

    def build(self) -> None:
        self._build_grasp_frame()
        self._build_object_point()

    def compute(self) -> torch.Tensor:
        dist = torch.norm(self._object_point_w() - self._grasp_pos_w(), dim=-1)
        return 1.0 - torch.tanh(self.coef * dist)


@REWARD_TERM_REGISTRY.register()
class HoverOverPoint(RewardTerm):
    """``[z_obj > lift_height] * (1 - tanh(coef * ||xy_obj - point_xy||))`` — robosuite's ``r_hover`` alignment.

    Rewards carrying the lifted object over a fixed world point (a bin quadrant centre, a peg); zero
    while the object is still below ``lift_height``. Stack's
    :class:`~eden.tasks.robosuite.stack.custom_terms.LiftedAlignXY` aligns to another *entity* instead.
    ``(N,)`` in ``[0, 1]``.
    """

    object_name: str = "object"
    point_xy: tuple[float, float] = (0.0, 0.0)
    lift_height: float = 0.0
    coef: float = 10.0

    def build(self) -> None:
        self.resolve_entity(self.object_name)
        self._point = torch.tensor(self.point_xy, device=self.device, dtype=torch.float32)

    def compute(self) -> torch.Tensor:
        # relative=True: ``point_xy`` / ``lift_height`` are authored from the bin and the asset's
        # own geometry, so the object has to be read in the frame it was authored in — the default
        # solver frame is re-origined onto a free body's centre of mass.
        pos = self.resolve_entity(self.object_name).get_pos(relative=True)
        lifted = (pos[:, 2] > self.lift_height).float()
        dist_xy = torch.norm(pos[:, :2] - self._point, dim=-1)
        return lifted * (1.0 - torch.tanh(self.coef * dist_xy))
