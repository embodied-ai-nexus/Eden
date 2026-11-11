"""Terms only the robosuite ``TwoArmHandover`` port needs.

:class:`LiftedReachTanh` is robosuite's second reaching stage: the receiving arm is only rewarded
for approaching the hammer once the giving arm has lifted it. The gate is a metric read by name so
the lift predicate lives in one place.
"""

from __future__ import annotations

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.tasks.robosuite.custom_terms.common import GraspFrameMixin, ObjectPointMixin


@REWARD_TERM_REGISTRY.register()
class LiftedReachTanh(ObjectPointMixin, GraspFrameMixin, RewardTerm):
    """``[gate] * (1 - tanh(coef * ||object_point - grasp_point||))`` — a reach gated on a metric.

    Pays only while the ``gate_metric_name`` metric is 1, read from the metric manager (computed
    earlier in the same step). ``object_link_name`` / ``object_offset`` pick the object point as in the
    shared ``ReachTanh``.
    """

    rig: str = "robot"
    object_name: str = "object"
    object_link_name: str | None = None
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gate_metric_name: str = ""
    coef: float = 1.0

    def build(self) -> None:
        self._build_grasp_frame()
        self._build_object_point()
        # The metric manager is constructed ahead of the reward manager (RLEnvBase._manager_order), so
        # the gate resolves here: a bad name raises at env construction, not on the first step.
        self._env.metric_manager.get_value(self.gate_metric_name)

    def compute(self) -> torch.Tensor:
        gate = self._env.metric_manager.get_value(self.gate_metric_name)
        dist = torch.norm(self._object_point_w() - self._grasp_pos_w(), dim=-1)
        return gate * (1.0 - torch.tanh(self.coef * dist))
