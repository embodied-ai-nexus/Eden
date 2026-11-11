"""Terms only the robosuite ``Stack`` port needs: :class:`LiftedAlignXY`, the carry-over-the-target shaping."""

from __future__ import annotations

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm


@REWARD_TERM_REGISTRY.register()
class LiftedAlignXY(RewardTerm):
    """``[z_obj > lift_height] * (1 - tanh(coef * ||xy_obj - xy_target||))`` — robosuite Stack's ``r_lift`` alignment.

    Rewards carrying the lifted object over the target; zero while it is still on the table.
    """

    object_name: str = "object"
    target_name: str = "target"
    lift_height: float = 0.0
    coef: float = 1.0

    def build(self) -> None:
        self.resolve_entity(self.object_name)
        self.resolve_entity(self.target_name)

    def compute(self) -> torch.Tensor:
        # relative=True on both: ``lift_height`` is an authored number, and a difference of two
        # solver-frame positions carries each body's centre-of-mass offset rotated by its own
        # orientation. Free MJCF objects are re-origined; primitives and welded bodies are not.
        obj = self.resolve_entity(self.object_name).get_pos(relative=True)
        target = self.resolve_entity(self.target_name).get_pos(relative=True)
        lifted = (obj[:, 2] > self.lift_height).float()
        dist_xy = torch.norm(obj[:, :2] - target[:, :2], dim=-1)
        return lifted * (1.0 - torch.tanh(self.coef * dist_xy))
