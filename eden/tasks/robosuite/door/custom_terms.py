"""Terms only the robosuite ``Door`` port needs: :class:`DofFractionOfRange`, the handle-rotation shaping,
and :class:`DofAbove`, the door-open success test.

The handle *site* robosuite reaches for (12.5 cm along the lever, 10 cm off the hub, in the ``latch``
body frame) is reproduced with the shared terms' ``object_link_name`` / ``object_offset``.
"""

from __future__ import annotations

import math

import torch
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm


@REWARD_TERM_REGISTRY.register()
class DofFractionOfRange(RewardTerm):
    """``clip(|q| / max_angle, 0, 1)`` of one DOF — robosuite Door's handle-rotation shaping. ``(N,)``.

    robosuite adds ``clip(0.25 * |handle_qpos| / (pi/2), -0.25, 0.25)``; configure this term with
    ``weight=0.25`` and the default ``max_angle`` to reproduce it.
    """

    entity_name: str = "object"
    dof_name: str = ""
    max_angle: float = math.pi / 2

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self._idx = entity.find_named_dofs([self.dof_name]).dofs_idx_local

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self.entity_name).get_dofs_pos(dofs_idx_local=self._idx)[:, 0]
        return torch.clamp(q.abs() / self.max_angle, 0.0, 1.0)


@METRIC_TERM_REGISTRY.register()
class DofAbove(MetricTerm):
    """1.0 when one DOF of an articulated object exceeds ``threshold`` (robosuite Door: ``hinge_qpos > 0.3``)."""

    entity_name: str = "object"
    dof_name: str = ""
    threshold: float = 0.0

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self._idx = entity.find_named_dofs([self.dof_name]).dofs_idx_local

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self.entity_name).get_dofs_pos(dofs_idx_local=self._idx)[:, 0]
        return (q > self.threshold).float()
