"""Metric terms for the robosuite ports: the predicates ``_check_success`` and ``reward()`` share.

Grasp detection and the success stages are computed once here and pulled into rewards
(``rewards.MetricReward``) and terminations (``terminations.OnMetricSuccess``) by name.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.tasks.robosuite.custom_terms.common import RigMixin, pad_link_global_indices, pads_touching

#: Metric fields that decide how a value is judged (or whether it counts), not what it is: a success
#: metric's child and the sibling metric computing the same quantity differ in exactly these.
_JUDGING_FIELDS = frozenset(
    {"contributes_to_success", "success_threshold", "direction", "metric_mode", "is_cumulative"}
)


def shared_per_metric_pass(compute: Callable[[MetricTerm], torch.Tensor]) -> Callable[[MetricTerm], torch.Tensor]:
    """Decorate a metric term's ``compute`` so each distinct configuration runs once per metric pass.

    A task's success metric often ANDs the same tests its sibling metrics already compute for the
    rewards (ToolHang's ``success`` is ``frame_assembled`` and ``tool_on_frame``'s tests together).
    Decorated, the second instance of a test is served the first one's value through
    :meth:`~eden.managers.base.ManagerBase.compute_once`, keyed by the term's class and every
    configured field except the judging ones (:data:`_JUDGING_FIELDS`) — so terms that differ in any
    input are never conflated. Outside a metric pass the term simply computes. The value is shared:
    callers must not modify it in place.
    """

    @functools.wraps(compute)
    def compute_shared(self: MetricTerm) -> torch.Tensor:
        manager = self._env.metric_manager
        if manager is None:
            return compute(self)
        key = (type(self).__qualname__, self.configured_fingerprint(exclude=_JUDGING_FIELDS))
        return manager.compute_once(key, lambda: compute(self))

    return compute_shared


@METRIC_TERM_REGISTRY.register()
class PadsGrasping(RigMixin, MetricTerm):
    """1.0 when both of the rig's pad links touch ``object_name`` — robosuite's ``_check_grasp``.

    Geometric contact, not force-aligned like :class:`~eden.managers.terms.metrics.grasp.IsGrasping`:
    robosuite counts any fingerpad-object contact pair. Configure with
    ``contributes_to_success=False`` when it is only a reward/gate input.
    """

    rig: str = "robot"
    object_name: str = "object"

    def build(self) -> None:
        rig = self._rig_options()
        self._gripper_name = rig.gripper_name
        gripper = self.resolve_entity(rig.gripper_name)
        self.resolve_entity(self.object_name)
        if len(rig.grasp_links) < 2:
            raise ValueError(
                f"PadsGrasping needs two pad links; rig '{self.rig}' declares grasp_links={rig.grasp_links!r}."
            )
        self._pad_idx = pad_link_global_indices(gripper, list(rig.grasp_links))

    def compute(self) -> torch.Tensor:
        gripper = self.resolve_entity(self._gripper_name)
        return pads_touching(gripper, self.resolve_entity(self.object_name), self._pad_idx).float()


@METRIC_TERM_REGISTRY.register()
class EntitiesInContact(MetricTerm):
    """1.0 when ``entity_a`` and ``entity_b`` share a contact pair (robosuite ``check_contact``).

    ``links_a`` / ``links_b`` restrict either side to named links (e.g. a bin's base slab).
    """

    entity_a_name: str = "object_a"
    entity_b_name: str = "object_b"
    links_a: list[str] = []
    links_b: list[str] = []

    def build(self) -> None:
        a = self.resolve_entity(self.entity_a_name)
        b = self.resolve_entity(self.entity_b_name)
        self._links_a = [a.get_link(n).idx for n in self.links_a] or None
        self._links_b = [b.get_link(n).idx for n in self.links_b] or None

    @staticmethod
    def _side_mask(contacts: dict[str, torch.Tensor], links: list[int] | None) -> torch.Tensor:
        if links is None:
            return torch.ones_like(contacts["valid_mask"])
        mask = torch.zeros_like(contacts["valid_mask"])
        for idx in links:
            mask |= (contacts["link_a"] == idx) | (contacts["link_b"] == idx)
        return mask

    @shared_per_metric_pass
    def compute(self) -> torch.Tensor:
        a = self.resolve_entity(self.entity_a_name)
        contacts = a.get_contacts(with_entity=self.resolve_entity(self.entity_b_name))
        hit = (
            contacts["valid_mask"] & self._side_mask(contacts, self._links_a) & self._side_mask(contacts, self._links_b)
        )
        return hit.any(dim=-1).float()
