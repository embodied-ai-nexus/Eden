"""Hand<->cube and hand<->hand contact state, shared by the rewards that need it.

Replaces four of ``wuji-mjlab``'s seven ``ContactSensorCfg`` declarations with two
:class:`MetricTerm`s over Genesis' contact queries. (The other three --
``palm_object_contact``, ``undesired_object_contact`` and ``robot_contact`` -- are declared
upstream but read by no term, so they are not ported.)

Genesis has no contact *sensor* with mjlab's ``found``/``netforce`` reductions;
:meth:`RigidEntity.get_grouped_contacts` gives the same information per (hand link, object
link) pair. Its ``valid`` flag means "the pair's summed contact-force magnitude is > 0",
which is the right analogue of mjlab's ``found``: upstream deliberately avoids a force
*threshold* here because the 120 g cube only presses with ~1.2 N, and ``> 0`` imposes no
threshold at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import METRIC_TERM_REGISTRY, MetricTerm
from eden.tasks.reorient.constants import DISTAL_LINKS, PALM_LINK, PROXIMAL_LINKS, TIP_COLLISION_LINKS

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.metrics import MetricTermOptions


@METRIC_TERM_REGISTRY.register()
class HandObjectContact(MetricTerm):
    """Which parts of the hand are touching the object, resolved once per step.

    ``compute()`` returns the number of fingertips in contact with the object -- upstream's
    ``fingertip_contact_count`` metric, and a direct read on whether the policy is holding
    the cube in its fingers at all. Consumers read the flags:

    * :attr:`tip_contact` -- ``(num_envs, n_tips)`` bool, per-fingertip contact. Gates the
      fingertip-slide penalty.
    * :attr:`palm_or_proximal_contact` -- ``(num_envs,)`` bool, object touching the palm or
      any proximal phalanx.
    * :attr:`distal_contact` -- ``(num_envs,)`` bool, object touching any distal phalanx.

    The last two are the two halves of the ``palm_detach`` reward: the cube should be held
    by the distal phalanges *and* clear of the palm.

    Parameters
    ----------
    hand_entity_name:
        Scene entity name of the hand.
    object_entity_name:
        Scene entity name of the object.
    tip_links:
        Link patterns for the fingertip contact row (upstream uses the distal ``link4``
        collision geoms; the tip frames carry no collision geometry of their own).
    palm_proximal_links:
        Link patterns for the "resting on the hand" set.
    distal_links:
        Link patterns for the "held in the fingers" set.
    """

    hand_entity_name: str = "robot"
    object_entity_name: str = "cube"
    tip_links: list[str] = [TIP_COLLISION_LINKS]
    palm_proximal_links: list[str] = [PALM_LINK, PROXIMAL_LINKS]
    distal_links: list[str] = [DISTAL_LINKS]

    contributes_to_success: bool = False

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env, options)
        self._tip_idx: torch.Tensor | None = None
        self._palm_proximal_idx: torch.Tensor | None = None
        self._distal_idx: torch.Tensor | None = None
        self._links_idx: torch.Tensor | None = None
        self.tip_contact = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.palm_or_proximal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.distal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def build(self) -> None:
        hand = self._env.entities[self.hand_entity_name]
        self._env.entities[self.object_entity_name]
        self._tip_idx = self._resolve(hand, self.tip_links, "tip_links")
        self._palm_proximal_idx = self._resolve(hand, self.palm_proximal_links, "palm_proximal_links")
        self._distal_idx = self._resolve(hand, self.distal_links, "distal_links")
        # Concatenated once: a fresh tensor per step would be bounds-checked (and synced on) every step.
        self._links_idx = torch.cat([self._tip_idx, self._palm_proximal_idx, self._distal_idx])
        self.tip_contact = torch.zeros(self.num_envs, self._tip_idx.numel(), dtype=torch.bool, device=self.device)

    def _resolve(self, hand, patterns: list[str], field: str) -> torch.Tensor:
        _, idx = hand.find_named_links_idx_local(patterns, preserve_order=True)
        if len(idx) == 0:
            raise ValueError(
                f"HandObjectContact: {field}={patterns!r} matched no links on entity "
                f"'{self.hand_entity_name}'. Available links: {[link.name for link in hand.links]}"
            )
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        hand = self.resolve_entity(self.hand_entity_name)
        obj = self.resolve_entity(self.object_entity_name)
        # One grouped query over every link of interest, sliced afterwards: the query cost is
        # dominated by the contact scan, not by the number of rows requested.
        valid = hand.get_grouped_contacts(obj, links_a_idx_local=self._links_idx)["valid"]  # (N, L, n_obj_links)
        touching = valid.any(dim=-1)  # (N, L)

        n_tip = self._tip_idx.numel()
        n_palm = self._palm_proximal_idx.numel()
        self.tip_contact = touching[:, :n_tip]
        self.palm_or_proximal_contact = touching[:, n_tip : n_tip + n_palm].any(dim=-1)
        self.distal_contact = touching[:, n_tip + n_palm :].any(dim=-1)

        return self.tip_contact.float().sum(dim=-1)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self.tip_contact[envs_idx] = False
        self.palm_or_proximal_contact[envs_idx] = False
        self.distal_contact[envs_idx] = False


@METRIC_TERM_REGISTRY.register()
class SelfCollisionCount(MetricTerm):
    """Number of active hand-internal contacts this step.

    Upstream's ``finger_collision`` sensor plus the reward and frequency metric that read
    it, collapsed into one term: the value is both the penalty magnitude (pulled by the
    ``finger_collision`` reward via ``rewards.MetricReward``) and the logged diagnostic.
    Genesis already filters parent-child pairs and any pair overlapping at the neutral
    configuration, so what is left is genuine finger-on-finger jamming.

    Requires ``EnvOptions(enable_self_collision=True)``; without it this is always zero and
    the penalty silently does nothing, so ``build`` refuses that configuration.

    Parameters
    ----------
    hand_entity_name:
        Scene entity name of the hand.
    """

    hand_entity_name: str = "robot"

    contributes_to_success: bool = False

    def build(self) -> None:
        self._env.entities[self.hand_entity_name]
        if not self._env.env_options.enable_self_collision:
            raise ValueError(
                "SelfCollisionCount requires EnvOptions(enable_self_collision=True); with self-collision "
                "off the query returns no contacts and the finger-collision penalty is silently inert."
            )

    def compute(self) -> torch.Tensor:
        hand = self.resolve_entity(self.hand_entity_name)
        contacts = hand.get_contacts(with_entity=hand, exclude_self_contact=False)
        return contacts["valid_mask"].sum(dim=1).float()


__all__ = ["HandObjectContact", "SelfCollisionCount"]
