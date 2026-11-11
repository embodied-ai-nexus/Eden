"""Palm-relative "cage" containment: the shared computation behind drop detection.

Port of ``wuji-mjlab``'s ``cage.py``. The cage is the axis-aligned bounding box of every
palm/finger link, expressed in the palm frame and padded by a margin. A cube outside it is
on its way to the floor; how far outside, and for how long, drives an escalating penalty
and eventually the episode termination.

Upstream computes this inside the *reward* term and publishes the counter by writing
``env._cage_penalty_counter``, which four other terms then read back -- an ownership
problem its own README flags. Eden forbids stashing term state on the env, so the
computation lives in one :class:`MetricTerm` here and the reward, termination, critic
observation and diagnostic metric all pull from it. The metric manager runs before
termination, reward and observation in :meth:`RLEnvBase.step`, so every consumer sees a
value computed this step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.managers import METRIC_TERM_REGISTRY, MetricTerm
from eden.tasks.reorient.constants import CAGE_LINKS, PALM_LINK

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.metrics import MetricTermOptions


@METRIC_TERM_REGISTRY.register()
class CageEscape(MetricTerm):
    """Escape distance from the palm-frame hand AABB, plus a time-escalating penalty.

    ``compute()`` returns the escalating penalty (zero while the cube is inside), which is
    what the ``cage_escape`` reward pulls via ``rewards.MetricReward``. Consumers that need
    the raw containment state read the public attributes:

    * :attr:`escape_dist` -- L1 distance outside the padded AABB, ``(num_envs,)``.
    * :attr:`outside` -- whether the cube is outside this step, ``(num_envs,)`` bool.
    * :attr:`penalty_counter` -- +1 per step outside, decaying by ``counter_decay_rate``
      per step inside and capped at ``counter_max``. Gradual decay (not an instant reset)
      is deliberate: it makes "recently outside" distinguishable from "securely held", and
      it is what the ``cage_drop`` termination thresholds on.

    Parameters
    ----------
    hand_entity_name:
        Scene entity name of the hand.
    object_entity_name:
        Scene entity name of the object being contained.
    palm_link:
        Link pattern selecting the palm; the cage is expressed in its frame, so the whole
        measure is invariant to wrist orientation.
    cage_links:
        Link patterns whose positions define the AABB.
    margin:
        Padding [m] added to the AABB on every side except +z.
    z_upper_margin:
        Padding [m] on the +z (finger-direction) face. Larger than ``margin`` upstream,
        because lifting the cube along the fingers is normal reorientation behaviour.
    max_outside_steps:
        Counter value that corresponds to a full-strength penalty (and, in the termination
        term, to a drop).
    drop_scale:
        Overall penalty gain.
    counter_decay_rate:
        Counter decay per step while inside.
    counter_max:
        Counter ceiling, so a long excursion cannot produce an unbounded penalty.
    """

    hand_entity_name: str = "robot"
    object_entity_name: str = "cube"
    palm_link: str = PALM_LINK
    cage_links: list[str] = CAGE_LINKS
    margin: float = 0.01
    z_upper_margin: float = 0.03
    max_outside_steps: int = 10
    drop_scale: float = 4.0
    counter_decay_rate: float = 0.5
    counter_max: float = 15.0

    contributes_to_success: bool = False

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env, options)
        self._palm_idx: torch.Tensor | None = None
        self._cage_idx: torch.Tensor | None = None
        self.escape_dist = torch.zeros(self.num_envs, device=self.device)
        self.outside = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.penalty_counter = torch.zeros(self.num_envs, device=self.device)

    def build(self) -> None:
        hand = self._env.entities[self.hand_entity_name]
        self._env.entities[self.object_entity_name]
        _, palm_idx = hand.find_named_links_idx_local(self.palm_link, preserve_order=True)
        if len(palm_idx) != 1:
            raise ValueError(
                f"CageEscape: palm_link={self.palm_link!r} matched {len(palm_idx)} links on entity "
                f"'{self.hand_entity_name}'; expected exactly one."
            )
        _, cage_idx = hand.find_named_links_idx_local(self.cage_links, preserve_order=True)
        if len(cage_idx) < 2:
            raise ValueError(
                f"CageEscape: cage_links={self.cage_links!r} matched {len(cage_idx)} links on entity "
                f"'{self.hand_entity_name}'; an AABB needs at least two."
            )
        self._palm_idx = torch.as_tensor(palm_idx, dtype=torch.long, device=self.device)
        self._cage_idx = torch.as_tensor(cage_idx, dtype=torch.long, device=self.device)
        if self.max_outside_steps < 1:
            raise ValueError(f"CageEscape: max_outside_steps must be >= 1, got {self.max_outside_steps}.")

    def compute(self) -> torch.Tensor:
        hand = self.resolve_entity(self.hand_entity_name)
        obj = self.resolve_entity(self.object_entity_name)

        palm_pos = hand.get_links_pos(ls_idx_local=self._palm_idx).squeeze(1)
        palm_quat = hand.get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)

        # Hand link cloud and the cube, both in the palm frame.
        cage_rel = hand.get_links_pos(ls_idx_local=self._cage_idx) - palm_pos.unsqueeze(1)
        palm_quat_b = palm_quat.unsqueeze(1).expand(-1, cage_rel.shape[1], -1)
        cage_in_palm = inv_transform_by_quat(cage_rel, palm_quat_b)
        cube_in_palm = inv_transform_by_quat(obj.get_pos() - palm_pos, palm_quat)

        lower = cage_in_palm.min(dim=1).values - self.margin
        upper = cage_in_palm.max(dim=1).values + self.margin
        # +z is the finger direction: allow more headroom there than on the other faces.
        upper[:, 2] = cage_in_palm[:, :, 2].max(dim=1).values + self.z_upper_margin

        outside_by = torch.relu(lower - cube_in_palm) + torch.relu(cube_in_palm - upper)
        self.escape_dist = outside_by.sum(dim=-1)
        self.outside = self.escape_dist > 0.0

        self.penalty_counter = torch.where(
            self.outside,
            self.penalty_counter + 1.0,
            (self.penalty_counter - self.counter_decay_rate).clamp(min=0.0),
        ).clamp(max=self.counter_max)

        # Escalates with how long the excursion has lasted, and is zero while inside -- so a
        # brief overshoot the policy recovers from costs far less than a slide out of the hand.
        return self.outside.float() * self.penalty_counter / float(self.max_outside_steps) * self.drop_scale

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self.escape_dist[envs_idx] = 0.0
        self.outside[envs_idx] = False
        self.penalty_counter[envs_idx] = 0.0


__all__ = ["CageEscape"]
