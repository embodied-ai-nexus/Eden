"""Contact-cloud Chamfer reward.

Scores policy-vs-demo per-link-pair contact configurations via masked
Chamfer distance, in either world / object / wrist frame, with exp or
sigmoid kernels. Builds on :meth:`RigidEntity.get_grouped_contacts`
(the policy contact-cloud aggregator) and :func:`eden.utils.pc.chamfer_distance`
(the distance kernel).

Reads the demo-side contact target from the motion manager
(``env.motion_manager.data[hand_entity_name].contact_points_w`` /
``contact_points_valid``). A dataset packs them through ``add_motion_sequence``;
when the loaded motion data does not carry them,
:meth:`MotionContactChamferReward.build` raises via :func:`assert_motion_field_present`.

Mirrors DexMachina ``RewardModule.compute_hand_contact_reward``
(``dexmachina/envs/rewards.py:293``) at single-frame, single-part
granularity. Per-part / multi-frame composition is left to the config
layer (one term per part / frame), keeping the reward kernel composable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal

import torch

from genesis.utils.geom import inv_transform_by_trans_quat

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.utils.motion_reference.lookup import (
    assert_motion_field_present,
    get_motion_link_idx,
)
from eden.utils.pc import chamfer_distance

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


EMPTY_DISTANCE_SENTINEL: Final[float] = 100.0
"""Per-direction Chamfer distance returned when one cloud is empty and the
other is not. Mirrors DexMachina ``reward_utils.chamfer_distance``
(``dexmachina/envs/reward_utils.py:44``) — large
enough that ``exp(-100/std**2) ≈ 0`` turns "policy contacts but demo does
not" (or vice-versa) into a hard penalty. ``chamfer_distance`` itself
defaults to ``0.0`` (no signal); this term overrides to the DexMachina
value."""


def _compute_contact_chamfer_reward(
    *,
    policy_pos: torch.Tensor,  # (B, n_a, n_b, 3)
    policy_valid: torch.Tensor,  # (B, n_a, n_b) bool
    policy_anchor_pos: torch.Tensor | None,  # (B, 3) or None
    policy_anchor_quat: torch.Tensor | None,  # (B, 4) wxyz or None
    demo_pos: torch.Tensor,  # (B, n_a, n_b, 3)
    demo_valid: torch.Tensor,  # (B, n_a, n_b) bool
    demo_anchor_pos: torch.Tensor | None,  # (B, 3) or None
    demo_anchor_quat: torch.Tensor | None,  # (B, 4) wxyz or None
    kernel: Literal["exp", "sigmoid"],
    std: float,
    sigmoid_a: float,
    sigmoid_b: float,
    sigmoid_offset: float,
    zero_contact_mode: Literal["zero", "one"],
    empty_distance: float,
) -> torch.Tensor:
    """Pure-tensor Chamfer-reward kernel with independent per-cloud anchors.

    Each cloud is brought into its own anchor's local frame before the
    Chamfer comparison — policy by ``policy_anchor_*``, demo by
    ``demo_anchor_*``. This matches DexMachina's
    ``compute_hand_contact_reward`` (``rewards.py:321-328``), where the
    demo is transformed by the demo's recorded object/wrist pose and the
    policy by the simulator's current object/wrist pose. The result is
    invariant to world-space offset between the two anchors — i.e. the
    Chamfer distance reflects contact-pattern error only, decoupled from
    object-tracking error.

    Either anchor may be ``None`` for "no transform" (world frame); the
    two anchors are independent — passing one and not the other is a
    valid mixed-frame configuration.

    Split out so unit tests can drive the math without an env. See
    :class:`MotionContactChamferReward` for the per-step orchestration.
    """
    if policy_anchor_pos is not None:
        a_pos = policy_anchor_pos.unsqueeze(1).unsqueeze(1)
        a_quat = policy_anchor_quat.unsqueeze(1).unsqueeze(1)
        policy_pos = inv_transform_by_trans_quat(policy_pos, a_pos, a_quat)
    if demo_anchor_pos is not None:
        d_pos = demo_anchor_pos.unsqueeze(1).unsqueeze(1)
        d_quat = demo_anchor_quat.unsqueeze(1).unsqueeze(1)
        demo_pos = inv_transform_by_trans_quat(demo_pos, d_pos, d_quat)

    B = policy_pos.shape[0]
    d = chamfer_distance(
        policy_pos.reshape(B, -1, 3),
        demo_pos.reshape(B, -1, 3),
        policy_valid.reshape(B, -1),
        demo_valid.reshape(B, -1),
        bidirectional=True,
        empty_distance=empty_distance,
    )

    if kernel == "exp":
        reward = torch.exp(-d / (std**2))
    else:
        # Negative slope: reward decreases as Chamfer distance grows (DexMachina's sigmoid(b - a*d)
        # contact kernel); the offset keeps reward=1 at d=0. Only monotonicity is unit-tested: if
        # contact-reward tracking misbehaves, verify sign/offset here against DexMachina.
        reward = torch.sigmoid(-sigmoid_a * d + sigmoid_b) + sigmoid_offset

    flat_policy = policy_valid.reshape(B, -1).any(dim=-1)
    flat_demo = demo_valid.reshape(B, -1).any(dim=-1)
    both_empty = (~flat_policy) & (~flat_demo)
    if zero_contact_mode == "zero":
        reward = torch.where(both_empty, torch.zeros_like(reward), reward)
    else:
        reward = torch.where(both_empty, torch.ones_like(reward), reward)
    return reward


@REWARD_TERM_REGISTRY.register()
class MotionContactChamferReward(RewardTerm):
    """Exp/sigmoid reward on the policy-vs-demo contact Chamfer distance.

    Per-step semantics:

    1. Policy contacts: ``hand.get_grouped_contacts(object, links_a, links_b)``
       — ``(B, n_a, n_b, 3)`` weighted-mean positions plus
       ``(B, n_a, n_b)`` validity mask.
    2. Demo contacts: ``env.motion_manager.data[hand_entity_name].contact_points_w[:, lookahead_index]``
       — same shape. Validity from ``contact_points_valid[:, lookahead_index]``.
    3. Each cloud is transformed by its **own** anchor before the Chamfer
       comparison (matches DexMachina ``rewards.py:321-328``):

       * ``frame="object"``: policy uses ``obj.get_pos()`` / ``get_quat()``;
         demo uses ``motion_manager.data[object_entity_name].base_pos_w``
         / ``base_quat_w`` at ``lookahead_index``.
       * ``frame="wrist"``: policy uses the named wrist link's current
         pose on the hand; demo uses ``links_pos_w`` / ``links_quat_w``
         on the hand's motion data at ``lookahead_index`` and the
         motion-side wrist index.
       * ``frame="world"``: no transform.

       Two instances composed at the config layer (``frame="object"``
       and ``frame="wrist"``, summed via term weights) reproduce
       DexMachina's ``multiply_frame_contact=True`` once weights are
       set up — the dual-anchor frame transform is what makes the
       composition meaningful.
    4. ``chamfer_distance`` over the flattened ``(n_a, n_b)`` →
       ``N`` clouds with ``empty_distance=EMPTY_DISTANCE_SENTINEL``
       (DexMachina sentinel, 100.0).
    5. Distance → reward via the configured kernel.
    6. Both-empty mask: per ``zero_contact_mode``.

    Parameters
    ----------
    hand_entity_name:
        Active entity (usually the hand). Indexes axis 1 of the
        grouped-contact tensors.
    object_entity_name:
        Contact partner (usually the manipulated object). Indexes
        axis 2.
    hand_links_name:
        :func:`resolve_matching_names`-compatible patterns selecting
        which links on ``hand_entity_name`` contribute. Empty → all
        links on the hand.
    object_links_name:
        Same, on the object side.
    frame:
        Frame the Chamfer distance is measured in. ``"world"`` (default),
        ``"object"``, or ``"wrist"``. Each cloud is anchored independently
        to its own object/wrist pose.
    wrist_link_name:
        Required iff ``frame="wrist"``. Names the link on
        ``hand_entity_name`` whose pose defines the local frame. Resolved
        on both the robot side (current sim pose) and the motion side
        (demo recorded pose) at ``build()``.
    lookahead_index:
        Frame index into the per-env motion-reference window. ``0``
        (default) reads the current target, the convention of the
        motion-tracking terms. ``>0`` reads future targets
        in the look-ahead window. Validated at ``build()`` against the
        per-env window size.
    kernel:
        ``"exp"`` (default) or ``"sigmoid"``.
    std:
        Exp-kernel scale; ``exp(-d / std**2)``. Default
        ``1/sqrt(30) ≈ 0.1826`` for exact parity with DexMachina
        ``contact_beta=30``. Mapping: ``std = 1/sqrt(beta)``.
    sigmoid_a, sigmoid_b:
        Sigmoid-kernel constants; ignored unless ``kernel="sigmoid"``.
        Defaults match DexMachina (``a=100``, ``b=5``).
    zero_contact_mode:
        ``"zero"`` (default, DexMachina ``mask_zero_contact=True``) or
        ``"one"``.
    empty_distance:
        Passed to :func:`chamfer_distance`. Default
        :data:`EMPTY_DISTANCE_SENTINEL` (100.0) mirrors DexMachina;
        ``chamfer_distance`` itself defaults to ``0.0``.

    Notes
    -----
    The demo-side fields ``contact_points_w`` and ``contact_points_valid`` of
    :class:`MotionReferenceData` are optional. When the loaded motion data does
    not carry them, :meth:`build` raises via :func:`assert_motion_field_present`,
    so the reward cannot silently score against zeros.

    The term assumes a free-floating hand and free-floating object —
    typical for dexterous manipulation. Arm-mounted hand setups are
    supported because the hand sub-tree remains free relative to the
    (possibly fixed) arm; the wrist anchor still resolves correctly.
    """

    hand_entity_name: str = "robot"
    object_entity_name: str = "object"
    hand_links_name: list[str] = []
    object_links_name: list[str] = []
    frame: Literal["world", "object", "wrist"] = "world"
    wrist_link_name: str | None = None
    lookahead_index: int = 0
    kernel: Literal["exp", "sigmoid"] = "exp"
    std: float = 1.0 / math.sqrt(30.0)
    sigmoid_a: float = 100.0
    sigmoid_b: float = 5.0
    zero_contact_mode: Literal["zero", "one"] = "zero"
    empty_distance: float = EMPTY_DISTANCE_SENTINEL

    def build(self) -> None:
        env = self._env
        hand: RigidEntity = env.get_entity(self.hand_entity_name)
        obj: RigidEntity = env.get_entity(self.object_entity_name)

        if self.lookahead_index < 0:
            raise ValueError(f"MotionContactChamferReward.lookahead_index must be >= 0, got {self.lookahead_index}.")

        self._hand_links_idx = self._resolve_links(hand, self.hand_links_name, "hand_links_name")
        self._object_links_idx = self._resolve_links(obj, self.object_links_name, "object_links_name")

        if self.frame == "wrist":
            if self.wrist_link_name is None:
                raise ValueError(
                    f"MotionContactChamferReward(frame='wrist') requires wrist_link_name on entity "
                    f"'{self.hand_entity_name}'."
                )
            _, wrist_idx = hand.find_named_links_idx_local([self.wrist_link_name], preserve_order=True)
            self._wrist_link_idx = torch.tensor([wrist_idx[0]], dtype=torch.long, device=self.device)

        # Fail-fast: require the demo contact cloud plus the demo-side anchor fields the chosen frame requires.
        assert_motion_field_present(env, self.hand_entity_name, "contact_points_w")
        assert_motion_field_present(env, self.hand_entity_name, "contact_points_valid")
        if self.frame == "object":
            assert_motion_field_present(env, self.object_entity_name, "base_pos_w")
            assert_motion_field_present(env, self.object_entity_name, "base_quat_w")
        elif self.frame == "wrist":
            assert_motion_field_present(env, self.hand_entity_name, "links_pos_w")
            assert_motion_field_present(env, self.hand_entity_name, "links_quat_w")
            # Resolve the wrist's motion-side index — may differ from the
            # robot-side index if the motion sequence tracks a permuted /
            # subset of the hand's links.
            self._motion_wrist_idx = get_motion_link_idx(env, self.hand_entity_name, [self.wrist_link_name])
            # A Python int, read once here: indexing with the 0-dim CUDA `idx[0]` syncs on every use.
            self._motion_wrist_col = int(self._motion_wrist_idx[0])

        # Per-env motion-reference window size. ``contact_points_w`` may not
        # be allocated on the buffer yet; use ``dofs_pos`` — always populated
        # by ``MotionReferenceData.make_empty`` — as the window-size oracle
        # (all per-frame fields share num_frames).
        ref = env.motion_manager.data[self.hand_entity_name]
        window = ref.dofs_pos.shape[1]
        if self.lookahead_index >= window:
            raise ValueError(
                f"MotionContactChamferReward.lookahead_index={self.lookahead_index} "
                f"is out of bounds for motion-reference window size {window} on "
                f"entity '{self.hand_entity_name}'."
            )

        # Sigmoid-offset pre-computation matches DexMachina sigmoid_offset:
        #   offset = 1 - sigmoid(b)  → sigmoid(-a*0+b) + offset = 1 at d=0.
        if self.kernel == "sigmoid":
            # The kernel is sigmoid(-a*d + b): with sigmoid_a > 0 it *decreases* as the Chamfer
            # distance d grows (worse tracking → lower reward). sigmoid_a <= 0 would flip that and
            # reward worse tracking, so require it strictly positive.
            if self.sigmoid_a <= 0:
                raise ValueError(f"MotionContactChamferReward.sigmoid_a must be > 0, got {self.sigmoid_a}.")
            self._sigmoid_offset = 1.0 - 1.0 / (1.0 + math.exp(-self.sigmoid_b))
        else:
            self._sigmoid_offset = 0.0

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        env = self._env
        hand: RigidEntity = env.get_entity(self.hand_entity_name)
        obj: RigidEntity = env.get_entity(self.object_entity_name)
        i = self.lookahead_index

        policy = hand.get_grouped_contacts(
            obj,
            links_a_idx_local=self._hand_links_idx,
            links_b_idx_local=self._object_links_idx,
        )
        hand_ref = env.motion_manager.data[self.hand_entity_name]

        policy_anchor_pos: torch.Tensor | None = None
        policy_anchor_quat: torch.Tensor | None = None
        demo_anchor_pos: torch.Tensor | None = None
        demo_anchor_quat: torch.Tensor | None = None

        if self.frame == "object":
            obj_ref = env.motion_manager.data[self.object_entity_name]
            policy_anchor_pos = obj.get_pos()
            policy_anchor_quat = obj.get_quat()
            demo_anchor_pos = obj_ref.base_pos_w[:, i]
            demo_anchor_quat = obj_ref.base_quat_w[:, i]
        elif self.frame == "wrist":
            policy_anchor_pos = hand.get_links_pos(ls_idx_local=self._wrist_link_idx)[:, 0]
            policy_anchor_quat = hand.get_links_quat(ls_idx_local=self._wrist_link_idx)[:, 0]
            demo_anchor_pos = hand_ref.links_pos_w[:, i, self._motion_wrist_col]
            demo_anchor_quat = hand_ref.links_quat_w[:, i, self._motion_wrist_col]

        return _compute_contact_chamfer_reward(
            policy_pos=policy["position"],
            policy_valid=policy["valid"],
            policy_anchor_pos=policy_anchor_pos,
            policy_anchor_quat=policy_anchor_quat,
            demo_pos=hand_ref.contact_points_w[:, i],
            demo_valid=hand_ref.contact_points_valid[:, i],
            demo_anchor_pos=demo_anchor_pos,
            demo_anchor_quat=demo_anchor_quat,
            kernel=self.kernel,
            std=self.std,
            sigmoid_a=self.sigmoid_a,
            sigmoid_b=self.sigmoid_b,
            sigmoid_offset=self._sigmoid_offset,
            zero_contact_mode=self.zero_contact_mode,
            empty_distance=self.empty_distance,
        )

    @staticmethod
    def _resolve_links(entity: "RigidEntity", patterns: list[str], kwarg: str) -> torch.Tensor | None:
        """Resolve link-name patterns to a 1-D long tensor of local indices.

        Returns ``None`` for empty patterns (consumed by
        ``get_grouped_contacts`` as "all links"), or a 1-D long tensor
        of local indices on ``entity.device``. Unknown patterns raise
        from the inner ``resolve_matching_names`` so config typos fail
        loudly at ``build()`` time.
        """
        if not patterns:
            return None
        _, link_idx = entity.find_named_links_idx_local(patterns, preserve_order=True)
        return torch.tensor(link_idx, dtype=torch.long, device=entity.device)


class _MotionContactTermBase(RewardTerm):
    """Shared build/lookup for per-link demo-vs-policy contact terms.

    Resolves the same link set on both sides — robot links (rows of
    ``get_grouped_contacts``) and motion links (rows of ``contact_points_*``)
    — so subclasses compare per-link pairs by position ``i`` without any
    runtime name resolution.

    Parameters
    ----------
    hand_links_name:
        Link patterns on ``hand_entity_name`` defining the row pairing;
        resolved identically on the robot and motion-ref sides. Required.
    object_links_name:
        Ordered object-link names declaring which object link each demo
        partner *column* corresponds to. The policy contact axis is reordered
        to this order (``links_b_idx_local``) so the demo↔policy column
        correspondence is name-anchored rather than tied to Genesis's internal
        link order. Empty selects all object links in local order — only
        unambiguous for a single-partner reference; a multi-partner reference
        without ``object_links_name`` raises in :meth:`build`.
    """

    hand_entity_name: str = "robot"
    object_entity_name: str = "object"
    hand_links_name: list[str] = []
    object_links_name: list[str] = []

    def build(self) -> None:
        env = self._env
        hand: RigidEntity = env.get_entity(self.hand_entity_name)
        if not self.hand_links_name:
            raise ValueError(f"{type(self).__name__}: hand_links_name is required (defines the link pairing).")
        self._robot_links_idx = MotionContactChamferReward._resolve_links(hand, self.hand_links_name, "hand_links_name")
        self._motion_links_idx = get_motion_link_idx(env, self.hand_entity_name, self.hand_links_name)
        assert_motion_field_present(env, self.hand_entity_name, "contact_points_valid")

        # Pair the demo partner columns with object links BY NAME. ``object_links_name``
        # declares, in order, which object link each demo partner column corresponds to,
        # and reorders the policy contact axis to match (via ``links_b_idx_local``) so the
        # correspondence never relies on Genesis's internal link order — which a dataset
        # author can't see and a count check can't verify. When omitted, the policy side
        # falls back to all object links in local order; that is only unambiguous for a
        # single partner, so a multi-partner reference without ``object_links_name`` is a
        # config error rather than a silent misalignment.
        obj: RigidEntity = env.get_entity(self.object_entity_name)
        num_demo_partners = env.motion_manager.data[self.hand_entity_name].contact_points_valid.shape[-1]
        if self.object_links_name:
            self._object_links_idx = MotionContactChamferReward._resolve_links(
                obj, self.object_links_name, "object_links_name"
            )
            num_object_partners = len(self.object_links_name)
            partner_source = f"object_links_name ({num_object_partners})"
        else:
            if num_demo_partners > 1:
                raise ValueError(
                    f"{type(self).__name__}: the demo contact reference has {num_demo_partners} partner "
                    f"columns, so 'object_links_name' is required to pin which object link each column "
                    f"pairs with (positional pairing against Genesis local link order is order-sensitive "
                    f"and cannot be verified). Object links: {[link.name for link in obj.links]}."
                )
            self._object_links_idx = None
            num_object_partners = obj.num_links
            partner_source = f"'{self.object_entity_name}'.num_links ({num_object_partners})"
        if num_object_partners != num_demo_partners:
            raise ValueError(
                f"{type(self).__name__}: {partner_source} does not match the demo contact reference's "
                f"{num_demo_partners} partner column(s); per-link pairing requires them to match."
            )

    def _policy_contacts(self) -> dict[str, torch.Tensor]:
        hand: RigidEntity = self.resolve_entity(self.hand_entity_name)
        obj: RigidEntity = self.resolve_entity(self.object_entity_name)
        return hand.get_grouped_contacts(
            obj, links_a_idx_local=self._robot_links_idx, links_b_idx_local=self._object_links_idx
        )

    def _demo_valid(self) -> torch.Tensor:
        """Demo contact flags at the current frame, ``(N, L, P)`` bool."""
        ref = self._env.motion_manager.data[self.hand_entity_name]
        return ref.contact_points_valid[:, 0].index_select(1, self._motion_links_idx)

    def _frame_valid(self) -> torch.Tensor:
        """Current-frame clip validity, ``(N,)`` float — zero past clip end."""
        return self._env.motion_manager.data[self.hand_entity_name].validity[:, 0].to(torch.float32)


@REWARD_TERM_REGISTRY.register()
class MotionContactMatchReward(_MotionContactTermBase):
    """Fraction of links whose contact state matches the demo's contact-ON links.

    ConTrack's contact-event overlap reward: per tracked link, credit 1 iff
    some partner column has BOTH the demo annotation and simulated contact
    (same-partner matching — ConTrack's per-object ``(ref & sim).any``);
    mean over the tracked links. Links the demo marks contact-free contribute
    0 either way — the term rewards making annotated contacts, not avoiding
    spurious ones (ConTrack ``contact_reward``).
    """

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        # Per-partner AND before the any-reduction: demo contact with partner A
        # plus sim contact with partner B earns nothing. The partner axes are
        # aligned by build() (name-anchored via object_links_name); for a single
        # partner this is identical to AND-ing the two any-flags.
        match = (self._policy_contacts()["valid"] & self._demo_valid()).any(dim=-1)  # (N, L)
        return match.to(torch.float32).mean(dim=-1) * self._frame_valid()


@REWARD_TERM_REGISTRY.register()
class MotionContactDistanceReward(_MotionContactTermBase):
    """Exp-kernel reward on per-link contact-point distance in the object frame.

    ConTrack's contact-point distance reward (``contact_distance_reward``):
    for each link in joint contact (demo AND sim), the simulated contact point
    is expressed in the CURRENT object frame and the demo point in the DEMO
    object frame (dual anchor — the comparison measures where on the object
    the link touches, decoupled from object tracking error), then scored with
    ``exp(-d² / std²)`` and averaged over jointly-contacting links. Returns 0
    when no link is jointly in contact.
    """

    std: float = 0.03

    def build(self) -> None:
        super().build()
        env = self._env
        assert_motion_field_present(env, self.hand_entity_name, "contact_points_w")
        assert_motion_field_present(env, self.object_entity_name, "base_pos_w")
        assert_motion_field_present(env, self.object_entity_name, "base_quat_w")

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        env = self._env
        obj: RigidEntity = env.get_entity(self.object_entity_name)
        policy = self._policy_contacts()  # position (N, L, n_b, 3), valid (N, L, n_b)
        hand_ref = env.motion_manager.data[self.hand_entity_name]
        obj_ref = env.motion_manager.data[self.object_entity_name]

        demo_pos_w = hand_ref.contact_points_w[:, 0].index_select(1, self._motion_links_idx)  # (N, L, P, 3)
        demo_valid = self._demo_valid()  # (N, L, P)

        sim_local = inv_transform_by_trans_quat(
            policy["position"],
            obj.get_pos().unsqueeze(1).unsqueeze(1),
            obj.get_quat().unsqueeze(1).unsqueeze(1),
        )
        demo_local = inv_transform_by_trans_quat(
            demo_pos_w,
            obj_ref.base_pos_w[:, 0].unsqueeze(1).unsqueeze(1),
            obj_ref.base_quat_w[:, 0].unsqueeze(1).unsqueeze(1),
        )

        # Partner axes are aligned in build(): the policy contact axis is ordered
        # to object_links_name (or all object links when single-partner), matching
        # the demo column order, so P == n_b and column i pairs with column i.
        joint = (demo_valid & policy["valid"]).to(torch.float32)  # (N, L, P)
        d_sq = torch.sum(torch.square(sim_local - demo_local), dim=-1)  # (N, L, P)
        # Per link: mean d² over jointly-contacting partners, exp kernel, then
        # average over links with at least one joint contact (0 if none).
        cnt = joint.sum(dim=-1)  # (N, L)
        link_has = (cnt > 0).to(torch.float32)
        mean_d_sq = (d_sq * joint).sum(dim=-1) / cnt.clamp(min=1.0)
        per_link = torch.exp(-mean_d_sq / (self.std**2)) * link_has
        reward = per_link.sum(dim=-1) / link_has.sum(dim=-1).clamp(min=1.0)
        return reward * self._frame_valid()
