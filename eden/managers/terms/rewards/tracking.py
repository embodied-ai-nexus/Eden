"""Reward terms that score robot state against ``env.motion_manager`` data.

All terms read motion-ref data through ``env.motion_manager.data[entity_name]``
(filled once per step at the start of ``MimicRLEnvBase.step()``) — NOT
``reference_frame``, which would force a recompute pass and a GPU sync.

Two families:

* **Direct world-frame tracking** (function-style, registry keys
  ``motion_{base,link,dofs}_*_tracking_exp``): scores robot state directly
  against world-frame motion-ref data. Use for free-floating objects (where
  world xy / yaw is meaningful), fixed-base entities, and DexMachina-style
  bimanual hand-object tasks.
* **Anchor-yaw-warped tracking** (class-style, registry keys
  ``motion_{relative_body,global_body}_*``): warps the motion targets through
  a yaw-only delta computed at an anchor link before scoring. Makes the
  reward invariant to the robot's heading drift on the floor — useful for
  free-floating humanoid imitation where absolute world yaw / xy is not the
  optimisation target. Requires both ``links_pos_w`` and ``links_quat_w``.

Exp-kernel parameter convention: these terms use ``std`` with the form
``exp(-error / std**2)``, matching beyondmimic's tracking rewards. Two other
conventions exist in the codebase that newcomers should be aware of:

* ``locomotion.tracking_lin_vel`` uses ``sigma`` with the form
  ``exp(-error / sigma)`` (linear, not squared).
* DexMachina papers use ``beta`` with ``exp(-beta * error)``. Mapping:
  ``std = 1 / sqrt(beta)``.

Each term calls :func:`assert_motion_field_present` at first read so a
dataset that doesn't supply the required field (e.g. a fixed-base motion
sequence with no ``base_pos_w``, or an Arctic hand sequence with no
``links_quat_w``) fails fast instead of silently scoring against the
make-empty default (zeros / identity quat).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.utils.geom import quat_error_magnitude
from eden.utils.motion_reference.lookup import (
    assert_motion_field_present,
    anchor_relative_links_once,
    current_motion_frame,
    get_motion_dofs_idx,
    get_motion_link_idx,
    get_robot_dofs_idx,
    get_robot_link_idx,
    resolve_frame_offsets,
    select_motion_frames,
)

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


# ---------------------------------------------------------------------------
# Multi-future-frame helpers
# ---------------------------------------------------------------------------
#
# The link, DOF, anchor-relative and link-velocity tracking rewards accept a ``frame_offsets`` kwarg that
# indexes the motion buffer along the time axis. The default ``(0,)`` scores only the
# current frame. Multi-frame (``frame_offsets=(0, k, m)``) averages the per-frame reward
# over the K axis, weighted by ``MotionReferenceData.validity`` so frames past the clip end
# contribute zero rather than a fake clamped-frame value (the buffer is filled with the last
# valid frame when ``validity[:, k]=False``).
#
# Pairs with the obs-side ``motion_future_*`` terms in
# ``eden/managers/terms/observations/tracking.py``: matching ``frame_offsets``
# on obs and reward keeps the gradient pointed at the new lookahead dims.
# Reading the same buffer slice on both sides also avoids any divergence
# between what the policy sees and what is being scored.


def _validity_mask_for(
    ref,
    offsets: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``validity[:, offsets]`` cast to ``dtype`` for elementwise masking.

    ``validity`` is always allocated by ``MotionReferenceData.make_empty``,
    so this is unconditional in production. Shape ``(N, K)``.
    """
    # validity is (N, F) — no link/dof axis, so just the frame window (sel_idx=None).
    return select_motion_frames(ref.validity, offsets).to(dtype)


def exp_kernel(
    error: torch.Tensor,
    std: float,
    *,
    mode: Literal["mean_of_exp", "exp_of_mean", "exp_of_sum"] = "exp_of_mean",
) -> torch.Tensor:
    """Gaussian tracking kernel reducing the last (per-item) axis of ``error``.

    ``error`` is the squared per-item error ``(..., M)`` where ``M`` is the link
    or DOF axis; returns ``(...)`` with that axis collapsed. ``mode`` names the
    one valid way the exp and the per-item reduction compose — each value is the
    literal formula, so there is no silently-ignored argument:

    * ``"mean_of_exp"`` (DexMachina ``exp_kpt_first``): per-item ``exp(-e/std²)``
      then mean over items — soft per-item credit.
    * ``"exp_of_mean"`` (default): one ``exp`` on the mean per-item error — a
      single soft pass on the average.
    * ``"exp_of_sum"`` (DexMachina ``obj_arti_rew``): one ``exp`` on the summed
      per-item error — a single hard pass/fail.

    Mirrors the ``exp(-error / std²)`` convention documented at module top.
    """
    if mode == "mean_of_exp":
        return torch.exp(-error / std**2).mean(-1)
    if mode == "exp_of_mean":
        return torch.exp(-error.mean(-1) / std**2)
    if mode == "exp_of_sum":
        return torch.exp(-error.sum(-1) / std**2)
    raise ValueError(f"Unknown exp_kernel mode {mode!r}. Use 'mean_of_exp', 'exp_of_mean', or 'exp_of_sum'.")


def validity_weighted_frame_mean(
    per_frame: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Average a per-frame reward ``(N, K)`` over the K future frames.

    Each frame is weighted by its ``valid`` mask ``(N, K)`` — 1.0 inside the
    clip, 0.0 for frames past the clip end (the buffer holds the last valid
    frame there, so an unmasked mean would score against a fake target). The
    denominator is clamped at 1 so an all-invalid row returns 0 rather than
    NaN. ``valid=None`` (no validity available) falls back to a plain mean over
    K. Returns ``(N,)``.
    """
    if valid is None:
        return per_frame.mean(-1)
    return (per_frame * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)


@REWARD_TERM_REGISTRY.register()
def motion_base_position_tracking_exp(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    std: float,
) -> torch.Tensor:
    """Exp-kernel reward on world-frame base position vs motion ref.

    DexMachina equivalent: ``obj_pos_rew``.
    """
    assert_motion_field_present(env, entity_name, "base_pos_w")
    entity = env.get_entity(entity_name)
    ref = env.motion_manager.data[entity_name]
    error = torch.sum(torch.square(entity.get_pos() - current_motion_frame(ref.base_pos_w)), dim=-1)
    return torch.exp(-error / std**2)


@REWARD_TERM_REGISTRY.register()
def motion_base_orientation_tracking_exp(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    std: float,
) -> torch.Tensor:
    """Exp-kernel reward on quat-error magnitude of the base vs motion ref (wxyz).

    DexMachina equivalent: ``obj_rot_rew``.
    """
    assert_motion_field_present(env, entity_name, "base_quat_w")
    entity = env.get_entity(entity_name)
    ref = env.motion_manager.data[entity_name]
    error = quat_error_magnitude(entity.get_quat(), current_motion_frame(ref.base_quat_w)) ** 2
    return torch.exp(-error / std**2)


@REWARD_TERM_REGISTRY.register()
def motion_link_position_tracking_exp(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    std: float,
    exp_first: bool = True,
    frame_offsets: Sequence[int] = (0,),
) -> torch.Tensor:
    """Exp-kernel reward on world-frame link positions vs motion ref.

    Single-link (``links_name=["wrist"]``) gives DexMachina wrist-pos /
    object-pos tracking; multi-link gives DexMachina fingertip imitation.

    ``exp_first=True`` (default, matches DexMachina ``exp_kpt_first=True``):
    mean over links of ``exp(-d² / std²)``.
    ``exp_first=False``: ``exp(-mean(d²) / std²)`` — one soft pass on the
    average error.

    ``frame_offsets=(0,)`` (default) scores only the current motion frame.
    Multi-frame (e.g. ``(0, 4, 8)``) averages the per-frame reward over the
    listed buffer indices, masked by ``MotionReferenceData.validity`` so
    frames past the clip end contribute zero. Pair with the obs-side
    :func:`~eden.managers.terms.observations.tracking.motion_future_links_pos_b`
    lookahead set to the same offsets so the reward scores what the policy sees.
    """
    assert_motion_field_present(env, entity_name, "links_pos_w")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    robot_idx = get_robot_link_idx(env, entity_name, links_name)
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "links_pos_w", frame_offsets, ref.links_pos_w.shape[1])
    # (N, K, L_all, 3) → (N, K, L, 3) and broadcast robot_pos to match.
    motion_pos = select_motion_frames(ref.links_pos_w, offsets, motion_idx)
    robot_pos = env.get_entity(entity_name).get_links_pos(ls_idx_local=robot_idx)  # (N, L, 3)
    sq = torch.sum(torch.square(motion_pos - robot_pos[:, None]), dim=-1)  # (N, K, L)
    valid = _validity_mask_for(ref, offsets, dtype=sq.dtype)  # (N, K)
    per_frame = exp_kernel(sq, std, mode="mean_of_exp" if exp_first else "exp_of_mean")  # (N, K)
    return validity_weighted_frame_mean(per_frame, valid)


@REWARD_TERM_REGISTRY.register()
def motion_link_orientation_tracking_exp(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    std: float,
    exp_first: bool = True,
    frame_offsets: Sequence[int] = (0,),
) -> torch.Tensor:
    """Exp-kernel reward on world-frame link orientations vs motion ref (wxyz).

    Single-link gives DexMachina wrist-rot / object-rot tracking; multi-link
    reduces the same way as the position variant.

    ``frame_offsets`` matches :func:`motion_link_position_tracking_exp` —
    default ``(0,)`` is the current frame; multi-frame masks past-clip
    frames via ``validity``.
    """
    assert_motion_field_present(env, entity_name, "links_quat_w")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    robot_idx = get_robot_link_idx(env, entity_name, links_name)
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "links_quat_w", frame_offsets, ref.links_quat_w.shape[1])
    # (N, K, L_all, 4) → (N, K, L, 4); robot_quat is broadcast.
    motion_quat = select_motion_frames(ref.links_quat_w, offsets, motion_idx)
    robot_quat = env.get_entity(entity_name).get_links_quat(ls_idx_local=robot_idx)  # (N, L, 4)
    K = offsets.shape[0]
    err = quat_error_magnitude(motion_quat, robot_quat[:, None].expand(-1, K, -1, -1)) ** 2  # (N, K, L)
    valid = _validity_mask_for(ref, offsets, dtype=err.dtype)  # (N, K)
    per_frame = exp_kernel(err, std, mode="mean_of_exp" if exp_first else "exp_of_mean")  # (N, K)
    return validity_weighted_frame_mean(per_frame, valid)


@REWARD_TERM_REGISTRY.register()
def motion_dofs_pos_tracking_exp(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    std: float,
    dofs_name: list[str] | None = None,
    reduction: Literal["sum", "per_dof_mean", "mean"] = "sum",
    frame_offsets: Sequence[int] = (0,),
) -> torch.Tensor:
    """Exp-kernel reward on per-DOF qpos vs motion ref.

    ``reduction="sum"`` (default): ``exp(-Σ(q − q_ref)² / std²)``. Single
    hard pass/fail; matches DexMachina ``obj_arti_rew``.
    ``reduction="per_dof_mean"``: mean over dofs of
    ``exp(-(q − q_ref)² / std²)``. Each DOF gets independent credit;
    matches DexMachina ``bc_rew`` for hand qpos behavior cloning.
    ``reduction="mean"``: ``exp(-mean((q − q_ref)²) / std²)`` — one soft
    pass on the MSE; matches ConTrack arm/finger joint tracking.

    ``dofs_name=None`` tracks every DOF the motion ref exposes for the
    entity.

    ``frame_offsets`` defaults to ``(0,)`` (current frame only). Multi-frame
    averages the per-frame ``exp(...)`` weighted by validity.
    """
    assert_motion_field_present(env, entity_name, "dofs_pos")
    entity = env.get_entity(entity_name)
    ref = env.motion_manager.data[entity_name]
    if dofs_name is None:
        dofs_name = env.motion_manager.get_target_dofs_name(entity_name)
        if dofs_name is None:
            raise ValueError(f"Entity '{entity_name}' is not tracked by any motion term.")
        if not dofs_name:
            raise ValueError(
                f"motion_dofs_pos_tracking_exp: entity '{entity_name}' has no DOFs in the "
                f"motion reference. Use motion_link_position_tracking_exp / "
                f"motion_base_position_tracking_exp for rigid or link-only entities, or "
                f"pass an explicit dofs_name list."
            )
    motion_idx = get_motion_dofs_idx(env, entity_name, dofs_name)
    robot_idx = get_robot_dofs_idx(env, entity_name, dofs_name)
    offsets = resolve_frame_offsets(env, "dofs_pos", frame_offsets, ref.dofs_pos.shape[1])
    # (N, K, D_all) → (N, K, D); robot_q is the current state, broadcast over K.
    motion_q = select_motion_frames(ref.dofs_pos, offsets, motion_idx)
    # Handed to the getter as `dofs_idx_local`, not used to gather from a full-width read:
    # `get_dofs_pos()` returns the entity's DOFs in *dofs_name* order, so `index_select`-ing it
    # with `robot_idx` (Genesis-local, tree-numbered, offset by a floating base's six) is out of
    # bounds on a floating base and a silent permutation on a fixed one whose declaration order
    # differs from the tree's. The getter takes local indices and returns them in the order given.
    robot_q = entity.get_dofs_pos(dofs_idx_local=robot_idx)  # (N, D)
    sq = torch.square(robot_q[:, None] - motion_q)  # (N, K, D)
    valid = _validity_mask_for(ref, offsets, dtype=sq.dtype)  # (N, K)
    kernel_mode = {"per_dof_mean": "mean_of_exp", "mean": "exp_of_mean", "sum": "exp_of_sum"}[reduction]
    per_frame = exp_kernel(sq, std, mode=kernel_mode)  # (N, K)
    return validity_weighted_frame_mean(per_frame, valid)


# ---------------------------------------------------------------------------
# Anchor-yaw-warped (heading-invariant) family
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
class MotionRelativeBodyPositionErrorExp(RewardTerm):
    """Yaw-warped multi-link position tracking (exp kernel, link mean).

    The yaw-warp uses both anchor positions and quaternions (via
    :func:`get_anchor_state` / :func:`get_links_state`), so the dataset must
    populate both ``links_pos_w`` and ``links_quat_w``. Use the direct
    :func:`motion_link_position_tracking_exp` for datasets that only have
    positions.

    ``frame_offsets`` defaults to ``(0,)`` (current frame only). Multi-frame
    averages per-future-frame ``exp(...)`` masked by validity so past-clip
    frames contribute zero — pair with the obs-side
    :func:`~eden.managers.terms.observations.tracking.motion_future_links_pos_b` set to the same offsets.
    """

    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"
    links_name: list[str] = []
    std: float = 0.1
    frame_offsets: tuple[int, ...] = (0,)

    def build(self) -> None:
        assert_motion_field_present(self._env, self.entity_name, "links_pos_w")
        assert_motion_field_present(self._env, self.entity_name, "links_quat_w")

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        # Shared with the other half's reward and the matching metrics within a step's graded pass.
        links_pos_rel, _, robot_links_pos, _, validity = anchor_relative_links_once(
            self._env, self.entity_name, self.anchor_link_name, self.links_name, self.frame_offsets
        )
        # links_pos_rel: (N, K, L, 3), robot_links_pos: (N, L, 3) → broadcast over K.
        error = torch.sum(torch.square(links_pos_rel - robot_links_pos[:, None]), dim=-1)  # (N, K, L)
        per_frame = exp_kernel(error, self.std)  # (N, K)
        valid = None if validity is None else validity.to(per_frame.dtype)
        return validity_weighted_frame_mean(per_frame, valid)


@REWARD_TERM_REGISTRY.register()
class MotionRelativeBodyOrientationErrorExp(RewardTerm):
    """Yaw-warped multi-link orientation tracking (exp kernel, link mean).

    Reads both ``links_pos_w`` and ``links_quat_w`` for the same reason as
    :class:`MotionRelativeBodyPositionErrorExp`.

    ``frame_offsets`` matches :class:`MotionRelativeBodyPositionErrorExp`.
    """

    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"
    links_name: list[str] = []
    std: float = 0.1
    frame_offsets: tuple[int, ...] = (0,)

    def build(self) -> None:
        assert_motion_field_present(self._env, self.entity_name, "links_pos_w")
        assert_motion_field_present(self._env, self.entity_name, "links_quat_w")

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        # Shared with the other half's reward and the matching metrics within a step's graded pass.
        _, links_quat_rel, _, robot_links_quat, validity = anchor_relative_links_once(
            self._env, self.entity_name, self.anchor_link_name, self.links_name, self.frame_offsets
        )
        K = links_quat_rel.shape[1]
        error = quat_error_magnitude(links_quat_rel, robot_links_quat[:, None].expand(-1, K, -1, -1)) ** 2  # (N, K, L)
        per_frame = exp_kernel(error, self.std)  # (N, K)
        valid = None if validity is None else validity.to(per_frame.dtype)
        return validity_weighted_frame_mean(per_frame, valid)


class _MotionGlobalBodyVelocityErrorExp(RewardTerm):
    """Base for linear/angular world-frame link-velocity tracking rewards.

    ``frame_offsets`` defaults to ``(0,)`` (current frame only). Multi-frame
    averages per-future-frame ``exp(...)`` masked by validity.
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    std: float = 0.1
    frame_offsets: tuple[int, ...] = (0,)

    # Subclasses set these
    _motion_vel_attr: str
    _robot_vel_method: str

    def build(self) -> None:
        assert_motion_field_present(self._env, self.entity_name, self._motion_vel_attr)
        self._motion_idx = get_motion_link_idx(self._env, self.entity_name, self.links_name)
        self._robot_idx = get_robot_link_idx(self._env, self.entity_name, self.links_name)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        ref = self._env.motion_manager.data[self.entity_name]
        motion_vel_buf = getattr(ref, self._motion_vel_attr)
        offsets = resolve_frame_offsets(self._env, self._motion_vel_attr, self.frame_offsets, motion_vel_buf.shape[1])
        # (N, K, L_all, 3) → (N, K, L, 3); robot_vel is broadcast over K.
        motion_vel = select_motion_frames(motion_vel_buf, offsets, self._motion_idx)
        robot_vel = getattr(self.resolve_entity(self.entity_name), self._robot_vel_method)(ls_idx_local=self._robot_idx)
        error = torch.sum(torch.square(motion_vel - robot_vel[:, None]), dim=-1)  # (N, K, L)
        per_frame = exp_kernel(error, self.std)  # (N, K)
        valid = _validity_mask_for(ref, offsets, dtype=per_frame.dtype)  # (N, K)
        return validity_weighted_frame_mean(per_frame, valid)


@REWARD_TERM_REGISTRY.register()
class MotionGlobalBodyLinearVelocityErrorExp(_MotionGlobalBodyVelocityErrorExp):
    _motion_vel_attr: str = "links_lin_vel_w"
    _robot_vel_method: str = "get_links_vel"


@REWARD_TERM_REGISTRY.register()
class MotionGlobalBodyAngularVelocityErrorExp(_MotionGlobalBodyVelocityErrorExp):
    _motion_vel_attr: str = "links_ang_vel_w"
    _robot_vel_method: str = "get_links_ang"
