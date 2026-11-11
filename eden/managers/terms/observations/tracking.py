"""Observation terms exposing motion-reference state to the policy.

Reads ``env.motion_manager.data[entity_name]`` (filled per step at the start
of ``MimicRLEnvBase.step()`` and refilled per env on reset) — NOT
``reference_frame``, which would force a recompute pass and a GPU sync.

Current vs future frames
------------------------

The motion-reference buffer is shaped ``(num_envs, num_frames, ...)`` where
``num_frames`` comes from :attr:`MotionManagerOptions.num_frames`. Terms in
this module split into two families:

* **Current-frame terms** (``motion_base_*_w``, ``motion_dofs_*``,
  ``motion_links_*_b``) index ``[:, 0, ...]``. With the default
  ``data_start_from="one_frame_interval"`` frame 0 is one tick ahead;
  with ``data_start_from="current_time"`` frame 0 is the current step.
* **Future-frame terms** (``motion_future_*``) index a caller-supplied
  ``frame_offsets`` tuple into the same buffer. They are intended for
  policies that consume a short lookahead (e.g. ProtoMotions'
  ``(1, 2, 4, 8)``-step target poses on top of the current frame). The
  caller must size ``num_frames`` to cover ``max(frame_offsets) + 1``;
  the term raises a clear error on the first call otherwise.

Shape convention
----------------

* Single-target terms (``motion_base_*_w``, ``motion_dofs_*``) return
  ``(num_envs, F)``.
* Multi-link terms (``motion_links_*``) return ``(num_envs, K*F)`` by
  default — flat, so they slot into rank-2 obs concat groups (the common
  case for MLP policies) without the caller having to pass ``flatten=True``
  every time. Pass ``flatten=False`` to keep the link axis as a separate
  ``(num_envs, K, F)`` dimension for structured policies (transformer /
  per-link MLP).
* Future-frame terms add a leading ``K = len(frame_offsets)`` axis:
  ``(num_envs, K*D)`` flat / ``(num_envs, K, D)`` unflattened for DOF
  terms; ``(num_envs, K*L*F)`` / ``(num_envs, K, L, F)`` for link terms.

Each term calls :func:`assert_motion_field_present` for the field(s) it
actually reads, so a dataset that doesn't supply the required field
(e.g. a fixed-base motion sequence with no ``base_pos_w``, or a hand
sequence with no ``links_quat_w``) fails fast on first read instead of
silently feeding the ``MotionReferenceData.make_empty`` default
(zeros / identity quat) to the policy. Position-only and orientation-only
terms intentionally assert *only* on the field they consume — a dataset
with only positions can still feed ``motion_links_pos_w`` /
``motion_links_pos_b``.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from eden.utils.geom import quat_to_rot6d_cols, quat_to_tan_norm, subtract_frame_transforms
from eden.utils.motion_reference.lookup import (
    assert_motion_field_present,
    current_motion_frame,
    get_motion_dofs_idx,
    get_motion_link_idx,
    get_robot_link_idx,
    resolve_frame_offsets,
    select_motion_frames,
)
from eden.utils.motion_reference.transforms import links_pos_in_anchor_frame, links_quat_in_anchor_frame

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


def _maybe_flatten(x: torch.Tensor, flatten: bool) -> torch.Tensor:
    return x.reshape(x.shape[0], -1) if flatten else x


def _orientation_6d(quat: torch.Tensor, representation: Literal["rot6d", "tan_norm"]) -> torch.Tensor:
    if representation == "rot6d":
        return quat_to_rot6d_cols(quat)  # [R·x̂, R·ŷ] interleaved
    if representation == "tan_norm":
        return quat_to_tan_norm(quat)  # [R·x̂, R·ẑ] concat
    raise ValueError(f"Unknown representation {representation!r}. Use 'rot6d' or 'tan_norm'.")


# Frame-offset validation + on-device caching lives in
# ``eden.utils.motion_reference.lookup.resolve_frame_offsets``. The windowed
# access helpers in that module are the single shared path the obs / reward /
# termination tracking terms run through: ``select_motion_frames`` (current +
# future) and ``current_motion_frame`` (the K=1 current-frame case).


@OBSERVATION_TERM_REGISTRY.register()
def motion_base_pos_w(env: MimicRLEnvBase, *, entity_name: str) -> torch.Tensor:
    """World-frame motion-ref base position. Shape ``(num_envs, 3)``.

    Free-base entities only — fixed-base motion sequences leave
    ``base_pos_w`` at the make-empty default and this term raises.
    """
    assert_motion_field_present(env, entity_name, "base_pos_w")
    ref = env.motion_manager.data[entity_name]
    return current_motion_frame(ref.base_pos_w)


@OBSERVATION_TERM_REGISTRY.register()
def motion_base_quat_w(env: MimicRLEnvBase, *, entity_name: str) -> torch.Tensor:
    """World-frame motion-ref base quaternion (wxyz). Shape ``(num_envs, 4)``.

    Quat is unit-norm — guaranteed by ``MotionSequence.__post_init__``.
    """
    assert_motion_field_present(env, entity_name, "base_quat_w")
    ref = env.motion_manager.data[entity_name]
    return current_motion_frame(ref.base_quat_w)


@OBSERVATION_TERM_REGISTRY.register()
def motion_links_pos_w(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    flatten: bool = True,
) -> torch.Tensor:
    """Return world-frame motion-ref link positions.

    Shape ``(num_envs, K*3)`` (or ``(num_envs, K, 3)`` if ``flatten=False``).

    Single-link (``links_name=["wrist"]``) gives e.g. dexmachina ref wrist
    position; multi-link gives ref keypoints (fingertips, etc.).
    """
    assert_motion_field_present(env, entity_name, "links_pos_w")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    ref = env.motion_manager.data[entity_name]
    return _maybe_flatten(current_motion_frame(ref.links_pos_w, motion_idx), flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_links_quat_w(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    flatten: bool = True,
) -> torch.Tensor:
    """Return world-frame motion-ref link quaternions (wxyz).

    Shape ``(num_envs, K*4)`` (or ``(num_envs, K, 4)`` if ``flatten=False``).

    Quats are unit-norm — guaranteed by ``MotionSequence.__post_init__``.
    """
    assert_motion_field_present(env, entity_name, "links_quat_w")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    ref = env.motion_manager.data[entity_name]
    return _maybe_flatten(current_motion_frame(ref.links_quat_w, motion_idx), flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_dofs_pos(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    dofs_name: list[str] | None = None,
) -> torch.Tensor:
    """Motion-ref qpos. Shape ``(num_envs, D)``.

    ``dofs_name=None`` returns every motion-ref DOF for the entity.
    Otherwise returns just the named DOFs (motion-ref-axis order).
    Sibling of :class:`eden.managers.terms.observations.proprio.DofsPos`,
    which reads the robot's own state.
    """
    assert_motion_field_present(env, entity_name, "dofs_pos")
    ref = env.motion_manager.data[entity_name]
    sel = None if dofs_name is None else get_motion_dofs_idx(env, entity_name, dofs_name)
    return current_motion_frame(ref.dofs_pos, sel)


@OBSERVATION_TERM_REGISTRY.register()
def motion_dofs_vel(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    dofs_name: list[str] | None = None,
) -> torch.Tensor:
    """Motion-ref qvel. Symmetric to :func:`motion_dofs_pos`.

    Sibling of :class:`eden.managers.terms.observations.proprio.DofsVel`.
    """
    assert_motion_field_present(env, entity_name, "dofs_vel")
    ref = env.motion_manager.data[entity_name]
    sel = None if dofs_name is None else get_motion_dofs_idx(env, entity_name, dofs_name)
    return current_motion_frame(ref.dofs_vel, sel)


@OBSERVATION_TERM_REGISTRY.register()
def motion_links_pos_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    anchor_link_name: str,
    flatten: bool = True,
) -> torch.Tensor:
    """Return motion-ref ``links_name`` positions in the robot's ``anchor_link_name`` frame.

    Shape ``(num_envs, K*3)`` (or ``(num_envs, K, 3)`` if ``flatten=False``).

    Position-only — reads ``links_pos_w`` (motion side) plus the robot's
    anchor link pose. Does **not** require ``links_quat_w`` from the
    motion sequence (a dataset with only positions still works here).

    To reproduce beyondmimic's ``motion_anchor_pos_b``,
    pass ``links_name=[anchor_link_name]`` (e.g.
    ``links_name=["torso_link"], anchor_link_name="torso_link"``) — that
    reproduces the original anchor-vs-anchor numerics and the
    ``(num_envs, 3)`` flat shape.
    """
    assert_motion_field_present(env, entity_name, "links_pos_w")

    anchor_robot_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])
    motion_idx = get_motion_link_idx(env, entity_name, links_name)

    entity = env.get_entity(entity_name)
    anchor_pos = entity.get_links_pos(ls_idx_local=anchor_robot_idx)[:, 0]  # (N, 3)
    anchor_quat = entity.get_links_quat(ls_idx_local=anchor_robot_idx)[:, 0]  # (N, 4)

    ref = env.motion_manager.data[entity_name]
    motion_pos = current_motion_frame(ref.links_pos_w, motion_idx)  # (N, L, 3)

    pos_b = links_pos_in_anchor_frame(anchor_pos, anchor_quat, motion_pos)  # (N, L, 3)
    return _maybe_flatten(pos_b, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_links_ori_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    anchor_link_name: str,
    flatten: bool = True,
) -> torch.Tensor:
    """Return motion-ref ``links_name`` orientations in the robot's ``anchor_link_name`` frame.

    Expressed as 2-column rotation matrices, shape ``(num_envs, K*6)``
    (or ``(num_envs, K, 6)`` if ``flatten=False``).

    Orientation-only — reads ``links_quat_w`` (motion side) plus the
    robot's anchor link orientation. Does **not** require ``links_pos_w``.

    To reproduce beyondmimic's ``motion_anchor_ori_b``, pass
    ``links_name=[anchor_link_name]``. The 2-column rotation matrix
    output (not quaternion) is beyondmimic's representation, so its
    pretrained policies work unchanged; configs that prefer quat can read
    :func:`motion_links_quat_w` instead.
    """
    assert_motion_field_present(env, entity_name, "links_quat_w")

    anchor_robot_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])
    motion_idx = get_motion_link_idx(env, entity_name, links_name)

    entity = env.get_entity(entity_name)
    anchor_quat = entity.get_links_quat(ls_idx_local=anchor_robot_idx)[:, 0]  # (N, 4)

    ref = env.motion_manager.data[entity_name]
    motion_quat = current_motion_frame(ref.links_quat_w, motion_idx)  # (N, L, 4)

    quat_b = links_quat_in_anchor_frame(anchor_quat, motion_quat)  # (N, L, 4)
    ori_b = quat_to_rot6d_cols(quat_b)  # (N, L, 6)
    return _maybe_flatten(ori_b, flatten)


def _robot_body_relative(
    env: MimicRLEnvBase,
    entity_name: str,
    anchor_link_name: str,
    tracked_link_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos_b, quat_b)`` of robot bodies relative to robot anchor.

    Once per :meth:`~eden.envs.base.EnvBase.compute_once` pass: :func:`robot_body_pos_b` and
    :func:`robot_body_ori_b` each read one half of it.
    """

    def producer() -> tuple[torch.Tensor, torch.Tensor]:
        anchor_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])
        body_idx = get_robot_link_idx(env, entity_name, tracked_link_names)

        entity = env.get_entity(entity_name)
        anchor_pos = entity.get_links_pos(ls_idx_local=anchor_idx)[:, 0]
        anchor_quat = entity.get_links_quat(ls_idx_local=anchor_idx)[:, 0]
        body_pos = entity.get_links_pos(ls_idx_local=body_idx)
        body_quat = entity.get_links_quat(ls_idx_local=body_idx)

        num_bodies = len(tracked_link_names)
        return subtract_frame_transforms(
            anchor_pos[:, None, :].expand(-1, num_bodies, -1),
            anchor_quat[:, None, :].expand(-1, num_bodies, -1),
            body_pos,
            body_quat,
        )

    return env.compute_once(("robot_body_relative", entity_name, anchor_link_name, tuple(tracked_link_names)), producer)


@OBSERVATION_TERM_REGISTRY.register()
def robot_body_pos_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    anchor_link_name: str,
    tracked_link_names: list[str],
    flatten: bool = True,
) -> torch.Tensor:
    """Return robot body positions relative to its own anchor link.

    Shape ``(N, K*3)`` (or ``(N, K, 3)`` if ``flatten=False``).

    Symmetric to :func:`motion_links_pos_b` but reads the robot's own state
    instead of the motion reference. Useful as a critic-only observation that
    sees the policy's body geometry in the same frame as the motion targets.
    """
    pos_b, _ = _robot_body_relative(env, entity_name, anchor_link_name, tracked_link_names)
    return _maybe_flatten(pos_b, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def robot_body_ori_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    anchor_link_name: str,
    tracked_link_names: list[str],
    flatten: bool = True,
) -> torch.Tensor:
    """Return robot body orientations relative to its own anchor, as 2-column rotation matrices.

    Shape ``(N, K*6)`` (or ``(N, K, 6)`` if ``flatten=False``).

    Symmetric to :func:`motion_links_ori_b`.
    """
    _, ori_b = _robot_body_relative(env, entity_name, anchor_link_name, tracked_link_names)
    ori_6d = quat_to_rot6d_cols(ori_b)
    return _maybe_flatten(ori_6d, flatten)


# ---------------------------------------------------------------------------
# Future-frame motion-ref lookahead terms
# ---------------------------------------------------------------------------
#
# These mirror the current-frame ``motion_dofs_*`` / ``motion_links_ori_b``
# terms but index a caller-supplied ``frame_offsets`` tuple into the
# ``(num_envs, num_frames, ...)`` buffer instead of ``[:, 0]``. They cover
# the ProtoMotions "target-poses" obs pattern (K future motion frames at
# fixed step offsets, fed alongside the current proprio).
#
# The motion buffer width is controlled by
# :attr:`MotionManagerOptions.num_frames` and the per-frame step size by
# :attr:`MotionManagerOptions.frame_interval_s_range`. The caller is
# responsible for sizing ``num_frames >= max(frame_offsets) + 1``;
# :func:`resolve_frame_offsets` validates this on first call.


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_dofs_pos(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    frame_offsets: Sequence[int],
    dofs_name: list[str] | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """Return motion-ref qpos at multiple future buffer frames.

    Shape ``(N, K*D)`` (or ``(N, K, D)`` if ``flatten=False``).

    ``frame_offsets`` are interpreted as **buffer indices** into
    ``env.motion_manager.data[entity_name].dofs_pos`` (shape ``(N, F, D)``).
    With ``data_start_from="current_time"`` offset 0 is the current step
    and ``(1, 2, 4, 8)`` are 1/2/4/8 buffer ticks ahead; with the default
    ``data_start_from="one_frame_interval"`` offset 0 is already one tick
    ahead. The per-tick wall-clock duration is
    :attr:`MotionManagerOptions.frame_interval_s_range`.

    ``dofs_name=None`` returns every motion-ref DOF for the entity.
    """
    assert_motion_field_present(env, entity_name, "dofs_pos")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "dofs_pos", frame_offsets, ref.dofs_pos.shape[1])
    sel = None if dofs_name is None else get_motion_dofs_idx(env, entity_name, dofs_name)
    future = select_motion_frames(ref.dofs_pos, offsets, sel)  # (N, K, D)
    return _maybe_flatten(future, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_dofs_vel(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    frame_offsets: Sequence[int],
    dofs_name: list[str] | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """Return motion-ref qvel at multiple future buffer frames.

    Symmetric to :func:`motion_future_dofs_pos`.
    """
    assert_motion_field_present(env, entity_name, "dofs_vel")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "dofs_vel", frame_offsets, ref.dofs_vel.shape[1])
    sel = None if dofs_name is None else get_motion_dofs_idx(env, entity_name, dofs_name)
    future = select_motion_frames(ref.dofs_vel, offsets, sel)
    return _maybe_flatten(future, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_links_ori_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    anchor_link_name: str,
    frame_offsets: Sequence[int],
    representation: Literal["rot6d", "tan_norm"] = "rot6d",
    flatten: bool = True,
) -> torch.Tensor:
    """Return future-frame motion-ref link orientations in the robot's current anchor frame.

    Shape ``(N, K*L*6)`` (or ``(N, K, L, 6)`` if ``flatten=False``).

    The relative rotation is ``R_rel = R_robot_anchor⁻¹ · R_motion_future``
    — i.e. the motion future orientation is brought into the robot's
    *current* anchor frame, matching the ProtoMotions reduced-coords target
    pose formulation. Two output representations:

    * ``representation="rot6d"``: 2-column rotation matrix
      ``[R_rel·x̂, R_rel·ŷ]`` (Zhou et al. 2019). Same convention as
      :func:`motion_links_ori_b`, so a config can swap the current-frame
      and future-frame variants without retraining the head.
    * ``representation="tan_norm"``: ProtoMotions ``quat_to_tan_norm``
      output ``[R_rel·x̂, R_rel·ẑ]``. Use this when feeding a checkpoint
      trained against ProtoMotions' target-poses obs (e.g. the
      ``g1-bones-deploy`` actor).

    Set ``links_name=[anchor_link_name]`` to reproduce the ProtoMotions
    "anchor-relative anchor future rotation" channel (single link, K
    futures).
    """
    assert_motion_field_present(env, entity_name, "links_quat_w")

    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "links_quat_w", frame_offsets, ref.links_quat_w.shape[1])
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    anchor_robot_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])

    # (N, K, L_all, 4) → (N, K, L, 4): pick K frames, then L motion links.
    motion_future_quat = select_motion_frames(ref.links_quat_w, offsets, motion_idx)

    entity = env.get_entity(entity_name)
    # Robot anchor quat is read at the current sim step (no future axis on the
    # robot side; only the motion ref looks ahead).
    anchor_quat = entity.get_links_quat(ls_idx_local=anchor_robot_idx)[:, 0]  # (N, 4)

    # Robot anchor has no future axis; broadcast it over the K motion futures so
    # (N, K) is the batch and L the link axis for the shared transform.
    K = offsets.shape[0]
    anchor_quat_k = anchor_quat[:, None, :].expand(-1, K, -1)  # (N, K, 4)
    quat_b = links_quat_in_anchor_frame(anchor_quat_k, motion_future_quat)  # (N, K, L, 4)
    return _maybe_flatten(_orientation_6d(quat_b, representation), flatten)  # (N, K, L, 6)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_links_pos_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    links_name: list[str],
    anchor_link_name: str,
    frame_offsets: Sequence[int],
    flatten: bool = True,
) -> torch.Tensor:
    """Return future-frame motion-ref ``links_name`` positions in the robot's current anchor frame.

    Shape ``(N, K*L*3)`` (or ``(N, K, L, 3)`` if ``flatten=False``).

    Position-only sibling of :func:`motion_future_links_ori_b`. The position
    delta is rotated into the robot's *current* anchor frame
    (``R_robot_anchor⁻¹ · (p_motion_future − p_robot_anchor)``), matching the
    ProtoMotions reduced-coords target-position formulation. Reads
    ``links_pos_w`` (motion side) plus the robot's anchor link pose.

    Set ``links_name=[anchor_link_name]`` to reproduce the ProtoMotions
    "anchor-relative anchor future position" channel (single link, K
    futures). The current-frame counterpart is :func:`motion_links_pos_b`.
    """
    assert_motion_field_present(env, entity_name, "links_pos_w")

    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "links_pos_w", frame_offsets, ref.links_pos_w.shape[1])
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    anchor_robot_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])

    # (N, K, L_all, 3) → (N, K, L, 3): pick K frames, then L motion links.
    motion_future_pos = select_motion_frames(ref.links_pos_w, offsets, motion_idx)

    entity = env.get_entity(entity_name)
    # Robot anchor pose is read at the current sim step (no future axis on the
    # robot side; only the motion ref looks ahead).
    anchor_pos = entity.get_links_pos(ls_idx_local=anchor_robot_idx)[:, 0]  # (N, 3)
    anchor_quat = entity.get_links_quat(ls_idx_local=anchor_robot_idx)[:, 0]  # (N, 4)

    # Robot anchor has no future axis; broadcast it over the K motion futures so
    # (N, K) is the batch and L the link axis for the shared transform.
    K = offsets.shape[0]
    anchor_pos_k = anchor_pos[:, None, :].expand(-1, K, -1)  # (N, K, 3)
    anchor_quat_k = anchor_quat[:, None, :].expand(-1, K, -1)  # (N, K, 4)
    pos_b = links_pos_in_anchor_frame(anchor_pos_k, anchor_quat_k, motion_future_pos)  # (N, K, L, 3)
    return _maybe_flatten(pos_b, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_base_pos_w(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    frame_offsets: Sequence[int],
    flatten: bool = True,
) -> torch.Tensor:
    """World-frame motion-ref base position at multiple buffer frames.

    Shape ``(N, K*3)`` (or ``(N, K, 3)`` if ``flatten=False``). The base-pose
    counterpart of :func:`motion_future_dofs_pos` — used e.g. for future
    object-pose targets in hand-object mimic tasks.
    """
    assert_motion_field_present(env, entity_name, "base_pos_w")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "base_pos_w", frame_offsets, ref.base_pos_w.shape[1])
    return _maybe_flatten(select_motion_frames(ref.base_pos_w, offsets), flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_base_quat_w(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    frame_offsets: Sequence[int],
    flatten: bool = True,
) -> torch.Tensor:
    """World-frame motion-ref base quaternion (wxyz) at multiple buffer frames.

    Shape ``(N, K*4)`` (or ``(N, K, 4)`` if ``flatten=False``).
    """
    assert_motion_field_present(env, entity_name, "base_quat_w")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "base_quat_w", frame_offsets, ref.base_quat_w.shape[1])
    return _maybe_flatten(select_motion_frames(ref.base_quat_w, offsets), flatten)


# ---------------------------------------------------------------------------
# Cross-entity terms: one entity's pose in another entity's link frame
# ---------------------------------------------------------------------------
#
# Humanoid-object tracking reads the object (its own entity, a free base with no links to anchor on) in the
# *robot's* anchor frame. The terms above take a single ``entity_name`` for both the target and the anchor, which
# cannot say that; these take the anchor separately (``anchor_entity_name`` + ``anchor_link_name``) and a free-base
# target through its base pose.


def _anchor_link_pose(env: MimicRLEnvBase, anchor_entity_name: str, anchor_link_name: str):
    """The live ``(pos, quat)`` of ``anchor_link_name`` on ``anchor_entity_name``, each ``(N, 3|4)``."""
    anchor_idx = get_robot_link_idx(env, anchor_entity_name, [anchor_link_name])
    anchor = env.get_entity(anchor_entity_name)
    return anchor.get_links_pos(ls_idx_local=anchor_idx)[:, 0], anchor.get_links_quat(ls_idx_local=anchor_idx)[:, 0]


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_base_pos_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    anchor_entity_name: str,
    anchor_link_name: str,
    frame_offsets: Sequence[int],
    flatten: bool = True,
) -> torch.Tensor:
    """Future-frame motion-ref base position of ``entity_name`` in another entity's *live* anchor-link frame.

    Shape ``(N, K*3)`` (or ``(N, K, 3)`` if ``flatten=False``). ``R_anchor⁻¹ · (p_ref_future − p_anchor)`` with
    the anchor read from the simulation at the current step — the full 6-DoF anchor frame, like
    :func:`motion_future_links_pos_b`. The object-in-humanoid-frame target of a humanoid-object tracker:
    ``entity_name="object", anchor_entity_name="robot", anchor_link_name="pelvis"``.
    """
    assert_motion_field_present(env, entity_name, "base_pos_w")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "base_pos_w", frame_offsets, ref.base_pos_w.shape[1])
    target = select_motion_frames(ref.base_pos_w, offsets)  # (N, K, 3)
    anchor_pos, anchor_quat = _anchor_link_pose(env, anchor_entity_name, anchor_link_name)
    pos_b = links_pos_in_anchor_frame(anchor_pos, anchor_quat, target)  # (N, K, 3): K plays the link axis
    return _maybe_flatten(pos_b, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_base_ori_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    anchor_entity_name: str,
    anchor_link_name: str,
    frame_offsets: Sequence[int],
    representation: Literal["rot6d", "tan_norm"] = "rot6d",
    flatten: bool = True,
) -> torch.Tensor:
    """Future-frame motion-ref base orientation of ``entity_name`` in another entity's live anchor-link frame.

    Shape ``(N, K*6)`` (or ``(N, K, 6)`` if ``flatten=False``). ``R_anchor⁻¹ · R_ref_future`` as a 6-D rotation,
    ``representation`` as in :func:`motion_future_links_ori_b`. The orientation sibling of
    :func:`motion_future_base_pos_b`.
    """
    assert_motion_field_present(env, entity_name, "base_quat_w")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "base_quat_w", frame_offsets, ref.base_quat_w.shape[1])
    target = select_motion_frames(ref.base_quat_w, offsets)  # (N, K, 4)
    _, anchor_quat = _anchor_link_pose(env, anchor_entity_name, anchor_link_name)
    ori_b = _orientation_6d(links_quat_in_anchor_frame(anchor_quat, target), representation)  # (N, K, 6)
    return _maybe_flatten(ori_b, flatten)


@OBSERVATION_TERM_REGISTRY.register()
def entity_base_pos_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    anchor_entity_name: str,
    anchor_link_name: str,
) -> torch.Tensor:
    """Live base position of ``entity_name`` in another entity's live anchor-link frame. Shape ``(N, 3)``.

    The simulated counterpart of :func:`motion_future_base_pos_b` — e.g. where the object is relative to the
    humanoid's pelvis. Reads ``get_pos()``, the entity's solver root frame: for a free rigid object that is its
    centre of mass unless the object is built with ``align=False`` (then the authored body frame, the frame a
    reference object pose is recorded in).
    """
    anchor_pos, anchor_quat = _anchor_link_pose(env, anchor_entity_name, anchor_link_name)
    target = env.get_entity(entity_name).get_pos()[:, None]  # (N, 1, 3)
    return links_pos_in_anchor_frame(anchor_pos, anchor_quat, target)[:, 0]


@OBSERVATION_TERM_REGISTRY.register()
def entity_base_ori_b(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    anchor_entity_name: str,
    anchor_link_name: str,
    representation: Literal["rot6d", "tan_norm"] = "rot6d",
) -> torch.Tensor:
    """Live base orientation of ``entity_name`` in another entity's live anchor-link frame, 6-D. Shape ``(N, 6)``.

    The orientation sibling of :func:`entity_base_pos_b`.
    """
    _, anchor_quat = _anchor_link_pose(env, anchor_entity_name, anchor_link_name)
    target = env.get_entity(entity_name).get_quat()[:, None]  # (N, 1, 4)
    return _orientation_6d(links_quat_in_anchor_frame(anchor_quat, target), representation)[:, 0]


@OBSERVATION_TERM_REGISTRY.register()
def motion_phase(env: MimicRLEnvBase, *, entity_name: str) -> torch.Tensor:
    """Normalized clip playback phase in ``[0, 1]``. Shape ``(num_envs, 1)``.

    ``frame / (clip_length - 1)`` at the current reference frame — unlike
    :func:`eden.managers.terms.observations.common.episode_phase`, a mid-clip
    RSI start reports its true position in the clip.

    :meth:`MotionManager.get_motion_phase` raises when no motion term tracks
    ``entity_name``, so a misspelled entity fails on the first compute instead of
    feeding the policy a constant-zero column.
    """
    return env.motion_manager.get_motion_phase(entity_name).unsqueeze(-1)
