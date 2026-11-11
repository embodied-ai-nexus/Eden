"""Index-resolution helpers for motion-reference-driven terms.

Cross-cutting utilities used by reward, termination, observation, and action
terms that read motion-manager data and need to align robot-side indices
(``find_named_*``) with motion-ref-side indices
(``MotionTerm.target_*_name``). Cached per env via a ``WeakKeyDictionary`` so
the cache evicts when the env is garbage-collected.

The ``get_*_idx`` helpers return a ``torch.Tensor`` on ``env.device`` so
callers can pass them directly to ``index_select`` without per-step
allocation; :func:`get_robot_link_idx` returns a ``range`` instead when the
links are contiguous.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Sequence

import torch

from eden.utils.misc import as_genesis_index
from eden.utils.motion_reference.transforms import anchor_relative_links

if TYPE_CHECKING:
    from eden.envs.base import EnvBase, MimicRLEnvBase


_INDEX_CACHE: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()

_FRAME_OFFSETS_CACHE: "weakref.WeakKeyDictionary[object, dict[tuple[str, tuple[int, ...]], torch.Tensor]]" = (
    weakref.WeakKeyDictionary()
)

# Kinematic base fields whose presence is meaningless for a fixed-base entity.
# Mask fields like ``base_pos_plane_mask`` are excluded — they're per-frame
# masks on ``MotionReferenceData``, not the kinematics themselves.
_FIXED_BASE_GUARDED_FIELDS = frozenset(
    {
        "base_pos_w",
        "base_quat_w",
        "base_lin_vel_w",
        "base_ang_vel_w",
    }
)


def _cache_for(env: EnvBase) -> dict:
    cache = _INDEX_CACHE.get(env)
    if cache is None:
        cache = {}
        _INDEX_CACHE[env] = cache
    return cache


def untracked_entity_error(env: MimicRLEnvBase, entity_name: str) -> ValueError:
    """The error for a motion-reference consumer pointed at an entity no motion term tracks."""
    tracked = sorted({name for term in env.motion_manager._terms.values() for name in term.entities_map})
    return ValueError(
        f"Entity '{entity_name}' is not tracked by any motion term; tracked entities: {tracked}. Add it to a "
        f"motion term's entities_map (entity name -> motion sequence key), or name one of the tracked entities."
    )


def get_robot_link_idx(env: EnvBase, entity_name: str, links_name: Sequence[str]) -> range | torch.Tensor:
    """Resolve robot-entity link indices for ``links_name`` (cached), for the entity's ``get_links_*`` getters.

    A ``range`` when the links are contiguous — always, for a single (anchor) link — else an index
    tensor (:func:`eden.utils.misc.as_genesis_index`). Genesis reads a one-element index tensor back
    with ``.item()``, a GPU->CPU sync on every call, and the anchor getters run several times a step.
    """
    if not links_name:
        raise ValueError(f"links_name must be non-empty (entity '{entity_name}').")
    cache = _cache_for(env)
    key = ("robot_link", entity_name, tuple(links_name))
    if key not in cache:
        _, indices = env.entities[entity_name].find_named_links_idx_local(list(links_name), preserve_order=True)
        cache[key] = as_genesis_index(torch.tensor(indices, dtype=torch.long, device=env.device))
    return cache[key]


def get_motion_link_idx(env: MimicRLEnvBase, entity_name: str, links_name: Sequence[str]) -> torch.Tensor:
    """Resolve motion-ref link indices for ``links_name`` (cached).

    Looks up which motion term covers ``entity_name`` and indexes the term's
    ``target_links_name`` for that entity. Raises if no motion term tracks
    the entity, or if any requested name is missing from the term's targets.
    """
    if not links_name:
        raise ValueError(f"links_name must be non-empty (entity '{entity_name}').")
    cache = _cache_for(env)
    key = ("motion_link", entity_name, tuple(links_name))
    if key not in cache:
        for term in env.motion_manager._terms.values():
            if entity_name in term.entities_map:
                seq_key = term.entities_map[entity_name]
                target_links = term.target_links_name[seq_key]
                missing = [n for n in links_name if n not in target_links]
                if missing:
                    raise ValueError(
                        f"Links {missing} are not present in the motion reference for entity "
                        f"'{entity_name}'. Motion-ref links: {target_links}"
                    )
                idx = [target_links.index(n) for n in links_name]
                cache[key] = torch.tensor(idx, dtype=torch.long, device=env.device)
                break
        else:
            raise untracked_entity_error(env, entity_name)
    return cache[key]


def get_robot_dofs_idx(env: EnvBase, entity_name: str, dofs_name: Sequence[str]) -> torch.Tensor:
    """Resolve robot-entity DOF indices for ``dofs_name`` (cached).

    Returned indices are *local* (Genesis kinematic-tree order) indices, for the
    ``dofs_idx_local=`` argument of the entity's ``set_dofs_*`` / ``get_dofs_*``
    calls. NOT columns of a full-width ``get_dofs_*`` read or of ``get_jacobian``
    (those follow the entity's resolved ``dofs_name``; see ``find_named_dofs``).
    """
    if not dofs_name:
        raise ValueError(f"dofs_name must be non-empty (entity '{entity_name}').")
    cache = _cache_for(env)
    key = ("robot_dofs", entity_name, tuple(dofs_name))
    if key not in cache:
        _, indices = env.entities[entity_name].find_named_dofs_idx_local(list(dofs_name), preserve_order=True)
        cache[key] = torch.tensor(indices, dtype=torch.long, device=env.device)
    return cache[key]


def get_motion_dofs_idx(env: MimicRLEnvBase, entity_name: str, dofs_name: Sequence[str]) -> torch.Tensor:
    """Resolve motion-ref DOF indices for ``dofs_name`` (cached).

    Raises if the entity is not tracked by any motion term, or if any requested
    name is missing from the motion-ref DOF axis.
    """
    if not dofs_name:
        raise ValueError(f"dofs_name must be non-empty (entity '{entity_name}').")
    cache = _cache_for(env)
    key = ("motion_dofs", entity_name, tuple(dofs_name))
    if key not in cache:
        ref_dofs_name = env.motion_manager.get_target_dofs_name(entity_name)
        if ref_dofs_name is None:
            raise untracked_entity_error(env, entity_name)
        name_to_idx = {n: i for i, n in enumerate(ref_dofs_name)}
        missing = [n for n in dofs_name if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"DOFs {missing} are not present in the motion reference for entity "
                f"'{entity_name}'. Motion-ref dofs: {ref_dofs_name}"
            )
        idx = [name_to_idx[n] for n in dofs_name]
        cache[key] = torch.tensor(idx, dtype=torch.long, device=env.device)
    return cache[key]


def resolve_frame_offsets(
    env: MimicRLEnvBase,
    field_name: str,
    frame_offsets: Sequence[int],
    available_frames: int,
) -> torch.Tensor:
    """Validate ``frame_offsets`` against the buffer width and return them on-device.

    Caches the resulting long-tensor in a ``WeakKeyDictionary`` keyed by the
    env so we don't reallocate per step (and the cache evicts when the env
    is garbage-collected). Raises a clear ``ValueError`` if any offset
    exceeds ``num_frames - 1``; the message points at the matching
    ``MotionManagerOptions.num_frames`` knob so the fix is obvious.

    Used by both observation terms (``motion_future_*`` in
    ``observations/tracking.py``) and tracking reward terms that accept a
    ``frame_offsets`` kwarg, so the validation and caching behaviour stays
    in lockstep.
    """
    if not frame_offsets:
        raise ValueError(f"frame_offsets must be non-empty (field '{field_name}').")
    offsets_tuple = tuple(int(o) for o in frame_offsets)
    bad = [o for o in offsets_tuple if o < 0 or o >= available_frames]
    if bad:
        raise ValueError(
            f"frame_offsets {bad} are out of range for '{field_name}' "
            f"(motion buffer has num_frames={available_frames}; valid offsets are "
            f"[0, {available_frames - 1}]). Bump MotionManagerOptions.num_frames "
            f"to cover max(frame_offsets)+1."
        )
    per_env = _FRAME_OFFSETS_CACHE.get(env)
    if per_env is None:
        per_env = {}
        _FRAME_OFFSETS_CACHE[env] = per_env
    key = (field_name, offsets_tuple)
    cached = per_env.get(key)
    if cached is None:
        cached = torch.as_tensor(offsets_tuple, dtype=torch.long, device=env.device)
        per_env[key] = cached
    return cached


def select_motion_frames(
    field: torch.Tensor,
    frames: torch.Tensor | slice,
    sel_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Window a ``(num_envs, num_frames, ...)`` motion-ref buffer to ``frames``,
    then optionally select links / dofs.

    Unifies current-frame and future-frame access onto one ``(N, K, ...)`` path:
    the frame window is taken on **axis 1** and ``sel_idx`` (links for ``links_*``
    fields, dofs for ``dofs_*`` fields) on **axis 2** — always, for both paths.

    ``frames`` is either a ``slice`` (the current-frame ``K=1`` case — callers pass
    ``slice(0, 1)``, a zero-copy view, then drop the singleton axis) or a long
    tensor of buffer-frame indices (future terms — ``resolve_frame_offsets`` output,
    a gather). Slicing keeps the hot current-frame path copy-free; only ``sel_idx``
    (when given) copies.
    """
    windowed = field[:, frames] if isinstance(frames, slice) else field.index_select(1, frames)  # (N, K, ...)
    if sel_idx is not None:
        windowed = windowed.index_select(2, sel_idx)  # (N, K, sel, ...)
    return windowed


def current_motion_frame(field: torch.Tensor, sel_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Current-frame motion read: the ``K=1`` case of :func:`select_motion_frames`.

    Windows ``field`` to buffer frame ``0`` via a ``slice`` (a zero-copy view),
    optionally selects ``sel_idx`` on the links/dofs axis (axis 2, as on the
    future path), and drops the singleton frame axis. Shared by the observation /
    reward / termination tracking terms so current-frame reads land on the same
    axis everywhere; the slice keeps this hot path copy-free (only ``sel_idx``,
    when given, copies).
    """
    return select_motion_frames(field, slice(0, 1), sel_idx)[:, 0]


def assert_motion_field_present(env: MimicRLEnvBase, entity_name: str, field_name: str) -> None:
    """Raise unless the motion sequence for ``entity_name`` populates ``field_name``.

    ``MotionReferenceData`` always allocates every field via ``make_empty`` (zeros
    or identity quat), so reading the per-step buffer never sees ``None``. If the
    underlying dataset doesn't expose ``field_name``, the buffer stays at the
    make_empty default forever and a tracking term silently scores against
    garbage targets. This helper inspects the *source* ``BatchMotionSequence``
    and fails fast on the first call.

    Also enforces the ``BatchMotionSequence.is_free_floating`` advisory: scoring
    a ``base_*`` field against a sequence flagged as fixed-base is almost
    certainly a config error (the base fields would be left at the make-empty
    default), so we raise even if the field is technically populated.

    Cached per (env, entity_name, field_name) so subsequent calls are a single
    dict lookup.
    """
    cache = _cache_for(env)
    key = ("motion_field", entity_name, field_name)
    if key in cache:
        return
    for term in env.motion_manager._terms.values():
        if entity_name not in term.entities_map:
            continue
        seq_key = term.entities_map[entity_name]
        seq = term.data_source.motion_sequences[seq_key]
        if not hasattr(seq, field_name):
            raise ValueError(f"Motion sequence '{seq_key}' (entity '{entity_name}') has no field '{field_name}'.")
        if getattr(seq, field_name) is None:
            raise ValueError(
                f"Motion sequence '{seq_key}' (entity '{entity_name}') does not "
                f"populate '{field_name}'. The tracking term cannot run against "
                f"this dataset."
            )
        if field_name in _FIXED_BASE_GUARDED_FIELDS and not getattr(seq, "is_free_floating", True):
            raise ValueError(
                f"Motion sequence '{seq_key}' (entity '{entity_name}') is flagged "
                f"is_free_floating=False but a tracking term is reading '{field_name}'. "
                f"Use motion_link_*_tracking_exp / motion_dofs_pos_tracking_exp for "
                f"fixed-base entities."
            )
        cache[key] = True
        return
    raise untracked_entity_error(env, entity_name)


def get_anchor_state(
    env: MimicRLEnvBase,
    entity_name: str,
    anchor_link_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(robot_pos, robot_quat, motion_pos, motion_quat)`` for the anchor link.

    All are shaped ``(num_envs, 3 or 4)`` (no link axis).
    """
    robot_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])
    motion_idx = get_motion_link_idx(env, entity_name, [anchor_link_name])
    entity = env.entities[entity_name]
    ref = env.motion_manager.data[entity_name]
    return (
        entity.get_links_pos(ls_idx_local=robot_idx)[:, 0],
        entity.get_links_quat(ls_idx_local=robot_idx)[:, 0],
        # `index_select`, not `[..., motion_idx[0]]`: indexing with a 0-dim CUDA tensor is a sync.
        ref.links_pos_w[:, 0].index_select(1, motion_idx)[:, 0],
        ref.links_quat_w[:, 0].index_select(1, motion_idx)[:, 0],
    )


def get_links_state(
    env: MimicRLEnvBase,
    entity_name: str,
    links_name: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(robot_pos, robot_quat, motion_pos, motion_quat)`` for tracked body links.

    All are shaped ``(num_envs, K, 3 or 4)``.
    """
    robot_idx = get_robot_link_idx(env, entity_name, links_name)
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    entity = env.entities[entity_name]
    ref = env.motion_manager.data[entity_name]
    return (
        entity.get_links_pos(ls_idx_local=robot_idx),
        entity.get_links_quat(ls_idx_local=robot_idx),
        ref.links_pos_w[:, 0].index_select(1, motion_idx),
        ref.links_quat_w[:, 0].index_select(1, motion_idx),
    )


def compute_anchor_relative_links(
    env: MimicRLEnvBase,
    entity_name: str,
    anchor_link_name: str,
    tracked_link_names: list[str],
    *,
    anchor_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Yaw-only-warped motion body data and raw robot body data.

    Returns ``(links_pos_relative_w, links_quat_relative_w, robot_links_pos_w,
    robot_links_quat_w)`` each of shape ``(N, K, 3 or 4)``. ``anchor_state`` is
    :func:`get_anchor_state`'s result for the same entity and anchor, for a caller that already
    read it this step; None reads it here.
    """
    if anchor_state is None:
        anchor_state = get_anchor_state(env, entity_name, anchor_link_name)
    robot_anchor_pos_w, robot_anchor_quat_w, motion_anchor_pos_w, motion_anchor_quat_w = anchor_state
    robot_links_pos_w, robot_links_quat_w, motion_links_pos_w, motion_links_quat_w = get_links_state(
        env, entity_name, tracked_link_names
    )

    links_pos_relative_w, links_quat_relative_w = anchor_relative_links(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        motion_anchor_pos_w,
        motion_anchor_quat_w,
        motion_links_pos_w,
        motion_links_quat_w,
    )

    return links_pos_relative_w, links_quat_relative_w, robot_links_pos_w, robot_links_quat_w


def compute_anchor_relative_links_multi_future(
    env: MimicRLEnvBase,
    entity_name: str,
    anchor_link_name: str,
    tracked_link_names: list[str],
    frame_offsets: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-future variant of :func:`compute_anchor_relative_links`.

    For each future buffer index in ``frame_offsets``, computes the
    yaw-warped motion body data using *that future frame's* anchor as the
    warp reference (robot anchor stays at the current sim step — only the
    motion ref looks ahead). With ``frame_offsets=(0,)`` the K=1 reduction
    is bit-identical to :func:`compute_anchor_relative_links`.

    Returns ``(links_pos_relative_w, links_quat_relative_w, robot_links_pos_w,
    robot_links_quat_w, validity)`` where the first two are ``(N, K, L, 3|4)``,
    the next two are ``(N, L, 3|4)`` (the robot only has one "current"
    state — broadcast over K), and ``validity`` is ``(N, K)`` for the reward
    side to mask past-clip frames to zero.
    """
    robot_anchor_idx = get_robot_link_idx(env, entity_name, [anchor_link_name])
    motion_anchor_idx = get_motion_link_idx(env, entity_name, [anchor_link_name])
    robot_body_idx = get_robot_link_idx(env, entity_name, tracked_link_names)
    motion_body_idx = get_motion_link_idx(env, entity_name, tracked_link_names)
    entity = env.entities[entity_name]
    ref = env.motion_manager.data[entity_name]

    offsets = resolve_frame_offsets(env, "links_pos_w", frame_offsets, ref.links_pos_w.shape[1])

    # (N, K, 3|4) for the motion-ref anchor across the future window.
    motion_anchor_pos_w = ref.links_pos_w.index_select(1, offsets).index_select(2, motion_anchor_idx)[:, :, 0]
    motion_anchor_quat_w = ref.links_quat_w.index_select(1, offsets).index_select(2, motion_anchor_idx)[:, :, 0]
    # (N, K, L, 3|4) for the motion-ref tracked bodies across the future window.
    motion_links_pos_w = ref.links_pos_w.index_select(1, offsets).index_select(2, motion_body_idx)
    motion_links_quat_w = ref.links_quat_w.index_select(1, offsets).index_select(2, motion_body_idx)

    robot_anchor_pos_w = entity.get_links_pos(ls_idx_local=robot_anchor_idx)[:, 0]  # (N, 3)
    robot_anchor_quat_w = entity.get_links_quat(ls_idx_local=robot_anchor_idx)[:, 0]  # (N, 4)
    robot_links_pos_w = entity.get_links_pos(ls_idx_local=robot_body_idx)  # (N, L, 3)
    robot_links_quat_w = entity.get_links_quat(ls_idx_local=robot_body_idx)  # (N, L, 4)

    K = offsets.shape[0]
    # Robot anchor is the current state, broadcast over the K future frames; the
    # motion anchor/bodies already carry the K axis. The shared warp inserts the
    # tracked-link axis itself, so this handles the (N, K, L, ...) case directly.
    links_pos_relative_w, links_quat_relative_w = anchor_relative_links(
        robot_anchor_pos_w[:, None, :].expand(-1, K, -1),
        robot_anchor_quat_w[:, None, :].expand(-1, K, -1),
        motion_anchor_pos_w,
        motion_anchor_quat_w,
        motion_links_pos_w,
        motion_links_quat_w,
    )

    validity = ref.validity.index_select(1, offsets) if getattr(ref, "validity", None) is not None else None

    return links_pos_relative_w, links_quat_relative_w, robot_links_pos_w, robot_links_quat_w, validity


def anchor_state_once(
    env: MimicRLEnvBase, entity_name: str, anchor_link_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`get_anchor_state`, once per :meth:`~eden.envs.base.EnvBase.compute_once` pass.

    The anchor-height and anchor-orientation terminations and the BeyondMimic anchor metrics all read it: within
    a step's graded pass they read one. The result is shared: treat it as read-only.
    """
    return env.compute_once(
        ("anchor_state", entity_name, anchor_link_name), lambda: get_anchor_state(env, entity_name, anchor_link_name)
    )


def anchor_relative_links_once(
    env: MimicRLEnvBase,
    entity_name: str,
    anchor_link_name: str,
    tracked_link_names: Sequence[str],
    frame_offsets: Sequence[int] = (0,),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`compute_anchor_relative_links_multi_future`, once per :meth:`~eden.envs.base.EnvBase.compute_once` pass.

    The one producer of the anchor-relative warp for every term that grades against it — the body-position
    and body-orientation rewards and metrics. ``RLEnvBase.step`` opens one pass over its graded phase, so
    within a step they all read one warp. A current-frame caller passes ``frame_offsets=(0,)`` and reads
    index 0 of the frame axis, which is bit-identical to :func:`compute_anchor_relative_links`. The result is
    shared: treat it as read-only.
    """
    offsets = tuple(frame_offsets)
    links = tuple(tracked_link_names)
    return env.compute_once(
        ("anchor_relative_links", entity_name, anchor_link_name, links, offsets),
        lambda: compute_anchor_relative_links_multi_future(env, entity_name, anchor_link_name, list(links), offsets),
    )
