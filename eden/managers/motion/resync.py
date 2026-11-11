"""Motion-end resync (opt-in): keep an episode running past a clip's end.

When ``MotionManagerOptions.resync_on_motion_end`` is set, an env whose reference
reaches the end of its clip is **soft-reset mid-episode** — a fresh start phase is
resampled, its playback clock restarted, and the robot teleported onto the new
reference start — WITHOUT ending the episode. This mirrors the clip-wrap behavior
of ``MotionCommand`` (the ``beyondmimic`` example); the default (off) freezes the
reference at the clip's last frame.

It is **folded into the single per-step ``fill_data(all)``**: :meth:`pre_fill`
detects wrapped envs from the just-advanced clock (before the fill) and resamples
them, so the one ``fill_data`` computes their new-phase reference (no second
fill); :meth:`post_fill` then teleports them onto that freshly-filled reference.
One instance per :class:`~eden.managers.motion_manager.MotionManager`, created in
its ``__init__``; it reads the manager's state through a back-reference.

Hot-path cost contract: no GPU->CPU sync. Whether any env wrapped is never
read on the host: every step resamples and teleports through the wrap mask, every
env's row with the mask picking the wrapped ones (``torch.where`` / ``masked_fill_``,
and Genesis' masked forward kinematics), so a step where nothing wrapped writes
nothing — it only runs the same few kernels over all envs. That fixed cost does
not outweigh the sync it replaces: BeyondMimic at 4096 envs (L40S) stepped in
24.12 ms against 24.16 ms with the ``nonzero()`` gate. (``MotionCommand`` keeps its
``.any()`` gate: its reset path adds pose noise and full-batch draws, ~1 ms.)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from eden.managers.motion_manager import MotionManager


class MotionEndResync:
    """Owns the opt-in motion-end resync operation for a ``MotionManager``."""

    def __init__(self, manager: MotionManager) -> None:
        self._mm = manager
        # RSI RESET-event terms, resolved lazily (the event manager is built after the
        # motion manager) but forced at build time via :meth:`validate` when enabled.
        self._rsi_terms: list | None = None
        # Env lookup hoisted out of the per-step post_fill.
        self._debug = os.environ.get("EDEN_RESYNC_DEBUG") == "1"

    @property
    def enabled(self) -> bool:
        return bool(self._mm._options.resync_on_motion_end)

    def validate(self) -> None:
        """Build-time check (a post-build hook, once the event manager exists): resolve
        the RSI teleport terms now so a missing one raises when ``resync_on_motion_end``
        is first honored — at env construction — instead of at the first mid-episode
        wrap. The resolution is cached, so the first wrap pays nothing.
        """
        self._reference_init_terms()

    def pre_fill(self) -> torch.Tensor | None:
        """Before ``fill_data(all)``: detect wrapped envs and resample them so the
        single fill computes their new-phase reference. Returns the ``(num_envs,)``
        wrap mask (to hand back to :meth:`post_fill`), or ``None`` when disabled.
        The mask is not read on the host, so it may select no env: everything below
        writes through it, which then writes nothing.
        """
        if not self.enabled:
            return None
        wrap_mask = self._wrapped_mask()
        self._resample(wrap_mask)
        return wrap_mask

    def post_fill(self, wrap_mask: torch.Tensor) -> None:
        """After ``fill_data(all)``: teleport the wrapped envs onto the freshly
        filled reference. With ``EDEN_RESYNC_DEBUG=1``, first assert the pre-fill wrap
        detection matched ``fill_data``'s validity (no env still past-end).
        """
        if self._debug:
            for _e, _ref in self._mm._data.items():
                assert not (~_ref.validity[:, 0]).any(), "resync: env still past clip end after resample+fill"
        self._teleport(wrap_mask)

    # -- internals -----------------------------------------------------------------

    def _wrapped_mask(self) -> torch.Tensor:
        """Bool mask (num_envs,) of envs whose CURRENT (frame-0) reference is past
        its clip end, from the just-advanced ``_timestamp`` — computed BEFORE
        ``fill_data``. Uses the term's ``_past_clip_end`` (bit-exact to the negation
        of ``fill_data``'s frame-0 validity — same product, floor dropped) at the
        frame-0 sample timestamp (``_timestamp`` + the frame-0 time-to-target offset:
        ``frame_interval`` for ``data_start_from == "one_frame_interval"``, else 0).
        Sharing that product guarantees the wrap detection can never drift from
        ``fill_data``'s validity across ``num_frames`` / ``data_start_from`` / clip
        framerates. Sync-free.
        """
        mm = self._mm
        t2t0_scale = 1.0 if mm._options.data_start_from == "one_frame_interval" else 0.0

        def _term_wrap(term, sl) -> torch.Tensor:
            assigned = term.assigned_ids_from_envs_idx(sl)
            # frame-0 sample timestamp for this term's envs, matching fill_data's
            # sample_timestamp[:, 0] (= _timestamp + time_to_target_frame[0]).
            sample_ts0 = (mm._timestamp[sl] + t2t0_scale * mm._frame_interval_s[sl]).unsqueeze(-1)
            term_wrap: torch.Tensor | None = None
            for _entity_name, motion_seq_key in term.entities_map.items():
                data = term.data_source.motion_sequences[motion_seq_key]
                # _past_clip_end shares fill_data's frame product (bit-exact validity)
                # but skips the floor/bool-cast the wrap check doesn't need.
                past = term._past_clip_end(assigned, sample_ts0, data)[:, 0]
                term_wrap = past if term_wrap is None else term_wrap | past
            return term_wrap

        # Single-term common case (beyondmimic): a term spanning all envs has a wrap mask
        # that IS the answer — no full-buffer alloc + scatter-or. One owning a slice gets
        # the scatter below, so the mask always has one entry per env.
        if len(mm._terms) == 1:
            ((name, term),) = mm._terms.items()
            sl = mm._motion_term_assignment[name]
            if sl.start == 0 and sl.stop == mm.num_envs:
                return _term_wrap(term, sl)
        wrap = torch.zeros(mm.num_envs, dtype=torch.bool, device=mm.device)
        for name, term in mm._terms.items():
            sl = mm._motion_term_assignment[name]
            wrap[sl] |= _term_wrap(term, sl)
        return wrap

    def _resample(self, wrap_mask: torch.Tensor) -> None:
        """Resample a fresh start phase + restart the clock for the wrapped envs.

        Through the mask: each term samples a start time for every env it owns and
        writes the wrapped ones (``_sample_env_motion_start_time``'s bool-mask path),
        with no ``nonzero()``. Does NOT touch ``episode_length_buf`` or fire any
        termination, so the episode keeps running past the clip end (instead of the
        default freeze-at-last-frame).
        """
        mm = self._mm
        for term in mm._terms.values():
            term._sample_env_motion_start_time(term.assigned_mask_from_envs_mask(wrap_mask))
        mm._timestamp.masked_fill_(wrap_mask, 0.0)

    def _teleport(self, wrap_mask: torch.Tensor) -> None:
        """Teleport the wrapped envs onto the just-filled ``_data`` reference.

        Delegates the per-entity seeding to each RSI term's ``_stage_teleport`` (branch-aware:
        a floating base, an immobile/welded child, or a free object each know their own
        payload and index), so the resync path holds no per-flavour knowledge — a new
        RSI flavour is picked up by construction. Reads the reference straight from
        ``_data`` (filled at the new phase by the single ``fill_data``), avoiding the
        RSI ``get_init_reference_state`` recompute. Every env's frame 0 is staged and
        Genesis is handed the wrap mask, which it writes through on the device:
        no ``nonzero()``, and on a step where nothing wrapped, no write.

        Every term stages its entity and ``teleport_batch`` writes them all at once, the
        same single write the reset path makes (see ``StagedStateWrite``); the
        free-joint body-frame angular-velocity convention lives in each term's
        ``_stage_teleport``. No reset perturbation noise is added — a wrap is mid-episode,
        so the scene snaps exactly onto the reference.
        """
        from eden.managers.terms.events.motion import ReferenceStateInitializationBase

        ReferenceStateInitializationBase.teleport_batch(self._reference_init_terms(), wrap_mask, self._mm._data)

    def _reference_init_terms(self) -> list:
        """The RSI RESET-event terms to teleport at a clip wrap, cached.

        Selected by ``isinstance`` against ``ReferenceStateInitializationBase`` so every
        RSI flavour — the robot (floating *or* immobile base) and a manipulated free
        object, plus any future one or subclass — is picked up, rather than only one
        exact class. The event manager
        is built after the motion manager, so this resolves lazily on first use; the
        ``validate`` post-build hook forces it at build time when resync is enabled.

        Every selected term must actually implement ``_stage_teleport`` (the base method only
        raises): a subclass that forgot to override it is rejected here — at build time
        when ``validate`` runs — rather than seeding nothing (or raising) at the first
        wrap, so ``resync`` never silently teleports a subset of the scene.
        """
        if self._rsi_terms is None:
            from eden.constants import EventMode
            from eden.managers.terms.events.motion import ReferenceStateInitializationBase

            em = getattr(self._mm._env, "event_manager", None)
            reset_terms = em._mode_reset_terms.get(EventMode.RESET, []) if em is not None else []
            self._rsi_terms = [t for t in reset_terms if isinstance(t, ReferenceStateInitializationBase)]
            if not self._rsi_terms:
                raise RuntimeError(
                    "resync_on_motion_end=True requires at least one ReferenceStateInitialization RESET "
                    "event term to teleport the scene at clip wrap, but none was found in the event manager."
                )
            unteleportable = [
                type(t).__name__
                for t in self._rsi_terms
                if type(t)._stage_teleport is ReferenceStateInitializationBase._stage_teleport
            ]
            if unteleportable:
                raise RuntimeError(
                    f"resync_on_motion_end=True found ReferenceStateInitialization RESET event term(s) that "
                    f"do not implement _stage_teleport(): {unteleportable}. Every RSI flavour must override "
                    f"_stage_teleport (not teleport, which only calls it) to seed its entity onto the resampled "
                    f"reference at a clip wrap."
                )
        return self._rsi_terms
