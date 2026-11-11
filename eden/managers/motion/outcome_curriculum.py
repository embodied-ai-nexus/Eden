"""Per-motion-id outcome curriculum (hard-motion mining).

Split out of ``MotionTerm``: owns the per-motion success/failure count
buffers, the drain cadence, the weight-rewrite dynamics (snapshot / opt-in
compound + uniform-mix regulariser), and the wandb summary/histogram reporting.
One instance per term that sets ``adaptive_motion_weights=True``, created in
``MotionTerm.__init__``.

``_motion_weights`` itself stays on the term — it is core sampling state read by
``_local_motion_weights`` / ``get_current_motion_weights``. This curriculum
*rewrites* it in place (through the back-reference) from the drained outcome
rates; everything else it touches (counts, summaries) it owns.

Cadence: ``MotionTerm._update_motion_weight_curriculum`` calls ``update`` once per
PPO iter; every ``motion_weight_update_every_n_iters``-th call drains, optionally
all-reduces across DDP, and rewrites the weights. The term piggybacks the
per-motion metric drain on the same cadence.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import torch

from eden.managers.motion.constants import _NORMALIZE_EPS

if TYPE_CHECKING:
    import numpy as np

    from eden.managers.motion_manager import MotionTerm


class MotionOutcomeCurriculum:
    """Owns the per-motion-id outcome counts and the weight-rewrite pipeline."""

    def __init__(self, term: MotionTerm) -> None:
        self._term = term
        # Fail fast on bad curriculum config — caught here at term setup rather
        # than at the first drain (which can be many iters into training when
        # ``motion_weight_update_every_n_iters > 1``).
        if not (0.0 <= term.motion_weight_uniform_mix <= 1.0):
            raise ValueError(
                f"motion_weight_uniform_mix must be in [0, 1], got "
                f"{term.motion_weight_uniform_mix}. The regulariser blends "
                f"``w := (1 - α) * w + α / N`` after the curriculum update; "
                f"values outside [0, 1] produce non-distributions."
            )
        if term.motion_weight_update_mode == "snapshot" and term.motion_weight_failure_rate_max_over_mean <= 0.0:
            # Without a positive clamp ceiling, ``_apply_snapshot_curriculum``
            # produces an all-zero ``_motion_weights`` (clip_upper ≤ 0 → every
            # clipped rate is 0 → division by clamp_min(_NORMALIZE_EPS) yields
            # all zeros). The downstream ``torch.multinomial`` would crash or
            # return garbage. Reject the config at setup time.
            raise ValueError(
                f"motion_weight_failure_rate_max_over_mean must be > 0 in snapshot mode, "
                f"got {term.motion_weight_failure_rate_max_over_mean}. SONIC's default is "
                f"200.0 (lenient clamp); tighter values like 5.0 are valid and bite the "
                f"distribution earlier."
            )
        n = term._global_num_motions
        # Sized to the GLOBAL motion-id space so DDP all_reduce stays
        # shape-symmetric across ranks regardless of mp_split_method.
        self._motion_success_counts = torch.zeros(n, dtype=torch.long, device=term.device)
        self._motion_failure_counts = torch.zeros(n, dtype=torch.long, device=term.device)
        self._iters_since_motion_weight_update = 0
        # ``_curriculum_has_run`` gates the floor-clip metric: pre-first-drain
        # the user-supplied raw weights have never been clamped, so reporting a
        # "fraction at floor" relative to the configured floor is noise (e.g.
        # raw weights ``[1e-4, …]`` look 100% floor-clipped though the
        # curriculum hasn't touched them). Flip on once the first apply runs.
        self._curriculum_has_run = False
        # Cached summaries (mixed float/str; forwarded verbatim to wandb.log via
        # the rsl_rl mixin). Refreshed only on drain / init, so the hot reset
        # path is a dict lookup rather than 5+ GPU→CPU syncs.
        self._last_curriculum_summary: dict[str, float | str] = {}
        self._last_curriculum_per_motion: dict[str, np.ndarray] | None = None
        self._last_weights_per_motion: np.ndarray | None = None
        self._last_weights_summary: dict[str, float | str] = self._compute_motion_weights_summary()

    def record_motion_outcomes(
        self,
        success_envs_idx: torch.Tensor,
        failure_envs_idx: torch.Tensor,
        success_timestamps: torch.Tensor,
        failure_timestamps: torch.Tensor,
        env_dt: float,
    ) -> None:
        """Increment per-motion success / failure counts for envs that just reset.

        Classification is driven by **motion completion**, not by which cohort
        an env arrived in (matches ProtoMotions's ``done_clip``: a clip is
        "done" when ``motion_times + env_dt >= end_times``, regardless of why
        the episode reset).

        - Completed (``start_time + episode_time + env_dt >= motion_length``)
          → success. This includes terminated envs whose motion happened to
          play to its natural end.
        - Terminated AND not completed → failure.
        - Not terminated AND not completed (mid-motion timeout) → dropped.

        Both ``*_envs_idx`` accept the same forms as ``envs_idx_to_assigned_ids``:
        a bool mask of length ``num_envs`` (preferred — avoids GPU sync) or an
        int-index tensor of global env ids. ``*_timestamps`` are compact tensors
        aligned with the corresponding cohort.
        """
        # Non-terminated cohort: only completed motions count as success; the
        # rest (mid-motion timeouts) are ambiguous and dropped on the floor.
        self._tally_outcomes(success_envs_idx, success_timestamps, env_dt, count_incomplete_as_failure=False)
        # Terminated cohort: completed motions still count as success (the
        # termination just coincided with natural end); the rest are failures.
        self._tally_outcomes(failure_envs_idx, failure_timestamps, env_dt, count_incomplete_as_failure=True)

    def _tally_outcomes(
        self,
        envs_idx: torch.Tensor,
        timestamps: torch.Tensor,
        env_dt: float,
        *,
        count_incomplete_as_failure: bool,
    ) -> None:
        """Increment success/failure counts for one cohort of resetting envs.

        Outcome buffers are sized to the GLOBAL motion-id space, while
        ``_assigned_env_motion_selection`` indexes into the per-rank local
        subset; map through ``_local_to_global`` before scattering so every
        rank's contribution lands in a shape-compatible tensor for the
        downstream ``dist.all_reduce``.

        ``scatter_add_`` is a no-op on empty index tensors, so empty cohorts
        flow through without explicit guards.
        """
        term = self._term
        assigned = term.envs_idx_to_assigned_ids(envs_idx)
        motion_ids_local = term._assigned_env_motion_selection[assigned]
        completed_mask = self._motion_completed_mask(assigned, motion_ids_local, timestamps, env_dt)

        motion_ids_global = term._local_to_global[motion_ids_local]
        # Every env's motion, weighted 1 or 0 by the outcome: the same whole-number counts as scattering
        # ones over the selected subset, without the bool-mask gather that picked it, a GPU->CPU sync.
        self._motion_success_counts.scatter_add_(
            0, motion_ids_global, completed_mask.to(self._motion_success_counts.dtype)
        )

        if count_incomplete_as_failure:
            self._motion_failure_counts.scatter_add_(
                0, motion_ids_global, (~completed_mask).to(self._motion_failure_counts.dtype)
            )

    def _motion_completed_mask(
        self,
        assigned_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        timestamps: torch.Tensor,
        env_dt: float,
    ) -> torch.Tensor:
        """Per-env boolean mask for whether the motion played to its natural end.

        ``played_time + env_dt >= motion_length`` matches ProtoMotions's
        ``done_clip = (motion_times + env_dt) >= end_times`` formulation.
        """
        term = self._term
        motion_lengths = term.data_source.buffer_length[motion_ids].to(timestamps.dtype) / term.data_source.framerate[
            motion_ids
        ].to(timestamps.dtype)
        played_time = term._motion_buffer_start_time_s[assigned_ids] + timestamps
        return played_time + env_dt >= motion_lengths

    def update(self, all_reduce_fn=None) -> bool:
        """Drain outcome counts, optionally reduce across DDP ranks, rewrite weights.

        Called once per PPO iteration via ``MotionTerm._update_motion_weight_curriculum``.
        The first ``motion_weight_update_every_n_iters - 1`` calls just bump the
        internal iter counter and return ``False`` (counts keep accumulating);
        the K-th call drains, optionally all-reduces, and applies the curriculum,
        returning ``True`` (which the term uses to gate the metric drain onto the
        same cadence).

        ``all_reduce_fn`` — if provided, applied in-place to both count tensors
        before they drive the curriculum (the manager supplies a
        ``dist.all_reduce`` lambda for DDP; tests pass ``None``).
        """
        self._iters_since_motion_weight_update += 1
        if self._iters_since_motion_weight_update < self._term.motion_weight_update_every_n_iters:
            return False
        self._iters_since_motion_weight_update = 0

        success = self._motion_success_counts.clone()
        failure = self._motion_failure_counts.clone()
        self._motion_success_counts.zero_()
        self._motion_failure_counts.zero_()

        if all_reduce_fn is not None:
            all_reduce_fn(success)
            all_reduce_fn(failure)

        self._record_curriculum_summary(success, failure)
        self._apply_motion_outcome_curriculum(success, failure)
        self._curriculum_has_run = True
        # Refresh the weights-summary cache now that ``_motion_weights`` has been
        # rewritten. Done here (not on every reset) so the hot reset path doesn't
        # pay 5 GPU→CPU syncs every time an env terminates.
        self._last_weights_summary = self._compute_motion_weights_summary()
        return True

    def _record_curriculum_summary(
        self,
        success_counts: torch.Tensor,
        failure_counts: torch.Tensor,
    ) -> None:
        """Stash per-iter outcome scalars for ``MotionManager.reset()`` to surface.

        Called after the post-drain ``all_reduce`` so the values are already
        byte-identical across DDP ranks. ``success`` / ``failure`` are global
        counts since the last drain; the resulting per-motion success rate is the
        rate over that drain window. Motions with no episodes this window are
        excluded from the rate statistics.
        """
        total = success_counts + failure_counts
        has_data = total > 0
        n_with_data = int(has_data.sum().item())
        episodes_total = int(total.sum().item())
        if n_with_data == 0:
            self._last_curriculum_summary = {
                "MotionOutcome/episodes_total": float(episodes_total),
                "MotionOutcome/motions_with_data": 0.0,
            }
            # Keep the histogram cache in lock-step with the scalar summary: an
            # empty drain window means no per-motion data to plot. Without this
            # reset, ``collect_per_motion_distributions`` would re-emit the
            # previous drain's outcome arrays, lagging the (empty) aggregates.
            self._last_curriculum_per_motion = None
            return
        # Per-motion success rate over the drain window. Float for quantile().
        sr_all = success_counts.float() / total.clamp_min(1).float()
        sr = sr_all[has_data]
        summary = {
            "MotionOutcome/episodes_total": float(episodes_total),
            "MotionOutcome/motions_with_data": float(n_with_data),
            "MotionOutcome/success_rate_mean": sr.mean().item(),
            "MotionOutcome/success_rate_p10": torch.quantile(sr, 0.10).item(),
            "MotionOutcome/success_rate_p50": torch.quantile(sr, 0.50).item(),
        }
        # Per-motion data cached as raw numpy arrays for the histogram emit path
        # (``collect_per_motion_distributions``). Histograms (not per-id scalars)
        # keep this usable at AMASS scale (~9.4k motions) — a per-id scalar
        # fan-out is ~38k series/iter, more than the wandb backend can take. One
        # ``torch.stack(...).cpu().numpy()`` round-trips all three in one sync.
        if self._term.log_per_motion_metrics:
            stacked = (
                torch.stack([success_counts.to(sr_all.dtype), failure_counts.to(sr_all.dtype), sr_all]).cpu().numpy()
            )
            succ_np, fail_np, sr_np = stacked
            self._last_curriculum_per_motion = {
                "success_count": succ_np,
                "failure_count": fail_np,
                "total": succ_np + fail_np,
                "success_rate": sr_np,
            }
        else:
            self._last_curriculum_per_motion = None
        self._last_curriculum_summary = summary

    def summarize_motion_weights(self) -> dict[str, float | str]:
        """Read the cached scalar summary of the motion-weights distribution.

        Returns the dict populated by ``_compute_motion_weights_summary``,
        refreshed only when ``_motion_weights`` actually changes (init, plus each
        ``update`` call that drains). The hot reset path reads this cache rather than
        recomputing from the GPU tensor (which would cost 5 GPU→CPU syncs).
        """
        return self._last_weights_summary

    @property
    def last_curriculum_summary(self) -> dict[str, float | str]:
        """Cached per-iter outcome scalars (``MotionOutcome/*``); empty until first drain."""
        return self._last_curriculum_summary

    def collect_per_motion_distributions(self, wandb_module) -> dict[str, Any]:
        """Build per-motion histograms for the wandb-direct emit path.

        Returns ``wandb.Histogram`` objects keyed under ``MotionPerMotion/*`` when
        ``log_per_motion_metrics=True``; an empty dict otherwise. Built from the
        cached numpy arrays ``_compute_motion_weights_summary`` /
        ``_record_curriculum_summary`` populate — one snapshot per drain, not per
        env step. Histograms (not per-id scalars) keep the fan-out constant-cost
        at AMASS / SEED scale.
        """
        if not self._term.log_per_motion_metrics:
            return {}
        out: dict[str, Any] = {}
        if self._last_weights_per_motion is not None:
            out["MotionPerMotion/Weights"] = wandb_module.Histogram(self._last_weights_per_motion)
        if self._last_curriculum_per_motion is not None:
            for field, arr in self._last_curriculum_per_motion.items():
                out[f"MotionPerMotion/Outcome/{field}"] = wandb_module.Histogram(arr)
        return out

    def _compute_motion_weights_summary(self) -> dict[str, float | str]:
        """Compute the weights summary from ``_motion_weights`` — slow path.

        Called once at init and once per curriculum update. Coalesces all
        GPU→CPU transfers into a single ``.tolist()`` so the cost is one
        synchronous round-trip instead of five.

        ``_motion_weights`` may hold raw user-supplied weights (e.g. ``[1, 2,
        1]``) rather than a normalized distribution — ``torch.multinomial``
        auto-normalizes at sample time. We normalize a copy internally so
        entropy / floor-clip describe the actual sampling distribution
        regardless of input scale.
        """
        weights = self._term._motion_weights
        if weights is None or len(weights) == 0:
            return {}
        raw = weights
        n = len(raw)
        total = raw.sum().clamp_min(_NORMALIZE_EPS)
        w = raw / total
        # Floor-clip detection mirrors the curriculum's own floor in normalized
        # space (the curriculum clamps in raw space at ``motion_weight_floor``).
        floor_normalized = self._term.motion_weight_floor / total
        floor_eps = (floor_normalized * 1e-3).clamp_min(1e-12)
        # All four aggregate scalars + the argmax index coalesced into one
        # ``.tolist()`` — a single CUDA sync.
        stats = torch.stack(
            [
                -(w * w.clamp_min(_NORMALIZE_EPS).log()).sum(),
                w.min(),
                w.max(),
                (w <= floor_normalized + floor_eps).float().mean(),
                w.argmax().to(w.dtype),
            ]
        ).tolist()
        entropy_nats, w_min, w_max, floor_clipped, argmax_id = stats
        summary: dict[str, float | str] = {
            "MotionWeights/entropy": entropy_nats,
            "MotionWeights/entropy_normalized": entropy_nats / math.log(n) if n > 1 else 0.0,
            "MotionWeights/min": w_min,
            "MotionWeights/max": w_max,
            # Cheapest "which motion is dominating" diagnostic — one int per
            # drain, works at any scale; logged even when log_per_motion_metrics
            # is off so SEED-scale runs can still identify the runaway clip.
            "MotionWeights/argmax_motion_id": float(int(argmax_id)),
        }
        # Only emit floor_clip_ratio once the curriculum has rewritten the
        # tensor — pre-first-drain it measures the *raw* user weights, which is
        # noise. After the first drain the weights are normalized + clamped.
        if self._curriculum_has_run:
            summary["MotionWeights/floor_clip_ratio"] = floor_clipped
        if self._term.log_per_motion_metrics:
            self._last_weights_per_motion = w.detach().cpu().numpy()
        else:
            self._last_weights_per_motion = None
        # Optional human-readable dominant-motion identifier.
        if self._term.log_motion_id_to_filename:
            filename = self._resolve_argmax_motion_filename(int(argmax_id))
            if filename is not None:
                summary["MotionWeights/argmax_motion_filename"] = filename
        return summary

    def _resolve_argmax_motion_filename(self, argmax_global_id: int) -> str | None:
        """Resolve a global motion-id to its dataset filename (basename).

        Returns ``None`` when the dataset doesn't expose ``all_motion_files``, or
        ``"<not_local>"`` when the global id is held by another DDP rank's shard
        (``_local_to_global`` doesn't contain it). ``all_motion_files`` is indexed
        in the LOCAL space, so we reverse-look-up ``_local_to_global``.
        """
        term = self._term
        if not hasattr(term.data_source, "all_motion_files"):
            return None
        files = term.data_source.all_motion_files
        if not files:
            return None
        mask = term._local_to_global == argmax_global_id
        if not mask.any():
            # argmax is on another rank's shard.
            return "<not_local>"
        local_idx = int(mask.nonzero(as_tuple=True)[0][0].item())
        if not (0 <= local_idx < len(files)):
            return None
        return os.path.basename(files[local_idx])

    def _apply_motion_outcome_curriculum(
        self,
        success_counts: torch.Tensor,
        failure_counts: torch.Tensor,
    ) -> None:
        """Rewrite ``_motion_weights`` from per-motion outcome counts.

        Dispatches on ``motion_weight_update_mode``:

        * ``"snapshot"`` (default): stateless rate-proportional update ported
          from SONIC. Always reflects the **current** drained rates.
        * ``"compound"`` (opt-in): multiplicative log-space update; collapses
          to a single motion at LAFAN1 scale; kept only for ablation.

        Both modes apply ``motion_weight_uniform_mix`` as a final regulariser.
        """
        mode = self._term.motion_weight_update_mode
        if mode == "compound":
            self._apply_compound_curriculum(success_counts, failure_counts)
        elif mode == "snapshot":
            self._apply_snapshot_curriculum(success_counts, failure_counts)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"Unknown motion_weight_update_mode '{mode}'. Valid: 'compound', 'snapshot'.")
        self._apply_uniform_mix_regulariser()

    def _apply_compound_curriculum(
        self,
        success_counts: torch.Tensor,
        failure_counts: torch.Tensor,
    ) -> None:
        """Multiplicative log-space update (``motion_weight_update_mode="compound"``).

        Each success applies one ``motion_weight_success_discount`` factor and
        each failure one ``motion_weight_failure_amplify`` factor, then weights
        are clamped to ``motion_weight_floor`` and re-normalised. Computed in
        log-space so the multiplicative factor cannot overflow float32.

        **Known failure mode**: at LAFAN1-scale this collapses the distribution
        to a single motion within a few thousand iters. Use
        ``motion_weight_update_mode="snapshot"`` instead.
        """
        term = self._term
        weights = term._motion_weights
        n_success = success_counts.to(weights.dtype)
        n_failure = failure_counts.to(weights.dtype)
        log_factor = n_success * math.log(term.motion_weight_success_discount) + n_failure * math.log(
            term.motion_weight_failure_amplify
        )
        log_w = weights.log() + log_factor
        if term.motion_weight_floor > 0:
            log_w.clamp_(min=math.log(term.motion_weight_floor))
        norm = log_w.logsumexp(0)
        if torch.isfinite(norm):
            weights[:] = (log_w - norm).exp()
        else:
            # Catastrophic underflow — fall back to uniform so the multinomial
            # stays a valid distribution.
            weights[:] = 1.0 / len(weights)

    def _apply_snapshot_curriculum(
        self,
        success_counts: torch.Tensor,
        failure_counts: torch.Tensor,
    ) -> None:
        """Stateless rate-proportional curriculum — SONIC's recipe.

        ``failure_rate = n_failure / (n_success + n_failure)`` per motion-id over
        the drain window. Outliers are clamped at ``mean(failure_rate) *
        motion_weight_failure_rate_max_over_mean`` so no single motion can absorb
        arbitrarily much mass; the clamped rates are normalised and written to
        ``_motion_weights`` — **no compound history**.

        Motions with zero episodes contribute ``failure_rate=0`` (weight 0 until
        the uniform-mix regulariser lifts it). If NO motion had episodes this window the
        weights are unchanged. ``success_discount`` / ``failure_amplify`` /
        ``floor`` are unused under this mode.
        """
        term = self._term
        weights = term._motion_weights
        total = (success_counts + failure_counts).to(weights.dtype)
        failure_rate = failure_counts.to(weights.dtype) / total.clamp_min(1.0)
        # Mean over ALL motions (matches SONIC: motions with no data contribute 0).
        mean_fr = failure_rate.mean()
        # Coalesce two scalar reads into one ``.tolist()`` (single CUDA sync).
        total_sum_f, mean_fr_f = torch.stack([total.sum(), mean_fr]).tolist()
        if total_sum_f == 0.0:
            # No outcomes this window — leave weights as-is.
            return
        if mean_fr_f == 0.0:
            # Every motion succeeded → no signal. Fall back to uniform.
            weights[:] = 1.0 / len(weights)
            return
        # ``clip_upper`` stays a 0-dim tensor so ``clamp_max`` broadcasts
        # on-device. ``failure_rate`` is non-negative by construction.
        clip_upper = mean_fr * term.motion_weight_failure_rate_max_over_mean
        clipped = failure_rate.clamp_max(clip_upper)
        s = clipped.sum().clamp_min(_NORMALIZE_EPS)
        weights[:] = clipped / s

    def _apply_uniform_mix_regulariser(self) -> None:
        """Blend the post-curriculum distribution with uniform.

        ``w := (1 - α) * w + α / N``. An affine combination of two distributions
        is itself a valid distribution, so sum-to-1 is preserved; any single
        motion is bounded to ``[α/N, (1 - α) + α/N]``.

        ``alpha`` is validated at init; the runtime check is defence-in-depth
        against post-init mutation. Validation runs before the ``alpha == 0``
        no-op so negative values raise rather than silently no-opping.
        """
        weights = self._term._motion_weights
        alpha = self._term.motion_weight_uniform_mix
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"motion_weight_uniform_mix must be in [0, 1], got {alpha}.")
        if alpha == 0:
            return
        n = weights.shape[0]
        weights.mul_(1.0 - alpha).add_(alpha / n)
