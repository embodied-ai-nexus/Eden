"""Intra-motion start-time bin curriculum (adaptive start-time sampling).

Split out of ``MotionTerm``. Owns the per-motion temporal **bin
weights** (a ``ConcatBatchTensor`` of per-bin sampling probabilities) and, when
``adaptive_sampling`` is on, the death-bin failure-count EMA + the vectorised weight-update
that biases start-time sampling toward the run-up to hard moments.

One instance per term that sets ``motion_bin_length_s`` (created/recreated in
``MotionTerm.enable_trajectories`` whenever the motion library changes). The bin
weights exist whenever ``motion_bin_length_s`` is set — they drive weighted
start-time sampling even without the curriculum; the adaptive EMA layer activates
only when ``adaptive_sampling=True``, so ``record_bin_outcomes`` / ``blend_failure_ema``
/ ``_update_adaptive_bin_weights`` keep their internal adaptive guards.

The term's start-time samplers (``_sample_env_motion_start_time`` /
``_sample_as_concat_motion_bins``) read the distribution via :attr:`weights`.
Term env-assignment state (``_assigned_env_motion_selection``,
``_motion_buffer_start_time_s``, ``envs_idx_to_assigned_ids``) is read through the
back-reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.motion.constants import _NORMALIZE_EPS
from eden.utils.buffers.cbt import ConcatBatchTensor

if TYPE_CHECKING:
    from eden.managers.motion_manager import MotionTerm


class AdaptiveBinSampler:
    """Owns the temporal bin weights and the adaptive start-bin curriculum."""

    def __init__(self, term: MotionTerm) -> None:
        """Initialize temporal bin weights for weighted start-time sampling."""
        self._term = term
        device = term.device

        num_motion_bins = torch.clamp(
            torch.floor(
                (term.data_source.buffer_length / term.data_source.framerate).to(device) / term.motion_bin_length_s
            ),
            min=1,
        ).to(torch.long)
        self._motion_bin_weights = ConcatBatchTensor(
            batch_sizes=num_motion_bins.tolist(),
            data_shape=tuple(),
            device=device,
        )
        for motion_idx in range(len(term.data_source)):
            motion_bin_probability = torch.ones(num_motion_bins[motion_idx], device=device, dtype=torch.float)
            if motion_bin_probability.numel() > 0:
                motion_bin_probability /= motion_bin_probability.sum()
            self._motion_bin_weights[motion_idx] = motion_bin_probability

        # Per-death-bin failure-count EMA (see ``record_bin_outcomes``).
        # ``None`` (allocated below) unless ``adaptive_sampling`` — the bin
        # weights themselves are used for weighted start sampling regardless.
        self._bin_failure_ema: torch.Tensor | None = None
        self._current_bin_failed: torch.Tensor | None = None
        self._adaptive_kernel: torch.Tensor | None = None
        # Precomputed per-bin coordinates + ``max(sizes)`` for the vectorised
        # weight update; depend only on sizes/starts (immutable for this library).
        self._adaptive_motion_ids_flat: torch.Tensor | None = None
        self._adaptive_pos_in_motion_flat: torch.Tensor | None = None
        self._adaptive_max_bins: int = 0
        self._adaptive_window_idx: torch.Tensor | None = None
        self._group_bin_mask: torch.Tensor | None = None

        if term.adaptive_sampling:
            total_bins = self._motion_bin_weights.concatenated_tensor.shape[0]
            self._bin_failure_ema = torch.zeros(total_bins, device=device, dtype=torch.float)
            self._current_bin_failed = torch.zeros(total_bins, device=device, dtype=torch.float)
            # Cache the (motion_id, position-within-motion) coordinates and
            # ``max(sizes)`` used by ``_update_adaptive_bin_weights`` — they depend
            # only on ``batch_sizes`` / ``batch_starts`` (immutable after this
            # returns), so caching here removes per-call allocations and the
            # per-call ``sizes.max().item()`` GPU→CPU sync.
            sizes_t = self._motion_bin_weights.batch_sizes
            starts_t = self._motion_bin_weights.batch_starts
            n_motions = len(sizes_t)
            self._adaptive_motion_ids_flat = torch.repeat_interleave(torch.arange(n_motions, device=device), sizes_t)
            self._adaptive_pos_in_motion_flat = (
                torch.arange(total_bins, device=device) - starts_t[self._adaptive_motion_ids_flat]
            )
            self._adaptive_max_bins = int(sizes_t.max().item()) if n_motions > 0 else 0
            if term.adaptive_kernel_size > 1:
                self._adaptive_kernel = torch.tensor(
                    [term.adaptive_lambda**i for i in range(term.adaptive_kernel_size)],
                    device=device,
                )
                self._adaptive_kernel = self._adaptive_kernel / self._adaptive_kernel.sum()
                # Flat index of every motion's bins, right-padded with its own last bin to
                # ``max_bins + K - 1`` columns: the look-ahead window the smoothing correlates with.
                cols = torch.arange(self._adaptive_max_bins + term.adaptive_kernel_size - 1, device=device)
                self._adaptive_window_idx = starts_t[:, None] + torch.minimum(cols[None, :], sizes_t[:, None] - 1)

    @property
    def group_bin_mask(self) -> torch.Tensor:
        """``(G, total_bins)`` membership of every start bin in each per-env motion group (``env_motion_groups``).

        Built on first use from the term's per-rank ``(G, M)`` clip mask, and cached: the bin layout is fixed for this
        library, and the sampler is rebuilt whenever the library changes.
        """
        if self._group_bin_mask is None:
            term = self._term
            if term._group_motion_mask is None:
                raise RuntimeError("group_bin_mask is only defined for a motion term with env_motion_groups set.")
            sizes = self._motion_bin_weights.batch_sizes
            bin_motion = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), sizes)
            self._group_bin_mask = term._group_motion_mask[:, bin_motion.to(term._group_motion_mask.device)]
        return self._group_bin_mask

    @property
    def weights(self) -> ConcatBatchTensor:
        """Per-motion temporal bin probabilities (read by the term's start-time samplers)."""
        return self._motion_bin_weights

    def record_bin_outcomes(self, envs_idx: torch.Tensor, terminated_mask: torch.Tensor, timestamps: torch.Tensor):
        """Accumulate per-episode failure counts for resetting envs.

        Failures are keyed by the bin the episode **died in**
        (``floor((start_time + timestamps) / bin_length)``) and accumulated as
        raw counts in ``_current_bin_failed`` for the terminated subset only.
        A failure *rate* keyed by the start bin instead starves the easy early
        frames a cold policy needs to bootstrap on multi-clip LAFAN1 and
        collapses tracking; the death-bin failure count is the curriculum
        signal that lets it learn.

        ``terminated_mask`` / ``timestamps`` are aligned with ``envs_idx``.
        """
        term = self._term
        if not term.adaptive_sampling or self._motion_bin_weights is None or len(envs_idx) == 0:
            return
        assigned_ids = term.envs_idx_to_assigned_ids(envs_idx)
        motion_ids = term._assigned_env_motion_selection[assigned_ids]
        death_time = term._motion_buffer_start_time_s[assigned_ids] + timestamps
        bin_indices = torch.clamp(torch.floor(death_time / term.motion_bin_length_s).long(), min=0)
        max_bins = self._motion_bin_weights.batch_sizes[motion_ids] - 1
        bin_indices = torch.min(bin_indices, max_bins)
        flat_indices = self._motion_bin_weights.batch_starts[motion_ids] + bin_indices
        # Every resetting env's bin, weighted 1 if it terminated and 0 if not: the same counts as scattering
        # ones over the terminated subset (whole numbers, so the order of the adds cannot matter), without
        # the bool-mask gather that picked the subset, a GPU->CPU sync.
        self._current_bin_failed.scatter_add_(0, flat_indices, terminated_mask.to(self._current_bin_failed.dtype))

    def blend_failure_ema(self) -> None:
        """Blend ``_current_bin_failed`` into ``_bin_failure_ema``, then zero it."""
        term = self._term
        if not term.adaptive_sampling or self._bin_failure_ema is None:
            return
        self._bin_failure_ema = (
            term.adaptive_alpha * self._current_bin_failed + (1 - term.adaptive_alpha) * self._bin_failure_ema
        )
        self._current_bin_failed.zero_()

    def _update_adaptive_bin_weights(self) -> None:
        """Compute smoothed probabilities from the failure EMA, write to the bin weights.

        Hot path: called from ``_sample_env_motion_start_time`` on every reset.
        Vectorised into a fixed number of GPU kernel launches independent of
        ``n_motions`` (a per-motion Python loop is the per-step bottleneck at
        AMASS scale).
        """
        term = self._term
        if not term.adaptive_sampling or self._bin_failure_ema is None:
            return

        cbt = self._motion_bin_weights
        sizes = cbt.batch_sizes
        n_motions = len(sizes)
        if n_motions == 0:
            return

        # Cached per-bin coordinates and ``max(sizes)`` (immutable until the
        # library is reset) — skip the per-call allocations + ``sizes.max()`` sync.
        motion_ids = self._adaptive_motion_ids_flat
        pos_in_motion = self._adaptive_pos_in_motion_flat
        max_bins = self._adaptive_max_bins

        # Per-death-bin failure COUNT EMA with a uniform floor added before
        # normalization. The floor (``adaptive_uniform_ratio / num_bins``) keeps
        # every bin sampleable so the easy early frames a cold policy needs to
        # bootstrap never get starved (see ``record_bin_outcomes``).
        flat = self._bin_failure_ema.clone()
        inv_sizes = term.adaptive_uniform_ratio / sizes.float()
        flat = flat + inv_sizes[motion_ids]

        # Optional kernel smoothing, per motion: bin ``j`` takes ``sum_k w[k] * x[j + k]``, so a failure in
        # bin ``b`` lands on ``b, b - 1, ..., b - K + 1`` (heaviest on ``b``) -- the run-up to it. As in
        # BeyondMimic's ``_adaptive_sampling`` and ``MotionCommand``, each motion is padded on the right with
        # its own last bin (``replicate``), so a flat distribution stays flat.
        if term.adaptive_kernel_size > 1 and max_bins > 1:
            smoothed = torch.nn.functional.conv1d(
                flat[self._adaptive_window_idx].unsqueeze(1),
                self._adaptive_kernel.view(1, 1, -1),
            )
            flat = smoothed[:, 0][motion_ids, pos_in_motion]

        # Per-motion L1 normalization into a proper per-motion distribution.
        motion_sums = torch.zeros(n_motions, device=term.device, dtype=flat.dtype).scatter_add_(0, motion_ids, flat)
        normalized = flat / motion_sums[motion_ids].clamp_min(_NORMALIZE_EPS)
        # Zero-sum motions (``adaptive_uniform_ratio == 0`` + fully-decayed EMA)
        # would leave their bins all-zero, breaking the per-motion sum-to-1
        # contract and making them unsampleable by ``_sample_as_concat_motion_bins``;
        # substitute a uniform 1/sz distribution.
        uniform_per_bin = 1.0 / sizes.float()
        zero_motion_mask = motion_sums <= _NORMALIZE_EPS
        flat = torch.where(zero_motion_mask[motion_ids], uniform_per_bin[motion_ids], normalized)
        cbt.concatenated_tensor.copy_(flat)
