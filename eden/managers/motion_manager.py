"""Motion manager and MotionTerm base: motion-reference data for imitation tasks.

A :class:`MotionTerm` behaves like a PyTorch ``Dataset`` of motion clips (some generate
references on the fly), and the :class:`MotionManager` fills per-env reference buffers
each step for :class:`~eden.envs.base.MimicRLEnvBase`.

Under DDP, each rank is assigned only a subset of trajectories (``data_source`` and the
``_local_to_global`` map shrink per rank) while the motion-weight and outcome buffers
stay sized to the full motion-id space, so cross-rank ``all_reduce`` stays
shape-symmetric.
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Container, Iterator, Sequence, Literal

import eden as en
from functools import cached_property

import torch
import torch.distributed as dist
import numpy as np
from eden.utils.format import format_table

from eden.managers.base import FlatManagerBase
from eden.managers.motion import AdaptiveBinSampler, MotionEndResync, MotionOutcomeCurriculum, PerMotionMetrics
from eden.utils.motion_reference.motion_data import (
    MotionReferenceData,
    MotionReferenceState,
)
from eden.utils.motion_reference.augmentation_mixin import SymmetricAugmentationMixin
from eden.managers.base import ConfigurableTermBase
from eden.utils.assets import load_text
from eden.utils.misc import assign_over_envs, env_indices, is_env_mask, resolve_named_entity, sanitize_envs_idx
from eden.utils.registry import TermRegistry
from eden.utils.string import resolve_matching_names
from eden.options.managers.motions import MotionGroupOptions, MotionManagerOptions, MotionTermOptions
from eden.datasets.base import DATASET_REGISTRY

from eden.utils.buffers.cbt import ConcatBatchTensor

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.datasets.base import DatasetBase
    from eden.options.datasets import DatasetOptions


def sample_start_times(
    motion_ids: torch.Tensor,
    motion_bin_weights: ConcatBatchTensor,
    buffer_device: torch.device,
    motion_bin_length_s: float | None,
    data_source: "DatasetBase | None" = None,
    uniform_bins: bool = False,
) -> torch.Tensor:
    """Sample a start time (seconds) per env from the bin-weight distribution.

    ``uniform_bins=True`` is a fast path for when the bin weights are known to be
    uniform (i.e. ``adaptive_sampling`` is off, so the ``torch.ones`` init is never
    rewritten): a ``multinomial`` over uniform weights is *exactly* a uniform draw
    over each motion's valid bins, so we draw ``floor(rand * n_bins)`` directly and
    skip the padded ``get_with_mask`` gather + ``multinomial`` launch. This is
    bit-for-bit the same DISTRIBUTION as the weighted path (verified in
    ``tests/managers/test_motion_manager.py``); the weighted path would dominate
    the motion-end resync's per-step cost, which calls this on every clip wrap.

    When ``data_source`` is provided, results are clamped per-env to
    ``(buffer_length[motion_id] - 1) / framerate[motion_id]``. This is the
    correct behavior whenever any clip is shorter than ``motion_bin_length_s``
    — without the clamp, the (after-``clamp(min=1)``) single bin spans the
    full ``motion_bin_length_s`` window, so the intra-bin uniform noise can
    return a start time past the end of the clip. ``fill_init_reference_state``
    then reads ``base_quat_w`` at an out-of-range frame index, ``ConcatBatchTensor``
    zero-fills it to ``(0,0,0,0)``, and the first genesis quaternion op on that
    degenerate quat — the rotation math normalizes by ``q·q``, so a zero quat is a
    divide-by-zero — NaN-poisons the policy observation on the first reset.

    The sibling ``_sample_as_concat_motion_bins`` path applies the same clamp
    inline (see ``max_start_time_s`` below). Keeping ``data_source`` optional
    preserves the existing call signature for tests that build a CBT directly
    without a full ``DatasetBase``; production callers should pass it.
    """
    if motion_bin_length_s is None:
        raise ValueError("motion_bin_length_s must be provided")

    num_assign = len(motion_ids)
    if num_assign == 0:
        return torch.empty(0, dtype=torch.float32, device=buffer_device)

    if uniform_bins:
        # Uniform weights => sampled bin is uniform over each motion's [0, n_bins).
        # batch_sizes[motion_ids] is that per-env bin count; floor(rand * n) draws it
        # directly (== multinomial over uniform weights) without the CBT gather.
        n_bins = motion_bin_weights.batch_sizes[motion_ids].to(torch.float)
        sampled_bins = torch.floor(torch.rand(num_assign, device=buffer_device) * n_bins)
    else:
        # Batched padded access: (num_assign, max_bins) with zeros for out-of-range bins
        # Padded to the longest motion's bin count, known on the host: the longest *selected* one is a sync.
        padded_weights, valid_mask = motion_bin_weights.get_with_mask(motion_ids, pad_to_longest_batch=True)
        # Guard against all-zero rows (would crash multinomial)
        safe_weights = padded_weights + valid_mask.float() * 1e-8

        # Single batched multinomial: one sample per env
        sampled_bins = torch.multinomial(safe_weights, 1, replacement=True).squeeze(-1).float()

    # Intra-bin uniform noise + scale to seconds
    start_times = (sampled_bins + torch.rand(num_assign, device=buffer_device)) * motion_bin_length_s
    if data_source is not None:
        max_start_time_s = (data_source.buffer_length[motion_ids].to(torch.float) - 1) / data_source.framerate[
            motion_ids
        ].to(torch.float)
        start_times = torch.clamp(start_times, max=max_start_time_s)
    return start_times


# Motion fields copied frame-by-frame from a sequence into the per-env reference buffer.
_MOTION_COPY_FIELDS = (
    "dofs_pos",
    "dofs_vel",
    "links_pos_b",
    "links_quat_b",
    "links_pos_w",
    "links_quat_w",
    "links_lin_vel_b",
    "links_ang_vel_b",
    "links_lin_vel_w",
    "links_ang_vel_w",
    "base_pos_w",
    "base_lin_vel_w",
    "base_quat_w",
    "base_ang_vel_w",
    "contact_points_w",
    "contact_points_valid",
    "contact_label",
)
# Velocity fields zeroed at clamped (past-end) frames — see ``MotionTerm._finish_rows``.
_MOTION_VELOCITY_FIELDS = (
    "dofs_vel",
    "links_lin_vel_b",
    "links_ang_vel_b",
    "links_lin_vel_w",
    "links_ang_vel_w",
    "base_lin_vel_w",
    "base_ang_vel_w",
)
# World-frame position fields shifted to each env's origin — see ``MotionTerm._finish_rows``.
_MOTION_ORIGIN_FIELDS = ("base_pos_w", "links_pos_w", "contact_points_w")


def _fast_gather_fields(data) -> frozenset:
    """Motion fields of ``data`` eligible for the shared flat-index fast gather.

    A field qualifies when it is a :class:`ConcatBatchTensor` whose per-motion
    ``batch_sizes`` equal the sequence's canonical ``buffer_length`` — the tensor
    the frame clamp in :meth:`MotionTerm._select_frames` (and the start-time clamp
    in :func:`sample_start_times`) is computed against. That exact structural match
    is what makes a pre-clamped frame index a valid row offset inside the field's
    own storage; comparing total row counts alone would let a field whose totals
    coincide but whose per-clip splits differ be gathered from the wrong clip,
    silently. Non-qualifying fields (plain-tensor test doubles; a CBT rebuilt by a
    modifier that skipped it) keep the general masked ``__getitem__`` path, and each
    exclusion of a CBT field is logged once.

    A sequence containing an empty clip (``buffer_length == 0``) disables the fast
    path entirely: the frame clamp then yields row ``-1``, which only the masked
    path absorbs (zero-fill). ``DatasetBase`` forbids empty clips at load, but a
    custom source that bypasses it degrades gracefully instead of reading out of
    bounds.

    The result is computed once per sequence OBJECT and cached on it (a one-time,
    off-hot-path GPU sync): the mutating selection APIs (``select`` /
    ``select_dofs`` / ``select_links``) all return new ``BatchMotionSequence``
    instances, and dataset modifiers run at load time before the first gather, so
    the cache cannot go stale.
    """
    cached = getattr(data, "_eden_fast_gather_fields", None)
    if cached is not None:
        return cached
    fields = set()
    buffer_length = getattr(data, "buffer_length", None)
    if buffer_length is not None and bool((buffer_length > 0).all()):
        canonical = buffer_length.to(torch.int64)
        for field in _MOTION_COPY_FIELDS:
            src = getattr(data, field, None)
            if not isinstance(src, ConcatBatchTensor):
                continue
            if src.batch_sizes.shape == canonical.shape and bool((src.batch_sizes == canonical).all()):
                fields.add(field)
            else:
                en.logger.warning(
                    f"[motion gather] field '{field}' has a per-motion batch structure different from "
                    f"the sequence's buffer_length; it will use the general (masked) ConcatBatchTensor "
                    f"path instead of the fast gather."
                )
    result = frozenset(fields)
    data._eden_fast_gather_fields = result
    return result


class MotionTerm(ConfigurableTermBase[MotionTermOptions]):
    """Base class for motion terms, which serve a similar role to ``Dataset`` in PyTorch.

    Some of the motion terms may return the motion reference by generating it on the fly.

    Parameters
    ----------
    dataset_options: DatasetOptions | None
        The options for the dataset.
    entities_map: dict[str, str]
        The mapping from the entity name in the environment to the entity name in the dataset.
        - key: entity name in the environment
        - value: motion sequence key in the dataset

    target_dofs_name: dict[str, str | list[str]]
        The names of the degrees of freedom to target.
    target_links_name: dict[str, str | list[str]]
        The names of the links to target.
    env_motion_groups : MotionGroupOptions | None
        Per-env clip groups keyed on a heterogeneous entity's variant; ``None`` samples every env from the whole
        library. See :class:`~eden.options.managers.motions.MotionGroupOptions`.
    """

    dataset_options: DatasetOptions | None = None
    entities_map: dict[str, str] = {}
    target_dofs_name: dict[str, str | list[str]] = {}
    target_links_name: dict[str, str | list[str]] = {}
    motion_weights: str | list[float] | None = None
    env_motion_groups: MotionGroupOptions | None = None
    env_starting_stub_sampling_strategy: Literal["independent", "concat_motion_bins"] = "independent"
    motion_start_from_middle_range: tuple[float, float] | list[float] = (0.0, 0.0)
    motion_bin_length_s: float | None = None
    adaptive_sampling: bool = False
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_failure_rate_max_over_mean: float = 10.0
    adaptive_motion_weights: bool = False
    motion_weight_success_discount: float = 0.999
    motion_weight_failure_amplify: float = 2.0
    motion_weight_floor: float = 1e-3
    motion_weight_update_mode: Literal["compound", "snapshot"] = "snapshot"
    motion_weight_uniform_mix: float = 0.0
    motion_weight_failure_rate_max_over_mean: float = 200.0
    motion_weight_update_every_n_iters: int = 1
    start_at_timestep_zero_prob: float = 0.0
    freeze_at_timestep_zero_prob: float = 0.0
    compute_per_motion_metrics: bool = False
    metric_anchor_link_name: str | None = None
    log_per_motion_metrics: bool = False
    log_motion_id_to_filename: bool = False

    if TYPE_CHECKING:
        data_source: DatasetBase

    # Optional collaborators (``eden.managers.motion``), assigned in
    # __init__ / set_envs_idx_assignments / enable_trajectories. Declared as
    # class-level None defaults so instances that bypass __init__ (lightweight
    # test doubles) still satisfy the manager's ``term._curriculum`` /
    # ``term._metrics`` / ``term._bin_sampler`` guards.
    _curriculum: MotionOutcomeCurriculum | None = None
    _metrics: PerMotionMetrics | None = None
    _bin_sampler: AdaptiveBinSampler | None = None

    def __init__(
        self,
        env: EnvBase,
        options: MotionTermOptions,
    ) -> None:
        super().__init__(env=env, options=options)

        assert options.dataset_options is not None, "dataset_options is required for motion terms."
        # NOTE: setup dataset
        self.data_source = DATASET_REGISTRY.get(options.dataset_options.name)(
            options.dataset_options,
            device=self.device,
        )

        for motion_sequence_key in self.entities_map.values():
            if motion_sequence_key not in self.data_source.motion_sequence_keys:
                raise ValueError(
                    f"Motion sequence key {motion_sequence_key} not found in the dataset. Available keys: {self.data_source.motion_sequence_keys}"
                )

        # NOTE: resolve the indices of the motion sequence keys
        target_dofs_idx: dict[str, list[int]] = {}
        target_links_idx: dict[str, list[int]] = {}
        for key in self.data_source.motion_sequence_keys:
            indices, names = resolve_matching_names(
                self.target_dofs_name[key],
                self.data_source.motion_dofs_name[key],
                preserve_order=True,
            )
            self.target_dofs_name[key] = names
            # Reorder on any non-identity selection, not only a subset: a full-length permuted target list
            # must permute the columns too, or every name lookup reads the wrong DOF.
            if indices != list(range(len(self.data_source.motion_dofs_name[key]))):
                target_dofs_idx[key] = indices
            indices, names = resolve_matching_names(
                self.target_links_name[key],
                self.data_source.motion_links_name[key],
                preserve_order=True,
            )
            self.target_links_name[key] = names
            if indices != list(range(len(self.data_source.motion_links_name[key]))):
                target_links_idx[key] = indices

        self.data_source.select_dofs(target_dofs_idx)
        self.data_source.select_links(target_links_idx)

        motion_weights_source: str | None = None
        if isinstance(self.motion_weights, str):
            motion_weights_source = self.motion_weights
            self.motion_weights = load_text(motion_weights_source).get("motion_weights", None)
            if self.motion_weights is None:
                raise ValueError(f"`motion_weights` key not found in {motion_weights_source}")
        if self.motion_weights is not None:
            if len(self.motion_weights) != len(self.data_source):
                # _motion_weights and the per-motion outcome buffers index a
                # shared GLOBAL motion-id space; a length mismatch with the
                # dataset corrupts every later read. Catch it at construction
                # rather than on the first sampling call.
                src = f" (loaded from {motion_weights_source!r})" if motion_weights_source else ""
                raise ValueError(
                    f"`motion_weights`{src} has length {len(self.motion_weights)}, but the "
                    f"dataset has {len(self.data_source)} trajectories. Each entry in "
                    f"`motion_weights` is the initial sampling weight for one motion id; "
                    f"supply exactly one weight per loaded trajectory."
                )
            self._motion_weights = torch.tensor(
                self.motion_weights,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self._motion_weights = torch.ones(
                len(self.data_source),
                device=self.device,
                dtype=torch.float,
            )

        # Pin the global motion-id space at construction time, before any per-rank
        # subsetting via enable_trajectories(). _motion_weights and the curriculum
        # outcome buffers are kept at this size on every rank — that is what lets
        # the iter-end dist.all_reduce shape-match across ranks even when each
        # rank only loads its assigned trajectory slice into VRAM.
        self._global_num_motions: int = len(self.data_source)
        self._local_to_global: torch.Tensor = torch.arange(
            self._global_num_motions, device=self.device, dtype=torch.long
        )

        # Per-env clip groups (``env_motion_groups``): each clip's group id in the GLOBAL motion-id space, resolved
        # here from the dataset's per-clip labels; the per-rank mask and the per-env group follow in
        # ``enable_trajectories`` / ``set_envs_idx_assignments``. All three stay ``None`` when the option is unset.
        self._global_motion_group: torch.Tensor | None = None
        self._group_motion_mask: torch.Tensor | None = None
        self._env_group: torch.Tensor | None = None
        if self.env_motion_groups is not None:
            self._global_motion_group = self._resolve_global_motion_groups(self.env_motion_groups)

        # Temporal start-time bin weights + adaptive curriculum live in an
        # AdaptiveBinSampler collaborator, (re)created in enable_trajectories when
        # motion_bin_length_s is set (class-level _bin_sampler=None default until then).
        if self.adaptive_sampling and self.motion_bin_length_s is None:
            raise ValueError("adaptive_sampling=True requires motion_bin_length_s to be set.")
        if self.compute_per_motion_metrics and not self.adaptive_motion_weights:
            # Per-motion metric drain piggybacks on the curriculum's DDP
            # cadence and ``all_reduce_fn`` to share one synced collective
            # per PPO iter. Without the curriculum nothing would drain the
            # metric buffers, so fail loudly rather than let them grow.
            raise ValueError(
                "compute_per_motion_metrics=True requires adaptive_motion_weights=True; "
                "the metric drain shares the curriculum's DDP cadence."
            )

        # Per-motion-id outcome curriculum (hard-motion mining), orthogonal to
        # the bin-level adaptive_sampling. Lives in MotionOutcomeCurriculum,
        # which owns the success/failure counts + summary caches and rewrites
        # ``_motion_weights`` from drained outcome rates. ``None`` when the
        # curriculum is off, so disabled terms pay nothing.
        self._curriculum: MotionOutcomeCurriculum | None = (
            MotionOutcomeCurriculum(self) if self.adaptive_motion_weights else None
        )

        # Per-motion tracking-error metrics live in a PerMotionMetrics
        # collaborator, created in ``set_envs_idx_assignments`` once
        # ``num_assigned_envs`` is known (and only when ``compute_per_motion_metrics``
        # is set, so disabled terms pay nothing). ``None`` until then / when disabled.
        self._metrics: PerMotionMetrics | None = None

    @cached_property
    def num_trajectories(self) -> int:
        """Return the number of trajectories in the motion buffer.

        For a generative motion buffer, this just returns a placeholder number.
        """
        return len(self.data_source)

    @property
    def num_assigned_envs(self) -> int:
        """The number of envs assigned to this motion buffer."""
        return self.assigned_env_slice.stop - self.assigned_env_slice.start

    def set_envs_idx_assignments(self, envs_idx: slice) -> None:
        """Assign the environment indices this motion buffer is responsible for.

        The motion buffer is expected to fill the motion reference data for the assigned envs only.

        Parameters
        ----------
        envs_idx : slice
            The environment IDs to assign to this motion buffer.
        """
        assert envs_idx.step is None, "Currently only support continuous env IDs."
        self.assigned_env_slice = envs_idx

        self._assigned_env_motion_selection = torch.zeros(
            self.num_assigned_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._motion_buffer_start_time_s = torch.zeros(
            self.num_assigned_envs,
            dtype=torch.float,
            device=self.device,
        )

        assigned_ids = self.assigned_ids_from_envs_idx(None)
        if self.env_starting_stub_sampling_strategy == "concat_motion_bins" and self.motion_bin_length_s is None:
            raise ValueError(
                "env_starting_stub_sampling_strategy='concat_motion_bins' requires motion_bin_length_s "
                "to be set so that motion bin weights are initialized. Got motion_bin_length_s=None."
            )
        if self._global_motion_group is not None:
            self._env_group = self._resolve_env_groups(self.env_motion_groups)
        if self.env_starting_stub_sampling_strategy == "independent":
            self._sample_assigned_env_motion_selection(assigned_ids)
            self._sample_env_motion_start_time(assigned_ids)
        elif self.env_starting_stub_sampling_strategy == "concat_motion_bins":
            self._sample_as_concat_motion_bins(assigned_ids)

        # Allocate the per-motion metric collaborator now that num_assigned_envs
        # is known. Gated on the term option so disabled terms pay nothing.
        if self.compute_per_motion_metrics:
            self._metrics = PerMotionMetrics(self)

    @property
    def complete_motion_lengths(self) -> torch.Tensor:
        """The full motion reference lengths for each assigned env. the unit is in seconds."""
        return (
            self.data_source.buffer_length[self._assigned_env_motion_selection]
            / self.data_source.framerate[self._assigned_env_motion_selection]
        )

    @property
    def assigned_motion_lengths(self) -> torch.Tensor:
        """The motion reference lengths for each assigned env. the unit is in seconds."""
        return self.complete_motion_lengths - self._motion_buffer_start_time_s

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        """Enable only the selected motion reference trajectories.

        Assuming all motion reference trajectories are stored in a sequential way, the motion manager will call this
        function to enable only the selected trajectories, depending on how multi-processing is implemented.
        By "enabled", it means the motion buffer will generate the motion data for only the enabled trajectories.

        ``_motion_weights`` and the per-motion outcome buffers stay sized to the
        full pre-subset motion-id space on every rank — only ``data_source`` and
        the new ``_local_to_global`` map shrink. This is what lets DDP
        ``all_reduce`` stay shape-symmetric while each rank holds only its
        assigned trajectory slice in VRAM.

        Parameters
        ----------
        traj_ids : torch.Tensor | slice | None, optional
            The trajectory IDs to enable. start from 0 to `self.num_trajectories - 1` in this motion buffer.
            If None, enable all trajectories.
            Out-of-range values are ignored, so no motion is enabled if every id is out of range.
        """
        if traj_ids is not None:
            # ``traj_ids`` are LOCAL ids relative to the term's *current*
            # ``num_trajectories`` (per the docstring). Compose them with the
            # existing ``_local_to_global`` BEFORE we shrink ``data_source`` so
            # repeated calls remain correct (e.g. global=[3, 7, 9] then a
            # local=[0, 2] subset must produce global=[3, 9], not [0, 2]).
            if isinstance(traj_ids, slice):
                start, stop, step = traj_ids.indices(len(self._local_to_global))
                local_idx = torch.arange(start, stop, step, device=self.device, dtype=torch.long)
            else:
                local_idx = torch.as_tensor(traj_ids, device=self.device, dtype=torch.long)
            self._local_to_global = self._local_to_global[local_idx]
            self.data_source.select_trajectories(traj_ids)
            # invalidate cached num_trajectories since the dataset has been subsetted
            self.__dict__.pop("num_trajectories", None)
        # (Re)create the bin sampler from the now-current (possibly subsetted)
        # data_source; its weights are per-motion so they must track the library.
        self._bin_sampler = AdaptiveBinSampler(self) if self.motion_bin_length_s is not None else None
        if self._global_motion_group is not None:
            # (G, M_local): which of this rank's clips each group may draw. Built once per library change, so the
            # per-reset draw is a single masked multinomial with no host sync.
            local_group = self._global_motion_group[self._local_to_global]
            num_groups = len(self._group_names)
            self._group_motion_mask = local_group.unsqueeze(0) == torch.arange(
                num_groups, device=self.device
            ).unsqueeze(1)
        # Outcome buffers are sized to the GLOBAL motion-id space and allocated
        # once in ``__init__`` — no re-init on subset, since the global shape
        # is invariant under per-rank trajectory selection.

    def rewind_to_start(self, envs_idx: torch.Tensor) -> None:
        """Rewind playback to frame 0 of each env's currently-assigned clip (no resampling).

        Zeros this term's per-env start offset. The per-term half of
        :meth:`MotionManager.rewind_to_start`; public so callers needn't poke
        ``_motion_buffer_start_time_s`` (allocated in :meth:`set_envs_idx_assignments` at
        manager construction, so it always exists by the time playback is rewound).
        """
        self._motion_buffer_start_time_s[envs_idx] = 0.0

    def reset(
        self,
        envs_idx: torch.Tensor | None,
        symmetric_augmentation_mask_buffer: torch.Tensor | None = None,
    ) -> None:
        """
        Reset the motion buffer for the specified envs_idx.

        Parameters
        ----------
        envs_idx : torch.Tensor | None
            The environment IDs to reset the motion buffer, which is among the entire num_envs.
            If None, reset all buffers.
        symmetric_augmentation_mask_buffer : torch.Tensor | None
            The buffer to store the symmetric augmentation mask for each env. The
            motion buffer is expected to sample and fill the symmetric augmentation mask for each env. Because some
            of the motions do or do not support symmetric augmentation.
            NOTE: the motion buffer does not apply the symmetric augmentation; it only updates the mask.
        """
        self._sample_assigned_env_starting_stub(envs_idx)

        if symmetric_augmentation_mask_buffer is not None:
            if envs_idx is not None and not isinstance(envs_idx, torch.Tensor):
                envs_idx = torch.as_tensor(envs_idx, device=self.device)
            if is_env_mask(envs_idx):
                # Over-sample full buffer, mask-write to avoid GPU→CPU sync
                rand_mask = torch.rand(len(symmetric_augmentation_mask_buffer), device=self.device) < 0.5
                symmetric_augmentation_mask_buffer[envs_idx] = rand_mask[envs_idx]
            else:
                num_envs = self.num_assigned_envs if envs_idx is None else len(envs_idx)
                symmetric_augmentation_mask_buffer[envs_idx] = torch.rand(num_envs, device=self.device) < 0.5

    def fill_init_reference_state(
        self,
        motion_sequence_key: str,
        envs_idx: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor | None,
        state_buffer: MotionReferenceState,
    ) -> None:
        """Fill the initial reference state into the buffer for the specified envs only.

        Parameters
        ----------
        motion_sequence_key : str
            The motion sequence key to fill the initial reference state from.
        envs_idx : Sequence[int] | torch.Tensor
            The environment IDs to fill the initial reference state.
        env_origins : torch.Tensor | None
            The origins for each robot/env, which is also affecting the motion reference data, shape (num_envs, 3).
            None: every env's origin is the world origin.
        state_buffer : MotionReferenceState
            The MotionReferenceState to fill the initial reference state, which is a buffer for all envs,
            shape (num_envs, ...).
        """
        assigned_ids = self.envs_idx_to_assigned_ids(envs_idx)
        data = self.data_source.motion_sequences[motion_sequence_key]
        motion_ids = self._assigned_env_motion_selection[assigned_ids]

        # Sampled start times are clamped to (buffer_length - 1) / framerate (see
        # ``sample_start_times`` / ``_sample_as_concat_motion_bins``), so the frame
        # index is in-range by construction; the explicit min() guards the float
        # round-trip of that clamp (start * framerate landing exactly on
        # buffer_length), holding the last frame instead of reading past the clip —
        # which the masked CBT path would zero-fill: the zero-quat NaN poison
        # described in ``sample_start_times``.
        frame_selection = torch.floor(self._motion_buffer_start_time_s[assigned_ids] * data.framerate[motion_ids]).to(
            torch.long
        )
        frame_selection = torch.minimum(frame_selection, (data.buffer_length[motion_ids] - 1).to(torch.long))

        # Same shared flat-index gather as ``_copy_frame_data``: the reset
        # path pays the identical per-field ConcatBatchTensor overhead otherwise.
        fast_fields = _fast_gather_fields(data)
        flat_rows: torch.Tensor | None = None

        def _gather(src, field: str) -> torch.Tensor:
            nonlocal flat_rows
            if field in fast_fields:
                if flat_rows is None:
                    flat_rows = src.batch_starts[motion_ids] + frame_selection
                return src.concatenated_tensor.index_select(0, flat_rows)
            return src[motion_ids, frame_selection]

        for field in (
            "base_quat_w",
            "base_lin_vel_w",
            "base_ang_vel_w",
            "dofs_pos",
            "dofs_vel",
        ):
            src = getattr(data, field, None)
            dst = getattr(state_buffer, field, None)
            if src is None or dst is None:
                continue
            dst[envs_idx] = _gather(src, field)

        if data.base_pos_w is not None and state_buffer.base_pos_w is not None:
            base_pos_w = _gather(data.base_pos_w, "base_pos_w")
            # NOTE: move the base position to the origin of the motion
            if env_origins is not None:
                base_pos_w = base_pos_w + env_origins[envs_idx]
            # TODO: check if any link is below the ground
            state_buffer.base_pos_w[envs_idx] = base_pos_w

    def compute(
        self,
        motion_sequence_key: str,
        envs_idx: slice | Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor | None,
        data_buffer: MotionReferenceData,
    ) -> None:
        """Fill the motion data into the buffer for the specified envs only.

        This should not modify any other data in the MotionReferenceData.

        Parameters
        ----------
        motion_sequence_key : str
            The motion sequence key to sample the motion data from.
        envs_idx : slice | Sequence[int] | torch.Tensor
            The environment IDs to fill the motion data. A slice (the per-step fill of every env) is
            gathered straight into the buffer, with no intermediate copy.
        sample_timestamp : torch.Tensor
            The time (s) to sample from the start of each env, shape (num_envs_idx, num_frames).
        env_origins : torch.Tensor | None
            The origins for each robot/env, which is also affecting the motion reference data, shape (num_envs, 3).
            None: every env's origin is the world origin.
        data_buffer : MotionReferenceData
            The MotionReferenceData to fill the motion data, which is a buffer for all envs,
            shape (num_envs, num_frames, ...).
        """
        assigned_ids = self.envs_idx_to_assigned_ids(envs_idx)
        data = self.data_source.motion_sequences[motion_sequence_key]

        motion_ids, frame_idx, validity = self._select_frames(
            assigned_ids, sample_timestamp, data, data_buffer, envs_idx
        )
        origin_offset = None if env_origins is None else env_origins[envs_idx]
        self._copy_frame_data(data, data_buffer, envs_idx, motion_ids, frame_idx, validity, origin_offset)

    def _select_frames(
        self,
        assigned_ids: torch.Tensor,
        sample_timestamp: torch.Tensor,
        data,
        data_buffer: MotionReferenceData,
        envs_idx: Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve the per-frame source indices and set ``data_buffer.validity[envs_idx]``.

        Frames past the end of a clip are marked invalid and clamped to the last frame
        (their velocities are zeroed later, in :meth:`_finish_rows`). Returns
        the per-env motion ids ``(len(envs_idx),)``, the clamped frame indices
        ``(len(envs_idx), num_frames)``, and the validity mask of the same shape —
        already-computed tensors the downstream steps would otherwise re-gather
        (this runs on the per-step hot path).
        """
        # Use the single-source frame/validity helper so the per-step fill and the
        # motion-end resync's wrap detection can never drift; return the 3-tuple
        # (motion_ids, clamped frame_idx, validity) the fast-gather path in
        # _copy_frame_data / _finish_rows consumes.
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        frame_selections, validity = self._frame_index_and_validity(assigned_ids, sample_timestamp, data)
        data_buffer.validity[envs_idx] = validity
        last_frame = data.buffer_length[motion_ids].unsqueeze(-1) - 1
        frame_selections = torch.where(validity, frame_selections, last_frame).to(torch.long)
        return motion_ids, frame_selections, validity

    def _frame_index_and_validity(
        self, assigned_ids: torch.Tensor, sample_timestamp: torch.Tensor, data
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single source of truth for the clip frame index + validity of a set of
        assigned envs at ``sample_timestamp`` (shape ``[len(assigned_ids), num_frames]``):

            frame = floor((start + sample_timestamp) * framerate)
            valid = frame < buffer_length              (i.e. not past the clip end)

        Used both by :meth:`_select_frames` (the per-step fill) and by the motion-end
        resync's wrap detection (:meth:`eden.managers.motion.resync.MotionEndResync._wrapped_mask`,
        via :meth:`_past_clip_end`), so the two can never drift out of sync. Returns
        ``(frame_float, valid_bool)`` — the caller clamps / uses index 0 as needed.
        """
        motion_sel = self._assigned_env_motion_selection[assigned_ids]
        # NOTE: this product expression is mirrored bit-for-bit in :meth:`_past_clip_end`;
        # keep them identical so the resync wrap check matches this validity exactly.
        prod = (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp) * data.framerate[
            motion_sel
        ].unsqueeze(-1)
        # valid == floor(prod) < L == prod < L for integer L (buffer_length), so compare
        # the un-floored product directly — bit-identical to floor-then-compare.
        valid = (prod < data.buffer_length[motion_sel].unsqueeze(-1)).to(torch.bool)
        return torch.floor(prod), valid

    def _past_clip_end(self, assigned_ids: torch.Tensor, sample_timestamp: torch.Tensor, data) -> torch.Tensor:
        """Bool mask: reference frame(s) past the clip end. Bit-exact to
        ``~_frame_index_and_validity(...).valid`` (same product; ``floor(prod) >= L`` iff
        ``prod >= L`` for integer ``L``), but skips the ``floor`` + bool-cast the wrap
        check doesn't need. Drives the motion-end resync's per-step wrap detection. The
        product below MUST stay identical to the one in :meth:`_frame_index_and_validity`.
        """
        motion_sel = self._assigned_env_motion_selection[assigned_ids]
        prod = (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp) * data.framerate[
            motion_sel
        ].unsqueeze(-1)
        return prod >= data.buffer_length[motion_sel].unsqueeze(-1)

    def _copy_frame_data(
        self,
        data,
        data_buffer: MotionReferenceData,
        envs_idx: slice | Sequence[int] | torch.Tensor,
        motion_ids: torch.Tensor,
        frame_idx: torch.Tensor,
        validity: torch.Tensor,
        origin_offset: torch.Tensor | None,
    ) -> None:
        """Gather each motion field at the selected ``(motion, frame)`` pairs into the buffer.

        ``frame_idx`` arrives pre-clamped in-range by :meth:`_select_frames`, so for
        every field that shares the sequence's canonical batch structure (see
        :func:`_fast_gather_fields`) the ``(motion, frame)`` → concatenated-row index
        is computed ONCE and the field reduces to a single ``index_select`` on the
        raw concatenated storage. Routing through ``ConcatBatchTensor.__getitem__``
        instead re-derives the flat index and re-applies validity masking per field
        (~a dozen extra CUDA kernels each, including a full-size masking multiply),
        which at 14+ fields per step would dominate the ``fill_data`` hot path.
        Non-qualifying fields keep the general ``__getitem__`` path.

        Each gathered field is finished before it is written (:meth:`_finish_rows`), on the selected
        rows only: a reset of a few envs must not touch every env's rows. A slice ``envs_idx``
        (the per-step fill of every env) is a contiguous view of the buffer, which a fast field is
        gathered straight into.
        """
        fast_fields = _fast_gather_fields(data)
        flat_rows: torch.Tensor | None = None
        flat_pair: tuple[torch.Tensor, torch.Tensor] | None = None
        clamped = ~validity
        for field in _MOTION_COPY_FIELDS:
            src = getattr(data, field, None)
            dst = getattr(data_buffer, field, None)
            if src is None or dst is None:
                continue
            # Zero-size fields (e.g. dofs_pos of a DOF-less rigid object) have
            # nothing to copy, and the -1 in the reshape below would be
            # ambiguous for a 0-element tensor.
            if dst.numel() == 0:
                continue
            if field in fast_fields:
                if flat_rows is None:
                    # All fast fields share batch_starts (batch_sizes == buffer_length),
                    # so the first one's offsets serve every field. Broadcast over the
                    # frame dim: one (num_envs,)-sized gather, no N*F index tensors.
                    flat_rows = (src.batch_starts[motion_ids].unsqueeze(-1) + frame_idx).reshape(-1)
                storage = src.concatenated_tensor
                rows = dst[envs_idx] if isinstance(envs_idx, slice) else None
                if (
                    rows is not None
                    and rows.is_contiguous()
                    and rows.dtype == storage.dtype
                    and rows.shape[2:] == storage.shape[1:]
                ):
                    # Straight into the buffer: no gathered copy to write back.
                    torch.index_select(storage, 0, flat_rows, out=rows.view(-1, *rows.shape[2:]))
                    self._finish_rows(field, rows, clamped, origin_offset)
                    continue
                selected = storage.index_select(0, flat_rows)
            else:
                if flat_pair is None:
                    flat_pair = (
                        motion_ids.unsqueeze(-1).expand_as(frame_idx).reshape(-1),
                        frame_idx.reshape(-1),
                    )
                # Tensor indices gather a copy, so the in-place finish below never reaches the source.
                selected = src[flat_pair[0], flat_pair[1]]
            selected = selected.reshape(-1, *dst.shape[1:])
            self._finish_rows(field, selected, clamped, origin_offset)
            dst[envs_idx] = selected

    @staticmethod
    def _finish_rows(field: str, rows: torch.Tensor, clamped: torch.Tensor, origin_offset: torch.Tensor | None) -> None:
        """Finish ``field``'s gathered ``rows`` ``(num_envs_idx, num_frames, ...)`` in place.

        * A velocity is zeroed at a ``clamped`` (past-end) frame, ``(num_envs_idx, num_frames)``. The index
          clamp in :meth:`_select_frames` holds position at the last valid frame, and copying that frame's
          (generally non-zero) velocity through would give a physically inconsistent signal (frozen pos +
          non-zero vel). Zero matches the "frozen at last pose, at rest" reading of the reward-side validity
          mask. No ``.any()`` short-circuit: it would cost a device sync, and the fill is a no-op on an
          all-valid mask.
        * A world-frame position moves to the env's origin (``origin_offset``, ``(num_envs_idx, 3)``; None
          leaves it at the world origin). A contact point's invalid rows become zeros + origin, which every
          consumer masks by ``contact_points_valid``.
        """
        if field in _MOTION_VELOCITY_FIELDS:
            rows.masked_fill_(clamped.view(*clamped.shape, *([1] * (rows.ndim - clamped.ndim))), 0.0)
        elif field in _MOTION_ORIGIN_FIELDS and origin_offset is not None:
            rows += origin_offset.view(len(origin_offset), *([1] * (rows.ndim - 2)), 3)

    def assigned_ids_from_envs_idx(
        self,
        envs_idx: slice | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Local int ids in ``[0, num_assigned_envs)`` for an *int-like* env selector.

        Accepts ``None`` (all assigned envs), a slice, or a long/int tensor (or
        ``Sequence[int]``) of **global** env ids. A bool mask is **not** accepted
        here — its assigned-space form is a bool mask, not int ids; use
        :meth:`assigned_mask_from_envs_mask`, or :meth:`envs_idx_to_assigned_ids`
        when the selector dtype is only known at runtime.
        """
        if envs_idx is None:
            return torch.arange(self.num_assigned_envs, device=self.device)
        if isinstance(envs_idx, slice):
            start = envs_idx.start if envs_idx.start is not None else self.assigned_env_slice.start
            stop = envs_idx.stop if envs_idx.stop is not None else self.assigned_env_slice.stop
            step = envs_idx.step if envs_idx.step is not None else 1
            indices = torch.arange(start, stop, step, device=self.device, dtype=torch.long)
            return indices - self.assigned_env_slice.start
        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = torch.tensor(envs_idx, device=self.device, dtype=torch.long)
        result = envs_idx - self.assigned_env_slice.start
        # Reading the check back is a GPU->CPU sync, and this runs on every step's reset and fill.
        # A buffer that starts at env 0 can only go below it on a negative id, which no in-tree
        # caller makes; device ids are trusted there, as `sanitize_envs_idx` trusts them.
        if self.assigned_env_slice.start > 0 or result.device.type == "cpu":
            assert (result >= 0).all(), (
                f"envs_idx contains indices below assigned_env_slice.start ({self.assigned_env_slice.start})"
            )
        return result

    def assigned_mask_from_envs_mask(self, envs_mask: torch.Tensor) -> torch.Tensor:
        """Local bool mask over ``[0, num_assigned_envs)`` for a *global* bool mask.

        The assigned-space form is the global mask sliced to the assigned env
        range, kept as a bool mask (not int ids) so downstream resampling can
        over-sample the full buffer and mask-write without a GPU→CPU sync.
        """
        return envs_mask[self.assigned_env_slice.start : self.assigned_env_slice.stop]

    def envs_idx_to_assigned_ids(
        self,
        envs_idx: slice | Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch a *runtime-typed* env selector to its assigned-space form.

        Returns a **local bool mask** when ``envs_idx`` is a bool tensor, otherwise
        **local int ids**. This dual return shape is load-bearing: a bool selector
        stays a bool mask so callers can mask-write (no GPU→CPU sync), while
        int/slice/``None`` selectors become int ids that index local buffers
        directly. When the selector dtype is statically known, call
        :meth:`assigned_ids_from_envs_idx` or :meth:`assigned_mask_from_envs_mask`
        directly instead of going through this dispatcher.
        """
        if is_env_mask(envs_idx):
            return self.assigned_mask_from_envs_mask(envs_idx)
        return self.assigned_ids_from_envs_idx(envs_idx)

    def envs_idx_is_assigned(self, envs_idx: slice | torch.Tensor) -> torch.Tensor:
        """
        Check if the envs_idx are assigned to this motion buffer.

        Parameters
        ----------
        envs_idx : torch.Tensor
            The environment IDs to check.

        Returns
        -------
        torch.Tensor
            A boolean tensor indicating assignment.
        """
        return (envs_idx >= self.assigned_env_slice.start) & (envs_idx < self.assigned_env_slice.stop)

    def get_current_motion_weights(self, envs_idx: slice | torch.Tensor | None = None) -> torch.Tensor:
        """Get the current motion weights for the specified envs.

        The motion weights are used to determine the probability of selecting a motion reference for each env.

        ``_assigned_env_motion_selection`` holds local motion ids; the
        canonical ``_motion_weights`` is global-sized, so we go through
        ``_local_to_global`` to index correctly.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor | None, optional
            The environment IDs (or bool mask) to get weights for; None for every assigned env.
        """
        if envs_idx is not None:
            assigned_ids = self.envs_idx_to_assigned_ids(envs_idx)
            local_ids = self._assigned_env_motion_selection[assigned_ids]
        else:
            local_ids = self._assigned_env_motion_selection
        return self._motion_weights[self._local_to_global[local_ids]]

    def match_scene(self, entity_name: str):
        """Let the motion buffer match the scene.

        For example a scene-interactive motion may need to match the motion starting point to the terrain.
        Recommended to call `scene.terrain.terrain_origins` and `scene.terrain.subterrain_specific_cfgs`
        to get some structured terrain information.

        Parameters
        ----------
        entity_name : str
            The name of the entity whose motion should be matched to the scene.
        """
        pass

    def _sample_env_motion_start_time(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Sample the start time for the assigned envs (at reset).

        This samples the start time for the assigned envs, which is independent from the motion selection.

        ``assigned_ids`` may be an int-index tensor or a bool mask.  For bool
        masks we over-sample the full buffer and mask-write.
        """
        assert assigned_ids is not None  # type: ignore
        _is_bool = is_env_mask(assigned_ids)
        n = self.num_assigned_envs if _is_bool else len(assigned_ids)

        if self._bin_sampler is not None:
            if not hasattr(self, "_warned_bin_vs_middle") and self.motion_start_from_middle_range[1] > 0.0:
                self._warned_bin_vs_middle = True
                en.logger.warning(
                    "Both motion_bin_weights and motion_start_from_middle_range are configured. "
                    "motion_bin_weights takes priority; motion_start_from_middle_range is ignored."
                )
            self._bin_sampler._update_adaptive_bin_weights()
            # adaptive off => bin weights are static-uniform, so sample uniform bins
            # directly (skips the per-call get_with_mask gather + multinomial; same
            # distribution). This is the motion-end resync's per-step hot path.
            uniform_bins = not self.adaptive_sampling
            if _is_bool:
                motion_ids = self._assigned_env_motion_selection  # full buffer
                start_times = sample_start_times(
                    motion_ids,
                    self._bin_sampler.weights,
                    self.device,
                    self.motion_bin_length_s,
                    data_source=self.data_source,
                    uniform_bins=uniform_bins,
                )
                # Through the mask with `torch.where`: a mask gather + write is a GPU->CPU sync each.
                assign_over_envs(self._motion_buffer_start_time_s, assigned_ids, start_times)
            else:
                motion_ids = self._assigned_env_motion_selection[assigned_ids]
                self._motion_buffer_start_time_s[assigned_ids] = sample_start_times(
                    motion_ids,
                    self._bin_sampler.weights,
                    self.device,
                    self.motion_bin_length_s,
                    data_source=self.data_source,
                    uniform_bins=uniform_bins,
                )
        elif self.motion_start_from_middle_range[1] > 0.0:
            assert (
                self.motion_start_from_middle_range[0] >= 0
                and self.motion_start_from_middle_range[1] >= self.motion_start_from_middle_range[0]
            ), (
                "motion_start_from_middle_range should be non-negative and the second value should be larger than the"
                " first one."
            )
            random_start_time = (
                torch.rand(n, device=self.device)
                * (self.motion_start_from_middle_range[1] - self.motion_start_from_middle_range[0])
                + self.motion_start_from_middle_range[0]
            )
            if _is_bool:
                sel = self._assigned_env_motion_selection
            else:
                sel = self._assigned_env_motion_selection[assigned_ids]
            random_start_time *= self.data_source.buffer_length[sel].to(torch.float)
            random_start_time /= self.data_source.framerate[sel].to(torch.float)
            if _is_bool:
                # Through the mask with `torch.where`: a mask gather + write is a GPU->CPU sync each.
                assign_over_envs(self._motion_buffer_start_time_s, assigned_ids, random_start_time)
            else:
                self._motion_buffer_start_time_s[assigned_ids] = random_start_time

    def _sample_as_concat_motion_bins(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Sample motion id and start time for the assigned envs as if all motions were concatenated into one.

        ``assigned_ids`` may be an int-index tensor or a bool mask.
        """
        _is_bool = is_env_mask(assigned_ids)
        n = self.num_assigned_envs if _is_bool else len(assigned_ids)
        bin_weights = self._bin_sampler.weights
        if self._env_group is not None:
            flattened_bin_ids = self._grouped_multinomial(
                bin_weights.concatenated_tensor, self._bin_sampler.group_bin_mask, assigned_ids
            )
        else:
            flattened_bin_ids = torch.multinomial(
                bin_weights.concatenated_tensor,
                n,
                replacement=True,
            ).to(self.device)
        # identify the motion_id and the actual bin id
        motion_ids, bin_ids = bin_weights.unwrap_flattened_idx(flattened_bin_ids)
        if _is_bool:
            assign_over_envs(self._assigned_env_motion_selection, assigned_ids, motion_ids)
        else:
            self._assigned_env_motion_selection[assigned_ids] = motion_ids
        max_start_time_s = (
            self.data_source.buffer_length[motion_ids].to(torch.float) - 1
        ) / self.data_source.framerate[motion_ids].to(torch.float)
        unclamped_start = (
            bin_ids * self.motion_bin_length_s + torch.rand(n, device=self.device) * self.motion_bin_length_s
        )
        start_times = torch.clamp(unclamped_start, max=max_start_time_s)
        if _is_bool:
            assign_over_envs(self._motion_buffer_start_time_s, assigned_ids, start_times)
        else:
            self._motion_buffer_start_time_s[assigned_ids] = start_times

    def _sample_assigned_env_starting_stub(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        """Sample motion file id and start time for the assigned envs.

        By "stub", it means the necessary information to sample the motion data for the assigned envs.
        """
        assigned_ids = self.envs_idx_to_assigned_ids(envs_idx)

        if self.env_starting_stub_sampling_strategy == "independent":
            self._sample_assigned_env_motion_selection(assigned_ids)
            self._sample_env_motion_start_time(assigned_ids)
        elif self.env_starting_stub_sampling_strategy == "concat_motion_bins":
            self._sample_as_concat_motion_bins(assigned_ids)
        else:
            raise ValueError(
                f"Unsupported env starting stub sampling method: {self.env_starting_stub_sampling_strategy}"
            )

        if self.start_at_timestep_zero_prob > 0.0:
            self._apply_start_at_timestep_zero(assigned_ids)

    def _apply_start_at_timestep_zero(self, assigned_ids: torch.Tensor) -> None:
        """Override sampled start times to 0 with probability ``start_at_timestep_zero_prob``."""
        _is_bool = is_env_mask(assigned_ids)
        # Over-sample full buffer for bool masks; mask-write to avoid GPU→CPU sync.
        n = self.num_assigned_envs if _is_bool else len(assigned_ids)
        force_zero = torch.rand(n, device=self.device) < self.start_at_timestep_zero_prob
        if _is_bool:
            target_mask = assigned_ids & force_zero
            self._motion_buffer_start_time_s[target_mask] = 0.0
        else:
            self._motion_buffer_start_time_s[assigned_ids[force_zero]] = 0.0

    def _local_motion_weights(self) -> torch.Tensor:
        """Return the sampling distribution restricted to this rank's motion subset.

        ``_motion_weights`` is global-shaped so the curriculum's
        ``dist.all_reduce`` stays shape-symmetric across ranks. The
        sampler, in contrast, can only produce motions this rank actually
        holds — so it needs the global weights gathered down to the local
        subset. Returns a 1-D tensor aligned with
        ``_assigned_env_motion_selection`` (i.e. local-id indexed).

        ``torch.multinomial`` re-normalizes its input, so the local slice need
        not sum to 1 (nor need ``_motion_weights`` before the first curriculum drain).
        """
        return self._motion_weights[self._local_to_global]

    def _sample_assigned_env_motion_selection(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Resample the motion selection for the assigned envs.

        ``assigned_ids`` may be an int-index tensor **or** a bool mask over the
        assigned range.  For bool masks we over-sample the full buffer and
        mask-write so no GPU→CPU sync is needed.
        """
        local_w = self._local_motion_weights()
        if self._env_group is not None:
            samples = self._grouped_multinomial(local_w, self._group_motion_mask, assigned_ids)
            if is_env_mask(assigned_ids):
                assign_over_envs(self._assigned_env_motion_selection, assigned_ids, samples)
            elif len(assigned_ids) > 0:
                self._assigned_env_motion_selection[assigned_ids] = samples
            return
        if is_env_mask(assigned_ids):
            samples = torch.multinomial(local_w, self.num_assigned_envs, replacement=True).to(self.device)
            # Through the mask with `torch.where`: a mask gather + write is a GPU->CPU sync each.
            assign_over_envs(self._assigned_env_motion_selection, assigned_ids, samples)
            return
        if len(assigned_ids) == 0:
            return
        self._assigned_env_motion_selection[assigned_ids] = torch.multinomial(
            local_w,
            len(assigned_ids),
            replacement=True,
        ).to(self.device)

    def _grouped_multinomial(
        self, weights: torch.Tensor, group_mask: torch.Tensor, assigned_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        """Draw one index per env from ``weights`` restricted to that env's group, with no host sync.

        ``weights`` is ``(K,)`` over clips or flattened start bins; ``group_mask`` is the matching ``(G, K)``
        membership. Every group draws ``n`` candidates in one ``(G, n)`` multinomial and each env keeps its own
        group's column — ``G`` is the variant count, so this costs ``G * n`` draws, not a per-group Python loop.
        ``multinomial`` renormalises each row, so only the within-group ratios of ``weights`` matter. A group whose
        weights are all zero (a curriculum can floor a whole group) falls back to uniform over its members rather
        than crashing the draw. For a bool ``assigned_ids`` mask the draw covers every assigned env (the caller
        writes through the mask); otherwise it is aligned with ``assigned_ids``.
        """
        is_bool = is_env_mask(assigned_ids)
        env_group = self._env_group if is_bool else self._env_group[assigned_ids]
        n = env_group.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        masked = weights.unsqueeze(0) * group_mask
        masked = torch.where(masked.sum(-1, keepdim=True) > 0, masked, group_mask.to(masked.dtype))
        draws = torch.multinomial(masked, n, replacement=True)
        return draws[env_group, torch.arange(n, device=self.device)]

    def _resolve_global_motion_groups(self, groups: MotionGroupOptions) -> torch.Tensor:
        """Map every clip to its group id via the dataset's per-clip labels; fail on an unmatched label either way."""
        labels = self.data_source.motion_labels(groups.metadata_key)
        if not groups.variant_labels:
            raise ValueError(
                f"env_motion_groups.variant_labels is empty; give one label per variant of entity "
                f"'{groups.entity_name}', in its GroupedEntityOptions.grouped_entities order."
            )
        self._group_names: list[str] = list(dict.fromkeys(groups.variant_labels))
        group_of = {name: i for i, name in enumerate(self._group_names)}
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        unknown = {label: c for label, c in counts.items() if label not in group_of}
        if unknown:
            raise ValueError(
                f"env_motion_groups: the dataset has clips labelled {unknown} (label: clip count) under "
                f"metadata_key={groups.metadata_key!r} that no variant of entity '{groups.entity_name}' matches "
                f"(variant_labels={groups.variant_labels}). Either add a variant for them, or filter them out of the "
                f"dataset (dataset_options include_pattern / exclude_list_file) — keeping them loaded would spend VRAM "
                f"on clips no env can ever play."
            )
        empty = [name for name in self._group_names if name not in counts]
        if empty:
            raise ValueError(
                f"env_motion_groups: variant label(s) {empty} of entity '{groups.entity_name}' match no clip in the "
                f"dataset, so envs holding them would have nothing to play. Labels present under "
                f"metadata_key={groups.metadata_key!r}: {counts}."
            )
        return torch.tensor([group_of[label] for label in labels], dtype=torch.long, device=self.device)

    def _resolve_env_groups(self, groups: MotionGroupOptions) -> torch.Tensor:
        """Each assigned env's group id, from the variant the heterogeneous entity gave it at build."""
        entity = resolve_named_entity(
            self._env.entities,
            groups.entity_name,
            hint="env_motion_groups.entity_name names it; set it to one of those.",
        )
        if not getattr(entity, "is_heterogeneous", False):
            raise TypeError(
                f"env_motion_groups.entity_name={groups.entity_name!r} names a {type(entity._options).__name__} "
                f"entity, but per-env motion groups need a heterogeneous one (GroupedEntityOptions) — every env of "
                f"this entity holds the same body, so there is no per-env variant to group by. Drop "
                f"env_motion_groups, or build '{groups.entity_name}' from GroupedEntityOptions."
            )
        if len(groups.variant_labels) != entity.num_variants:
            raise ValueError(
                f"env_motion_groups.variant_labels has {len(groups.variant_labels)} labels but entity "
                f"'{groups.entity_name}' has {entity.num_variants} variants; give exactly one label per "
                f"GroupedEntityOptions.grouped_entities entry, in that order."
            )
        variant_to_group = torch.tensor(
            [self._group_names.index(label) for label in groups.variant_labels], dtype=torch.long, device=self.device
        )
        env_group = variant_to_group[entity.env_variant_idx[self.assigned_env_slice].to(self.device)]
        # Build-time check (one sync, never on the hot path): a group some env here holds must have clips on
        # this rank, or its envs would sample from an all-zero row.
        present = set(env_group.unique().tolist())
        local_counts = self._group_motion_mask.sum(-1).tolist()
        starved = [self._group_names[g] for g in sorted(present) if local_counts[g] == 0]
        if starved:
            raise ValueError(
                f"env_motion_groups: envs on this rank hold variant(s) {starved} of '{groups.entity_name}', but "
                f"this rank's trajectory subset has no clip of them (per-group local clip counts: "
                f"{dict(zip(self._group_names, local_counts))}). With DDP, use motion_options.mp_split_method='Even' "
                f"(round-robin gives every rank each group) or 'None'; 'Segment' hands ranks contiguous clip ranges, "
                f"which a per-object library groups by object."
            )
        return env_group

    def _update_motion_weight_curriculum(self, all_reduce_fn=None) -> bool:
        """Advance the per-motion outcome curriculum and the metric drain together.

        Called once per PPO iteration by the runner mixin. Delegates to the
        ``MotionOutcomeCurriculum`` collaborator (which owns the every-N-iters
        drain gate); when it actually drains this call, the per-motion metric
        drain is piggybacked on the same DDP cadence. Returns whether a drain
        happened (the runner uses it for cadence assertions / DDP verification).
        """
        if self._curriculum is None:
            return False
        drained = self._curriculum.update(all_reduce_fn=all_reduce_fn)
        if drained and self._metrics is not None:
            self._metrics.drain(all_reduce_fn=all_reduce_fn)
        return drained

    def summarize_per_motion_metrics(self) -> dict[str, torch.Tensor]:
        """Per-motion-id tensors for CSV dump; ``{}`` when metrics are disabled.

        Public accessor for external dump tooling (see
        ``MotionTermOptions.compute_per_motion_metrics``); delegates to the
        ``PerMotionMetrics`` collaborator that owns the buffers.
        """
        return self._metrics.per_motion_tensors() if self._metrics is not None else {}


MOTION_TERM_REGISTRY = TermRegistry("MOTION_TERM", MotionTerm)
MOTION_TERM_REGISTRY.register()(MotionTerm)


class MotionManager(FlatManagerBase[MotionManagerOptions], SymmetricAugmentationMixin):
    """The manager to handle all motion references."""

    _registry = MOTION_TERM_REGISTRY

    def __init__(
        self,
        env: EnvBase,
        options: MotionManagerOptions,
    ):
        # NOTE: these must be set BEFORE super().__init__() because
        # FlatManagerBase.__init__ -> ManagerBase.__init__ calls _prepare_terms() which uses them.
        self._is_visualizing = False
        self._timestamp = torch.zeros(env.num_envs, device=env.device)
        # Opt-in motion-end resync (keeps episodes running past a clip's end).
        # No-op unless options.resync_on_motion_end; owns its own state.
        self._resync = MotionEndResync(self)
        self._frame_interval_s = torch.ones(env.num_envs, device=env.device) * options.frame_interval_s_range[0]
        self.require_frame_interval_s_resample = (
            options.frame_interval_s_range[1] - options.frame_interval_s_range[0]
        ) > 0

        self._env_symmetric_augmentation_mask = None
        if options.symmetric_augmentation:
            grouped = [
                name
                for name in options.term_keys()
                if getattr(getattr(options, name), "env_motion_groups", None) is not None
            ]
            if grouped:
                # The mirror applies one joint/link map per entity: it cannot swap a left and a right hand that are
                # separate entities, nor mirror the object a grouped clip manipulates.
                raise ValueError(
                    f"symmetric_augmentation=True cannot be combined with env_motion_groups (set on motion term(s) "
                    f"{grouped}): mirroring a clip would not mirror the per-variant entity it is grouped by. Set "
                    f"motion_options.symmetric_augmentation=False."
                )
            self._env_symmetric_augmentation_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        # The envs of the reset in progress whose init state is frame 0 of their window, and whether that
        # window has been filled yet (see ``resetting``). None outside a reset.
        self._resetting_envs: torch.Tensor | None = None
        self._resetting_window_filled = False
        # Inside ``managers_resetting``: the manager resets of a reset, during which ``data`` raises.
        self._managers_resetting = False
        # Per entity a reset-time reader asked for (see ``record_ended_episode_validity``): the window's
        # ``validity[:, 0]`` as each env's last episode ended.
        self._ended_episode_validity: dict[str, torch.Tensor] = {}

        super().__init__(env=env, options=options)

        # Fail fast: when motion-end resync is enabled, resolve + validate its RSI teleport
        # terms via a post-build hook (the event manager is built after this manager), so a
        # missing RSI term raises at env-construction time rather than at the first wrap.
        if self._resync.enabled:
            env.register_post_build_hook(self._resync.validate)

    def summary(self):
        """Get the tabular information of the motion reference managed buffer."""
        msg = f"<MotionManager> contains {len(self._terms)} motion terms to use.\n"

        msg += format_table(
            "Active Motion Terms",
            ["Name", "Num Trajectories"],
            ([name, term.num_trajectories] for name, term in self._terms.items()),
            align={"Name": "l", "Num Trajectories": "r"},
        )
        msg += "\n"

        return msg

    @property
    def data(self) -> dict[str, MotionReferenceData]:
        """The motion reference data for all entities, keyed by entity name.

        Filled by :meth:`compute` at the start of each ``MimicRLEnvBase.step()``
        and refilled per-env by :meth:`resetting` in :meth:`MimicRLEnvBase._reset_idx`.
        Fresh for all envs during reward, termination, action, metric, and
        observation computation in ``RLEnvBase.step()``.

        **Stale window**: :meth:`reset` is called inside
        ``MimicRLEnvBase._reset_idx`` *before* ``super()._reset_idx`` and only
        resamples motion selection / rolls back ``_timestamp`` — it does
        **not** refill the buffer. :meth:`resetting`, which brackets the scene
        reset and the RESET events, does: at the first init-state request inside
        it (``data_start_from="current_time"``), else on its exit. A RESET-mode
        event may therefore see *stale* ``data`` for reset envs (last step's
        compute fill); use :meth:`get_reset_reference_state` instead — it is the
        one ``ReferenceStateInitialization`` uses.

        **Raises inside the manager resets** (:meth:`managers_resetting`): the reset
        envs' window already holds the new episode there, so a ``reset()`` reading
        it to grade the ended episode would grade the wrong one, silently.
        Grade the ended episode off :meth:`ended_episode_validity`; read the new
        episode's reference in ``post_reset()``.
        """
        if self._managers_resetting:
            raise RuntimeError(
                "motion_manager.data was read inside a manager reset(), where the reset envs' window already "
                "holds the new episode's reference, not the one that just ended. To grade the ended "
                "episode, call motion_manager.record_ended_episode_validity(<entity>) in the term's build() and "
                "read motion_manager.ended_episode_validity(<entity>) in reset(); to read the new episode's "
                "reference, move the read to the term's post_reset(). Catching this error reads the new "
                "episode as the ended one."
            )
        return self._data

    @property
    def reference_frame(self) -> dict[str, MotionReferenceData]:
        """The reference state of all entities at current time step, keyed by entity name."""
        current_time = self._timestamp

        outdated_mask = torch.logical_or(
            (self._reference_frame_timestamp - current_time).abs() > 1e-6,
            self._reference_frame_timestamp < 1e-6,
        )
        if outdated_mask.any():
            envs_idx_to_update = self._all_indices[outdated_mask]
            for name, term in self._terms.items():
                envs_idx_this_buffer, _ = self._filter_envs_for_term(envs_idx_to_update, name)
                for entity_name, motion_seq_key in term.entities_map.items():
                    term.compute(
                        motion_seq_key,
                        envs_idx_this_buffer,
                        current_time[envs_idx_this_buffer].unsqueeze(-1),
                        self.env_origins,
                        self._reference_frame[entity_name],
                    )
                    if self._symmetric_augmentation_configured:
                        self._symmetric_augment_reference_data(self._reference_frame[entity_name], envs_idx_this_buffer)
            self._reference_frame_timestamp[outdated_mask] = current_time[outdated_mask]
        return self._reference_frame

    @property
    def num_frames(self) -> int:
        return self._options.num_frames

    @property
    def env_origins(self) -> torch.Tensor | None:
        """The env's ``env_origins`` ``(num_envs, 3)``, which shift the reference; None for an env without them.

        None, not zeros, so that a fill does not add zeros to every position field of every entity.
        """
        return getattr(self._env, "env_origins", None)

    @cached_property
    def _all_indices(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device)

    def _filter_envs_for_term(self, envs_idx: torch.Tensor, term_name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Filter envs_idx to only those assigned to the given motion term.

        Supports both int-index tensors and bool masks.

        Returns
        -------
        envs_idx_filtered : torch.Tensor
            The subset of envs_idx assigned to this term (bool mask or int indices,
            matching the input type).
        mask : torch.Tensor | None
            Boolean mask into ``envs_idx`` for the assigned subset.  When
            ``envs_idx`` is a bool mask this is the same tensor as
            ``envs_idx_filtered`` (both are full-length bool masks). ``None`` for
            int ids the term owns entirely — everything is selected, and gathering
            through an all-True mask would still be a GPU->CPU sync.
        """
        assignment = self._motion_term_assignment[term_name]
        assert assignment.step is None, "Only support continuous envs_idx assignment for now."
        if assignment.start <= 0 and assignment.stop >= self.num_envs:
            # The term owns every env (the usual single-term case): nothing to filter, and filtering
            # int ids would gather through a bool mask, a GPU->CPU sync.
            return envs_idx, (envs_idx if envs_idx.dtype == torch.bool else None)
        if envs_idx.dtype == torch.bool:
            # Zero out entries outside the assignment range while keeping the
            # bool-mask representation (non-blocking).
            filtered = envs_idx.clone()
            if assignment.start > 0:
                filtered[: assignment.start] = False
            if assignment.stop < len(filtered):
                filtered[assignment.stop :] = False
            return filtered, filtered
        mask = (envs_idx >= assignment.start) & (envs_idx < assignment.stop)
        return envs_idx[mask], mask

    @property
    def _motion_term_num_trajectories(self) -> dict[str, int]:
        """Get the number of trajectories for each motion term."""
        return {name: term.num_trajectories for name, term in self._terms.items()}

    def reset(self, envs_idx: Sequence[int] | torch.Tensor | None = None) -> dict[str, Any]:
        """Reset the motion reference manager.

        Resamples motion selection, rolls back ``_timestamp[envs_idx]`` to 0,
        and updates adaptive-sampling bookkeeping. **Does not refill**
        ``self._data`` — callers that need fresh per-step data must call
        :meth:`fill_data` themselves. ``MimicRLEnvBase`` does it through
        :meth:`resetting`, around the scene reset and RESET events (so the
        refill picks up any ``env_origins`` update from ``scene.reset`` /
        RESET events, as the reference-state seeding does).
        """
        if envs_idx is None:
            envs_idx = self._all_indices
        else:
            envs_idx = torch.as_tensor(envs_idx, device=self.device)

        self._record_adaptive_bin_outcomes(envs_idx)
        self._record_motion_outcomes(envs_idx)
        for name, ended in self._ended_episode_validity.items():
            assign_over_envs(ended, envs_idx, self._data[name].validity[:, 0])
        self._timestamp[envs_idx] = 0.0
        self._reset_motion_terms(envs_idx)
        self._resample_buffer_collate_params(envs_idx)

        return self._collect_log_extras()

    @torch.inference_mode()
    def post_reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        """Fan out to motion terms, restricted to each term's assigned environments.

        Overrides the flat fan-out (which would hand every term the full reset index)
        because a motion term owns a contiguous env slice and its buffers are sized to
        that slice — the same per-term narrowing :meth:`compute` and :meth:`fill_data`
        do. Terms are unique per name here, so the base class's dedupe is not needed.
        """
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs, prefer_slice=False)
        for name, term in self._terms.items():
            envs_idx_this_term, _ = self._filter_envs_for_term(envs_idx, name)
            term.post_reset(envs_idx_this_term)

    def _collect_log_extras(self) -> dict[str, float | str]:
        """Per-term scalar summaries surfaced into ``extras["log"]`` for rsl-rl.

        Two channels read off the term's ``MotionOutcomeCurriculum``, plus a third
        when the term computes per-motion metrics:
          - ``curriculum.summarize_motion_weights()`` — cached snapshot of the
            current ``_motion_weights``, refreshed only when the curriculum runs
            (the cache is what makes this method cheap on the hot path).
          - ``curriculum.last_curriculum_summary`` — outcome scalars populated
            after each drain. Empty until the first drain fires.
          - ``PerMotionMetrics.summary()`` — tracking-error quantiles, cached on
            the same drain.

        Multi-term envs prefix each key with the term name
        (``<term>/MotionWeights/...``) to avoid collisions; single-term
        envs (the common case) emit unprefixed keys for cleaner wandb
        panels. The prefix is at the front of the key so wandb's tree
        view groups all keys for a term together.

        Note on emission cadence: ``MotionManager.reset()`` is called on
        every step (with a possibly-empty ``envs_idx``), not only on
        actual reset events — this matches Eden's convention for other
        managers (see e.g. ``termination_manager.reset()``). Because both
        channels we emit are constants between curriculum updates, every
        rsl-rl per-iter averager sees the same value across all steps in
        the iter and the average equals the value. Numerically stable;
        no reset-event gating needed.
        """
        if not self._terms:
            return {}
        # String values flow through when ``log_motion_id_to_filename=True``;
        # the downstream rsl_rl mixin path forwards this dict straight to
        # ``wandb.log()`` so mixed types are fine.
        extras: dict[str, float | str] = {}
        multi_term = len(self._terms) > 1
        for name, term in self._terms.items():
            if term._curriculum is None:
                continue
            term_extras: dict[str, float | str] = {}
            term_extras.update(term._curriculum.summarize_motion_weights())
            term_extras.update(term._curriculum.last_curriculum_summary)
            # Per-motion tracking-error metrics; empty
            # dict on terms that haven't opted in. Same cache-on-drain
            # discipline as the curriculum summary above — no GPU work here.
            if term._metrics is not None:
                term_extras.update(term._metrics.summary())
            if multi_term:
                term_extras = {f"{name}/{k}": v for k, v in term_extras.items()}
            extras.update(term_extras)
        return extras

    def collect_per_motion_distributions(self, wandb_module) -> dict[str, "Any"]:
        """Per-term per-motion histograms for the wandb-direct emit path.

        Kept off the ``_collect_log_extras`` / ``extras["log"]`` channel: those
        flow through rsl_rl's ``Logger.ep_extras`` which iterates and prints
        every key per iter — at AMASS scale that loop is the per-step
        bottleneck. The wandb-direct mixin path (see
        ``MotionWeightCurriculumMixin``) calls this method once per learning
        iter and merges the result into the same ``wandb.log()`` payload.
        Empty when no term has ``log_per_motion_metrics=True``.
        """
        out: dict[str, "Any"] = {}
        if not self._terms:
            return out
        multi_term = len(self._terms) > 1
        for name, term in self._terms.items():
            if term._curriculum is None:
                continue
            sub = term._curriculum.collect_per_motion_distributions(wandb_module)
            if not sub:
                continue
            if multi_term:
                sub = {f"{name}/{k}": v for k, v in sub.items()}
            out.update(sub)
        return out

    def compute(self, dt: float):
        """Advance ``_timestamp`` and refill ``self._data`` for all envs.

        Called once per ``MimicRLEnvBase.step`` (before reward / termination
        / observation compute). The reset path fills the reset-env subset
        through :meth:`resetting` (see :meth:`fill_data`).

        When any motion term sets ``freeze_at_timestep_zero_prob > 0``, envs
        currently at ``_timestamp == 0`` and assigned to that term are
        per-step-randomly held at 0 (re-rolled every step — see
        :class:`MotionTermOptions.freeze_at_timestep_zero_prob`).
        """
        self._advance_timestamp(dt)
        # Blend the death-bin failure counts recorded at the previous reset
        # (see ``AdaptiveBinSampler.record_bin_outcomes``) into their EMA.
        self._blend_adaptive_ema()
        # Motion-end resync (opt-in), folded into the single per-step fill_data(all):
        # pre_fill detects wrapped envs from the advanced clock (before fill) and
        # resamples them so the ONE fill_data below fills them at the new phase (no
        # second fill_data); post_fill then teleports them onto it, through the wrap
        # mask, so no step reads on the host whether any env wrapped. None
        # when disabled. See eden.managers.motion.resync.MotionEndResync.
        wrap_mask = self._resync.pre_fill()
        self.fill_data()
        if wrap_mask is not None:
            self._resync.post_fill(wrap_mask)
        self._accumulate_per_motion_metrics()

    def _accumulate_per_motion_metrics(self) -> None:
        """Per-step batch metric accumulation for terms that opted in.

        Called once per ``compute()`` after ``fill_data`` has refreshed
        ``self._data``. Iterates opted-in terms × tracked entities and hands each
        ``(entity, ref)`` to the term's ``PerMotionMetrics.record_step`` (which
        owns the link-index cache and the error geometry). Skips entirely when no
        term has metrics enabled via the ``_any_per_motion_metrics`` early-return.
        """
        if not self._any_per_motion_metrics:
            return
        for term in self._terms.values():
            if term._metrics is None:
                continue
            for env_name, ref in self._data.items():
                entity = self._env.entities.get(env_name)
                if entity is None:
                    continue
                term._metrics.record_step(entity, ref, env_name)

    @cached_property
    def _any_per_motion_metrics(self) -> bool:
        """Fast bool: at least one term has opted into per-motion metrics."""
        return any(getattr(t, "compute_per_motion_metrics", False) for t in self._terms.values())

    def _advance_timestamp(self, dt: float) -> None:
        """Advance ``_timestamp`` by ``dt``, honouring per-term freeze probabilities.

        Each env belongs to exactly one term (via ``_motion_term_assignment``).
        We build a per-env probability tensor and re-roll ONCE per step, so an
        env's effective freeze probability is exactly the prob of its owning
        term — no multi-term AND composition.
        """
        any_freeze = any(term.freeze_at_timestep_zero_prob > 0.0 for term in self._terms.values())
        if not any_freeze:
            self._timestamp += dt
            return

        per_env_freeze_prob = torch.zeros_like(self._timestamp)
        for name, term in self._terms.items():
            if term.freeze_at_timestep_zero_prob <= 0.0:
                continue
            per_env_freeze_prob[self._motion_term_assignment[name]] = term.freeze_at_timestep_zero_prob

        zero_mask = self._timestamp == 0.0
        rand_vals = torch.rand_like(self._timestamp)
        freeze_mask = (rand_vals < per_env_freeze_prob) & zero_mask
        advance_mask = ~freeze_mask
        self._timestamp += advance_mask.to(self._timestamp.dtype) * dt

    def rewind_to_start(self, envs_idx: torch.Tensor) -> None:
        """Rewind playback to frame 0 for ``envs_idx`` WITHOUT resampling.

        Distinct from :meth:`reset`: ``reset`` re-samples motion selection and start phase
        (the per-episode randomized restart), whereas this replays each env's CURRENTLY-assigned
        clip deterministically from the beginning (e.g. a deploy starting from a stand). Zeros
        the manager timestamp and fans out to each term's :meth:`MotionTerm.rewind_to_start`.
        Does not refill the reference buffer — call :meth:`fill_data` afterwards.
        """
        self._timestamp[envs_idx] = 0.0
        # MOTION_TERM_REGISTRY enforces MotionTerm, so every term has rewind_to_start.
        for term in self._terms.values():
            term.rewind_to_start(envs_idx)

    def fill_data(self, envs_idx: torch.Tensor | None = None) -> None:
        """Refill ``self._data[entity][envs_idx]`` from ``self._timestamp[envs_idx]``; None fills every env.

        Called by :meth:`compute` (per step, every env) and by :meth:`resetting` (reset envs only),
        around the scene reset and RESET events of ``MimicRLEnvBase._reset_idx``. Bool-mask input
        becomes int ids through :func:`~eden.utils.misc.env_indices` — free for the auto-reset's mask,
        whose ids the env took once for the whole reset.

        Only the filled envs' rows are written. A fill of every env (None) hands each term its
        assignment as a slice, which the term gathers straight into the buffer; the ids of
        every env write the same rows, through the ids.
        """
        every_env = envs_idx is None
        envs_idx = self._all_indices if every_env else env_indices(envs_idx)

        n = envs_idx.shape[0]
        if n == 0:
            return

        time_to_target_frame = torch.arange(self.num_frames, device=self.device, dtype=torch.float32)
        if self._options.data_start_from == "one_frame_interval":
            time_to_target_frame += 1
        time_to_target_frame = time_to_target_frame.unsqueeze(0).repeat(n, 1)
        time_to_target_frame *= self._frame_interval_s[envs_idx].unsqueeze(-1)

        for name, term in self._terms.items():
            if every_env:
                # Row i of every env's tensors is env i, so the term's envs are its assignment slice.
                envs_idx_this_buffer = self._motion_term_assignment[name]
                time_to_target_frame_this_buffer = time_to_target_frame[envs_idx_this_buffer]
                ids_this_buffer = self._all_indices[envs_idx_this_buffer]
            else:
                envs_idx_this_buffer, envs_idx_this_buffer_mask = self._filter_envs_for_term(envs_idx, name)
                time_to_target_frame_this_buffer = (
                    time_to_target_frame
                    if envs_idx_this_buffer_mask is None
                    else time_to_target_frame[envs_idx_this_buffer_mask]
                )
                ids_this_buffer = envs_idx_this_buffer
            for entity_name, motion_seq_key in term.entities_map.items():
                term.compute(
                    motion_seq_key,
                    envs_idx_this_buffer,
                    self._timestamp[envs_idx_this_buffer].unsqueeze(-1) + time_to_target_frame_this_buffer,
                    self.env_origins,
                    self._data[entity_name],
                )
                if self._symmetric_augmentation_configured:
                    self._symmetric_augment_reference_data(self._data[entity_name], ids_this_buffer)

        rows = slice(None) if every_env else envs_idx
        for entity_name, entity_data in self._data.items():
            # An invalid frame's time is -1. The other envs' rows are as their own fill left them.
            entity_data.time_to_target_frame[rows] = torch.where(entity_data.validity[rows], time_to_target_frame, -1.0)

    def get_target_dofs_name(self, entity_name: str) -> list[str] | None:
        """Return the resolved DOF names that the motion reference exposes for ``entity_name``.

        These are the names that index into ``reference_frame[entity_name].dofs_pos``
        along the last axis. Returns ``None`` when no motion term targets the entity.

        When multiple terms target the same entity, the first term is used; the
        :class:`MotionManager` already requires consistent num_dofs across terms.
        """
        for term in self._terms.values():
            motion_seq_key = term.entities_map.get(entity_name)
            if motion_seq_key is None:
                continue
            names = term.target_dofs_name.get(motion_seq_key)
            if isinstance(names, str):
                return [names]
            return list(names) if names is not None else []
        return None

    def get_motion_phase(self, entity_name: str) -> torch.Tensor:
        """Per-env normalized playback phase of the clip tracking ``entity_name``.

        ``frame / (clip_length - 1)`` in ``[0, 1]`` at the CURRENT reference frame
        (the same floor-of-timestamp frame selection as :meth:`MotionTerm._select_frames`),
        so mid-clip RSI starts report their true position in the clip rather than
        the episode-step phase. Envs not assigned to a term tracking the entity
        report 0. Shape ``(num_envs,)``.

        Raises if *no* term tracks ``entity_name`` at all: that is a config error
        (a misspelled entity), and returning the all-zero buffer would hand a
        consumer a silently-constant signal forever.
        """
        tracking_terms = [term for term in self._terms.values() if entity_name in term.entities_map]
        if not tracking_terms:
            tracked_names = sorted({name for term in self._terms.values() for name in term.entities_map})
            raise ValueError(
                f"Entity '{entity_name}' is not tracked by any motion term, so its playback phase is "
                f"undefined. Tracked entities: {tracked_names}"
            )
        phase = torch.zeros(self.num_envs, device=self.device)
        for term in tracking_terms:
            data = term.data_source.motion_sequences[term.entities_map[entity_name]]
            assigned = term.assigned_env_slice
            selection = term._assigned_env_motion_selection
            length = data.buffer_length[selection].to(torch.float32)
            frame = torch.floor(
                (term._motion_buffer_start_time_s + self._timestamp[assigned]) * data.framerate[selection]
            )
            phase[assigned] = torch.clamp(frame, max=length - 1.0) / (length - 1.0).clamp(min=1.0)
        return phase

    def get_init_reference_state(
        self,
        envs_idx: Sequence[int] | torch.Tensor | None = None,
        entity_names: Sequence[str] | None = None,
    ) -> dict[str, MotionReferenceState]:
        """Get the initial reference state for the given envs_idx, keyed by entity name.

        ``entity_names`` limits the gather to those entities (``None``: every tracked one). Each
        entity costs one gather per field, so the reset path asks only for the entities it seeds
        rather than refilling all of them once per RSI term. Valid for any envs, at any time;
        a RESET event seeding its own ``envs_idx`` calls :meth:`get_reset_reference_state`, which
        saves the gather.
        """
        wanted = self._wanted_entities(entity_names, "get_init_reference_state")
        if envs_idx is None:
            envs_idx = self._all_indices
        else:
            # Int ids, not a bool mask: every field below is gathered and written for the selected
            # envs, a GPU->CPU sync per op through a mask and none through ids.
            envs_idx = env_indices(torch.as_tensor(envs_idx, device=self.device))
        for name, term in self._terms.items():
            entities = [(entity, key) for entity, key in term.entities_map.items() if entity in wanted]
            if not entities:
                continue
            envs_idx_this_buffer, _ = self._filter_envs_for_term(envs_idx, name)
            for entity_name, motion_seq_key in entities:
                term.fill_init_reference_state(
                    motion_seq_key,
                    envs_idx_this_buffer,
                    self.env_origins,
                    self._init_reference_state[entity_name],
                )
                if self._symmetric_augmentation_configured:
                    self._symmetric_augment_reference_data(
                        self._init_reference_state[entity_name],
                        envs_idx_this_buffer,
                    )
        return {name: state[envs_idx] for name, state in self._init_reference_state.items() if name in wanted}

    def get_reset_reference_state(
        self,
        envs_idx: Sequence[int] | torch.Tensor | None = None,
        entity_names: Sequence[str] | None = None,
    ) -> dict[str, MotionReferenceState]:
        """:meth:`get_init_reference_state` for a RESET event seeding its own ``envs_idx``.

        Inside :meth:`resetting` with ``data_start_from="current_time"``, the window's first frame *is*
        each reset env's init state, so the first call fills the reset envs' window and every call
        reads the state off its frame 0: the reset gathers each reference row once, not once for the
        window and again for reference-state initialization. Anywhere else it is
        :meth:`get_init_reference_state`.

        ``envs_idx`` must be envs of the reset in progress, which a RESET event's ``envs_idx`` always
        is (a throttled event's is a subset). Frame 0 of any other env's window is a mid-episode
        reference row, and it is not checked here, since that would cost a GPU->CPU sync every reset:
        call :meth:`get_init_reference_state` for envs that need not be resetting.
        """
        if self._resetting_envs is None or envs_idx is None:
            return self.get_init_reference_state(envs_idx, entity_names)
        wanted = self._wanted_entities(entity_names, "get_reset_reference_state")
        # Int ids, as in ``get_init_reference_state``.
        return self._init_state_from_window(env_indices(torch.as_tensor(envs_idx, device=self.device)), wanted)

    def _wanted_entities(self, entity_names: Sequence[str] | None, caller: str) -> Container[str]:
        """``entity_names`` checked against the tracked entities (``None``: every tracked one)."""
        if entity_names is None:
            return self._init_reference_state.keys()
        if isinstance(entity_names, str):
            raise TypeError(
                f"{caller}: entity_names takes a sequence of names, got the string "
                f"{entity_names!r}; pass [{entity_names!r}]."
            )
        wanted = set(entity_names)
        if unknown := sorted(wanted - self._init_reference_state.keys()):
            raise ValueError(
                f"{caller}: no motion term tracks {unknown}; tracked entities: "
                f"{sorted(self._init_reference_state)}. Add the entity to a motion term's sequences, "
                f"or ask for one of the tracked names."
            )
        return wanted

    def record_ended_episode_validity(self, entity_name: str) -> None:
        """Keep, from now on, ``entity_name``'s :meth:`ended_episode_validity` (call it in a term's ``build()``).

        For a term that grades the episode a reset ends: the manager resets run after the reset envs'
        window has been refilled for the new episode (see :meth:`resetting`), so they cannot read the
        ended one off ``data``.
        """
        if entity_name not in self._data:
            raise ValueError(
                f"record_ended_episode_validity: no motion term tracks '{entity_name}'; tracked entities: "
                f"{sorted(self._data)}. Name one of them, or add the entity to a motion term's entities_map."
            )
        self._ended_episode_validity.setdefault(
            entity_name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

    def ended_episode_validity(self, entity_name: str) -> torch.Tensor:
        """``validity[:, 0]`` of ``entity_name``'s window as each env's last episode ended, ``(num_envs,)``.

        False where that episode played its clip out. Taken by :meth:`reset`, before the window moves
        on; an env that never ran an episode reads False, the empty window's validity.
        """
        ended = self._ended_episode_validity.get(entity_name)
        if ended is None:
            raise ValueError(
                f"ended_episode_validity('{entity_name}'): nothing recorded it. Call "
                f"motion_manager.record_ended_episode_validity('{entity_name}') in the term's build(); reading "
                f"data['{entity_name}'].validity in a reset() instead sees the new episode's window."
            )
        return ended

    @contextmanager
    def resetting(self, envs_idx: torch.Tensor) -> Iterator[None]:
        """Bracket the scene reset and RESET events of a reset of ``envs_idx``, and leave its window filled.

        Enter after :meth:`reset`; on a normal exit ``data`` holds the reset envs' new window, filled
        by :meth:`fill_data` once, so every manager reset after it sees the new episode (the ended one's
        clip state is :meth:`ended_episode_validity`). With ``data_start_from="current_time"`` the
        window's first frame *is* each reset env's init state, so the first
        :meth:`get_reset_reference_state` call inside fills the window there and every call reads the
        state off it. Otherwise, and when nothing asks, the window is filled on exit, after the
        RESET events. The window reads ``env_origins`` when it is filled: at the first such call, or
        on exit.
        """
        if self._options.data_start_from == "current_time":
            self._resetting_envs = envs_idx
        try:
            yield
            filled = self._resetting_window_filled
        finally:
            self._resetting_envs = None
            self._resetting_window_filled = False
        if not filled:
            self.fill_data(envs_idx)

    @contextmanager
    def managers_resetting(self) -> Iterator[None]:
        """Bracket the manager resets that follow :meth:`resetting`: reading :attr:`data` inside raises.

        The reset envs' window holds the new episode by then, so a manager ``reset()`` reading it to
        grade the episode that ended would silently grade the new one.
        """
        self._managers_resetting = True
        try:
            yield
        finally:
            self._managers_resetting = False

    def _init_state_from_window(
        self, envs_idx: torch.Tensor, entity_names: Container[str]
    ) -> dict[str, MotionReferenceState]:
        """:meth:`get_reset_reference_state` inside :meth:`resetting`: frame 0 of the reset envs' window."""
        if not self._resetting_window_filled:
            self.fill_data(self._resetting_envs)
            self._resetting_window_filled = True
        return {
            name: MotionReferenceState(
                dofs_pos=data.dofs_pos[envs_idx, 0],
                dofs_vel=data.dofs_vel[envs_idx, 0],
                base_pos_w=data.base_pos_w[envs_idx, 0],
                base_quat_w=data.base_quat_w[envs_idx, 0],
                base_lin_vel_w=data.base_lin_vel_w[envs_idx, 0],
                base_ang_vel_w=data.base_ang_vel_w[envs_idx, 0],
            )
            for name, data in self._data.items()
            if name in entity_names
        }

    def _reset_motion_terms(self, envs_idx: torch.Tensor):
        """Reset each motion terms if is selected by the given envs_idx."""
        for name in self._terms.keys():
            envs_idx_this_term, _ = self._filter_envs_for_term(envs_idx, name)
            self._terms[name].reset(
                envs_idx=envs_idx_this_term,
                symmetric_augmentation_mask_buffer=self._env_symmetric_augmentation_mask,
            )

    def _record_adaptive_bin_outcomes(self, envs_idx: torch.Tensor) -> None:
        """Record per-episode failure outcomes for adaptive sampling.

        Called once per reset boundary, before the timestamp / start-time is
        reset, with *all* resetting envs; the terminated subset contributes a
        failure count in the bin it **died in** (see
        :meth:`AdaptiveBinSampler.record_bin_outcomes`).

        Bootstrap resets (``episode_length_buf == 0``) carry no real outcome
        and add nothing, matching :meth:`_record_motion_outcomes`. Supports
        bool masks and int-index tensors.
        """
        if not hasattr(self._env, "termination_manager"):
            return
        # Every resetting env, the bootstrap ones too, weighted out rather than filtered out: the failure
        # count is a scatter-add of the terminated flag, so an env that adds 0 is one that was never
        # selected. The ids are the auto-reset's own, taken once for the whole reset; selecting the
        # active subset would take another nonzero(), a GPU->CPU sync.
        failed = self._env.termination_manager.terminated & (self._env.episode_length_buf > 0)
        envs_idx = env_indices(envs_idx)
        for name, term in self._terms.items():
            if term._bin_sampler is None:
                continue
            envs_for_term, _ = self._filter_envs_for_term(envs_idx, name)
            if len(envs_for_term) == 0:
                continue
            term._bin_sampler.record_bin_outcomes(
                envs_for_term,
                failed[envs_for_term],
                self._timestamp[envs_for_term],
            )

    def _blend_adaptive_ema(self) -> None:
        """Blend the failure-count EMA of every adaptive motion term."""
        for term in self._terms.values():
            if term._bin_sampler is not None:
                term._bin_sampler.blend_failure_ema()

    def _record_motion_outcomes(self, envs_idx: torch.Tensor) -> None:
        """Record per-motion-id success / failure outcomes for envs being reset.

        Mirrors :meth:`_record_adaptive_bin_outcomes` (which is keyed on the
        *bin* within a motion the episode died in) but classifies *every* resetting env as
        success-candidate or failure and dispatches by ``motion_id``.

        Two filters are applied here:

        1. ``episode_length_buf > 0`` — envs at their bootstrap reset have no
           real outcome; skip them.
        2. ``terminated`` — splits resetting envs into a failure cohort
           (counted unconditionally) and a success-candidate cohort. The term
           applies the second-stage filter (motion actually played to its
           natural end) using the per-env timestamp we pass through.

        Supports both bool-mask and int-index ``envs_idx``; bool-mask is
        non-blocking on the GPU.
        """
        if not self.has_adaptive_motion_weights:
            return
        if not hasattr(self._env, "termination_manager"):
            return
        terminated = self._env.termination_manager.terminated
        active = self._env.episode_length_buf > 0
        env_dt = float(self._env.dt)

        if envs_idx.dtype == torch.bool:
            # Int ids, once per cohort: every per-term gather below would sync through a mask.
            success_envs = env_indices(envs_idx & active & ~terminated)
            failure_envs = env_indices(envs_idx & active & terminated)
            envs_idx = env_indices(envs_idx) if any(t._metrics is not None for t in self._terms.values()) else envs_idx
        else:
            active_mask = active[envs_idx]
            if not active_mask.any():
                return
            active_envs = envs_idx[active_mask]
            terminated_subset = terminated[active_envs]
            success_envs = active_envs[~terminated_subset]
            failure_envs = active_envs[terminated_subset]

        episode_lengths = self._env.episode_length_buf
        for name, term in self._terms.items():
            if term._curriculum is None:
                continue
            success_for_term, _ = self._filter_envs_for_term(success_envs, name)
            failure_for_term, _ = self._filter_envs_for_term(failure_envs, name)
            term._curriculum.record_motion_outcomes(
                success_for_term,
                failure_for_term,
                self._timestamp[success_for_term],
                self._timestamp[failure_for_term],
                env_dt,
            )
            # Metric scatter runs over the full reset cohort (success +
            # failure) since the per-motion mean is meaningful regardless
            # of outcome. Cheap no-op slot when the term hasn't enabled metrics.
            if term._metrics is not None:
                resetting_for_term, _ = self._filter_envs_for_term(envs_idx, name)
                term._metrics.scatter_at_reset(resetting_for_term, episode_lengths)

    @property
    def has_adaptive_motion_weights(self) -> bool:
        """``True`` iff at least one motion term enables hard-motion mining."""
        return bool(self._options.adaptive_motion_terms())

    def update_motion_weight_curriculum(self) -> None:
        """Run one curriculum step across all adaptive motion terms.

        Called once per PPO iteration by the rsl-rl runner mixin (see
        ``eden.envs.wrappers.rsl_rl_env``). Terms that haven't yet reached
        their per-term ``motion_weight_update_every_n_iters`` cadence are
        cheap no-ops; terms that *have* reached cadence drain their counts,
        all-reduce them across DDP ranks (so every rank applies the same
        update — a hard requirement for keeping ``_motion_weights``
        byte-identical), then rewrite ``_motion_weights`` per
        ``motion_weight_update_mode``.

        DDP shape symmetry holds because every rank's outcome buffers and
        ``_motion_weights`` are sized to the GLOBAL motion-id space,
        independent of ``mp_split_method`` or each rank's local trajectory
        slice. Per-rank contributions land in the same global slots; after
        the SUM all_reduce every rank applies the same update and
        the resulting ``_motion_weights`` is byte-identical across ranks.
        """
        do_reduce = dist.is_initialized() and dist.get_world_size() > 1
        all_reduce_fn = (lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM)) if do_reduce else None
        for term in self._terms.values():
            term._update_motion_weight_curriculum(all_reduce_fn=all_reduce_fn)

    def _prepare_terms(self):
        self._data: dict[str, MotionReferenceData] = dict()
        self._reference_frame: dict[str, MotionReferenceData] = dict()
        self._init_reference_state: dict[str, MotionReferenceState] = dict()
        self._reference_frame_timestamp = torch.zeros(self.num_envs, device=self.device)

        for term_name in self._options.term_keys():
            term_option: MotionTermOptions = getattr(self._options, term_name)
            term: MotionTerm = self._build_term(term_name)
            self._term_names.append(term_name)
            self._terms[term_name] = term

            for entity_name, motion_seq_key in term_option.entities_map.items():
                motion_seq = term.data_source.motion_sequences[motion_seq_key]
                num_dofs = motion_seq.dofs_pos._data_shape[0] if motion_seq.dofs_pos is not None else 0
                # The first link field present: a dataset may omit the body-frame ``links_*_b`` fields (no tracking
                # term reads them) to fit a large library on the GPU, and ``links_pos_b`` is then ``None``.
                link_field = next(
                    (
                        getattr(motion_seq, f)
                        for f in ("links_pos_w", "links_pos_b", "links_quat_w", "links_quat_b")
                        if getattr(motion_seq, f) is not None
                    ),
                    None,
                )
                num_links = link_field._data_shape[0] if link_field is not None else 0
                # Contact partner axis follows the dataset (e.g. one column per
                # object link for articulated partners); 1 when absent.
                num_contact_partners = motion_seq.num_contact_partners

                if entity_name in self._data:
                    # TODO: support different number of dofs and links for the same entity
                    if self._data[entity_name].dofs_pos.shape[-1] != num_dofs:
                        raise ValueError(
                            f"The number of dofs has to be the same for the same entity for now. "
                            f"Got {self._data[entity_name].dofs_pos.shape[-1]} and {num_dofs}."
                        )
                    if self._data[entity_name].links_pos_b.shape[-2] != num_links:
                        raise ValueError(
                            f"The number of links has to be the same for the same entity for now. "
                            f"Got {self._data[entity_name].links_pos_b.shape[-2]} and {num_links}."
                        )
                    continue

                self._data[entity_name] = MotionReferenceData.make_empty(
                    self.num_envs,
                    self.num_frames,
                    num_dofs,
                    num_links,
                    device=self.device,
                    num_contact_partners=num_contact_partners,
                )
                self._reference_frame[entity_name] = MotionReferenceData.make_empty(
                    self.num_envs,
                    1,
                    num_dofs,
                    num_links,
                    device=self.device,
                    num_contact_partners=num_contact_partners,
                )
                self._init_reference_state[entity_name] = MotionReferenceState.make_empty(
                    self.num_envs,
                    num_dofs,
                    device=self.device,
                )

        # NOTE: assign motion trajectories to this process from rank/world_size (when multi-processing),
        # referring to how pytorch distributed launch is designed at https://pytorch.org/docs/stable/elastic/run.html
        if dist.is_initialized():
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            local_rank = 0
            world_size = None
        self._assign_motion_trajectories_for_this_process(local_rank, world_size)
        self._assign_motion_term_for_envs()

        self._resample_buffer_collate_params()

    def _assign_motion_trajectories_for_this_process(self, local_rank: int = 0, world_size: int | None = None):
        """Assign the motion trajectories for the current process based on rank_id/world_size.

        `self._motion_term_num_trajectories` will be updated.
        As for the current process, only a subset of trajectories will be enabled.
        Not-used motion term (python object) will be deleted.

        ``adaptive_motion_weights=True`` is compatible with every
        ``mp_split_method``: each ``MotionTerm`` keeps its
        ``_motion_weights`` and outcome buffers sized to the GLOBAL motion-id
        space, so the iter-end ``dist.all_reduce`` shape-matches across ranks
        even when each rank only holds its assigned trajectory slice in VRAM.
        Adaptive terms with an empty local subset on this rank are kept (rather
        than dropped) so every rank iterates the same set of terms in the same
        order, which is a hard requirement for collective ops.
        """
        if world_size is None or self._options.mp_split_method == "None":
            # NOTE: single process, enable all trajectories for all motion buffers.
            # NOTE: no split method, enable all trajectories for all motion buffers.
            for term in self._terms.values():
                term.enable_trajectories()
        elif self._options.mp_split_method == "Even":
            # split the trajectories evenly (round-robin) for each process.
            # |----motion1-----|----motion2-----|----motion3-----|----motion4-----|----motion5-----|
            # [123412341234123412341234123412341234123412341234123412341234123412341234123412341234]
            idxs = torch.arange(
                sum(self._motion_term_num_trajectories.values()),
                device=self.device,
                dtype=torch.long,
            )
            if len(idxs) % world_size != 0:
                dropped = len(idxs) % world_size
                en.logger.warning(
                    f"Even split: dropping {dropped} trailing trajectory(ies) "
                    f"({len(idxs)} total not divisible by world_size={world_size})."
                )
                idxs = idxs[:-dropped]
            idxs = idxs.reshape(-1, world_size)
            idxs = idxs[:, local_rank]
            buffer_traj_start = 0
            for term_name in list(self._terms.keys()):
                term = self._terms[term_name]
                buffer_traj_end = term.num_trajectories + buffer_traj_start
                idxs_this_buffer = idxs[(idxs >= buffer_traj_start) & (idxs < buffer_traj_end)] - buffer_traj_start
                if len(idxs_this_buffer) > 0:
                    term.enable_trajectories(idxs_this_buffer)
                else:
                    if getattr(term, "adaptive_motion_weights", False):
                        # An adaptive term needs ≥1 trajectory on every rank so
                        # it participates uniformly in the iter-end all_reduce.
                        # Hitting this means world_size exceeds the term's
                        # trajectory count.
                        raise ValueError(
                            f"Motion term '{term_name}' has adaptive_motion_weights=True "
                            f"but received zero local trajectories under "
                            f"mp_split_method='Even' with world_size={world_size}. "
                            f"Resolve by either: reducing world_size below the term's "
                            f"num_trajectories ({term.num_trajectories}); switching "
                            f"mp_split_method='None' (every rank holds the full library, "
                            f"trading VRAM for not having to subset); or disabling "
                            f"adaptive_motion_weights."
                        )
                    term = self._terms.pop(term_name)
                    del term
                buffer_traj_start = buffer_traj_end
        elif self._options.mp_split_method == "Segment":
            # split the trajectories based on the total number of trajectories.
            # Assuming all trajectories are put in a single sequence, each process will be assigned a segment.
            # |----motion1-----|----motion2-----|----motion3-----|----motion4-----|----motion5-----|
            # |-------------------process1-------------------|-------------------process2----------|
            total_traj_global = sum(self._motion_term_num_trajectories.values())
            num_traj_per_process = total_traj_global // world_size
            process_traj_start = num_traj_per_process * local_rank
            process_traj_end = (
                num_traj_per_process * (local_rank + 1) if local_rank < world_size - 1 else total_traj_global
            )
            term_traj_start = 0
            for term_name in list(self._terms.keys()):
                term = self._terms[term_name]
                term_traj_end = term.num_trajectories + term_traj_start
                term_slice_start = np.clip(process_traj_start, term_traj_start, term_traj_end) - term_traj_start
                term_slice_end = np.clip(process_traj_end, term_traj_start, term_traj_end) - term_traj_start
                if term_slice_start < term_slice_end:
                    term.enable_trajectories(slice(term_slice_start, term_slice_end))
                else:
                    if getattr(term, "adaptive_motion_weights", False):
                        # See the matching branch in the Even path for rationale.
                        raise ValueError(
                            f"Motion term '{term_name}' has adaptive_motion_weights=True "
                            f"but received zero local trajectories under "
                            f"mp_split_method='Segment' with world_size={world_size}. "
                            f"Resolve by either: reducing world_size below the term's "
                            f"num_trajectories ({term.num_trajectories}); switching "
                            f"mp_split_method='None' (every rank holds the full library); "
                            f"or disabling adaptive_motion_weights."
                        )
                    term = self._terms.pop(term_name)
                    del term
                term_traj_start = term_traj_end
        else:
            raise ValueError(f"Unknown mp_split_method: {self._options.mp_split_method}")

    def _assign_motion_term_for_envs(self):
        """Assign which motion term is used for each env.

        Defining this at the startup stage leads to more stable term data behavior. It is just like
        the fixed terrain type in a curriculum-based locomotion training setting.
        """
        term_names = list(self._motion_term_num_trajectories.keys())
        num_terms = len(term_names)

        if num_terms == 0:
            self._motion_term_assignment = {}
            return

        if self.num_envs < num_terms:
            en.logger.warning(
                f"num_envs ({self.num_envs}) < number of motion terms ({num_terms}). "
                f"Only the first {self.num_envs} terms will receive 1 env each."
            )
            num_env_assignment_d = {}
            for i, name in enumerate(term_names):
                num_env_assignment_d[name] = 1 if i < self.num_envs else 0
        else:
            # Largest-remainder method with a floor of 1 env per term.
            # Since num_envs >= num_terms, every term is guaranteed at least 1.
            total_traj = sum(self._motion_term_num_trajectories.values())
            num_env_assignment_d = {name: 1 for name in term_names}
            remaining = self.num_envs - num_terms
            if remaining > 0:
                # Distribute the remaining envs proportionally
                exact_extra = {
                    name: (num_traj / total_traj) * remaining
                    for name, num_traj in self._motion_term_num_trajectories.items()
                }
                floored_extra = {name: int(v) for name, v in exact_extra.items()}
                for name, v in floored_extra.items():
                    num_env_assignment_d[name] += v
                # Give leftover envs to terms with the largest fractional remainders
                leftover = remaining - sum(floored_extra.values())
                if leftover > 0:
                    remainders = {name: exact_extra[name] - floored_extra[name] for name in term_names}
                    sorted_by_remainder = sorted(
                        remainders,
                        key=remainders.get,
                        reverse=True,  # type: ignore
                    )
                    for i in range(leftover):
                        num_env_assignment_d[sorted_by_remainder[i]] += 1

        # assign the motion term for each envs
        self._motion_term_assignment: dict[str, slice] = dict()
        start_idx = 0
        for name, num_env in num_env_assignment_d.items():
            self._motion_term_assignment[name] = slice(start_idx, start_idx + num_env)
            start_idx += num_env

        # Remove terms with 0-env assignments and inform the rest.
        for name in list(self._terms.keys()):
            assignment = self._motion_term_assignment[name]
            if assignment.start < assignment.stop:
                self._terms[name].set_envs_idx_assignments(assignment)
            else:
                del self._terms[name]
                if name in self._term_names:
                    self._term_names.remove(name)

    def _resample_buffer_collate_params(self, envs_idx: slice | torch.Tensor | None = None):
        if not self.require_frame_interval_s_resample:
            return

        if envs_idx is None:
            envs_idx = slice(None)
            num_envs = self.num_envs
        elif isinstance(envs_idx, slice):
            num_envs = len(range(*envs_idx.indices(self.num_envs)))
        elif is_env_mask(envs_idx):
            num_envs = self.num_envs  # over-sample full buffer, mask-write
        elif isinstance(envs_idx, torch.Tensor):
            num_envs = len(envs_idx)
        else:
            raise ValueError(f"Invalid environment indices: {envs_idx}")

        samples = (
            torch.rand(num_envs, device=self.device)
            * (self._options.frame_interval_s_range[1] - self._options.frame_interval_s_range[0])
            + self._options.frame_interval_s_range[0]
        )
        if is_env_mask(envs_idx):
            self._frame_interval_s[envs_idx] = samples[envs_idx]
        else:
            self._frame_interval_s[envs_idx] = samples
