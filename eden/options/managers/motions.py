"""Motion manager and motion-term configuration options."""

from __future__ import annotations
from typing import Literal, Sequence

from eden.options.datasets import DatasetOptions
from eden.options.managers.base import ManagerOptions, ManagerTermOptions
from eden.options.options import ConfigurableOptions


class MotionGroupOptions(ConfigurableOptions):
    """Restrict each env to the clips that match the variant of a heterogeneous entity it holds.

    A heterogeneous entity (``GroupedEntityOptions``) gives each env one variant for the whole run — one object
    mesh out of a bank. A reference clip that manipulates object A is meaningless in an env holding object B, so
    with this set every motion draw (initial assignment, reset, adaptive start bins) samples only among the clips
    whose per-clip label equals the env's variant label.

    Parameters
    ----------
    entity_name : str
        The heterogeneous scene entity whose per-env variant selects the group, e.g. ``"object"``.
    metadata_key : str
        The dataset's per-clip label list the groups are matched against (``dataset.motion_labels(metadata_key)``),
        e.g. ``"objects"``. Defaults to ``"objects"``.
    variant_labels : list[str]
        One label per variant, in the entity's ``grouped_entities`` order: ``variant_labels[i]`` is the per-clip
        label of the clips an env holding variant ``i`` may play. Derive it from the same ordered constant that
        builds ``grouped_entities`` so the two cannot drift. Labels may repeat (variants then share a group).
    """

    entity_name: str
    metadata_key: str = "objects"
    variant_labels: list[str]


class MotionTermOptions(ManagerTermOptions):
    """
    Configuration for the motion buffer.

    Parameters
    ----------
    dataset_options: DatasetOptions | None
        The options for the dataset.
    entities_map: dict[str, str]
        Maps each environment entity name (key) to its motion sequence key in the dataset (value).
    target_dofs_name: dict[str, str | list[str]]
        Per entity, the names of the degrees of freedom to target. The reference's DOF columns follow this
        list's order, which need not match the dataset's.
    target_links_name: dict[str, str | list[str]]
        Per entity, the names of the links to target. The reference's link rows follow this list's order, which
        need not match the dataset's.
    env_starting_stub_sampling_strategy: Literal["independent", "concat_motion_bins"]
        How a motion and its start time are drawn. ``"independent"`` (default) draws the motion, then its start
        time. ``"concat_motion_bins"`` draws one ``motion_bin_length_s`` bin from all motions' bins concatenated
        (so a longer motion is proportionally likelier) and starts inside it; requires ``motion_bin_length_s``.
    motion_start_from_middle_range: tuple[float, float] | list[float]
        ``(min, max)`` fractions of the clip duration the start time is drawn uniformly from. Used only when
        ``max > 0`` and ``motion_bin_length_s`` is None (bin sampling takes priority). Defaults to ``(0.0, 0.0)``:
        every clip starts at its first frame.
    motion_bin_length_s: float | None
        Width [s] of the start-time bins. Setting it enables bin-weighted start sampling; required by
        ``adaptive_sampling`` and ``"concat_motion_bins"``. Defaults to None.
    adaptive_sampling: bool
        Master toggle for adaptive *start-time* (intra-motion bin) sampling. Defaults to False.
    adaptive_kernel_size: int
        Temporal smoothing kernel width ``K``. Smooths each failure onto the bin it died in and the ``K - 1``
        bins before it, so starts move onto the run-up to a hard moment (BeyondMimic's look-ahead kernel).
        ``1`` disables smoothing. Defaults to 1.
    adaptive_lambda: float
        Exponential decay for kernel weights: the death bin weighs ``1``, the bin ``i`` before it
        ``adaptive_lambda ** i``, normalised to sum to 1. Defaults to 0.8.
    adaptive_uniform_ratio: float
        Minimum uniform floor in sampling probabilities. Defaults to 0.1.
    adaptive_alpha: float
        EMA blend rate for failure counts. Defaults to 0.001.
    adaptive_failure_rate_max_over_mean: float
        Per-START-bin concentration clamp for adaptive start-time sampling.
        A bin whose (smoothed) failure rate exceeds ``mean * this`` (per-motion
        mean over its bins) is clipped to that ceiling before normalization, so
        no single bin can absorb arbitrarily much probability mass even if its
        failure rate is pathologically high (e.g. a momentarily impossible
        frame). Bin-level analogue of
        ``motion_weight_failure_rate_max_over_mean``; without it the uniform
        mix alone only bounds the floor, not the peak, so the start
        distribution can still collapse onto one bin. ``<= 0`` disables the
        clamp. Defaults to 10.0.
    motion_weights: str | list[float] | None
        Initial per-motion-id sampling weights, one entry per loaded trajectory
        in motion-id order. ``None`` (default) weights every motion equally. A
        ``list[float]`` supplies the weights inline; a ``str`` is a path to a
        text/JSON file (loaded via ``load_text``) holding a ``"motion_weights"``
        key with that list — convenient for large clip libraries. The length
        must equal the number of loaded trajectories or ``build`` raises. When
        ``adaptive_motion_weights`` is True these are the starting weights the
        curriculum then rewrites.
    adaptive_motion_weights: bool
        Master toggle for adaptive *per-motion-id* sampling (hard-motion mining).
        Orthogonal to ``adaptive_sampling`` (which controls intra-motion start-time
        bins). When True, ``_motion_weights`` is updated each training iteration
        based on per-motion success / early-termination outcomes during rollouts.
        Defaults to False.
    motion_weight_success_discount: float
        Per-success multiplicative decay applied to a motion's weight when its
        motion plays to its natural end (regardless of whether the env also
        terminated this step). ``< 1.0`` shrinks the weight of easy motions;
        ``= 1.0`` disables decay. Defaults to 0.999 (matches ProtoMotions
        reference).
    motion_weight_failure_amplify: float
        Per-failure multiplicative gain applied to a motion's weight when an
        episode terminates early *without* the motion completing. ``> 1.0``
        boosts the weight of hard motions; ``= 1.0`` disables amplification.
        Defaults to 2.0 (matches ProtoMotions reference). For a gentler,
        symmetric per-iter pressure, set this to a value just above 1
        (e.g. 1.001) alongside ``motion_weight_success_discount=0.999``.
    motion_weight_floor: float
        Lower clamp on per-motion weight (before re-normalization), so no motion
        can be permanently zeroed out. Defaults to 1e-3. Only consulted by
        ``motion_weight_update_mode="compound"``; the snapshot mode bounds
        weights via ``motion_weight_failure_rate_max_over_mean`` +
        ``motion_weight_uniform_mix`` instead.
    motion_weight_update_mode: Literal["compound", "snapshot"]
        How ``_motion_weights`` is rewritten from the drained outcome counts:

        * ``"snapshot"`` (default): stateless rate-proportional formulation
          ported from SONIC's ``update_adaptive_sampling_probabilities``
          (motion_lib_base.py:2570-2585). Computes
          ``failure_rate = n_failure / (n_success + n_failure)``, clamps
          outliers at ``mean_failure_rate * failure_rate_max_over_mean``,
          normalises, then blends with uniform via ``uniform_mix``. The
          result is always the same function of the **current** drained
          counts — no history compounds. ``success_discount`` /
          ``failure_amplify`` / ``floor`` are ignored under this mode.
        * ``"compound"`` (opt-in): log-space additive update
          ``log_w += n_success * log(success_discount) + n_failure *
          log(failure_amplify)``. Integrates **all historical outcomes**
          forever; any persistent failure-rate bias compounds exponentially
          until a motion saturates at the floor or at 1.0. Observed to
          collapse to a single motion at LAFAN1 scale; kept only for
          ablation runs.
    motion_weight_uniform_mix: float
        Applied AFTER the curriculum update under either mode:
        ``w := (1 - α) * w + α / N``. Caps maximum concentration at
        ``(1 - α) + α/N`` and floors minimum at ``α/N``, independent of
        the update mode's own dynamics. Defaults to 0.0 (off).
    motion_weight_failure_rate_max_over_mean: float
        Snapshot-mode-only outlier clamp. A motion with failure rate
        greater than ``mean * this`` is clipped to that ceiling before
        being converted to a probability. With ``N=12`` and clamp=200,
        the worst case is one motion captures up to ~95% of the pre-mix
        probability mass; combined with ``uniform_mix=0.1`` the worst-case
        cap becomes ~85%. Defaults to 200.0 (matches SONIC's
        ``adp_samp_failure_rate_max_over_mean``).
    motion_weight_update_every_n_iters: int
        Apply the curriculum update every N PPO iterations. Outcomes accumulate
        in between. Defaults to 1.
    log_per_motion_metrics: bool
        Emit per-motion-id distributions as ``wandb.Histogram`` panels under
        ``MotionPerMotion/Weights`` (current sampling weights) and
        ``MotionPerMotion/Outcome/{success_count, failure_count, total,
        success_rate}`` (per-drain counts). One histogram per metric per iter
        — independent of motion count.

        Useful for diagnosing curriculum collapse: a long tail forming in the
        weight distribution shows up immediately as a skewed histogram. One
        scalar series per motion-id would be ~47k series per iteration at
        AMASS scale (~9.4k motions × 5 fields), which overwhelms the wandb
        backend and slows rsl-rl's logger; histograms keep the diagnostic at
        constant cost. ``MotionWeights/argmax_motion_id`` (a
        single int) stays on the aggregate panel and is always emitted —
        that's the cheapest way to identify the dominant clip and works at
        any scale. Defaults to False.
    log_motion_id_to_filename: bool
        When True, additionally emit ``MotionWeights/argmax_motion_filename``
        as a wandb **string** field — the basename of the dataset file
        corresponding to the current argmax motion-id. Resolved against the
        dataset's ``all_motion_files`` list (the post-``select_trajectories``
        per-rank slice), so DDP-sharded runs may emit ``"<not_local>"`` when
        the argmax is held by another rank. Off by default because the
        integer ``argmax_motion_id`` is enough to identify the clip via the
        dataset's saved file order, and the string field is heavier to handle
        (wandb plot-by-string is awkward; mostly useful in the wandb run table
        for at-a-glance correlation with the integer time series).
    start_at_timestep_zero_prob: float
        On reset, with this probability force the start time to 0 instead of
        sampling via ``motion_start_from_middle_range`` / bin-weighted
        sampling. Useful for eval-mode determinism and curriculum warm-ups
        where every clip should begin at frame 0. Defaults to 0.0.
    freeze_at_timestep_zero_prob: float
        Per-step probability that ``MotionManager.compute(dt)`` does NOT
        advance ``_timestamp`` for envs currently at timestamp 0. Re-rolled
        every step (mirrors Holosoma's WBT semantic), so a setting of 0.5
        gives an env at t=0 a 50% chance of advancing each step — useful
        for mild start-time jitter, NOT for permanent freeze (an env
        frozen on step N can advance on step N+1). Defaults to 0.0.
    compute_per_motion_metrics: bool
        When True, accumulate per-step tracking error against the
        reference (MPJPE in world frame, root position L2 error, root
        quaternion geodesic error) into per-env running sums, then
        scatter to per-motion-id aggregators at episode reset. Drained
        alongside the success/failure curriculum counts (same
        ``all_reduce_fn`` across DDP ranks; same cadence as
        ``motion_weight_update_every_n_iters``). Aggregate scalars
        (``Mimic/mpjpe_{mean,p50,p95}`` etc.) flow into ``extras["log"]``;
        per-motion tensors are available via ``summarize_per_motion_metrics()``
        for CSV dump.

        Hot-path cost when enabled: one extra ``get_links_pos`` per step
        + 3 per-env reductions + 3 in-place adds. No CPU↔GPU sync. When
        disabled (default), buffers are unallocated and the accumulator
        is a single early-return check. Defaults to False.
    metric_anchor_link_name: str | None
        Optional anchor link (e.g. ``"torso_link"``). When set together
        with ``compute_per_motion_metrics=True``, additionally accumulates
        G-MPJPE — link MPJPE measured in the anchor frame (subtract the
        anchor link's world-frame position from all tracked link positions
        before the L2 norm). Surfaces ``Mimic/g_mpjpe_*`` scalars in
        ``extras["log"]``. ``None`` → don't compute G-MPJPE, keep the
        baseline 3-metric set (MPJPE, root_pos_err, root_quat_err) only.
        Adds one extra entity ``find_named_links_idx_local`` lookup at
        first compute, then one ``get_links_pos`` row (for the anchor)
        and one subtract on the hot path. Defaults to None.
    env_motion_groups: MotionGroupOptions | None
        Per-env clip groups keyed on a heterogeneous entity's variant (see :class:`MotionGroupOptions`), for a
        reference that is only valid with one variant of that entity — an object-manipulation clip and the object
        it manipulates. ``None`` (the default) samples every env from the whole library.
    """

    dataset_options: DatasetOptions | None = None
    entities_map: dict[str, str] = {}
    target_dofs_name: dict[str, str | list[str]] = {}
    target_links_name: dict[str, str | list[str]] = {}
    motion_weights: str | list[float] | None = None
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
    env_motion_groups: MotionGroupOptions | None = None


class MotionManagerOptions(ManagerOptions[MotionTermOptions]):
    """
    Configuration for the motion reference manager.

    Parameters
    ----------
    entities_map: dict[str, str]
        Not read by the motion manager; each term's own ``entities_map`` maps entities to sequences.
    num_frames: int
        The number of future reference frames generated per step. Defaults to 1.
    data_start_from: Literal["one_frame_interval", "current_time"]
        Where the 0-th reference frame sits: ``"one_frame_interval"`` (default) puts it one frame interval
        after the current time ``t``, ``"current_time"`` at ``t``.
    frame_interval_s_range: tuple[float, float]
        ``(min, max)`` time between reference frames [s], sampled uniformly per env on reset. Defaults to
        ``(0.1, 0.1)``.
    mp_split_method: Literal["None", "Even", "Segment"]
        How the motion library is split across DDP ranks: ``"None"`` (default) gives every rank all of it,
        ``"Even"`` splits it evenly (round-robin), ``"Segment"`` assigns each rank a contiguous segment.
    symmetric_augmentation: bool
        Whether to enable symmetric augmentation of motion data (mirroring across the x-z plane).
        When True, the required mapping fields below must also be provided.
    symmetric_augmentation_joint_mapping: Sequence[int] | None
        The joint index permutation for symmetric augmentation. Maps each joint to its mirrored counterpart.
    symmetric_augmentation_joint_reverse_buf: Sequence | None
        Per-joint sign multipliers for symmetric augmentation (e.g., -1.0 for joints that flip sign when mirrored).
        Required when symmetric_augmentation is True.
    symmetric_augmentation_links_mapping: Sequence[int] | None
        Link index permutation for symmetric augmentation (in the order of links_of_interests).
        Required when symmetric_augmentation is True and links are referenced.
    reference_prim_path: str | None
        Not read by the motion manager.
    resync_on_motion_end: bool
        When True, an env whose reference reaches the end of its clip is soft-reset
        mid-episode: a fresh start phase is resampled and the robot is teleported
        onto the new reference start (via the ReferenceStateInitialization event
        term), WITHOUT ending the episode. Mirrors the clip-wrap behavior of
        ``MotionCommand`` (used by the ``beyondmimic`` example). Default False:
        the reference freezes at the last frame (velocities zeroed) and the robot
        is not re-synced.
    <motion_term_name>: MotionTermOptions
        The motion term configuration to be used.
    """

    entities_map: dict[str, str] = {}
    num_frames: int = 1
    data_start_from: Literal["one_frame_interval", "current_time"] = "one_frame_interval"
    frame_interval_s_range: tuple[float, float] = (0.1, 0.1)
    mp_split_method: Literal["None", "Even", "Segment"] = "None"
    symmetric_augmentation: bool = False
    symmetric_augmentation_joint_mapping: Sequence[int] | None = None
    symmetric_augmentation_joint_reverse_buf: Sequence | None = None
    symmetric_augmentation_links_mapping: Sequence[int] | None = None
    reference_prim_path: str | None = None
    resync_on_motion_end: bool = False

    def adaptive_motion_terms(self) -> list[str]:
        """The motion terms that mine hard motions (``adaptive_motion_weights``), whose weights a curriculum steps."""
        return [name for name in self.term_keys() if getattr(self, name).adaptive_motion_weights]
