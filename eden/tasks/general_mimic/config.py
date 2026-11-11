"""General-purpose motion-tracking task — 10~100 clip scale, G1 robot.

Goal: ship a one-command-runnable Eden mimic policy for a multi-motion
library that ProtoMotions would call a "motion tracker" (as opposed to
the single-clip BeyondMimic policy). The defaults below target the
LAFAN1-retargeted-G1 dataset (~75 locomotion / dance clips); swap to any
other ``DatasetBase`` via ``GeneralMimicDataMod``.

Design choices (vs the BeyondMimic single-clip task):

* **Action**: ExplicitPDController on G1 with BeyondMimic's
  ``G1_ACTION_SCALE``, but residual on the motion reference
  (``ReferenceSource.MOTION_REF``) where BeyondMimic offsets from the
  default pose (``ReferenceSource.DEFAULT``).
* **Dataset**: LAFAN1-retargeted-G1 by default (auto-downloads from HF on
  first run via :class:`LAFAN1RetargetedG1Dataset`). Pass ``dataset_name``
  on the data TaskMod to switch — currently ``"lafan1"`` and ``"amass"``
  are wired in; extend ``_DATASET_CLS_BY_NAME`` for further G1-compatible
  motion libraries.
* **Adaptive motion-id curriculum** is **enabled** — the whole point of a
  multi-motion task. Defaults to SONIC's ``snapshot`` update (drain every
  iter); the ``compound`` mode's knobs keep ProtoMotions' values
  (success-discount 0.999, failure-amplify 2.0, floor 1e-3). Per-motion
  metrics (``MotionWeights/*`` + ``MotionOutcome/*``) flow through to wandb
  automatically; no extra observability wiring needed.
* **Rewards / observations / terminations**: ProtoMotions
  ``motion_tracker/g1-bones-deploy`` numbers verbatim (same weights and
  sigmas as BeyondMimic — both lineages converged on the same recipe).
* **Episode length**: 5s (shorter than BeyondMimic's 10s) so the
  curriculum drain per PPO iter sees more episode ends → richer
  per-motion success/failure signal.
* **DDP scaling**: ``mp_split_method='None'`` by default — at 10~100
  motions, the full library fits comfortably per-rank. For larger libraries,
  ``'Even'`` shards trajectories per rank without touching the rest of this
  config.
"""

from typing import Annotated, Literal

import eden as en
from eden.terms import (
    ObsTerm,
    RewardTerm,
    DoneTerm,
)
from eden import actions, events, observations, rewards, terminations
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.managers.motions import MotionManagerOptions
from eden.managers.motion_manager import MotionTerm
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskMod
from eden.utils.configs import (
    EdenMimicConfig,
    EnvOptions,
    RewardManagerOptions,
    SceneOptions,
    ActionManagerOptions,
    ObservationManagerOptions,
    TerminationManagerOptions,
    EventManagerOptions,
    MetricManagerOptions,
)

from eden.datasets.amass_dataset import AMASSRetargetedG1Dataset
from eden.datasets.base import DatasetBase
from eden.datasets.lafan1_dataset import LAFAN1RetargetedG1Dataset
from eden.datasets.modifiers import GroundFirstFrame, SmoothStartTransition
from eden.managers.terms.events.motion import ReferenceStateInitialization
from eden.managers.terms.observations.tracking import (
    motion_dofs_pos,
    motion_dofs_vel,
    motion_future_dofs_pos,
    motion_future_dofs_vel,
    motion_future_links_ori_b,
    motion_future_links_pos_b,
    motion_links_ori_b,
    motion_links_pos_b,
    robot_body_ori_b,
    robot_body_pos_b,
)
from eden.managers.terms.rewards.tracking import (
    MotionGlobalBodyAngularVelocityErrorExp,
    MotionGlobalBodyLinearVelocityErrorExp,
    MotionRelativeBodyOrientationErrorExp,
    MotionRelativeBodyPositionErrorExp,
    motion_link_orientation_tracking_exp,
    motion_link_position_tracking_exp,
)
from eden.managers.terms.terminations.tracking import (
    BadMotionLinkPosZOnly,
    bad_anchor_ori,
    bad_anchor_pos_z_only,
)

from eden.options.robots.unitree_g1 import G1_ACTION_SCALE, G1_IK_PROFILE, G1Dofs29
from eden.constants import ReferenceSource

_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

_G1Options = G1Dofs29()

# Datasets accepted by ``GeneralMimicDataMod.dataset_name``. Restricted to a
# tiny explicit set rather than dispatched through ``DATASET_REGISTRY``: the
# registry holds object-manipulation datasets (DexYCB, OMOMO, Arctic, etc.)
# whose ``motion_*_name`` shapes don't match the G1 motion-tracking config in
# this task — exposing them via a free-form string knob would be a footgun.
# Each entry's class must expose ``motion_dofs_name['g1']`` (29 DOFs in the
# canonical G1 order) and ``motion_links_name['g1']``.
_DATASET_CLS_BY_NAME: dict[str, type[DatasetBase]] = {
    "lafan1": LAFAN1RetargetedG1Dataset,
    "amass": AMASSRetargetedG1Dataset,
}

# ProtoMotions-style lookahead schedule (matches ``g1-bones-deploy``):
# the policy sees the motion reference 1/2/4/8 motion-ticks ahead in
# addition to the current frame. With ``frame_interval_s = 0.02`` that's
# +20/40/80/160 ms. ``num_frames`` is sized to cover ``max(_FUTURE_FRAME_OFFSETS) + 1``
# so the motion buffer slot at index 8 is populated.
_FUTURE_FRAME_OFFSETS: tuple[int, ...] = (1, 2, 4, 8)
_MOTION_NUM_FRAMES: int = max(_FUTURE_FRAME_OFFSETS) + 1

# 14-link tracking set: pelvis + per-side hip / knee / ankle + torso +
# per-side shoulder / elbow / wrist. Matches BeyondMimic exactly. The
# anchor (torso_link) is intentionally included so the
# relative_body_pos/ori rewards include the anchor at its own coordinate
# (offset = 0 by construction — acts as a no-op for that link).
_TRACKED_LINKS = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
_ANCHOR_LINK = "torso_link"


class GeneralMimicDataMod(TaskMod):
    r"""Configure dataset + motion-manager options for the general mimic task.

    Defaults to LAFAN1-retargeted-G1. Switch to AMASS-G1 (incl. ProtoMotions
    retargetings like ``retargeted_g1_soma``) by passing ``dataset_name="amass"``.
    Both options expose the same 29-DOF G1 motion shape, so the
    motion-manager / reward / observation config below is unchanged across
    them. New G1-compatible datasets are added by extending
    ``_DATASET_CLS_BY_NAME`` at module scope — this is intentionally an
    explicit allowlist (not ``DATASET_REGISTRY`` lookup) so that the CLI
    Literal rejects nonsensical choices (DexYCB, OMOMO, etc.) at configure
    time rather than at apply / train time.

    The mutual-exclusion-locked adaptive curriculum knobs live here so a
    user can tune the curriculum without forking the whole config. The
    ProtoMotions reference values (success-discount 0.999, failure-amplify
    2.0, floor 1e-3) match the ``motion_tracker/g1-bones-deploy`` config.

    CLI usage::

        eden train --task general_mimic \\
            --dataset_name amass \\
            --raw_data_path /my/amass_g1_dir \\
            --target_framerate 50.0
    """

    def __init__(
        self,
        dataset_name: Annotated[
            Literal["lafan1", "amass"],
            "Which G1-retargeted motion dataset to load. 'lafan1' (default) is the "
            "ember-lab-berkeley LAFAN1-retargeted-G1 set (~75 locomotion / dance clips, "
            "CSV-on-disk). 'amass' is the IsaacLab-AMP-format AMASS-G1 plus ProtoMotions "
            "retargetings (e.g. ``retargeted_g1_soma`` — ~9k clips, NPZ-on-disk). Both "
            "expose the same 29-DOF G1 motion shape, so the motion-manager / reward / "
            "observation config is unchanged.",
        ] = "lafan1",
        raw_data_path: Annotated[str | None, "Directory with retargeted-G1 motion files. None → auto-download."] = None,
        processed_data_path: Annotated[str | None, "Path to a processed-motion cache .pt file."] = None,
        target_framerate: Annotated[float, "Target framerate (Hz) for retiming motion clips."] = 50.0,
        include_pattern: Annotated[
            str | None,
            "Regex (re.search) on each motion-file basename; only matching clips are kept. "
            "E.g. '^walk' for LAFAN1 walks only.",
        ] = None,
        exclude_pattern: Annotated[
            str | None,
            "Regex (re.search) on each motion-file basename; matching clips are dropped after include filtering.",
        ] = None,
        include_list_file: Annotated[
            str | None,
            "Path to a YAML or JSON file enumerating motion files to keep. Accepts a flat list "
            "of paths/basenames, a list of dicts each with a ``file:`` key (ProtoMotions "
            "``amass_*_train.yaml`` shape), or a dict wrapping either under ``motions:``. "
            "Matched on basename with extension stripped, so ``CMU/45/45_01_stageii.motion`` "
            "matches the local ``45_01_stageii.npz``.",
        ] = None,
        exclude_list_file: Annotated[
            str | None,
            "Path to a YAML or JSON file enumerating motion files to drop. Same accepted shapes "
            "and basename-stem matching as ``include_list_file``.",
        ] = None,
        adaptive_motion_weights: Annotated[bool, "Enable per-motion-id adaptive curriculum."] = True,
        motion_weight_success_discount: Annotated[float, "Per-success decay; <1.0 shrinks easy motions."] = 0.999,
        motion_weight_failure_amplify: Annotated[float, "Per-failure boost; >1.0 amplifies hard motions."] = 2.0,
        motion_weight_floor: Annotated[float, "Lower clamp on per-motion weight (compound mode only)."] = 1e-3,
        motion_weight_update_mode: Annotated[
            Literal["compound", "snapshot"],
            "Curriculum update dynamics. 'snapshot' (default) is SONIC's stateless rate-proportional "
            "formulation — w_t is a function of THIS drain's failure rates, no history compounds. "
            "'compound' is the log-space multiplicative update, which collapses to a single motion "
            "at LAFAN1 scale.",
        ] = "snapshot",
        motion_weight_uniform_mix: Annotated[
            float,
            "Final regulariser blending uniform into the post-curriculum distribution: "
            "w := (1-α)·w + α/N. Caps max at (1-α)+α/N, floors min at α/N. Default 0.1.",
        ] = 0.1,
        motion_weight_failure_rate_max_over_mean: Annotated[
            float,
            "Snapshot-mode outlier clamp: a motion with failure rate > mean·this is clipped to the "
            "ceiling before being normalised to a probability. Default 200.0 matches SONIC.",
        ] = 200.0,
        motion_weight_update_every_n_iters: Annotated[int, "Curriculum drain cadence (in PPO iters)."] = 1,
        compute_per_motion_metrics: Annotated[
            bool,
            "Surface per-motion tracking-error metrics (MPJPE, root pos / quat err, G-MPJPE) to wandb. "
            "Defaults to True since general_mimic's primary purpose is multi-motion observability.",
        ] = True,
        metric_anchor_link_name: Annotated[
            str | None,
            "Anchor link for G-MPJPE (anchor-frame link MPJPE). Defaults to torso_link to match the "
            "reward terms' anchor; set to None to skip G-MPJPE.",
        ] = _ANCHOR_LINK,
        log_per_motion_metrics: Annotated[
            bool,
            "Emit per-motion-id wandb scalars under a separate top-level prefix "
            "(``MotionPerMotion/Weights/m{id}`` and "
            "``MotionPerMotion/Outcome/m{id}/{success_count, failure_count, total, success_rate}``). "
            "Aggregate panels under ``MotionWeights/*`` and ``MotionOutcome/*`` stay free of "
            "per-motion noise in wandb's left-nav. Defaults to True at LAFAN1 scale "
            "(5 extra scalars per motion per iter); turn OFF for SEED-scale datasets "
            "(100k+ motions). ``MotionWeights/argmax_motion_id`` is always emitted regardless.",
        ] = True,
        log_motion_id_to_filename: Annotated[
            bool,
            "Resolve ``MotionWeights/argmax_motion_id`` (an int) to the dataset filename and emit "
            "as ``MotionWeights/argmax_motion_filename`` (a wandb string field). Off by default — "
            "the integer id already pins the clip and the string field is mainly useful in the "
            "wandb run table for at-a-glance correlation. Set to True for ad-hoc diagnostic runs.",
        ] = False,
        smooth_start_transition: Annotated[
            bool,
            "Pre-pend each clip with a stand-up → IK-step → SLERP-merge segment so RSI from "
            "the default pose to clip-frame-0 is no longer a teleport (OpenTrack-style). "
            "Off by default — flipping ON re-runs the cache build the next time the dataset "
            "is loaded (extends each clip by ~2 s at 50 Hz, ~0.4 s of IK per clip). For large "
            "libraries, build the cache out-of-band first with "
            "``eden preprocess --task general_mimic --smooth_start_transition True``, "
            "ideally sharded across GPUs via ``--num_batches N --batch_idx K``.",
        ] = False,
        ground_first_frame: Annotated[
            bool,
            "Auto-shift each clip uniformly along +Z so frame 0's robot AABB sits at-or-above "
            "the floor. Useful when retargeting leaves clips slightly penetrating the ground "
            "at clip start. Spins up a transient Genesis env once per cache load to run a "
            "batched frame-0 FK; clips already above the floor pass through with no shift, "
            "so the cost is just the env build (~15 s kernel compile, then cached per process). "
            "On by default — set to ``False`` if clips are already grounded and the per-load "
            "FK pass isn't worth it, or for SEED-class libraries that would OOM with one env "
            "per clip (alternatively, set ``ground_first_frame_chunk_size``).",
        ] = True,
        ground_first_frame_margin_m: Annotated[
            float,
            "Safety margin (m) above z=0 enforced by ``ground_first_frame``. ``dz`` is "
            "computed as ``max(0, margin - aabb_min_z)``, so a positive value nudges the "
            "foot off the floor by that amount.",
        ] = 0.0,
        ground_first_frame_chunk_size: Annotated[
            int | None,
            "Max clips per Genesis env when ``ground_first_frame`` is on. ``None`` (default) "
            "allocates one env per clip — fastest for LAFAN1-class libraries. Set to a "
            "few hundred / thousand for SEED-class libraries (100k+ clips) where "
            "one-env-per-clip would OOM. The Genesis env is built once and re-used across "
            "chunks via repeated ``set_qpos``.",
        ] = None,
    ):
        if dataset_name not in _DATASET_CLS_BY_NAME:
            raise ValueError(f"dataset_name must be one of {sorted(_DATASET_CLS_BY_NAME)}; got {dataset_name!r}.")
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.target_framerate = target_framerate
        self.include_pattern = include_pattern
        self.exclude_pattern = exclude_pattern
        self.include_list_file = include_list_file
        self.exclude_list_file = exclude_list_file
        self.adaptive_motion_weights = adaptive_motion_weights
        self.motion_weight_success_discount = motion_weight_success_discount
        self.motion_weight_failure_amplify = motion_weight_failure_amplify
        self.motion_weight_floor = motion_weight_floor
        self.motion_weight_update_mode = motion_weight_update_mode
        self.motion_weight_uniform_mix = motion_weight_uniform_mix
        self.motion_weight_failure_rate_max_over_mean = motion_weight_failure_rate_max_over_mean
        self.motion_weight_update_every_n_iters = motion_weight_update_every_n_iters
        self.compute_per_motion_metrics = compute_per_motion_metrics
        self.metric_anchor_link_name = metric_anchor_link_name
        self.log_per_motion_metrics = log_per_motion_metrics
        self.log_motion_id_to_filename = log_motion_id_to_filename
        self.smooth_start_transition = smooth_start_transition
        self.ground_first_frame = ground_first_frame
        self.ground_first_frame_margin_m = ground_first_frame_margin_m
        self.ground_first_frame_chunk_size = ground_first_frame_chunk_size

    def apply(self, config: "GeneralMimicConfig") -> "GeneralMimicConfig":
        # Resolve the dataset class once and reuse for the modifier wiring,
        # the dataset_options below, and the target_links_name on the motion
        # term. All three sites need to agree on the class so the link-name
        # map (which differs slightly between retargetings) stays consistent.
        dataset_cls = _DATASET_CLS_BY_NAME[self.dataset_name]

        modifiers = []
        # Ground-shift runs before smooth_start: the synthesized prefix is
        # IK'd against the (now-grounded) frame-0 pose, so the stand-up is
        # consistent with the clip's actual floor.
        if self.ground_first_frame:
            modifiers.append(
                GroundFirstFrame.configure(
                    robot_options=_G1Options,
                    safety_margin_m=self.ground_first_frame_margin_m,
                    chunk_size=self.ground_first_frame_chunk_size,
                )
            )
        if self.smooth_start_transition:
            # The Genesis-based IK reads DOFs in ``dofs_name`` order, which is
            # what every Eden dataset already uses, so no DOF permutation is
            # needed. The link-name map is what lets the modifier re-FK the
            # per-link world poses through Genesis after it extends the clip.
            modifiers.append(
                SmoothStartTransition.configure(
                    ik_profile=G1_IK_PROFILE,
                    motion_links_name_per_key=dict(dataset_cls.motion_links_name),
                )
            )

        config.motion_options = MotionManagerOptions(
            # Pre-populate one current frame (index 0) plus enough future
            # frames to cover the lookahead schedule in
            # ``_FUTURE_FRAME_OFFSETS``. ``motion_future_*`` observation
            # terms index this buffer directly — see the
            # ``motion_future_*`` keys in ``_CommonObservations``.
            num_frames=_MOTION_NUM_FRAMES,
            data_start_from="current_time",
            frame_interval_s_range=(0.02, 0.02),
            # At 10~100 motions the whole library fits per rank, so sharding buys
            # nothing; flip to 'Even' for larger subsets (sharding keeps the
            # adaptive curriculum identical across ranks).
            mp_split_method="None",
            default=MotionTerm.configure(
                dataset_options=dataset_cls.configure(
                    raw_data_path=self.raw_data_path,
                    processed_data_path=self.processed_data_path,
                    target_framerate=self.target_framerate,
                    include_pattern=self.include_pattern,
                    exclude_pattern=self.exclude_pattern,
                    include_list_file=self.include_list_file,
                    exclude_list_file=self.exclude_list_file,
                    modifiers=modifiers,
                ),
                entities_map={"robot": "g1"},
                target_dofs_name={"g1": _G1Options.dofs_name},
                target_links_name={"g1": dataset_cls.motion_links_name["g1"]},
                # Intra-motion start-time bin curriculum + free choice of
                # start point. Keeps episodes diverse even on a small clip
                # library and lets the bin curriculum concentrate sampling
                # on the hard chunks of each clip.
                motion_bin_length_s=1.0,
                adaptive_sampling=True,
                motion_start_from_middle_range=(0.0, 1.0),
                # Per-motion-id hard-motion mining. Drains drive the
                # ``MotionOutcome/success_rate_*`` and ``MotionWeights/*`` scalars
                # into ``extras["log"]`` for rsl-rl — no extra metric wiring needed.
                adaptive_motion_weights=self.adaptive_motion_weights,
                motion_weight_success_discount=self.motion_weight_success_discount,
                motion_weight_failure_amplify=self.motion_weight_failure_amplify,
                motion_weight_floor=self.motion_weight_floor,
                motion_weight_update_mode=self.motion_weight_update_mode,
                motion_weight_uniform_mix=self.motion_weight_uniform_mix,
                motion_weight_failure_rate_max_over_mean=self.motion_weight_failure_rate_max_over_mean,
                motion_weight_update_every_n_iters=self.motion_weight_update_every_n_iters,
                # Per-motion tracking-error metrics.
                # Surfaces ``Mimic/{mpjpe, g_mpjpe, root_pos_err,
                # root_quat_err}_*`` scalars (mean / p10 / p50 / p95 across
                # motions with data) alongside the existing
                # ``MotionOutcome/*`` aggregates. Hot-path cost: one extra
                # ``get_links_pos`` per step + three (or four w/ anchor)
                # in-place adds. Shares the curriculum's DDP cadence.
                compute_per_motion_metrics=self.compute_per_motion_metrics,
                metric_anchor_link_name=self.metric_anchor_link_name,
                log_per_motion_metrics=self.log_per_motion_metrics,
                log_motion_id_to_filename=self.log_motion_id_to_filename,
            ),
        )
        return config


class GeneralMimicArchMod(TaskMod):
    """Swap the actor/critic network architecture by preset.

    The ``general_mimic`` defaults (3 layers, ``[512, 256, 128]``,
    ELU, scalar Gaussian with ``init_std=1.0``) were tuned at LAFAN1-scale
    (~75 motions). At AMASS-scale (~9 k motions) the clip-pose entropy is
    high enough that the small default trunk underfits — SONIC ships with
    a 6-layer ``[2048, 2048, 1024, 1024, 512, 512]`` MLP for exactly this
    regime.

    Presets
    -------
    ``default``
        3-layer MLP ``[512, 256, 128]``, ELU, scalar Gaussian with
        ``init_std=1.0``. Identical to the unmodified
        ``GeneralMimicConfig.runner_options.actor/critic`` defaults, so
        applying this preset is a no-op.

    ``sonic_flat``
        6-layer MLP ``[2048, 2048, 1024, 1024, 512, 512]``, Swish (SiLU)
        activation, per-DOF log Gaussian with ``init_std=0.05``. Matches the
        SONIC release's flat-MLP variant (``gear_sonic/.../actor_critic/mlp.yaml``).
        The tight ``init_std`` is load-bearing: with the much larger trunk and the
        default ``init_std=1.0``, action variance at step-0 saturates and stalls
        PPO — SONIC pairs the larger net with much less exploratory init noise.

    Only the actor/critic submodules of ``runner_options`` are touched. PPO
    hyperparameters, optimizer settings, num_steps_per_env, etc. stay at the
    ``GeneralMimicConfig`` defaults so this mod composes cleanly with the
    other ``general_mimic`` knobs.

    CLI usage::

        eden train --task general_mimic --arch_preset sonic_flat ...
    """

    def __init__(
        self,
        arch_preset: Annotated[
            Literal["default", "sonic_flat"],
            "Network architecture preset. 'default' is the 3-layer LAFAN1 recipe "
            "(no-op vs the unmodified config); 'sonic_flat' is SONIC's 6-layer "
            "MLP variant for AMASS-scale libraries.",
        ] = "default",
    ):
        self.arch_preset = arch_preset

    def apply(self, config: "GeneralMimicConfig") -> "GeneralMimicConfig":
        if self.arch_preset == "default":
            # No-op: ``GeneralMimicConfig.runner_options`` already holds these values, so a
            # serialized config that round-trips through this mod is identical to one that never touched it.
            return config
        if self.arch_preset == "sonic_flat":
            hidden_dims = [2048, 2048, 1024, 1024, 512, 512]
            # rsl_rl exposes SiLU under the name "swish" (see
            # ``rsl_rl.utils.resolve_nn_activation``); ``"silu"`` is rejected.
            activation = "swish"
            config.runner_options.actor.hidden_dims = list(hidden_dims)
            config.runner_options.actor.activation = activation
            config.runner_options.actor.distribution_cfg.init_std = 0.05
            config.runner_options.actor.distribution_cfg.std_type = "log"
            config.runner_options.critic.hidden_dims = list(hidden_dims)
            config.runner_options.critic.activation = activation
            return config
        raise ValueError(f"Unknown arch_preset {self.arch_preset!r}; expected one of 'default', 'sonic_flat'.")


_CommonObservations = dict(
    # ProtoMotions-style reference target encoded as anchor-relative pos /
    # ori on the anchor link itself (so the policy sees "where do I need
    # to put my torso next"). The current-frame anchor pos / ori below are
    # complemented by a ProtoMotions-style multi-future lookahead further
    # down (``motion_future_*``) — together they reproduce the
    # ``g1-bones-deploy`` reduced-coords target-poses obs.
    motion_dofs_pos=ObsTerm.configure(
        func=motion_dofs_pos,
        params={"entity_name": "robot"},
    ),
    motion_dofs_vel=ObsTerm.configure(
        func=motion_dofs_vel,
        params={"entity_name": "robot"},
    ),
    motion_anchor_pos_b=ObsTerm.configure(
        func=motion_links_pos_b,
        params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "anchor_link_name": _ANCHOR_LINK},
        noise=UniformNoise.configure(n_min=-0.25, n_max=0.25),
    ),
    motion_anchor_ori_b=ObsTerm.configure(
        func=motion_links_ori_b,
        params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "anchor_link_name": _ANCHOR_LINK},
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    # ProtoMotions ``g1-bones-deploy`` lookahead: K=4 future motion frames
    # at +1/+2/+4/+8 buffer ticks (+20/40/80/160 ms at frame_interval_s=0.02).
    # The future anchor orientation is brought into the robot's *current*
    # anchor frame; ``rot6d`` representation keeps the per-link 6D layout
    # consistent with ``motion_anchor_ori_b`` above so a single rotation
    # head can score both. Swap to ``representation="tan_norm"`` when
    # producing obs for a ProtoMotions-trained checkpoint
    # (see :class:`ProtoMotionV3Policy`).
    motion_future_dofs_pos=ObsTerm.configure(
        func=motion_future_dofs_pos,
        params={"entity_name": "robot", "frame_offsets": _FUTURE_FRAME_OFFSETS},
    ),
    motion_future_dofs_vel=ObsTerm.configure(
        func=motion_future_dofs_vel,
        params={"entity_name": "robot", "frame_offsets": _FUTURE_FRAME_OFFSETS},
    ),
    motion_future_anchor_pos_b=ObsTerm.configure(
        func=motion_future_links_pos_b,
        params={
            "entity_name": "robot",
            "links_name": [_ANCHOR_LINK],
            "anchor_link_name": _ANCHOR_LINK,
            "frame_offsets": _FUTURE_FRAME_OFFSETS,
        },
        noise=UniformNoise.configure(n_min=-0.25, n_max=0.25),
    ),
    motion_future_anchor_ori_b=ObsTerm.configure(
        func=motion_future_links_ori_b,
        params={
            "entity_name": "robot",
            "links_name": [_ANCHOR_LINK],
            "anchor_link_name": _ANCHOR_LINK,
            "frame_offsets": _FUTURE_FRAME_OFFSETS,
            "representation": "rot6d",
        },
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    # Proprio — body-frame base velocities + DOF state + last action.
    # Noise ranges match the BeyondMimic reference (small DOF noise, large
    # velocity noise — velocities are the most-corrupted real-robot
    # signal).
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.5, n_max=0.5),
    ),
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
    ),
    dofs_pos=observations.DofsPos.configure(
        entity_name="robot", offset_from_default=True, noise=UniformNoise.configure(n_min=-0.01, n_max=0.01)
    ),
    dofs_vel=observations.DofsVel.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-0.5, n_max=0.5)),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(modifiers=(GeneralMimicDataMod(), GeneralMimicArchMod()))
class GeneralMimicConfig(EdenMimicConfig):
    """Multi-motion G1 tracking policy.

    Build the policy with everything wired::

        from eden.tasks import TASK_REGISTRY
        cfg = TASK_REGISTRY.build("general_mimic")

    A bare ``GeneralMimicConfig()`` populates every field *except*
    ``motion_options``; the registered :class:`GeneralMimicDataMod`
    finishes setup (auto-applied on registry build). To override dataset
    paths or curriculum knobs::

        cfg = TASK_REGISTRY.build(
            "general_mimic",
            raw_data_path="/my/custom_subset",
            motion_weight_failure_amplify=1.5,
        )
    """

    # A field default, not a model_post_init assignment: model_post_init re-runs on every
    # reconstruction (load / --config merge) and would clobber user overrides back to this default.
    scene_options: SceneOptions = en.scenes.Empty(robot=_G1Options)

    env_options: EnvOptions = EnvOptions(
        # 4096 envs x 24 steps = ~98k env-steps per rsl-rl iter; with 5 s
        # (250-step) episodes at least ~400 episodes end per iter (more with
        # failures) — enough per-iter signal for the adaptive curriculum to
        # drive a stable update without batching multiple iters of outcomes.
        num_envs=4096,
        episode_length_s=5.0,
        iterations=10,
        ls_iterations=20,
        max_collision_pairs=20,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,
        batch_dofs_info=False,
        batch_links_info=True,  # RandomizeComShift writes per-env link COMs
        use_gjk_collision=False,
        enable_self_collision=False,
        enable_multi_contact=False,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=True,
            **_CommonObservations,
        ),
        # Critic gets the full anchor-relative body state — privileged
        # info that the policy can't see (per-tracked-link pos and ori in
        # the anchor frame). Closes the gap to ProtoMotions's reduced-
        # coords critic without needing a separate target-poses obs.
        critic=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=False,
            **_CommonObservations,
            body_pos_b=ObsTerm.configure(
                func=robot_body_pos_b,
                params={
                    "entity_name": "robot",
                    "anchor_link_name": _ANCHOR_LINK,
                    "tracked_link_names": _TRACKED_LINKS,
                },
            ),
            body_ori_b=ObsTerm.configure(
                func=robot_body_ori_b,
                params={
                    "entity_name": "robot",
                    "anchor_link_name": _ANCHOR_LINK,
                    "tracked_link_names": _TRACKED_LINKS,
                },
            ),
        ),
    )

    # ProtoMotions ``motion_tracker/g1-bones-deploy`` reference weights /
    # sigmas verbatim (BeyondMimic uses the same family). ProtoMotions'
    # per-region weighting (``use_region_weights=True``, auto-discovered from
    # the kinematic tree) is not ported; for this 14-link uniform tracking
    # set its effect is small.
    reward_options: RewardManagerOptions = RewardManagerOptions(
        motion_global_anchor_pos=RewardTerm.configure(
            func=motion_link_position_tracking_exp,
            weight=0.5,
            params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "std": 0.3},
        ),
        motion_global_anchor_ori=RewardTerm.configure(
            func=motion_link_orientation_tracking_exp,
            weight=0.5,
            params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "std": 0.4},
        ),
        motion_body_pos=MotionRelativeBodyPositionErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
            std=0.3,
        ),
        motion_body_ori=MotionRelativeBodyOrientationErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
            std=0.4,
        ),
        motion_body_lin_vel=MotionGlobalBodyLinearVelocityErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            links_name=_TRACKED_LINKS,
            std=1.0,
        ),
        motion_body_ang_vel=MotionGlobalBodyAngularVelocityErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            links_name=_TRACKED_LINKS,
            std=3.14,
        ),
        # Multi-future reward leg — scores the same buffer slice the policy
        # observes via ``motion_future_*``. At 0.5× the current-frame weight
        # so the current-frame term remains dominant while the future-frame
        # term provides a gradient pointed at the lookahead obs dims so PPO
        # can actually learn to use them. Frames past the clip end are
        # masked to zero via ``MotionReferenceData.validity`` in the reward
        # implementation, so envs near their motion's end don't get a fake
        # clamped-frame penalty.
        motion_global_anchor_pos_future=RewardTerm.configure(
            func=motion_link_position_tracking_exp,
            weight=0.25,
            params={
                "entity_name": "robot",
                "links_name": [_ANCHOR_LINK],
                "std": 0.3,
                "frame_offsets": _FUTURE_FRAME_OFFSETS,
            },
        ),
        motion_global_anchor_ori_future=RewardTerm.configure(
            func=motion_link_orientation_tracking_exp,
            weight=0.25,
            params={
                "entity_name": "robot",
                "links_name": [_ANCHOR_LINK],
                "std": 0.4,
                "frame_offsets": _FUTURE_FRAME_OFFSETS,
            },
        ),
        motion_body_pos_future=MotionRelativeBodyPositionErrorExp.configure(
            weight=0.5,
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
            std=0.3,
            frame_offsets=_FUTURE_FRAME_OFFSETS,
        ),
        motion_body_ori_future=MotionRelativeBodyOrientationErrorExp.configure(
            weight=0.5,
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
            std=0.4,
            frame_offsets=_FUTURE_FRAME_OFFSETS,
        ),
        action_rate_l2=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-1e-1,
        ),
        joint_limit=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
        self_collisions=RewardTerm.configure(
            func=rewards.self_collision_cost,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        # NOTE: ``motion_ends`` is intentionally NOT in the default termination
        # set. Empirically (single cartwheel clip; LAFAN1 multi-motion) it
        # collapses training: motion_ends-truncated envs leave only mid-clip
        # ``terminated`` failures driving the adaptive_sampling bin-failure EMA,
        # so the bin curriculum concentrates RSI starts on physically
        # infeasible mid-flip states. The motion-id curriculum's success
        # signal still works without it: ``MotionManager._record_motion_outcomes``
        # counts a non-terminated reset as a success only when its clip played
        # to its end, not by which term fired.
        # Adaptive thresholds: the gate stays at the base ``threshold``
        # while the motion ref is upright (anchor z >= ``root_height_threshold``)
        # and ramps up linearly to ``threshold * (1 + down_threshold)`` as
        # the ref drops to z=0. So crawl / kneel clips don't get killed for
        # tracking errors that would be unacceptable on a walk.
        #
        # Base thresholds 0.25 / 0.8 / 0.25 outperformed a tightened
        # (0.15, 0.5, 0.15) variant on the upright-majority LAFAN1
        # distribution (≈94% of frames have torso_z >= 0.5 m for G1). The
        # adaptive part only *widens* the gate for the long-tail crawl / fall
        # frames; the upright gate is the plain static threshold.
        anchor_pos=DoneTerm.configure(
            func=bad_anchor_pos_z_only,
            params={
                "entity_name": "robot",
                "anchor_link_name": _ANCHOR_LINK,
                "threshold": 0.25,
                "threshold_adaptive": True,
                "down_threshold": 0.75,
                "root_height_threshold": 0.5,
            },
        ),
        # On the unit gravity direction, as in BeyondMimic: ~78 deg off an upright reference.
        anchor_ori=DoneTerm.configure(
            func=bad_anchor_ori,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK, "threshold": 0.8},
        ),
        ee_body_pos=BadMotionLinkPosZOnly.configure(
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            threshold=0.25,
            threshold_adaptive=True,
            down_threshold=0.75,
            root_height_threshold=0.5,
            links_name=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Reference-state init: every reset spawns the env at the sampled
        # motion's chosen frame, ±small jitter. Without RSI the policy
        # would need many thousands of episodes per motion to discover the
        # right starting pose. Priority=100 → runs before other RESET
        # events that might fight the init pose.
        reset_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=_VELOCITY_RANGE,
            dofs_position_range=(-0.1, 0.1),
            priority=100,
        ),
        # Domain randomization — same recipe as BeyondMimic. Push +
        # COM-shift + DOF-default + foot friction cover the most-impactful
        # sim-to-real gaps for G1.
        push_robot=events.PushByVelocity.configure(
            mode=en.EventMode.INTERVAL,
            interval_range_s=(1.0, 3.0),
            entity_name="robot",
            velocity_range=_VELOCITY_RANGE,
        ),
        base_com_shift=events.RandomizeComShift.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["pelvis"],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.05, 0.05),
            com_shift_z_range=(-0.05, 0.05),
        ),
        dofs_default_pos=events.RandomizeDofsPosOffset.configure(
            mode=en.EventMode.STARTUP,
            action_term_name="dofs_pos_controller",
            entity_name="robot",
            dofs_pos_range=(-0.01, 0.01),
        ),
        foot_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["*_ankle_roll_link"],
            friction_range=(0.3, 1.2),
        ),
    )

    # Explicit PD on the 29 DOFs with BeyondMimic's ``G1_ACTION_SCALE``; the
    # action is a residual on the motion reference (``ReferenceSource.MOTION_REF``).
    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ExplicitPDController.configure(
            entity_name="robot",
            dofs_name=_G1Options.dofs_name,
            scale=G1_ACTION_SCALE,
            reference_source=ReferenceSource.MOTION_REF,
        )
    )

    # No metric terms: the motion manager already logs ``MotionWeights/*``,
    # ``MotionOutcome/*`` and (``compute_per_motion_metrics``) the per-motion
    # ``Mimic/*`` tracking errors.
    metric_options: MetricManagerOptions = MetricManagerOptions()

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        # 24 steps per env × 4096 envs = ~98k transitions per iter.
        num_steps_per_env=24,
        max_iterations=150000,
        save_interval=500,
        experiment_name="mimic_general",
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
        ),
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=0.001,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
