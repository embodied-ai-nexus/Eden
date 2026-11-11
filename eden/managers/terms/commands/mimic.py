"""Motion-clip command term (single- and multi-clip) and npz loaders.

A :class:`CommandTerm` that drives motion-imitation / tracking policies from one
or more BeyondMimic-format ``.npz`` clips. Single-clip is the ``M == 1`` case of
the multi-clip library. Shared by the frozen ONNX policy ports (sonic, twist2,
humanoid_gpt, protomotionv3, beyondmimic_moves) and the beyondmimic example.

Which motion path to use
------------------------
Eden has two motion-reference systems; pick by use case:

* **This command-term path** (``MotionCommand``) — lightweight, for
  **frozen-policy inference / single-clip tracking** on a plain ``RLEnvBase``.
  Loads npz clips verbatim and exposes the current frame (plus arbitrary future
  frames via :meth:`MotionCommand.gather_motion_frames`) directly off the
  command term. No dataset pipeline.
* **The manager path** (``eden.managers.motion_manager.MotionManager`` +
  ``eden.utils.motion_reference``) — **training-grade**: dataset adapters,
  continuous-time interpolation, a future-frame window, symmetry augmentation,
  and motion-id / start-time curricula. Owned by ``MimicRLEnvBase`` and consumed
  by the tracking reward / observation / termination terms. See
  ``docs/guides/motion-tracking.md``.

Both expose the same world-frame vocabulary — ``links_*_w`` (tracked-link poses)
and ``base_*_w`` (floating-base root) — so reward / metric code reads the same
field names regardless of source.

Config vs data vocabulary
-------------------------
The configurable ``links_name`` list names the tracked links; their per-frame
data is read as ``links_*_w``. Two ``links_name`` members are special: the
floating-base body (``base_link_name``, default ``links_name[0]``; exposed as
``base_*_w``, used to reset the robot root) and ``anchor_link_name`` (the
yaw-warp reference, exposed as ``anchor_*_w``). Both are resolved and validated
against ``links_name`` in :meth:`build`. The npz on-disk keys stay ``body_*_w``
(file format); only the Python accessors are ``links_*``.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from genesis.utils.geom import inv_transform_by_quat

import eden as en
from eden.managers.command_manager import COMMAND_TERM_REGISTRY, CommandTerm
from eden.options.robots.unitree_g1 import G1_DOFS29_TRACKED_LINKS_NAME
from eden.utils.geom import quat_from_euler_xyz, quat_mul
from eden.utils.misc import contiguous_range
from eden.utils.sample import sample_uniform
from eden.utils.string import columns_by_name, term_subject
from eden.utils.motion_reference.transforms import anchor_relative_links

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


_LEGACY_BFS_WARNING = (
    "motion npz has no 'body_names'; assuming the legacy pre-1.2.0 breadth-first G1 link order to "
    "resolve body columns by name (correct for clips generated before the genesis 1.2.0 "
    "breadth-first -> depth-first reorder). This assumes the clip's body axis is the FULL canonical "
    "G1 layout (G1_DOFS29_TRACKED_LINKS_NAME, 30 bodies in that exact order): a clip whose body axis "
    "is partial or reordered is silently mis-mapped (or indexes out of range) with no error. "
    "Regenerate it with examples/projects/mimic/beyondmimic/scripts/csv_to_npz.py, which embeds 'body_names'."
)

_LEGACY_NPZ_WARNING = (
    "motion npz has no 'body_names' and its links do not match the default G1 body order; falling "
    "back to Genesis link indices to slice body columns. This stays correct only if the npz was "
    "generated under the same internal link ordering as runtime. Regenerate it with "
    "examples/projects/mimic/beyondmimic/scripts/csv_to_npz.py so body columns resolve by name (required "
    "after the genesis 1.2.0 breadth-first -> depth-first kinematic-tree reordering)."
)


def resolve_body_cols(clip, links_idx, links_name) -> "torch.Tensor | list[int]":
    """Column indices into a clip's ``body_*_w`` arrays for the requested links.

    Resolution order:

    1. If the npz stores ``body_names``, resolve by name — independent of Genesis's internal link
       ordering (which changed breadth-first -> depth-first in genesis 1.2.0).
    2. Else, if the canonical G1 breadth-first link list (``G1_DOFS29_TRACKED_LINKS_NAME``) covers
       every requested link, resolve against that order. Every ``body_names``-less npz predates
       genesis 1.2.0 and was therefore written in the old breadth-first order, so this makes legacy
       G1 clips correct under a depth-first runtime instead of silently mis-sliced by ``idx_local``.
    3. Else (unrecognized body set, e.g. a non-G1 clip) fall back to the Genesis ``idx_local`` in
       ``links_idx`` — order-coupled, correct only when generated under the runtime's ordering.
    """
    if links_name is not None:
        if "body_names" in clip:
            return columns_by_name(clip["body_names"], links_name, what="motion npz 'body_names'")
        if all(n in G1_DOFS29_TRACKED_LINKS_NAME for n in links_name):
            en.logger.warning(_LEGACY_BFS_WARNING)
            return columns_by_name(G1_DOFS29_TRACKED_LINKS_NAME, links_name, what="G1_DOFS29_TRACKED_LINKS_NAME")
    en.logger.warning(_LEGACY_NPZ_WARNING)
    return links_idx


class MotionLoader:
    """
    Loader for a single motion clip.

    A standalone verbatim loader for callers that only need one clip;
    :class:`MotionCommand` itself loads through :class:`MultiMotionLoader`
    (single-clip is its M=1 case).

    Parameters
    ----------
    motion_file: str
        Path to the motion file.
    links_idx: torch.Tensor
        Genesis ``idx_local`` of the bodies to load; legacy fallback used only when the npz has
        no ``body_names``.
    device: torch.device
        Device to load the data on.
    links_name: list[str] | None
        Names of the bodies to load, in the desired output order. When given and the npz carries
        ``body_names``, columns are resolved by name (order-independent of Genesis); otherwise
        ``links_idx`` is used.
    """

    def __init__(
        self,
        motion_file: str,
        links_idx: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        links_name: list[str] | None = None,
    ) -> None:
        data = np.load(motion_file)
        body_cols = resolve_body_cols(data, links_idx, links_name)

        def _load(key: str) -> torch.Tensor:
            return torch.tensor(data[key], dtype=torch.float32, device=device)

        self.dofs_pos = _load("joint_pos")
        self.dofs_vel = _load("joint_vel")
        self.time_step_total = self.dofs_pos.shape[0]
        # Body axes are sliced to the requested links; quats are wxyz (MuJoCo convention).
        self.links_pos_w = _load("body_pos_w")[:, body_cols]
        self.links_quat_w = _load("body_quat_w")[:, body_cols]
        self.links_lin_vel_w = _load("body_lin_vel_w")[:, body_cols]
        self.links_ang_vel_w = _load("body_ang_vel_w")[:, body_cols]


class MultiMotionLoader:
    """Batched verbatim loader for a list of BeyondMimic ``.npz`` clips.

    Mirrors :class:`MotionLoader` field-for-field, but stacks ``M`` clips of
    possibly different lengths into padded ``(M, T_max, ...)`` tensors plus a
    per-clip ``frame_total (M,)``. Fields are loaded VERBATIM (no interpolation,
    no velocity recompute) so each clip is bit-identical to the single-clip path.
    With ``M == 1`` (the single-clip case) ``T_max == T`` and no padding occurs,
    so ``library[0, t]`` is plain single-clip indexing.

    Padding edge-replicates each clip's last valid frame (never zeros): a zero
    quaternion would NaN ``yaw_quat``, and an over-read then degrades to a
    "hold last frame" end-of-motion behavior (the per-env wrap keeps reads in
    range anyway).
    """

    def __init__(
        self,
        motion_files: list[str],
        links_idx: torch.Tensor,
        device: torch.device,
        links_name: list[str] | None = None,
    ) -> None:
        if not motion_files:
            raise ValueError("MultiMotionLoader: motion_files is empty")
        self.motion_refs = [os.path.splitext(os.path.basename(f))[0] for f in motion_files]
        clips = [np.load(f) for f in motion_files]
        lengths = [int(c["joint_pos"].shape[0]) for c in clips]
        # All fields of a clip must share its frame count; otherwise the per-field
        # padding below would misalign body vs joint frames silently.
        for f, c in zip(motion_files, clips):
            for k in ("joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                if k in c and int(c[k].shape[0]) != int(c["joint_pos"].shape[0]):
                    raise ValueError(
                        f"{f}: field '{k}' has {c[k].shape[0]} frames but joint_pos has {c['joint_pos'].shape[0]}"
                    )
        # Resolve each clip's body columns by name (order-independent of Genesis) when the npz
        # provides ``body_names``, else fall back to the Genesis idx_local in ``links_idx``.
        body_cols = [resolve_body_cols(c, links_idx, links_name) for c in clips]
        m, t_max = len(clips), max(lengths)
        self.frame_total = torch.tensor(lengths, dtype=torch.long, device=device)
        self.time_step_total = t_max  # nominal; per-clip lengths live in frame_total

        def _stack(key: str, link_sliced: bool) -> torch.Tensor:
            out: torch.Tensor | None = None
            for i, clip in enumerate(clips):
                t = torch.tensor(clip[key], dtype=torch.float32, device=device)
                if link_sliced:
                    t = t[:, body_cols[i]]
                if out is None:
                    out = t.new_zeros((m, t_max, *t.shape[1:]))
                length = t.shape[0]
                out[i, :length] = t
                if length < t_max:  # edge-replicate the last valid frame
                    out[i, length:] = t[length - 1]
            return out

        self.dofs_pos = _stack("joint_pos", False)  # (M, T_max, 29)
        self.dofs_vel = _stack("joint_vel", False)  # (M, T_max, 29)
        self.links_pos_w = _stack("body_pos_w", True)  # (M, T_max, n_bodies, 3)
        self.links_quat_w = _stack("body_quat_w", True)  # (M, T_max, n_bodies, 4)
        self.links_lin_vel_w = _stack("body_lin_vel_w", True)
        self.links_ang_vel_w = _stack("body_ang_vel_w", True)


_RANGE_AXES: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")


@COMMAND_TERM_REGISTRY.register()
class MotionCommand(CommandTerm):
    """Motion command for mimicking one or more motion clips.

    Backed by a :class:`MultiMotionLoader` clip library and a per-env
    ``(motion_id, time_steps)`` gather. A single clip is just the ``M == 1``
    case: every env stays on clip 0 and the gather reduces to plain
    single-clip indexing.
    Passing ``motion_files=[...]`` (or a list to :func:`motion_command_options`)
    assigns each env its own clip, re-sampled on every reset/wrap so all clips
    are exercised regardless of ``num_envs`` vs ``num_clips``.

    Sampling modes: ``start`` / ``uniform`` work for any clip count; ``adaptive``
    (the failure-bin start-time curriculum) requires a single clip, since the
    bins partition one clip's timeline.
    """

    entity_name: str = ""
    ghost_entity_name: str = ""
    motion_file: str = "motion.npz"
    # Multi-clip spelling. When non-empty, takes precedence over ``motion_file``;
    # ``motion_file`` is the single-clip shorthand for ``[motion_file]``.
    motion_files: list[str] = []
    anchor_link_name: str = "torso_link"
    # Floating-base (root) body; a member of ``links_name``. ``None`` selects ``links_name[0]`` (the
    # "row 0 is the base" convention). Naming it (or reordering ``links_name``) is safe: ``build`` resolves and
    # validates it, and ``base_*_w`` writes this body's pose as the robot root.
    base_link_name: str | None = None
    # None auto-detects the free base joint by joint TYPE (works for MJCF and URDF robots
    # alike); name it only to disambiguate a multi-free-joint entity.
    base_joint_name: str | None = None
    links_name: list[str] = [
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
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    dofs_position_range: tuple[float, float] = (-0.1, 0.1)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_failure_rate_max_over_mean: float = 10.0
    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        self._cache_robot_dofs_pos = None
        self._cache_robot_dofs_vel = None
        self._cache_robot_links_pos_w = None
        self._cache_robot_links_quat_w = None
        self._cache_robot_links_lin_vel_w = None
        self._cache_robot_links_ang_vel_w = None

        self.pose_ranges = self._build_range_tensor(self.pose_range)
        self.velocity_ranges = self._build_range_tensor(self.velocity_range)

        self.robot = None
        self.robot_base_dofs_index = None
        self.base_and_dofs_index = None
        self._reset_through_mask = False
        self._reset_dofs_idx = None
        self._reset_qs_idx = None
        self.motion_anchor_link_index = None
        self.base_link_index = None
        self.links_idx = None

        self.motion = None
        self.num_clips = None
        self.time_steps = None
        # Per-env assigned clip (all zeros for single-clip) and that clip's length.
        self.motion_id = None
        self.frame_total = None
        self.links_pos_relative_w = None
        self.links_quat_relative_w = None

        self.bin_count = None
        # Both EMAs are keyed by the bin the episode STARTED in and accumulated
        # at the same resample events, so they stay phase-aligned. Their ratio
        # is a bounded per-episode failure RATE in [0, 1] (see _adaptive_sampling):
        #   bin_failed_count[k] — episodes that started in bin k AND terminated
        #   bin_starts_count[k] — episodes that started in bin k (any outcome)
        self.bin_failed_count = None
        self._current_bin_failed = None
        self.bin_starts_count = None
        self._current_bin_starts = None
        # Per-env start bin of the currently-running episode (the bin we
        # attribute that episode's outcome to when it ends).
        self.start_bin = None
        self.kernel = None

    def _build_range_tensor(self, ranges: dict[str, tuple[float, float]]) -> torch.Tensor:
        return torch.tensor(
            [ranges.get(axis, (0.0, 0.0)) for axis in _RANGE_AXES],
            device=self.device,
        )

    def build(self) -> None:
        self.robot = self.resolve_entity(self.entity_name)

        # NOTE: avoid tensor indexing by passing int
        robot_base_dofs_index = self.robot.find_free_base_dofs_idx_local(self.base_joint_name)
        self.robot_base_dofs_index = torch.as_tensor(
            robot_base_dofs_index, dtype=torch.long, device=self.device
        ).contiguous()
        self.base_and_dofs_index = torch.cat([self.robot_base_dofs_index, self.robot.dofs_idx_local]).contiguous()

        # Resolve a links_name member to its row in ``links_*_w``, failing loudly on a mistyped or reordered
        # name instead of a bare index error or (for the base) a silent wrong-link root reset.
        def links_index(link_name: str, field_name: str) -> int:
            if link_name not in self.links_name:
                raise ValueError(
                    f"{term_subject(self)}: {field_name}={link_name!r} must be a member "
                    f"of links_name; got links_name={self.links_name}"
                )
            return self.links_name.index(link_name)

        self.motion_anchor_link_index = links_index(self.anchor_link_name, "anchor_link_name")
        self.base_link_index = 0 if self.base_link_name is None else links_index(self.base_link_name, "base_link_name")

        self.links_idx = torch.tensor(
            self.robot.find_named_links_idx_local(self.links_name, preserve_order=True)[1],
            dtype=torch.long,
            device=self.device,
        )

        # Single-clip is the M=1 case of the clip library: ``motion_file`` is the
        # single-clip shorthand for ``motion_files=[motion_file]``.
        motion_files = list(self.motion_files) if self.motion_files else [self.motion_file]
        self.num_clips = len(motion_files)
        if self.num_clips > 1 and self.sampling_mode == "adaptive":
            raise ValueError(
                "adaptive sampling requires a single clip (the failure-bin curriculum partitions "
                f"one clip's timeline); got {self.num_clips} clips. Use sampling_mode 'start' or 'uniform'."
            )

        self.motion = MultiMotionLoader(motion_files, self.links_idx, device=self.device, links_name=self.links_name)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_total = self.motion.frame_total[self.motion_id]  # (N,) per-env clip length
        self.links_pos_relative_w = torch.zeros(self.num_envs, len(self.links_name), 3, device=self.device)
        self.links_quat_relative_w = torch.zeros(self.num_envs, len(self.links_name), 4, device=self.device)
        self.links_quat_relative_w[:, :, 0].fill_(1.0)

        # Adaptive failure-bin curriculum state. The bins partition the (single)
        # clip's timeline; for multi-clip these stay allocated but unused
        # (``sampling_mode == "adaptive"`` is rejected above for M > 1).
        self.bin_count = int(self.motion.time_step_total // (1 / self._env.dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.bin_starts_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_starts = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # How `_resample_command` hands the reset state to Genesis. Where its setters can keep the
        # bool env mask on the device — a `torch.where` into the state and a masked
        # forward-kinematics kernel — they take every env's row and a `range` for each DOF axis
        # (`RigidEntity.setters_take_every_row`). Otherwise, e.g. under a `dofs_name` reordered away from the
        # asset's or with hibernation on, they take the selected rows and the DOF indices: Genesis'
        # indexed path, which pays a `nonzero()` GPU->CPU sync per setter to turn the mask into indices.
        dofs_range = contiguous_range(self.base_and_dofs_index)
        qs_range = contiguous_range(self.robot.qs_idx_local)
        self._reset_through_mask = self.robot.setters_take_every_row and dofs_range is not None and qs_range is not None
        self._reset_dofs_idx = dofs_range if self._reset_through_mask else self.base_and_dofs_index
        self._reset_qs_idx = qs_range if self._reset_through_mask else self.robot.qs_idx_local

        self._init_sampling_stats()
        # Initial per-env clip assignment. Gated on M > 1 so the single-clip path
        # draws no RNG here (motion_id is already all-zeros) and its sampling
        # stream does not depend on the clip library.
        if self.num_clips > 1:
            self._sample_motions(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._update_robot_state_cache()

    def _init_sampling_stats(self) -> None:
        """Register the per-env sampling-telemetry stat keys (logged by the command
        manager). The adaptive curriculum fills them in :meth:`_adaptive_sampling`;
        for multi-clip / non-adaptive modes they stay at their defaults so the
        logged schema is consistent across clip counts and sampling modes.
        """
        self.stats["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.stats["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.stats["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float) -> None:
        self._update_robot_state_cache()
        super().compute(dt)

    def post_physics_refresh(self) -> None:
        # The robot-state cache is refreshed at TWO distinct points per control step, by
        # design — they snapshot different states:
        #   1. here (post-physics, before reward) so reward/termination grade S_N, the
        #      pre-reset post-physics state;
        #   2. in compute() (which the command manager runs AFTER _reset_idx) so the
        #      cache then reflects the POST-reset state of just-reset envs, which the
        #      command and the end-of-step observation read.
        # Dropping either leaves one of those consumers a control step stale. See RLEnvBase.step.
        self._update_robot_state_cache()
        # Advance-then-grade: the clip clock was advanced (and wrapped) in
        # pre_physics_advance, so recompute the anchor-relative transforms against
        # the advanced reference and the just-refreshed post-physics robot cache.
        # This is what makes reward/termination grade the temporally-aligned frame.
        self._update_relative_transforms()

    def _update_robot_state_cache(self):
        self._cache_robot_dofs_pos = self.robot.get_dofs_pos()
        self._cache_robot_dofs_vel = self.robot.get_dofs_vel()
        self._cache_robot_links_pos_w = self.robot.get_links_pos(ls_idx_local=self.links_idx)
        self._cache_robot_links_quat_w = self.robot.get_links_quat(ls_idx_local=self.links_idx)
        # csv_to_npz generates the reference body_lin_vel_w (npz key) with get_links_vel() at its
        # DEFAULT ref="link_origin", so the robot velocity in the reward must also be link-origin.
        # A COM-frame ref="link_com" velocity is off by omega x r_com — large and time-varying on fast
        # motions, making the velocity reward unsatisfiable and high-variance. mjlab/IsaacLab use
        # link-origin (body_link_lin_vel_w) on both sides.
        self._cache_robot_links_lin_vel_w = self.robot.get_links_vel(ls_idx_local=self.links_idx, ref="link_origin")
        self._cache_robot_links_ang_vel_w = self.robot.get_links_ang(ls_idx_local=self.links_idx)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.dofs_pos, self.dofs_vel], dim=1)

    # --- per-env-motion gathers ---
    # Every motion-side read gathers ``library[motion_id, time_steps]`` so N envs
    # roll N (possibly different) clips at once. With M == 1 the motion_id axis is
    # all zeros and this reduces to plain single-clip indexing.

    def _gather_now(self, field: torch.Tensor) -> torch.Tensor:
        """``field (M, T_max, ...)`` -> ``(N, ...)`` at each env's (motion, frame)."""
        return field[self.motion_id, self.time_steps]

    def _anchor(self, field_name: str) -> torch.Tensor:
        """Anchor body's row of motion field ``field_name`` at each env's (motion, frame)."""
        return getattr(self.motion, field_name)[self.motion_id, self.time_steps, self.motion_anchor_link_index]

    @property
    def dofs_pos(self) -> torch.Tensor:
        return self._gather_now(self.motion.dofs_pos)

    @property
    def dofs_vel(self) -> torch.Tensor:
        return self._gather_now(self.motion.dofs_vel)

    @property
    def links_pos_w(self) -> torch.Tensor:
        return self._gather_now(self.motion.links_pos_w)

    @property
    def links_quat_w(self) -> torch.Tensor:
        return self._gather_now(self.motion.links_quat_w)

    @property
    def links_lin_vel_w(self) -> torch.Tensor:
        return self._gather_now(self.motion.links_lin_vel_w)

    @property
    def links_ang_vel_w(self) -> torch.Tensor:
        return self._gather_now(self.motion.links_ang_vel_w)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._anchor("links_pos_w")

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._anchor("links_quat_w")

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._anchor("links_lin_vel_w")

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._anchor("links_ang_vel_w")

    # --- floating-base (root) accessors ---
    # ``base_link_index`` row of each ``links_*_w`` (see ``base_link_name``); mirror ``MotionReferenceData``'s
    # ``base_*`` fields.

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.links_pos_w[:, self.base_link_index]

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.links_quat_w[:, self.base_link_index]

    @property
    def base_lin_vel_w(self) -> torch.Tensor:
        return self.links_lin_vel_w[:, self.base_link_index]

    @property
    def base_ang_vel_w(self) -> torch.Tensor:
        return self.links_ang_vel_w[:, self.base_link_index]

    # --- motion-source-agnostic accessors ---
    # A frozen policy (e.g. SONIC) that reads future motion frames directly off
    # the command term uses these instead of indexing ``self.motion.X``, so the
    # same policy code drives any clip count without changes.

    def frame_total_per_env(self) -> torch.Tensor:
        """Per-env motion length in frames (each env's assigned clip length;
        a constant vector when there is a single clip).
        """
        return self.frame_total

    def clip_finished(self) -> torch.Tensor:
        """Per-env ``(N,)`` bool: has this env's clip reached its final frame?

        The command path's answer to "is the motion over", and the counterpart of
        :meth:`~eden.managers.motion_manager.MotionManager.get_motion_phase` ``>= 1`` on the
        manager path. It is only meaningful because :meth:`advance_reference` *holds* the clock
        at ``frame_total - 1`` instead of wrapping: in simulation
        :meth:`pre_physics_advance` resamples at the boundary, so the clock regresses and this
        goes ``False`` again, which is correct — a wrapped clip is not a finished one.

        ``DeploymentBase.motion_finished`` reduces this for the deploy loop's end-of-clip
        handover; nothing in the training loop reads it.
        """
        return self.time_steps >= self.frame_total - 1

    def gather_motion_frames(self, field_name: str, frame_idx: torch.Tensor) -> torch.Tensor:
        """Gather motion field ``field_name`` at per-env frame indices.

        ``frame_idx`` is ``(N, K)`` long -> returns ``(N, K, ...)``. Each env
        indexes its own assigned clip (``motion_id``); with a single clip every
        env indexes the one shared clip's time axis.
        """
        field = getattr(self.motion, field_name)  # (M, T_max, ...)
        n, k = frame_idx.shape
        mid = self.motion_id[:, None].expand(n, k)
        return field[mid, frame_idx]  # (N, K, ...)

    @property
    def robot_dofs_pos(self) -> torch.Tensor:
        return self._cache_robot_dofs_pos

    @property
    def robot_dofs_vel(self) -> torch.Tensor:
        return self._cache_robot_dofs_vel

    @property
    def robot_links_pos_w(self) -> torch.Tensor:
        return self._cache_robot_links_pos_w

    @property
    def robot_links_quat_w(self) -> torch.Tensor:
        return self._cache_robot_links_quat_w

    @property
    def robot_links_lin_vel_w(self) -> torch.Tensor:
        return self._cache_robot_links_lin_vel_w

    @property
    def robot_links_ang_vel_w(self) -> torch.Tensor:
        return self._cache_robot_links_ang_vel_w

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self._cache_robot_links_pos_w[:, self.motion_anchor_link_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self._cache_robot_links_quat_w[:, self.motion_anchor_link_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self._cache_robot_links_lin_vel_w[:, self.motion_anchor_link_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self._cache_robot_links_ang_vel_w[:, self.motion_anchor_link_index]

    # NOTE: the sampling helpers below and `_resample_command` take the `(num_envs,)` bool mask
    # `_resample_command` is handed, and never index with it: every draw is full-batch and kept
    # through `torch.where`, and counts are weighted scatter-adds. A bool mask used as an index is
    # sized on the host — a GPU->CPU sync per gather or write, ~20 of them on this path — and it
    # runs on every reset and on every clip wrap, which at training scale is most control steps.

    def _sample_motions(self, envs_idx: torch.Tensor) -> None:
        """Assign each selected env a fresh random clip (uniform over the M loaded
        clips) and refresh its per-env clip length. Re-run on every reset/wrap so
        coverage of all clips does not depend on num_envs vs num_clips. (Single
        clip: a no-op assignment of clip 0 — callers gate this on M > 1.)
        """
        m = len(self.motion.motion_refs)
        motion_id = torch.randint(0, m, (self.num_envs,), device=self.device)
        self.motion_id.copy_(torch.where(envs_idx, motion_id, self.motion_id))
        self.frame_total = self.motion.frame_total[self.motion_id]  # (N,) per-env clip length

    def _adaptive_sampling(self, envs_idx: torch.Tensor):
        # Record the ENDING episodes' outcomes against the bin they STARTED in
        # (not the bin they died in). ``start_bin`` was stored when these envs
        # were last (re)sampled; ``bin_starts_count`` is the per-episode
        # denominator and ``bin_failed_count`` the numerator, so their ratio is
        # a bounded failure RATE in [0, 1]: "of the episodes that started in
        # bin k, what fraction terminated early". Accumulate (not overwrite) so
        # all-False / multi-call resets within a step don't wipe data before
        # _update_command() consumes it in the EMA.
        # Bootstrap resets (``episode_length_buf == 0``) carry no real outcome:
        # the env hasn't run yet, so its ``start_bin`` is still the initial 0.
        # Recording those would seed the starts EMA with artificial bin-0 starts
        # and skew the failure-rate denominator for early adaptive sampling. Drop
        # them here, mirroring ``MotionManager._record_adaptive_bin_outcomes``.
        ended = envs_idx & (self._env.episode_length_buf > 0)
        failed = ended & self._env.termination_manager.terminated
        # Every env scatters into its start bin, weighted 1 if its episode is being counted and 0
        # otherwise — the same counts as a `bincount` of the selected bins, without the gather.
        self._current_bin_starts.scatter_add_(0, self.start_bin, ended.float())
        self._current_bin_failed.scatter_add_(0, self.start_bin, failed.float())

        # Per-START-bin failure RATE. A bounded per-episode rate (unlike an
        # unbounded per-step hazard) self-regulates: as the policy
        # masters a region its start-bin failure rate falls toward the rest, the
        # normalized distribution flattens back toward uniform, and no part of
        # the clip is starved. Attributing to the start bin also seeds episodes
        # just before hard moments (a run-up) rather than cold-starting at them.
        bin_failure_rate = self.bin_failed_count / self.bin_starts_count.clamp_min(1e-12)
        sampling_probabilities = torch.nn.functional.pad(
            bin_failure_rate.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        # Concentration clamp: cap each bin at ``max_over_mean`` times the mean
        # rate so no single bin can absorb arbitrarily much mass (the uniform
        # mix below only bounds the floor, not the peak).
        if self.adaptive_failure_rate_max_over_mean > 0:
            mean_rate = sampling_probabilities.mean()
            sampling_probabilities = sampling_probabilities.clamp_max(
                mean_rate * self.adaptive_failure_rate_max_over_mean
            )

        uniform = torch.full_like(sampling_probabilities, 1.0 / float(self.bin_count))
        total = sampling_probabilities.sum()
        # All-zero rate (e.g. before the first failures, or a fully-mastered
        # clip) -> fall back to uniform.
        p_fail = torch.where(total > 1e-12, sampling_probabilities / total.clamp_min(1e-12), uniform)
        r = self.adaptive_uniform_ratio
        sampling_probabilities = (1.0 - r) * p_fail + r * uniform

        sampled_bins = torch.multinomial(sampling_probabilities, self._env.num_envs, replacement=True)
        # Remember the start bin so the next reset of these envs attributes the
        # episode's outcome to where it started.
        self.start_bin.copy_(torch.where(envs_idx, sampled_bins, self.start_bin))
        time_steps = (
            (sampled_bins + sample_uniform(0.0, 1.0, (self._env.num_envs,), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        self.time_steps.copy_(torch.where(envs_idx, time_steps, self.time_steps))

        # Update stats.
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.stats["sampling_entropy"][:] = H_norm
        self.stats["sampling_top1_prob"][:] = pmax
        self.stats["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_sampling(self, envs_idx: torch.Tensor):
        # Per-env clip-length-aware uniform start frame in [0, frame_total). With
        # a single clip every env shares the one clip length (uniform over the
        # whole clip).
        ft = self.frame_total
        sampled = (torch.rand(ft.shape, device=self.device) * ft.to(torch.float32)).long()
        self.time_steps.copy_(torch.where(envs_idx, torch.minimum(sampled, ft - 1), self.time_steps))
        self.stats["sampling_entropy"].fill_(1.0)  # Maximum entropy for uniform.
        self.stats["sampling_top1_prob"].fill_(1.0 / self.bin_count)
        self.stats["sampling_top1_bin"].fill_(0.5)  # No specific bin preference.

    def _sample_time_steps(self, envs_idx: torch.Tensor):
        """Set per-env ``(motion_id, time_steps)`` for ``envs_idx`` per the
        sampling mode. Multi-clip first draws a fresh clip per env; ``adaptive``
        is single-clip only (enforced in :meth:`build`).
        """
        if self.num_clips > 1:
            self._sample_motions(envs_idx)  # fresh clip per env, then a start frame within it
        if self.sampling_mode == "start":
            self.time_steps[envs_idx] = 0  # a scalar write through the mask: a masked fill, no sync
        elif self.sampling_mode == "adaptive":
            self._adaptive_sampling(envs_idx)
        elif self.sampling_mode == "uniform":
            self._uniform_sampling(envs_idx)

    def _resample_command(self, envs_idx: torch.Tensor):
        self._sample_time_steps(envs_idx)

        # Every env's reset state, full-batch (see the NOTE above `_sample_motions`); the setters
        # write only the envs `envs_idx` selects.
        rand_samples = sample_uniform(
            self.pose_ranges[:, 0],
            self.pose_ranges[:, 1],
            (self._env.num_envs, 6),
            device=self.device,
        )
        root_pos = self.base_pos_w + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, self.base_quat_w)
        rand_samples = sample_uniform(
            self.velocity_ranges[:, 0],
            self.velocity_ranges[:, 1],
            (self._env.num_envs, 6),
            device=self.device,
        )
        root_lin_vel = self.base_lin_vel_w + rand_samples[:, :3]
        root_ang_vel = self.base_ang_vel_w + rand_samples[:, 3:]

        dofs_pos = self.dofs_pos
        dofs_pos = dofs_pos + sample_uniform(
            lower=self.dofs_position_range[0],
            upper=self.dofs_position_range[1],
            size=dofs_pos.shape,
            device=dofs_pos.device,  # type: ignore
        )
        soft_dofs_pos_limits = self.robot.soft_dofs_pos_limits
        dofs_pos = torch.clip(dofs_pos, soft_dofs_pos_limits[:, :, 0], soft_dofs_pos_limits[:, :, 1])

        # Genesis stores a free joint's rotational DOF velocity in the base BODY frame (MuJoCo
        # qvel convention), but root_ang_vel is world-frame (base_ang_vel_w + a world-axis
        # perturbation). Rotate world->body using root_ori (the same, possibly-perturbed, quat
        # written to qpos below) so the body frame matches the reset pose. Linear DOFs stay world.
        root_ang_vel_b = inv_transform_by_quat(root_ang_vel, root_ori)
        # Every env's row where Genesis keeps the mask on the device, else just the selected rows
        # (`build` says which).
        rows = slice(None) if self._reset_through_mask else envs_idx
        # Order matters: set_dofs_vel(skip_forward=True) FIRST, set_qpos(zero_velocity=False) SECOND, so
        # set_qpos's FK sees the fresh qvel and leaves both solver forward flags True (the next scene.step
        # skips a full-batch velocity FK). See MotionEndResync._teleport for the analysis.
        self.robot.set_dofs_vel(
            torch.cat([root_lin_vel, root_ang_vel_b, self.dofs_vel], dim=1)[rows],
            dofs_idx_local=self._reset_dofs_idx,
            envs_idx=envs_idx,
            skip_forward=True,
        )
        self.robot.set_qpos(
            torch.cat([root_pos, root_ori, dofs_pos], dim=1)[rows],
            qs_idx_local=self._reset_qs_idx,
            envs_idx=envs_idx,
            zero_velocity=False,
        )

    def pre_physics_advance(self) -> None:
        """Advance the clip clock (+ wrap-teleport) BEFORE the physics sub-steps.

        Advance-then-grade: the reference frame is advanced before physics so the
        post-physics robot (graded at reward/termination time) is compared against
        the temporally-aligned frame — matching :class:`MotionManager`, whose clock
        advances at the start of ``MimicRLEnvBase.step``. Advancing in
        :meth:`compute` (after reward) would grade against a
        one-control-step-stale reference. Wrap-teleport runs here, before physics,
        so the teleported robot is forward-stepped and its forward kinematics stay
        fresh at reward time (no stale-FK spurious terminations).
        """
        self.time_steps += 1
        wrapped = self.time_steps >= self.frame_total  # per-env clip length
        # The bool mask `_resample_command` gets from `_resample` too, so the wrap costs no sync of
        # its own. The `.any()` gate is one: an all-False mask would write nothing, but the
        # full-batch reset draws, math and setter calls still cost ~1 ms at 4096 envs, far more
        # than the sync.
        if wrapped.any():
            self._resample_command(wrapped)

    def advance_reference(self) -> None:
        """Advance deployment playback, holding the last frame without a wrap teleport."""
        self.time_steps.copy_(torch.minimum(self.time_steps + 1, self.frame_total - 1))

    def rewind_reference(self) -> None:
        """Rewind the clip clock to frame 0 for every env — the deploy path's handover.

        Clock only: no :meth:`_resample_command`, so the robot is neither teleported nor
        pose-randomized and the clip assignment stands. That is the whole point on hardware,
        where the sim's reset-state initialization has nothing to act on, and it mirrors what
        ``MotionManager.rewind_to_start`` does for the manager path.
        """
        if self.time_steps is not None:
            self.time_steps.zero_()
            # Deployment restarts only on reset/START, not on the training timer.
            # Otherwise compute() can still resample and initialize the robot pose.
            self.time_left.fill_(float("inf"))

    def _update_command(self):
        # Simulation wraps in pre_physics_advance; deployment holds at the end
        # through advance_reference. compute() only publishes transforms for
        # the current frame and updates adaptive-sampling statistics.
        self._update_relative_transforms()

        if self.sampling_mode == "adaptive":
            # Blend the per-episode start-bin accumulators (filled at resample
            # in _adaptive_sampling, so numerator and denominator stay
            # phase-aligned) into their EMAs, then zero the accumulators.
            self.bin_failed_count = (
                self.adaptive_alpha * self._current_bin_failed + (1 - self.adaptive_alpha) * self.bin_failed_count
            )
            self.bin_starts_count = (
                self.adaptive_alpha * self._current_bin_starts + (1 - self.adaptive_alpha) * self.bin_starts_count
            )
            self._current_bin_failed.zero_()
            self._current_bin_starts.zero_()

    def _update_relative_transforms(self) -> None:
        """Compute anchor-relative body pose buffers from the current motion ref
        and robot state. Reads the motion via the ``anchor_*``/``links_*``
        properties, so per-env-motion gathers give per-env transforms for free.
        """
        # Shared yaw-warp (also used by the motion-manager tracking helpers); it
        # applies the robot-anchor NaN guard internally.
        self.links_pos_relative_w, self.links_quat_relative_w = anchor_relative_links(
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            self.anchor_pos_w,
            self.anchor_quat_w,
            self.links_pos_w,
            self.links_quat_w,
        )

    def draw_vis(self) -> None:
        if not self.ghost_entity_name:
            return

        if self._env.training:
            en.logger.warning("Drawing ghost is expensive in training. Please make sure if this is intended.")

        ghost_entity: Entity = self.resolve_entity(self.ghost_entity_name)
        # NOTE: fixed base entity must use set_pos and set_quat to set the pose
        ghost_entity.set_pos(self.base_pos_w)
        ghost_entity.set_quat(self.base_quat_w)
        ghost_entity.set_dofs_pos(self.dofs_pos)


def motion_command_options(
    motion_file: str | list[str],
    *,
    anchor_link_name: str,
    links_name: list[str],
    entity_name: str = "robot",
    ghost_entity_name: str = "",
    debug_vis: bool = False,
    sampling_mode: str = "start",
    **overrides,
) -> dict[str, "CommandTermOptions"]:
    """Build the ``{"motion": <term>}`` command_options for a motion-tracking policy.

    Always builds a single :class:`MotionCommand`; a list of clip paths populates
    its ``motion_files`` (each env rolls its own assigned clip) and a single path
    populates ``motion_file`` (the M=1 case). The accessor interface is identical
    either way, so a policy reads motion source-agnostically — a new ONNX motion
    policy reuses this helper by passing only its body/anchor names. ``overrides``
    replace shared term fields.
    """
    shared = dict(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=debug_vis,
        entity_name=entity_name,
        ghost_entity_name=ghost_entity_name,
        anchor_link_name=anchor_link_name,
        links_name=links_name,
        pose_range={},
        velocity_range={},
        dofs_position_range=(0.0, 0.0),
        sampling_mode=sampling_mode,
    )
    shared.update(overrides)
    if isinstance(motion_file, (list, tuple)):
        return dict(motion=MotionCommand.configure(motion_files=list(motion_file), **shared))
    return dict(motion=MotionCommand.configure(motion_file=motion_file, **shared))
