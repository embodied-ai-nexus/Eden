"""TCRS terrain-adaptation dataset modifier.

Runs the TCRS terrain-adaptive motion optimizer
(:mod:`eden.extensions.motion_optimization.tcrs`) over every clip of a motion dataset:
MPPI foot-swing planning against a terrain height field, support-aware pelvis
height re-targeting, and whole-body Jacobian IK on a MuJoCo model of the
robot. Flat-ground clips come out with footsteps and pelvis height that track
the configured terrain.

The terrain can be described three ways (all reduce to one height field):

- ``TerrainOptions`` — Eden's procedural terrain (the generator grid is run
  once, off-scene),
- ``EntityOptions`` — a 3D terrain mesh file, rasterized by downward raycasts
  at ``mesh_resolution``,
- ``str`` — path to a height-field ``.npz`` exported via
  :func:`eden.utils.terrains.io.export_height_field`.

The optimization costs minutes per clip and dataset modifiers rerun on every
dataset load (the dataset cache is deliberately saved pre-modifier), so
results are cached on disk keyed by a content hash of (clip data, terrain,
parameters, robot MJCF) — the second load of an unchanged dataset is
near-free. Set ``cache_dir=None`` to disable.

This modifier adapts clips 1→1 in place at a single ``terrain_offset``
placement. Upstream-style 1→N random-placement augmentation is out of scope
here; see ``examples/recipes/motion_optimization/tcrs`` for batch variant generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from typing import TYPE_CHECKING, ClassVar

import numpy as np

import eden as en
from eden.datasets.modifiers.base import DATASET_MODIFIER_REGISTRY, DatasetModifier
from eden.options.entities import EntityOptions, TerrainOptions
from eden.options.extensions.motion_optimization import TCRSPlannerOptions, TCRSRobotOptions
from eden.utils import distributed as eden_dist
from eden.utils.distribution import get_eden_version
from eden.utils.motion_reference.utils import derive_body_frame_link_fields, estimate_motion_velocities
from eden.utils.torch import numpy_to_tensor, to_numpy

if TYPE_CHECKING:
    from eden.extensions.motion_optimization.tcrs.pipeline import TCRSMotionAdapter
    from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain
    from eden.utils.motion_reference.motion_data import BatchMotionSequence

# Bumped when the cached-result layout or the pipeline behavior changes in a
# way that invalidates previously cached adaptations.
_TCRS_CACHE_VERSION = 4


@DATASET_MODIFIER_REGISTRY.register()
class TCRSTerrainAdapt(DatasetModifier):
    """Adapt flat-ground clips to a terrain with the TCRS optimizer.

    Parameters
    ----------
    terrain : TerrainOptions | EntityOptions | str
        The terrain to adapt to: procedural ``TerrainOptions``, a mesh-file
        ``EntityOptions``, or the path of an exported height-field ``.npz``.
        Required.
    robot_options : EntityOptions
        Eden robot options whose ``file`` resolves to the robot's MJCF; the
        IK solves against this model. Must match the robot the dataset is
        retargeted to (e.g. ``en.robots.G1Dofs29()``). Required.
    terrain_offset : tuple[float, float, float]
        SE(2) placement ``(dx, dy, dyaw)`` applied to every clip before
        adaptation: the clip is yaw-rotated by ``dyaw`` (radians) about its
        first-frame pelvis position, then translated by ``(dx, dy)``. Use it
        to position the motion on the terrain feature of interest.
    mesh_resolution : float
        Rasterization cell size in meters when ``terrain`` is a mesh-file
        ``EntityOptions``. Ignored for the other terrain kinds.
    robot_spec : TCRSRobotOptions | dict | None
        A :class:`~eden.options.extensions.motion_optimization.TCRSRobotOptions` (or
        its dict form / a partial override dict, e.g. ``{"dofs_name": [...]}``
        when the dataset's DOF order differs). Unspecified fields keep the G1
        defaults — ``dofs_name``/``links_name`` default to Eden's canonical G1
        orderings, which is what Eden datasets produce. Unknown keys raise at
        construction. Partial ``ik_config`` dicts deep-merge onto the defaults
        (``DEFAULT_IK_CONFIG``): solver keys merge individually, link-weight
        entries merge per link name.
    planner : TCRSPlannerOptions | dict | None
        A :class:`~eden.options.extensions.motion_optimization.TCRSPlannerOptions`
        (or its dict form / a partial override dict) for the pipeline knobs
        (MPPI sampling, stabilizer, lowpass cutoff, ...). Unknown keys raise
        at construction. IK solver weights live in the robot spec: override
        via ``robot_spec={"ik_config": {...}}``.
    cache_dir : str | None
        Directory for the content-addressed result cache. ``None`` disables
        caching — every dataset load then re-runs the full optimization.

    Notes
    -----
    ``links_pos_w``/``links_quat_w`` must be populated on the sequence (Eden
    mimic datasets fill them via FK). ``contact_points_w`` is left untouched
    and a warning is emitted when present: world-frame contact points recorded
    on flat ground are stale after terrain adaptation.
    """

    # ``= None`` is a framework requirement: ``ConfigurableMixin.get_parameter_names``
    # discovers fields via class ``__dict__``, so a value-less typed-only field
    # vanishes from the modifier's parameter list (see GroundFirstFrame).
    terrain: TerrainOptions | EntityOptions | str = None
    robot_options: EntityOptions = None
    terrain_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_resolution: float = 0.02
    robot_spec: TCRSRobotOptions | dict | None = None
    planner: TCRSPlannerOptions | dict | None = None
    cache_dir: str | None = "~/.cache/eden/tcrs"
    _breaks_body_model: ClassVar[str | None] = "its IK re-poses the links, leaving the body-model pose stale"

    def apply(self, motion_sequences):
        if self.terrain is None:
            raise ValueError(
                "TCRSTerrainAdapt: 'terrain' is required — pass TerrainOptions, a mesh-file EntityOptions, "
                "or the path of an exported height-field .npz."
            )
        if self.robot_options is None:
            raise ValueError(
                "TCRSTerrainAdapt: 'robot_options' is required — pass the robot's EntityOptions "
                "(e.g. en.robots.G1Dofs29()) so the IK can load its MJCF."
            )
        offset = tuple(float(v) for v in self.terrain_offset)
        if len(offset) != 3:
            raise ValueError(f"TCRSTerrainAdapt: terrain_offset must be (dx, dy, dyaw); got {self.terrain_offset}.")

        keys = self._selected_keys(motion_sequences, single=True)

        robot_options = self._coerce_options(self.robot_options, EntityOptions, "robot_options")
        if not robot_options.file:
            raise ValueError(
                "TCRSTerrainAdapt: robot_options.file is empty — it must name the robot's MJCF "
                "(an empty pattern would trigger a full asset-registry download)."
            )
        mjcf_path = robot_options.fetch_asset()
        if not str(mjcf_path).endswith(".xml"):
            raise ValueError(
                f"TCRSTerrainAdapt: robot_options.file must resolve to a MuJoCo MJCF (.xml); got '{mjcf_path}'. "
                "The TCRS IK runs on a MuJoCo model of the robot."
            )

        terrain = self._build_terrain()
        terrain.print_info()
        adapter = self._build_adapter(mjcf_path)

        out = dict(motion_sequences)
        for key in keys:
            out[key] = self._apply_to_key(out[key], key, terrain, adapter, mjcf_path, offset)
        return out

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_options(value, options_cls, field_name):
        """Accept a live options object or its dict form (post-yaml load)."""
        if isinstance(value, options_cls):
            return value
        if isinstance(value, dict):
            return options_cls(**value)
        raise TypeError(
            f"TCRSTerrainAdapt: {field_name} must be {options_cls.__name__} (or its dict form); "
            f"got {type(value).__name__}."
        )

    def _build_terrain(self) -> "HeightFieldTerrain":
        from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain

        terrain = self.terrain
        if isinstance(terrain, str):
            path = os.path.expanduser(terrain)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"TCRSTerrainAdapt: terrain file '{terrain}' does not exist. Expected a height-field .npz "
                    "exported via eden.utils.terrains.io.export_height_field."
                )
            return HeightFieldTerrain.from_file(path)
        if isinstance(terrain, dict):
            # Post-yaml dict form: only a mesh-file EntityOptions carries a non-empty "file";
            # everything else is TerrainOptions (which may rely on its default generators).
            cls = EntityOptions if terrain.get("file") else TerrainOptions
            terrain = self._coerce_options(terrain, cls, "terrain")
        if isinstance(terrain, TerrainOptions):
            if terrain.randomize:
                raise ValueError(
                    "TCRSTerrainAdapt: terrain.randomize=True is not supported — the adapted clips must match "
                    "a reproducible terrain surface (and the result cache could never hit). Set randomize=False."
                )
            return HeightFieldTerrain.from_terrain_options(terrain)
        if isinstance(terrain, EntityOptions):
            if not terrain.file:
                raise ValueError(
                    "TCRSTerrainAdapt: terrain EntityOptions.file is empty — it must name the terrain mesh "
                    "(an empty pattern would trigger a full asset-registry download)."
                )
            return HeightFieldTerrain.from_entity_options(terrain, resolution=self.mesh_resolution)
        raise TypeError(
            "TCRSTerrainAdapt: terrain must be TerrainOptions (procedural), EntityOptions (mesh file), or a "
            f"height-field .npz path; got {type(terrain).__name__}."
        )

    def _build_robot_spec(self) -> TCRSRobotOptions:
        if self.robot_spec is None:
            return TCRSRobotOptions()
        return self._coerce_options(self.robot_spec, TCRSRobotOptions, "robot_spec")

    def _build_planner(self) -> TCRSPlannerOptions:
        if self.planner is None:
            return TCRSPlannerOptions()
        return self._coerce_options(self.planner, TCRSPlannerOptions, "planner")

    def _build_adapter(self, mjcf_path: str) -> "TCRSMotionAdapter":
        from eden.extensions.motion_optimization.tcrs.pipeline import TCRSMotionAdapter  # mujoco stays lazy

        return TCRSMotionAdapter(
            model_path=mjcf_path, robot_spec=self._build_robot_spec(), params=self._build_planner()
        )

    # ------------------------------------------------------------------
    # Per-key application
    # ------------------------------------------------------------------

    def _apply_to_key(
        self,
        motion_seq: "BatchMotionSequence",
        key: str,
        terrain: "HeightFieldTerrain",
        adapter: "TCRSMotionAdapter",
        mjcf_path: str,
        offset: tuple[float, float, float],
    ) -> "BatchMotionSequence":
        from eden.extensions.motion_optimization.tcrs.pipeline import ClipArrays

        for field in ("dofs_pos", "base_pos_w", "base_quat_w", "links_pos_w", "links_quat_w"):
            if getattr(motion_seq, field) is None:
                raise ValueError(
                    f"TCRSTerrainAdapt: motion_sequences['{key}'].{field} is not populated. The TCRS pipeline "
                    "needs joint positions and world-frame link poses (Eden mimic datasets fill links via FK)."
                )
        if not motion_seq.is_free_floating:
            raise ValueError(
                f"TCRSTerrainAdapt: motion_sequences['{key}'] is not free-floating; terrain adaptation "
                "re-targets the floating base and does not apply to fixed-base entities."
            )
        if motion_seq.contact_points_w is not None or motion_seq.contact_points_valid is not None:
            en.logger.warning(
                f"TCRSTerrainAdapt: motion_sequences['{key}'] carries contact_points_w/contact_points_valid "
                "recorded on the original (flat) motion; they are NOT re-derived after terrain adaptation "
                "and will be stale."
            )
        min_frames = int(motion_seq.buffer_length.min())
        if min_frames < 3:
            raise ValueError(
                f"TCRSTerrainAdapt: motion_sequences['{key}'] contains a clip with only {min_frames} frame(s); "
                "at least 3 frames per clip are required, because the solver's foot-collision pass smooths each clip "
                "with a 3-tap kernel (_resolve_lower_leg_foot_collisions in tcrs/pipeline.py). Leave the clip out with "
                "the dataset's exclude_pattern or exclude_list_file, or lower drop_first_n_frames / drop_last_n_frames."
            )

        cache_dir = os.path.expanduser(self.cache_dir) if self.cache_dir is not None else None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        context_hash = self._context_hash(terrain, adapter, mjcf_path, offset)

        num_seqs = int(motion_seq.buffer_length.shape[0])
        if num_seqs == 0:
            return motion_seq
        starts = motion_seq.base_pos_w.batch_starts
        sizes = motion_seq.base_pos_w.batch_sizes
        pelvis_idx = adapter.robot_spec.pelvis_index
        num_cache_hits = 0
        ik_max_mm: list[float] = []
        bad_frame_counts: list[int] = []
        # Stamped into every entry as a staleness tripwire. The release version, not a checkout's commit: a stamp that
        # changed with every commit would warn on every run and so warn of nothing (the entry layout has its own
        # _TCRS_CACHE_VERSION).
        eden_version = get_eden_version()
        stale_cache_versions: set[str] = set()

        # DDP: with a shared cache, let rank 0 fill it first so the other ranks'
        # phase 1 turns into pure cache hits (same rank-0-prepares protocol as
        # ``DatasetBase._initialize``). Non-main ranks still RUN phase 1 — with
        # cache_dir=None or a partially-warm cache they just recompute — so no
        # rank ever depends on another rank's success for correctness.
        coordinate_ranks = eden_dist.is_distributed() and cache_dir is not None
        if coordinate_ranks and not eden_dist.is_main_process():
            eden_dist.barrier()  # wait for rank 0 to finish phase 1 below

        # Phase 1: adapt (or cache-load) every clip. Nothing is written into the
        # shared buffers yet, so a failure on clip k leaves the input sequence
        # untouched instead of half-adapted.
        adapted_clips: list[dict] = []
        try:
            for i in range(num_seqs):
                sl = slice(int(starts[i]), int(starts[i]) + int(sizes[i]))
                fps = float(motion_seq.framerate[i])
                links_lin_vel = (
                    motion_seq.links_lin_vel_w.concatenated_tensor[sl]
                    if motion_seq.links_lin_vel_w is not None
                    else None
                )

                # Hash the float32 buffers directly (no float64 blow-up on the
                # cache-hit path); the float64 pipeline input is built from these
                # same arrays on a miss, so hash and pipeline input stay 1:1.
                clip32 = ClipArrays(
                    fps=fps,
                    dofs_pos=to_numpy(motion_seq.dofs_pos.concatenated_tensor[sl], np.float32),
                    links_pos_w=to_numpy(motion_seq.links_pos_w.concatenated_tensor[sl], np.float32),
                    links_quat_w=to_numpy(motion_seq.links_quat_w.concatenated_tensor[sl], np.float32),
                    links_lin_vel_w=to_numpy(links_lin_vel, np.float32) if links_lin_vel is not None else None,
                )

                cache_path = None
                adapted = None
                if cache_dir is not None:
                    cache_path = os.path.join(cache_dir, f"{self._clip_hash(context_hash, clip32)}.npz")
                    adapted = self._load_cached_clip(cache_path, clip32)
                    if adapted is not None:
                        num_cache_hits += 1
                        cached_version = str(adapted.pop("eden_version"))
                        if self._stamped_by_another_version(cached_version, eden_version):
                            stale_cache_versions.add(cached_version)
                if adapted is None:
                    clip = ClipArrays(
                        fps=fps,
                        dofs_pos=clip32.dofs_pos.astype(np.float64),
                        links_pos_w=clip32.links_pos_w.astype(np.float64),
                        links_quat_w=clip32.links_quat_w.astype(np.float64),
                        links_lin_vel_w=(
                            clip32.links_lin_vel_w.astype(np.float64) if clip32.links_lin_vel_w is not None else None
                        ),
                    )
                    result = adapter.adapt(clip, terrain, dx=offset[0], dy=offset[1], dyaw=offset[2])
                    # float32 results: halves cache size and phase-1 RAM; the
                    # write-back casts to the (float32) buffer dtype anyway.
                    adapted = {
                        "dofs_pos": result.optimized.dofs_pos.astype(np.float32),
                        "links_pos_w": result.optimized.links_pos_w.astype(np.float32),
                        "links_quat_w": result.optimized.links_quat_w.astype(np.float32),
                        "ik_pos_errors": result.ik_pos_errors.astype(np.float32),
                        "ik_bad_frame_count": np.int64(result.ik_bad_frame_count),
                    }
                    if cache_path is not None:
                        self._save_cached_clip(cache_path, adapted, eden_version)

                ik_max_mm.append(float(np.max(adapted["ik_pos_errors"])) if len(adapted["ik_pos_errors"]) else 0.0)
                bad_frame_counts.append(int(adapted["ik_bad_frame_count"]))
                adapted_clips.append(adapted)
        finally:
            # Release the peers even when phase 1 raises on rank 0 — they then
            # recompute their misses instead of hanging on the barrier.
            if coordinate_ranks and eden_dist.is_main_process():
                eden_dist.barrier()

        if stale_cache_versions:
            en.logger.warning(
                f"TCRSTerrainAdapt: some cached results under '{cache_dir}' were produced by a different eden "
                f"version ({sorted(stale_cache_versions)}; current {eden_version}). Using them anyway — delete "
                "the cache directory to force recomputation."
            )

        # Phase 2: write everything back in place.
        for i, adapted in enumerate(adapted_clips):
            sl = slice(int(starts[i]), int(starts[i]) + int(sizes[i]))
            self._write_back_clip(motion_seq, sl, adapted, pelvis_idx, float(motion_seq.framerate[i]))

        total_bad = int(np.sum(bad_frame_counts))
        clips_with_bad = int(np.sum(np.asarray(bad_frame_counts) > 0))
        en.logger.info(
            f"TCRSTerrainAdapt: adapted {num_seqs} clip(s) of '{key}' "
            f"({num_cache_hits} cache hit(s), offset dx={offset[0]:+.2f} dy={offset[1]:+.2f} dyaw={offset[2]:+.3f}; "
            f"IK max err p50={np.median(ik_max_mm):.0f}mm max={np.max(ik_max_mm):.0f}mm)"
        )
        if total_bad > 0:
            en.logger.warning(
                f"TCRSTerrainAdapt: {clips_with_bad}/{num_seqs} clip(s) have frames with IK error above the "
                f"rejection threshold ({total_bad} frame(s) total). Inspect these clips before training on them."
            )
        return motion_seq

    def _write_back_clip(self, motion_seq, sl, adapted, pelvis_idx, fps):
        """Write the optimized clip back into the packed buffers and refresh derived fields."""
        dofs = motion_seq.dofs_pos.concatenated_tensor
        links_pos = motion_seq.links_pos_w.concatenated_tensor
        links_quat = motion_seq.links_quat_w.concatenated_tensor
        for target, field in ((dofs, "dofs_pos"), (links_pos, "links_pos_w"), (links_quat, "links_quat_w")):
            target[sl] = numpy_to_tensor(adapted[field], dtype=target.dtype, device=target.device)
        motion_seq.base_pos_w.concatenated_tensor[sl] = links_pos[sl][:, pelvis_idx]
        motion_seq.base_quat_w.concatenated_tensor[sl] = links_quat[sl][:, pelvis_idx]

        # Recompute every populated velocity field from the adapted poses (the
        # IK changed joints and link poses; stale velocities would leak into
        # tracking rewards).
        base_lin_vel, base_ang_vel, dofs_vel, links_lin_vel, links_ang_vel = estimate_motion_velocities(
            root_trans=motion_seq.base_pos_w.concatenated_tensor[sl],
            root_quat=motion_seq.base_quat_w.concatenated_tensor[sl],
            dofs_pos=dofs[sl],
            links_pos_w=links_pos[sl],
            links_quat_w=links_quat[sl],
            dt=1.0 / fps,
        )
        for field, value in (
            ("base_lin_vel_w", base_lin_vel),
            ("base_ang_vel_w", base_ang_vel),
            ("dofs_vel", dofs_vel),
            ("links_lin_vel_w", links_lin_vel),
            ("links_ang_vel_w", links_ang_vel),
        ):
            buf = getattr(motion_seq, field)
            if buf is not None and value is not None:
                buf.concatenated_tensor[sl] = value.to(dtype=buf.concatenated_tensor.dtype)

        # Refresh the passive base-frame link fields when the sequence carries them.
        if motion_seq.links_pos_b is not None or motion_seq.links_quat_b is not None:
            derived = derive_body_frame_link_fields(
                base_pos_w=motion_seq.base_pos_w.concatenated_tensor[sl],
                base_quat_w=motion_seq.base_quat_w.concatenated_tensor[sl],
                base_lin_vel_w=base_lin_vel,
                base_ang_vel_w=base_ang_vel,
                links_pos_w=links_pos[sl],
                links_quat_w=links_quat[sl],
                links_lin_vel_w=links_lin_vel,
                links_ang_vel_w=links_ang_vel,
            )
            for field, value in zip(("links_pos_b", "links_quat_b", "links_lin_vel_b", "links_ang_vel_b"), derived):
                buf = getattr(motion_seq, field)
                if buf is not None and value is not None:
                    buf.concatenated_tensor[sl] = value.to(dtype=buf.concatenated_tensor.dtype)

    # ------------------------------------------------------------------
    # Content-addressed cache
    # ------------------------------------------------------------------

    def _context_hash(self, terrain, adapter, mjcf_path, offset) -> str:
        """Hash everything that affects the result except the clip itself."""
        h = hashlib.sha256()
        h.update(f"tcrs-cache-v{_TCRS_CACHE_VERSION}".encode())
        h.update(terrain.state_bytes())
        h.update(json.dumps(adapter.params.model_dump(), sort_keys=True).encode())
        h.update(json.dumps(adapter.robot_spec.model_dump(), sort_keys=True).encode())
        h.update(np.asarray(offset, dtype=np.float64).tobytes())
        # The MJCF xml determines the kinematics the IK solves against. Its
        # referenced meshes only affect visuals/collisions, which the pipeline
        # never touches — hashing the xml alone is sufficient.
        with open(mjcf_path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    @staticmethod
    def _clip_hash(context_hash: str, clip) -> str:
        h = hashlib.sha256()
        h.update(context_hash.encode())
        h.update(np.float64(clip.fps).tobytes())
        for array in (clip.dofs_pos, clip.links_pos_w, clip.links_quat_w):
            h.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        # Velocities feed contact detection (contact_speed_threshold), so both
        # their presence and their contents must invalidate the cache.
        if clip.links_lin_vel_w is not None:
            h.update(np.ascontiguousarray(clip.links_lin_vel_w, dtype=np.float32).tobytes())
        else:
            h.update(b"no-links-lin-vel")
        return h.hexdigest()

    @staticmethod
    def _load_cached_clip(cache_path: str, clip) -> dict | None:
        if not os.path.isfile(cache_path):
            return None
        try:
            with np.load(cache_path) as raw:
                adapted = {k: raw[k] for k in raw.files}
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            en.logger.warning(f"TCRSTerrainAdapt: discarding unreadable cache entry '{cache_path}' ({exc}).")
            return None
        expected = {"dofs_pos", "links_pos_w", "links_quat_w", "ik_pos_errors", "ik_bad_frame_count", "eden_version"}
        if set(adapted) != expected or adapted["dofs_pos"].shape != clip.dofs_pos.shape:
            en.logger.warning(f"TCRSTerrainAdapt: discarding stale cache entry '{cache_path}'.")
            return None
        return adapted

    @staticmethod
    def _stamped_by_another_version(cached: str, current: str) -> bool:
        """Whether an entry was stamped by a release other than this one, as far as this run can tell.

        :func:`get_eden_version` returns ``""`` when Eden is not installed as the ``eden-world`` distribution (a
        source tree on ``PYTHONPATH``, say). Such a run cannot compare, so it calls nothing stale: otherwise it would
        tell the reader to delete a cache that is fine. An entry it wrote, stamped ``""``, is of unknown release, so a
        run that knows its own version does call it stale.
        """
        return bool(current) and cached != current

    @staticmethod
    def _save_cached_clip(cache_path: str, adapted: dict, eden_version: str) -> None:
        # Atomic write: modifiers run on every DDP rank; os.replace keeps
        # concurrent writers from ever exposing a torn file.
        fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(cache_path))
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, eden_version=eden_version, **adapted)
            os.replace(tmp_path, cache_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
