"""DexYCB hand-object grasping dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar

import functools
import glob
import os

import numpy as np
import yaml
from genesis.utils.geom import (
    R_to_quat,
    axis_angle_to_quat,
    quat_to_R,
    transform_by_quat,
    transform_quat_by_quat,
    xyz_to_quat,
)

import eden as en
from eden.datasets.base import _MIN_CLIP_FRAMES, DATASET_REGISTRY, DatasetBase
from eden.datasets.constants import MANO_KEYPOINTS_NAME
from eden.datasets.utils.body_model import axis_angles_to_body_pose_quat, mano_model_name, mano_pose_from_release
from eden.utils.assets import get_asset_path
from eden.utils.geom import xyzw_to_wxyz
from eden.utils.misc import get_assets_dir
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.torch import numpy_to_tensor

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence


# Maps ``meta.yml::ycb_ids`` integers to YCB folder names; mirrors
# ``dex_ycb_toolkit._YCB_CLASSES``.
DEX_YCB_OBJECTS: dict[int, str] = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

DEX_YCB_FRAMERATE: float = 30.0


# Order of the 21 keypoints DexYCB ships in ``labels_*.npz::joint_3d`` —
# per the dex-ycb-toolkit MANO convention, distinct from the Eden order
# in :data:`eden.datasets.constants.MANO_KEYPOINTS_NAME`. We reorder the
# raw labels into the Eden order at load time via :data:`_DEX_YCB_TO_EDEN_KEYPOINTS`.
_DEX_YCB_KEYPOINT_ORDER: tuple[str, ...] = (
    "wrist",
    "thumb1", "thumb2", "thumb3", "thumb_tip",
    "index1", "index2", "index3", "index_tip",
    "middle1", "middle2", "middle3", "middle_tip",
    "ring1", "ring2", "ring3", "ring_tip",
    "pinky1", "pinky2", "pinky3", "pinky_tip",
)  # fmt: skip
_DEX_YCB_TO_EDEN_KEYPOINTS: tuple[int, ...] = tuple(_DEX_YCB_KEYPOINT_ORDER.index(name) for name in MANO_KEYPOINTS_NAME)


def _broadcast_to(value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Tile a per-clip constant to a per-frame shape with a contiguous copy.

    Genesis's numba kernels assert ``v.shape == quat.shape`` and reject
    the strided views that ``np.broadcast_to`` alone would return.
    """
    return np.broadcast_to(np.asarray(value, dtype=dtype), shape).copy()


def _apply_world_left(
    pos_src: np.ndarray,  # (T, 3)
    quat_wxyz_src: np.ndarray,  # (T, 4)
    R: np.ndarray,  # (3, 3)
    t: np.ndarray,  # (3,)
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame left-multiply: ``new_pose = (R, t) ∘ src_pose``."""
    pos_dst = pos_src @ R.T + t
    # `transform_quat_by_quat(v, u)` = quatmul(u, v), so passing
    # (v=quat_src, u=q_R) gives q_R ⊗ quat_src.
    q_R = _broadcast_to(R_to_quat(R.astype(np.float32)), quat_wxyz_src.shape, quat_wxyz_src.dtype)
    quat_wxyz_dst = transform_quat_by_quat(quat_wxyz_src, q_R)
    return pos_dst.astype(np.float32), quat_wxyz_dst.astype(np.float32)


def _apply_mesh_local_right(
    pos_src: np.ndarray,  # (T, 3)
    quat_wxyz_src: np.ndarray,  # (T, 4)
    q_corr_wxyz: np.ndarray,  # (4,) constant
    t_corr: np.ndarray,  # (3,) constant
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh-local right-multiply: ``new_pose = src_pose ∘ (q_corr, t_corr)``."""
    q_corr = _broadcast_to(q_corr_wxyz, quat_wxyz_src.shape, quat_wxyz_src.dtype)
    t_broadcast = _broadcast_to(t_corr, (quat_wxyz_src.shape[0], 3), quat_wxyz_src.dtype)
    quat_wxyz_dst = transform_quat_by_quat(q_corr, quat_wxyz_src)
    pos_dst = pos_src + transform_by_quat(t_broadcast, quat_wxyz_src)
    return pos_dst.astype(np.float32), quat_wxyz_dst.astype(np.float32)


@functools.lru_cache(maxsize=64)
def _ycb_mesh_aabb(folder_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Cache the mesh-frame AABB ``(min, max)`` for one YCB object.

    Returns ``None`` when the asset isn't extracted locally.
    """
    path = os.path.join(get_assets_dir(), "ycb", "models", folder_name, "textured.obj")
    if not os.path.exists(path):
        return None
    try:
        import trimesh
    except ImportError:
        return None
    mesh = trimesh.load_mesh(path, force="mesh")
    v = mesh.vertices
    return v.min(axis=0).astype(np.float32), v.max(axis=0).astype(np.float32)


def _grounding_z_shift(
    quat_0_wxyz: np.ndarray,
    pos_0: np.ndarray,
    mesh_aabb_min: np.ndarray,
    mesh_aabb_max: np.ndarray,
) -> float:
    """Return the z-translation that grounds the world AABB bottom to z=0.

    World-z of a mesh vertex ``v`` is ``R_0[2, :] @ v + pos_0[2]``; its
    minimum over the AABB is the linear-functional minimum over an
    axis-aligned box (one of the 8 corners, chosen per-axis by the sign
    of ``R_0[2, :]``).
    """
    R_0 = quat_to_R(np.asarray(quat_0_wxyz, dtype=np.float32))
    z_row = R_0[2, :]
    corner = np.where(z_row >= 0, mesh_aabb_min, mesh_aabb_max)
    return float(-(z_row @ corner + pos_0[2]))


# DexYCB recordings start with the object on the table; a measurable
# post-grounding dip means a bug rather than a real motion. Tolerance
# absorbs accumulated float rounding.
_GROUND_PENETRATION_TOLERANCE: float = 0.01


# Per-object correction aligning Eden's body frame (after Genesis bakes
# in ``YCBOptions.up``/``front``) with DexYCB's annotation frame.
# Applied as a mesh-local right-multiply on the object trajectory.
#
# Each entry is ``(rpy_deg, offset_m)`` = ``T_ED · R_align⁻¹``:
#   * ``T_ED`` (raw mesh ICP): DexYCB ships BOP-formatted scans, Eden
#     ships Berkeley scans — they agree up to a per-object yaw of
#     0°-180°.
#   * ``R_align⁻¹ = R_z(+90°)``: undoes Genesis's bake-in for the
#     current defaults ``up=(0,0,1), front=(0,1,0)`` (which rotates
#     mesh-Y → body-X). Eliminated if ``YCBOptions.front`` is
#     ever flipped to ``(1, 0, 0)``.
DEX_YCB_OBJECT_CORRECTIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "002_master_chef_can":     ((  -0.1,   +0.0,  -90.1), (-0.001, +0.000, -0.003)),
    "003_cracker_box":         ((  +0.0,   +0.2,  +90.0), (-0.001, -0.002, -0.004)),
    "004_sugar_box":           ((  +3.2,   -0.9,  -92.3), (-0.000, -0.002, -0.003)),
    "005_tomato_soup_can":     ((  -0.0,   -0.1,  +89.9), (-0.000, +0.002, -0.008)),
    "006_mustard_bottle":      ((  -0.1,   +0.5,  +61.9), (-0.001, -0.002, +0.012)),
    "007_tuna_fish_can":       ((  -0.1,   +0.0,   -0.1), (+0.000, -0.000, -0.004)),
    "008_pudding_box":         ((  +0.0,   +0.0, +115.7), (-0.001, +0.004, -0.003)),
    "009_gelatin_box":         ((  +0.1,   +0.0, +100.5), (-0.001, +0.001, -0.004)),
    "010_potted_meat_can":     ((  -0.1,   +0.0,  +92.4), (-0.002, -0.001, -0.008)),
    "011_banana":              ((  +1.0,   -0.3,  +67.7), (+0.001, -0.005, +0.002)),
    "019_pitcher_base":        ((  -2.4,   -2.5,  +91.6), (-0.000, -0.003, -0.025)),
    "021_bleach_cleanser":     ((  +3.0,   -2.0,  +87.1), (-0.001, -0.002, +0.011)),
    "024_bowl":                ((  -0.4,   -0.4, -180.0), (+0.000, -0.000, +0.002)),
    "025_mug":                 ((  +0.7,   +0.3,  +89.7), (+0.000, +0.002, +0.001)),
    "035_power_drill":         ((  -0.1,   +0.0,  +91.4), (-0.006, -0.021, +0.002)),
    "036_wood_block":          ((  -0.3,   -0.4,  +77.5), (-0.002, +0.002, -0.017)),
    "037_scissors":            (( -10.7,   -1.5,  -88.2), (+0.014, +0.019, +0.006)),
    "040_large_marker":        ((  +0.0,   +0.0,  +91.0), (-0.000, +0.000, -0.000)),
    "051_large_clamp":         ((  +4.0,   -1.6,  +97.8), (-0.001, -0.002, -0.000)),
    "052_extra_large_clamp":   ((  -0.1,   +0.0,  +96.7), (-0.003, +0.000, -0.000)),
    "061_foam_brick":          ((  +0.1,   -0.1,  +89.2), (+0.000, -0.000, +0.008)),
}  # fmt: skip


# DexYCB's ``extrinsics.yml`` emits a ``!!python/tuple`` tag for every
# 12-float entry; ``safe_load`` refuses it. Register a constructor that
# drops the tag to a plain list — safer than ``yaml.full_load``, which
# evaluates arbitrary tags.
class _DexYCBYamlLoader(yaml.SafeLoader):
    pass


_DexYCBYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node),
)


@DATASET_REGISTRY.register()
class DexYCBDataset(DatasetBase):
    """DexYCB single-hand YCB-grasping dataset.

    Each sequence is one MANO hand (left or right) interacting with up to four
    YCB objects; the loader keeps only the *grasped* object indicated by
    ``meta.yml::ycb_grasp_ind``. Frames where ``pose_m`` is all-zero (no MANO
    annotation, typically pre-grasp) are trimmed from the leading edge; clips
    whose invalid frames are not contiguously at the start are skipped with a
    warning so ``BatchMotionSequence`` always sees a contiguous range.

    Expected on-disk layout (after extracting ``dex-ycb-20210415.tar.gz`` or
    the per-subject tarballs from https://dex-ycb.github.io/)::

        <root>/<YYYYMMDD-subject-NN>/<YYYYMMDD_HHMMSS>/
            ├── meta.yml          # ycb_ids, ycb_grasp_ind, mano_sides, ...
            ├── pose.npz          # world-frame ground-truth poses
            └── <camera_serial>/  # color/depth/per-camera labels (unused here)

    ``pose.npz`` carries:

    * ``pose_y`` — shape ``(T, num_objects, 7)``, layout
      ``[qx, qy, qz, qw, tx, ty, tz]``. Quaternion in xyzw, converted to
      Eden's wxyz convention here.
    * ``pose_m`` — shape ``(T, 1, 51)``, layout ``[pose(48), trans(3)]``
      in dex-ycb-toolkit's MANO parameterisation (manopth: the global
      wrist axis-angle, 45 finger PCA coefficients relative to the mean
      hand, and MANO's translation). ``pose[0:3]`` is loaded into
      ``base_quat_w``. The loader picks up DexYCB's precomputed MANO joint
      positions from each clip's
      ``<seq>/<master_serial>/labels_*.npz::joint_3d`` and packs them
      into ``links_pos_w`` (21 keypoints in
      :data:`eden.datasets.constants.MANO_KEYPOINTS_NAME` order);
      ``base_pos_w`` is their wrist. (MANO's translation is not the wrist:
      the wrist sits a shape-dependent offset, ~9 cm, away from it.)

    With ``body_model_params=True`` the hand also carries its MANO pose and
    shape (``body_pose_quat`` / ``body_shape``, ``body_model`` ``mano_<side>``):
    the PCA expanded into the 15 finger rotations relative to the flat hand,
    and the subject's ``betas`` from ``calibration/mano_<id>/mano.yml``. That
    needs the MANO model files (``body_model_dir``) for the PCA basis and the
    mean hand.

    Raw ``pose.npz`` poses live in the **master camera's frame** (x-right,
    y-down, z-forward, the first entry in ``meta.yml::serials``).
    ``calibration_dir`` (the extracted ``calibration/`` directory) is
    required: the loader applies each clip's master-camera extrinsic to
    bring poses into the **AprilTag world frame** (z-up, origin on the
    table tag) so a Genesis viewer renders correctly. Preparing the dataset
    from raw data without ``calibration_dir`` raises ``FileNotFoundError``.

    Captures run at 30 fps.

    No HuggingFace mirror is provided: the upstream is gated behind a Google
    Drive consent form, so users must download and extract manually, then
    pass the extracted root as ``raw_data_path``.

    References
    ----------
    - https://dex-ycb.github.io/
    - https://github.com/NVlabs/dex-ycb-toolkit


    Parameters
    ----------
    calibration_dir: str | None
        Path to DexYCB's ``calibration/`` directory. Required: each
        clip's ``apriltag`` extrinsic is left-applied so poses land in
        the z-up AprilTag world frame. ``prepare()`` raises
        ``FileNotFoundError`` when this is unset or doesn't exist.
    ground_initial_pose: bool
        Translates each clip uniformly along world-Z so the object's
        world AABB bottom touches z=0 at frame 0; orientation passes
        through unchanged.
    apply_mesh_correction: bool
        Applies :data:`DEX_YCB_OBJECT_CORRECTIONS` as a mesh-local
        right-multiply on the object trajectory. Disable when rendering
        against a non-Eden YCB asset version.
    models_dir: str | None
        DexYCB's ``models/`` directory, read by :meth:`object_mesh` when
        ``apply_mesh_correction`` is off (the poses then refer to the
        release's meshes). Defaults to the ``models/`` next to
        ``calibration_dir``.
    retargeted_label: str | None
        When set, also load each clip's ``retargeted_<label>.npz`` as a third
        motion key ``"robot"`` (see the comment on the attribute).
    """

    _applies_frame_window: ClassVar[bool] = True
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames", "drop_last_n_frames")

    calibration_dir: str | None = None
    ground_initial_pose: bool = True
    apply_mesh_correction: bool = True
    models_dir: str | None = None
    # When set, the loader also reads ``<seq_dir>/retargeted_<label>.npz``
    # (written by ``examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py``)
    # and exposes it as a third motion key ``"robot"`` whose
    # ``motion_dofs_name`` reflects the saved robot's ``dofs_name``. Clips
    # missing the retargeted file are dropped with a warning so the
    # surviving "hand" / "object" / "robot" sequences stay aligned.
    retargeted_label: str | None = None

    _supports_body_model_params: ClassVar[bool] = True
    # Refuses caches whose hand base_pos_w is MANO's ``transl`` (~9 cm off the wrist) rather than the labelled wrist.
    _cache_semantics: ClassVar[str | None] = "base_pos_w is the labelled MANO wrist (joint 0), not MANO transl"
    _cache_identity_fields: ClassVar[tuple[str, ...]] = (
        "ground_initial_pose",
        "apply_mesh_correction",
        "retargeted_label",
    )

    # NOTE: narrowed from the base `glob("**/*.npz")` to skip per-frame `labels_NNN.npz` under each camera serial.
    file_extension: ClassVar[str] = "pose.npz"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("hand", "object")
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {
        "hand": [],
        "object": [],
    }
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "hand": list(MANO_KEYPOINTS_NAME),
        "object": [],
    }

    # Keys in ``self.metadata`` that hold one entry per clip; ``DatasetBase.select_trajectories``
    # reindexes them together with the trajectories.
    _PER_CLIP_METADATA_KEYS: ClassVar[tuple[str, ...]] = (
        "sequence_ids",
        "objects",
        "sides",
        "first_valid_frames",
        "master_serials",
        "grounding_z_shifts",
    )

    def _initialize(self):
        self._meshes: dict[str, Any] = {}  # object_mesh's, per instance
        super()._initialize()

    def _get_all_motion_files(self, data_path: str, extension: str = "pose.npz") -> list[str]:
        # ``sorted`` keeps the cached ``all_motion_files`` order stable
        # across filesystem walks.
        if os.path.isdir(data_path):
            return sorted(glob.glob(os.path.join(data_path, "**", "pose.npz"), recursive=True))
        if not data_path.endswith("pose.npz"):
            raise ValueError(f"DexYCB raw_data_path must be a directory or a pose.npz file, got {data_path}")
        return [data_path]

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        if not self.all_motion_files:
            raise FileNotFoundError(
                f"No DexYCB sequences found under {self.raw_data_path}. "
                "Expected `<root>/<subject>/<sequence>/pose.npz` after extracting "
                "`dex-ycb-20210415.tar.gz` or the per-subject archives."
            )

        if self.calibration_dir is None or not os.path.isdir(self.calibration_dir):
            raise FileNotFoundError(
                f"DexYCB: `calibration_dir` is required for the apriltag-world "
                f"transform (got: {self.calibration_dir!r}). Pass "
                "`DexYCBDataset.configure(calibration_dir=...)` pointing at an "
                "extracted `calibration/` directory."
            )

        results = self._parallel_read_motion_files(
            self.all_motion_files,
            self._read_single_sequence,
            desc="DexYCB: reading raw pose.npz",
        )

        valid_files: list[str] = []
        per_clip: dict[str, list] = {
            k: [] for k in ("sequence_ids", "objects", "sides", "first_valid_frames", "master_serials")
        }
        hand_seqs: list[MotionSequence] = []
        object_seqs: list[MotionSequence] = []
        grounding_z_shifts: list[float] = []
        for file_path, result in zip(self.all_motion_files, results):
            if result is None:
                en.logger.warning(f"DexYCB: skipping {file_path} (no contiguous valid hand frames)")
                continue
            hand_seq, object_seq, info = result
            valid_files.append(file_path)
            # ``os.path.basename`` is always ``"pose.npz"``; use a relpath
            # to keep ids unique and human-readable.
            per_clip["sequence_ids"].append(os.path.relpath(os.path.dirname(file_path), self.raw_data_path))
            per_clip["objects"].append(info["object"])
            per_clip["sides"].append(info["side"])
            per_clip["first_valid_frames"].append(info["first_valid_frame"])
            per_clip["master_serials"].append(info["master_serial"])
            grounding_z_shifts.append(info["grounding_z_shift"])
            hand_seqs.append(hand_seq)
            object_seqs.append(object_seq)

        if not hand_seqs:
            raise RuntimeError(
                "All DexYCB sequences were filtered out. Expect at least one clip "
                "with a contiguous run of valid `pose_m` frames; inspect the "
                "`pose.npz::pose_m` arrays for unexpected zero gaps."
            )

        # ------------------------------------------------------------------
        # Optional second pass: read inline `retargeted_<label>.npz` files
        # next to each surviving pose.npz. Clips missing the retargeted file
        # are dropped here so the resulting hand/object/robot sequences stay
        # aligned across motion keys.
        # ------------------------------------------------------------------
        robot_seqs: list[MotionSequence] = []
        robot_dofs_name: list[str] = []
        if self.retargeted_label is not None:
            keep_idx: list[int] = []
            for i, file_path in enumerate(valid_files):
                robot_seq, this_dofs_name = self._read_retargeted_robot(file_path)
                if robot_seq is None:
                    continue
                if not robot_dofs_name:
                    robot_dofs_name = this_dofs_name
                elif this_dofs_name != robot_dofs_name:
                    raise ValueError(
                        f"DexYCB: `retargeted_<{self.retargeted_label}>.npz` files have "
                        f"inconsistent `dofs_name` across clips. First clip declared "
                        f"{robot_dofs_name}; {file_path} declared {this_dofs_name}."
                    )
                robot_seqs.append(robot_seq)
                keep_idx.append(i)

            if not robot_seqs:
                raise RuntimeError(
                    f"DexYCB: `retargeted_label={self.retargeted_label!r}` but no clip "
                    f"under {self.raw_data_path} has a matching "
                    f"`retargeted_{self.retargeted_label}.npz`. Run "
                    f"`examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py` first."
                )
            # Filter hand/object/per-clip metadata down to the clips that
            # actually had a retargeted file.
            if len(keep_idx) < len(valid_files):
                en.logger.warning(
                    f"DexYCB: dropping {len(valid_files) - len(keep_idx)} clip(s) "
                    f"missing `retargeted_{self.retargeted_label}.npz`; "
                    f"{len(keep_idx)} retained."
                )
                hand_seqs = [hand_seqs[i] for i in keep_idx]
                object_seqs = [object_seqs[i] for i in keep_idx]
                grounding_z_shifts = [grounding_z_shifts[i] for i in keep_idx]
                for key in per_clip:
                    per_clip[key] = [per_clip[key][i] for i in keep_idx]
                valid_files = [valid_files[i] for i in keep_idx]

        # Keep the base class's post-prepare ``metadata['all_motion_files']``
        # in sync with the surviving motion buffers.
        self.all_motion_files = valid_files

        data = {
            "hand": BatchMotionSequence.from_motion_sequences(hand_seqs).to(self.device),
            "object": BatchMotionSequence.from_motion_sequences(object_seqs).to(self.device),
        }
        if self.retargeted_label is not None:
            data["robot"] = BatchMotionSequence.from_motion_sequences(robot_seqs).to(self.device)
            # Shadow the ClassVars at the instance level so downstream
            # consumers (and ``select_trajectories``) see the third key.
            self.motion_sequence_keys = (*type(self).motion_sequence_keys, "robot")
            self.motion_dofs_name = {**type(self).motion_dofs_name, "robot": robot_dofs_name}

        metadata: dict[str, Any] = {
            **per_clip,
            # Already applied to ``base_pos_w``; surfaced so downstream
            # code re-loading positions from the raw source (e.g. MANO
            # keypoints from per-frame labels) can match the dataset frame.
            "grounding_z_shifts": np.asarray(grounding_z_shifts, dtype=np.float32),
        }
        if self.retargeted_label is not None:
            metadata["retargeted_label"] = self.retargeted_label
        return data, metadata

    def _read_single_sequence(self, file_path: str) -> tuple[MotionSequence, MotionSequence, dict[str, Any]] | None:
        seq_dir = os.path.dirname(file_path)
        meta_path = os.path.join(seq_dir, "meta.yml")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"DexYCB meta.yml not found next to {file_path}")
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        ycb_id = int(meta["ycb_ids"][int(meta["ycb_grasp_ind"])])
        if ycb_id not in DEX_YCB_OBJECTS:
            raise ValueError(f"Unknown DexYCB ycb_id {ycb_id} at {meta_path} (known: {sorted(DEX_YCB_OBJECTS)})")
        object_name = DEX_YCB_OBJECTS[ycb_id]
        side = list(meta["mano_sides"])[0]

        data = np.load(file_path)
        pose_y = data["pose_y"]  # (T, num_obj, 7)
        pose_m = data["pose_m"]  # (T, 1, 51)
        if pose_y.ndim != 3 or pose_y.shape[-1] != 7:
            raise ValueError(f"Unexpected pose_y shape {pose_y.shape} in {file_path}")
        if pose_m.ndim != 3 or pose_m.shape[1:] != (1, 51):
            raise ValueError(f"Unexpected pose_m shape {pose_m.shape} in {file_path}")

        # Pre-grasp frames have ``pose_m == 0``; trim leading invalid
        # frames and skip clips with non-contiguous gaps (which would
        # break ``BatchMotionSequence``'s contiguous-range invariant).
        valid = ~np.all(pose_m[:, 0] == 0.0, axis=-1)
        if not valid.any():
            return None
        first_valid = int(np.argmax(valid))
        if not valid[first_valid:].all():
            return None
        window = self._frame_window()
        pose_y = pose_y[first_valid:][window]
        pose_m = pose_m[first_valid:][window]
        if pose_m.shape[0] < _MIN_CLIP_FRAMES:
            # Raised before the per-frame labels are read — an over-trimmed window has none to stack — as the error
            # ``_pack_motion_sequence`` gives every other short clip, which the shared read leaves out with a warning.
            raise self._short_clip_error(pose_m.shape[0], DEX_YCB_FRAMERATE, None)

        obj = pose_y[:, int(meta["ycb_grasp_ind"])].astype(np.float32, copy=False)
        obj_quat_wxyz = xyzw_to_wxyz(obj[:, :4])
        obj_pos = obj[:, 4:7].copy()

        m = pose_m[:, 0].astype(np.float32, copy=False)
        wrist_rotvec = m[:, 0:3]
        # ``axis_angle_to_quat`` normalises internally and handles
        # zero-magnitude rotvecs (pre-grasp identity) safely.
        hand_quat_wxyz = axis_angle_to_quat(np.linalg.norm(wrist_rotvec, axis=-1), wrist_rotvec)

        # ``prepare()`` guarantees ``calibration_dir`` is set and exists.
        T_apriltag_master, master_serial = self.load_apriltag_master_calibration(str(meta["extrinsics"]))
        if T_apriltag_master is None:
            raise FileNotFoundError(
                f"DexYCB: extrinsics for clip {os.path.basename(seq_dir)} "
                f"(extrinsics_id={meta['extrinsics']!r}) missing under {self.calibration_dir}"
            )
        R, t = T_apriltag_master[:3, :3], T_apriltag_master[:3, 3]
        obj_pos, obj_quat_wxyz = _apply_world_left(obj_pos, obj_quat_wxyz, R, t)
        _, hand_quat_wxyz = _apply_world_left(m[:, 48:51], hand_quat_wxyz, R, t)

        # MANO joint positions per frame (master-cam frame in labels;
        # apriltag-applied here to match the rest of the trajectory).
        # Reordered from the dex-ycb-toolkit MANO order to Eden's
        # ``MANO_KEYPOINTS_NAME`` order so downstream code can index by name.
        labels_start = first_valid + window.start
        hand_links_pos_w = self._load_master_cam_keypoints(
            seq_dir, master_serial, labels_start, labels_start + pose_y.shape[0]
        )
        hand_links_pos_w = hand_links_pos_w[:, _DEX_YCB_TO_EDEN_KEYPOINTS, :]
        hand_links_pos_w = hand_links_pos_w @ R.T + t
        # The root is the labelled wrist, MANO's root joint: ``pose_m``'s translation sits ~9 cm off it.
        hand_pos = hand_links_pos_w[:, MANO_KEYPOINTS_NAME.index("wrist")].copy()

        body: dict[str, Any] = {}
        if self.body_model_params:
            calib = os.path.join(self.calibration_dir, f"mano_{meta['mano_calib'][0]}", "mano.yml")
            with open(calib) as f:
                betas = np.asarray(yaml.safe_load(f)["betas"], dtype=np.float32)
            fingers = mano_pose_from_release(
                m[:, 3:48], self.body_model_dir, side, num_pca_comps=45, flat_hand_mean=False
            )
            body = dict(
                body_pose_quat=axis_angles_to_body_pose_quat(fingers),
                body_shape=numpy_to_tensor(np.broadcast_to(betas, (len(m), len(betas)))),
                body_model=mano_model_name(side),
            )

        # Mesh correction is mesh-local (right-multiply) and applies to
        # the object only — the hand isn't tied to a YCB asset.
        if self.apply_mesh_correction and object_name in DEX_YCB_OBJECT_CORRECTIONS:
            rpy_deg, offset = DEX_YCB_OBJECT_CORRECTIONS[object_name]
            q_corr = xyz_to_quat(np.asarray(rpy_deg, dtype=np.float32), rpy=False, degrees=True)
            t_corr = np.asarray(offset, dtype=np.float32)
            obj_pos, obj_quat_wxyz = _apply_mesh_local_right(obj_pos, obj_quat_wxyz, q_corr, t_corr)

        z_shift = 0.0
        if self.ground_initial_pose:
            mesh_aabb = _ycb_mesh_aabb(object_name)
            if mesh_aabb is None:
                en.logger.warning(
                    f"DexYCB: {object_name} mesh missing under "
                    f"{os.path.join(get_assets_dir(), 'ycb', 'models')} — "
                    "leaving clip ungrounded."
                )
            else:
                z_shift = _grounding_z_shift(obj_quat_wxyz[0], obj_pos[0], *mesh_aabb)
                obj_pos[:, 2] += z_shift
                hand_pos[:, 2] += z_shift
                hand_links_pos_w[:, :, 2] += z_shift

                min_obj_z = float(obj_pos[:, 2].min())
                if min_obj_z < -_GROUND_PENETRATION_TOLERANCE:
                    en.logger.warning(
                        f"DexYCB: clip {os.path.basename(seq_dir)} ({object_name}) — "
                        f"object centroid dips to {min_obj_z:+.4f} m after grounding "
                        f"(>{_GROUND_PENETRATION_TOLERANCE:.3f} m below floor)."
                    )

        hand_seq = self._pack_motion_sequence(
            framerate=DEX_YCB_FRAMERATE,
            base_pos_w=hand_pos,
            base_quat_w=hand_quat_wxyz,
            links_pos_w=hand_links_pos_w,
            **body,
        )
        object_seq = self._pack_motion_sequence(
            framerate=DEX_YCB_FRAMERATE,
            base_pos_w=obj_pos,
            base_quat_w=obj_quat_wxyz,
        )
        return (
            hand_seq,
            object_seq,
            {
                "object": object_name,
                "side": side,
                "first_valid_frame": first_valid,
                "master_serial": master_serial,
                "grounding_z_shift": float(z_shift),
            },
        )

    def object_mesh(self, object_name: str, materials: bool = False):
        """The ``trimesh.Trimesh`` (metres) that clip object poses of ``object_name`` place: its geometry, loaded once
        per dataset.

        With ``apply_mesh_correction`` the poses were moved onto Eden's YCB asset (``ycb/models/<name>/textured.obj``);
        without it they are the release's, on ``<models_dir>/<name>/textured_simple.obj``. With ``materials``, read
        again with its material and texture, and not cached: trimesh decodes the 4096^2 texture as it loads it
        (~50 MB per object), which only an export of the mesh needs.
        """
        if materials:
            return self._load_object_mesh(object_name, materials=True)
        if object_name not in self._meshes:  # an import asks once per clip (its floor) for the ~20 YCB objects
            self._meshes[object_name] = self._load_object_mesh(object_name)
        return self._meshes[object_name]

    def _load_object_mesh(self, object_name: str, materials: bool = False):
        import trimesh

        if self.apply_mesh_correction:
            path = get_asset_path(f"ycb/models/{object_name}/textured.obj")
        else:
            models = self.models_dir or (
                os.path.join(os.path.dirname(os.path.normpath(self.calibration_dir)), "models")
                if self.calibration_dir
                else None
            )
            if models is None or not os.path.isdir(models):
                raise FileNotFoundError(
                    f"DexYCB: object meshes for apply_mesh_correction=False come from the release's models/ "
                    f"directory, not found at {models!r}. Set models_dir to it."
                )
            path = os.path.join(models, object_name, "textured_simple.obj")
        return trimesh.load(path, force="mesh", process=False, skip_materials=not materials)

    def _read_retargeted_robot(self, pose_npz_path: str) -> tuple[MotionSequence | None, list[str]]:
        """Load ``<seq_dir>/retargeted_<label>.npz`` and pack a robot MotionSequence.

        Returns ``(None, [])`` when the file is missing so the caller can drop
        the corresponding clip. The expected schema (written by
        ``examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py``):

        - ``dofs_pos``: ``(T, n_dofs)``
        - ``dofs_name``: ``(n_dofs,)`` string array
        - ``base_pos_w``: ``(T, 3)`` — world frame (MANO wrist, matches hand seq)
        - ``base_quat_w``: ``(T, 4)`` wxyz
        - ``framerate``: scalar

        T must match the clip's hand-sequence length; mismatches raise.
        """
        seq_dir = os.path.dirname(pose_npz_path)
        path = os.path.join(seq_dir, f"retargeted_{self.retargeted_label}.npz")
        if not os.path.exists(path):
            # Per-clip log at debug level only — the dataset loader emits a
            # single WARNING with the total dropped count after the scan.
            en.logger.debug(f"DexYCB: retargeted file missing for {seq_dir}: {path}")
            return None, []

        npz = np.load(path)
        dofs_pos = np.asarray(npz["dofs_pos"], dtype=np.float32)
        base_pos_w = np.asarray(npz["base_pos_w"], dtype=np.float32)
        base_quat_w = np.asarray(npz["base_quat_w"], dtype=np.float32)
        dofs_name = [str(n) for n in np.asarray(npz["dofs_name"]).tolist()]
        framerate = float(npz["framerate"]) if "framerate" in npz.files else DEX_YCB_FRAMERATE

        if dofs_pos.ndim != 2:
            raise ValueError(f"DexYCB: {path} has dofs_pos shape {dofs_pos.shape}; expected (T, n_dofs).")
        if base_pos_w.shape != (dofs_pos.shape[0], 3) or base_quat_w.shape != (dofs_pos.shape[0], 4):
            raise ValueError(
                f"DexYCB: {path} base pose shapes "
                f"(base_pos_w={base_pos_w.shape}, base_quat_w={base_quat_w.shape}) do not "
                f"match T={dofs_pos.shape[0]}."
            )
        if len(dofs_name) != dofs_pos.shape[1]:
            raise ValueError(
                f"DexYCB: {path} dofs_name length {len(dofs_name)} disagrees with dofs_pos width {dofs_pos.shape[1]}."
            )

        robot_seq = self._pack_motion_sequence(
            framerate=framerate,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            dofs_pos=dofs_pos,
        )
        return robot_seq, dofs_name

    @staticmethod
    def _load_master_cam_keypoints(seq_dir: str, master_serial: str, first_valid: int, end_excl: int) -> np.ndarray:
        """Stack ``joint_3d`` from ``labels_NNNNNN.npz`` over the given frame range.

        Returned in the master-camera frame, dex-ycb-toolkit MANO order.
        """
        labels = []
        for frame in range(first_valid, end_excl):
            path = os.path.join(seq_dir, master_serial, f"labels_{frame:06d}.npz")
            labels.append(np.load(path)["joint_3d"][0])
        return np.stack(labels, dtype=np.float32)

    def load_apriltag_master_calibration(self, extrinsics_id: str) -> tuple[np.ndarray | None, str]:
        """Read ``<calibration_dir>/extrinsics_<id>/extrinsics.yml``.

        Returns ``(T_apriltag_master, master_serial)``. The ``apriltag``
        entry in the file is ``T_master_apriltag`` (AprilTag pose in the
        master camera's frame); we invert it so left-applying brings
        master-cam-frame poses into the z-up AprilTag world.
        ``master_serial`` is the calibration's top-level ``master``
        field, used by downstream code that reads per-camera files
        (e.g. MANO keypoints from ``labels_*.npz``).
        """
        if self.calibration_dir is None:
            return None, ""
        path = os.path.join(self.calibration_dir, f"extrinsics_{extrinsics_id}", "extrinsics.yml")
        if not os.path.exists(path):
            en.logger.warning(f"DexYCB: extrinsics file missing: {path}")
            return None, ""
        with open(path, "r") as f:
            ext = yaml.load(f, Loader=_DexYCBYamlLoader)
        block = ext.get("extrinsics", {})
        raw = block.get("apriltag")
        if raw is None:
            en.logger.warning(f"DexYCB: 'apriltag' entry missing in {path}; available: {sorted(block)}")
            return None, str(ext.get("master", ""))
        T_master_apriltag = np.asarray(raw, dtype=np.float32).reshape(3, 4)
        R = T_master_apriltag[:3, :3]
        t = T_master_apriltag[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        return np.concatenate([R_inv, t_inv[:, None]], axis=1), str(ext.get("master", ""))
