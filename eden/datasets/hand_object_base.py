"""Shared base of the MANO hand-object releases (TACO, OakInk2, HOT3D): per-side hand and held-object sequences."""

from __future__ import annotations
from typing import Any, ClassVar, Literal

import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from genesis.utils.geom import R_to_quat
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import eden as en
from eden.datasets.base import DatasetBase, _ShortClipError
from eden.datasets.constants import MANO_KEYPOINTS_NAME
from eden.datasets.utils.body_model import (
    axis_angles_to_body_pose_quat,
    body_model_forward,
    mano_keypoints,
    mano_model_name,
)
from eden.utils.geom import quat_from_axis_angle
from eden.utils.motion_reference.motion_data import BatchMotionSequence

SIDES = ("right", "left")


@dataclass
class MANOHand:
    """One hand of a clip in the dataset's z-up world, at the release's frame rate.

    Parameters
    ----------
    side : str
        ``right`` or ``left``: which MANO model it is.
    root_pos : np.ndarray
        (T, 3) MANO root joint (the wrist).
    root_rotvec : np.ndarray
        (T, 3) global orientation, axis-angle.
    finger_pose : np.ndarray
        (T, 45) finger axis-angles relative to the flat hand.
    betas : np.ndarray
        (T, B) or (B,) shape coefficients.
    keypoints : np.ndarray | None
        (T, 21, 3) float32 :data:`~eden.datasets.constants.MANO_KEYPOINTS_NAME` keypoints, once a reader that ran the
        model for its own analysis stored them (:meth:`keep_keypoints`); kept by :meth:`frames`. The vertices are
        never kept: at 778 x 3 per frame they would hold a whole release in memory until it is packed.
    """

    side: str
    root_pos: np.ndarray
    root_rotvec: np.ndarray
    finger_pose: np.ndarray
    betas: np.ndarray
    keypoints: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.root_pos)

    def frames(self, idx: np.ndarray | slice) -> MANOHand:
        """The hand on frames ``idx`` (a per-frame ``betas`` is sliced, a constant one kept)."""
        betas = self.betas[idx] if self.betas.ndim == 2 else self.betas
        keypoints = None if self.keypoints is None else self.keypoints[idx]
        return MANOHand(self.side, self.root_pos[idx], self.root_rotvec[idx], self.finger_pose[idx], betas, keypoints)

    def rotated(self, R: np.ndarray) -> MANOHand:
        """The hand with the world rotated by ``R``; finger rotations are local and stay, keypoints are dropped."""
        rotvec = (Rotation.from_matrix(R) * Rotation.from_rotvec(self.root_rotvec)).as_rotvec()
        return MANOHand(self.side, self.root_pos @ R.T, rotvec, self.finger_pose, self.betas)

    def forward(self, model_dir: str | None) -> tuple[np.ndarray, np.ndarray]:
        """(T, 16, 3) joints and (T, 778, 3) vertices, float64."""
        joints, verts = body_model_forward(
            mano_model_name(self.side),
            model_dir,
            torch.from_numpy(self.root_pos),
            quat_from_axis_angle(torch.from_numpy(self.root_rotvec)),
            axis_angles_to_body_pose_quat(self.finger_pose),
            torch.from_numpy(np.broadcast_to(self.betas, (len(self), self.betas.shape[-1])).copy()),
        )
        return joints.double().numpy(), verts.double().numpy()

    def keep_keypoints(self, joints: np.ndarray, verts: np.ndarray) -> None:
        """Store the keypoints of this hand's :meth:`forward` output, so packing does not run the model again."""
        self.keypoints = mano_keypoints(torch.from_numpy(joints), torch.from_numpy(verts)).float().numpy()

    def keypoint_track(self, model_dir: str | None) -> np.ndarray:
        """(T, 21, 3) keypoints: the stored ones, or the model's."""
        if self.keypoints is None:
            return mano_keypoints(*map(torch.from_numpy, self.forward(model_dir))).float().numpy()
        return self.keypoints


@dataclass
class HandObjectClip:
    """One clip a release's reader cut: hands, the object each holds, and object poses (object -> world).

    Parameters
    ----------
    name : str
        Unique clip name.
    recording : str
        The recording it was cut from; clips of one recording share
        :meth:`HandObjectDatasetBase.recording_object_poses`.
    hands : dict[str, MANOHand]
        Hand side -> hand.
    objects : dict[str, str]
        Hand side -> the object id it holds (both sides may name one object).
    poses : dict[str, np.ndarray]
        Object id -> (T, 4, 4) pose.
    """

    name: str
    recording: str
    hands: dict[str, MANOHand]
    objects: dict[str, str]
    poses: dict[str, np.ndarray]


class HandObjectDatasetBase(DatasetBase):
    """Base of the MANO hand-object releases: each clip is the hands of ``sides`` and the object each one holds.

    Keys: ``<side>_hand`` (base = the MANO wrist, ``links_pos_w`` = the 21
    :data:`~eden.datasets.constants.MANO_KEYPOINTS_NAME` keypoints) and ``object_<side>`` (the held object's pose)
    for each side in ``sides``. A clip where one of those hands holds nothing is left out and named in
    ``metadata["skipped"]``, as is any clip its reader rejects. With ``body_model_params=True`` each hand also carries
    its MANO pose and shape. The releases ship MANO parameters, not keypoints, so the MANO model files
    (``body_model_dir``) are always needed.

    Per-clip metadata: ``clip_names``, ``recordings``, ``objects_<side>`` (the held object id); the poses of every
    object of a recording come from :meth:`recording_object_poses`, read on demand from the release.

    The file filters (``include_pattern``, the list files, ...) match each clip's ``<unit>/<clip name>`` path, on a
    fresh build as on a cache load: a clip is only known once its unit is read, so a build reads every unit first.

    Parameters
    ----------
    sides : str
        ``both``, ``right`` or ``left``: the hands loaded, each of which must hold an object.
    """

    sides: Literal["both", "right", "left"] = "both"

    _supports_body_model_params: ClassVar[bool] = True
    _applies_frame_window: ClassVar[bool] = True
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames", "drop_last_n_frames")
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("sides",)
    #: The release's frame rate (pass it as ``target_framerate`` to keep the clips unretimed).
    source_framerate: ClassVar[float]
    #: Whether clips are cut from longer recordings, whose every object :meth:`recording_object_poses` gives.
    has_recording_context: ClassVar[bool] = False

    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("right_hand", "object_right", "left_hand", "object_left")
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {k: [] for k in motion_sequence_keys}
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "right_hand": list(MANO_KEYPOINTS_NAME),
        "left_hand": list(MANO_KEYPOINTS_NAME),
        "object_right": [],
        "object_left": [],
    }
    _PER_CLIP_METADATA_KEYS: ClassVar[tuple[str, ...]] = ("clip_names", "recordings", "objects_right", "objects_left")

    def _initialize(self):
        if self.sides not in ("both", *SIDES):
            raise ValueError(f"{type(self).__name__}.sides must be 'both', 'right' or 'left', got {self.sides!r}.")
        self._sides = SIDES if self.sides == "both" else (self.sides,)  # the hands loaded, right first
        # Per instance: the mixin mirrors class values back onto the instance, so a class-level narrowing would leak.
        self.motion_sequence_keys = tuple(k for s in self._sides for k in (f"{s}_hand", f"object_{s}"))
        self._vertex_trees: dict[str, tuple] = {}
        self._meshes: dict[str, Any] = {}
        # The file filters match clips; a build lists units, whose clips are known once read (see _apply_file_filter).
        self._clips_known = True
        super()._initialize()

    # ---- what a release's reader provides ------------------------------------------------------------------------

    def units(self, root: Path) -> list[str]:
        """The release's units of work (recordings or sequences), as paths under ``root``; raises naming the expected
        layout when ``root`` is not the release.
        """
        raise NotImplementedError

    def read_unit(self, unit: str) -> list[HandObjectClip | tuple[str, str]]:
        """The clips of ``unit``; a ``(name, reason)`` pair is a clip the reader rejected."""
        raise NotImplementedError

    def _load_object_mesh(self, obj_id: str, materials: bool = False):
        """Read the ``trimesh.Trimesh`` (metres) of object ``obj_id`` from the release, in its poses' frame; with
        ``materials``, its materials and textures too.
        """
        raise NotImplementedError

    def object_mesh(self, obj_id: str, materials: bool = False):
        """The ``trimesh.Trimesh`` (metres) of object ``obj_id``, in the frame its poses place: its geometry, cached.

        With ``materials``, read again with its materials and textures, and not cached: trimesh decodes the textures
        as it loads them (a 4096^2 texture holds ~50 MB), which only an export of the mesh needs.
        """
        if materials:
            return self._load_object_mesh(obj_id, materials=True)
        if obj_id not in self._meshes:
            self._meshes[obj_id] = self._load_object_mesh(obj_id)
        return self._meshes[obj_id]

    def recording_object_poses(self, recording: str) -> dict[str, np.ndarray]:
        """Every object's (T, 4, 4) world pose over the whole recording (read from the release on demand)."""
        raise NotImplementedError(
            f"{type(self).__name__} clips are whole recordings; their own object poses are the recording's."
        )

    # ---- shared ------------------------------------------------------------------------------------------------

    def _get_all_motion_files(self, data_path: str, extension: str = "") -> list[str]:
        self._clips_known = False  # these are units: prepare() reads them, then filters their clips
        return self.units(Path(data_path))

    def _apply_file_filter(self, files: list[str]) -> tuple[list[str], list[int] | None]:
        # Clips are only known once a unit is read, and a cache holds clips: the filters always match the clips'
        # ``<unit>/<clip name>`` paths, so a pattern selects the same clips whether the dataset is built or loaded.
        if not self._clips_known:
            return files, None
        return super()._apply_file_filter(files)

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        if self.body_model_dir is None:
            raise ValueError(
                f"{type(self).__name__} ships MANO parameters, not keypoints, so it needs the MANO model files: set "
                f"body_model_dir to the directory holding MANO_RIGHT.pkl / MANO_LEFT.pkl (mano.is.tue.mpg.de)."
            )
        results = self._parallel_read_motion_files(
            self.all_motion_files, self.read_unit, desc=f"{type(self).__name__}: reading"
        )
        sides = self._sides
        skipped: list[tuple[str, str]] = []
        accepted: list[tuple[str, HandObjectClip]] = []
        for unit, clips in zip(self.all_motion_files, results):
            for clip in clips:
                if isinstance(clip, tuple):
                    skipped.append(clip)
                    continue
                missing = [s for s in sides if s not in clip.hands or s not in clip.objects]
                if missing:
                    skipped.append((clip.name, f"no hand with an object ({'/'.join(missing)})"))
                    continue
                lengths = {len(clip.hands[s]) for s in sides} | {len(clip.poses[clip.objects[s]]) for s in sides}
                if len(lengths) > 1:
                    skipped.append((clip.name, f"frame counts differ ({sorted(lengths)})"))
                    continue
                clip = self._drop_frames(clip, lengths.pop())
                if isinstance(clip, tuple):
                    skipped.append(clip)
                    continue
                accepted.append((os.path.join(unit, clip.name), clip))
        self._clips_known = True
        if accepted:
            _, keep = self._apply_file_filter([path for path, _ in accepted])
            if keep is not None:
                accepted = [accepted[i] for i in keep]

        sequences: dict[str, list] = {key: [] for key in self.motion_sequence_keys}
        meta: dict[str, list] = {k: [] for k in self._PER_CLIP_METADATA_KEYS}
        files: list[str] = []
        for path, clip in accepted:
            try:  # every side packed before any is kept, so a clip is kept whole or not at all
                packed = {
                    side: (self._pack_hand(clip, side), self._pack_object(clip.poses[clip.objects[side]]))
                    for side in sides
                }
            except _ShortClipError:
                skipped.append((clip.name, "too short after retiming to target_framerate"))
                continue
            for side, (hand_seq, object_seq) in packed.items():
                sequences[f"{side}_hand"].append(hand_seq)
                sequences[f"object_{side}"].append(object_seq)
                meta[f"objects_{side}"].append(clip.objects[side])
            for side in SIDES:
                if side not in sides:
                    meta[f"objects_{side}"].append("")
            meta["clip_names"].append(clip.name)
            meta["recordings"].append(clip.recording)
            files.append(path)
        if skipped:
            reasons = count_reasons(reason for _, reason in skipped)
            en.logger.info(f"{type(self).__name__}: kept {len(files)} clips, skipped {len(skipped)} {reasons}")
        if not files:
            raise RuntimeError(
                f"{type(self).__name__}: every clip under {self.raw_data_path} was skipped "
                f"({len(skipped)}; first: {skipped[:3]}). sides={self.sides!r} needs each of those hands to hold an "
                f"object."
            )
        # prepare() owns the clip order; _initialize stores all_motion_files as the per-trajectory metadata.
        self.all_motion_files = files
        data = {}
        for key, seqs in sequences.items():
            batch = BatchMotionSequence.from_motion_sequences(
                seqs, num_dofs=0, num_links=len(self.motion_links_name[key])
            )
            batch.entity_kind = "hand" if key.endswith("_hand") else "rigid_object"
            data[key] = batch.to(self.device)
        return data, {**meta, "skipped": skipped}

    def _drop_frames(self, clip: HandObjectClip, frames: int) -> HandObjectClip | tuple[str, str]:
        """``clip`` without its first ``drop_first_n_frames`` and last ``drop_last_n_frames`` frames, or
        ``(name, reason)`` when fewer than 2 remain.
        """
        first, last = self.drop_first_n_frames, self.drop_last_n_frames
        if frames - first - last < 2:
            return clip.name, (
                f"too short ({frames} frames, {max(frames - first - last, 0)} after drop_first_n_frames={first} / "
                f"drop_last_n_frames={last})"
            )
        if not first and not last:
            return clip
        keep = self._frame_window()
        hands = {s: h.frames(keep) for s, h in clip.hands.items()}
        poses = {o: p[keep] for o, p in clip.poses.items()}
        return HandObjectClip(clip.name, clip.recording, hands, clip.objects, poses)

    def _pack_hand(self, clip: HandObjectClip, side: str):
        hand = clip.hands[side]
        keypoints = torch.from_numpy(hand.keypoint_track(self.body_model_dir))
        body: dict[str, Any] = {}
        if self.body_model_params:
            betas = np.broadcast_to(hand.betas, (len(hand), hand.betas.shape[-1]))
            body = dict(
                body_pose_quat=axis_angles_to_body_pose_quat(hand.finger_pose),
                body_shape=torch.from_numpy(np.ascontiguousarray(betas, dtype=np.float32)),
                body_model=mano_model_name(side),
            )
        return self._pack_motion_sequence(
            framerate=self.source_framerate,
            base_pos_w=torch.from_numpy(hand.root_pos).float(),
            base_quat_w=quat_from_axis_angle(torch.from_numpy(hand.root_rotvec)).float(),
            links_pos_w=keypoints.float(),
            **body,
        )

    def _pack_object(self, pose: np.ndarray):
        return self._pack_motion_sequence(
            framerate=self.source_framerate,
            base_pos_w=torch.from_numpy(np.ascontiguousarray(pose[:, :3, 3])).float(),
            base_quat_w=torch.from_numpy(R_to_quat(np.ascontiguousarray(pose[:, :3, :3], dtype=np.float32))).float(),
        )

    # ---- helpers for the readers -------------------------------------------------------------------------------

    def near(self, obj_id: str, pts: np.ndarray, bound: float) -> np.ndarray:
        """Distance from each (N, 3) object-frame point to the nearest mesh vertex where below ``bound``, else inf;
        points outside the vertices' box padded by ``bound`` skip the KD-tree.
        """
        if obj_id not in self._vertex_trees:
            v = np.asarray(self.object_mesh(obj_id).vertices)
            self._vertex_trees[obj_id] = cKDTree(v), v.min(0), v.max(0)
        tree, lo, hi = self._vertex_trees[obj_id]
        d = np.full(len(pts), np.inf)
        box = ((pts >= lo - bound) & (pts <= hi + bound)).all(1)
        d[box] = tree.query(pts[box], distance_upper_bound=bound)[0]
        return d


def count_reasons(reasons: Iterable[str]) -> dict[str, int]:
    """How many skips each reason accounts for, by its text before any ``" (detail)"``."""
    return dict(Counter(reason.split(" (")[0] for reason in reasons))


def rotate_poses(poses: dict[str, np.ndarray], R: np.ndarray) -> dict[str, np.ndarray]:
    """Object poses with the world rotated by ``R``."""
    out = {}
    for oid, pose in poses.items():
        p = np.array(pose, dtype=np.float64)
        p[:, :3, :3] = R @ p[:, :3, :3]
        p[:, :3, 3] = p[:, :3, 3] @ R.T
        out[oid] = p
    return out
