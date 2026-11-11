"""OakInk2 (Zhan et al., CVPR 2024): bimanual object manipulation, one clip per program stage."""

from __future__ import annotations
from typing import ClassVar

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from eden.datasets.base import DATASET_REGISTRY
from eden.datasets.hand_object_base import HandObjectClip, HandObjectDatasetBase, MANOHand, rotate_poses
from eden.datasets.utils.body_model import mano_vertex_links

#: OakInk2's y-up world turned z-up.
YUP_TO_ZUP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)


@DATASET_REGISTRY.register()
class OakInk2Dataset(HandObjectDatasetBase):
    """OakInk2: each recording cut into the stages of its program, one clip per stage.

    Layout::

        <root>/anno_preview/<seq>.pkl               # raw_mano and obj_transf per mocap frame
        <root>/program/program_info/<seq>.json      # the program's stages: frame spans, objects per hand
        <root>/object_repair/align_ds/<id>/model.obj

    120 Hz, y-up turned z-up. A stage is the frames from its first span's start to its last span's end (clip
    ``<seq>__stNN``). Each hand holds the object its 16 MANO links touch most (a vertex within ``contact`` of the
    object's mesh, counted per link and frame), as CHORD assigns them, since the program can name the other hand's
    object; the program's own object when it touches none. A stage with several objects in one loaded hand
    (``sides``), or with a mocap frame missing from its span, is skipped, as is a recording whose mocap frame ids
    step by more than 1 (packed at 120 Hz, it would play too fast).
    :meth:`recording_object_poses` gives every object with a mesh over the whole recording. Needs the MANO model
    files (``body_model_dir``).

    Parameters
    ----------
    contact : float
        Distance (m) under which a MANO vertex touches an object.
    """

    contact: float = 0.01

    source_framerate: ClassVar[float] = 120.0
    has_recording_context: ClassVar[bool] = True
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("contact",)

    def units(self, root: Path) -> list[str]:
        missing = [
            d for d in ("anno_preview", "program/program_info", "object_repair/align_ds") if not (root / d).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"{root} is not an OakInk2 release: {missing} missing. Point raw_data_path at the directory holding "
                "anno_preview/, program/ and object_repair/."
            )
        programs = root / "program" / "program_info"
        return sorted(str(p) for p in (root / "anno_preview").glob("*.pkl") if (programs / f"{p.stem}.json").exists())

    def read_unit(self, unit: str) -> list[HandObjectClip | tuple[str, str]]:
        path = Path(unit)
        seq = path.stem
        with open(path, "rb") as f:
            anno = pickle.load(f)
        with open(Path(self.raw_data_path) / "program" / "program_info" / f"{seq}.json") as f:
            program = json.load(f)
        frame_ids = list(anno["mocap_frame_id_list"])
        # The recording's mocap stride (its most common step). Packed at source_framerate, a recording that steps by
        # more than 1 would play stride x too fast, so it is skipped; a larger step inside a stage is a missing frame.
        steps = np.diff(frame_ids)
        stride = int(np.bincount(steps[steps > 0]).argmax()) if (steps > 0).any() else 1
        if stride != 1:
            return [(seq, f"mocap frame ids step by {stride}, not 1 ({self.source_framerate:g} Hz)")]
        context = self._context(anno)
        out: list[HandObjectClip | tuple[str, str]] = []
        for i, (key, stage) in enumerate(program.items()):
            name = f"{seq}__st{i:02d}"
            spans = [s for s in ast.literal_eval(key) if s is not None]
            lists = {"right": stage.get("obj_list_rh") or [], "left": stage.get("obj_list_lh") or []}
            if not spans:
                out.append((name, "no stage interval"))
                continue
            if any(len(lists[side]) > 1 for side in self._sides):
                out.append((name, "several objects in one hand"))
                continue
            start, end = min(s[0] for s in spans), max(s[1] for s in spans)
            idx = np.array([k for k, f in enumerate(frame_ids) if start <= f < end], dtype=int)
            if not len(idx):
                out.append((name, f"no mocap frame in the stage interval (frames {start}-{end})"))
                continue
            frames = [frame_ids[k] for k in idx]
            gaps = np.flatnonzero(np.diff(frames) > 1)
            if len(gaps):
                out.append((name, f"missing frames ({len(gaps)} gap(s), first after mocap frame {frames[gaps[0]]})"))
                continue
            hands = {side: self._hand(anno, frames, side) for side in self._sides}
            verts = {}
            for side, hand in hands.items():
                joints, verts[side] = hand.forward(self.body_model_dir)
                hand.keep_keypoints(joints, verts[side])
            objects = self._held(verts, {o: p[idx] for o, p in context.items()}, lists)
            # The program's object when no hand touches one with a mesh: it can lack a mesh or a pose track.
            tracked = {str(o) for o in anno["obj_transf"]}
            unusable = sorted(o for o in set(objects.values()) if o not in tracked or not self._mesh_path(o).exists())
            if unusable:
                what = "pose track" if unusable[0] not in tracked else "mesh"
                out.append((name, f"held object has no {what} ({unusable[0]})"))
                continue
            # From the annotation, as FlashDexRetarget reads them.
            poses = _object_poses(anno, frames, set(objects.values()))
            out.append(HandObjectClip(name, seq, hands, objects, poses))
        return out

    def recording_object_poses(self, recording: str) -> dict[str, np.ndarray]:
        with open(Path(self.raw_data_path) / "anno_preview" / f"{recording}.pkl", "rb") as f:
            return self._context(pickle.load(f))

    def _context(self, anno: dict) -> dict[str, np.ndarray]:
        """Every object with a mesh, over every mocap frame of the recording, in the z-up world."""
        ids = [str(o) for o in anno["obj_transf"] if self._mesh_path(str(o)).exists()]
        return _object_poses(anno, anno["mocap_frame_id_list"], ids)

    def _held(self, verts: dict[str, np.ndarray], candidates: dict[str, np.ndarray], lists: dict) -> dict[str, str]:
        """Hand side -> the object its 16 MANO links touch the most link-frames; the program's when none."""
        out = {}
        for side, v in verts.items():
            link_of = mano_vertex_links(self.body_model_dir, side)
            best, most = (lists[side] or [None])[0], 0
            for o, pose in candidates.items():
                local = np.matmul(v - pose[:, None, :3, 3], pose[:, :3, :3])
                near = self.near(o, local.reshape(-1, 3), self.contact).reshape(v.shape[:2]) < self.contact
                n = sum(int(near[:, link_of == k].any(1).sum()) for k in range(16))
                if n > most:
                    best, most = o, n
            if best is not None:
                out[side] = str(best)
        return out

    @staticmethod
    def _hand(anno: dict, frames: list[int], side: str) -> MANOHand:
        pre = {"right": "rh", "left": "lh"}[side]
        mano = [anno["raw_mano"][f] for f in frames]
        quat = torch.cat([m[f"{pre}__pose_coeffs"] for m in mano]).numpy().astype(np.float64)
        rotvec = Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(frames), 16, 3)
        hand = MANOHand(
            side,
            torch.cat([m[f"{pre}__tsl"] for m in mano]).numpy().astype(np.float64),
            rotvec[:, 0],
            rotvec[:, 1:].reshape(len(frames), 45),
            torch.cat([m[f"{pre}__betas"] for m in mano]).numpy().astype(np.float64),
        )
        return hand.rotated(YUP_TO_ZUP)

    def _mesh_path(self, obj_id: str) -> Path:
        return Path(self.raw_data_path) / "object_repair" / "align_ds" / obj_id / "model.obj"

    def _load_object_mesh(self, obj_id: str, materials: bool = False):
        import trimesh

        path = self._mesh_path(obj_id)
        if not path.exists():
            raise FileNotFoundError(
                f"OakInk2 object {obj_id!r} has no mesh at {path}: object_repair/align_ds/ lacks it. Its clips need "
                f"one; add it, or leave those clips out with exclude_pattern."
            )
        return trimesh.load(path, force="mesh", process=False, skip_materials=not materials)


def _object_poses(anno: dict, frames: list[int], ids) -> dict[str, np.ndarray]:
    """Object id -> its (T, 4, 4) z-up world poses over the mocap ``frames``."""
    keys = {str(o): o for o in anno["obj_transf"]}
    poses = {i: np.stack([anno["obj_transf"][keys[i]][f] for f in frames]).astype(np.float64) for i in ids}
    return rotate_poses(poses, YUP_TO_ZUP)
