"""TACO (Liu et al., CVPR 2024): bimanual tool-action-object recordings."""

from __future__ import annotations
from typing import ClassVar

import itertools
import pickle
from pathlib import Path

import numpy as np

from eden.datasets.base import DATASET_REGISTRY
from eden.datasets.hand_object_base import HandObjectClip, HandObjectDatasetBase, MANOHand, rotate_poses

#: TACO's world (z-up) turned 180 degrees about z, CHORD's convention.
_TACO_TO_WORLD = np.diag([-1.0, -1.0, 1.0])


@DATASET_REGISTRY.register()
class TacoDataset(HandObjectDatasetBase):
    """TACO: one clip per recording, both hands and the tool and target objects.

    Layout (github.com/Satorinal/TACO)::

        <root>/Hand_Poses/<(action, tool, target)>/<seq>/{right,left}_hand.pkl, {right,left}_hand_shape.pkl
        <root>/Object_Poses/<(action, tool, target)>/<seq>/{tool,target}_<id>.npy
        <root>/object_models_released/<id>_cm.obj

    Hands are MANO relative to the flat hand, with the wrist given; objects (4, 4) per frame; 30 Hz, z-up, turned
    180 degrees about z (CHORD's frame). Each hand holds the object it stays nearest to over the clip (both hands
    the one object when there is only one). Clip names ``taco_<action>__<tool>__<target>_<seq>``. Needs the MANO
    model files (``body_model_dir``); see :class:`~eden.datasets.hand_object_base.HandObjectDatasetBase`.
    """

    source_framerate: ClassVar[float] = 30.0

    def units(self, root: Path) -> list[str]:
        missing = [d for d in ("Hand_Poses", "Object_Poses", "object_models_released") if not (root / d).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"{root} is not a TACO release: {missing} missing. Point raw_data_path at the directory holding "
                "Hand_Poses/, Object_Poses/ and object_models_released/ (github.com/Satorinal/TACO)."
            )
        return sorted(str(d) for d in (root / "Hand_Poses").glob("*/*") if d.is_dir())

    def read_unit(self, unit: str) -> list[HandObjectClip | tuple[str, str]]:
        hand_dir = Path(unit)
        triplet, seq = hand_dir.parent.name, hand_dir.name
        name = "taco_" + "__".join(w.replace(" ", "_") for w in triplet.strip("()").split(", ")) + f"_{seq}"
        pose_dir = Path(self.raw_data_path) / "Object_Poses" / triplet / seq
        poses = {}
        for kind in ("tool", "target"):
            files = sorted(pose_dir.glob(f"{kind}_*.npy"))
            if len(files) == 1:
                poses[files[0].stem.split("_", 1)[1]] = np.load(files[0]).astype(np.float64)
        hands = {}
        for side in ("right", "left"):
            if (hand_dir / f"{side}_hand.pkl").exists() and (hand_dir / f"{side}_hand_shape.pkl").exists():
                with open(hand_dir / f"{side}_hand.pkl", "rb") as f:
                    frames = pickle.load(f)
                keys = sorted(frames)
                pose = np.stack([frames[k]["hand_pose"].numpy() for k in keys]).astype(np.float64)
                with open(hand_dir / f"{side}_hand_shape.pkl", "rb") as f:
                    betas = pickle.load(f)["hand_shape"].numpy().astype(np.float64)
                wrist = np.stack([frames[k]["hand_trans"].numpy() for k in keys]).astype(np.float64)
                hands[side] = MANOHand(side, wrist, pose[:, :3], pose[:, 3:], betas).rotated(_TACO_TO_WORLD)
        if not poses:
            return [(name, "no tool or target object")]
        if len({len(p) for p in poses.values()} | {len(h) for h in hands.values()}) > 1:
            return [(name, "frame counts differ")]
        poses = rotate_poses(poses, _TACO_TO_WORLD)
        centres = {o: p[:, :3, :3] @ self.object_mesh(o).vertices.mean(0) + p[:, :3, 3] for o, p in poses.items()}
        objects = assign_nearest({s: h.root_pos for s, h in hands.items()}, centres)
        return [HandObjectClip(name, f"{triplet}/{seq}", hands, objects, poses)]

    def _load_object_mesh(self, obj_id: str, materials: bool = False):
        import trimesh

        mesh = trimesh.load(
            Path(self.raw_data_path) / "object_models_released" / f"{obj_id}_cm.obj",
            force="mesh",
            process=False,
            skip_materials=not materials,
        )
        mesh.apply_scale(0.01)
        return mesh


def assign_nearest(wrists: dict[str, np.ndarray], centres: dict[str, np.ndarray]) -> dict[str, str]:
    """Hand side -> object with the least total mean wrist-centre distance; a single object goes to every hand."""
    if len(centres) == 1:
        return {s: next(iter(centres)) for s in wrists}
    dist = {(s, o): float(np.linalg.norm(w - c, axis=1).mean()) for s, w in wrists.items() for o, c in centres.items()}
    pick = min(itertools.permutations(centres, len(wrists)), key=lambda p: sum(dist[k] for k in zip(wrists, p)))
    return dict(zip(wrists, pick))
