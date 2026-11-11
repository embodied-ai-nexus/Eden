"""Synthetic TACO, OakInk2 and HOT3D releases on disk, in each release's own layout, for the datasets that read them.

The hands are driven through the synthetic MANO model of ``tests/datasets/conftest.py`` (``mano_dir``); where a reader
decides contact from MANO vertices (OakInk2's held object, HOT3D's segments), the object meshes are built around the
hand's own vertices so the contact is unambiguous.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from eden.datasets.utils.body_model import CHORD_TIP_VERTICES
from tests.datasets.conftest import pose_track, write_box


def write_taco(raw: Path, triplet: str, seq: str, tool_pos, target_pos, flip_target_at: int | None = None, T: int = 31):
    """One TACO recording: the right wrist hovers over the tool, the left over the target (unambiguous nearest); the tool
    is textured :data:`TEXTURE_RGB`.
    """
    hands, poses = raw / "Hand_Poses" / triplet / seq, raw / "Object_Poses" / triplet / seq
    hands.mkdir(parents=True)
    poses.mkdir(parents=True)
    tool, target = triplet.strip("()").split(", ")[1:]
    target_track = pose_track(np.tile(target_pos, (T, 1)))
    if flip_target_at is not None:
        target_track[flip_target_at:, :3, :3] = np.diag([-1.0, -1.0, 1.0])  # 180 deg about z in one frame
    np.save(poses / f"tool_{tool}01.npy", pose_track(np.tile(tool_pos, (T, 1))))
    np.save(poses / f"target_{target}02.npy", target_track)
    for side, near in (("right", tool_pos), ("left", target_pos)):
        wrist = np.tile(np.asarray(near) + (0.0, 0.05, 0.1), (T, 1))
        frames = {
            f"{t:05d}": {"hand_pose": torch.zeros(48), "hand_trans": torch.tensor(w)} for t, w in enumerate(wrist)
        }
        with open(hands / f"{side}_hand.pkl", "wb") as f:
            pickle.dump(frames, f)
        with open(hands / f"{side}_hand_shape.pkl", "wb") as f:
            pickle.dump({"hand_shape": torch.zeros(10)}, f)
    # TACO meshes are in cm; the tool's textured
    write_textured_box(raw / "object_models_released" / f"{tool}01_cm.obj", scale=100.0)
    write_box(raw / "object_models_released" / f"{target}02_cm.obj", scale=100.0)


def _hand_vertices(mano_dir: Path, side: str, **kwargs) -> np.ndarray:
    """(778, 3) synthetic MANO vertices of one hand (``kwargs`` are smplx's, batch 1)."""
    import smplx  # the `dev` group's reference implementation; the mano_dir fixture skips without it

    layer = smplx.create(
        str(mano_dir / f"MANO_{side.upper()}.pkl"),
        model_type="mano",
        is_rhand=side == "right",
        use_pca=kwargs.pop("use_pca", False),
        num_pca_comps=kwargs.pop("num_pca_comps", 6),
        flat_hand_mean=kwargs.pop("flat_hand_mean", True),
    )
    batch = {k: torch.as_tensor(np.asarray(v, dtype=np.float32))[None] for k, v in kwargs.items()}
    return layer(**batch).vertices.detach()[0].numpy().astype(np.float64)


def _point_cloud_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """A mesh whose vertices include ``points``: a small tetrahedron at each."""
    tet = trimesh.creation.icosphere(subdivisions=0, radius=0.002)
    return trimesh.util.concatenate([tet.copy().apply_translation(p) for p in points])


OAKINK2_SEQ = "scene_01__A001"


def write_oakink2(
    raw: Path,
    mano_dir: Path,
    T: int = 40,
    missing: tuple[int, ...] = (),
    left_objects: tuple[str, ...] = (),
    stride: int = 1,
) -> dict:
    """One OakInk2 recording (y-up, 120 Hz) with four program stages over ``T`` mocap frames:

    - stage 0 (frames 0-19): the program says the right hand holds ``obj_a``, but its MANO vertices touch ``obj_b``
      (a mesh around them) and nothing else; the left hand touches nothing and the program names ``left_objects``
      for it;
    - stage 1 (20-39): the program names two objects for the right hand;
    - stage 2: no frame span;
    - stage 3: a span past the recording's mocap frames.

    The mocap frames ``missing`` (ids from 100, ``stride`` apart) have no annotation. Returns the source-frame (y-up)
    positions of the objects and the right wrist.
    """
    frames = [f for f in range(100, 100 + stride * T, stride) if f not in missing]  # mocap ids, not indices
    wrist = {"right": np.array([0.0, 0.3, 0.1]), "left": np.array([0.5, 0.3, 0.1])}
    template = _hand_vertices(mano_dir, "right", transl=np.zeros(3))
    right_verts = template - template[0] + wrist["right"]  # the zero-pose hand, its wrist (vertex 0) on the wrist
    obj_pos = {"obj_a": np.array([-0.6, 0.05, 0.0]), "obj_b": np.zeros(3)}
    raw_mano = {}
    for f in frames:
        entry = {}
        for side, pre in (("right", "rh"), ("left", "lh")):
            coeffs = torch.zeros(1, 16, 4)
            coeffs[..., 0] = 1.0  # wxyz identities
            entry[f"{pre}__pose_coeffs"] = coeffs
            entry[f"{pre}__tsl"] = torch.tensor(wrist[side], dtype=torch.float32)[None]
            entry[f"{pre}__betas"] = torch.zeros(1, 10)
        raw_mano[f] = entry
    anno = {
        "mocap_frame_id_list": frames,
        "raw_mano": raw_mano,
        "obj_transf": {o: {f: pose_track(p[None])[0] for f in frames} for o, p in obj_pos.items()},
    }
    (raw / "anno_preview").mkdir(parents=True)
    with open(raw / "anno_preview" / f"{OAKINK2_SEQ}.pkl", "wb") as f:
        pickle.dump(anno, f)
    program = {
        str(((100, 120), None)): {"obj_list_rh": ["obj_a"], "obj_list_lh": list(left_objects)},
        str(((120, 140), None)): {"obj_list_rh": ["obj_a", "obj_b"], "obj_list_lh": []},
        str((None, None)): {"obj_list_rh": [], "obj_list_lh": []},
        str(((500, 520), None)): {"obj_list_rh": ["obj_a"], "obj_list_lh": []},
    }
    (raw / "program" / "program_info").mkdir(parents=True)
    (raw / "program" / "program_info" / f"{OAKINK2_SEQ}.json").write_text(json.dumps(program))
    write_textured_box(raw / "object_repair" / "align_ds" / "obj_a" / "model.obj")
    mesh_b = raw / "object_repair" / "align_ds" / "obj_b" / "model.obj"
    mesh_b.parent.mkdir(parents=True)
    _point_cloud_mesh(right_verts[::25]).export(mesh_b)  # obj_b sits at the origin, its vertices on the hand's
    return {"obj_a": obj_pos["obj_a"], "obj_b": obj_pos["obj_b"], "right_wrist": wrist["right"]}


HOT3D_RECORDING = "P0001_aaaa"


def write_hot3d(
    raw: Path,
    mano_dir: Path,
    T: int = 100,
    touch: tuple[int, int] = (30, 80),
    no_object_at: tuple[int, ...] = (),
    idle: str | None = None,
) -> None:
    """One HOT3D recording (Aria, 30 Hz) where both hands (all but the ``idle`` one) rest on object ``111`` over
    frames ``touch`` and hover 1 m above everything otherwise; object ``222``, textured, stays far away, and has no
    pose on frames ``no_object_at``. Frame 5 also has a hand row 2 ms later (another camera stream of the same frame),
    which the reader merges.
    """
    rec = raw / "dataset" / HOT3D_RECORDING
    rec.mkdir(parents=True)
    (rec / "metadata.json").write_text(
        json.dumps({"object_uids": ["111", "222"], "have_hand_object_pose_gt": True, "headset": "Aria"})
    )
    stamps = [1_000_000_000 + 33_333_333 * t for t in range(T)]
    near_transl = {"right": np.array([0.2, 0.0, 0.3]), "left": np.array([0.2, 0.0, 0.3])}
    with open(rec / "mano_hand_pose_trajectory.jsonl", "w") as f:
        for t, stamp in enumerate(stamps):
            lift = 0.0 if touch[0] <= t < touch[1] else 1.0
            poses = {
                key: {
                    "wrist_xform": {
                        "q_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "t_xyz": (near_transl[side] + (0, 0, 1.0 if side == idle else lift)).tolist(),
                    },
                    "pose": [0.0] * 15,
                    "betas": [0.0] * 10,
                }
                for side, key in (("right", "1"), ("left", "0"))
            }
            f.write(json.dumps({"timestamp_ns": stamp, "hand_poses": poses}) + "\n")
            if t == 5:
                f.write(json.dumps({"timestamp_ns": stamp + 2_000_000, "hand_poses": poses}) + "\n")
    with open(rec / "dynamic_objects.csv", "w", newline="") as f:
        cols = [
            "timestamp[ns]",
            "object_uid",
            "t_wo_x[m]",
            "t_wo_y[m]",
            "t_wo_z[m]",
            "q_wo_w",
            "q_wo_x",
            "q_wo_y",
            "q_wo_z",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for t, stamp in enumerate(stamps):
            for uid, pos in (("111", (0.0, 0.0, 0.0)), ("222", (3.0, 3.0, 0.0))):
                if uid != "222" or t not in no_object_at:
                    writer.writerow(dict(zip(cols, [stamp, uid, *pos, 1.0, 0.0, 0.0, 0.0])))
    with open(rec / "headset_trajectory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"])
        writer.writeheader()
        writer.writerow({"t_wo_x[m]": 0.0, "t_wo_y[m]": -1.0, "t_wo_z[m]": 0.5})
    # Object 111's vertices are the resting hands' fingertips and every 20th vertex (on the mean hand: HOT3D's PCA).
    tips = list(CHORD_TIP_VERTICES.values())
    points = []
    for side in ("right", "left"):
        v = _hand_vertices(
            mano_dir, side, use_pca=True, num_pca_comps=15, flat_hand_mean=False, transl=near_transl[side]
        )
        points += [v[tips], v[::20]]
    assets = raw / "_assets" / "assets"
    assets.mkdir(parents=True)
    _point_cloud_mesh(np.concatenate(points)).export(assets / "111.glb")
    textured_box((200, 10, 10)).export(assets / "222.glb")  # textured, as the release's scans are


#: The texture colour of the synthetic releases' textured objects: DexYCB's, TACO's tool and OakInk2's obj_a.
TEXTURE_RGB = (10, 120, 200)


def textured_box(
    rgb: tuple[int, int, int] = TEXTURE_RGB, extents=(0.1, 0.1, 0.1), material: str = "material_0", uv_seed: int = 0
) -> trimesh.Trimesh:
    """A box textured with one colour; ``material`` names its material, which trimesh names the texture file after."""
    box = trimesh.creation.box(extents=extents)
    uv = np.random.default_rng(uv_seed).random((len(box.vertices), 2))
    box.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new("RGB", (4, 4), rgb))
    box.visual.material.name = material
    return box


def write_textured_box(path: Path, scale: float = 1.0, **kwargs) -> None:
    """:func:`textured_box` (``kwargs``) as an OBJ with its material and texture beside it, in ``scale`` units per
    metre.
    """
    mesh = textured_box(**kwargs)
    mesh.apply_scale(scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def write_dexycb(raw: Path, mano_dir: Path, pca: np.ndarray, T: int = 32) -> None:
    """One DexYCB sequence (30 Hz, right hand) in a camera 1 m above the AprilTag looking down, its first two frames
    without a hand; the master camera's labelled keypoints are the synthetic MANO of ``pose_m``, as DexYCB's are. Its
    object is a box textured :data:`TEXTURE_RGB`, its OBJ's material (``texture_map``) and texture beside it.
    """
    import yaml

    from eden.datasets.dex_ycb_dataset import _DEX_YCB_TO_EDEN_KEYPOINTS
    from eden.datasets.utils.body_model import mano_keypoints

    calib = raw / "calibration"
    (calib / "mano_subject_right").mkdir(parents=True)
    (calib / "extrinsics_cam").mkdir(parents=True)
    with open(calib / "mano_subject_right" / "mano.yml", "w") as f:
        yaml.safe_dump({"betas": [0.0] * 10}, f)
    # x_cam = diag(1, -1, -1) x_tag + (0, 0, 1).
    tag = np.hstack([np.diag([1.0, -1.0, -1.0]), [[0.0], [0.0], [1.0]]]).reshape(-1).tolist()
    with open(calib / "extrinsics_cam" / "extrinsics.yml", "w") as f:
        yaml.safe_dump({"extrinsics": {"apriltag": tag, "111": [0.0] * 12}, "master": "111"}, f)
    write_textured_box(raw / "models" / "002_master_chef_can" / "textured_simple.obj", material="texture_map")
    seq = raw / "subject" / "seq"
    seq.mkdir(parents=True)
    meta = {"extrinsics": "cam", "ycb_ids": [1], "ycb_grasp_ind": 0, "mano_sides": ["right"]}
    meta["mano_calib"] = ["subject_right"]
    with open(seq / "meta.yml", "w") as f:
        yaml.safe_dump(meta, f)
    pose_y = np.tile([0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.9], (T, 1, 1))
    pose_m = np.zeros((T, 1, 51))
    pose_m[2:, 0, 3:48] = pca  # the first two frames have no hand: trimmed
    pose_m[2:, 0, 48:51] = (0.1, 0.25, 0.8)
    np.savez(seq / "pose.npz", pose_y=pose_y, pose_m=pose_m)
    import smplx  # the `dev` group's reference implementation; the mano_dir fixture skips without it

    layer = smplx.create(str(mano_dir / "MANO_RIGHT.pkl"), model_type="mano", use_pca=False, flat_hand_mean=False)
    out = layer(
        hand_pose=torch.tensor(pca @ np.eye(45)[::-1], dtype=torch.float32)[None],  # the synthetic PCA basis
        transl=torch.tensor([[0.1, 0.25, 0.8]]),
    )
    eden_order = mano_keypoints(out.joints.detach(), out.vertices.detach()).numpy()[0]
    labels = np.zeros_like(eden_order)
    labels[list(_DEX_YCB_TO_EDEN_KEYPOINTS)] = eden_order
    (seq / "111").mkdir()
    for frame in range(T):
        np.savez(seq / "111" / f"labels_{frame:06d}.npz", joint_3d=labels[None].astype(np.float32))
