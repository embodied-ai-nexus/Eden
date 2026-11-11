"""Import: a released dataset -> ``<dataset>/human_demo/<clip>.npz`` + ``<dataset>/objects/<id>/visual.obj``.

The release is read by its :mod:`eden.datasets` loader (``TacoDataset``, ``OakInk2Dataset``, ``Hot3dDataset``,
``DexYCBDataset``) with its MANO parameters (``body_model_params``), in the loader's z-up world at the release's
frame rate; the loader cuts the clips and picks the object each hand holds. Per clip, this runs MANO, moves the
surface the objects rest on to ``table_z``, takes the wrist and fingertip reference the rewards compare against, and
resamples to ``fps``. A clip is skipped when an object flips more than ``max_obj_rot_step_deg`` in one source frame.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import gc
import hashlib
import itertools
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import eden as en
from eden.datasets.hand_object_base import count_reasons
from eden.datasets.utils.body_model import body_model_forward
from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.utils.geom import quat_to_R

if TYPE_CHECKING:
    from eden.datasets.base import DatasetBase
    from eden.options.extensions.flashdexretarget import FDRImportOptions

#: ``source.name`` -> the :data:`~eden.datasets.DATASET_REGISTRY` key of the loader that reads it.
SOURCE_DATASETS = {
    "taco": "taco_dataset",
    "oakink2": "oak_ink2_dataset",
    "hot3d": "hot3d_dataset",
    "dexycb": "dex_ycb_dataset",
}


@dataclass
class _Hand:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    pose_quat: torch.Tensor
    shape: torch.Tensor
    model: str


@dataclass
class _Clip:
    name: str
    recording: str
    fps: float
    hands: dict[str, _Hand]
    objects: dict[str, str]
    poses: dict[str, np.ndarray]


def open_dataset(cfg: FDRImportOptions) -> DatasetBase:
    """The :mod:`eden.datasets` loader of ``cfg.source``, built over ``cfg.raw`` with its MANO parameters."""
    from eden.datasets import DATASET_REGISTRY

    cls = DATASET_REGISTRY.get(SOURCE_DATASETS[cfg.source.name])
    params: dict[str, Any] = {
        "raw_data_path": cfg.raw,
        "body_model_params": True,
        "body_model_dir": cfg.mano_dir,
    }
    if cfg.source.name == "dexycb":
        from eden.datasets.dex_ycb_dataset import DEX_YCB_FRAMERATE

        raw = Path(cfg.raw)
        params |= {
            "target_framerate": DEX_YCB_FRAMERATE,
            "calibration_dir": str(raw / "calibration"),
            "models_dir": str(raw / "models"),
            # FlashDexRetarget fits its own support disks on the release's poses and meshes.
            "ground_initial_pose": False,
            "apply_mesh_correction": False,
        }
    else:
        params |= {"target_framerate": cls.source_framerate, "sides": cfg.sides}
    taken = sorted(set(cfg.source.dataset) & set(params))
    if taken:
        raise ValueError(
            f"source.dataset sets {taken}, which the import sets itself (from raw, mano_dir, sides and the release's "
            f"frame rate). Set them on the import options instead, or drop them from source.dataset."
        )
    return cls(cls.configure(**params, **cfg.source.dataset), torch.device("cpu"))


def _clips(dataset: DatasetBase, cfg: FDRImportOptions) -> tuple[list[_Clip], list[tuple[str, str]]]:
    """The dataset's clips with the hands of ``cfg.sides``, and ``(name, reason)`` for the ones it cannot give."""
    seqs, meta, fps = dataset.motion_sequences, dataset.metadata, float(dataset.target_framerate)
    starts = {key: [0, *torch.cumsum(seq.buffer_length, 0).tolist()] for key, seq in seqs.items()}

    def rows(key: str, i: int) -> slice:
        return slice(starts[key][i], starts[key][i + 1])

    def hand(key: str, i: int) -> _Hand:
        seq, r = seqs[key], rows(key, i)
        return _Hand(
            seq.base_pos_w.concatenated_tensor[r],
            seq.base_quat_w.concatenated_tensor[r],
            seq.body_pose_quat.concatenated_tensor[r],
            seq.body_shape.concatenated_tensor[r],
            seq.body_model[i],
        )

    def pose(key: str, i: int) -> np.ndarray:
        seq, r = seqs[key], rows(key, i)
        out = np.tile(np.eye(4), (r.stop - r.start, 1, 1))
        out[:, :3, :3] = quat_to_R(seq.base_quat_w.concatenated_tensor[r]).double().numpy()
        out[:, :3, 3] = seq.base_pos_w.concatenated_tensor[r].double().numpy()
        return out

    sides = C.SIDES if cfg.sides == "both" else (cfg.sides,)
    out, skipped = [], [tuple(s) for s in meta.get("skipped", [])]
    if "hand" in seqs:  # DexYCB: one hand per clip, its side in metadata["sides"]
        for i, side in enumerate(meta["sides"]):
            name = "dexycb_" + meta["sequence_ids"][i].replace("/", "_")
            missing = [s for s in sides if s != side]
            if missing:
                skipped.append((name, f"no hand with an object ({'/'.join(missing)})"))
                continue
            obj = meta["objects"][i]
            out.append(_Clip(name, name, fps, {side: hand("hand", i)}, {side: obj}, {obj: pose("object", i)}))
        return out, skipped
    for i, name in enumerate(meta["clip_names"]):
        objects = {s: meta[f"objects_{s}"][i] for s in sides}
        poses: dict[str, np.ndarray] = {}
        for s in sides:  # an object both hands hold has the same pose under both keys
            poses.setdefault(objects[s], pose(f"object_{s}", i))
        hands = {s: hand(f"{s}_hand", i) for s in sides}
        out.append(_Clip(name, meta["recordings"][i], fps, hands, objects, poses))
    return out, skipped


def _max_rot_step_deg(rot: np.ndarray) -> float:
    if len(rot) < 2:
        return 0.0
    step = Rotation.from_matrix(rot[1:] @ np.swapaxes(rot[:-1], 1, 2)).magnitude()
    return float(np.degrees(step.max()))


def _floor(poses: dict[str, np.ndarray], dataset: DatasetBase, scales: dict[str, float], fps: float) -> float:
    """Lowest point of the objects over the frames they stand still (< 0.1 m/s): the surface they rest on."""
    lows = []
    for oid, pose in poses.items():
        mesh = dataset.object_mesh(oid)
        s = scales.get(oid, 1.0)
        centre = pose[:, :3, :3] @ (mesh.vertices.mean(0) * s) + pose[:, :3, 3]
        speed = np.linalg.norm(np.diff(centre, axis=0), axis=1) * fps
        still = np.concatenate([speed < 0.1, speed[-1:] < 0.1])
        if still.any():
            hull = np.asarray(mesh.convex_hull.vertices) * s
            pts = np.einsum("tij,vj->tvi", pose[still, :3, :3], hull) + pose[still, None, :3, 3]
            lows.append(float(pts[..., 2].min()))
    return min(lows) if lows else 0.0


def _resample(x: np.ndarray, src: float, dst: float, rot: bool = False) -> np.ndarray:
    if src == dst:
        return x
    t_src = np.arange(len(x)) / src
    t_dst = np.arange(int((len(x) - 1) * dst / src + 1e-6) + 1) / dst
    if rot:
        return Slerp(t_src, Rotation.from_matrix(x))(t_dst).as_matrix()
    return interp1d(t_src, x, axis=0)(t_dst)


def _to_clip(pose: np.ndarray, z0: float, src: float, dst: float) -> np.ndarray:
    """World poses lowered by ``z0`` and resampled to the clip rate, float32."""
    out = np.tile(np.eye(4), (len(_resample(pose[:, 0, 3], src, dst)), 1, 1))
    out[:, :3, :3] = _resample(pose[:, :3, :3], src, dst, rot=True)
    out[:, :3, 3] = _resample(pose[:, :3, 3], src, dst)
    out[:, 2, 3] -= z0
    return out.astype(np.float32)


@dataclass
class _Recording:
    """What the clips of one recording share: every object's poses (``context``, empty for a release whose clips are
    whole recordings), the height they are lowered by when it comes from them (``z0``), and the desk poses.
    """

    context: dict[str, np.ndarray]
    z0: float | None
    desk: np.ndarray | None


def _recording(dataset: DatasetBase, recording: str, cfg: FDRImportOptions, fps: float) -> _Recording:
    source, scales = cfg.source, getattr(cfg.source, "scales", {})
    context = dataset.recording_object_poses(recording) if getattr(dataset, "has_recording_context", False) else {}
    if not context:  # no object of the recording has a mesh: each clip's own objects give its floor, and no desk
        return _Recording({}, None if source.table_z is not None else 0.0, None)
    # Over the whole recording, so all its clips share one height.
    z0 = 0.0 if source.table_z is None else _floor(context, dataset, scales, fps) - source.table_z
    desk = None
    if getattr(source, "desk_over_recording", False):
        desk = np.stack([_to_clip(context[o], z0, fps, cfg.fps) for o in sorted(context)])
    return _Recording(context, z0, desk)


def _write(dataset: DatasetBase, clip: _Clip, cfg: FDRImportOptions, out_dir: Path, rec: _Recording) -> dict:
    """Write one clip of recording ``rec``."""
    row = {"clip": clip.name}
    sides = tuple(clip.hands)
    held = sorted(set(clip.objects.values()))
    frames = len(next(iter(clip.poses.values())))
    if frames < 2:  # nothing to resample, and no motion to retarget
        return {**row, "skip": f"too short ({frames} frame)"}
    flips = [o for o in held if _max_rot_step_deg(clip.poses[o][:, :3, :3]) > cfg.max_obj_rot_step_deg]
    if flips:
        return {**row, "skip": f"object flips ({flips[0]})"}
    scales = getattr(cfg.source, "scales", {})
    z0 = rec.z0 if rec.z0 is not None else _floor(clip.poses, dataset, scales, clip.fps) - cfg.source.table_z
    out: dict = {"fps": np.float32(cfg.fps)}
    for side in sides:
        hand = clip.hands[side]
        joints, verts = body_model_forward(
            hand.model, cfg.mano_dir, hand.root_pos, hand.root_quat, hand.pose_quat, hand.shape
        )
        joints, verts = joints.double().numpy(), verts.double().numpy()
        joints[..., 2] -= z0
        verts[..., 2] -= z0
        rot = quat_to_R(hand.root_quat).double().numpy()
        wrist_pos, wrist_rot, mano_joints = C.mano_reference(joints, verts, rot, side)
        out[f"mano_{side}_wrist_pos"] = _resample(wrist_pos, clip.fps, cfg.fps).astype(np.float32)
        out[f"mano_{side}_wrist_rot"] = _resample(wrist_rot.astype(np.float64), clip.fps, cfg.fps, rot=True).astype(
            np.float32
        )
        out[f"mano_{side}_joints"] = _resample(mano_joints, clip.fps, cfg.fps).astype(np.float32)
        oid = clip.objects[side]
        out[f"obj_{side}_pose"] = _to_clip(clip.poses[oid], z0, clip.fps, cfg.fps)
        out[f"obj_{side}_id"] = np.array(oid)
        if scales.get(oid, 1.0) != 1.0:
            out[f"obj_{side}_scale"] = np.float32(scales[oid])
    if rec.desk is not None:
        ids = sorted(rec.context)
        out["desk_context_ids"] = np.array(ids)
        out["desk_context_scales"] = np.array([scales.get(o, 1.0) for o in ids], dtype=np.float32)
        out["desk_context_poses"] = rec.desk
    C.save_clip(out_dir / f"{clip.name}.npz", out)
    return {
        **row,
        "frames": len(out[f"mano_{sides[0]}_wrist_pos"]),
        "objects": " ".join(held),
        # The recording's other objects, whose meshes the import exports: only a clip that keeps them (its desk is
        # fitted under every object of the recording) needs them.
        "context": " ".join(sorted(rec.context)) if rec.desk is not None else "",
    }


#: Next to ``visual.obj``: the geometry and the materials it was exported from (:func:`_geometry_sha1`,
#: :func:`_material_sha1`).
MESH_SOURCE_FILE = "visual.json"


def _geometry_sha1(mesh) -> str:
    """Digest of a mesh's vertices (to the micrometre) and faces: what the import compares, since the OBJ text
    trimesh writes for one mesh can differ (normals it happened to compute, its version's formatting).
    """
    h = hashlib.sha1(np.round(np.asarray(mesh.vertices, dtype=np.float64), 6).tobytes())
    h.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()


#: What a trimesh material draws with, across its OBJ (``SimpleMaterial``) and glTF (``PBRMaterial``) materials.
_MATERIAL_FIELDS = (
    "name",
    "image",
    "ambient",
    "diffuse",
    "specular",
    "glossiness",
    "baseColorFactor",
    "baseColorTexture",
    "metallicFactor",
    "roughnessFactor",
    "metallicRoughnessTexture",
    "normalTexture",
    "occlusionTexture",
    "emissiveFactor",
    "emissiveTexture",
    "alphaMode",
    "alphaCutoff",
    "doubleSided",
)


def _digest(h, value) -> None:
    """Feed one material field (an image with its mode and size, a string, a number or an array) to ``h``."""
    if value is None:
        h.update(b"none")
    elif hasattr(value, "mode") and hasattr(value, "size") and hasattr(value, "tobytes"):  # a PIL image
        h.update(f"image {value.mode} {value.size}".encode())
        h.update(value.tobytes())
    elif isinstance(value, str):
        h.update(value.encode())
    else:
        a = np.asarray(value)
        h.update(f"{a.dtype} {a.shape}".encode())
        h.update(a.tobytes())


def _material_sha1(mesh) -> str:
    """Digest of what a mesh is drawn with: its UVs and every field of its material (texture images with their mode
    and size), or its vertex or face colours.
    """
    visual = mesh.visual
    h = hashlib.sha1(str(visual.kind).encode())
    if visual.kind == "texture":
        _digest(h, None if visual.uv is None else np.round(np.asarray(visual.uv, dtype=np.float64), 6))
        for field in _MATERIAL_FIELDS:
            h.update(field.encode())
            _digest(h, getattr(visual.material, field, None))
    elif visual.kind == "vertex":
        _digest(h, visual.vertex_colors)
    elif visual.kind == "face":
        _digest(h, visual.face_colors)
    return h.hexdigest()


def _directives(path: Path, *keywords: str) -> list[str]:
    """The arguments of an OBJ or MTL file's lines that start with one of ``keywords`` (a ``map_`` line's last token:
    its options come first), in file order.
    """
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            word, _, rest = line.strip().partition(" ")
            if any(word == k or (k.endswith("_") and word.startswith(k)) for k in keywords) and rest.strip():
                out.append(rest.split()[-1] if word.startswith("map_") else rest.strip())
    return out


def _export_files(obj_dir: Path) -> set[str]:
    """What an export wrote into ``obj_dir``: ``visual.obj``, the material libraries it names and the textures those
    name (files of ``obj_dir`` itself only).
    """
    obj = obj_dir / "visual.obj"
    if not obj.exists():
        return set()
    files = {"visual.obj"}
    for lib in _directives(obj, "mtllib"):
        files.add(lib)
        if (obj_dir / lib).is_file():
            files.update(_directives(obj_dir / lib, "map_", "bump", "disp", "norm"))
    return {f for f in files if Path(f).name == f and f != MESH_SOURCE_FILE}


def _same_file(a: Path, b: Path) -> bool:
    return b.is_file() and a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def _export_mesh(dataset: DatasetBase, oid: str, obj_dir: Path, textures: bool) -> str:
    """Write object ``oid``'s mesh to ``obj_dir/visual.obj`` unless it already holds that geometry (and, with
    ``textures``, the release's materials and textures), and remove what an earlier export wrote that this one does not.
    Without ``textures`` the mesh is the dataset's geometry-only one and the materials are not compared.

    Returns ``"kept"``; ``"written"`` (new, or exported before its geometry was recorded); ``"retextured"`` (the same
    geometry with other materials or textures: only those are rewritten, so ``visual.obj``, and the convex parts and
    SDF grids keyed on its bytes, stay as they are); ``"rewritten"`` (the same geometry, but its ``visual.obj`` comes
    out other bytes: other UVs, or another trimesh's formatting) or ``"changed"`` (another geometry). The last two
    replace ``visual.obj``, so its convex parts and SDF grids are redone.
    """
    # Loaded for this export alone: the dataset caches only the geometry, as decoded textures are large.
    mesh = dataset.object_mesh(oid, materials=True) if textures else dataset.object_mesh(oid)
    digest = {"geometry_sha1": _geometry_sha1(mesh)}
    if textures:
        digest["material_sha1"] = _material_sha1(mesh)
    path, source = obj_dir / "visual.obj", obj_dir / MESH_SOURCE_FILE
    recorded = json.loads(source.read_text()) if source.exists() and path.exists() else {}
    if all(recorded.get(k) == v for k, v in digest.items()):
        return "kept"
    same_geometry = recorded.get("geometry_sha1") == digest["geometry_sha1"]
    names = _directives(path, "usemtl") if same_geometry else []
    if len(names) == 1 and mesh.visual.kind == "texture":
        # Under the name visual.obj already uses (trimesh writes the material's own name, which a mesh loaded without
        # its materials has as material_0), so that only the material files can differ. A copy: the dataset caches
        # the mesh.
        mesh = mesh.copy()
        mesh.visual.material = mesh.visual.material.copy()
        mesh.visual.material.name = names[0]
    old = _export_files(obj_dir)
    # Exported by path, so trimesh writes material.mtl and the textures beside it; through a directory of this
    # process's own, visual.json (what marks the object done) written last.
    tmp = obj_dir.parent / f".{oid}.{os.getpid()}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        mesh.export(tmp / "visual.obj")
        new = {f.name for f in tmp.iterdir()}
        if same_geometry and (tmp / "visual.obj").read_bytes() == path.read_bytes():
            moved = [f for f in sorted(new - {"visual.obj"}) if not _same_file(tmp / f, obj_dir / f)]
            status = "retextured" if moved or old - new else "kept"
        else:
            moved = sorted(new, key=lambda f: f == "visual.obj")
            status = "written" if "geometry_sha1" not in recorded else "rewritten" if same_geometry else "changed"
            obj_dir.mkdir(parents=True, exist_ok=True)
        for f in moved:
            os.replace(tmp / f, obj_dir / f)
        for f in sorted(old - new):  # e.g. a texture an earlier export wrote under another name
            (obj_dir / f).unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tmp_source = source.with_name(f"{source.name}.{os.getpid()}.tmp")
    tmp_source.write_text(json.dumps(digest))
    os.replace(tmp_source, source)
    return status


def import_demos(cfg: FDRImportOptions) -> list[dict]:
    """``cfg.raw`` (a released dataset, ``cfg.source``) -> ``<dataset>/human_demo/<clip>.npz`` and
    ``<dataset>/objects/<id>/visual.obj``.

    Parameters
    ----------
    cfg : FDRImportOptions
        Import settings.

    Returns
    -------
    list[dict]
        One row per clip: its name and frame count and objects, or the reason it was skipped (``skip``).
    """
    mano_dir = Path(cfg.mano_dir)
    for f in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
        if not (mano_dir / f).exists():
            raise FileNotFoundError(
                f"{mano_dir / f} missing. The import needs the MANO hand models, which are not redistributable: "
                "register at mano.is.tue.mpg.de, download 'Models & Code' (mano_v1_2.zip) and point `mano_dir` at "
                "its models/ directory."
            )
    en.logger.info(f"[import] {cfg.source.name}: reading {cfg.raw}")
    dataset = open_dataset(cfg)
    clips, skipped = _clips(dataset, cfg)
    out_dir = Path(cfg.dataset) / "human_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p.stem for p in C.list_clips(out_dir)}
    rows = [{"clip": name, "skip": reason} for name, reason in skipped]
    done = 0
    for recording, group in itertools.groupby(sorted(clips, key=lambda c: c.recording), key=lambda c: c.recording):
        group = list(group)
        rec = _recording(dataset, recording, cfg, group[0].fps)
        rows += [_write(dataset, clip, cfg, out_dir, rec) for clip in group]
        if (done + len(group)) // 200 > done // 200 or done + len(group) == len(clips):
            en.logger.info(f"[import] {done + len(group)}/{len(clips)} clips")
        done += len(group)
    kept = [r for r in rows if "skip" not in r]
    reasons = count_reasons(r["skip"] for r in rows if "skip" in r)
    if not kept:
        # Before any earlier clip is removed below: an import that keeps nothing is a misconfiguration, not a release
        # whose clips all went away.
        raise RuntimeError(
            f"[import] every clip of {cfg.raw} was skipped ({len(rows)}: {reasons}), so nothing was written to "
            f"{out_dir} and its earlier clips are left as they are. First skips: {[r['skip'] for r in rows[:3]]}. "
            f"Check `sides` (now {cfg.sides!r}: each of those hands must hold an object) and the source's settings."
        )
    exported: dict[str, list[str]] = {}
    for oid in sorted({o for r in kept for k in ("objects", "context") if r.get(k) for o in r[k].split()}):
        status = _export_mesh(dataset, oid, Path(cfg.dataset) / "objects" / oid, cfg.textures)
        exported.setdefault(status, []).append(oid)
        if cfg.textures:
            # A trimesh mesh sits in reference cycles, so the textured one just exported is freed by the cycle
            # collector, which runs on object counts, not bytes: without this, the decoded textures (~50 MB each at
            # 4096^2) pile up, 1.5 GB over DexYCB's 20 objects.
            gc.collect()
    objects = Path(cfg.dataset) / "objects"
    changed, rewritten = exported.get("changed", []), exported.get("rewritten", [])
    retextured = exported.get("retextured", [])
    if retextured:
        en.logger.info(
            f"[import] {len(retextured)} objects already in {objects} got this release's materials and textures, their "
            f"visual.obj (and the convex parts and SDF grids made from it) unchanged, e.g. {retextured[:3]}"
        )
    if rewritten:
        en.logger.warning(
            f"[import] {len(rewritten)} objects already in {objects} have this release's geometry, but their "
            f"visual.obj was written again (other UVs, or another trimesh version's formatting), e.g. "
            f"{rewritten[:3]}; their convex parts, SDF grids and retargeted clips are redone."
        )
    if changed:
        en.logger.warning(
            f"[import] {len(changed)} objects already in {objects} had another mesh, now replaced "
            f"by this release's, e.g. {changed[:3]}; their convex parts, SDF grids and retargeted clips are redone."
        )
    en.logger.info(f"[import] {len(kept)} clips -> {out_dir}, skipped {len(rows) - len(kept)} {reasons}")
    # A clip this import skips but an earlier one wrote would still be retargeted: it goes.
    stale = sorted(before & {r["clip"] for r in rows if "skip" in r})
    for name in stale:
        (out_dir / f"{name}.npz").unlink()
    if stale:
        en.logger.info(
            f"[import] removed {len(stale)} clips of an earlier import that this one skips, e.g. {stale[:3]}"
        )
    others = sorted(before - {r["clip"] for r in rows})
    if others:
        en.logger.warning(
            f"[import] {out_dir} also holds {len(others)} clips this import did not write (another release or other "
            f"source settings), e.g. {others[:3]}; `eden retarget flashdex kinematic` retargets them too. Delete them "
            "if they are left over."
        )
    return rows
