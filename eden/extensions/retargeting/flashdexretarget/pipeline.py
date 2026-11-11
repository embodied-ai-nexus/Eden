"""Kinematic retargeting: imported clips -> ``<dataset>/<robot>/motion.pt`` for one robot.

=================  ==================================================  ===========================================
step               does                                                writes (under the dataset)
=================  ==================================================  ===========================================
preprocess         support disks, reject rules                         ``<robot>/report.csv``
convex             convex decomposition (CoACD or CuACD)               ``objects/<id>/convex/*.obj``
retarget           MuJoCo physics IK                                   ``<robot>/retargeted/<clip>.npz``
pack               kept clips into one file                            ``<robot>/motion.pt``
urdf               object URDFs for simulation                         ``objects/<id>/object.urdf``
sdf                object SDF grids                                    ``objects/<id>/.sdf_cache/``
=================  ==================================================  ===========================================
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import csv
import os
from pathlib import Path

import numpy as np
import torch

import eden as en
from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.clips import clip_fps, list_clips, object_ids
from eden.extensions.retargeting.flashdexretarget.convex import (
    check_cuacd,
    check_parts,
    decompose_objects,
    parts_sha1,
)
from eden.extensions.retargeting.flashdexretarget.objects import write_object_urdfs
from eden.extensions.retargeting.flashdexretarget.pack import build, object_meshes
from eden.extensions.retargeting.flashdexretarget.preprocess import preprocess_clips
from eden.extensions.retargeting.flashdexretarget.retarget import cache_key, retarget_clips
from eden.extensions.retargeting.flashdexretarget.scene import robot_sha1
from eden.extensions.retargeting.flashdexretarget.sdf import bake, check_sdf

if TYPE_CHECKING:
    from eden.options.extensions.flashdexretarget import FDRKinematicRetargetingOptions


def _write_report(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # never a truncated report
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["clip"]))
    os.replace(tmp, path)


def _cache_keys(kept: dict, dataset: Path, cfg: FDRKinematicRetargetingOptions) -> dict[str, dict]:
    """Clip name -> the ``cache_key`` of its IK (each robot model and object's parts hashed once); raises for an
    object whose convex parts are missing or older than its mesh (:func:`check_parts`), the convex step off or not.
    """
    robots: dict[tuple[str, ...], str] = {}
    parts: dict[str, str] = {}
    keys = {}
    for path, disks in kept.items():
        with np.load(path, allow_pickle=False) as z:
            sides = tuple(s for s in C.SIDES if f"mano_{s}_wrist_pos" in z.files)
            ids = {str(z[f"obj_{s}_id"]) for s in sides}
        if sides not in robots:
            robots[sides] = robot_sha1(cfg.robot, sides)
        for oid in ids - parts.keys():
            check_parts(dataset / "objects" / oid)
            parts[oid] = parts_sha1(dataset / "objects" / oid)
        keys[path.stem] = cache_key(path, disks, cfg.robot, robots[sides], {o: parts[o] for o in ids}, cfg.retarget)
    return keys


def run_kinematic_retargeting(cfg: FDRKinematicRetargetingOptions) -> Path | None:
    """Run the enabled steps over ``cfg.dataset``.

    The robot's outputs go to ``<dataset>/<robot>/`` (``report.csv``, ``retargeted/``, ``motion.pt``), so each robot
    retargeted from one import keeps its own; ``human_demo/`` and ``objects/`` are shared. A dataset without
    ``human_demo/`` but with the robot's ``motion.pt`` only gets its objects prepared (convex, sdf).

    Parameters
    ----------
    cfg : FDRKinematicRetargetingOptions
        Pipeline settings.

    Returns
    -------
    Path | None
        ``<dataset>/<robot>/motion.pt`` when it exists at the end, else ``None``.
    """
    workers = cfg.workers or max(1, (os.cpu_count() or 2) // 2)  # half the CPU cores by default
    steps = cfg.steps
    # What the last steps need, checked before the hours of IK ahead of them.
    if steps.convex and cfg.convex.name == "cuacd":
        check_cuacd(cfg.convex)
    if steps.sdf:
        check_sdf(cfg.sdf)
    dataset = Path(cfg.dataset)
    # Per robot: retargeting another robot keeps this one's results, and two robots' runs never share a file.
    demo, out = dataset / "human_demo", dataset / cfg.robot
    retargeted, motion, report = out / "retargeted", out / "motion.pt", out / "report.csv"
    old = [p for p in (dataset / "motion.pt", dataset / "report.csv", dataset / "retargeted") if p.exists()]
    if old:  # the layout before the per-robot directories: a training run pointed at them would load the stale ones
        en.logger.warning(
            f"[kinematic_retargeting] {dataset} still holds {[p.name for p in old]} of an earlier version, which wrote "
            f"one robot's outputs into the dataset itself; this run writes {out}/. Delete them."
        )
    from_clips = demo.is_dir()
    if from_clips:
        fps = clip_fps(list_clips(demo))
        en.logger.info(f"[kinematic_retargeting] preprocess: {demo} ({fps:g} Hz)")
        out.mkdir(exist_ok=True)
        rows, kept = preprocess_clips(demo, dataset, cfg, fps, workers)
        _write_report(report, rows)
        if not kept:
            raise RuntimeError(f"Every clip was rejected; the reasons are in {report} (column 'reject').")
        ids = object_ids(kept)
    elif motion.exists():
        en.logger.info(f"[kinematic_retargeting] no {demo}: preparing the objects of {motion}")
        packed = torch.load(motion, map_location="cpu", weights_only=True)
        ids = sorted({mesh.parent.name for mesh, _ in object_meshes(packed, motion.parent)})
    else:
        raise FileNotFoundError(
            f"{dataset} has neither human_demo/ nor {cfg.robot}/motion.pt; import a release into it first with "
            "`eden retarget flashdex import`."
        )

    if steps.convex:
        en.logger.info(f"[kinematic_retargeting] convex: {len(ids)} objects (<object>/convex/, {cfg.convex.name})")
        decompose_objects(dataset, ids, cfg.convex, workers)
    if from_clips and (steps.retarget or steps.pack):
        keys = _cache_keys(kept, dataset, cfg)
        names = set(keys)
        if steps.retarget:
            en.logger.info(f"[kinematic_retargeting] retarget: {len(kept)} clips -> {retargeted} ({cfg.robot})")
            ik = {
                r["clip"]: r
                for r in retarget_clips(
                    list(kept.items()), dataset, retargeted, cfg.robot, cfg.retarget, fps, workers, keys
                )
            }
            for r in rows:
                r.update(ik.get(r["clip"], {}))
            _write_report(report, rows)
            failed = {name for name, r in ik.items() if r.get("error")}
            if failed == names:
                raise RuntimeError(f"Every clip failed to retarget; the errors are in {report} (column 'error').")
            names -= failed
        if steps.pack:
            en.logger.info(f"[kinematic_retargeting] pack: {retargeted} -> {motion}")
            build(retargeted, dataset, motion, fps, cfg.pack, cfg.robot, {n: keys[n] for n in sorted(names)})
    if steps.urdf and motion.exists():
        packed = torch.load(motion, map_location="cpu", weights_only=True)
        write_object_urdfs(dataset, [(mesh.parent.name, scale) for mesh, scale in object_meshes(packed, motion.parent)])
    if steps.sdf and not motion.exists():
        en.logger.info(f"[kinematic_retargeting] sdf: skipped, no {motion} yet")
    elif steps.sdf:
        en.logger.info(f"[kinematic_retargeting] sdf: <object>/.sdf_cache/, {cfg.sdf.grid_n}^3")
        bake(motion, cfg.sdf, workers)
    return motion if motion.exists() else None
