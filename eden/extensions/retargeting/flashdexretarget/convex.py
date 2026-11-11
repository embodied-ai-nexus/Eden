"""Convex parts of every object (``<dataset>/objects/<id>/convex/<i>.obj``): the shapes the IK and the simulator
collide with. CoACD on CPU workers, or CuACD on GPUs followed by a port of CoACD's merge.

``convex/source.json`` records what the parts were made from (the ``visual.obj`` bytes, the backend and its
settings); parts made from anything else are redone.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import atexit
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull, cKDTree

import eden as en
from eden.extensions.retargeting.flashdexretarget import clips as C

if TYPE_CHECKING:
    from eden.options.extensions.flashdexretarget import FDRConvexOptions, FDRCuacdConvexOptions

_CUACD = None


def visible_gpus() -> list[str]:
    """The ``CUDA_VISIBLE_DEVICES`` entries, or every device index torch sees."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",") if visible else [str(g) for g in range(torch.cuda.device_count())]


def pick_gpus(indices: list[int] | None, field: str, cpu_fix: str) -> list[str]:
    """The visible GPUs at ``indices`` (every one for None); raises naming ``field`` when there are none or an index
    is out of range.
    """
    visible = visible_gpus()
    if not visible:
        raise RuntimeError(f"{field} asks for a GPU but none is visible (CUDA_VISIBLE_DEVICES, torch); {cpu_fix}.")
    bad = [g for g in indices or () if g >= len(visible)]
    if bad:
        raise ValueError(
            f"{field}={indices} indexes the visible GPUs, which are {len(visible)} ({','.join(visible)}): "
            f"{bad} out of range. Use indices 0..{len(visible) - 1}, or null for every one."
        )
    return [visible[g] for g in indices] if indices is not None else visible


def check_cuacd(cfg: FDRCuacdConvexOptions) -> list[str]:
    """The GPUs CuACD runs on; raises when ``cuacd`` is missing or no requested GPU is visible."""
    try:
        import cuacd  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "convex.name='cuacd' needs the `cuacd` package (`uv pip install cuacd`); or decompose on CPU with "
            "`--override 'convex: {name: coacd}'`."
        ) from e
    return pick_gpus(cfg.gpus, "convex.gpus", "decompose on CPU with `--override 'convex: {name: coacd}'`")


#: What :func:`_write` records next to the parts.
SOURCE_FILE = "source.json"


def has_parts(convex_dir: Path) -> bool:
    return convex_dir.is_dir() and any(p.suffix == ".obj" and p.stem.isdigit() for p in convex_dir.iterdir())


def parts_source(obj_dir: Path, cfg: FDRConvexOptions) -> dict:
    """What the parts of ``obj_dir`` are made from under ``cfg``: the mesh's bytes, the backend and its settings
    (``force`` and the GPUs it runs on aside).
    """
    return {"visual_sha1": _visual_sha1(obj_dir), "convex": cfg.model_dump(mode="json", exclude={"force", "gpus"})}


def _visual_sha1(obj_dir: Path) -> str:
    return hashlib.sha1((Path(obj_dir) / "visual.obj").read_bytes()).hexdigest()


def check_parts(obj_dir: Path) -> None:
    """Raise unless ``obj_dir`` has parts made from its current ``visual.obj``, whatever the convex settings: the IK
    collides with the parts while pack and the SDF grids read the mesh, so parts of an older mesh mix two shapes.
    """
    convex = Path(obj_dir) / "convex"
    if not has_parts(convex):
        raise FileNotFoundError(
            f"{obj_dir} has no convex parts ({convex}/<i>.obj), which the IK collides with; make them with the convex "
            "step (steps.convex: true)."
        )
    try:
        made_from = json.loads((convex / SOURCE_FILE).read_text())["visual_sha1"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        made_from = None
    if made_from != _visual_sha1(obj_dir):
        raise ValueError(
            f"The convex parts in {convex} were not made from the current {Path(obj_dir) / 'visual.obj'} (re-imported "
            f"since, or the parts predate {convex / SOURCE_FILE}), so IK on them would collide with another shape than "
            "the one packed and baked. Redo them with the convex step (steps.convex: true)."
        )


def parts_current(obj_dir: Path, source: dict) -> bool:
    """``obj_dir`` has parts, made from ``source`` (:func:`parts_source`)."""
    convex = Path(obj_dir) / "convex"
    if not has_parts(convex) or not (convex / SOURCE_FILE).exists():
        return False
    try:
        return json.loads((convex / SOURCE_FILE).read_text()) == source
    except json.JSONDecodeError:  # a damaged record: the parts are redone, as for none
        return False


def parts_sha1(obj_dir: Path) -> str:
    """Digest of the parts' bytes, which the IK collides with (what a retargeted clip's ``cache_key`` keeps)."""
    h = hashlib.sha1()
    for part in convex_parts(obj_dir):
        h.update(part.name.encode())
        h.update(part.read_bytes())
    return h.hexdigest()


def convex_parts(obj_dir: Path) -> list[Path]:
    """``<obj_dir>/convex/<i>.obj`` in part order."""
    convex = Path(obj_dir) / "convex"
    return sorted((p for p in convex.glob("*.obj") if p.stem.isdigit()), key=lambda p: int(p.stem))


def _load(obj: Path, decimate_faces: int) -> trimesh.Trimesh:
    mesh = trimesh.load(obj / "visual.obj", process=False, skip_materials=True, force="mesh")
    if decimate_faces and len(mesh.faces) > decimate_faces:  # dense scans make CoACD very slow
        mesh = mesh.simplify_quadric_decimation(face_count=decimate_faces)
    return mesh


def _write(obj: Path, parts: list[trimesh.Trimesh], source: dict) -> None:
    """``<obj>/convex/<i>.obj`` and ``source.json`` (``source``), swapped in whole: an interrupted run leaves the old
    parts or none, never a partial set that :func:`has_parts` would take for done.
    """
    if not parts:
        raise ValueError(
            f"The convex decomposition of {obj / 'visual.obj'} has no part (CuACD drops hulls under 4 vertices), so "
            "the object would collide with nothing. Check the mesh, or decompose it with CoACD "
            "(`--override 'convex: {name: coacd}'`)."
        )
    out, tmp = obj / "convex", obj / f"convex.{os.getpid()}.tmp"  # this process's own: two runs never share one
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for i, part in enumerate(parts):
        part.export(tmp / f"{i}.obj")
    (tmp / SOURCE_FILE).write_text(json.dumps(source, sort_keys=True))
    shutil.rmtree(out, ignore_errors=True)
    try:
        os.replace(tmp, out)
    except OSError:
        # Another run on this dataset swapped its parts in between: they stand if made from the same source.
        shutil.rmtree(tmp, ignore_errors=True)
        if not parts_current(obj, source):
            raise


def decompose(
    obj_dir: str, coacd_kwargs: dict, source: dict, decimate_faces: int = 0, seed: int = 1
) -> tuple[str, int, float]:
    """CoACD parts of ``<obj_dir>/visual.obj`` (made from ``source``); returns the object's name, its part count and
    the seconds taken.
    """
    import coacd

    t0 = time.time()
    obj = Path(obj_dir)
    mesh = _load(obj, decimate_faces)
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), seed=seed, **coacd_kwargs)
    _write(obj, [trimesh.Trimesh(vs, np.asarray(fs, dtype=int)) for vs, fs in parts], source)
    return obj.name, len(parts), time.time() - t0


def _cuacd_init(counter, gpus: list[str]) -> None:
    global _CUACD
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
    _CUACD = _cuacd_context()


def _cuacd_context():
    import cuacd

    ctx = cuacd.Context()
    atexit.register(ctx.close)  # before interpreter teardown, where its __del__ fails
    return ctx


def _cuacd_hulls(mesh: trimesh.Trimesh, cfg: FDRCuacdConvexOptions) -> list[np.ndarray]:
    verts, tris = mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
    if _CUACD.check_mesh(verts, tris)["needs_remesh"]:
        verts, tris = _CUACD.preprocess(verts, tris, resolution=cfg.preprocess_resolution)
    lo, hi = verts.min(0), verts.max(0)
    c, s = (lo + hi) / 2, float((hi - lo).max()) / 2  # CuACD works in a ~2-unit box
    out = _CUACD.lookahead_decompose(
        ((verts - c) / s).astype(np.float32),
        tris,
        threshold=cfg.threshold,
        depth=cfg.depth,
        decompose_components=True,
        merge_hulls=True,
    )
    return [hv * s + c for _, _, hv, _ in out if len(hv) >= 4]


def decompose_cuacd(
    obj_dir: str, cfg: FDRCuacdConvexOptions, source: dict, decimate_faces: int = 0
) -> tuple[str, int, float]:
    """CuACD on this worker's GPU, then CoACD's merge (:func:`merge_parts`); the parts are made from ``source``."""
    global _CUACD

    t0 = time.time()
    obj = Path(obj_dir)
    mesh = _load(obj, decimate_faces)
    try:
        hulls = _cuacd_hulls(mesh, cfg)
    except RuntimeError:  # a GPU error leaves the context unusable: retry once on a fresh one, as cuacd's CLI does
        _CUACD.close()
        _CUACD = _cuacd_context()
        hulls = _cuacd_hulls(mesh, cfg)
    parts = merge_parts(hulls, cfg.threshold, cfg.max_convex_hull)
    _write(obj, [trimesh.convex.convex_hull(p) for p in parts], source)
    return obj.name, len(parts), time.time() - t0


def decompose_objects(pool: str | Path, ids: list[str], cfg: FDRConvexOptions, workers: int) -> None:
    """Convex parts of every object of ``ids`` that has none made from its current ``visual.obj`` with ``cfg``'s
    backend and settings (all of them with ``cfg.force``).

    Parameters
    ----------
    pool : str | Path
        The dataset directory holding ``objects/<id>/visual.obj``.
    ids : list[str]
        Object ids.
    cfg : FDRConvexOptions
        The backend and its settings.
    workers : int
        CPU processes for CoACD (CuACD runs one per GPU).
    """
    root = Path(pool) / "objects"
    missing = [str(root / i) for i in ids if not (root / i / "visual.obj").exists()]
    if missing:
        raise FileNotFoundError(
            f"visual.obj missing for {len(missing)} objects, e.g. {missing[0]}; the import writes them "
            "(`eden retarget flashdex import`)."
        )
    sources = {root / i: parts_source(root / i, cfg) for i in ids}
    todo = [p for p, src in sources.items() if cfg.force or not parts_current(p, src)]
    if cfg.name == "coacd":
        where = f"{workers} CPU workers"
    else:
        gpus = check_cuacd(cfg)
        workers, where = len(gpus), f"GPU {','.join(gpus)}"
    en.logger.info(f"[convex] {len(todo)} of {len(ids)} objects need convex parts ({cfg.name}, {where})")
    if not todo:
        return
    with C.single_threaded_workers():  # one OpenMP thread in each worker, started inside
        if cfg.name == "coacd":
            kwargs = cfg.coacd.model_dump()
            # Spawned, not forked: a child forked from a process that has run CoACD inherits its OpenMP pool's locks and
            # hangs in its first run_coacd.
            ex = ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=mp.get_context("spawn"))
            futures = {ex.submit(decompose, str(p), kwargs, sources[p], cfg.decimate_faces, cfg.seed): p for p in todo}
        else:
            ctx = mp.get_context("spawn")
            ex = ProcessPoolExecutor(
                max_workers=min(workers, len(todo)),
                mp_context=ctx,
                initializer=_cuacd_init,
                initargs=(ctx.Value("i", 0), gpus),
            )
            futures = {ex.submit(decompose_cuacd, str(p), cfg, sources[p], cfg.decimate_faces): p for p in todo}
        with ex:
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    name, n, sec = fut.result()
                except Exception as e:
                    for f in futures:  # do not sit through every other object before the error shows
                        f.cancel()
                    raise RuntimeError(
                        f"Convex decomposition ({cfg.name}) of {futures[fut] / 'visual.obj'} failed: {e!r}"
                    ) from e
                en.logger.info(f"[convex] {i}/{len(todo)} {name}: {n} parts, {sec:.0f} s")


# --------------------------------------------------------------------------------------------------------------------
# CoACD's merge over CuACD's parts
# --------------------------------------------------------------------------------------------------------------------


def _inside(hulls: list[ConvexHull], pts: np.ndarray, tol: float) -> np.ndarray:
    ins = np.zeros(len(pts), bool)
    for h in hulls:
        ins |= (pts @ h.equations[:, :3].T + h.equations[:, 3] <= tol).all(1)
    return ins


def _surface(h: ConvexHull, n: int, seed: int) -> np.ndarray:
    return trimesh.sample.sample_surface(trimesh.Trimesh(h.points, h.simplices), n, seed=seed)[0]


class _Parts:
    """Merged hulls, each with the unmerged parts it covers: stick-out is measured against those, so a chain of
    merges cannot pile its allowances up.
    """

    def __init__(self, parts: list[np.ndarray]):
        self.raw = [ConvexHull(np.asarray(p, dtype=np.float64)) for p in parts]
        self.raw_pts = [_surface(h, 1500, i) for i, h in enumerate(self.raw)]
        self.hulls, self.trees, self.boxes, self.members = [], [], [], []
        for i, h in enumerate(self.raw):
            self.add(h, [i])

    def add(self, h: ConvexHull, members: list[int]) -> int:
        v = h.points[h.vertices]
        self.hulls.append(ConvexHull(v))
        self.trees.append(cKDTree(np.concatenate([self.raw_pts[m] for m in members])))
        self.boxes.append((v.min(0), v.max(0)))
        self.members.append(members)
        return len(self.hulls) - 1

    def joint(self, i: int, j: int) -> ConvexHull:
        return ConvexHull(np.concatenate([self.hulls[i].points, self.hulls[j].points]))

    def stick_out(self, i: int, j: int, limit: float) -> tuple[float, ConvexHull | None]:
        """Max distance from the joint hull's surface to the unmerged parts of the two (inf when farther apart
        than ``2 * limit``).
        """
        (lo_i, hi_i), (lo_j, hi_j) = self.boxes[i], self.boxes[j]
        if np.maximum(0, np.maximum(lo_i - hi_j, lo_j - hi_i)).max() > 2 * limit:
            return np.inf, None
        h = self.joint(i, j)
        x = _surface(h, 1000, i * 7919 + j)
        x = x[~_inside([self.raw[m] for m in self.members[i] + self.members[j]], x, 1e-6)]
        if not len(x):
            return 0.0, h
        return float(np.minimum(self.trees[i].query(x)[0], self.trees[j].query(x)[0]).max()), h


def _greedy(parts: _Parts, alive: set[int], cost, stop) -> set[int]:
    costs = {(i, j): cost(i, j) for i, j in itertools.combinations(sorted(alive), 2)}
    while costs and len(alive) > 1:
        (i, j), (c, h) = min(costs.items(), key=lambda kv: kv[1][0])
        if stop(c, len(alive)):
            break
        k = parts.add(h, parts.members[i] + parts.members[j])
        alive -= {i, j}
        costs = {p: v for p, v in costs.items() if i not in p and j not in p}
        costs.update({(o, k): cost(o, k) for o in alive})
        alive.add(k)
    return alive


def merge_parts(parts: list[np.ndarray], threshold: float, max_parts: int) -> list[np.ndarray]:
    """Convex parts (vertex arrays) after CoACD's merge: cheapest first, pairs whose joint hull sticks out of their
    unmerged parts by at most ``threshold`` times half the overall extent; then, above ``max_parts``, the pairs
    sticking out least.
    """
    if len(parts) < 2:
        return parts
    parts = [np.asarray(p, dtype=np.float64) for p in parts]
    parts = sorted((p[np.lexsort(p.T[::-1])] for p in parts), key=lambda p: (*np.round(p.mean(0), 6), len(p)))
    allp = np.concatenate(parts)
    half = float((allp.max(0) - allp.min(0)).max()) / 2
    limit = threshold * half
    p = _Parts(parts)
    alive = _greedy(p, set(range(len(parts))), lambda i, j: p.stick_out(i, j, limit), lambda c, n: c > limit)
    if len(alive) > max_parts:
        alive = _greedy(p, alive, lambda i, j: p.stick_out(i, j, half), lambda c, n: n <= max_parts)
    return [p.hulls[i].points for i in sorted(alive)]
