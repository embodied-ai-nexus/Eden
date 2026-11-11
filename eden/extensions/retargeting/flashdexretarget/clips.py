"""The imported-clip format (``<dataset>/human_demo/<clip>.npz``) and the object-mesh cache the pipeline shares.

All arrays are at the clip's ``fps``, in a z-up world, in metres; ``s`` is ``right`` or ``left``:

============================================  ===========================  ============================================
key                                           shape                        content
============================================  ===========================  ============================================
``fps``                                       ()                           frame rate (one per dataset)
``mano_s_wrist_pos``, ``mano_s_wrist_rot``    (T, 3), (T, 3, 3)            MANO wrist, rotation in the robot palm's
                                                                           convention (see :func:`mano_reference`)
``mano_s_joints``                             (T, 20, 3)                   MANO finger joints and tips,
                                                                           :data:`MANO_JOINT_NAMES` order
``obj_s_pose``, ``obj_s_id``                  (T, 4, 4), ()                object pose and its folder under
                                                                           ``objects/``
``obj_s_scale``, ``obj_s_mass``               ()                           optional mesh scale and mass
============================================  ===========================  ============================================
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
#: Longest edge (m) of the subdivided mesh whose vertices stand in for an object's surface in distance queries: every
#: surface point is within SURFACE_SPACING / sqrt(3) (1.2 mm) of one, where the raw vertices of a coarse mesh can be
#: centimetres from the middle of a face.
SURFACE_SPACING = 0.002
MANO_JOINT_NAMES = tuple(sorted(f"{f}_{p}" for f in FINGERS for p in ("proximal", "intermediate", "distal", "tip")))
SIDES = ("right", "left")


def mano_reference(
    joints: np.ndarray, verts: np.ndarray, global_rot: np.ndarray, side: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MANO layer output -> the clip's hand arrays.

    Parameters
    ----------
    joints : np.ndarray
        (T, 16, 3) MANO joints.
    verts : np.ndarray
        (T, 778, 3) MANO vertices.
    global_rot : np.ndarray
        (T, 3, 3) MANO global orientation.
    side : str
        ``right`` or ``left``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Wrist position (T, 3), wrist rotation rebased to the robot palm frame (T, 3, 3), and the 20 named joints
        (T, 20, 3) in :data:`MANO_JOINT_NAMES` order, tips taken from
        :data:`eden.datasets.utils.body_model.CHORD_TIP_VERTICES`.
    """
    # Imported here: importing eden.datasets loads every dataset module, which only the import step needs.
    from eden.datasets.utils.body_model import CHORD_TIP_VERTICES

    first_joint = {"index": 1, "middle": 4, "pinky": 7, "ring": 10, "thumb": 13}
    wrist = joints[:, 0]
    named = {}
    for finger, j in first_joint.items():
        for k, part in enumerate(("proximal", "intermediate", "distal")):
            named[f"{finger}_{part}"] = joints[:, j + k]
        named[f"{finger}_tip"] = verts[:, CHORD_TIP_VERTICES[finger]]
    joints20 = np.stack([named[n] for n in MANO_JOINT_NAMES], axis=1)
    rebase = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    if side == "left":
        rebase = rebase @ np.diag([-1.0, -1.0, 1.0])
    rot = np.asarray(global_rot, dtype=np.float64) @ rebase
    return wrist.astype(np.float32), rot.astype(np.float32), joints20.astype(np.float32)


#: The thread counts BLAS and OpenMP read when they load.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
#: What the CPU workers of :func:`worker_pool` run, imported once by the server they fork from.
WORKER_MODULES = (
    "eden.extensions.retargeting.flashdexretarget.preprocess",
    "eden.extensions.retargeting.flashdexretarget.retarget",
)


@contextlib.contextmanager
def single_threaded_workers() -> Iterator[None]:
    """Each of :data:`THREAD_VARS` the caller has not set is ``1`` while worker processes start inside, then removed
    again: a fresh worker process reads them as it loads numpy (or OpenMP), so N workers keep to N cores, and the
    caller's own environment is left as it was.
    """
    added = [v for v in THREAD_VARS if v not in os.environ]
    os.environ.update(dict.fromkeys(added, "1"))
    try:
        yield
    finally:
        for v in added:
            os.environ.pop(v, None)


@contextlib.contextmanager
def worker_pool(workers: int) -> Iterator[ProcessPoolExecutor]:
    """``workers`` processes for the CPU steps (preprocess, IK), each with one BLAS / OpenMP thread.

    They fork from a ``forkserver``, a fresh process that imports :data:`WORKER_MODULES` once under
    :func:`single_threaded_workers`, not from this one: a fork of this process would keep its thread pools as they
    are (numpy's sized to every core), and a spawned worker would spend seconds importing Eden. The server lives as
    long as this process, so the thread counts are the ones of the first pool.
    """
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(list(WORKER_MODULES))
    with single_threaded_workers(), ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        yield ex


def load_clip(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_clip(path: str | Path, clip: dict[str, np.ndarray]) -> None:
    """Write through a temporary file of this process's own, so an interrupted run never leaves a truncated clip and
    two runs never write one file.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **clip)
    os.replace(tmp, path)


def list_clips(clip_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(clip_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))


def sides(clip: dict) -> tuple[str, ...]:
    return tuple(s for s in SIDES if f"mano_{s}_wrist_pos" in clip)


def num_frames(clip: dict) -> int:
    return len(clip[f"mano_{sides(clip)[0]}_wrist_pos"])


def joints(clip: dict, side: str) -> np.ndarray:
    """(T, 20, 3) MANO joints in :data:`MANO_JOINT_NAMES` order."""
    names = MANO_JOINT_NAMES
    if f"mano_{side}_joint_names" in clip:
        names = [str(n) for n in clip[f"mano_{side}_joint_names"]]
    j = clip[f"mano_{side}_joints"]
    return j[:, [names.index(n) for n in MANO_JOINT_NAMES]].astype(np.float64)


def tips(clip: dict, side: str) -> np.ndarray:
    """(T, 5, 3) MANO fingertips, thumb to pinky."""
    cols = [MANO_JOINT_NAMES.index(f"{f}_tip") for f in FINGERS]
    return joints(clip, side)[:, cols]


def obj_id(clip: dict, side: str) -> str:
    return str(clip[f"obj_{side}_id"])


def obj_scale(clip: dict, side: str) -> float:
    return float(clip[f"obj_{side}_scale"]) if f"obj_{side}_scale" in clip else 1.0


def obj_pose(clip: dict, side: str) -> np.ndarray:
    return clip[f"obj_{side}_pose"].astype(np.float64)


def shared_object(clip: dict) -> bool:
    """Both hands on one object: the same id and the same pose track."""
    if sides(clip) != SIDES or obj_id(clip, "right") != obj_id(clip, "left"):
        return False
    return bool(np.allclose(obj_pose(clip, "right"), obj_pose(clip, "left"), atol=1e-6))


def objects(clip: dict) -> list[str]:
    """The clip's distinct object bodies, as hand sides (a shared object counts once)."""
    return ["right"] if shared_object(clip) else list(sides(clip))


class ObjectMeshes:
    """``objects/<id>/visual.obj`` (scaled) per (object, scale): vertices, convex hull, centre of mass, and the
    surface points (``surface``, the mesh subdivided to :data:`SURFACE_SPACING`) with their KD-tree (``tree``), which
    measures distances to the surface.

    Parameters
    ----------
    pool : str | Path
        The dataset directory holding ``objects/``.
    """

    def __init__(self, pool: str | Path) -> None:
        self.pool = Path(pool)
        self._cache: dict[tuple[str, float], dict] = {}

    def get(self, oid: str, scale: float = 1.0) -> dict:
        key = (oid, float(scale))
        if key not in self._cache:
            path = self.pool / "objects" / oid / "visual.obj"
            mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
            mesh.apply_scale(scale)
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            com = mesh.center_mass if mesh.is_watertight else verts.mean(0)
            surface, _ = trimesh.remesh.subdivide_to_size(verts, mesh.faces, max_edge=SURFACE_SPACING, max_iter=30)
            self._cache[key] = {
                "verts": verts,
                "surface": surface,
                "tree": cKDTree(surface),
                "hull": np.asarray(mesh.convex_hull.vertices, dtype=np.float64),
                "com": np.asarray(com, dtype=np.float64),
            }
        return self._cache[key]


def clip_fps(paths) -> float:
    """The frame rate the clips share (every clip stores its own ``fps``)."""
    rates = set()
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            if "fps" not in z.files:
                raise ValueError(f"{p.name} has no 'fps' entry; re-import it with `eden retarget flashdex import`.")
            rates.add(float(z["fps"]))
    if len(rates) != 1:
        raise ValueError(
            f"The clips of one dataset need one fps, got {sorted(rates)}; re-import them with one `fps` setting."
        )
    return rates.pop()


def object_ids(paths) -> list[str]:
    ids = set()
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            ids.update(str(z[f"obj_{s}_id"]) for s in SIDES if f"obj_{s}_id" in z.files)
    return sorted(ids)
