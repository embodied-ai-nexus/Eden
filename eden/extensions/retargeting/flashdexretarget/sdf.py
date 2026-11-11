"""Object SDF grids: ``(4, N, N, N)`` signed distance (negative inside) and its gradient over ``[-extent, extent]^3``
in the object frame, baked with warp's BVH closest-point query and winding-number sign and cached in
``<object>/.sdf_cache/``. The training task reads them for its SDF observations, rewards and contact gating.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import hashlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import trimesh

import eden as en
from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.convex import pick_gpus, visible_gpus
from eden.extensions.retargeting.flashdexretarget.pack import object_meshes
from eden.utils.distribution import install_hint

if TYPE_CHECKING:
    from eden.options.extensions.flashdexretarget import FDRSdfOptions

_DEVICE = "cpu"


def cache_path(mesh_path: str, extent: float, N: int, mesh_scale: float) -> Path:
    """Where the grid of one (mesh, scale, extent, N) is cached, keyed on the mesh file's bytes: a re-imported mesh
    gets a grid of its own instead of the old one's.
    """
    mesh_path_p = Path(mesh_path)
    mesh_sha1 = hashlib.sha1(mesh_path_p.read_bytes()).hexdigest()
    key = hashlib.md5(f"{mesh_sha1}|method=v2_analytic|e={extent}|n={N}|s={mesh_scale}".encode()).hexdigest()[:12]
    return mesh_path_p.parent / ".sdf_cache" / f"sdf_v3_e{extent}_n{N}_s{mesh_scale}_{key}.pt"


def _preprocess_mesh(mesh_path: str, mesh_scale: float, max_faces: int = 16000) -> trimesh.Trimesh:
    """Load, scale and decimate to ``max_faces`` (detail below a voxel only slows the closest-point query)."""
    mesh = trimesh.load(Path(mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"SDF bake expected a single trimesh.Trimesh, got {type(mesh)} from {mesh_path!r}")
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)
    if len(mesh.faces) > max_faces:
        decim = mesh.simplify_quadric_decimation(face_count=max_faces)
        if isinstance(decim, trimesh.Trimesh) and len(decim.faces) > 0:
            mesh = decim
    return mesh


def _import_warp():
    try:
        import warp as wp

        from eden.extensions.retargeting.flashdexretarget import _sdf_kernel
    except ImportError as e:
        raise ImportError(
            f"Baking object SDF grids needs `warp-lang`: install it with {install_hint('flashdexretarget')}. Or skip "
            "the step with `--override 'steps: {sdf: false}'`."
        ) from e
    return wp, _sdf_kernel.closest_point


def _build_object_sdf(mesh: trimesh.Trimesh, extent: float, N: int, device: str = "cpu") -> np.ndarray:
    """BVH closest-point queries: seconds and < 1 GB for 128^3, where ``trimesh.proximity`` takes hours and ~17 GB
    (it expands every candidate face of every grid point far from the mesh).
    """
    wp, kernel = _import_warp()
    xs = np.linspace(-extent, extent, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    wp.init()
    wmesh = wp.Mesh(
        points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(mesh.faces.astype(np.int32).reshape(-1), dtype=wp.int32, device=device),
        support_winding_number=True,
    )
    closest_wp = wp.empty(len(pts), dtype=wp.vec3, device=device)
    sign_wp = wp.empty(len(pts), dtype=float, device=device)
    wp.launch(
        kernel,
        dim=len(pts),
        device=device,
        inputs=[wmesh.id, wp.array(pts, dtype=wp.vec3, device=device), 1.0e6, closest_wp, sign_wp],
    )
    closest, sign = closest_wp.numpy(), sign_wp.numpy().astype(np.float32)  # sign: -1 inside (winding number)

    disp = (pts - closest).astype(np.float32)
    dist = np.linalg.norm(disp, axis=-1)
    sdf = (sign * dist).astype(np.float32).reshape(N, N, N)
    unit = disp / np.maximum(dist[:, None], 1e-8)
    grad = (sign[:, None] * unit).astype(np.float32).reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    return np.concatenate([sdf[None], grad], axis=0)


def bake_object_sdf_grid(
    mesh_path: str, mesh_scale: float, extent: float, N: int, device: str, bake_device: str | None = None
) -> torch.Tensor:
    """The ``(4, N, N, N)`` grid of one mesh, from the cache or baked on ``bake_device`` (default ``device``).

    Parameters
    ----------
    mesh_path : str
        The object's ``visual.obj``.
    mesh_scale : float
        Scale applied to the mesh.
    extent : float
        Half-size (m) of the grid.
    N : int
        Grid points per axis.
    device : str
        Device of the returned tensor.
    bake_device : str | None
        warp device of the bake.

    Returns
    -------
    torch.Tensor
        Channel 0 the signed distance (negative inside), channels 1-3 its gradient; axis order x, y, z.
    """
    cache_file = cache_path(mesh_path, extent, N, mesh_scale)
    if cache_file.exists():
        return torch.load(cache_file, map_location=device, weights_only=True).to(device=device, dtype=torch.float32)
    grid = _build_object_sdf(_preprocess_mesh(mesh_path, mesh_scale), extent, N, str(bake_device or device))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # Through a file of this process's own, so an interrupted bake (or a concurrent one) never leaves a truncated
    # grid that every later run would load.
    tmp = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
    torch.save(torch.from_numpy(grid), tmp)
    os.replace(tmp, cache_file)
    return torch.from_numpy(grid).to(device=device, dtype=torch.float32)


def _init_worker(counter, gpus: list[str], device: str) -> None:
    global _DEVICE
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    # one visible GPU per worker: warp init over every visible GPU takes ~20 s with 8 workers
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)] if gpus else ""
    _DEVICE = "cuda:0" if device == "cuda" else "cpu"
    torch.set_num_threads(1)


def _bake(mesh: str, scale: float, extent: float, n: int) -> tuple[str, float]:
    t0 = time.time()
    bake_object_sdf_grid(mesh, scale, extent, n, device="cpu", bake_device=_DEVICE)
    return Path(mesh).parent.name, time.time() - t0


def check_sdf(cfg: FDRSdfOptions) -> list[str]:
    """The GPUs the bake runs on (none on CPU); raises when warp is missing or no requested GPU is visible."""
    _import_warp()
    if cfg.device != "cuda":
        return []
    return pick_gpus(cfg.gpus, "sdf.gpus", "bake on CPU with `--override 'sdf: {device: cpu}'`")


def bake(motion: Path, cfg: FDRSdfOptions, workers: int) -> None:
    """Grids of every (mesh, scale) ``motion`` references that has none cached.

    Parameters
    ----------
    motion : Path
        The ``motion.pt``; meshes resolve next to it.
    cfg : FDRSdfOptions
        Grid and device settings.
    workers : int
        CPU processes when ``cfg.device`` is ``cpu``.
    """
    gpus = check_sdf(cfg)
    packed = torch.load(motion, map_location="cpu", weights_only=True)
    meshes = object_meshes(packed, Path(motion).parent)
    missing = [str(m) for m, _ in meshes if not m.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} meshes {Path(motion).name} references are missing, e.g. {missing[0]}")
    todo = [(m, s) for m, s in meshes if not cache_path(str(m), cfg.extent, cfg.grid_n, s).exists()]
    if cfg.device == "cuda":
        workers = cfg.workers_per_gpu * len(gpus)
    else:
        gpus = visible_gpus()[:1]  # warp still initializes CUDA
    en.logger.info(
        f"[sdf] {len(todo)} of {len(meshes)} meshes need a grid, {cfg.grid_n}^3 over +-{cfg.extent} m, "
        f"{workers} workers on {'GPU ' + ','.join(gpus) if cfg.device == 'cuda' else 'CPU'}"
    )
    if not todo:
        return
    ctx = mp.get_context("spawn")
    with (
        C.single_threaded_workers(),
        ProcessPoolExecutor(
            max_workers=min(workers, len(todo)),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(ctx.Value("i", 0), gpus, cfg.device),
        ) as ex,
    ):
        futures = {ex.submit(_bake, str(m), s, cfg.extent, cfg.grid_n): m for m, s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                name, sec = fut.result()
            except Exception as e:
                for f in futures:  # do not sit through every other mesh before the error shows
                    f.cancel()
                raise RuntimeError(f"SDF bake of {futures[fut]} failed: {e!r}") from e
            en.logger.info(f"[sdf] {i}/{len(todo)} {name} {sec:.1f} s")
