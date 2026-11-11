"""Preprocess: support disks and reject rules over every imported clip (writes only the report rows)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import functools
from pathlib import Path

import numpy as np

import eden as en
from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.desk import support_disks
from eden.extensions.retargeting.flashdexretarget.filters import features, reject_reasons

if TYPE_CHECKING:
    from eden.options.extensions.flashdexretarget import FDRKinematicRetargetingOptions


@functools.cache
def _meshes(pool: str) -> C.ObjectMeshes:
    return C.ObjectMeshes(pool)


def _process(path: str, pool: str, cfg: FDRKinematicRetargetingOptions, fps: float) -> tuple[dict, np.ndarray | None]:
    """A clip's report row and support disks. A clip the rules reject has ``reject`` set; anything that raises (a
    missing mesh, a malformed clip) is not a verdict on the clip and stops the run, naming it.
    """
    meshes = _meshes(pool)
    path = Path(path)
    row: dict = {"clip": path.stem}
    try:
        clip = C.load_clip(path)
        desk, filt = cfg.desk.mode, cfg.filters.enable
        if desk == "fit" or (desk == "keep" and "support_disks" not in clip):
            clip["support_disks"] = support_disks(clip, meshes, cfg.desk, fps)
        if filt:
            row.update(features(clip, meshes, cfg.filters, fps))
    except Exception as e:
        raise RuntimeError(
            f"Preprocessing {path} failed: {e!r}. Re-import the clip with `eden retarget flashdex import`, or move it "
            f"out of {path.parent} to go on without it."
        ) from e
    reasons = reject_reasons(row, cfg.filters) if filt else []
    row["disks"] = len(np.asarray(clip.get("support_disks", np.zeros((0, 5)))).reshape(-1, 5))
    row["reject"] = ";".join(reasons)
    return row, clip.get("support_disks")


def preprocess_clips(
    clips: Path, pool: Path, cfg: FDRKinematicRetargetingOptions, fps: float, workers: int
) -> tuple[list[dict], dict[Path, np.ndarray | None]]:
    """Report rows of every clip, and the kept clips with their support disks.

    Parameters
    ----------
    clips : Path
        The ``human_demo/`` directory.
    pool : Path
        The dataset directory holding ``objects/``.
    cfg : FDRKinematicRetargetingOptions
        Pipeline settings (``desk``, ``filters``).
    fps : float
        The clips' frame rate.
    workers : int
        Processes.

    Returns
    -------
    tuple[list[dict], dict[Path, np.ndarray | None]]
        One report row per clip (``reject`` empty when kept), and kept clip path -> its ``(K, 5)`` support disks.
    """
    paths = C.list_clips(clips)
    en.logger.info(f"[preprocess] {len(paths)} clips, desk={cfg.desk.mode}, filter={cfg.filters.enable}")
    jobs = [(str(p), str(pool), cfg, fps) for p in paths]
    with C.worker_pool(max(1, min(workers, len(jobs)))) as ex:
        results = list(ex.map(_process, *zip(*jobs))) if jobs else []

    rows = [r for r, _ in results]
    kept = {p: disks for p, (r, disks) in zip(paths, results) if not r["reject"]}
    counts: dict[str, int] = {}
    for r in rows:
        for reason in filter(None, r["reject"].split(";")):
            counts[reason] = counts.get(reason, 0) + 1
    en.logger.info(f"[preprocess] kept {len(kept)} / {len(rows)}  {counts}")
    return rows, kept
