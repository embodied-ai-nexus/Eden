"""Retarget step: every IK attempt of every kept clip as a job of its own, the best attempt per clip saved to
``<robot>/retargeted/<clip>.npz`` and scored for the report.

A saved clip is a cache keyed (``cache_key``) on everything its IK read: the robot and its model files, the imported
clip's bytes, its support disks, its objects' convex parts and the retarget options (``force`` and the ``quality``
flags, which are scored afresh every run, aside), so a re-import, an updated asset, redone convex parts or a changed
setting redoes it.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import NamedTuple

import numpy as np

import eden as en
from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.ik import attempt, best_run, check_length
from eden.extensions.retargeting.flashdexretarget.quality import flags


#: Row key (never written to report.csv): the clip failed after its IK result was saved, which stays.
_KEEP = "_keep_result"


class _Failure(NamedTuple):
    """An IK attempt that raised: the exception's repr (report.csv) and its traceback (the log)."""

    error: str
    traceback: str


def cache_key(
    clip_path: Path, disks: np.ndarray | None, robot: str, robot_sha1: str, convex_sha1: dict[str, str], cfg
) -> dict:
    """What a retargeted clip was made from (cfg: retarget); compared field by field against a saved clip's.

    ``robot_sha1`` is :func:`~eden.extensions.retargeting.flashdexretarget.scene.robot_sha1` of the clip's hands and
    ``convex_sha1`` maps each of its objects to
    :func:`~eden.extensions.retargeting.flashdexretarget.convex.parts_sha1`.
    """
    return {
        "robot": robot,
        "robot_sha1": robot_sha1,
        "clip_sha1": hashlib.sha1(Path(clip_path).read_bytes()).hexdigest(),
        "support_disks_sha1": None
        if disks is None
        else hashlib.sha1(np.ascontiguousarray(disks, np.float64)).hexdigest(),
        "convex_sha1": dict(sorted(convex_sha1.items())),
        "retarget": cfg.model_dump(mode="json", exclude={"force", "quality"}),
    }


def saved_key(path: Path) -> dict | None:
    """The ``cache_key`` a retargeted clip was saved with; None when it has none."""
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["cache_key"])) if "cache_key" in z.files else None


def key_mismatch(saved: dict | None, key: dict) -> list[str]:
    """The fields of ``key`` a saved clip's key differs in (``['cache_key']`` when it has none)."""
    if saved is None:
        return ["cache_key"]
    return [k for k in key if saved.get(k) != key[k]]


def _crop(clip: dict, n: int) -> dict:
    """Drop the first n frames of every per-frame array."""
    T = C.num_frames(clip)
    out = {}
    for k, v in clip.items():
        per_frame = isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == T
        out[k] = v[n:] if per_frame and not k.endswith(("support_disks", "_joint_names")) else v
    return out


def _load(path: str, disks) -> dict:
    clip = C.load_clip(path)
    if disks is not None:
        clip["support_disks"] = disks
    return clip


def _needs_ik(key: dict, out: Path, cfg) -> bool:
    """No saved result made from exactly this (``cache_key``), or cfg.force."""
    return cfg.force or not out.exists() or bool(key_mismatch(saved_key(out), key))


def _scores(clip: dict, run: tuple, robot: str, pool: Path, cfg, fps: float) -> dict:
    """The quality flags (report.csv's) of one IK run, smoothed and cropped as it would be saved."""
    res = best_run([run], cfg)
    clip = _crop(clip, res["crop"])
    clip["qpos"] = res["qpos"]
    return flags(clip, robot, pool, cfg, fps)


def _attempt(path: str, disks, pool: str, robot: str, cfg, fps: float, k: int):
    t0 = time.time()
    try:
        clip = _load(path, disks)
        run = attempt(clip, robot, Path(pool), cfg, fps, Path(path).stem, k)
        scores = _scores(clip, run, robot, Path(pool), cfg, fps)
        # (qpos, errs, log, penetration: best_run's 4th element, the flags: the report row of the kept run)
        return (*run, scores["hand_obj_pen_max_m"], scores), time.time() - t0
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the batch; the row and the log carry it
        return _Failure(repr(e), traceback.format_exc()), time.time() - t0


def _finish(
    path: str, disks, pool: str, out_dir: str, robot: str, cfg, fps: float, runs, seconds: float, key: dict
) -> dict:
    """Save the best IK run, with the quality flags its attempt scored on the same qpos; a cached clip (runs None) is
    replayed for them.

    A replay that raises leaves the clip out of this run (an ``error``) but keeps its saved result, which is this
    run's IK: the next run rescores it rather than redoing the IK.
    """
    path, out = Path(path), Path(out_dir) / Path(path).name
    row: dict = {"clip": path.stem}
    scores = None
    t0 = time.time()
    try:
        if runs is None:
            clip, row["source"] = C.load_clip(out), "cached"
        else:
            clip = _load(str(path), disks)
            res = best_run(runs, cfg)
            scores = runs[res["run"]][4]
            clip = _crop(clip, res["crop"])
            clip["qpos"] = res["qpos"]
            clip["ik_tip_err_mean_m"] = np.float64(res["tip_error_mean_m"])
            clip["ik_restarts"] = np.int64(len(res["log"]) - 1)
            clip["cache_key"] = np.array(json.dumps(key, sort_keys=True))
            row["source"] = "ik"
            C.save_clip(out, clip)
        if "ik_restarts" in clip:
            row.update(ik_tip_err_mean_m=float(clip["ik_tip_err_mean_m"]), ik_restarts=int(clip["ik_restarts"]))
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the batch; the row and the log carry it
        row["error"] = repr(e)
        row["traceback"] = traceback.format_exc()
    else:
        try:
            row.update(scores if scores is not None else flags(clip, robot, Path(pool), cfg, fps))
        except Exception as e:  # noqa: BLE001 - as above
            row["error"] = f"the quality replay raised {e!r}; the IK result is kept in {out} and rescored next run"
            row["traceback"] = traceback.format_exc()
            row[_KEEP] = True
    row["retarget_seconds"] = round(seconds + time.time() - t0, 1)
    return row


def _failed(out: Path, row: dict, done: int, total: int) -> dict:
    """Log a failed clip with its traceback and drop its saved result: made from other inputs (or redone with
    force), it must not be packed in place of this run's. A result this run's IK made is kept (``row[_KEEP]``).
    """
    if not row.pop(_KEEP, False):
        out.unlink(missing_ok=True)
    tb = row.pop("traceback", "")
    en.logger.warning(f"[retarget] {done}/{total} {row['clip']}: {row['error']}" + (f"\n{tb}" if tb else ""))
    return row


def retarget_clips(
    clips: list[tuple[Path, np.ndarray | None]],
    pool: Path,
    out: Path,
    robot: str,
    cfg,
    fps: float,
    workers: int,
    keys: dict[str, dict],
) -> list[dict]:
    """IK of (clip file, support disks) pairs into out, a cache: a clip whose saved result has its ``keys[stem]``
    (:func:`cache_key`) is not redone unless cfg.force (cfg: retarget). Every IK run of every clip (cfg.attempts per
    clip) is a job of its own, so a long clip takes one run's time. Returns the report rows; retarget_seconds sums a
    clip's runs. A clip that fails has an ``error`` and no file in out.
    """
    out.mkdir(parents=True, exist_ok=True)
    en.logger.info(f"[retarget] {len(clips)} clips to {robot}")
    args = (str(pool), robot, cfg, fps)
    rows, runs, spent, jobs = [], {}, {}, {}

    def finish(p: Path, d, finished, seconds: float):
        return ex.submit(_finish, str(p), d, str(pool), str(out), robot, cfg, fps, finished, seconds, keys[p.stem])

    with C.worker_pool(max(1, workers)) as ex:
        for p, d in clips:
            try:
                check_length(C.num_frames(C.load_clip(p)), cfg.smooth)
            except ValueError as e:
                rows.append(_failed(out / p.name, {"clip": p.stem, "error": str(e)}, len(rows) + 1, len(clips)))
            else:
                if _needs_ik(keys[p.stem], out / p.name, cfg):
                    runs[p], spent[p] = [None] * cfg.attempts, 0.0
                    for k in range(cfg.attempts):
                        jobs[ex.submit(_attempt, str(p), d, *args, k)] = (p, d, k)
                else:
                    jobs[finish(p, d, None, 0.0)] = (p, d, None)
        while jobs:
            done, _ = wait(jobs, return_when=FIRST_COMPLETED)
            for fut in done:
                p, d, k = jobs.pop(fut)
                if k is None:
                    r = fut.result()
                    if r.get("error"):
                        rows.append(_failed(out / p.name, r, len(rows) + 1, len(clips)))
                        continue
                    rows.append(r)
                    msg = f"{r['source']}, tip {r.get('ik_tip_err_mean_m', float('nan')) * 1e3:.1f} mm {r.get('flags') or 'ok'}"
                    en.logger.info(
                        f"[retarget] {len(rows)}/{len(clips)} {r['clip']} ({r['retarget_seconds']} s): {msg}"
                    )
                    continue
                runs[p][k], seconds = fut.result()
                spent[p] += seconds
                if any(r is None for r in runs[p]):
                    continue
                finished = runs.pop(p)
                failures = [(i, r) for i, r in enumerate(finished) if isinstance(r, _Failure)]
                if failures:
                    distinct = list(dict.fromkeys(r.error for _, r in failures))
                    row = {
                        "clip": p.stem,
                        "error": f"{len(failures)}/{len(finished)} IK attempts raised: " + "; ".join(distinct),
                        "traceback": f"attempt {failures[0][0]}:\n{failures[0][1].traceback}",
                        "retarget_seconds": round(spent[p], 1),
                    }
                    rows.append(_failed(out / p.name, row, len(rows) + 1, len(clips)))
                    continue
                jobs[finish(p, d, finished, spent[p])] = (p, d, None)
    errors = sum(1 for r in rows if r.get("error"))
    flagged = sum(1 for r in rows if r.get("flags"))
    en.logger.info(f"[retarget] {len(rows) - errors} clips in {out} ({errors} errors, {flagged} flagged)")
    return rows
