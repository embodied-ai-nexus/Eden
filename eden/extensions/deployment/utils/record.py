"""Crash-safe streaming recorder for a deployment's per-step history.

Holding every step in memory and writing one ``.npz`` from ``close()`` is the wrong way
round on a robot: the runs worth recording are the ones
that end badly -- a fall, an e-stop, an operator who hits Ctrl-C twice -- and those are
exactly the runs where ``close()`` may not finish. A recording you only get when the run
went well is not a diagnostic.

So the history is streamed to disk in fixed-size chunks as it is produced. Three properties
matter, in this order:

* **It survives a kill.** Each chunk is written to a temporary name and then ``os.replace``-d
  into place, so a process killed mid-write leaves whole chunks or nothing, never a corrupt
  one. At most :attr:`StreamingRecorder.flush_steps` steps are ever at risk.
* **It does not stall the control loop.** :meth:`StreamingRecorder.append` is O(1) and never
  touches the disk; a background thread does the writing. Chunking (rather than rewriting a
  growing file) keeps both the memory and the per-write cost constant, so a ten-minute run
  costs the same per step as a ten-second one.
* **Every artifact is a plain ``.npz``.** A clean run merges the chunks into the single file
  ``--debug_record_path`` names. A killed run
  leaves the chunk directory, and :func:`load_deploy_record` reads either without the caller
  needing to know which happened.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
from typing import Any

import numpy as np

import eden as en

#: Sentinel pushed onto the write queue to stop the writer thread.
_STOP = object()


def _parts_dir(path: str) -> str:
    """Return the chunk directory that belongs to a final recording path."""
    stem = path.removesuffix(".npz")
    return f"{stem}.parts"


class StreamingRecorder:
    """Accumulate per-step rows and stream them to disk in crash-safe chunks.

    Parameters
    ----------
    path: str
        Final ``.npz`` the chunks are merged into by :meth:`close`.
    dofs_name: list[str]
        DOF order the recorded columns are in. Written into every chunk so a chunk
        recovered from a killed run is self-describing.
    flush_steps: int
        Steps per chunk, i.e. the most that can be lost to a kill. At a 50 Hz control rate
        the default 100 is two seconds.
    """

    def __init__(self, path: str, dofs_name: list[str], flush_steps: int = 100) -> None:
        if flush_steps < 1:
            raise ValueError(f"flush_steps must be >= 1, got {flush_steps}")
        self.path = path
        self.dofs_name = list(dofs_name)
        self.flush_steps = int(flush_steps)
        self._dir = _parts_dir(path)
        self._next = 0
        self._written = 0
        self._closed = False
        # Bounded so a disk slower than the control loop applies back-pressure instead of
        # growing the queue without limit -- on a long run that is the difference between a
        # late chunk and an out-of-memory kill. 64 chunks is >2 min of slack at the default.
        self._q: queue.Queue[Any] = queue.Queue(maxsize=64)
        os.makedirs(self._dir, exist_ok=True)
        self._thread = threading.Thread(target=self._writer_loop, name="deploy-recorder", daemon=True)
        self._thread.start()

    # -- producer side (control thread) -----------------------------------

    def append(self, rows: dict[str, list]) -> bool:
        """Hand off ``rows`` as a chunk if it has reached ``flush_steps``.

        Returns ``True`` when the chunk was handed to the writer, so the caller can start a
        fresh buffer. The caller keeps ownership of the dict it passed in and must not reuse
        it after a ``True`` -- the writer thread reads it.

        Parameters
        ----------
        rows: dict[str, list]
            The pending chunk: one list of per-step values per recorded channel.

        Returns
        -------
        bool
            Whether the chunk was flushed.
        """
        if self._closed or not rows or len(rows.get("stamp", ())) < self.flush_steps:
            return False
        self._enqueue(rows)
        return True

    def _enqueue(self, rows: dict[str, list]) -> None:
        self._q.put((self._next, rows))
        self._next += 1

    # -- consumer side (writer thread) ------------------------------------

    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    return
                index, rows = item
                self._write_chunk(index, rows)
            except Exception as exc:  # noqa: BLE001 — a recorder must never take the run down
                en.logger.warning(f"[deploy] debug_record: dropped a chunk ({exc}).")
            finally:
                self._q.task_done()

    def _write_chunk(self, index: int, rows: dict[str, list]) -> None:
        """Write one chunk atomically: a temp file, then ``os.replace`` into place."""
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        arrays["dofs_name"] = np.asarray(self.dofs_name)
        final = os.path.join(self._dir, f"{index:05d}.npz")
        tmp = f"{final}.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        self._written += len(arrays["stamp"])

    # -- teardown ---------------------------------------------------------

    def close(self, rows: dict[str, list] | None = None) -> int:
        """Flush a final partial chunk, stop the writer, and merge everything into ``path``.

        Idempotent. Returns the number of steps in the merged file (0 if nothing was
        recorded, in which case no file and no chunk directory are left behind).

        Parameters
        ----------
        rows: dict[str, list] | None
            The last, partial chunk still in the caller's buffer.

        Returns
        -------
        int
            Steps written to ``path``.
        """
        if self._closed:
            return self._written
        if rows and rows.get("stamp"):
            self._enqueue(rows)
        self._closed = True
        self._q.put(_STOP)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            en.logger.warning("[deploy] debug_record: the writer thread did not finish; chunks are kept as-is.")
            return self._written
        return self._merge()

    def _merge(self) -> int:
        """Concatenate the chunk files into ``path`` and remove the chunk directory."""
        merged, steps = _merge_parts(self._dir)
        if not steps:
            shutil.rmtree(self._dir, ignore_errors=True)
            return 0
        np.savez(self.path, **merged)
        shutil.rmtree(self._dir, ignore_errors=True)
        return steps


def _merge_parts(parts_dir: str) -> tuple[dict[str, np.ndarray], int]:
    """Concatenate every chunk in ``parts_dir`` in index order.

    Channels are merged on the **intersection** of the chunks' keys. A backend is expected to
    populate the same ``state.extra`` keys every step, but if one did not, dropping the
    ragged channel with a warning is the only option that neither fabricates rows nor throws
    the rest of the recording away.

    Parameters
    ----------
    parts_dir: str
        Directory of ``NNNNN.npz`` chunks.

    Returns
    -------
    tuple[dict[str, np.ndarray], int]
        The merged channels (plus ``dofs_name``) and the total step count.
    """
    if not os.path.isdir(parts_dir):
        return {}, 0
    files = sorted(f for f in os.listdir(parts_dir) if f.endswith(".npz"))
    chunks = []
    for name in files:
        try:
            with np.load(os.path.join(parts_dir, name), allow_pickle=False) as z:
                chunks.append({k: z[k] for k in z.files})
        except Exception as exc:  # noqa: BLE001 — a truncated tail chunk must not lose the rest
            en.logger.warning(f"[deploy] debug_record: skipping unreadable chunk {name} ({exc}).")
    if not chunks:
        return {}, 0

    dofs_name = chunks[0].get("dofs_name")
    keys = set.intersection(*(set(c) for c in chunks)) - {"dofs_name"}
    dropped = sorted((set.union(*(set(c) for c in chunks)) - {"dofs_name"}) - keys)
    if dropped:
        en.logger.warning(f"[deploy] debug_record: channels {dropped} are missing from some chunks; dropped.")
    merged = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in sorted(keys)}
    if dofs_name is not None:
        merged["dofs_name"] = dofs_name
    return merged, len(merged["stamp"]) if "stamp" in merged else 0


def load_deploy_record(path: str) -> dict[str, np.ndarray]:
    """Load a deployment recording, whether the run closed cleanly or was killed.

    Reads ``path`` when the merged file exists; otherwise merges whatever chunks survive in
    the sibling ``<stem>.parts`` directory, which is what a killed run leaves behind. Use
    this instead of ``np.load`` in analysis code so both cases read the same.

    Parameters
    ----------
    path: str
        The ``--debug_record_path`` the run was given.

    Returns
    -------
    dict[str, np.ndarray]
        One array per recorded channel, plus ``dofs_name``.

    Raises
    ------
    FileNotFoundError
        When neither the merged file nor any readable chunk exists.
    """
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    merged, steps = _merge_parts(_parts_dir(path))
    if not steps:
        raise FileNotFoundError(f"No deployment recording at {path}, and no readable chunks in {_parts_dir(path)}.")
    en.logger.info(f"[deploy] debug_record: recovered {steps} steps from {_parts_dir(path)} (the run did not close).")
    return merged
