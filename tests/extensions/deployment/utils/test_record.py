"""``StreamingRecorder`` / ``load_deploy_record``: a recording that survives a run which never closes."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from eden.extensions.deployment.utils.record import (
    StreamingRecorder,
    _parts_dir,
    load_deploy_record,
)


def _rows(n: int, start: int = 0, extra: dict | None = None) -> dict[str, list]:
    rows = {
        "stamp": [float(start + i) for i in range(n)],
        "q": [np.full(2, start + i, dtype=np.float64) for i in range(n)],
    }
    for k, v in (extra or {}).items():
        rows[k] = [np.asarray(v, dtype=np.float64) for _ in range(n)]
    return rows


def _finished_chunks(rec: StreamingRecorder) -> list[str]:
    """Only the chunks that have been ``os.replace``-d into place, never an in-flight ``.tmp``."""
    return sorted(f for f in os.listdir(rec._dir) if f.endswith(".npz"))


def _wait_for_chunks(rec: StreamingRecorder, count: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(_finished_chunks(rec)) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"only {len(_finished_chunks(rec))} chunk(s) appeared, wanted {count}")


def test_chunks_land_on_disk_before_close(tmp_path):
    """The point of the change: the data is durable *during* the run, not at the end."""
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=10)
    assert rec.append(_rows(10)) is True
    _wait_for_chunks(rec, 1)
    assert _finished_chunks(rec) == ["00000.npz"]
    assert not os.path.exists(path), "nothing should be merged yet"
    rec.close()


def test_append_below_the_threshold_does_not_flush(tmp_path):
    rec = StreamingRecorder(str(tmp_path / "rec.npz"), ["a", "b"], flush_steps=10)
    assert rec.append(_rows(9)) is False
    assert _finished_chunks(rec) == []
    rec.close()


def test_clean_close_merges_into_one_npz_and_removes_the_chunks(tmp_path):
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=10)
    for i in range(3):
        assert rec.append(_rows(10, start=10 * i)) is True
    assert rec.close(_rows(4, start=30)) == 34

    assert not os.path.exists(_parts_dir(path)), "chunk directory should be cleaned up"
    with np.load(path) as z:
        assert z["stamp"].shape == (34,)
        assert z["q"].shape == (34, 2)
        assert list(z["dofs_name"]) == ["a", "b"]
        # order preserved across chunk boundaries and into the partial tail
        assert z["stamp"] == pytest.approx(np.arange(34, dtype=float))


def test_a_run_that_never_closes_is_still_recoverable(tmp_path):
    """The Ctrl-C case: no ``close()``, so no merged file -- the chunks carry the run."""
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=10)
    for i in range(3):
        rec.append(_rows(10, start=10 * i))
    _wait_for_chunks(rec, 3)
    # ... process dies here: no close(), no merged file.
    assert not os.path.exists(path)

    got = load_deploy_record(path)
    assert got["stamp"].shape == (30,)
    assert got["stamp"] == pytest.approx(np.arange(30, dtype=float))
    assert list(got["dofs_name"]) == ["a", "b"]


def test_load_prefers_the_merged_file_and_reads_both_shapes_the_same(tmp_path):
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=10)
    rec.append(_rows(10))
    _wait_for_chunks(rec, 1)
    from_chunks = load_deploy_record(path)
    rec.close()
    from_merged = load_deploy_record(path)
    assert set(from_chunks) == set(from_merged)
    assert from_chunks["stamp"] == pytest.approx(from_merged["stamp"])


def test_no_temp_files_survive_a_chunk_write(tmp_path):
    """Chunks are written to a temp name and ``os.replace``-d, so a kill leaves no partial."""
    rec = StreamingRecorder(str(tmp_path / "rec.npz"), ["a", "b"], flush_steps=5)
    for i in range(4):
        rec.append(_rows(5, start=5 * i))
    _wait_for_chunks(rec, 4)
    assert [f for f in os.listdir(rec._dir) if f.endswith(".tmp")] == []
    rec.close()


def test_recording_nothing_leaves_no_file_and_no_directory(tmp_path):
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=10)
    assert rec.close() == 0
    assert not os.path.exists(path)
    assert not os.path.exists(_parts_dir(path))


def test_close_is_idempotent(tmp_path):
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=10)
    assert rec.close(_rows(3)) == 3
    assert rec.close(_rows(3)) == 3  # second close records nothing more
    with np.load(path) as z:
        assert z["stamp"].shape == (3,)


def test_flush_steps_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="flush_steps"):
        StreamingRecorder(str(tmp_path / "rec.npz"), ["a"], flush_steps=0)


def test_load_raises_when_there_is_nothing_at_all(tmp_path):
    with pytest.raises(FileNotFoundError, match="No deployment recording"):
        load_deploy_record(str(tmp_path / "missing.npz"))


def test_a_channel_missing_from_some_chunks_is_dropped_not_fabricated(tmp_path, eden_caplog):
    """Ragged ``state.extra`` keys must not invent rows, and must not lose the rest."""
    path = str(tmp_path / "rec.npz")
    rec = StreamingRecorder(path, ["a", "b"], flush_steps=5)
    rec.append(_rows(5, start=0, extra={"x_temp": [1.0, 2.0]}))
    rec.append(_rows(5, start=5))  # no x_temp in this one
    steps = rec.close()
    assert steps == 10
    with np.load(path) as z:
        assert "x_temp" not in z.files
        assert z["stamp"].shape == (10,)
    assert "x_temp" in eden_caplog.text
