"""Tests for ``eden.extensions.retargeting.flashdexretarget.clips``: the worker processes of the CPU steps."""

from __future__ import annotations

import os

from eden.extensions.retargeting.flashdexretarget import clips as C


def test_a_worker_runs_one_blas_thread_and_the_callers_environment_is_left_as_it_was(monkeypatch):
    # The server the workers fork from keeps the thread counts it started with: the caller's, else 1.
    expected = {v: os.environ.get(v, "1") for v in C.THREAD_VARS}
    for v in C.THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    with C.worker_pool(1) as ex:
        assert {v: ex.submit(os.getenv, v).result() for v in C.THREAD_VARS} == expected
    assert not any(v in os.environ for v in C.THREAD_VARS)


def test_a_thread_count_the_caller_set_is_kept(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with C.single_threaded_workers():
        assert os.environ["OMP_NUM_THREADS"] == "4" and os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "4" and "MKL_NUM_THREADS" not in os.environ
