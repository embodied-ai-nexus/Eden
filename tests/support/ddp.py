"""Shared helpers for spawning ``torch.distributed`` subprocess fixtures in tests.

DDP-touching code (the curriculum all-reduce, the dataset rank-0 prepare/barrier,
upcoming per-motion metric reductions) needs subprocess-level coverage to catch
regressions like rank-0 self-deadlocks or shape-mismatched all-reduces. Each
callsite re-deriving the env-var / process-group / output-channel boilerplate
duplicates ~30 lines per test and hides the failure modes that boilerplate is
trying to prevent.

Usage
-----

Define a top-level worker that takes ``(rank, world_size, out_dir, *args)``::

    def _my_worker(rank, world_size, out_dir, payload):
        import torch
        import torch.distributed as dist

        x = torch.tensor([float(rank)])
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.save({"x": x, "payload": payload}, f"{out_dir}/rank_{rank}.pt")

Then in the test::

    from tests.support.ddp import ddp_spawn

    def test_my_thing():
        with ddp_spawn(_my_worker, "hello", world_size=2) as out_dir:
            r0 = torch.load(out_dir / "rank_0.pt", weights_only=False)
            r1 = torch.load(out_dir / "rank_1.pt", weights_only=False)
        assert r0["x"].item() == r1["x"].item() == 1.0  # 0 + 1

The helper sets ``MASTER_ADDR`` / ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``,
initializes the process group via ``file://`` init (no port races), runs the
target in a try/finally that always tears the group down, and gives the test a
shared output directory for per-rank artifacts. Both the init file and the
output directory are cleaned up when the context manager exits.

Skip semantics
--------------

If ``torch.multiprocessing.spawn`` is unavailable (Windows, certain CI images),
``ddp_spawn`` raises ``pytest.skip``. Tests using it therefore degrade
gracefully on platforms that can't run gloo/file:// init.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest


def _ddp_subprocess_entry(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    backend: str,
    target: Callable[..., None],
    user_args: tuple,
) -> None:
    """Top-level entry point ``torch.multiprocessing.spawn`` invokes per rank.

    Sets the rank-local env vars, initializes the process group, and calls the
    user's ``target(rank, world_size, out_dir, *user_args)``. The process group
    is destroyed in a ``finally`` so a failing target never leaks NCCL/gloo
    state into a subsequent test.

    Defined at module top level (not inside ``ddp_spawn``) so the spawn pickler
    can find it.
    """
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # ``LOCAL_RANK`` mirrors ``RANK`` for single-node ``mp.spawn`` runs: every
    # subprocess sees its own rank, so Eden's ``CUDA_VISIBLE_DEVICES`` /
    # per-rank Quadrants cache-path isolation in ``eden/__init__.py`` keys off
    # a unique value per process. Hard-coding ``"0"`` would silently make
    # every rank fight over ``cuda:0`` and reuse the same cache lock.
    os.environ["LOCAL_RANK"] = str(rank)
    # ``MASTER_PORT`` intentionally unset — we use ``file://`` rendezvous
    # (no TCP), and ``eden.utils.distributed.setup()`` (the only place in
    # Eden that would consume it) is a no-op once the process group is
    # initialized below.

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        target(rank, world_size, out_dir, *user_args)
    finally:
        dist.destroy_process_group()


@contextmanager
def ddp_spawn(
    target: Callable[..., None],
    *args: Any,
    world_size: int = 2,
    backend: str = "gloo",
) -> Iterator[Path]:
    """Spawn ``world_size`` subprocesses each running ``target``.

    Parameters
    ----------
    target
        A picklable, top-level callable with signature
        ``(rank: int, world_size: int, out_dir: str, *args) -> None``. Inner
        functions and lambdas are not picklable by ``spawn`` and will fail.
    args
        Forwarded verbatim to ``target`` after the helper-supplied
        ``rank, world_size, out_dir``.
    world_size
        Number of ranks to spawn. Default 2 (smallest non-trivial DDP setup).
    backend
        ``torch.distributed`` backend. Default ``"gloo"`` so tests run on
        CPU-only CI. Pass ``"nccl"`` only for GPU-resident tests; the helper
        does not gate on CUDA availability.

    Yields
    ------
    Path
        A directory where each rank can write its artifacts. By convention,
        rank ``N`` writes ``rank_<N>.pt`` and the parent test reads them back
        **inside** the ``with`` block. The directory is cleaned up on context
        exit; the parent should not rely on its contents after the block ends.

    Skips
    -----
    On Windows (gloo file:// init is unreliable) or when
    ``torch.multiprocessing.spawn`` is unavailable in the runtime, the test is
    skipped via ``pytest.skip``.

    Notes
    -----
    Workers should ``import eden`` (and any module that triggers eden import)
    *inside* the worker function, not at module top level. ``LOCAL_RANK`` /
    ``CUDA_VISIBLE_DEVICES`` are set on this subprocess by
    ``_ddp_subprocess_entry`` immediately before ``init_process_group``;
    importing eden before that env-var setup runs will see stale values and
    skip the per-rank GPU/cache isolation in ``eden/__init__.py``.
    """
    if sys.platform == "win32":
        pytest.skip("gloo file:// init unreliable on Windows")

    import torch.multiprocessing as mp

    try:
        mp.get_context("spawn")
    except (ImportError, RuntimeError):
        pytest.skip("torch.multiprocessing.spawn unavailable")

    if backend == "nccl":
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            pytest.skip(
                f"backend='nccl' needs >= {world_size} CUDA devices, "
                f"found {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
            )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        init_file = td_path / "ddp_init"
        out_dir = td_path / "out"
        out_dir.mkdir()

        mp.spawn(
            _ddp_subprocess_entry,
            args=(world_size, str(init_file), str(out_dir), backend, target, args),
            nprocs=world_size,
            join=True,
        )
        yield out_dir
