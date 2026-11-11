"""``tests.support.ddp.ddp_spawn``, the harness itself: per-rank env vars, the process group, argument
forwarding and a failing worker. A failure here means every DDP test is broken before it starts.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.support.ddp import ddp_spawn


def _all_reduce_worker(rank: int, world_size: int, out_dir: str) -> None:
    import torch.distributed as dist

    x = torch.tensor([float(rank)])
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    env = {name: os.environ.get(name) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR")}
    torch.save({"world_size": world_size, "reduced": x.item(), "env": env}, os.path.join(out_dir, f"rank_{rank}.pt"))


def _raising_worker(rank: int, world_size: int, out_dir: str) -> None:
    raise RuntimeError(f"intentional failure on rank {rank}")


def _arg_forwarding_worker(rank: int, world_size: int, out_dir: str, payload: str, multiplier: int) -> None:
    torch.save({"payload": payload, "product": rank * multiplier}, os.path.join(out_dir, f"rank_{rank}.pt"))


def _results(out_dir, world_size: int) -> list[dict]:
    return [torch.load(out_dir / f"rank_{rank}.pt", weights_only=False) for rank in range(world_size)]


@pytest.mark.parametrize("world_size", [1, 2])
def test_each_rank_sees_the_group_and_its_own_env_vars(world_size):
    """``LOCAL_RANK`` must be the rank: ``eden/__init__.py`` keys the per-rank GPU and cache isolation on it."""
    with ddp_spawn(_all_reduce_worker, world_size=world_size) as out_dir:
        results = _results(out_dir, world_size)

    for rank, result in enumerate(results):
        assert result["reduced"] == sum(range(world_size))
        assert result["world_size"] == world_size
        assert result["env"] == {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
        }


def test_arguments_after_the_target_are_forwarded_verbatim():
    with ddp_spawn(_arg_forwarding_worker, "hello", 7, world_size=2) as out_dir:
        results = _results(out_dir, 2)

    assert [(r["payload"], r["product"]) for r in results] == [("hello", 0), ("hello", 7)]


def test_a_failing_worker_surfaces_and_does_not_poison_the_next_spawn():
    with pytest.raises(Exception, match="intentional failure"), ddp_spawn(_raising_worker, world_size=2):
        pass

    with ddp_spawn(_all_reduce_worker, world_size=2) as out_dir:
        assert [r["reduced"] for r in _results(out_dir, 2)] == [1.0, 1.0]
