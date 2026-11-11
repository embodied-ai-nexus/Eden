"""Tests for ``eden.utils.random``: seed validation and the per-rank run seed."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from eden.utils.random import apply_run_seed, validate_seed


def _fingerprint() -> tuple[float, float, float]:
    return torch.rand(1).item(), float(np.random.rand()), random.random()


def _fingerprint_of(seed: int) -> tuple[float, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return _fingerprint()


@pytest.mark.parametrize("seed", [0, 123, 7100])
def test_a_non_negative_seed_passes_validation(seed):
    assert validate_seed(seed) == seed


def test_the_old_draw_one_sentinel_is_an_error():
    """``-1`` drew off a stream ``gs.init`` had just seeded to a constant: the same value every run."""
    for check in (validate_seed, apply_run_seed):
        with pytest.raises(ValueError, match="seed must be >= 0"):
            check(-1)


def test_apply_run_seed_seeds_torch_numpy_and_random_and_reports_the_seed():
    assert apply_run_seed(7100) == 7100
    assert _fingerprint() == _fingerprint_of(7100)


def test_each_ddp_rank_offsets_the_base_by_its_rank_and_reapplying_does_not_compound(monkeypatch):
    """Without the offset every rank replays rank 0's rollout; compounded, rank ``r`` gets ``base + 2r``."""
    monkeypatch.setenv("RANK", "3")

    assert [apply_run_seed(100) for _ in range(3)] == [103, 103, 103]
    assert _fingerprint() == _fingerprint_of(103)
