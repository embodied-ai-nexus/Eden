"""Tests for ``eden.utils.sample``: the global Bernoulli gate and uniform range sampling."""

from __future__ import annotations

import pytest
import torch

from eden.utils.sample import apply_probability, sample_uniform, sample_uniform_ranges


@pytest.mark.parametrize("probability, expected", [(1.0, [True, False, True]), (0.0, [False, False, False])])
def test_the_bounds_keep_or_clear_the_mask_and_stay_bool(probability, expected):
    """Bool always: the termination manager OR-combines it with other bool masks."""
    out = apply_probability(torch.tensor([True, False, True]), probability=probability)

    assert out.dtype == torch.bool
    assert out.tolist() == expected


def test_the_gate_is_one_draw_per_call_for_every_env():
    """Either every violating env keeps its True or none does."""
    torch.manual_seed(0)
    mask = torch.tensor([True, True, False, True])

    outcomes = {tuple(apply_probability(mask, probability=0.5).tolist()) for _ in range(50)}

    assert outcomes == {tuple(mask.tolist()), (False,) * 4}


@pytest.mark.parametrize("probability", [float("nan"), -0.1, 1.5])
def test_a_probability_outside_the_unit_interval_raises(probability):
    """A silent no-op would mask a config bug."""
    with pytest.raises(ValueError):
        apply_probability(torch.tensor([True, False]), probability=probability)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_sample_uniform_ranges_draws_what_sample_uniform_draws(device):
    """The ranges form replaced per-axis bound tensors without changing a seeded run's draw."""
    ranges = ((0.0, 1.0), (-2.0, 2.0), (0.5, 0.5))
    torch.manual_seed(0)
    new = sample_uniform_ranges(ranges, 6, device)
    torch.manual_seed(0)
    low = torch.tensor([r[0] for r in ranges], device=device)
    high = torch.tensor([r[1] for r in ranges], device=device)
    assert torch.equal(new, sample_uniform(low, high, (6, 3), device=device))


def test_sample_uniform_ranges_is_sync_free_once_warm(count_cuda_syncs):
    ranges = ((0.0, 1.0), (-2.0, 2.0))
    sample_uniform_ranges(ranges, 6, "cuda")
    assert count_cuda_syncs(lambda: sample_uniform_ranges(ranges, 6, "cuda")) == 0
