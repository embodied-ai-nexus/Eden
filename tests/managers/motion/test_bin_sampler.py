"""Tests for ``eden.managers.motion.bin_sampler``: the adaptive start-time bin curriculum.

Failures are recorded into a real term's ``AdaptiveBinSampler`` at chosen death times; a reset
then re-derives the bin weights the next start times are drawn from.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.motion_manager import sample_start_times
from tests.support.motion import FRAMERATE, make_manager, pin_clips, term_options, write_library

BIN_S = 0.1  # five frames per bin


def _sampler(env, tmp_path, bins_per_motion, **term_kwargs):
    """A term over clips of the given bin counts, with ``adaptive_alpha=1`` unless overridden."""
    # Two frames past the last bin edge, so the bin count never rides a float boundary.
    root = write_library(tmp_path, [int(b * BIN_S * FRAMERATE) + 2 for b in bins_per_motion])
    term_kwargs = {"adaptive_sampling": True, "adaptive_alpha": 1.0, "adaptive_uniform_ratio": 0.0} | term_kwargs
    manager = make_manager(env, terms={"default": term_options(root, motion_bin_length_s=BIN_S, **term_kwargs)})
    term = manager.get_term("default")
    return manager, term, term._bin_sampler


def _record_failures(term, sampler, failures: dict[tuple[int, int], int]) -> None:
    """Record ``count`` terminated episodes dying in bin ``b`` of motion ``m`` for each ``(m, b): count``."""
    for (motion, bin_idx), count in failures.items():
        pin_clips(term, [motion] * term.num_assigned_envs)
        died_at = torch.full((count,), (bin_idx + 0.5) * BIN_S)
        sampler.record_bin_outcomes(torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.bool), died_at)


def _refresh(manager) -> None:
    # A reset draws new start times, which first re-derives the bin weights from the failure EMA.
    manager.reset(torch.arange(manager.num_envs))


def _weights(sampler, motion: int) -> torch.Tensor:
    return sampler.weights[motion].cpu()


def test_each_motion_gets_its_clip_length_in_bins_and_at_least_one(mimic_env, tmp_path):
    root = write_library(tmp_path, [3, 7, 52])  # 0.06 s, 0.14 s and 1.04 s
    term = make_manager(mimic_env, terms={"default": term_options(root, motion_bin_length_s=BIN_S)}).get_term("default")

    assert term._bin_sampler.weights.batch_sizes.tolist() == [1, 1, 10]
    torch.testing.assert_close(_weights(term._bin_sampler, 2), torch.full((10,), 0.1))


def test_failures_count_in_the_bin_the_episode_died_in_not_the_one_it_started_in(mimic_env, tmp_path):
    manager, term, sampler = _sampler(mimic_env, tmp_path, [10, 8])
    # Two envs start in bin 2 of motion 0; only the one that terminated counts, in its death bin.
    pin_clips(term, [0, 0, 1, 1], start_s=[0.24, 0.29, 0.15, 0.32])
    sampler.record_bin_outcomes(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([True, False, True, True]),
        torch.tensor([0.5, 0.01, 0.0, 0.0]),
    )
    sampler.blend_failure_ema()

    _refresh(manager)

    assert _weights(sampler, 0).tolist() == [0.0] * 7 + [1.0] + [0.0] * 2
    # Motion 1: one death at 0.15 s (bin 1), one at 0.32 s (bin 3).
    assert _weights(sampler, 1).tolist() == [0.0, 0.5, 0.0, 0.5] + [0.0] * 4


@pytest.mark.parametrize(
    "bins, failures, uniform_ratio, expected",
    [
        ([2], {(0, 0): 1, (0, 1): 9}, 0.0, [[0.1, 0.9]]),
        # The floor uniform_ratio / num_bins is added before the per-motion L1 normalisation.
        ([4], {(0, 2): 5}, 0.1, [[0.025 / 5.1, 0.025 / 5.1, 5.025 / 5.1, 0.025 / 5.1]]),
        ([5], {(0, b): 5 for b in range(5)}, 0.0, [[0.2] * 5]),
        ([10], {}, 0.1, [[0.1] * 10]),
        # Motions normalise independently; one with no failures (and no floor) falls back to uniform.
        ([5, 8], {(1, 2): 1}, 0.0, [[0.2] * 5, [0.0, 0.0, 1.0] + [0.0] * 5]),
    ],
    ids=["count-proportional", "floor-before-normalise", "equal-failures", "no-failures", "zero-sum-motion"],
)
def test_bin_weights_are_the_normalised_failure_counts(mimic_env, tmp_path, bins, failures, uniform_ratio, expected):
    manager, term, sampler = _sampler(mimic_env, tmp_path, bins, adaptive_uniform_ratio=uniform_ratio)
    _record_failures(term, sampler, failures)
    sampler.blend_failure_ema()

    _refresh(manager)

    for motion, row in enumerate(expected):
        torch.testing.assert_close(_weights(sampler, motion), torch.tensor(row), atol=1e-6, rtol=0)


def test_blend_is_an_ema_of_the_failures_since_the_last_blend(mimic_env, tmp_path):
    manager, term, sampler = _sampler(mimic_env, tmp_path, [3], adaptive_alpha=0.5)
    _record_failures(term, sampler, {(0, 0): 2})
    sampler.blend_failure_ema()  # EMA [1, 0, 0]
    _record_failures(term, sampler, {(0, 2): 2})
    sampler.blend_failure_ema()  # EMA [0.5, 0, 1]: the first batch is not counted twice

    _refresh(manager)

    torch.testing.assert_close(_weights(sampler, 0), torch.tensor([1 / 3, 0.0, 2 / 3]))


def test_without_adaptive_sampling_failures_leave_the_bins_uniform(mimic_env, tmp_path):
    manager, term, sampler = _sampler(mimic_env, tmp_path, [10], adaptive_sampling=False)
    _record_failures(term, sampler, {(0, 3): 100})
    sampler.blend_failure_ema()

    _refresh(manager)

    torch.testing.assert_close(_weights(sampler, 0), torch.full((10,), 0.1))


def test_kernel_smoothing_spreads_a_spike_over_kernel_size_bins(mimic_env, tmp_path):
    manager, term, sampler = _sampler(mimic_env, tmp_path, [20, 1], adaptive_kernel_size=3, adaptive_lambda=0.5)
    _record_failures(term, sampler, {(0, 10): 1, (1, 0): 1})
    sampler.blend_failure_ema()

    _refresh(manager)

    probs = _weights(sampler, 0)
    assert (probs > 0).sum() == 3 and probs.sum().item() == pytest.approx(1.0)
    assert probs[:8].sum() == 0 and probs[11:].sum() == 0
    assert _weights(sampler, 1).tolist() == [1.0]  # a single-bin motion survives the padding


def test_kernel_smoothing_moves_starts_onto_the_run_up_to_a_failure(mimic_env, tmp_path):
    manager, term, sampler = _sampler(mimic_env, tmp_path, [20], adaptive_kernel_size=3, adaptive_lambda=0.5)
    _record_failures(term, sampler, {(0, 10): 1})
    sampler.blend_failure_ema()

    _refresh(manager)

    # Kernel [1, 0.5, 0.25]: the death bin itself heaviest, decaying into the bins before it.
    expected = torch.zeros(20)
    expected[[10, 9, 8]] = torch.tensor([1.0, 0.5, 0.25]) / 1.75
    torch.testing.assert_close(_weights(sampler, 0), expected)


def test_kernel_smoothing_pads_each_motion_with_its_own_last_bin(mimic_env, tmp_path):
    # A failure in motion 0's last bin keeps its whole run-up, and the replicated last bin adds the kernel's
    # tail back onto it (BeyondMimic's `replicate` padding): bins 17..19 get [w2, w1 + w2, w0 + w1 + w2].
    manager, term, sampler = _sampler(mimic_env, tmp_path, [20, 6], adaptive_kernel_size=3, adaptive_lambda=0.5)
    _record_failures(term, sampler, {(0, 19): 1, (1, 5): 1})
    sampler.blend_failure_ema()

    _refresh(manager)

    run_up = torch.tensor([0.25, 0.75, 1.75]) / 2.75
    torch.testing.assert_close(_weights(sampler, 0), torch.cat([torch.zeros(17), run_up]))
    # The shorter motion replicates its own last bin, not the batch's zero padding beyond it.
    torch.testing.assert_close(_weights(sampler, 1), torch.cat([torch.zeros(3), run_up]))


def test_kernel_smoothing_keeps_a_failure_free_floor_uniform(mimic_env, tmp_path):
    # Zero padding would thin the floor in the bins next to it, at the start or end of every motion.
    manager, term, sampler = _sampler(
        mimic_env, tmp_path, [20, 6], adaptive_kernel_size=3, adaptive_lambda=0.5, adaptive_uniform_ratio=0.1
    )

    _refresh(manager)

    torch.testing.assert_close(_weights(sampler, 0), torch.full((20,), 1 / 20), atol=1e-6, rtol=0)
    torch.testing.assert_close(_weights(sampler, 1), torch.full((6,), 1 / 6), atol=1e-6, rtol=0)


def test_repeated_failures_move_starts_to_where_episodes_die(mimic_env, tmp_path):
    """Attributing failures to the start bin (or weighting by rate) never moves starts onto the hard
    region, and multi-clip LAFAN1 tracking collapses.
    """
    manager, term, sampler = _sampler(mimic_env, tmp_path, [10], adaptive_alpha=0.5, adaptive_uniform_ratio=0.05)
    starts = torch.tensor([0.03, 0.12, 0.25, 0.25])  # bins 0, 1, 2
    pin_clips(term, [0, 0, 0, 0], start_s=starts)
    for _ in range(20):
        sampler.record_bin_outcomes(torch.arange(4), torch.ones(4, dtype=torch.bool), 0.71 - starts)  # all die in bin 7
        sampler.blend_failure_ema()

    _refresh(manager)

    probs = _weights(sampler, 0)
    assert int(probs.argmax()) == 7
    drawn = sample_start_times(torch.zeros(10_000, dtype=torch.long), sampler.weights, "cpu", BIN_S)
    assert ((drawn >= 0.7) & (drawn < 0.8)).float().mean() > 0.5


def test_failures_spread_evenly_keep_starts_uniform(mimic_env, tmp_path):
    manager, term, sampler = _sampler(mimic_env, tmp_path, [8], adaptive_alpha=0.5, adaptive_uniform_ratio=0.1)
    for _ in range(10):
        for first in (0, 4):
            pin_clips(term, [0] * 4, start_s=(torch.arange(first, first + 4) + 0.5) * BIN_S)
            sampler.record_bin_outcomes(torch.arange(4), torch.ones(4, dtype=torch.bool), torch.zeros(4))
        sampler.blend_failure_ema()

    _refresh(manager)

    torch.testing.assert_close(_weights(sampler, 0), torch.full((8,), 1 / 8), atol=1e-6, rtol=0)
