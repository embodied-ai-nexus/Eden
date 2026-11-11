"""Tests for ``RslRlFlashSacRunnerOptions``: the run's length is set one way only."""

from __future__ import annotations

import pytest

from eden.learning.wrappers.rsl_rl.flashsac.options import RslRlFlashSacRunnerOptions


def test_iterations_and_env_steps_together_are_refused_not_one_silently_ignored():
    with pytest.raises(ValueError, match=r"exactly one of max_iterations \(1000\) and num_env_steps \(1000000\)"):
        RslRlFlashSacRunnerOptions(max_iterations=1000, num_env_steps=1_000_000)


def test_a_run_with_no_length_is_refused():
    with pytest.raises(ValueError, match="exactly one of max_iterations"):
        RslRlFlashSacRunnerOptions(max_iterations=None)


def test_env_steps_become_iterations_for_the_env_count():
    sized = RslRlFlashSacRunnerOptions(max_iterations=None, num_env_steps=1_000_000, num_steps_per_env=2)
    assert sized.for_num_envs(1000, world_size=1).max_iterations == 500
    assert sized.for_num_envs(10_000_000, world_size=1).max_iterations == 1  # never a zero-length run
    iterations = RslRlFlashSacRunnerOptions(max_iterations=1000)
    assert iterations.for_num_envs(4096, world_size=1).max_iterations == 1000


def test_env_steps_are_split_across_ranks_not_repeated_on_each():
    """Under ``torchrun`` every rank runs ``max_iterations`` of its own envs: two GPUs halve the iterations."""
    sized = RslRlFlashSacRunnerOptions(max_iterations=None, num_env_steps=1_000_000, num_steps_per_env=2)
    assert sized.for_num_envs(1000, world_size=2).max_iterations == 250
    iterations = RslRlFlashSacRunnerOptions(max_iterations=1000)
    assert iterations.for_num_envs(4096, world_size=2).max_iterations == 1000  # an iteration count is per rank
