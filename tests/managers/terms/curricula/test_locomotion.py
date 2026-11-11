"""The locomotion curricula in ``eden/managers/terms/curricula/locomotion.py``: ``AverageEpisodeLengthTracker``.

An EMA of finished episodes' lengths whose window scales with ``num_envs / BASE_NUM_ENVS``, so its time
constant in episodes does not depend on the env count. The tests write the envs' ``episode_length_buf``.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.curricula.locomotion import AverageEpisodeLengthTracker
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


pytestmark = pytest.mark.requires_genesis


@pytest.fixture(scope="module")
def env(box_env):
    """The shared four-env box scene."""
    return box_env


@pytest.fixture(scope="module")
def env_of_two():
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        )
    )


def _tracker(env, num_compute_average_epl: int) -> AverageEpisodeLengthTracker:
    return AverageEpisodeLengthTracker(
        env=env, options=AverageEpisodeLengthTracker.configure(num_compute_average_epl=num_compute_average_epl)
    )


def _finish(env, tracker, length: int, envs_idx) -> float:
    """End the episodes of ``envs_idx`` at ``length`` steps; return the tracker's average after."""
    env.episode_length_buf[:] = length
    try:
        tracker.reset(envs_idx=envs_idx)
    finally:
        env.episode_length_buf[:] = 0
    return tracker.compute()["average_episode_length"]


def test_one_reset_of_every_env_moves_the_average_by_the_same_fraction_at_any_env_count(env, env_of_two):
    """8192 at the 4096-env reference is a window of 8 resets at 4 envs and 4 at 2: both weigh a full reset 0.5."""
    for built in (env, env_of_two):
        tracker = _tracker(built, num_compute_average_epl=8192)
        assert _finish(built, tracker, 100, torch.arange(built.num_envs)) == pytest.approx(50.0)


def test_a_partial_reset_is_weighted_by_its_share_of_the_window(env):
    tracker = _tracker(env, num_compute_average_epl=8192)  # window 8

    assert _finish(env, tracker, 100, torch.tensor([0, 1])) == pytest.approx(25.0)  # 0 + (100 - 0) * 2 / 8
    assert _finish(env, tracker, 300, torch.tensor([2, 3])) == pytest.approx(93.75)  # 25 + (300 - 25) * 2 / 8


def test_the_window_floors_at_one_reset(env):
    """Rounded down, 1 * 4 / 4096 would be a zero window; floored, each reset replaces the average."""
    tracker = _tracker(env, num_compute_average_epl=1)

    assert _finish(env, tracker, 250, torch.tensor([0])) == pytest.approx(250.0)
    assert _finish(env, tracker, 100, torch.tensor([1])) == pytest.approx(100.0)


def test_it_starts_at_zero_and_unprimed_and_only_a_real_reset_primes_it(env):
    """``envs_idx=None`` (a startup-style call) and an empty selection record no episode."""
    tracker = _tracker(env, num_compute_average_epl=1000)
    assert tracker.compute() == {"average_episode_length": 0.0, "is_primed": False}

    _finish(env, tracker, 50, None)
    _finish(env, tracker, 50, torch.tensor([], dtype=torch.long))
    assert tracker.compute() == {"average_episode_length": 0.0, "is_primed": False}

    _finish(env, tracker, 50, torch.arange(env.num_envs))
    assert tracker.compute()["is_primed"] is True


def test_the_state_dict_round_trips_the_average_and_the_primed_flag(env):
    source = _tracker(env, num_compute_average_epl=1000)
    saved = _finish(env, source, 123, torch.arange(env.num_envs))

    restored = _tracker(env, num_compute_average_epl=1000)
    restored.load_state_dict(source.state_dict())

    assert restored.compute() == {"average_episode_length": pytest.approx(saved), "is_primed": True}
    untouched = _tracker(env, num_compute_average_epl=1000)
    untouched.load_state_dict({})  # a missing key leaves the running mean as it is
    assert untouched.compute() == {"average_episode_length": 0.0, "is_primed": False}
