"""``eden.envs.wrappers.core``: what every env wrapper shares, through the rsl_rl and rl_games wrappers.

The gymnasium, SB3 and skrl wrappers reach the same core through ``VecEnvMixin``, whose tests
(``test_vec_env_mixin.py``) cover it on that side.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from eden.envs.wrappers.core import observation_space, time_outs
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from tests.envs.wrappers.conftest import gym_pendulum_config, rsl_rl_runner_options, unbuildable_gym_pendulum_config


def _rl_games_wrapper():
    pytest.importorskip("rl_games", reason="the rl-games extra is not installed")
    from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper

    return RlGamesVecEnvWrapper


def _runner_options(wrapper_cls, **overrides):
    if wrapper_cls is RslRlVecEnvWrapper:
        return rsl_rl_runner_options(**overrides)
    from eden.options.learning.rl_games import RlGamesPpoRunnerOptions

    return RlGamesPpoRunnerOptions(device="cpu", **overrides)


@pytest.fixture(params=["rsl_rl", "rl_games"])
def runner_wrapper_cls(request):
    """Each wrapper ``eden train`` drives."""
    return RslRlVecEnvWrapper if request.param == "rsl_rl" else _rl_games_wrapper()


def _pendulum_config(pendulum_file, wrapper_cls):
    config = gym_pendulum_config(pendulum_file, num_envs=1)
    config.runner_options = _runner_options(wrapper_cls)
    return config


def test_observation_space_keeps_each_terms_dtype_and_layout():
    obs = {
        "policy": torch.zeros(4, 3),
        "contacts": {"touching": torch.zeros(4, 2, dtype=torch.bool), "count": torch.zeros(4, 1, dtype=torch.int32)},
    }

    space = observation_space(obs)

    assert space["policy"] == gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    assert space["contacts"]["touching"] == gym.spaces.Box(0, 1, shape=(2,), dtype=np.bool_)
    count = space["contacts"]["count"]
    assert count.dtype == np.int32 and count.low.item() == np.iinfo(np.int32).min and count.shape == (1,)


def test_time_outs_leave_out_an_env_that_also_terminated():
    terminated = torch.tensor([False, True, True, False])
    truncated = torch.tensor([True, True, False, False])

    assert time_outs(terminated, truncated).tolist() == [True, False, False, False]


@pytest.mark.requires_genesis
def test_close_destroys_the_env_from_config_built_but_not_one_passed_in(pendulum_file, runner_wrapper_cls):
    config = _pendulum_config(pendulum_file, runner_wrapper_cls)
    owner = runner_wrapper_cls.from_config(config)
    env = owner.unwrapped

    # A wrapper around an env it did not build leaves the scene to its builder.
    runner_wrapper_cls(env, options=config.runner_options).close()
    assert env.scene.is_built

    owner.close()
    assert not env.scene.is_built
    owner.close()  # a second close does nothing


@pytest.mark.requires_genesis
def test_from_config_destroys_the_env_when_the_wrapper_fails_to_construct(pendulum_file, runner_wrapper_cls):
    built = []

    class FailingWrapper(runner_wrapper_cls):
        def __init__(self, env, options):
            built.append(env)
            raise RuntimeError("the wrapper failed")

    with pytest.raises(RuntimeError, match=r"the wrapper failed"):
        FailingWrapper.from_config(_pendulum_config(pendulum_file, runner_wrapper_cls))
    (env,) = built
    assert not env.scene.is_built


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_a_bad_clip_actions_assigned_after_validation_is_refused_before_the_build(tmp_path, runner_wrapper_cls, bound):
    """The options refuse it at construction, but not when assigned afterwards: the wrapper checks again.

    The config's pendulum file is missing, so anything but this error would have come from the build.
    """
    config = unbuildable_gym_pendulum_config(tmp_path)
    config.runner_options = _runner_options(runner_wrapper_cls)
    config.runner_options.clip_actions = bound

    with pytest.raises(ValueError, match=r"runner_options\.clip_actions=.* must be a finite positive bound"):
        runner_wrapper_cls.from_config(config)
