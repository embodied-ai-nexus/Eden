"""``eden.envs.wrappers.vec_env_mixin``: the core every library wrapper shares, through the gymnasium wrapper.

``GymnasiumVecEnvWrapper`` is the mixin with nothing but gymnasium's interface on top, so it exercises the
mixin as it is. The tests share one built ``gym_pendulum_config`` env; each starts from a ``reset()``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.envs.wrappers.gymnasium_env import GymnasiumVecEnvWrapper
from tests.envs.wrappers.conftest import (
    built_gym_pendulum_env,
    ending,
    gym_pendulum_config,
    unbuildable_gym_pendulum_config,
)

NUM_ENVS = 3


@pytest.fixture(scope="module")
def source_config(pendulum_file):
    """The config the shared wrapper was built from."""
    return gym_pendulum_config(pendulum_file, num_envs=NUM_ENVS)


@pytest.fixture(scope="module")
def wrapper(source_config):
    wrapper = GymnasiumVecEnvWrapper.from_config(source_config, clip_actions=2.0)
    yield wrapper
    wrapper.close()


@pytest.mark.requires_genesis
def test_from_config_records_final_observations_without_touching_the_config(source_config, wrapper):
    assert wrapper.env.env_options.record_final_observations
    assert not source_config.env_options.record_final_observations


@pytest.mark.requires_genesis
def test_a_caller_built_env_needs_final_observations_unless_opted_out_and_survives_close(pendulum_file):
    env = built_gym_pendulum_env(pendulum_file, num_envs=1, record_final_observations=False)
    try:
        with pytest.raises(ValueError, match=r"record_final_observations=False.*final_observations=False"):
            GymnasiumVecEnvWrapper(env)

        wrapper = GymnasiumVecEnvWrapper(env, final_observations=False)
        wrapper.reset()
        _, _, _, _, info = wrapper.step(torch.zeros(1, 1))
        assert "final_obs" not in info and "_final_obs" not in info and "episode" in info

        # The env is the caller's: closing the wrapper leaves its scene alive.
        wrapper.close()
        assert env.scene.is_built
        env.step(torch.zeros(1, 1))
    finally:
        env.destroy()


@pytest.mark.requires_genesis
def test_without_final_observations_from_config_does_not_record_them_and_close_destroys_its_env(pendulum_file):
    wrapper = GymnasiumVecEnvWrapper.from_config(
        gym_pendulum_config(pendulum_file, num_envs=1), final_observations=False
    )

    assert not wrapper.env.env_options.record_final_observations
    wrapper.close()
    assert not wrapper.env.scene.is_built


@pytest.mark.requires_genesis
def test_random_episode_starts_stagger_the_episodes_reproducibly(pendulum_file):
    num_envs = 16
    wrapper = GymnasiumVecEnvWrapper.from_config(
        gym_pendulum_config(pendulum_file, num_envs=num_envs), random_episode_starts=True
    )
    try:
        max_length = wrapper.env.max_episode_length
        wrapper.reset(seed=3)
        starts = wrapper.env.episode_length_buf.clone()
        wrapper.reset(seed=3)
        assert torch.equal(wrapper.env.episode_length_buf, starts)
        assert ((starts >= 0) & (starts < max_length)).all() and len(set(starts.tolist())) > 1

        # Each env times out after the steps its clock had left, and reports the steps it actually took.
        lengths = torch.zeros(num_envs, dtype=torch.long, device=wrapper.device)
        for _ in range(max_length):
            _, _, _, truncated, info = wrapper.step(torch.zeros(num_envs, 1))
            first_end = truncated & (lengths == 0)
            lengths[first_end] = info["episode"]["l"][first_end]
        assert torch.equal(lengths, max_length - starts)
    finally:
        wrapper.close()


def test_a_config_without_rewards_is_refused_before_the_build():
    from eden.utils.configs import EdenConfig

    with pytest.raises(TypeError, match=r"'EdenConfig' is not an EdenRLConfig"):
        GymnasiumVecEnvWrapper.from_config(EdenConfig())


@pytest.mark.requires_genesis
@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_clip_actions_must_be_a_finite_positive_bound(wrapper, bound):
    with pytest.raises(ValueError, match=r"must be a finite positive bound"):
        GymnasiumVecEnvWrapper(wrapper.env, clip_actions=bound)


def test_from_config_refuses_a_bad_clip_actions_before_the_build(tmp_path):
    with pytest.raises(ValueError, match=r"clip_actions=0.0 must be a finite positive bound"):
        GymnasiumVecEnvWrapper.from_config(unbuildable_gym_pendulum_config(tmp_path), clip_actions=0.0)


@pytest.mark.requires_genesis
def test_from_config_destroys_the_env_when_the_wrapper_fails_to_construct(pendulum_file):
    built = []

    class FailingWrapper(GymnasiumVecEnvWrapper):
        def __init__(self, env, **kwargs):
            built.append((self, env))
            raise RuntimeError("the wrapper failed")

    with pytest.raises(RuntimeError, match=r"the wrapper failed"):
        FailingWrapper.from_config(gym_pendulum_config(pendulum_file, num_envs=1))
    ((wrapper, env),) = built
    assert not env.scene.is_built
    wrapper.close()  # a wrapper whose __init__ raised owns no env: closing it does nothing


@pytest.mark.requires_genesis
def test_a_failed_teardown_does_not_replace_the_error_that_stopped_from_config(pendulum_file):
    """The teardown's error is only logged: the one to read is why the wrapper could not be built."""
    built = []

    class FailingWrapper(GymnasiumVecEnvWrapper):
        def __init__(self, env, **kwargs):
            built.append(env)
            destroy = env.destroy

            def destroy_then_fail():
                destroy()
                raise OSError("the teardown failed")

            env.destroy = destroy_then_fail
            raise RuntimeError("the wrapper failed")

    with pytest.raises(RuntimeError, match=r"the wrapper failed"):
        FailingWrapper.from_config(gym_pendulum_config(pendulum_file, num_envs=1))
    assert not built[0].scene.is_built


@pytest.mark.requires_genesis
def test_terminated_and_truncated_are_split_by_time_out(wrapper):
    wrapper.reset()
    for _ in range(wrapper.env.max_episode_length - 1):
        _, _, terminated, truncated, _ = wrapper.step(torch.zeros(NUM_ENVS, 1))
        assert not (terminated | truncated).any()

    with ending(wrapper.env, [True, False, False]):
        _, _, terminated, truncated, info = wrapper.step(torch.zeros(NUM_ENVS, 1))

    # Env 0 ended by its termination term on the step its episode timed out: it is both, as gymnasium allows.
    assert terminated.tolist() == [True, False, False]
    assert truncated.tolist() == [True, True, True]
    assert info["_final_obs"].tolist() == info["_episode"].tolist() == [True, True, True]


@pytest.mark.requires_genesis
def test_a_done_env_returns_its_next_episode_and_reports_the_one_it_ended(wrapper):
    wrapper.reset()
    actions = torch.tensor([[1.0], [-0.5], [0.25]])
    returns = torch.zeros(NUM_ENVS)
    for _ in range(3):
        _, reward, _, _, _ = wrapper.step(actions)
        returns += reward

    with ending(wrapper.env, [False, True, False]):
        obs, reward, terminated, _, info = wrapper.step(actions)
    returns += reward

    assert terminated.tolist() == [False, True, False]
    # Env 1 is back at the pendulum's rest pose; the pose it swung to is its final observation.
    assert obs["policy"][1].abs().sum() == 0
    assert info["final_obs"]["policy"][1].abs().sum() > 0
    assert info["final_obs"]["critic"]["dofs_pos"][1] == info["final_obs"]["policy"][1, 0]
    assert info["final_obs"]["policy"][[0, 2]].abs().sum() == 0
    assert info["episode"]["l"].tolist() == [0, 4, 0]
    assert info["episode"]["r"][1].item() == pytest.approx(returns[1].item())
    assert info["episode"]["r"][[0, 2]].abs().sum() == 0

    # The ended env starts counting again; the others keep their running episode.
    for _ in range(wrapper.env.max_episode_length - 4):
        _, _, terminated, truncated, info = wrapper.step(actions)
    assert truncated.tolist() == [True, False, True]
    assert info["episode"]["l"].tolist() == [wrapper.env.max_episode_length, 0, wrapper.env.max_episode_length]


@pytest.mark.requires_genesis
def test_reset_restarts_the_episode_statistics(wrapper):
    wrapper.reset()
    for _ in range(3):
        wrapper.step(torch.ones(NUM_ENVS, 1))
    wrapper.reset()
    for _ in range(wrapper.env.max_episode_length):
        _, _, _, truncated, info = wrapper.step(torch.zeros(NUM_ENVS, 1))

    assert truncated.all()
    assert info["episode"]["l"].tolist() == [wrapper.env.max_episode_length] * NUM_ENVS
    assert info["episode"]["r"].abs().sum() == 0  # zero actions after the reset: the action-norm reward is zero


@pytest.mark.requires_genesis
def test_what_a_step_returns_is_not_overwritten_by_the_next(wrapper):
    """The env reuses its reward, done and some observation buffers in place; a learner keeps them a step longer."""
    wrapper.reset()
    with ending(wrapper.env, [True, False, False]):
        obs, reward, terminated, truncated, info = wrapper.step(torch.ones(NUM_ENVS, 1))
    kept = [obs["policy"], obs["critic"]["dofs_pos"], reward, terminated, truncated, info["final_obs"]["policy"]]
    snapshot = [value.clone() for value in kept]

    wrapper.step(torch.full((NUM_ENVS, 1), -1.0))

    assert all(torch.equal(value, before) for value, before in zip(kept, snapshot))
    assert info["_final_obs"] is not info["_episode"]


@pytest.mark.requires_genesis
def test_actions_are_clamped_to_the_clip_and_may_be_numpy(wrapper):
    wrapper.reset()
    wrapper.step(np.array([[5.0], [-5.0], [1.0]], dtype=np.float32))

    assert wrapper.env.action_manager.action.flatten().tolist() == [2.0, -2.0, 1.0]


@pytest.mark.requires_genesis
def test_an_unknown_observation_group_names_the_available_ones(wrapper):
    with pytest.raises(KeyError, match=r"'actor' is not one of the task's observation groups \['critic', 'policy'\]"):
        wrapper.group_space("actor")
