"""``eden.envs.wrappers.skrl_env``: flattened torch observations and an optional critic state for skrl."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("skrl", reason="the skrl extra is not installed")

from skrl.utils.runner.torch import Runner  # noqa: E402

from eden.envs.wrappers.skrl_env import SkrlVecEnvWrapper  # noqa: E402
from tests.envs.wrappers.conftest import (  # noqa: E402
    built_gym_pendulum_env,
    ending,
    gym_pendulum_config,
    unbuildable_gym_pendulum_config,
)

NUM_ENVS = 3


@pytest.fixture(scope="module")
def rl_env(pendulum_file):
    env = built_gym_pendulum_env(pendulum_file, num_envs=NUM_ENVS)
    yield env
    env.destroy()


@pytest.mark.requires_genesis
def test_observation_and_state_are_flat_tensors_of_their_groups(rl_env):
    env = SkrlVecEnvWrapper(rl_env, state_group="critic", final_observations=True)
    obs, _ = env.reset()

    assert env.num_envs == NUM_ENVS and env.device == torch.device(rl_env.device)
    assert env.observation_space == env.single_observation_space["policy"]
    assert env.state_space == env.single_observation_space["critic"]
    assert obs.shape == (NUM_ENVS, 2) and env.state().shape == (NUM_ENVS, 2)
    obs, reward, terminated, truncated, info = env.step(torch.ones(NUM_ENVS, 1))
    assert reward.shape == terminated.shape == truncated.shape == (NUM_ENVS, 1)
    # The critic's terms, flattened in skrl's (sorted) order, read the same state as the policy group.
    assert torch.equal(env.state(), obs)
    assert "final_obs" in info


@pytest.mark.requires_genesis
def test_the_episode_statistics_are_tensors_skrl_records(rl_env):
    """The skrl trainers record an ``info[environment_info]`` value only if it is a one-element tensor."""
    env = SkrlVecEnvWrapper(rl_env)
    env.reset()
    for _ in range(rl_env.max_episode_length):
        _, _, _, truncated, info = env.step(torch.zeros(NUM_ENVS, 1))

    assert truncated.all()
    assert isinstance(rl_env.reset_info["Curriculum/action_weight/weight"], float)
    assert "Episode_Reward/action" in info["log"]
    assert all(isinstance(value, torch.Tensor) and value.numel() == 1 for value in info["log"].values())


@pytest.mark.requires_genesis
def test_an_env_that_terminates_as_it_times_out_is_not_truncated(rl_env):
    """An skrl agent bootstraps every ``truncated`` env; one that also terminated is terminal."""
    env = SkrlVecEnvWrapper(rl_env)
    env.reset()
    for _ in range(rl_env.max_episode_length - 1):
        env.step(torch.zeros(NUM_ENVS, 1))

    with ending(rl_env, [True, False, False]):
        _, _, terminated, truncated, _ = env.step(torch.zeros(NUM_ENVS, 1))

    assert terminated.flatten().tolist() == [True, False, False]
    assert truncated.flatten().tolist() == [False, True, True]


@pytest.mark.requires_genesis
def test_without_a_state_group_there_is_no_state(rl_env):
    env = SkrlVecEnvWrapper(rl_env)
    env.reset()

    assert env.state_space is None and env.state() is None


def test_from_config_refuses_an_unknown_state_group_before_the_build(tmp_path):
    with pytest.raises(KeyError, match=r"'actor' is not one of the task's observation groups \['critic', 'policy'\]"):
        SkrlVecEnvWrapper.from_config(unbuildable_gym_pendulum_config(tmp_path), state_group="actor")


@pytest.mark.requires_genesis
def test_skrl_runner_trains_ppo(rl_env):
    network = [{"name": "net", "input": "OBSERVATIONS", "layers": [8], "activations": "elu"}]
    cfg = {
        "seed": 0,
        "models": {
            "separate": True,
            "policy": {"class": "GaussianMixin", "network": network, "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "network": network, "output": "ONE"},
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 4,
            "learning_epochs": 1,
            "mini_batches": 1,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": 8,
            "disable_progressbar": True,
            "close_environment_at_exit": False,  # the fixture destroys it
        },
    }
    runner = Runner(SkrlVecEnvWrapper(rl_env), cfg)
    runner.run("train")

    assert runner.agent.memory.get_tensor_by_name("rewards").shape == (4, NUM_ENVS, 1)


@pytest.mark.requires_genesis
def test_from_config_leaves_final_observations_off(pendulum_file):
    """The skrl agents never read them, so the adapter does not pay the extra observation compute."""
    env = SkrlVecEnvWrapper.from_config(gym_pendulum_config(pendulum_file, num_envs=1))
    try:
        env.reset()
        *_, info = env.step(torch.zeros(1, 1))

        assert not env.env.env_options.record_final_observations
        assert "final_obs" not in info
    finally:
        env.close()
    assert not env.env.scene.is_built
    env.close()  # a second close does nothing


@pytest.mark.requires_genesis
def test_only_the_groups_skrl_reads_are_copied_and_returned(rl_env):
    env = SkrlVecEnvWrapper(rl_env, final_observations=True)
    env.reset()
    with ending(rl_env, [True, False, False]):
        *_, info = env.step(torch.zeros(NUM_ENVS, 1))

    assert set(info["final_obs"]) == {"policy"}
    assert info["_final_obs"].tolist() == [True, False, False]


@pytest.mark.requires_genesis
def test_random_episode_starts_stagger_the_clocks_on_reset(rl_env):
    env = SkrlVecEnvWrapper(rl_env, random_episode_starts=True)
    env.reset()

    starts = rl_env.episode_length_buf
    assert ((starts >= 0) & (starts < rl_env.max_episode_length)).all()
    assert env.step(torch.zeros(NUM_ENVS, 1))[0].shape == (NUM_ENVS, 2)
