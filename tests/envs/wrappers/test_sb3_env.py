"""``eden.envs.wrappers.sb3_env``: one observation group as numpy, and the per-env infos SB3 reads at an episode end."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3", reason="the sb3 extra is not installed")

from stable_baselines3 import PPO  # noqa: E402

from eden.envs.wrappers.sb3_env import Sb3VecEnvWrapper  # noqa: E402
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
def test_the_policy_observes_one_group_as_numpy(rl_env):
    env = Sb3VecEnvWrapper(rl_env, clip_actions=1.0)
    obs = env.reset()

    assert env.observation_space == env.single_observation_space["policy"]
    assert env.action_space == env.single_action_space
    assert isinstance(obs, np.ndarray) and obs.shape == (NUM_ENVS, 2)
    obs, reward, done, infos = env.step(np.zeros((NUM_ENVS, 1), dtype=np.float32))
    assert obs.shape == (NUM_ENVS, 2) and reward.shape == done.shape == (NUM_ENVS,)
    assert infos == [{}, {}, {}]


@pytest.mark.requires_genesis
def test_random_episode_starts_follow_sb3s_seed(rl_env):
    env = Sb3VecEnvWrapper(rl_env, clip_actions=1.0, random_episode_starts=True)
    env.seed(3)
    env.reset()
    starts = rl_env.episode_length_buf.clone()
    env.seed(3)
    env.reset()

    assert torch.equal(rl_env.episode_length_buf, starts)
    assert ((starts >= 0) & (starts < rl_env.max_episode_length)).all()


@pytest.mark.requires_genesis
def test_a_dict_group_is_a_dict_observation(rl_env):
    env = Sb3VecEnvWrapper(rl_env, clip_actions=1.0, obs_group="critic")
    obs = env.reset()

    assert set(obs) == {"dofs_pos", "dofs_vel"} and obs["dofs_pos"].shape == (NUM_ENVS, 1)


@pytest.mark.requires_genesis
@pytest.mark.parametrize("obs_group", ["policy", "critic"])
def test_a_done_env_reports_its_terminal_observation_time_out_and_episode(rl_env, obs_group):
    env = Sb3VecEnvWrapper(rl_env, clip_actions=1.0, obs_group=obs_group)
    env.reset()
    actions = np.array([[1.0], [-0.5], [0.25]], dtype=np.float32)
    returns = np.zeros(NUM_ENVS)
    for _ in range(rl_env.max_episode_length - 1):
        returns += env.step(actions)[1]

    with ending(rl_env, [True, False, False]):
        obs, reward, done, infos = env.step(actions)
    returns += reward

    assert done.tolist() == [True, True, True]
    # Env 0 terminated as it timed out: terminal, so SB3 must not bootstrap it.
    assert [info["TimeLimit.truncated"] for info in infos] == [False, True, True]
    assert [info["episode"]["l"] for info in infos] == [rl_env.max_episode_length] * NUM_ENVS
    assert [info["episode"]["r"] for info in infos] == pytest.approx(returns.tolist())
    # Every env is back at rest; each terminal observation is the swung pose its episode ended in.
    terminal = [info["terminal_observation"] for info in infos]
    if obs_group == "policy":
        assert np.abs(obs).sum() == 0
        assert all(row.shape == (2,) and np.abs(row).sum() > 0 for row in terminal)
    else:
        assert np.abs(obs["dofs_pos"]).sum() == 0
        assert all(row["dofs_pos"].shape == (1,) and np.abs(row["dofs_pos"]).sum() > 0 for row in terminal)


@pytest.mark.requires_genesis
def test_reset_options_are_refused_not_dropped(rl_env):
    env = Sb3VecEnvWrapper(rl_env, clip_actions=1.0)
    env.set_options({"difficulty": "hard"})

    with pytest.raises(TypeError, match=r"takes no reset options"):
        env.reset()


@pytest.mark.requires_genesis
def test_sb3_ppo_trains_and_logs_episodes(rl_env):
    env = Sb3VecEnvWrapper(rl_env, clip_actions=1.0)
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=NUM_ENVS * 8, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=NUM_ENVS * 8 * 2)

    assert model.num_timesteps == NUM_ENVS * 8 * 2
    assert len(model.ep_info_buffer) > 0


@pytest.mark.requires_genesis
def test_an_unknown_group_names_the_available_ones(rl_env):
    with pytest.raises(KeyError, match=r"'actor' is not one of the task's observation groups \['critic', 'policy'\]"):
        Sb3VecEnvWrapper(rl_env, clip_actions=1.0, obs_group="actor")


@pytest.mark.requires_genesis
def test_an_env_without_final_observations_is_refused(pendulum_file):
    """SB3 would otherwise replay a done env's reset observation as its next observation."""
    env = built_gym_pendulum_env(pendulum_file, num_envs=1, record_final_observations=False)
    try:
        with pytest.raises(ValueError, match=r"record_final_observations=False.*SB3 cannot do without them.*Do NOT"):
            Sb3VecEnvWrapper(env, clip_actions=1.0)
    finally:
        env.destroy()


@pytest.mark.requires_genesis
def test_an_unbounded_action_space_is_refused_before_sb3_asserts(rl_env):
    with pytest.raises(ValueError, match=r"clip_actions=None.*from_config\(cfg, clip_actions=1.0\)"):
        Sb3VecEnvWrapper(rl_env, clip_actions=None)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"clip_actions": None}, ValueError, r"clip_actions=None"),
        ({"clip_actions": 1.0, "obs_group": "actor"}, KeyError, r"'actor' is not one of the task's observation groups"),
    ],
)
def test_from_config_refuses_a_bad_argument_before_the_build(tmp_path, kwargs, error, match):
    with pytest.raises(error, match=match):
        Sb3VecEnvWrapper.from_config(unbuildable_gym_pendulum_config(tmp_path), **kwargs)


@pytest.mark.requires_genesis
def test_from_config_records_final_observations_and_close_destroys_its_env(pendulum_file):
    config = gym_pendulum_config(pendulum_file, num_envs=1)
    env = Sb3VecEnvWrapper.from_config(config, clip_actions=1.0)

    assert env.env.env_options.record_final_observations and not config.env_options.record_final_observations
    env.close()
    assert not env.env.scene.is_built
    env.close()  # a second close does nothing
