"""``eden.envs.wrappers.gymnasium_env``: gymnasium's interface on the shared core, and the ``Eden/<task>`` registration.

What the core does (final observations, episode statistics, copies, clamping) is pinned in
``test_vec_env_mixin.py``.
"""

from __future__ import annotations

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

from eden.envs.wrappers.gymnasium_env import GymnasiumVecEnvWrapper, make_vec_env, register_gymnasium_envs
from tests.envs.wrappers.conftest import gym_pendulum_config

NUM_ENVS = 3


@pytest.fixture(scope="module")
def wrapper(pendulum_file):
    wrapper = GymnasiumVecEnvWrapper.from_config(
        gym_pendulum_config(pendulum_file, num_envs=NUM_ENVS), clip_actions=2.0
    )
    yield wrapper
    wrapper.close()


@pytest.mark.requires_genesis
def test_the_spaces_mirror_the_observation_groups_and_the_action_clip(wrapper):
    single = wrapper.single_observation_space
    assert isinstance(single["policy"], gym.spaces.Box) and single["policy"].shape == (2,)
    assert isinstance(single["critic"], gym.spaces.Dict) and set(single["critic"].spaces) == {"dofs_pos", "dofs_vel"}
    assert wrapper.observation_space["policy"].shape == (NUM_ENVS, 2)
    assert wrapper.single_action_space == gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
    assert wrapper.action_space.shape == (NUM_ENVS, 1)
    assert wrapper.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP

    obs, _ = wrapper.reset()
    assert single.contains({k: _row(v, 0) for k, v in obs.items()})


@pytest.mark.requires_genesis
def test_reset_refuses_per_env_seeds_and_options(wrapper):
    with pytest.raises(TypeError, match=r"single int seed"):
        wrapper.reset(seed=[0, 1, 2])
    with pytest.raises(TypeError, match=r"takes no reset options"):
        wrapper.reset(options={"difficulty": "hard"})


# ----------------------------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------------------------


def test_registration_builds_nothing_and_make_points_at_make_vec():
    """Registering every task imports no Genesis; ``make_vec`` then refuses until ``eden.init()``.

    A subprocess, so neither the session's Genesis nor an earlier registration can mask what this one imports.
    """
    script = """
import sys
import eden
from eden.envs.wrappers.gymnasium_env import register_gymnasium_envs
from eden.tasks import TASK_REGISTRY
ids = register_gymnasium_envs()
assert ids == [f"Eden/{task}" for task in TASK_REGISTRY.list_tasks()], ids
assert "genesis" not in sys.modules
assert register_gymnasium_envs() == []
import gymnasium as gym
for make in (lambda: gym.make("Eden/reacher"), lambda: gym.make_vec("Eden/reacher", num_envs=2)):
    try:
        make()
    except Exception as error:
        print("ERROR", type(error).__name__, error)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    make_error, make_vec_error = [line for line in result.stdout.splitlines() if line.startswith("ERROR")]
    assert make_error.startswith("ERROR TypeError") and "gym.make_vec('Eden/reacher', num_envs=N)" in make_error
    assert make_vec_error.startswith("ERROR RuntimeError") and "eden.init(backend=gs.gpu)" in make_vec_error


@pytest.mark.requires_genesis
def test_make_vec_refuses_a_non_positive_num_envs_before_building():
    with pytest.raises(ValueError, match=r"num_envs=0\): num_envs must be >= 1"):
        make_vec_env(0, task="reacher")


@pytest.mark.requires_genesis
def test_make_vec_builds_the_registered_task_with_num_envs(pendulum_file, task_registry_alias):
    from eden.utils.configs import EdenRLConfig

    defaults = gym_pendulum_config(pendulum_file)

    class GymPendulumConfig(EdenRLConfig):
        env_options: type(defaults.env_options) = defaults.env_options
        scene_options: type(defaults.scene_options) = defaults.scene_options
        action_options: type(defaults.action_options) = defaults.action_options
        observation_options: type(defaults.observation_options) = defaults.observation_options
        reward_options: type(defaults.reward_options) = defaults.reward_options
        termination_options: type(defaults.termination_options) = defaults.termination_options

    task_registry_alias(GymPendulumConfig, "gym_pendulum_test")
    try:
        assert "Eden/gym_pendulum_test" in register_gymnasium_envs()
        envs = gym.make_vec("Eden/gym_pendulum_test", num_envs=2, clip_actions=1.0)
        try:
            assert isinstance(envs, GymnasiumVecEnvWrapper)
            assert envs.num_envs == envs.env.num_envs == 2
            assert envs.single_action_space.high.tolist() == [1.0]
            obs, _ = envs.reset(seed=0)
            assert obs["policy"].shape == (2, 2)
        finally:
            envs.close()
    finally:
        gym.registry.pop("Eden/gym_pendulum_test", None)


def _row(value, i):
    if isinstance(value, dict):
        return {key: _row(item, i) for key, item in value.items()}
    return value[i].cpu().numpy()
