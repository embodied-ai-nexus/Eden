"""The SuperMimic runner on the supermimic task: its world model grades as the simulator does, and it trains."""

import pytest
import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.learning.wrappers.rsl_rl.supermimic.runner import make_supermimic_runner
from tests.tasks.supermimic.conftest import supermimic_config, write_lafan1_library

pytestmark = [pytest.mark.requires_genesis]


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    cfg = supermimic_config(write_lafan1_library(tmp_path_factory.mktemp("lafan1")))
    o = cfg.runner_options
    o.device, o.logger, o.num_steps_per_env, o.buffer_size = "cpu", "tensorboard", 4, 32
    o.world_model.hidden_dims, o.world_model.batch_size, o.world_model.num_batches = (32,), 8, 2
    o.policy.actor_hidden_dims, o.policy.critic_hidden_dims, o.policy.batch_size = (32,), (16,), 8
    o.policy.horizon, o.replay_ratio = 4, 64.0
    env = RslRlVecEnvWrapper.from_config(cfg)
    return make_supermimic_runner(env, log_dir=str(tmp_path_factory.mktemp("run")))


def test_the_world_model_reward_and_termination_grade_a_real_transition_as_the_task_does(runner):
    env, rep, wm_env = runner.env, runner.representation, runner.wm_env
    obs = env.get_observations()
    torch.manual_seed(0)
    for _ in range(6):
        action = 0.5 * torch.randn(env.num_envs, env.num_actions)
        last = env.unwrapped.action_manager.action.clone()
        obs, reward, dones, _ = env.step(action)
        kept = ~dones.bool()
        state = obs["full_state"]
        target = wm_env.reference_bodies(runner.motion.frame_index)
        wm_reward = rep.reward(rep.aux_state(state), rep.represent(state), action, last, target)
        torch.testing.assert_close(wm_reward[kept], reward[kept], atol=1e-5, rtol=1e-4)


def test_it_trains_and_a_checkpoint_restores_the_policy(runner, tmp_path):
    runner.learn(num_learning_iterations=3)
    assert runner.current_learning_iteration == 3
    assert runner.replay_buffer.num_transitions == 12
    path = str(tmp_path / "model_3.pt")
    runner.save(path)
    obs = runner.env.get_observations()
    policy = runner.get_inference_policy()
    before = policy(obs)
    for p in runner.actor.parameters():
        p.data.add_(1.0)
    runner.load(path)
    torch.testing.assert_close(runner.get_inference_policy()(obs), before)
    assert runner.current_learning_iteration == 3
