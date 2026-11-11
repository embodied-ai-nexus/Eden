"""Both phases train through Eden's runner path (what ``eden train`` / ``eden inference`` call) on a 2x2-cell env.

Teacher: the task's runner options build rsl_rl's ``OnPolicyRunner`` with ``RmaPPO``; its checkpoint replays as the
deployed composition. Student: the student task's runner options build ``RmaDepthDistillationRunner``, which distills
that teacher checkpoint and resumes from its own. The depth renderer is the conftest's counting stub.
"""

from __future__ import annotations

import pytest
import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version
from eden.tasks import TASK_REGISTRY
from eden.utils.assets import model_checkpoints
from tests.support.envs import build_env

pytestmark = [
    pytest.mark.requires_genesis,
    pytest.mark.slow,
    pytest.mark.skipif(detect_rsl_rl_version() < MIN_RSL_RL_VERSION, reason="the RMA stack needs rsl-rl-lib >= 5.5.1"),
]

STEPS_PER_ITERATION = 5  # one depth-refresh window


@pytest.fixture(scope="module")
def teacher_env(small_teacher_config):
    return build_env(small_teacher_config(), reset=True)


@pytest.fixture(scope="module")
def student_env(small_teacher_config):
    """Its own env: the teacher's rollouts run under ``torch.inference_mode``, which turns the env's buffers into
    inference tensors the student's gradient-carrying loop cannot update — one phase per process, as in real runs.
    """
    return build_env(small_teacher_config(), reset=True)


def _wrap(env, task: str) -> RslRlVecEnvWrapper:
    options = TASK_REGISTRY.build(task).runner_options
    options.num_steps_per_env = STEPS_PER_ITERATION
    options.logger = "tensorboard"
    options.device = "auto"
    return RslRlVecEnvWrapper(env, options)


def _latest(log_dir) -> str:
    return str(log_dir / model_checkpoints(str(log_dir))[-1][1])


@pytest.fixture(scope="module")
def teacher_checkpoint(teacher_env, tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("teacher")
    runner = make_rsl_rl_runner(_wrap(teacher_env, "extreme_parkour_teacher"), log_dir=str(log_dir))
    assert type(runner.alg).__name__ == "RmaPPO"
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    return _latest(log_dir)


def test_the_teacher_checkpoint_replays_as_the_deployed_policy(teacher_env, teacher_checkpoint, tmp_path):
    wrapper = _wrap(teacher_env, "extreme_parkour_teacher")
    runner = make_rsl_rl_runner(wrapper, checkpoint=teacher_checkpoint, log_dir=str(tmp_path))
    policy = runner.get_inference_policy(device=wrapper.device)
    obs = wrapper.get_observations()
    with torch.no_grad():
        actions = policy(obs)
        flat = obs["policy"]
        deployed = runner.alg.actor.net(runner.alg.actor.with_estimate(flat), True)  # estimator + history latent
    assert actions.shape == (wrapper.num_envs, wrapper.num_actions)
    torch.testing.assert_close(actions, deployed)
    assert runner.alg.iteration == runner.current_learning_iteration  # the DAgger cadence resumes in step


def test_the_student_distills_the_teacher_and_resumes_from_its_own_checkpoint(
    student_env, teacher_checkpoint, tmp_path
):
    wrapper = _wrap(student_env, "extreme_parkour_student")
    runner = make_rsl_rl_runner(wrapper, checkpoint=teacher_checkpoint, log_dir=str(tmp_path))
    assert type(runner).__name__ == "RmaDepthDistillationRunner"
    alg = runner.alg
    for key, value in alg.teacher.net.state_dict().items():  # a fresh distillation starts from the teacher's actor
        assert torch.equal(alg.depth_actor.state_dict()[key], value), key

    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    student_checkpoint = _latest(tmp_path)
    assert any(not torch.equal(alg.depth_actor.state_dict()[k], v) for k, v in alg.teacher.net.state_dict().items())

    resumed = make_rsl_rl_runner(wrapper, checkpoint=student_checkpoint, log_dir=str(tmp_path / "resumed"))
    assert resumed.current_learning_iteration == 1
    for key, value in alg.depth_encoder.state_dict().items():
        assert torch.equal(resumed.alg.depth_encoder.state_dict()[key], value), key
    actions = resumed.get_inference_policy(device=wrapper.device)(wrapper.get_observations())
    assert actions.shape == (wrapper.num_envs, wrapper.num_actions)
    assert torch.isfinite(actions).all()


def test_the_student_refuses_to_train_without_a_teacher(student_env, tmp_path):
    runner = make_rsl_rl_runner(_wrap(student_env, "extreme_parkour_student"), log_dir=str(tmp_path))
    with pytest.raises(ValueError, match=r"needs a teacher: pass --checkpoint"):
        runner.learn(num_learning_iterations=1)


def test_the_student_rejects_an_encoder_cadence_off_the_depth_refresh(student_env, tmp_path):
    wrapper = _wrap(student_env, "extreme_parkour_student")
    wrapper.runner_options.algorithm.update_interval = 3
    wrapper.runner_options.num_steps_per_env = 6
    with pytest.raises(
        ValueError, match=r"update_interval=3 does not match the env's refresh cadence of \{.*'depth_camera/depth': 5"
    ):
        make_rsl_rl_runner(wrapper, log_dir=str(tmp_path))


def test_a_teacher_checkpoint_does_not_replay_as_an_untrained_student(student_env, teacher_checkpoint, tmp_path):
    runner = make_rsl_rl_runner(
        _wrap(student_env, "extreme_parkour_student"), checkpoint=teacher_checkpoint, log_dir=str(tmp_path)
    )
    with pytest.raises(RuntimeError, match=r"depth student has not been trained.*--task extreme_parkour_teacher"):
        runner.get_inference_policy()


def test_the_student_loop_spreads_episode_clocks_and_checks_for_nan(
    student_env, teacher_checkpoint, tmp_path, monkeypatch
):
    wrapper = _wrap(student_env, "extreme_parkour_student")
    runner = make_rsl_rl_runner(wrapper, checkpoint=teacher_checkpoint, log_dir=str(tmp_path))
    wrapper.unwrapped.episode_length_buf.zero_()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    assert wrapper.unwrapped.episode_length_buf.unique().numel() > 1  # not every env started at step 0

    real_step = wrapper.step

    def nan_rewards(actions):
        obs, rewards, dones, extras = real_step(actions)
        return obs, torch.full_like(rewards, float("nan")), dones, extras

    monkeypatch.setattr(wrapper, "step", nan_rewards)
    with pytest.raises(ValueError, match=r"rewards returned by the environment contain NaN"):
        runner.learn(num_learning_iterations=1)
