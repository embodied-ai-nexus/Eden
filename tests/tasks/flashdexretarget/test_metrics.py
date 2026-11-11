"""Tests for the task's metrics and evaluation: completion, tracking errors and the evaluation's start frame."""

from __future__ import annotations

import pytest
import torch

from tests.tasks.flashdexretarget.conftest import build_task

pytestmark = [pytest.mark.slow, pytest.mark.requires_genesis]

#: From each clip's first frame, with no early reset on the zero action's loose tracking.
_FULL_CLIPS = {
    "motion_options.default.motion_start_from_middle_range": (0.0, 0.0),
    "termination_options.task_reward_early_reset.params": {
        "reward_term": "object_pcd",
        "per_step_threshold": -1.0,
        "check_interval": 5,
    },
}


def _first_reset_log(env) -> tuple[dict, torch.Tensor]:
    hand = env.action_manager.get_term("hand")
    for _ in range(200):
        _, _, terminated, timeouts, extras = env.step(torch.zeros(env.num_envs, hand.action_dim))
        if (terminated | timeouts).any():
            return extras["log"], timeouts
    raise AssertionError("no episode ended within 200 steps")


def test_an_episode_that_plays_its_clip_out_succeeds_and_reports_its_mean_tracking_errors(motion_file):
    """The assist (at its starting gains, training) holds each object on its reference through a zero action, so the
    clip runs out: ``motion_clip_ended`` ends the episode, judged a success, with an object error near zero.
    """
    env = build_task(motion_file, **_FULL_CLIPS)
    log, timeouts = _first_reset_log(env)
    assert bool(timeouts.all())
    assert float(log["Metrics/completed"]) == 1.0
    assert float(log["Metrics/Success"]) == 1.0
    assert 0.0 < float(log["Metrics/object_pos_error"]) < 0.02
    assert 0.0 <= float(log["Metrics/object_rot_error"]) < 0.2
    assert 0.0 < float(log["Metrics/tip_pos_error"]) < 0.2


def test_an_episode_a_failure_term_ends_is_not_a_success(motion_file):
    """A 0 m fingertip threshold ends every episode on its first step past the grace period, long before its clip."""
    env = build_task(
        motion_file,
        **_FULL_CLIPS,
        **{
            "termination_options.fingertip_diverged.threshold": 0.0,
            "termination_options.fingertip_diverged.grace_steps": 0,
        },
    )
    log, timeouts = _first_reset_log(env)
    assert not bool(timeouts.any())
    assert float(log["Metrics/completed"]) == 0.0
    assert float(log["Metrics/Success"]) == 0.0


def test_an_episode_the_time_limit_ends_before_its_clip_does_succeeds(motion_file):
    """A 0.05 s episode limit (3 control steps) ends each episode long before its clip: ``time_out`` is a time-out like
    ``motion_clip_ended``, so the episode completed, as the assist curriculum counts it.
    """
    env = build_task(motion_file, **_FULL_CLIPS, **{"env_options.episode_length_s": 0.05})
    log, timeouts = _first_reset_log(env)
    assert bool(timeouts.all())
    assert float(log["Metrics/completed"]) == 1.0
    assert float(log["Metrics/Success"]) == 1.0


@pytest.mark.parametrize("eval_mode", [False, True], ids=["training", "eval"])
def test_an_evaluation_starts_every_episode_at_its_clips_first_frame(motion_file, eval_mode):
    """Training keeps FlashDexRetarget's uniform start in [0, 0.99 T); an eval build starts at frame 0, so a completed
    eval episode tracked its whole clip.
    """
    env = build_task(motion_file, num_envs=16, eval_mode=eval_mode)
    start = env.motion_manager.get_term("default")._motion_buffer_start_time_s
    if eval_mode:
        assert bool((start == 0.0).all())
    else:
        assert int((start > 0.0).sum()) >= 12  # P(a uniform start lands on frame 0) is ~1/40 per env
