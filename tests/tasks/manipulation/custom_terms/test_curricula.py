"""Tests for ``eden.tasks.manipulation.custom_terms.curricula``: ``GoalRangeCurriculum``'s success-gated level walk."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.commands.oracle import UniformT3Command
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.managers.curricula import CurriculumManagerOptions, CurriculumTermOptions
from eden.tasks.manipulation.custom_terms.curricula import GoalRangeCurriculum
from eden.utils.configs import CommandManagerOptions, EdenRLConfig
from tests.support.envs import build_env

NUM_ENVS = 4
FULL_X, EASY_X = (0.2, 0.5), 0.28


@pytest.fixture(scope="module")
def env():
    """A fixed "robot" box at the origin, a free cube, and the lift task's goal command and curriculum."""
    return build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                robot=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.05), is_fixed_base=True),
                cube=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.3, 0.0, 0.025)),
            ),
            command_options=CommandManagerOptions(
                lift_target=UniformT3Command.configure(
                    x_range=FULL_X, y_range=(-0.3, 0.3), z_range=(0.6, 0.9), resampling_time_range=(100.0, 100.0)
                )
            ),
            curriculum_options=CurriculumManagerOptions(
                # ema_window scales to num_envs / 4096 of itself, so 4096 is a window of exactly NUM_ENVS episodes
                goal_range=GoalRangeCurriculum.configure(
                    advance_thresh=0.3, retreat_thresh=0.2, easy_center=(EASY_X, 0.0, 0.66)
                ),
            ),
        ),
        reset=True,
    )


def test_level_advances_on_reach_success_and_retreats_on_failure(env):
    """Hand-walked: step_up = step_down = 0.05, scaled by the fraction of envs whose episode ended."""
    term = env.curriculum_manager.get_term("goal_range")
    goal_w = env.entities["robot"].get_pos() + env.command_manager.get_term("lift_target").command
    away = goal_w + torch.tensor([5.0, 0.0, 0.0], device=env.device)
    ended = torch.tensor([True, True, False, False], device=env.device)
    everyone = torch.ones(NUM_ENVS, dtype=torch.bool, device=env.device)
    steps = [  # (cube position, episodes that just ended, level, goal_ema) after the step
        (goal_w, ~everyone, 0.0, 0.0),  # every cube reaches its goal; nothing ended yet
        (goal_w, ended, 0.025, 0.5),  # 2 of 4 end, all reached: ema lerps halfway to 1 > 0.3, level += 0.05 * 2/4
        (away, everyone, 0.075, 1.0),  # all end having reached earlier: ema -> 1, level += 0.05
        (away, everyone, 0.025, 0.0),  # all end, none reached since: ema -> 0 < 0.2, level -= 0.05
    ]
    try:
        for cube_pos, done, level, goal_ema in steps:
            env.entities["cube"].set_pos(cube_pos)
            env.episode_length_buf[:] = torch.where(done, 0, 1)
            state = term.compute()
            torch.testing.assert_close(state["level"].cpu(), torch.tensor(level))
            torch.testing.assert_close(state["goal_ema"].cpu(), torch.tensor(goal_ema))
        assert bool(state["primed"])
        # the command's x range interpolates from the easy point toward the full range by the level
        x_range = env.command_manager.get_term("lift_target").x_range
        expected = tuple(EASY_X + (end - EASY_X) * 0.025 for end in FULL_X)
        assert x_range == pytest.approx(expected, abs=1e-6)
    finally:
        env.reset()


def test_compute_syncs_once(count_cuda_syncs, cuda_goal_env):
    """The one sync left reads the level back, because the command samples from Python-float ranges."""
    term = GoalRangeCurriculum(
        env=cuda_goal_env, options=CurriculumTermOptions(name="goal_range_curriculum", easy_center=(EASY_X, 0.0, 0.66))
    )
    term.compute()
    assert count_cuda_syncs(term.compute) == 1


def test_easy_center_is_required_where_the_config_is_written():
    """It is in the robot base frame, so a default fits only a task whose arm stands at the world origin.

    ``configure()`` refuses it before any env exists, not after a scene build.
    """
    with pytest.raises(ValueError, match="easy_center is required.*'lift_target' command's frame"):
        GoalRangeCurriculum.configure(advance_thresh=0.3)


def test_easy_center_is_required_of_options_that_skipped_configure(env):
    """A loaded ``config.json`` or an in-place edit never calls ``configure()``; the build still refuses it."""
    with pytest.raises(ValueError, match="easy_center is required"):
        GoalRangeCurriculum(env=env, options=CurriculumTermOptions(name="goal_range_curriculum"))
