"""``OMPLPlanner`` on a Franka beside a fixed box, with whichever OMPL bindings are installed (1.x or 2.x)."""

from __future__ import annotations

# isort: off
import eden as en
# isort: on

import pytest
import torch

from eden.extensions.motion_planning.ompl.planner import IS_OMPL_AVAILABLE, OMPLPlanner
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions
from eden.options.extensions.motion_planning import PlannerOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# The planner's own guard, so a partial OMPL install (``ompl`` without ``ompl.base``) skips rather than errors.
pytestmark = pytest.mark.skipif(not IS_OMPL_AVAILABLE, reason="OMPL is optional: `pip install ompl`")

BOX_POS = (0.45, 0.35, 0.2)
DOWN = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # wxyz: the hand pointing at the floor


@pytest.fixture(scope="module")
def env():
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=en.robots.FrankaEmikaPanda(),
            box=BoxOptions(size=(0.1, 0.1, 0.1), fixed=True, default_root_pos=BOX_POS),
        ),
    )
    return build_env(config, reset=True)


@pytest.fixture(scope="module")
def planner(env):
    return OMPLPlanner(env, PlannerOptions(entity_name="robot"))


def _goal(env, pos) -> torch.Tensor:
    robot = env.entities["robot"]
    return robot.inverse_kinematics(links_name="hand", poss=torch.tensor([pos]), quats=DOWN)[0]


def test_a_free_path_runs_from_start_to_goal_within_the_joint_limits(env, planner):
    robot = env.entities["robot"]
    start = robot.get_qpos(envs_idx=0).squeeze(0).clone()
    goal = _goal(env, (0.35, 0.25, 0.5))

    waypoints = planner.plan_motion(qpos_goal=goal, num_waypoints=20, envs_idx=0)

    assert len(waypoints) == 20
    path = torch.stack(waypoints).to(start.device)
    torch.testing.assert_close(path[0], start, atol=1e-4, rtol=0.0)
    torch.testing.assert_close(path[-1], goal.to(start.device), atol=1e-4, rtol=0.0)
    lower, upper = (limit[0].to(start.device) for limit in robot.get_qpos_limit(0))
    assert ((path >= lower - 1e-6) & (path <= upper + 1e-6)).all()
    # Planning moves the robot through every state it checks; it puts the start back.
    torch.testing.assert_close(robot.get_qpos(envs_idx=0).squeeze(0), start)


def test_a_goal_inside_the_box_is_refused_by_the_collision_check(env, planner, eden_caplog):
    goal = _goal(env, (BOX_POS[0], BOX_POS[1], BOX_POS[2] + 0.06))  # the fingertips in the box

    assert planner.plan_motion(qpos_goal=goal, timeout=1.0, envs_idx=0) == []
    assert "Path planning infeasible" in eden_caplog.text  # an invalid goal, not a timeout
    # With the check off the same goal plans, so the refusal above is the collision check's.
    assert planner.plan_motion(qpos_goal=goal, ignore_collision=True, num_waypoints=5, envs_idx=0)


def test_a_planner_that_times_out_retries_up_to_max_retry(env, planner, eden_caplog):
    goal = _goal(env, (0.35, 0.25, 0.5))

    planner.plan_motion(qpos_goal=goal, timeout=1e-9, max_retry=3, envs_idx=0)

    assert eden_caplog.text.count("Trying again") == 2
    assert "Path planning did not converge. Returning" in eden_caplog.text


def test_an_unknown_planner_is_refused_with_the_installed_ones(env, planner):
    goal = env.entities["robot"].get_qpos(envs_idx=0).squeeze(0)

    with pytest.raises(ValueError, match=r"planner='NoSuchPlanner' is not an ompl\.geometric planner.*RRTConnect"):
        planner.plan_motion(qpos_goal=goal, planner="NoSuchPlanner", envs_idx=0)
