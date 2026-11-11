"""Block-world demonstrators on the real robosuite ``Stack`` scene: setup checks, the rig's frames, subgoal decisions.

One ``robosuite_stack`` build (two envs) serves every test; decisions are asked of the controller on
hand-placed block layouts, so each expected subgoal is known exactly.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import numpy as np
import pytest
import torch

from eden.extensions.scripted_demos.block_world import LiftController, TowerController
from eden.extensions.scripted_demos.controller import ObjectSnapshot
from eden.extensions.scripted_demos.primitives import tool_quat
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.common import TABLE_TOP_Z
from eden.utils.geom import transform_by_quat, yaw_from_quat
from tests.support.envs import build_env

H_B, H_A = 0.025, 0.02  # robosuite Stack: cubeB is 5 cm, cubeA 4 cm
BOUNDS = {
    "table_z": TABLE_TOP_Z,
    "spot_bounds": ((-0.12, -0.15), (0.12, 0.15)),
    "workspace": ((-0.35, -0.35, TABLE_TOP_Z), (0.35, 0.35, TABLE_TOP_Z + 0.45)),
    "effector_bounds": ((-0.2, -0.15, TABLE_TOP_Z + 0.12), (0.05, 0.15, TABLE_TOP_Z + 0.3)),
}


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_stack")
    config.env_options.num_envs = 2
    return build_env(config, reset=True)


def _tower(env, **kwargs) -> TowerController:
    return TowerController(env, ["cubeB", "cubeA"], **{**BOUNDS, "p_mistake": 0.0, **kwargs})


def _on_table(x, y, half):
    return [x, y, TABLE_TOP_Z + half]


# --- setup ------------------------------------------------------------------------------------------


def test_the_task_registers_its_demonstrator(env):
    controller = SCRIPTED_DEMO_REGISTRY.get("robosuite_stack")(env, seed=0)
    assert isinstance(controller, TowerController)
    assert controller.object_names == ["cubeB", "cubeA"]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rig": "cubeA"}, TypeError, r"'cubeA' is a BoxOptions.*ManipulatorOptions composite"),
        ({"block_names": ["cubeB", "mug"]}, TypeError, r"Block 'mug' must be a BoxOptions"),
    ],
)
def test_a_bad_binding_fails_at_construction_naming_the_value(env, kwargs, error, match):
    names = kwargs.pop("block_names", ["cubeB", "cubeA"])
    with pytest.raises(error, match=match):
        TowerController(env, names, **{**BOUNDS, **kwargs})


def test_the_straight_down_frame_points_the_approach_axis_at_the_floor(env):
    controller = _tower(env)
    _, rot = controller.tool_pose()
    approach = transform_by_quat(
        torch.tensor([[0.0, 0.0, 1.0]]), tool_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), controller.down_quat)
    )
    torch.testing.assert_close(approach, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5, rtol=0)
    # At robosuite's home pose the hand is within a few degrees of straight down.
    tilt = torch.acos(transform_by_quat(torch.tensor([[0.0, 0.0, 1.0]]).expand(2, 3), rot)[:, 2].clamp(-1, 1))
    assert float(tilt.max()) < math.radians(10)


def test_the_wrist_window_spans_the_last_joints_range_less_the_margins(env):
    """Panda joint7 turns +-2.9 rad about a near-vertical axis: about 5.1 rad of heading once 0.35 rad is kept off each stop."""
    window = _tower(env).yaw_window()
    width = window[:, 1] - window[:, 0]
    assert float(width.min()) == pytest.approx(2 * 2.8973 - 2 * 0.35, abs=0.15)
    _, rot = _tower(env).tool_pose()
    yaw = yaw_from_quat(rot)
    assert bool(((window[:, 0] < yaw) & (yaw < window[:, 1])).all())


def test_act_commands_the_arm_and_gripper_terms_within_the_ik_squash(env):
    controller = _tower(env)
    controller.reset()
    action = controller.act()
    assert set(action) == {"robot_arm", "robot_gripper"}
    assert action["robot_arm"].shape == (2, 6) and action["robot_gripper"].shape == (2, 1)
    assert float(action["robot_arm"].abs().max()) <= math.atanh(0.9) + 1e-5
    # The first command of a plan is the hand's own pose: almost no motion asked for, gripper open.
    assert float(action["robot_arm"].abs().max()) < 0.05
    assert bool((action["robot_gripper"] < 0).all())
    env.reset()


# --- tower decisions -------------------------------------------------------------------------------


def _decide(controller, pos, yaw=(0.3, -0.7)):
    return controller.goal_subgoal(0, np.array(pos, dtype=np.float64), np.array(yaw, dtype=np.float64))


def test_the_top_block_goes_onto_the_one_below_aligned_with_it(env):
    subgoal = _decide(_tower(env), [_on_table(0.05, 0.05, H_B), _on_table(-0.05, -0.05, H_A)])
    assert (subgoal.object_index, subgoal.kind, subgoal.release) == (1, "goal", True)
    np.testing.assert_allclose(subgoal.goal_pos, [0.05, 0.05, TABLE_TOP_Z + 2 * H_B + H_A])
    assert subgoal.goal_yaw == pytest.approx(0.3)


def test_a_built_tower_is_complete(env):
    stacked = [_on_table(0.05, 0.05, H_B), [0.055, 0.05, TABLE_TOP_Z + 2 * H_B + H_A]]
    assert _decide(_tower(env), stacked) is None


def test_an_inverted_tower_is_taken_apart_first(env):
    """The base (cubeB) sits on cubeA: it is not on the table, so it is moved to a free spot clear of cubeA."""
    controller = _tower(env)
    inverted = [[0.0, 0.0, TABLE_TOP_Z + 2 * H_A + H_B], _on_table(0.0, 0.0, H_A)]
    subgoal = _decide(controller, inverted)
    assert (subgoal.object_index, subgoal.kind) == (0, "clear")
    assert subgoal.goal_pos[2] == pytest.approx(TABLE_TOP_Z + H_B)
    assert np.linalg.norm(subgoal.goal_pos[:2]) >= controller.min_spot_distance


def test_with_certain_mistakes_every_subgoal_is_a_mistake(env):
    controller = _tower(env, p_mistake=1.0)
    layout = np.array([[_on_table(0.05, 0.05, H_B), _on_table(-0.05, -0.05, H_A)]] * 2)
    subgoals = controller.choose_subgoals([0, 1], ObjectSnapshot(pos=layout, yaw=np.zeros((2, 2))))
    assert [sg.kind for sg in subgoals] == ["mistake", "mistake"]


# --- lift decisions --------------------------------------------------------------------------------


def test_a_lift_holds_the_block_above_the_table_until_it_is_up(env):
    lift = LiftController(env, ["cubeA"], **{**BOUNDS, "p_mistake": 0.0})
    resting = lift.goal_subgoal(0, np.array([_on_table(0.0, 0.0, H_A)]), np.array([0.2]))
    assert (resting.release, resting.kind) == (False, "goal")
    assert 0.1 <= resting.goal_pos[2] - (TABLE_TOP_Z + H_A) <= 0.2
    held = np.array([[0.0, 0.0, TABLE_TOP_Z + H_A + 0.12]])
    assert lift.goal_subgoal(0, held, np.array([0.2])) is None
