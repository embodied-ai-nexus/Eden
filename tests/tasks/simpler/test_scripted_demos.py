"""The SIMPLER demonstrators on the real WidowX scenes: a robot with its own fingers, driven at 5 Hz.

One cube-stacking build and one ``simpler_put_eggplant_in_basket`` build (two envs each, MPR collision — see
``docs/guides/scripted-demos.md`` on GJK) serve the tests.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.extensions.scripted_demos.block_world import TowerController
from eden.extensions.scripted_demos.primitives import CONTACT_GATE_SLOTS, GATE_SLOTS, tool_quat
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.simpler.bridge_sink.config import BASKET_REGION, BASIN_FLOOR_Z
from eden.utils.geom import transform_by_quat
from tests.support.envs import build_env

TASK = "simpler_stack_green_cube_on_yellow_cube"


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build(TASK)
    config.env_options.num_envs = 2
    config.env_options.use_gjk_collision = False
    return build_env(config, reset=True)


@pytest.fixture
def controller(env):
    env.reset()
    controller = SCRIPTED_DEMO_REGISTRY.get(TASK)(env, seed=0)
    controller.reset()
    return controller


def test_the_task_names_its_action_terms_its_own_way_and_they_are_found_by_what_they_drive(controller):
    assert isinstance(controller, TowerController)
    action = controller.act()
    assert set(action) == {"ee_controller", "gripper_controller"}
    assert action["ee_controller"].shape == (2, 6)


def test_the_widowx_approaches_along_its_gripper_links_x_axis(controller):
    """``tcp_offset`` is along +x of ``gripper_link``: pointing straight down puts that axis on -z."""
    straight_down = tool_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), controller.down_quat)
    approach = transform_by_quat(torch.tensor([[1.0, 0.0, 0.0]]), straight_down)
    torch.testing.assert_close(approach, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-5, rtol=0)


def test_the_tool_floor_keeps_the_fingertips_off_the_ground(controller):
    """The fingers reach ~1.9 cm past the pads — more than half a 3 cm cube — so the pads stop above its centre."""
    assert controller.tip_reach == pytest.approx(0.0187, abs=0.002)
    floor = float(
        controller._workspace[0][2]
    )  # the clipped floor is what the rig is sent to; nothing public exposes it
    assert floor == pytest.approx(0.002 + controller.tip_reach)
    assert floor > 0.015  # above the cube centre a centred grasp would aim at


def test_the_gripper_closes_only_once_the_hand_is_at_the_grasp_pose(controller, env):
    """At 5 Hz the WidowX trails a descent by several cm; the contact gate holds the plan until it arrives.

    Every time an env's gripper command flips to closed, its tool point is within reach of the grasp
    pose it was sent to: the 1 cm gate tolerance plus the few mm the plan moves after it (the contact
    offsets). Measured against that pose rather than the object, which a finger may have nudged.
    """
    grasp_gate = GATE_SLOTS.index(CONTACT_GATE_SLOTS[0])  # pick_start
    closed = torch.zeros(2, dtype=torch.bool)
    closings = 0
    for _ in range(150):
        controller.update_subgoals()
        action = controller.act()
        now_closed = (action["gripper_controller"][:, 0] > 0).cpu()
        tool, _ = controller.tool_pose()
        # private: the grasp pose the hand was sent to is the gate the contract is about
        grasp_pose = controller._gate_pos[:, grasp_gate]
        for env_id in (now_closed & ~closed).nonzero().flatten().tolist():
            closings += 1
            assert float((tool[env_id] - grasp_pose[env_id]).norm()) < 0.025, env_id
        closed = now_closed
        env.step(action)
    assert closings > 0, "no grasp in 30 s of rollout: the test checked nothing"
    env.reset()


def test_a_held_cube_is_not_judged_dropped_while_the_lagging_hand_lifts_it(controller, env):
    """The pick check waits for the hand to rise and asks whether the cube came along.

    Checked at plan time ``postpick`` instead (OCBench), a hand trailing the plan by ~4 cm has not lifted
    the cube 2.5 cm yet, so good grasps read as failed and were redone. On this seed every grasp holds.
    """
    grasps = 0
    closed = torch.zeros(2, dtype=torch.bool)
    for _ in range(100):
        controller.update_subgoals()
        action = controller.act()
        now_closed = (action["gripper_controller"][:, 0] > 0).cpu()
        grasps += int((now_closed & ~closed).sum())
        closed = now_closed
        env.step(action)
    assert grasps >= 2, "fewer than one grasp per env in 20 s: the test checked nothing"
    # private: the retry counter is the only record of a pick judged failed
    assert controller._num_retries.tolist() == [0, 0]
    env.reset()


# --- the sink scene ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sink_env():
    config = TASK_REGISTRY.build("simpler_put_eggplant_in_basket")
    config.env_options.num_envs = 2
    config.env_options.use_gjk_collision = False
    return build_env(config, reset=True)


def test_the_eggplant_starts_on_the_basin_floor(sink_env):
    for seed in range(3):
        sink_env.reset(seed=seed)
        pos = sink_env.entities["source"].get_pos()
        # On the basin floor (its centre ~2 cm up), in the basin's +y half, and not in the basket.
        assert ((pos[:, 2] - BASIN_FLOOR_Z) < 0.03).all(), pos
        assert ((pos[:, 1] > -0.17) & (pos[:, 1] < -0.08)).all(), pos
        assert not sink_env.metric_manager.success_buf.any()


def test_the_routine_puts_the_eggplant_in_the_basket(sink_env):
    sink_env.reset(seed=0)
    demo = SCRIPTED_DEMO_REGISTRY.get("simpler_put_eggplant_in_basket")(sink_env, seed=0, p_mistake=0.0)
    demo.reset()
    for _ in range(250):
        if bool(demo.update_subgoals().all()):
            break
        sink_env.step(demo.act())
    assert demo.done.all()
    assert sink_env.metric_manager.success_buf.all()
    low, high = BASKET_REGION
    pos = sink_env.entities["source"].get_pos()
    assert ((pos[:, 0] > low[0]) & (pos[:, 0] < high[0]) & (pos[:, 1] > low[1]) & (pos[:, 1] < high[1])).all()
