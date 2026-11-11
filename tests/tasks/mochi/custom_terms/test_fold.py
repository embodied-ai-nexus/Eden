"""Tests for the fold measures, RoboSimStudio's feature latch and letting go, on ``mochi_fold_the_towel``."""

import math

import pytest
import torch

from eden.tasks.mochi.common import YAM_FINGER_SPEED, YAM_FINGERS_OPEN
from eden.tasks.mochi.fold_the_towel import config as fold_config
from tests.tasks.mochi.conftest import action, build_task, place_tcp, tool_centre


@pytest.fixture(scope="module")
def env():
    return build_task("mochi_fold_the_towel")


def _fold_terms(env):
    return env.metric_manager.get_term("fold_score"), env.metric_manager.get_term("fold_extent")


def _close_fully(env) -> None:
    """Step with the left jaws told to close until their travel from open is done: the latch seizes then."""
    for _ in range(math.ceil(YAM_FINGERS_OPEN["left_finger"] / YAM_FINGER_SPEED / env.dt) + 1):
        env.step(action(env, close={"left_gripper": True}))


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_perfect_fold_scores_one_and_halves_the_extent(env):
    """Env 0 lays the far half exactly onto the near half one layer up; env 1 stays flat where it started."""
    env.reset()
    towel = env.entities["towel"]
    score, extent = _fold_terms(env)
    geometry = score._fold
    verts = towel.get_vertex_positions().clone()
    folded = verts[0].clone()
    layer = torch.tensor([0.0, 0.0, fold_config.LAYER_THICKNESS], device=verts.device)
    folded[geometry.moving] = verts[0, geometry.pairs] + layer
    verts[0] = folded
    towel.set_vertex_positions(verts)
    torch.testing.assert_close(score.compute(), torch.tensor([1.0, 0.0], device=verts.device), atol=1e-4, rtol=0)
    torch.testing.assert_close(extent.compute(), torch.tensor([0.5, 1.0], device=verts.device), atol=1e-3, rtol=0)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_dragging_the_near_half_costs_one_point_two_times_its_drift(env):
    env.reset()
    towel = env.entities["towel"]
    score, _ = _fold_terms(env)
    rest_gap, layer = score._fold.rest_gap, score.layer_thickness
    verts = towel.get_vertex_positions().clone()
    verts[1, :, 0] += 0.02  # the whole flat towel slides 2 cm: its near half drifted 2 cm, its gaps unchanged
    towel.set_vertex_positions(verts)
    expected = 1.0 - (rest_gap - layer + 1.2 * 0.02) / (rest_gap - layer)
    assert float(score.compute()[1]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_latch_seizes_the_edge_strip_in_reach_and_nothing_else(env):
    """Env 0 closes on the far edge where the demo pinches it; env 1 closes on the towel's middle, off the strip."""
    env.reset()
    towel = env.entities["towel"]
    latch = env.action_manager.get_term("left_gripper")
    verts = towel.get_vertex_positions()
    edge = verts[:, fold_config.EDGE_GRASP[2]]
    middle = verts.mean(dim=1)
    place_tcp(
        env, "left", torch.stack([edge[0], middle[1]]) + torch.tensor([0.0, 0.0, 0.012], device=env.device), yaw=1.5708
    )
    env.step(action(env, close={"left_gripper": True}))
    assert latch.is_latched.tolist() == [False, False]  # told to close, but the jaws are still on their way
    _close_fully(env)
    assert latch.is_latched.tolist() == [True, False]
    verts = towel.get_vertex_positions()
    pinned = latch._pinned[0][latch._pinned[0] >= 0].tolist()
    assert set(pinned) <= set(fold_config.GRASP_STRIP)
    assert float((verts[0, pinned] - tool_centre(env, "left")[0]).norm(dim=-1).max()) <= latch.radius + 1e-3
    assert env.metric_manager.get_value("holds").tolist() == [1.0, 0.0]

    lift = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    for _ in range(30):
        env.step(action(env, close={"left_gripper": True}, move={"left_hand": lift}))
    edge_now = towel.get_vertex_positions()[:, fold_config.EDGE_GRASP[2]]
    assert float(edge_now[0, 2] - edge[0, 2]) > 0.03  # carried up with the hand
    env.step(action(env, close={"left_gripper": False}))
    assert latch.is_latched.tolist() == [False, False]


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_reset_releases_the_latch(env):
    env.reset()
    towel = env.entities["towel"]
    latch = env.action_manager.get_term("left_gripper")
    edge = towel.get_vertex_positions()[:, fold_config.EDGE_GRASP[2]]
    place_tcp(env, "left", edge + torch.tensor([0.0, 0.0, 0.012], device=env.device), yaw=1.5708)
    _close_fully(env)
    assert latch.is_latched.tolist() == [True, True]
    env.reset(envs_idx=torch.tensor([1], device=env.device))
    assert latch.is_latched.tolist() == [True, False]


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_hands_clear_is_the_nearest_tool_centre_to_the_towel_and_a_flat_towel_is_still(env):
    env.reset()
    towel = env.entities["towel"]
    verts = towel.get_vertex_positions()
    tools = torch.stack([tool_centre(env, "left"), tool_centre(env, "right")], dim=1)
    expected = torch.cdist(tools, verts).amin(dim=(-1, -2))
    torch.testing.assert_close(env.metric_manager.get_term("hands_clear").compute(), expected)
    for _ in range(5):
        env.step(action(env))
    assert (env.metric_manager.get_value("settled") <= 0.02).all()
