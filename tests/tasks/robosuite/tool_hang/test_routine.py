"""The ToolHang routine: the held-object geometry it turns the hand by, the ring's drop pose, and real runs."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import numpy as np
import pytest
from genesis.utils.geom import axis_angle_to_R

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.tool_hang.routine import (
    BAR_FREE_END_HEADING,
    FRAME_APPROACH,
    FRAME_UPRIGHT,
    RING_DROP,
    RING_LEAN,
    RING_SEGMENT_RADIUS,
    Grasp,
    lean,
    wrench_hanging,
)
from tests.support.envs import build_env

UP = np.array((0.0, 0.0, 1.0))
#: The hole1 ring's segments (``tool_hang_wrench.xml``): half their radial depth and half their thickness.
SEGMENT_RADIAL_HALF = 0.0032336
SEGMENT_AXIAL_HALF = 0.0035
#: The hook's lip (``tool_hang_frame.xml``): how far it runs in from the bar's free end, and its top above
#: the bar's axis.
LIP_LENGTH = 0.0075
LIP_TOP = 0.01575


def test_a_grasp_turns_the_object_with_the_hand():
    hand = axis_angle_to_R(np.array((0.0, 0.0, 1.0)), np.array(0.4))
    object_rot = axis_angle_to_R(np.array((1.0, 0.0, 0.0)), np.array(0.3)) @ hand
    tool, object_pos = np.array((0.1, -0.2, 1.0)), np.array((0.12, -0.18, 0.97))
    grasp = Grasp("tool", hand.T @ object_rot, hand.T @ (object_pos - tool))

    # Asked for where it already is, the hand stays where it is, and so does the tool point.
    np.testing.assert_allclose(grasp.hand_for(object_rot), hand, atol=1e-12)
    site = np.array((-0.09, 0.0, 0.0))
    np.testing.assert_allclose(grasp.tool_for(hand, site, object_pos + object_rot @ site), tool, atol=1e-12)
    # Asked for a quarter turn about the vertical, the hand turns by it.
    turn = axis_angle_to_R(np.array((0.0, 0.0, 1.0)), np.array(math.pi / 2))
    np.testing.assert_allclose(grasp.hand_for(turn @ object_rot), turn @ hand, atol=1e-12)


def test_the_hanging_wrench_has_its_handle_down_and_its_ring_axis_back_along_the_bar():
    along = np.array((math.cos(0.7), math.sin(0.7), 0.0))
    rot = wrench_hanging(along)
    np.testing.assert_allclose(rot @ np.array((1.0, 0.0, 0.0)), -UP, atol=1e-12)  # handle down, ring on top
    np.testing.assert_allclose(rot @ np.array((0.0, 0.0, 1.0)), -along, atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    # Leaned, its ring's up direction tips toward the rod by the lean.
    leaned = lean(rot, along, RING_LEAN)
    ring_up = leaned @ np.array((-1.0, 0.0, 0.0))
    np.testing.assert_allclose(ring_up, math.cos(RING_LEAN) * UP + math.sin(RING_LEAN) * along, atol=1e-12)


def test_the_upright_frame_is_held_with_the_hand_level_and_across_its_bar():
    free_end = -FRAME_UPRIGHT[:, 0]
    assert math.atan2(free_end[1], free_end[0]) == pytest.approx(BAR_FREE_END_HEADING)
    np.testing.assert_allclose(FRAME_UPRIGHT[:, 2], UP, atol=1e-12)  # rod up
    assert FRAME_APPROACH[2] == pytest.approx(0.0, abs=1e-12)
    assert FRAME_APPROACH @ free_end == pytest.approx(0.0, abs=1e-12)


def test_the_ring_is_let_go_with_its_top_past_the_lip_and_its_bottom_short_of_the_bars_end():
    """The drop pose clears the lip both ways, the margin the routine's threading rests on.

    In the bar's frame (along the bar from its free end, up from its axis), with the ring leaned by
    ``RING_LEAN``: its top segment must be wholly past the lip and above it, its bottom segment wholly
    short of the bar's end — the bar and lip (19.5 mm) do not fit through the ring's hole (19.4 mm) at once.
    """
    along_up = np.array((math.sin(RING_LEAN), math.cos(RING_LEAN)))  # the ring's up, (along, up)
    along_axis = np.array((math.cos(RING_LEAN), -math.sin(RING_LEAN)))  # its axis, toward the free end's side
    top = np.array(RING_DROP)
    bottom = top - 2 * RING_SEGMENT_RADIUS * along_up

    def extent(centre, component):
        """Smallest and largest ``component`` (0 along, 1 up) of a segment's cross-section."""
        reach = SEGMENT_RADIAL_HALF * abs(along_up[component]) + SEGMENT_AXIAL_HALF * abs(along_axis[component])
        return centre[component] - reach, centre[component] + reach

    assert extent(top, 0)[0] > LIP_LENGTH + 0.001
    assert extent(top, 1)[0] > LIP_TOP + 0.002
    assert extent(bottom, 0)[1] < -0.003


def test_the_demonstrator_gets_the_time_it_needs():
    # The frame takes ~60 s and the wrench ~40 s more, past the generator's 90 s default.
    assert SCRIPTED_DEMO_REGISTRY.get("robosuite_tool_hang").episode_length_s == 180.0


@pytest.mark.slow
def test_the_routine_assembles_the_frame_and_hangs_the_wrench():
    config = TASK_REGISTRY.build("robosuite_tool_hang")
    config.env_options.num_envs = 2
    config.env_options.episode_length_s = 150.0  # the routine takes ~100 s; robosuite's RL episodes are shorter
    env = build_env(config)
    env.reset(seed=0)
    demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_tool_hang")(env, seed=0)
    demo.reset()
    for _ in range(2600):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())

    assert demo.done.all()
    metrics = env.metric_manager
    assert (metrics.get_value("frame_assembled") > 0.5).all()
    assert (metrics.get_value("tool_on_frame") > 0.5).all()
    assert metrics.success_buf.all()
    for env_id in range(2):
        assert any("wrench: let go" in line for line in demo.trace(env_id))
