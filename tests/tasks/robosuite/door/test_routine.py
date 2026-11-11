"""The door-opening routine: the arc it swings the hand along, and real runs on both robosuite doors."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import numpy as np
import pytest
import torch

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.door.routine import OPEN_ANGLE, arc, swing
from eden.utils.geom import quat_from_axis_angle
from tests.support.envs import build_env


def test_swing_turns_a_point_about_a_line_through_a_centre():
    centre, up = np.array([1.0, 2.0, 0.5]), np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(swing(centre + (0.3, 0.0, 0.2), centre, up, math.pi / 2), centre + (0.0, 0.3, 0.2))
    # About a horizontal axis the height changes and the offset along the axis does not.
    side = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(swing(centre + (0.1, 0.4, 0.0), centre, side, -math.pi / 2), centre + (0.0, 0.4, 0.1))


def test_the_arc_ends_at_the_full_angle_in_equal_steps_and_turns_the_hand_with_it():
    centre, up = np.zeros(3), np.array([0.0, 0.0, 1.0])
    waypoints = arc(np.array([0.31, 0.0, 1.0]), (1.0, 0.0, 0.0, 0.0), centre, up, 0.38, 0.1)
    assert len(waypoints) == 4  # 0.38 rad in steps of at most 0.1
    point, rot = waypoints[-1]
    np.testing.assert_allclose(point, (0.31 * math.cos(0.38), 0.31 * math.sin(0.38), 1.0), atol=1e-12)
    expected = quat_from_axis_angle(torch.tensor([[0.0, 0.0, 0.38]]))[0]
    np.testing.assert_allclose(rot, expected.numpy(), atol=1e-6)
    gaps = [np.linalg.norm(b[0] - a[0]) for a, b in zip(waypoints, waypoints[1:])]
    assert max(gaps) - min(gaps) < 1e-12


@pytest.mark.parametrize("latch", ["latch", "no_latch"])
def test_the_routine_opens_the_door_past_robosuites_success_angle(latch):
    config = TASK_REGISTRY.build("robosuite_door", latch=latch)
    config.env_options.num_envs = 2
    config.env_options.episode_length_s = 60.0  # the routine takes ~30 s; robosuite's RL episodes are shorter
    env = build_env(config)
    env.reset(seed=0)
    door = env.entities["door"]
    # robosuite's hinge range (0, 0.4) is in radians, as the file is compiled.
    assert float(door.get_dofs_limit()[1][0, 0]) == pytest.approx(0.4)

    demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_door")(env, seed=0, p_mistake=0.0)
    demo.reset()
    for _ in range(900):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())

    assert demo.done.all()
    hinge = door.get_dofs_pos(dofs_idx_local=door.find_named_dofs(["hinge"]).dofs_idx_local)[:, 0]
    assert (hinge > OPEN_ANGLE).all(), hinge
    assert env.metric_manager.success_buf.all()
    for env_id in range(2):
        assert any("door open at" in line for line in demo.trace(env_id))
