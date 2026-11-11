"""The proximity metrics in ``eden/managers/terms/metrics/proximity.py``."""

from __future__ import annotations

import pytest
import torch

from eden.constants import MetricDirection
from eden.managers.metric_manager import MetricManager, MetricTermFuncWrapper
from eden.managers.terms.metrics.proximity import (
    EeNearEntity,
    entity_distance,
    entity_distance_xy,
    entity_near_entity_xy,
    entity_near_target,
    object_a_within_b_xy,
)
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions

pytestmark = pytest.mark.requires_genesis

FINGERS = ["left_finger", "right_finger"]


@pytest.fixture
def place(panda_env, restore_state):
    """``place(a_rows, b_rows)``: put ``cube_a`` / ``cube_b`` at the given per-env positions."""

    def put(a, b=None):
        restore_state(panda_env.entities["cube_a"]).set_pos(torch.tensor(a))
        if b is not None:
            restore_state(panda_env.entities["cube_b"]).set_pos(torch.tensor(b))

    return put


def _fingertip_midpoint(robot):
    _, idx = robot.find_named_links_idx_local(FINGERS)
    return robot.get_links_pos(ls_idx_local=idx).mean(dim=1)


def test_entity_distance_is_the_3d_distance(panda_env, place):
    place([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(
        entity_distance(panda_env, entity_a_name="cube_a", entity_b_name="cube_b"), torch.tensor([5.0, 0.0])
    )
    torch.testing.assert_close(
        entity_distance(panda_env, entity_a_name="cube_a", entity_b_name="cube_a"), torch.zeros(2)
    )


def test_entity_distance_xy_ignores_z(panda_env, place):
    place([[0.0, 0.0, 0.0], [0.5, 0.3, 0.15]], [[3.0, 4.0, 99.0], [0.5, 0.3, 0.05]])  # env 1: apple on plate
    torch.testing.assert_close(
        entity_distance_xy(panda_env, entity_a_name="cube_a", entity_b_name="cube_b"), torch.tensor([5.0, 0.0])
    )


def test_entity_near_entity_xy_compares_the_xy_distance(panda_env, place):
    place([[0.5, 0.3, 0.15], [0.0, 0.0, 0.0]], [[0.5, 0.3, 0.05], [0.1, 0.0, 0.0]])
    near = entity_near_entity_xy(panda_env, entity_a_name="cube_a", entity_b_name="cube_b", threshold=0.05)
    assert near.tolist() == [True, False]


def test_entity_near_target_compares_the_distance_to_a_fixed_point(panda_env, place):
    place([[0.5, 0.0, 0.3], [10.0, 10.0, 10.0]])
    near = entity_near_target(panda_env, entity_name="cube_a", target_pos=[0.5, 0.0, 0.3], threshold=0.05)
    assert near.tolist() == [True, False]


@pytest.mark.parametrize("direction", [None, MetricDirection.LIB], ids=["registered-default", "explicit"])
@pytest.mark.parametrize("path", ["options", "configure"])
def test_entity_distance_succeeds_when_close(panda_env, place, install_manager, path, direction):
    """``entity_distance`` is registered lower-is-better, so a task naming it succeeds on a small distance."""
    fields = {} if direction is None else {"direction": direction}
    params = {"entity_a_name": "cube_a", "entity_b_name": "cube_b"}
    if path == "options":
        dist = MetricTermOptions(name="entity_distance", params=params, success_threshold=0.1, **fields)
    else:
        dist = MetricTermFuncWrapper.configure(func=entity_distance, params=params, success_threshold=0.1, **fields)
    manager = install_manager(panda_env, "metric_manager", MetricManager(panda_env, MetricManagerOptions(dist=dist)))
    place([[0.0, 0.0, 0.0]] * 2, [[0.01, 0.0, 0.0], [100.0, 0.0, 0.0]])
    manager.compute()
    assert manager.success_buf.tolist() == [True, False]


def test_ee_near_entity_compares_the_link_midpoint_to_the_entity(panda_env, make_term, place):
    midpoint = _fingertip_midpoint(panda_env.entities["robot"])
    place((midpoint + torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.0, 0.2]])).tolist())
    term = make_term(EeNearEntity, panda_env, robot_name="robot", ee_link_names=FINGERS, entity_name="cube_a")
    assert term.compute().tolist() == [True, False]


def test_ee_near_entity_rejects_an_unknown_link_at_build(panda_env, make_term):
    with pytest.raises(ValueError, match="finger_typo"):
        make_term(EeNearEntity, panda_env, robot_name="robot", ee_link_names=["finger_typo"], entity_name="cube_a")


@pytest.mark.parametrize(("dx", "expected"), [(0.0, True), (0.009, True), (0.011, False), (0.2, False)])
def test_object_a_within_b_xy_is_the_share_of_a_footprint_over_b(panda_env, place, dx, expected):
    """Two 5 cm cubes offset by ``dx``: ``(5 - dx) / 5`` of cube_a's footprint lies over cube_b's, whatever the heights."""
    place([[0.5 + dx, 0.0, 0.3]] * 2, [[0.5, 0.0, 0.1]] * 2)
    within = object_a_within_b_xy(panda_env, entity_a_name="cube_a", entity_b_name="cube_b", min_fraction=0.8)
    assert within.tolist() == [expected] * 2
