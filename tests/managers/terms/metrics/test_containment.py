"""The containment metrics in ``eden/managers/terms/metrics/containment.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.metrics.containment import object_above_height, object_bottom_above_height, object_in_region

pytestmark = pytest.mark.requires_genesis


def _place(env, restore_state, rows):
    restore_state(env.entities["robot"]).set_pos(torch.tensor(rows))


def test_object_above_height_is_strictly_above(box_env, restore_state):
    _place(box_env, restore_state, [[0.0, 0.0, 1.5], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
    assert object_above_height(box_env, entity_name="robot", height=1.0).tolist() == [True, False, False, True]


def test_object_in_region_includes_its_boundary(box_env, restore_state):
    _place(box_env, restore_state, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    inside = object_in_region(box_env, entity_name="robot", region_min=[0.0, 0.0, 0.0], region_max=[1.0, 1.0, 1.0])
    assert inside.tolist() == [True, False, True, False]


def test_object_bottom_above_height_reads_the_lowest_point_not_the_origin(panda_env, restore_state):
    """A 5 cm cube centred at z clears ``height`` once z - 2.5 cm does, whatever its origin."""
    restore_state(panda_env.entities["cube_a"]).set_pos(torch.tensor([[1.0, 1.0, 0.04], [1.0, 1.0, 0.06]]))
    lifted = object_bottom_above_height(panda_env, entity_name="cube_a", height=0.025)
    assert lifted.tolist() == [False, True]
