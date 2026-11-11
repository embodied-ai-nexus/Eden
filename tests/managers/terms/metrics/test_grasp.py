"""The grasp metrics in ``eden/managers/terms/metrics/grasp.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.metrics.grasp import object_lifted

pytestmark = pytest.mark.requires_genesis


def test_object_lifted_is_strictly_above_the_minimum_height(box_env, restore_state):
    box = restore_state(box_env.entities["robot"])
    box.set_pos(torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.1], [0.0, 0.0, 0.3], [0.0, 0.0, 0.31]]))
    assert object_lifted(box_env, entity_name="robot", min_height=0.3).tolist() == [True, False, False, True]
