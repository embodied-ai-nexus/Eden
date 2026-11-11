"""The articulation metrics in ``eden/managers/terms/metrics/articulation.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.metrics.articulation import joint_angle_at_target

pytestmark = pytest.mark.requires_genesis


@pytest.mark.parametrize(
    ("joint1", "target", "threshold", "expected"),
    [
        ([1.57, 0.0], 1.57, 0.05, [True, False]),
        ([0.52, 0.99], 0.5, 0.05, [True, False]),  # within the threshold either side
        ([0.99, 1.01], 1.0, 0.05, [True, True]),
        ([0.0, 0.0], 1.0, 0.1, [False, False]),
    ],
)
def test_joint_angle_at_target_is_within_the_threshold(panda_env, restore_state, joint1, target, threshold, expected):
    robot = restore_state(panda_env.entities["robot"])
    dofs_pos = robot.get_dofs_pos().clone()
    dofs_pos[:, 0] = torch.tensor(joint1)
    robot.set_dofs_pos(dofs_pos)
    result = joint_angle_at_target(
        panda_env, entity_name="robot", joint_name="joint1", target_angle=target, threshold=threshold
    )
    assert result.tolist() == expected
