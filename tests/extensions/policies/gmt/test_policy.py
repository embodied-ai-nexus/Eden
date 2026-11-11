"""``gmt_g1_robot``: the ``**overrides`` hatch reaches the robot without clobbering the trained values."""

from __future__ import annotations

from eden.extensions.policies.gmt.policy import gmt_g1_robot


def test_overrides_reach_the_robot_and_keep_the_training_values():
    """e.g. ``file=`` for exact-MJCF fidelity."""
    stock = gmt_g1_robot()
    robot = gmt_g1_robot(file="custom_g1.xml")
    assert robot.file == "custom_g1.xml"
    assert robot.default_dofs_pos == stock.default_dofs_pos and robot.default_dofs_pos
    assert robot.default_dofs_armature == stock.default_dofs_armature and robot.default_dofs_armature
