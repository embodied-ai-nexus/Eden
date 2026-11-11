"""The UR5e / UR10e manipulators: the Droid's pad-only Robotiq on a UR arm, welded at the UR flange."""

from __future__ import annotations

import pytest

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, Droid, UR5eRobotiq, UR10eRobotiq


@pytest.mark.parametrize("name, cls, arm", [("ur5e_2f85", UR5eRobotiq, "UR5e"), ("ur10e_2f85", UR10eRobotiq, "UR10e")])
def test_the_rig_welds_the_gripper_at_the_ur_flange(name, cls, arm):
    rig = cls(base_pos=(0.0, 0.0, 0.4))
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {"robot": arm, "gripper": "Robotiq2f85"}
    assert attachments == {"gripper": ("robot", "ee_link")}
    assert len(rig.arm_dofs) == 6 and rig.grasp_links == ["left_pad", "right_pad"]
    assert COMPOSITE_ROBOT_REGISTRY.get(name) is cls


def test_every_manipulator_builds_the_same_gripper():
    """The shared factory keeps the gripper, and everything keyed off it, embodiment-invariant."""
    droid, ur = Droid().gripper, UR5eRobotiq().gripper
    assert type(droid) is type(ur)
    assert droid.collision_link_patterns == ur.collision_link_patterns == [".*pad"]
    assert (droid.open_dofs_pos, droid.close_dofs_pos) == (ur.open_dofs_pos, ur.close_dofs_pos)
