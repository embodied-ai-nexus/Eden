"""``Droid``: a Franka arm and a pad-only Robotiq 2F-85 welded at the dedicated ``attachment`` body."""

from __future__ import annotations

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, Droid, FrankaXHand


def test_the_rig_is_a_panda_arm_with_a_robotiq_on_the_attachment_body():
    droid = Droid()
    entities, attachments = droid.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {
        "robot": "FrankaEmikaPandaNoHand",
        "gripper": "Robotiq2f85",
    }
    assert attachments == {"gripper": ("robot", "attachment")}
    assert droid.arm_dofs == [f"joint{i}" for i in range(1, 8)]
    assert droid.grasp_links == ["left_pad", "right_pad"]
    assert droid.gripper_open == {"left_driver_joint": 0.0, "right_driver_joint": 0.0}
    assert droid.gripper_dofs == list(droid.gripper_open) == list(droid.gripper_close)
    assert COMPOSITE_ROBOT_REGISTRY.get("droid") is Droid


def test_droid_and_franka_xhand_share_one_franka_home_pose():
    """Imported, not copied, so the arm cannot drift between the gripper and dexterous variants."""
    assert Droid().arm.default_dofs_pos == FrankaXHand().arm.default_dofs_pos
