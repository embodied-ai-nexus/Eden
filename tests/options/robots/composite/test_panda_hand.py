"""``PandaFrankaHand``: the Panda as an arm + Franka Hand composite, swappable with the other manipulators."""

from __future__ import annotations

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, PandaFrankaHand


def test_the_rig_welds_the_franka_hand_at_identity_on_the_shared_flange():
    rig = PandaFrankaHand(base_pos=(-0.56, 0.0, 0.912), arm_name="robot0", gripper_name="robot0_gripper")
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {
        "robot0": "FrankaEmikaPandaNoHand",
        "robot0_gripper": "FrankaHand",
    }
    assert attachments == {"robot0_gripper": ("robot0", "attachment")}
    assert rig.gripper_dofs == ["finger_joint1", "finger_joint2"]
    assert tuple(entities["robot0"].default_root_pos) == (-0.56, 0.0, 0.912)
    assert COMPOSITE_ROBOT_REGISTRY.get("panda_franka_hand") is PandaFrankaHand
