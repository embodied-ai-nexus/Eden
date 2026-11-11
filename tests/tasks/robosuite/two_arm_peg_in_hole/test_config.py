"""TwoArmPegInHole: the peg and plate are welded into the grippers of whichever rigs hold them."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest

from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.rigs import ROBOT_NAMES, RobosuiteRobotMod, rig_name, rig_of
from eden.tasks.robosuite.two_arm_peg_in_hole.config import _GRIP_DEPTH, _PEG_LENGTH


def _rigs(cfg):
    return [rig_of(cfg, slot) for slot in type(cfg).robot_slots]


@pytest.mark.parametrize("robot", ROBOT_NAMES)
@pytest.mark.parametrize("arrangement", ["opposed", "parallel"])
def test_objects_sit_in_the_grippers(robot, arrangement):
    """Both objects hang off the tool point (pad midpoint) with the fingers closed on them.

    robosuite mounts the peg / plate on a bare flange; Eden's rigs have grippers there.
    """
    cfg = TASK_REGISTRY.build("robosuite_two_arm_peg_in_hole", robot=robot, arrangement=arrangement)
    rig0, rig1 = _rigs(cfg)
    peg, hole = cfg.scene_options.attachments_dict["peg"], cfg.scene_options.attachments_dict["hole"]
    assert (peg[0], peg[1]) == (rig0.arm_name, rig0.ee_link)
    assert (hole[0], hole[1]) == (rig1.arm_name, rig1.ee_link)
    # The peg's near end and the plate's near edge (17 cm behind its hole) both reach _GRIP_DEPTH past the pads.
    assert peg[2][2] - _PEG_LENGTH / 2 == pytest.approx(rig0.tcp_offset[2] - _GRIP_DEPTH)
    assert hole[2][2] - 0.17 == pytest.approx(rig1.tcp_offset[2] - _GRIP_DEPTH)
    assert hole[2][0] == pytest.approx(0.11)  # robosuite's lateral offset puts the hole on the hand axis
    for rig, width in ((rig0, 0.04), (rig1, 0.02)):
        defaults = rig.gripper.default_dofs_pos
        for dof, value in rig.hold_width(width).items():
            assert defaults[dof] == pytest.approx(value), f"{robot} {rig.arm_name}: fingers not preset on the object"
            assert value != rig.gripper_open[dof]


def test_robot_mod_alone_rewelds():
    """The robot mod must not rely on the arrangement mod running after it to fix the welds."""
    cfg = RobosuiteRobotMod("droid").apply(TASK_REGISTRY.build("robosuite_two_arm_peg_in_hole"))
    rig0, rig1 = _rigs(cfg)
    assert rig_name(rig0) == "droid"
    assert cfg.scene_options.attachments_dict["peg"][1] == rig0.ee_link == "attachment"
    assert cfg.scene_options.attachments_dict["hole"][1] == rig1.ee_link == "attachment"
    assert rig0.gripper.default_dofs_pos["left_driver_joint"] > 0.0
