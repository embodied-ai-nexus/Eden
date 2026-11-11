"""UR5e / UR10e + Robotiq 2F85 composite manipulators.

Universal Robots arms carrying the same pad-only Robotiq 2F85 as :class:`Droid`.
The gripper welds at identity onto the UR URDF's fixed ``ee_link`` flange frame, so
no offset is needed here; a rig that does need a tool-frame offset can set
``attach_offset`` / ``attach_quat`` on the composite (the Franka instead carries a dedicated
``attachment`` body). Registered under the same lowercase names the
``manipulation_lift_box --manipulator`` flag uses, so a task can swap the arm by name via
:data:`COMPOSITE_ROBOT_REGISTRY`.
"""

from __future__ import annotations

from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, ManipulatorOptions
from eden.options.robots.composite.droid import _droid_gripper
from eden.options.robots.universal_ur10e import UR10e
from eden.options.robots.universal_ur5e import UR5e

# Ready-to-grasp homes for the lift task: Robotiq pointing straight DOWN with the
# pad midpoint over the table centerline ~10 cm above the cube rest height. Two
# traps these encode: the UR option classes' own default pose points the tool
# *horizontally* (useless as a grasp start), and reusing the UR5e angles on the
# UR10e dips its longer forearm onto the table — each arm is IK-derived for its own
# geometry (ee_link to ~(0.30, 0, 0.80), tool-down). The negative shoulder_pan
# cancels the UR wrist's lateral (y) offset; wrist_3 cancels the resulting yaw.
_UR5E_HOME = {
    "shoulder_pan": -0.463,
    "shoulder_lift": -2.1226,
    "elbow": 2.1862,
    "wrist_1": -1.6344,
    "wrist_2": -1.5708,
    "wrist_3": -0.463,
}
_UR10E_HOME = {
    "shoulder_pan": -0.6193,
    "shoulder_lift": -2.3717,
    "elbow": 2.5309,
    "wrist_1": -1.7301,
    "wrist_2": -1.5708,
    "wrist_3": -0.6193,
}


def _ur5e_arm() -> RobotOptions:
    return UR5e(
        default_dofs_pos=dict(_UR5E_HOME),
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )


def _ur10e_arm() -> RobotOptions:
    return UR10e(
        default_dofs_pos=dict(_UR10E_HOME),
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )


@COMPOSITE_ROBOT_REGISTRY.register(name="ur5e_2f85")
class UR5eRobotiq(ManipulatorOptions):
    """UR5e + Robotiq 2F85, welded at the arm's ``ee_link`` flange.

    Registered ``ur5e_2f85``, not ``ur5e``: the bare arm already owns that key in
    ``ROBOT_REGISTRY``, and a rig named for its arm alone both collides with it and hides which
    end-effector is on. ``<arm>_<end-effector>`` is what the rest of this package already does
    (``franka_x_hand``, ``kuka_sharpa``, ``xarm7_xhand1_r``, ``oli_wuji``).
    """

    arm: RobotOptions = Field(default_factory=_ur5e_arm)
    gripper: RobotOptions = Field(default_factory=_droid_gripper)
    ee_link: str = "ee_link"
    grasp_links: list[str] = ["left_pad", "right_pad"]


@COMPOSITE_ROBOT_REGISTRY.register(name="ur10e_2f85")
class UR10eRobotiq(ManipulatorOptions):
    """UR10e + Robotiq 2F85, welded at the arm's ``ee_link`` flange (see :class:`UR5eRobotiq`)."""

    arm: RobotOptions = Field(default_factory=_ur10e_arm)
    gripper: RobotOptions = Field(default_factory=_droid_gripper)
    ee_link: str = "ee_link"
    grasp_links: list[str] = ["left_pad", "right_pad"]
