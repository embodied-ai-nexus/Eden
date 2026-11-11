"""Franka Emika Panda and Franka Research 3 arm configurations."""

from __future__ import annotations

from typing import ClassVar

from genesis.typing import Vec3FType

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import MetadataOptions, RobotOptions


@ROBOT_REGISTRY.register()
class FrankaResearch3(RobotOptions):
    """The Franka Research 3, Franka Robotics' successor to the Panda.

    From MuJoCo Menagerie's ``franka_fr3`` (Apache-2.0), which derives it from Franka's own
    ``franka_description``, with two Eden edits baked into the shipped asset: its links and joints
    are renamed ``link0``…``link7`` / ``joint1``…``joint7`` (the naming every other arm here uses,
    and what this class's DOF maps and the composite rigs address), and the bare
    ``attachment_site`` becomes an ``attachment`` *body* carrying the same mount frame as
    ``panda_nohand.xml`` — 0.107 m up from ``link7``, rotated 135 degrees about z — so a gripper
    welds onto either Franka arm unchanged.

    The FR3 is the same arm mechanically as the Panda (identical link frames, identical collision
    hulls, 855 mm reach, 3 kg payload, ±87 / ±12 N·m joint torques), and differs in what it
    declares: its own joint limits (A6 is shifted by 26 degrees, so a Panda home pose is not
    automatically reachable), Franka's published inertials, and joint friction identified in
    torque control (``armature`` 0.195 / 0.074, ``damping`` 0.21, ``frictionloss`` 1.137 … 0.248)
    where the Panda model has none.
    """

    # Not ``arms/franka_research3/fr3.xml``: that file in the asset dataset is a mislabelled
    # hand-less Panda.
    file: str = "arms/franka_fr3/fr3.xml"

    is_fixed_base: bool = True

    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)  # x,y,z [m]
    links_to_keep: list[str] = ["attachment"]

    # The FR3 ships no gripper, so its tool frame is the bare flange body `attachment`
    # — the same link the composite rigs weld an end-effector onto (``Droid.ee_link``).
    ee_links_name: ClassVar[list[str]] = ["attachment"]
    #: The seven arm joints in MuJoCo XML declaration order (the order DROID policies emit).
    arm_dofs_name: ClassVar[list[str]] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -2.27,
        "joint5": 0.0,
        "joint6": 2.27,
        "joint7": 0.785398,
    }
    default_dofs_kp: dict[str, float] = {
        "joint1": 4500,
        "joint2": 4500,
        "joint3": 3500,
        "joint4": 3500,
        "joint5": 2000,
        "joint6": 2000,
        "joint7": 2000,
    }
    default_dofs_kd: dict[str, float] = {
        "joint1": 450,
        "joint2": 450,
        "joint3": 350,
        "joint4": 350,
        "joint5": 200,
        "joint6": 200,
        "joint7": 200,
    }


@ROBOT_REGISTRY.register()
class FrankaEmikaPanda(RobotOptions):
    file: str = "manipulators/franka_panda/panda.xml"

    is_fixed_base: bool = True
    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)  # x,y,z [m]
    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -2.27,
        "joint5": 0.0,
        "joint6": 2.27,
        "joint7": 0.785398,
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    links_to_keep: list[str] = ["left_gripper_pad", "right_gripper_pad"]

    # dofs_vel_limit: dict[str, float] = {
    #     "joint1": 2.61799,  # rad/s
    #     "joint2": 2.61799,
    #     "joint3": 2.61799,
    #     "joint4": 2.61799,
    #     "joint5": 3.14159,
    #     "joint6": 3.14159,
    #     "joint7": 3.14159,
    #     "finger_joint1": 0.05,  # 50 mm/s per finger
    #     "finger_joint2": 0.05,
    # }
    default_dofs_kp: dict[str, float] = {
        "joint1": 4500,  # * 0.8,
        "joint2": 4500,  # * 0.8,
        "joint3": 3500,  # * 0.8,
        "joint4": 3500,  # * 0.8,
        "joint5": 2000,  # * 0.8,
        "joint6": 2000,  # * 0.8,
        "joint7": 2000,  # * 0.8,
        "finger_joint1": 100,
        "finger_joint2": 100,
    }
    default_dofs_kd: dict[str, float] = {
        "joint1": 450,
        "joint2": 450,
        "joint3": 350,
        "joint4": 350,
        "joint5": 200,
        "joint6": 200,
        "joint7": 200,
        "finger_joint1": 10,
        "finger_joint2": 10,
    }

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        ee_link_names=[
            "hand",
        ]
    )

    # The gripper *base* frame (0.107 m up from link7, rotated -45 deg about z), matching
    # ``metadata.ee_link_names`` — not a TCP between the fingertips, so IK against it
    # servos ~10 cm behind the fingers.
    ee_links_name: ClassVar[list[str]] = ["hand"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.0,
        "finger_joint2": 0.0,
    }


@ROBOT_REGISTRY.register()
class FrankaEmikaPandaNoHand(RobotOptions):
    """The Panda arm without its hand: ``panda_nohand.xml``, the seven joints ending in a flange.

    The Panda counterpart of :class:`FrankaResearch3`, and what a composite rig wants when the
    end-effector is a separate entity — :class:`~eden.options.robots.FrankaEmikaPanda` is one MJCF
    with the fingers inside it, so its hand cannot be swapped or welded to. The two arms expose the
    same ``attachment`` mount frame, so a gripper written for one mounts on the other unchanged.

    The Panda's own gains and joint limits, which are *not* the FR3's: see
    :class:`FrankaResearch3` for what differs between the two robots.
    """

    file: str = "manipulators/franka_panda/panda_nohand.xml"

    is_fixed_base: bool = True
    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)  # x,y,z [m]
    links_to_keep: list[str] = ["attachment"]

    ee_links_name: ClassVar[list[str]] = ["attachment"]
    metadata: ClassVar[MetadataOptions] = MetadataOptions(ee_link_names=["attachment"])

    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -2.27,
        "joint5": 0.0,
        "joint6": 2.27,
        "joint7": 0.785398,
    }
    default_dofs_kp: dict[str, float] = {
        "joint1": 4500,
        "joint2": 4500,
        "joint3": 3500,
        "joint4": 3500,
        "joint5": 2000,
        "joint6": 2000,
        "joint7": 2000,
    }
    default_dofs_kd: dict[str, float] = {
        "joint1": 450,
        "joint2": 450,
        "joint3": 350,
        "joint4": 350,
        "joint5": 200,
        "joint6": 200,
        "joint7": 200,
    }
