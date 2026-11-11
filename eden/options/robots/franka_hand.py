"""Franka Hand parallel-jaw gripper configuration."""

from __future__ import annotations
from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions


@ROBOT_REGISTRY.register()
class FrankaHand(RobotOptions):
    file: str = "grippers/franka_hand/franka_hand.xml"

    is_fixed_base: bool = True
    links_to_keep: list[str] = ["hand", "left_gripper_pad", "right_gripper_pad"]

    dofs_name: list[str] = [
        "finger_joint1",
        "finger_joint2",
    ]

    default_dofs_pos: dict[str, float] = {
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    default_dofs_kp: dict[str, float] = {
        "finger_joint1": 100.0,
        "finger_joint2": 100.0,
    }
    default_dofs_kd: dict[str, float] = {
        "finger_joint1": 10.0,
        "finger_joint2": 10.0,
    }

    # The gripper base frame, not a fingertip TCP — see FrankaEmikaPanda.ee_links_name.
    ee_links_name: ClassVar[list[str]] = ["hand"]

    open_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.04,
        "finger_joint2": 0.04,
    }
    close_dofs_pos: ClassVar[dict[str, float]] = {
        "finger_joint1": 0.0,
        "finger_joint2": 0.0,
    }

    # Pad midpoint along the hand's z axis: Franka's published flange-to-TCP distance, which the
    # pad collision geoms of this asset reproduce to within 0.3 mm. Fingers open 0.04 m each.
    tcp_offset: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.1034)
    stroke: ClassVar[float] = 0.08
