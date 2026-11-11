"""UFACTORY xArm7 arm configuration."""

from __future__ import annotations
from typing import ClassVar

import numpy as np

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions


@ROBOT_REGISTRY.register()
class Xarm7(RobotOptions):
    file: str = "arms/ufactory_xarm7/xarm.urdf"

    is_fixed_base: bool = True
    # Keep URDF fixed end-effector link so it can be referenced by controllers
    # and attachment hooks (e.g., gripper mount point).
    links_to_keep: list[str] = ["link_eef"]

    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    # The xArm7 zero pose is the fully folded shipping pose; bend the elbow (joint4)
    # and wrist (joint6) to a tabletop ready pose with the end effector in front of
    # the base (~(0.46, 0, 0.36) m) pointing straight down.
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": np.pi / 3,
        "joint5": 0.0,
        "joint6": np.pi / 3,
        "joint7": 0.0,
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

    ee_links_name: ClassVar[list[str]] = ["link_eef"]
