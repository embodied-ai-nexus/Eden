"""KUKA LBR iiwa 14 arm configuration."""

from __future__ import annotations
from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions


@ROBOT_REGISTRY.register()
class Iiwa14(RobotOptions):
    file: str = "arms/kuka_iiwa14/iiwa14.xml"

    is_fixed_base: bool = True

    dofs_name: list[str] = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    # The MJCF "home" keyframe: elbow bent with the end effector in front of the base.
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "joint1": 0.0,
        "joint2": 0.785398,
        "joint3": 0.0,
        "joint4": -1.5708,
        "joint5": 0.0,
        "joint6": 0.0,
        "joint7": 0.0,
    }
    # Match the MJCF's own position actuators (gainprm=2000, biasprm=(0, -2000, -200)).
    default_dofs_kp: dict[str, float] = {
        "joint1": 2000,
        "joint2": 2000,
        "joint3": 2000,
        "joint4": 2000,
        "joint5": 2000,
        "joint6": 2000,
        "joint7": 2000,
    }
    default_dofs_kd: dict[str, float] = {
        "joint1": 200,
        "joint2": 200,
        "joint3": 200,
        "joint4": 200,
        "joint5": 200,
        "joint6": 200,
        "joint7": 200,
    }

    # The MJCF declares neither joint damping nor a forcerange (its ``<general>`` actuators
    # carry only a ctrlrange), so without these the arm loads undamped with an unbounded
    # drive — it rings under load and the drive can command torques no iiwa14 can produce.
    # Values come from KUKA's own URDF for this arm (``<dynamics damping="0.5">``,
    # ``<limit effort="300">``; the datasheet limit is 320 N.m on the first four axes).
    #
    # These are all-or-nothing, NOT a floor: overriding either dict replaces it wholesale, so
    # a partial dict (one joint) drops the remaining six back to the MJCF's 0 damping and
    # unbounded range rather than keeping the values here, and neither property warns. Restate
    # every joint you want kept. Both behaviours pinned by ``tests/options/robots/test_kuka_iiwa14.py``.
    default_dofs_damping: dict[str, float] = {f"joint{i}": 0.5 for i in range(1, 8)}
    default_dofs_force_limits: dict[str, float] = {f"joint{i}": 300.0 for i in range(1, 8)}

    # The flange body; the MJCF's attachment_site sits on it (+0.045 m along local z).
    ee_links_name: ClassVar[list[str]] = ["link7"]
