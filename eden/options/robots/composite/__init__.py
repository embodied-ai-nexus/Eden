"""Composite robot presets — multi-entity rigs (arm + gripper / dex hand, and kin).

Each concrete rig lives in its own module and registers itself into
:data:`COMPOSITE_ROBOT_REGISTRY`; new arms or dexterous hands slot in as new modules. See
:mod:`eden.options.robots.composite.base` for the shared bases (:class:`ManipulatorOptions` for arm + gripper,
:class:`DexManipulatorOptions` for arm + dex hand, :class:`HumanoidOptions` for a humanoid root
with an end-effector per wrist, e.g. :class:`OliWuji`, :class:`G1Wuji`).

Usage::

    from eden.options.robots.composite import Droid, FrankaXHand, COMPOSITE_ROBOT_REGISTRY

    droid = Droid(base_pos=(0.0, 0.0, 0.4))
    scene = en.scenes.Empty(is_support_enabled=True, robot=droid)
    cls = COMPOSITE_ROBOT_REGISTRY.get("droid")
"""

from eden.options.robots.composite.base import (
    COMPOSITE_ROBOT_REGISTRY,
    DexManipulatorOptions,
    HumanoidOptions,
    ManipulatorOptions,
    has_frame_alignment,
    weld_quat,
)
from eden.options.robots.composite.droid import Droid
from eden.options.robots.composite.franka_xhand import FrankaXHand
from eden.options.robots.composite.g1_inspire import G1Inspire
from eden.options.robots.composite.g1_wuji import G1Wuji
from eden.options.robots.composite.kuka_sharpa import KukaSharpa
from eden.options.robots.composite.oli_wuji import OliWuji
from eden.options.robots.composite.panda_hand import PandaFrankaHand
from eden.options.robots.composite.ur_robotiq import UR5eRobotiq, UR10eRobotiq
from eden.options.robots.composite.xarm_xhand import XArm7XHand1_L, XArm7XHand1_R

__all__ = [
    "COMPOSITE_ROBOT_REGISTRY",
    "ManipulatorOptions",
    "DexManipulatorOptions",
    "HumanoidOptions",
    "Droid",
    "FrankaXHand",
    "G1Inspire",
    "G1Wuji",
    "KukaSharpa",
    "OliWuji",
    "PandaFrankaHand",
    "UR5eRobotiq",
    "UR10eRobotiq",
    "XArm7XHand1_L",
    "XArm7XHand1_R",
    "has_frame_alignment",
    "weld_quat",
]
