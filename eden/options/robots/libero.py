"""LIBERO Panda and mounted-rig options."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from eden.options.entities import BaseEntityOptions, CompositeEntityOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.robosuite import RethinkMount
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda

LIBERO_MOUNT_ENTITY_NAME = "mount"
LIBERO_ROBOT_ENTITY_NAME = "robot"


class LiberoPanda(FrankaEmikaPanda):
    """Panda dynamics used by every LIBERO family."""

    arm_dofs_name: ClassVar[tuple[str, ...]] = tuple(f"joint{index}" for index in range(1, 8))
    gripper_dofs_name: ClassVar[tuple[str, ...]] = ("finger_joint1", "finger_joint2")
    grip_site_pos_local: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.0965)
    grip_site_quat_local: ClassVar[tuple[float, float, float, float]] = (
        0.7071067811865475,
        0.0,
        0.0,
        -0.7071067811865475,
    )

    # The bundle owns reset qpos; these arm values are the fixed OSC nullspace reference.
    default_dofs_pos: dict[str, float] = Field(
        default_factory=lambda: {
            "joint1": 0.0,
            "joint2": -0.161037389,
            "joint3": 0.0,
            "joint4": -2.44459747,
            "joint5": 0.0,
            "joint6": 2.2267522,
            "joint7": 0.7853981633974483,
        }
    )
    default_dofs_kp: dict[str, float] = Field(
        default_factory=lambda: {"finger_joint1": 1000.0, "finger_joint2": 1000.0}
    )
    default_dofs_kd: dict[str, float] = Field(default_factory=lambda: {"finger_joint1": 0.0, "finger_joint2": 0.0})
    default_dofs_damping: dict[str, float] = Field(
        default_factory=lambda: {
            "joint1": 0.1,
            "joint2": 0.1,
            "joint3": 0.1,
            "joint4": 0.1,
            "joint5": 0.1,
            "joint6": 0.01,
            "joint7": 0.01,
            "finger_joint1": 100.0,
            "finger_joint2": 100.0,
        }
    )
    default_dofs_armature: dict[str, float] = Field(
        default_factory=lambda: (
            {name: 5.0 / index for index, name in enumerate(LiberoPanda.arm_dofs_name, start=1)}
            | {"finger_joint1": 1.0, "finger_joint2": 1.0}
        )
    )
    default_dofs_frictionloss: dict[str, float] = Field(
        default_factory=lambda: (
            {name: 0.1 for name in LiberoPanda.arm_dofs_name} | {"finger_joint1": 1.0, "finger_joint2": 1.0}
        )
    )
    default_dofs_force_limits: dict[str, float] = Field(
        default_factory=lambda: {"finger_joint1": 20.0, "finger_joint2": 20.0}
    )
    material: RigidMaterialOptions = Field(default_factory=lambda: RigidMaterialOptions(gravity_compensation=1.0))


class LiberoMountedPanda(CompositeEntityOptions):
    """Rethink mount and Panda represented as one validated welded rig."""

    mount: RethinkMount = Field(
        default_factory=lambda: RethinkMount(material=RigidMaterialOptions(gravity_compensation=1.0))
    )
    robot: LiberoPanda = Field(
        default_factory=lambda: LiberoPanda(default_root_pos=(0.0, 0.0, -0.01), is_fixed_base=False)
    )

    def entity_parts(self) -> list[tuple[str, BaseEntityOptions]]:
        return [(LIBERO_MOUNT_ENTITY_NAME, self.mount), (LIBERO_ROBOT_ENTITY_NAME, self.robot)]

    def attachments(self) -> dict[str, tuple[str, str]]:
        return {
            LIBERO_ROBOT_ENTITY_NAME: (
                LIBERO_MOUNT_ENTITY_NAME,
                "base",
            )
        }


__all__ = [
    "LIBERO_MOUNT_ENTITY_NAME",
    "LIBERO_ROBOT_ENTITY_NAME",
    "LiberoMountedPanda",
    "LiberoPanda",
]
