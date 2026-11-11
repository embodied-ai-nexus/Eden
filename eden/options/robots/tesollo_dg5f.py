"""Tesollo DG-5F dexterous hand configurations (left/right)."""

from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions, MetadataOptions


class _TesolloDG5F(RobotOptions):
    default_dofs_pos: dict[str, float] = {"*": 0.0}

    default_dofs_kp: dict[str, float] = {"*": 100.0}

    default_dofs_kd: dict[str, float] = {"*": 10.0}

    # Per-joint armature set to a realistic finger rotor inertia (~3.3e-3). URDF carries no armature field, so otherwise
    # every joint takes Genesis' non-physical 0.1 morph default (~30x heavier); it dominates inertia only at low kd.
    default_dofs_armature: dict[str, float] = {"*": 1.0 / 300}


@ROBOT_REGISTRY.register(name="tesollo_dg5f_r")
class TesolloDG5F_R(_TesolloDG5F):
    file: str = "hands/tesollo_hand/dg5f_right.urdf"

    dofs_name: list[str] = [
        "rj_dg_1_1",
        "rj_dg_1_2",
        "rj_dg_1_3",
        "rj_dg_1_4",
        "rj_dg_2_1",
        "rj_dg_2_2",
        "rj_dg_2_3",
        "rj_dg_2_4",
        "rj_dg_3_1",
        "rj_dg_3_2",
        "rj_dg_3_3",
        "rj_dg_3_4",
        "rj_dg_4_1",
        "rj_dg_4_2",
        "rj_dg_4_3",
        "rj_dg_4_4",
        "rj_dg_5_1",
        "rj_dg_5_2",
        "rj_dg_5_3",
        "rj_dg_5_4",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(side="right")


@ROBOT_REGISTRY.register(name="tesollo_dg5f_l")
class TesolloDG5F_L(_TesolloDG5F):
    file: str = "hands/tesollo_hand/dg5f_left.urdf"

    dofs_name: list[str] = [
        "lj_dg_1_1",
        "lj_dg_1_2",
        "lj_dg_1_3",
        "lj_dg_1_4",
        "lj_dg_2_1",
        "lj_dg_2_2",
        "lj_dg_2_3",
        "lj_dg_2_4",
        "lj_dg_3_1",
        "lj_dg_3_2",
        "lj_dg_3_3",
        "lj_dg_3_4",
        "lj_dg_4_1",
        "lj_dg_4_2",
        "lj_dg_4_3",
        "lj_dg_4_4",
        "lj_dg_5_1",
        "lj_dg_5_2",
        "lj_dg_5_3",
        "lj_dg_5_4",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(side="left")
