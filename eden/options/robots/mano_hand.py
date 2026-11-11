"""MANO hand model configurations (left/right)."""

from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.entities import MetadataOptions


class _ManoHand(RobotOptions):
    """Base class for MANO Hand model from https://mano.is.tue.mpg.de/."""

    # MANO 21-DOF: 20 revolute joints in URDF + wrist (root). Wrist is fixed in hand-only URDF.
    dofs_name: list[str] = [
        "j_index1y",
        "j_index1x",
        "j_index2",
        "j_index3",
        "j_middle1y",
        "j_middle1x",
        "j_middle2",
        "j_middle3",
        "j_pinky1y",
        "j_pinky1x",
        "j_pinky2",
        "j_pinky3",
        "j_ring1y",
        "j_ring1x",
        "j_ring2",
        "j_ring3",
        "j_thumb1y",
        "j_thumb1z",
        "j_thumb2",
        "j_thumb3",
        # "j_wrist"
    ]
    default_dofs_pos: dict[str, float] = {
        "*": 0.0,
    }
    default_dofs_kp: dict[str, float] = {
        "*": 100.0,
    }
    default_dofs_kd: dict[str, float] = {
        "*": 10.0,
    }

    # Per-joint armature set to a realistic finger rotor inertia (~3.3e-3). URDF carries no armature field, so otherwise
    # every joint takes Genesis' non-physical 0.1 morph default (~30x heavier); it dominates inertia only at low kd.
    default_dofs_armature: dict[str, float] = {"*": 1.0 / 300}


@ROBOT_REGISTRY.register()
class ManoHand_R(_ManoHand):
    file: str = "hands/mano_hand/mano_right.urdf"

    # Hand-axis convention: ``up`` = fingers, ``front`` = palm normal.
    # MANO right-hand URDF has fingers along -X and palm normal along -Y.
    up: tuple[int, int, int] = (-1, 0, 0)
    front: tuple[int, int, int] = (0, -1, 0)

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
    )


@ROBOT_REGISTRY.register()
class ManoHand_L(_ManoHand):
    file: str = "hands/mano_hand/mano_left.urdf"

    # MANO left-hand mirror: palm normal along URDF +Y.
    up: tuple[int, int, int] = (-1, 0, 0)
    front: tuple[int, int, int] = (0, 1, 0)

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
    )
