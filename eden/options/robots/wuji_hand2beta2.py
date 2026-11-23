"""Wuji Hand2 (beta2) dexterous hand configurations (left/right, with or without the arm mount).

Twenty-DOF anthropomorphic hand from Wuji Technology (5 fingers x 4 DOF), in two
flavours per side: the bare hand (rooted at the wrist) and the ``WithMount`` variant
that carries the arm-flange adapter plate as its root link. Assets (URDF + STL meshes,
MIT-licensed) live under ``hands/wuji_hand2beta2`` in the ``Kashu7100/eden_assets``
dataset, converted from the upstream ``wuji-technology/wuji-description``
``hand2/hand2_beta2`` package.
"""

from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions, MetadataOptions


class _WujiHand2Beta2(RobotOptions):
    default_dofs_pos: dict[str, float] = {"*": 0.0}

    # Per-joint calibration from wuji-mjlab v2026.10.22 ``pen_spin/config/wuji_hand2/hand2_constants.py``
    # (identified on a right hand2_2, whose joints match hand2_beta2). Gains are the firmware MIT gains
    # (kp 3.0, kd 0.075/0.07/0.05/0.05) times kt * 1.9115 A/unit.
    default_dofs_kp: dict[str, float] = {
        ".*_thumb_cmc_flex": 2.37175,
        ".*_thumb_cmc_abd": 2.03285,
        ".*_thumb_mcp": 1.27075,
        ".*_thumb_ip": 0.762103,
        ".*_index_finger_mcp_flex": 2.37175,
        ".*_index_finger_mcp_abd": 0.959367,
        ".*_index_finger_pip": 0.959367,
        ".*_index_finger_dip": 0.959367,
        ".*_middle_finger_mcp_flex": 2.37175,
        ".*_middle_finger_mcp_abd": 0.959367,
        ".*_middle_finger_pip": 0.959367,
        ".*_middle_finger_dip": 0.959367,
        ".*_ring_finger_mcp_flex": 2.37175,
        ".*_ring_finger_mcp_abd": 0.959367,
        ".*_ring_finger_pip": 0.959367,
        ".*_ring_finger_dip": 0.959367,
        ".*_pinky_mcp_flex": 2.55812,
        ".*_pinky_mcp_abd": 0.639387,
        ".*_pinky_pip": 0.639387,
        ".*_pinky_dip": 0.639387,
    }

    default_dofs_kd: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.0592938,
        ".*_thumb_cmc_abd": 0.0474332,
        ".*_thumb_mcp": 0.0211791,
        ".*_thumb_ip": 0.0127017,
        ".*_index_finger_mcp_flex": 0.0592938,
        ".*_index_finger_mcp_abd": 0.0223852,
        ".*_index_finger_pip": 0.0159895,
        ".*_index_finger_dip": 0.0159895,
        ".*_middle_finger_mcp_flex": 0.0592938,
        ".*_middle_finger_mcp_abd": 0.0223852,
        ".*_middle_finger_pip": 0.0159895,
        ".*_middle_finger_dip": 0.0159895,
        ".*_ring_finger_mcp_flex": 0.0592938,
        ".*_ring_finger_mcp_abd": 0.0223852,
        ".*_ring_finger_pip": 0.0159895,
        ".*_ring_finger_dip": 0.0159895,
        ".*_pinky_mcp_flex": 0.063953,
        ".*_pinky_mcp_abd": 0.014919,
        ".*_pinky_pip": 0.0106565,
        ".*_pinky_dip": 0.0106565,
    }

    # kt * 1.5 A current limit.
    default_dofs_force_limits: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.6204,
        ".*_thumb_cmc_abd": 0.53175,
        ".*_thumb_mcp": 0.3324,
        ".*_thumb_ip": 0.19935,
        ".*_index_finger_mcp_flex": 0.6204,
        ".*_index_finger_mcp_abd": 0.25095,
        ".*_index_finger_pip": 0.25095,
        ".*_index_finger_dip": 0.25095,
        ".*_middle_finger_mcp_flex": 0.6204,
        ".*_middle_finger_mcp_abd": 0.25095,
        ".*_middle_finger_pip": 0.25095,
        ".*_middle_finger_dip": 0.25095,
        ".*_ring_finger_mcp_flex": 0.6204,
        ".*_ring_finger_mcp_abd": 0.25095,
        ".*_ring_finger_pip": 0.25095,
        ".*_ring_finger_dip": 0.25095,
        ".*_pinky_mcp_flex": 0.66915,
        ".*_pinky_mcp_abd": 0.16725,
        ".*_pinky_pip": 0.16725,
        ".*_pinky_dip": 0.16725,
    }

    # The URDF has no armature; without this Genesis would use 0.1.
    default_dofs_armature: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.000145716,
        ".*_thumb_cmc_abd": 0.000136639,
        ".*_thumb_mcp": 0.000127355,
        ".*_thumb_ip": 8.7689e-05,
        ".*_index_finger_mcp_flex": 0.000113047,
        ".*_index_finger_mcp_abd": 5.12716e-05,
        ".*_index_finger_pip": 4.84378e-05,
        ".*_index_finger_dip": 7.90619e-05,
        ".*_middle_finger_mcp_flex": 0.000201105,
        ".*_middle_finger_mcp_abd": 7.10665e-05,
        ".*_middle_finger_pip": 6.05475e-05,
        ".*_middle_finger_dip": 5.65927e-05,
        ".*_ring_finger_mcp_flex": 0.00013939,
        ".*_ring_finger_mcp_abd": 3.54411e-05,
        ".*_ring_finger_pip": 5.23397e-05,
        ".*_ring_finger_dip": 6.04795e-05,
        ".*_pinky_mcp_flex": 9.70626e-05,
        ".*_pinky_mcp_abd": 3.82605e-05,
        ".*_pinky_pip": 4.66446e-05,
        ".*_pinky_dip": 4.74378e-05,
    }

    default_dofs_damping: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.0026915,
        ".*_thumb_cmc_abd": 0.00428122,
        ".*_thumb_mcp": 0.00141357,
        ".*_thumb_ip": 0.000513181,
        ".*_index_finger_mcp_flex": 0.00105109,
        ".*_index_finger_mcp_abd": 0.0,
        ".*_index_finger_pip": 0.00103028,
        ".*_index_finger_dip": 6.63526e-05,
        ".*_middle_finger_mcp_flex": 0.00215877,
        ".*_middle_finger_mcp_abd": 0.000884286,
        ".*_middle_finger_pip": 0.00126105,
        ".*_middle_finger_dip": 0.000112849,
        ".*_ring_finger_mcp_flex": 0.00145065,
        ".*_ring_finger_mcp_abd": 0.0,
        ".*_ring_finger_pip": 0.000714964,
        ".*_ring_finger_dip": 0.0,
        ".*_pinky_mcp_flex": 0.0,
        ".*_pinky_mcp_abd": 0.0,
        ".*_pinky_pip": 0.00162465,
        ".*_pinky_dip": 0.0,
    }

    default_dofs_frictionloss: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.0367355,
        ".*_thumb_cmc_abd": 0.0342266,
        ".*_thumb_mcp": 0.0187921,
        ".*_thumb_ip": 0.0122546,
        ".*_index_finger_mcp_flex": 0.0377085,
        ".*_index_finger_mcp_abd": 0.0177933,
        ".*_index_finger_pip": 0.0297927,
        ".*_index_finger_dip": 0.0161822,
        ".*_middle_finger_mcp_flex": 0.0355645,
        ".*_middle_finger_mcp_abd": 0.0240617,
        ".*_middle_finger_pip": 0.0127756,
        ".*_middle_finger_dip": 0.010917,
        ".*_ring_finger_mcp_flex": 0.0528258,
        ".*_ring_finger_mcp_abd": 0.0266927,
        ".*_ring_finger_pip": 0.0148677,
        ".*_ring_finger_dip": 0.0215947,
        ".*_pinky_mcp_flex": 0.0338252,
        ".*_pinky_mcp_abd": 0.0261331,
        ".*_pinky_pip": 0.0142009,
        ".*_pinky_dip": 0.0129471,
    }


# Finger anatomy (per hand, 4 DOF each): CMC/MCP flexion + abduction, then the two
# distal flexion joints (PIP/DIP for fingers, MCP/IP for the thumb). The empty
# ``*_tip`` links are the fingertip frames kept for retargeting / task use.


@ROBOT_REGISTRY.register(name="wuji_hand2beta2_r")
class WujiHand2Beta2_R(_WujiHand2Beta2):
    file: str = "hands/wuji_hand2beta2/wuji_hand2beta2_right.urdf"

    # Hand-axis convention via Eden's standard ``up`` / ``front`` fields:
    #   ``up``    = URDF axis the fingers extend along
    #   ``front`` = URDF axis the palm normal points along
    # The Wuji URDF has the fingers extending along -Z (all MCP origins sit at
    # negative z in the wrist frame) and the palm normal along +Y (positive
    # flexion curls the fingertips toward +Y). ``align_up_and_front`` maps these
    # to world +Z / +X at load, so the hand spawns fingers-up, palm-forward.
    up: tuple[int, int, int] = (0, 0, -1)
    front: tuple[int, int, int] = (0, 1, 0)

    dofs_name: list[str] = [
        "r_thumb_cmc_flex",
        "r_thumb_cmc_abd",
        "r_thumb_mcp",
        "r_thumb_ip",
        "r_index_finger_mcp_flex",
        "r_index_finger_mcp_abd",
        "r_index_finger_pip",
        "r_index_finger_dip",
        "r_middle_finger_mcp_flex",
        "r_middle_finger_mcp_abd",
        "r_middle_finger_pip",
        "r_middle_finger_dip",
        "r_ring_finger_mcp_flex",
        "r_ring_finger_mcp_abd",
        "r_ring_finger_pip",
        "r_ring_finger_dip",
        "r_pinky_mcp_flex",
        "r_pinky_mcp_abd",
        "r_pinky_pip",
        "r_pinky_dip",
    ]

    links_to_keep: list[str] = [
        "r_thumb_tip",
        "r_index_finger_tip",
        "r_middle_finger_tip",
        "r_ring_finger_tip",
        "r_pinky_tip",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # NOTE: the order of ``target_hand_links`` matches the order of ``mano_hand_idx``.
        target_hand_links=[
            "r_thumb_tip",
            "r_index_finger_tip",
            "r_middle_finger_tip",
            "r_ring_finger_tip",
            "r_pinky_tip",
        ],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


@ROBOT_REGISTRY.register(name="wuji_hand2beta2_l")
class WujiHand2Beta2_L(_WujiHand2Beta2):
    file: str = "hands/wuji_hand2beta2/wuji_hand2beta2_left.urdf"

    # Same finger-extension axis as the right hand (fingers along URDF -Z). The
    # left URDF mirrors the right about the palm plane, flipping the palm normal
    # to -Y, so ``front`` flips sign while ``up`` stays the same.
    up: tuple[int, int, int] = (0, 0, -1)
    front: tuple[int, int, int] = (0, -1, 0)

    dofs_name: list[str] = [
        "l_thumb_cmc_flex",
        "l_thumb_cmc_abd",
        "l_thumb_mcp",
        "l_thumb_ip",
        "l_index_finger_mcp_flex",
        "l_index_finger_mcp_abd",
        "l_index_finger_pip",
        "l_index_finger_dip",
        "l_middle_finger_mcp_flex",
        "l_middle_finger_mcp_abd",
        "l_middle_finger_pip",
        "l_middle_finger_dip",
        "l_ring_finger_mcp_flex",
        "l_ring_finger_mcp_abd",
        "l_ring_finger_pip",
        "l_ring_finger_dip",
        "l_pinky_mcp_flex",
        "l_pinky_mcp_abd",
        "l_pinky_pip",
        "l_pinky_dip",
    ]

    links_to_keep: list[str] = [
        "l_thumb_tip",
        "l_index_finger_tip",
        "l_middle_finger_tip",
        "l_ring_finger_tip",
        "l_pinky_tip",
    ]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        # NOTE: the order of ``target_hand_links`` matches the order of ``mano_hand_idx``.
        target_hand_links=[
            "l_thumb_tip",
            "l_index_finger_tip",
            "l_middle_finger_tip",
            "l_ring_finger_tip",
            "l_pinky_tip",
        ],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


# --- mounted variants ------------------------------------------------------- #
# Upstream also ships ``{side}_with_mount.urdf``: the same 20-DOF hand with the wrist
# adapter plate (``{r,l}_mount``, 69 g) prepended as a new root link, fixed to
# ``{r,l}_wrist``. That plate is the flange the hand bolts to an arm through, so these
# are the variants to reach for when composing an arm + hand — the mount frame is the
# attach point, and its mass/inertia stay in the payload instead of being dropped.
# Joint names, DOF order and the ``up``/``front`` axes are unchanged; what differs is
# the base link (``{r,l}_mount``, not ``{r,l}_wrist``) and hence the frame every root
# pose getter/setter and ``default_root_pos`` refers to.


@ROBOT_REGISTRY.register(name="wuji_hand2beta2_with_mount_r")
class WujiHand2Beta2WithMount_R(WujiHand2Beta2_R):
    file: str = "hands/wuji_hand2beta2/wuji_hand2beta2_right_with_mount.urdf"

    # ``r_wrist`` is a fixed child of the ``r_mount`` root here, so Genesis would merge
    # it into the base; keep it so link lookups written against the unmounted hand
    # (which is rooted at ``r_wrist``) still resolve.
    links_to_keep: list[str] = [
        "r_wrist",
        "r_thumb_tip",
        "r_index_finger_tip",
        "r_middle_finger_tip",
        "r_ring_finger_tip",
        "r_pinky_tip",
    ]


@ROBOT_REGISTRY.register(name="wuji_hand2beta2_with_mount_l")
class WujiHand2Beta2WithMount_L(WujiHand2Beta2_L):
    file: str = "hands/wuji_hand2beta2/wuji_hand2beta2_left_with_mount.urdf"

    links_to_keep: list[str] = [
        "l_wrist",
        "l_thumb_tip",
        "l_index_finger_tip",
        "l_middle_finger_tip",
        "l_ring_finger_tip",
        "l_pinky_tip",
    ]
