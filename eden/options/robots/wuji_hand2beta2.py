"""Wuji Hand2 (beta2) dexterous hand configurations (left/right, with or without the arm mount).

Twenty-DOF anthropomorphic hand from Wuji Technology (5 fingers x 4 DOF), in two
flavours per side: the bare hand (rooted at the wrist) and the ``WithMount`` variant
that carries the arm-flange adapter plate as its root link. Assets (URDF + STL meshes,
MIT-licensed) live under ``hands/wuji_hand2beta2`` in the ``Kashu7100/eden_assets``
dataset, converted from the upstream ``wuji-technology/wuji-description``
``hand2/hand2_beta2`` package.
"""

from typing import ClassVar

from eden.options.actuators import (
    ActuatorSpecOptions,
    WujiHand2Actuator,
    WujiHand2Finger,
    WujiHand2Flex,
    WujiHand2Pinky,
    WujiHand2PinkyFlex,
    WujiHand2ThumbAbd,
    WujiHand2ThumbIp,
    WujiHand2ThumbMcp,
)
from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import MetadataOptions, RobotOptions


# The pen-spin operating point from wuji-mjlab v2026.10.22 ``pen_spin/config/wuji_hand2/hand2_constants.py``,
# per joint pattern: the motor, and the firmware MIT kd by joint position within the finger. Every joint runs
# at MIT kp 3.0 and a 1.5 A current limit.
_PEN_SPIN_MOTORS: dict[str, tuple[type[WujiHand2Actuator], float]] = {
    ".*_thumb_cmc_flex": (WujiHand2Flex, 0.075),
    ".*_thumb_cmc_abd": (WujiHand2ThumbAbd, 0.07),
    ".*_thumb_mcp": (WujiHand2ThumbMcp, 0.05),
    ".*_thumb_ip": (WujiHand2ThumbIp, 0.05),
    ".*_(index|middle|ring)_finger_mcp_flex": (WujiHand2Flex, 0.075),
    ".*_(index|middle|ring)_finger_mcp_abd": (WujiHand2Finger, 0.07),
    ".*_(index|middle|ring)_finger_(pip|dip)": (WujiHand2Finger, 0.05),
    ".*_pinky_mcp_flex": (WujiHand2PinkyFlex, 0.075),
    ".*_pinky_mcp_abd": (WujiHand2Pinky, 0.07),
    ".*_pinky_(pip|dip)": (WujiHand2Pinky, 0.05),
}
_MIT_KP = 3.0
_CURRENT_LIMIT_A = 1.5


class _WujiHand2Beta2(RobotOptions):
    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = {p: m for p, (m, _) in _PEN_SPIN_MOTORS.items()}

    # The current the force limits below are trained at; ``WujiHandDeployment`` programs it when
    # ``effort_limit_a`` is unset.
    current_limit_a: ClassVar[float] = _CURRENT_LIMIT_A

    default_dofs_pos: dict[str, float] = {"*": 0.0}

    default_dofs_kp: dict[str, float] = {p: m.JOINT_GAIN(_MIT_KP) for p, (m, _) in _PEN_SPIN_MOTORS.items()}
    default_dofs_kd: dict[str, float] = {p: m.JOINT_GAIN(kd) for p, (m, kd) in _PEN_SPIN_MOTORS.items()}
    default_dofs_force_limits: dict[str, float] = {
        p: m.TORQUE_CONSTANT * _CURRENT_LIMIT_A for p, (m, _) in _PEN_SPIN_MOTORS.items()
    }

    # Identified per joint (same source), so not part of the motor specs.
    # The URDF has no armature; without this Genesis would use 0.1.
    default_dofs_armature: dict[str, float] = {
        ".*_thumb_cmc_flex": 1.457162e-4,
        ".*_thumb_cmc_abd": 1.366389e-4,
        ".*_thumb_mcp": 1.273553e-4,
        ".*_thumb_ip": 8.768902e-5,
        ".*_index_finger_mcp_flex": 1.130474e-4,
        ".*_index_finger_mcp_abd": 5.127161e-5,
        ".*_index_finger_pip": 4.843783e-5,
        ".*_index_finger_dip": 7.906193e-5,
        ".*_middle_finger_mcp_flex": 2.011048e-4,
        ".*_middle_finger_mcp_abd": 7.106655e-5,
        ".*_middle_finger_pip": 6.054746e-5,
        ".*_middle_finger_dip": 5.659272e-5,
        ".*_ring_finger_mcp_flex": 1.393897e-4,
        ".*_ring_finger_mcp_abd": 3.544108e-5,
        ".*_ring_finger_pip": 5.233974e-5,
        ".*_ring_finger_dip": 6.047948e-5,
        ".*_pinky_mcp_flex": 9.706259e-5,
        ".*_pinky_mcp_abd": 3.826054e-5,
        ".*_pinky_pip": 4.664456e-5,
        ".*_pinky_dip": 4.743778e-5,
    }

    default_dofs_damping: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.0026915050,
        ".*_thumb_cmc_abd": 0.0042812198,
        ".*_thumb_mcp": 0.0014135711,
        ".*_thumb_ip": 0.0005131806,
        ".*_index_finger_mcp_flex": 0.0010510923,
        ".*_index_finger_mcp_abd": 0.0,
        ".*_index_finger_pip": 0.0010302809,
        ".*_index_finger_dip": 0.0000663526,
        ".*_middle_finger_mcp_flex": 0.0021587672,
        ".*_middle_finger_mcp_abd": 0.0008842859,
        ".*_middle_finger_pip": 0.0012610488,
        ".*_middle_finger_dip": 0.0001128493,
        ".*_ring_finger_mcp_flex": 0.0014506542,
        ".*_ring_finger_mcp_abd": 0.0,
        ".*_ring_finger_pip": 0.0007149635,
        ".*_ring_finger_dip": 0.0,
        ".*_pinky_mcp_flex": 0.0,
        ".*_pinky_mcp_abd": 0.0,
        ".*_pinky_pip": 0.0016246495,
        ".*_pinky_dip": 0.0,
    }

    default_dofs_frictionloss: dict[str, float] = {
        ".*_thumb_cmc_flex": 0.0367355430,
        ".*_thumb_cmc_abd": 0.0342266392,
        ".*_thumb_mcp": 0.0187921182,
        ".*_thumb_ip": 0.0122545518,
        ".*_index_finger_mcp_flex": 0.0377084683,
        ".*_index_finger_mcp_abd": 0.0177932607,
        ".*_index_finger_pip": 0.0297927367,
        ".*_index_finger_dip": 0.0161822467,
        ".*_middle_finger_mcp_flex": 0.0355644840,
        ".*_middle_finger_mcp_abd": 0.0240616629,
        ".*_middle_finger_pip": 0.0127755677,
        ".*_middle_finger_dip": 0.0109169800,
        ".*_ring_finger_mcp_flex": 0.0528257763,
        ".*_ring_finger_mcp_abd": 0.0266927477,
        ".*_ring_finger_pip": 0.0148677454,
        ".*_ring_finger_dip": 0.0215947297,
        ".*_pinky_mcp_flex": 0.0338251954,
        ".*_pinky_mcp_abd": 0.0261331247,
        ".*_pinky_pip": 0.0142008972,
        ".*_pinky_dip": 0.0129471168,
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
