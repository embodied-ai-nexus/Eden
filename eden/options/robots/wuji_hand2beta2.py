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

    default_dofs_kp: dict[str, float] = {"*": 50.0}

    default_dofs_kd: dict[str, float] = {"*": 5.0}

    # Per-joint effort ceiling. The URDF/MJCF joint efforts (0.2-2.0 Nm) leave the
    # PD torque-starved (the kd term eats most of the budget), so the fingers can't
    # close at hand speed. Widened for teleop / manipulation responsiveness, matching
    # the sibling dexterous hands. NOTE: this exceeds the real Wuji actuator spec —
    # override it if you need hardware-faithful torques (the MJCF forceranges are
    # 0.6 Nm thumb CMC, 2.0 Nm finger MCP-flex, 0.2-0.3 Nm elsewhere).
    default_dofs_force_limits: dict[str, float] = {"*": 5.0}

    # Rotor inertia: a uniform 5e-4, the manufacturer MJCF's value for the driven
    # CMC/MCP-flex joints (``mjcf/{right,left}.xml`` joint ``armature``; it declares 2e-4 on
    # the rest). The URDF carries no armature field, so without this every joint would take
    # Genesis' non-physical 0.1 morph default (~200x heavier) and the fingers would barely move.
    default_dofs_armature: dict[str, float] = {"*": 5.0e-4}


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
