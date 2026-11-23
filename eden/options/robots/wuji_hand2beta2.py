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


# --- identified actuator calibration ---------------------------------------- #
# From wuji-mjlab v2026.10.22, ``tasks/pen_spin/config/wuji_hand2/hand2_constants.py``:
# the calibration Wuji identified on a right Hand 2 for its pen-spin policy. Upstream
# loads it onto ``wuji-description`` ``hand2/hand2_2``, the release that replaced
# ``hand2_beta2``. The two MJCFs declare the same joint names, axes, ranges and actuator
# limits; only the finger-root placements differ (by about 2 mm), so the table applies to
# this hand joint by joint. The left hand reuses the right hand's values, since no left
# hand was identified. Every tuple below is in the SDK's finger-major order, which is
# also the order of ``dofs_name``.
_JOINTS = (
    "thumb_cmc_flex",
    "thumb_cmc_abd",
    "thumb_mcp",
    "thumb_ip",
    *(
        f"{finger}_{joint}"
        for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")
        for joint in ("mcp_flex", "mcp_abd", "pip", "dip")
    ),
)

# The hand runs an MIT impedance loop in firmware. Upstream deploys it at kp 3.0 and the
# per-position kd below, with a 1.5 A current limit, and converts MIT units to joint torque
# through each motor's output torque constant (N*m/A) and the SDK's amps per MIT unit.
_MIT_KP = 3.0
_MIT_KD = (0.075, 0.07, 0.05, 0.05)  # joints 1-4 of every finger: flexion, abduction, then the two distal joints
_CURRENT_LIMIT_A = 1.5
_AMPS_PER_MIT_UNIT = 1.9114709702965849
_TORQUE_CONSTANT = (
    0.4136, 0.3545, 0.2216, 0.1329,  # thumb
    0.4136, 0.1673, 0.1673, 0.1673,  # index
    0.4136, 0.1673, 0.1673, 0.1673,  # middle
    0.4136, 0.1673, 0.1673, 0.1673,  # ring
    0.4461, 0.1115, 0.1115, 0.1115,  # pinky
)  # fmt: skip

_ARMATURE = (
    1.457162e-4, 1.366389e-4, 1.273553e-4, 8.768902e-5,
    1.130474e-4, 5.127161e-5, 4.843783e-5, 7.906193e-5,
    2.011048e-4, 7.106655e-5, 6.054746e-5, 5.659272e-5,
    1.393897e-4, 3.544108e-5, 5.233974e-5, 6.047948e-5,
    9.706259e-5, 3.826054e-5, 4.664456e-5, 4.743778e-5,
)  # fmt: skip

_DAMPING = (
    0.0026915050, 0.0042812198, 0.0014135711, 0.0005131806,
    0.0010510923, 0.0, 0.0010302809, 0.0000663526,
    0.0021587672, 0.0008842859, 0.0012610488, 0.0001128493,
    0.0014506542, 0.0, 0.0007149635, 0.0,
    0.0, 0.0, 0.0016246495, 0.0,
)  # fmt: skip

_FRICTIONLOSS = (
    0.0367355430, 0.0342266392, 0.0187921182, 0.0122545518,
    0.0377084683, 0.0177932607, 0.0297927367, 0.0161822467,
    0.0355644840, 0.0240616629, 0.0127755677, 0.0109169800,
    0.0528257763, 0.0266927477, 0.0148677454, 0.0215947297,
    0.0338251954, 0.0261331247, 0.0142008972, 0.0129471168,
)  # fmt: skip


def _per_joint(values) -> dict[str, float]:
    """Key one value per joint by a side-agnostic pattern, so both chiralities read one table."""
    return {f".*_{joint}": float(value) for joint, value in zip(_JOINTS, values, strict=True)}


class _WujiHand2Beta2(RobotOptions):
    """Shared 20-DOF kinematics and the identified actuator calibration (see the tables above)."""

    default_dofs_pos: dict[str, float] = {"*": 0.0}

    # Joint-space PD gains equal to the firmware's MIT gains: kp = kt * A/unit * MIT kp, and
    # kd likewise. kp runs 0.64-2.56 N*m/rad and kd 0.011-0.064 N*m*s/rad.
    default_dofs_kp: dict[str, float] = _per_joint(kt * _AMPS_PER_MIT_UNIT * _MIT_KP for kt in _TORQUE_CONSTANT)

    default_dofs_kd: dict[str, float] = _per_joint(
        kt * _AMPS_PER_MIT_UNIT * _MIT_KD[i % 4] for i, kt in enumerate(_TORQUE_CONSTANT)
    )

    # The torque the 1.5 A current limit allows, kt * 1.5: 0.17-0.67 N*m. Upstream also caps
    # each joint's total actuator force at the older pen-spin values (0.21-1.06 N*m), but
    # those are larger on every joint, so this limit is the one that binds.
    default_dofs_force_limits: dict[str, float] = _per_joint(kt * _CURRENT_LIMIT_A for kt in _TORQUE_CONSTANT)

    # Identified rotor inertia, 3.5e-5 to 2.0e-4. The URDF carries no armature, so without
    # this every joint would take Genesis' 0.1 morph default, 500 to 2800 times heavier.
    # Under ``EnvOptions.integrator="euler"`` only, ``post_build()`` adds ``kd * substep_dt``
    # to it; at kd 0.06 and a 1 ms substep that is 6e-5, a large fraction of these values.
    default_dofs_armature: dict[str, float] = _per_joint(_ARMATURE)

    # Identified passive joint damping and dry friction. The URDF declares neither.
    default_dofs_damping: dict[str, float] = _per_joint(_DAMPING)

    default_dofs_frictionloss: dict[str, float] = _per_joint(_FRICTIONLOSS)


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
