"""Wuji Hand 1 dexterous hand configurations (left/right, full-mesh or primitive collision).

Twenty-DOF anthropomorphic hand from Wuji Technology (5 fingers x 4 DOF) with a silicone
pad on the thumb -- the ``hand/body-with-soft`` package, i.e. the hand the upstream
in-hand cube-reorientation RL task (``wuji-technology/wuji-mjlab``) is built around. The
successor generation is :mod:`eden.options.robots.wuji_hand2beta2`; the two are different
hands, not variants (different link names, joint names and kinematics).

Two collision flavours per side:

* ``WujiHand1_{R,L}`` -- one full visual mesh per link, reused as the collision mesh.
  Genesis convexifies each, so the palm becomes a single convex hull that bulges over the
  finger sockets. Fine for visualization, retargeting and teleop.
* ``WujiHand1Simplified_{R,L}`` -- primitive collision (cylinder + end spheres on the
  proximal links, simplified meshes for the palm, the thumb pad and the five distal
  links). This reproduces the collision model the upstream MJCF -- and therefore the
  upstream RL task -- was tuned against, and is much cheaper at thousands of envs. Use
  this flavour for contact-rich manipulation.

Assets (URDF + STL meshes, MIT-licensed) live under ``hands/wuji_hand1`` in the
``Kashu7100/eden_assets`` dataset, converted from the upstream
``wuji-technology/wuji-description`` ``hand/body-with-soft`` package.

Not to be confused with the older, unreferenced ``hands/wuji_hand`` directory in the same
dataset: that is the upstream ``hand/body`` package at v1.0.1 -- same kinematics, but no
thumb pad, no simplified-collision flavour, unprefixed link/joint names, and looser joint
limits and effort ceilings. Nothing in Eden loads it.
"""

from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions, MetadataOptions


class _WujiHand1(RobotOptions):
    """Shared 20-DOF kinematics and the manufacturer's actuator table.

    Every ``default_dofs_*`` dict below is keyed by side-agnostic regex
    (``.*_fingerN_jointM``) so both chiralities inherit one table. Note the dicts must
    *partition* the DOF set -- ``resolve_matching_names_values`` raises on a DOF matched
    by two keys -- so there is no ``"*"`` catch-all alongside the specific entries.
    """

    # Manufacturer keyframe (upstream ``wuji_hand_cfg.WUJI_HAND_HOME_KEYFRAME``): a
    # relaxed, grasp-ready pose. NOT zeros -- ``finger1_joint1`` has a lower limit of
    # +0.0475 rad, so an all-zero default would start the thumb outside its own range.
    default_dofs_pos: dict[str, float] = {
        ".*_finger1_joint1": 0.34,
        ".*_finger1_joint2": -0.08,
        ".*_finger1_joint3": 0.97,
        ".*_finger1_joint4": 0.78,
        ".*_finger2_joint1": 0.93,
        ".*_finger2_joint2": 0.17,
        ".*_finger2_joint3": 0.96,
        ".*_finger2_joint4": 0.80,
        ".*_finger3_joint1": 1.02,
        ".*_finger3_joint2": -0.02,
        ".*_finger3_joint3": 0.83,
        ".*_finger3_joint4": 0.76,
        ".*_finger4_joint1": 0.98,
        ".*_finger4_joint2": -0.16,
        ".*_finger4_joint3": 0.71,
        ".*_finger4_joint4": 0.88,
        ".*_finger5_joint1": 0.83,
        ".*_finger5_joint2": -0.21,
        ".*_finger5_joint3": 0.76,
        ".*_finger5_joint4": 1.23,
    }

    # Per-joint PD gains, effort ceilings and rotor inertia from the manufacturer's
    # system-identification table (``hand/body-with-soft/params.csv``, mirrored into the
    # upstream MJCF's ``position`` actuators as kp/kv/forcerange and its joint
    # ``armature``). These are NOT interchangeable with the blanket ``{"*": 50.0}`` gains
    # the other in-tree dexterous hands carry: this hand's fingers are driven at
    # kp ~0.2-0.7 Nm/rad against ~0.15-0.65 Nm actuators, three orders of magnitude
    # softer, and a stiff-gain override makes the fingertips behave like flywheels
    # instead of compliant contacts.
    default_dofs_kp: dict[str, float] = {
        ".*_finger1_joint1": 0.408447,
        ".*_finger1_joint2": 0.68586,
        ".*_finger1_joint3": 0.239111,
        ".*_finger1_joint4": 0.207361,
        ".*_finger2_joint1": 0.373522,
        ".*_finger2_joint2": 0.455924,
        ".*_finger2_joint3": 0.243684,
        ".*_finger2_joint4": 0.18027,
        ".*_finger3_joint1": 0.368709,
        ".*_finger3_joint2": 0.416425,
        ".*_finger3_joint3": 0.222186,
        ".*_finger3_joint4": 0.194276,
        ".*_finger4_joint1": 0.357182,
        ".*_finger4_joint2": 0.429773,
        ".*_finger4_joint3": 0.249302,
        ".*_finger4_joint4": 0.228503,
        ".*_finger5_joint1": 0.365533,
        ".*_finger5_joint2": 0.413931,
        ".*_finger5_joint3": 0.227294,
        ".*_finger5_joint4": 0.196472,
    }

    default_dofs_kd: dict[str, float] = {
        ".*_finger1_joint1": 0.020882,
        ".*_finger1_joint2": 0.0306109,
        ".*_finger1_joint3": 0.0101815,
        ".*_finger1_joint4": 0.00909699,
        ".*_finger2_joint1": 0.0188227,
        ".*_finger2_joint2": 0.0197982,
        ".*_finger2_joint3": 0.010477,
        ".*_finger2_joint4": 0.00824021,
        ".*_finger3_joint1": 0.0184862,
        ".*_finger3_joint2": 0.0180329,
        ".*_finger3_joint3": 0.0095922,
        ".*_finger3_joint4": 0.00915299,
        ".*_finger4_joint1": 0.0183766,
        ".*_finger4_joint2": 0.018677,
        ".*_finger4_joint3": 0.0105951,
        ".*_finger4_joint4": 0.0099176,
        ".*_finger5_joint1": 0.018617,
        ".*_finger5_joint2": 0.0187322,
        ".*_finger5_joint3": 0.00951005,
        ".*_finger5_joint4": 0.0090173,
    }

    # Joint effort limits, matching both the URDF ``<limit effort>`` and the MJCF
    # ``actuatorfrcrange``. Left at the hardware spec deliberately: the upstream RL task
    # normalizes its torque penalty by these, so widening them (as
    # ``wuji_hand2beta2`` does for teleop responsiveness) would silently rescale that
    # reward term.
    default_dofs_force_limits: dict[str, float] = {
        ".*_finger1_joint1": 0.4452,
        ".*_finger1_joint2": 0.4259,
        ".*_finger1_joint3": 0.1888,
        ".*_finger1_joint4": 0.1468,
        ".*_finger2_joint1": 0.6188,
        ".*_finger2_joint2": 0.1822,
        ".*_finger2_joint3": 0.2251,
        ".*_finger2_joint4": 0.217,
        ".*_finger3_joint1": 0.6494,
        ".*_finger3_joint2": 0.1827,
        ".*_finger3_joint3": 0.2078,
        ".*_finger3_joint4": 0.2018,
        ".*_finger4_joint1": 0.6389,
        ".*_finger4_joint2": 0.1832,
        ".*_finger4_joint3": 0.2249,
        ".*_finger4_joint4": 0.2044,
        ".*_finger5_joint1": 0.6441,
        ".*_finger5_joint2": 0.1798,
        ".*_finger5_joint3": 0.2384,
        ".*_finger5_joint4": 0.1866,
    }

    # Rotor inertia: 5e-4 on the six driven proximal joints (the thumb's two CMC joints
    # and each finger's MCP flexion), 2e-4 on the rest. URDF has no armature element, so
    # without this every joint would take Genesis' non-physical 0.1 morph default --
    # 200-500x too heavy here, enough that the fingers cannot grasp at all.
    # NOTE: under ``EnvOptions.integrator="euler"`` only, ``post_build()`` adds
    # ``kd * substep_dt`` for actuated entities, so the effective armature exceeds these
    # values; with kd ~0.01 that addition is ~1e-5 at a 1 ms substep, i.e. small against
    # 2e-4, but it grows with the substep size. The implicit integrators add nothing.
    default_dofs_armature: dict[str, float] = {
        ".*_finger1_joint[12]": 5.0e-4,
        ".*_finger[2-5]_joint1": 5.0e-4,
        ".*_finger1_joint[34]": 2.0e-4,
        ".*_finger[2-5]_joint[2-4]": 2.0e-4,
    }

    # The URDF declares no <dynamics> damping and params.csv lists damping=0 for every
    # joint, so joint damping is left at Genesis' zero default -- all dissipation is in
    # the PD kd term above.


# Finger numbering is the upstream convention: finger1 is the thumb, finger2..finger5 are
# index -> pinky. Each has 4 DOF: joint1/joint2 are the proximal pair (flexion +
# abduction; the thumb's are its two CMC axes) and joint3/joint4 the two distal flexion
# joints. The five empty ``*_tip_link`` frames are the fingertip frames kept for
# retargeting and for task-side contact/velocity queries.

_DOFS_NAME_TEMPLATE = [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
_TIP_LINKS_TEMPLATE = [f"finger{f}_tip_link" for f in range(1, 6)]


def _sided(names: list[str], side: str) -> list[str]:
    return [f"{side}_{name}" for name in names]


@ROBOT_REGISTRY.register(name="wuji_hand1_r")
class WujiHand1_R(_WujiHand1):
    file: str = "hands/wuji_hand1/wuji_hand1_right.urdf"

    # No ``up`` / ``front`` override: this URDF already satisfies Eden's defaults
    # (``up=(0, 0, 1)``, ``front=(1, 0, 0)``). Verified by FK -- the MCP joint origins and
    # the fingertip frames sit at positive z (fingers extend along URDF +Z) and positive
    # flexion displaces the fingertips along URDF +X (the palm normal). Both hold for
    # BOTH chiralities here, because the left URDF mirrors about the XZ plane, which
    # leaves the palm normal on +X. (Contrast wuji_hand2beta2, whose mirror flips
    # ``front``.) Pinned in ``tests/options/robots/test_wuji_hand1.py``.

    dofs_name: list[str] = _sided(_DOFS_NAME_TEMPLATE, "right")

    # ``*_tip_link`` are massless fixed children of ``*_link4``, and
    # ``finger1_link2_softbody`` is the thumb's silicone pad (a fixed child of link2 that
    # carries the pad's own collision mesh). Both would be merged into their parents
    # without this.
    links_to_keep: list[str] = _sided(_TIP_LINKS_TEMPLATE, "right") + ["right_finger1_link2_softbody"]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # NOTE: the order of ``target_hand_links`` matches the order of ``mano_hand_idx``.
        target_hand_links=_sided(_TIP_LINKS_TEMPLATE, "right"),
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


@ROBOT_REGISTRY.register(name="wuji_hand1_l")
class WujiHand1_L(_WujiHand1):
    file: str = "hands/wuji_hand1/wuji_hand1_left.urdf"

    # Same axes as the right hand -- see the note there.

    dofs_name: list[str] = _sided(_DOFS_NAME_TEMPLATE, "left")

    links_to_keep: list[str] = _sided(_TIP_LINKS_TEMPLATE, "left") + ["left_finger1_link2_softbody"]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        # NOTE: the order of ``target_hand_links`` matches the order of ``mano_hand_idx``.
        target_hand_links=_sided(_TIP_LINKS_TEMPLATE, "left"),
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


# --- primitive-collision flavours ------------------------------------------- #
# Upstream also ships ``{side}_simplified.urdf``: the same 20-DOF hand, same link and
# joint names, same inertials, differing only in the <collision> geometry -- a cylinder
# plus two end spheres (a capsule in all but name) on each proximal link, and reduced
# meshes for the palm, the thumb pad and the five distal links. That is the collision
# model the upstream MJCF and its trained reorientation policy use, so these are the
# flavours to reach for in contact-rich RL: the full-mesh palm hull bulges over the
# finger sockets, and mesh-mesh contact costs far more per step.


@ROBOT_REGISTRY.register(name="wuji_hand1_simplified_r")
class WujiHand1Simplified_R(WujiHand1_R):
    file: str = "hands/wuji_hand1/wuji_hand1_right_simplified.urdf"


@ROBOT_REGISTRY.register(name="wuji_hand1_simplified_l")
class WujiHand1Simplified_L(WujiHand1_L):
    file: str = "hands/wuji_hand1/wuji_hand1_left_simplified.urdf"
