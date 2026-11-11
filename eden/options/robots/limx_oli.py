"""LimX Dynamics Oli (HU_D04) humanoid configuration (31 DoF)."""

from __future__ import annotations
from typing import ClassVar, Any

import numpy as np
from genesis.typing import Vec3FType

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.actuators import (
    ActuatorSpecOptions,
    LimxOliHipKnee,
    LimxOliAnkleWaist,
    LimxOliShoulderElbow,
    LimxOliWristHead,
)
from eden.utils.actuator import (
    raibert_heuristic_kp,
    raibert_heuristic_kd,
    raibert_heuristic_action_scale,
)


# 31 actuated joints in LimX's canonical order (HU_D04_01 config.yaml); hands excluded.
OLI_DOFS_NAME: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]


# Per-joint actuator assignment; the gain/armature/action-scale dicts below all
# derive from it. ankle + waist are parallel-linkage joints (see LimxOliAnkleWaist).
_OLI_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    "*_hip_*_joint": LimxOliHipKnee,
    "*_knee_joint": LimxOliHipKnee,
    "*_ankle_*_joint": LimxOliAnkleWaist,
    "waist_*_joint": LimxOliAnkleWaist,
    "*_shoulder_*_joint": LimxOliShoulderElbow,
    "*_elbow_joint": LimxOliShoulderElbow,
    "*_wrist_*_joint": LimxOliWristHead,
    "head_*_joint": LimxOliWristHead,
}


# Oli's shipped gains fit the Raibert heuristic at omega_n = 2*pi*5 (zeta=2),
# reverse-engineered from the deployed controllers: kp/kd is constant across joint
# groups and kp = armature*omega_n**2 matches the RL MJCF armatures. Half of Eden's
# global NATURAL_FREQ (10 Hz); passed explicitly below to reproduce the shipped gains.
OLI_NATURAL_FREQ: float = 2.0 * np.pi * 5.0  # rad/s


# The ankle-roll links are the foot frames: what Oli stands on, what every locomotion term
# resolves, and what Genesis must not merge away — hence they feed both ``feet_links_name`` and
# ``links_to_keep``. ``OliNoHands`` extends the *latter* with its hand mounts (which are not
# feet) rather than restating it, so a link added here reaches both variants.
OLI_FEET_LINKS_NAME: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

# Where a bring-up gantry hangs Oli: the two shoulders. These are the most proximal link of each
# shoulder, so their frame origins sit on the shoulder-pitch axis at the top of the torso and do
# not move as the arms swing — the rig stays put through a clip that throws the arms around.
OLI_GANTRY_LINKS_NAME: tuple[str, ...] = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")


@ROBOT_REGISTRY.register()
class Oli(RobotOptions):
    """LimX Dynamics Oli (HU_D04) 31-DoF humanoid configuration.

    Assets: ``humanoids/limx_oli`` in the ``Kashu7100/eden_assets`` dataset. The
    serial ``HU_D04_01.urdf`` is used (its 31-DoF tree matches ``dofs_name``);
    the sibling ``HU_D04_01.xml`` MJCF is the *parallel* model whose ankle/waist
    are expressed as ``*_achilles`` / ``waist_A/B`` closed-loop joints and does
    not expose the serial ankle_pitch/roll DoFs directly.
    """

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = _OLI_DOFS_SPEC

    file: str = "humanoids/limx_oli/HU_D04_01.urdf"
    links_to_keep: list[str] = list(OLI_FEET_LINKS_NAME)
    feet_links_name: ClassVar[tuple[str, ...]] = OLI_FEET_LINKS_NAME
    gantry_links_name: ClassVar[tuple[str, ...]] = OLI_GANTRY_LINKS_NAME

    dofs_name: list[str] = OLI_DOFS_NAME
    default_root_pos: Vec3FType = (0.0, 0.0, 0.903)  # pelvis height, feet flat [m]

    # Deployed WalkHumanoidController default pose, as a feet-flat symmetric stance:
    # ankle_roll=0 (shipped ±0.16 is a policy-frame offset) and ankle_pitch=-0.16 on
    # both (URDF ankle_pitch axis is non-mirrored, so symmetry needs equal values).
    default_dofs_pos: dict[str, float] = {
        "left_hip_pitch_joint": -0.16,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": -0.05,
        "left_knee_joint": 0.28,
        "left_ankle_pitch_joint": -0.16,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.16,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.05,
        "right_knee_joint": 0.28,
        "right_ankle_pitch_joint": -0.16,  # shipped +0.16 (policy frame)
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "head_yaw_joint": 0.0,
        "head_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": -0.063,
        "left_shoulder_roll_joint": 0.146,
        "left_shoulder_yaw_joint": -0.297,
        "left_elbow_joint": -0.086,
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": -0.063,
        "right_shoulder_roll_joint": -0.146,
        "right_shoulder_yaw_joint": 0.297,
        "right_elbow_joint": -0.086,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
    }

    # Derived from dofs_spec via the Raibert heuristic at OLI_NATURAL_FREQ (5 Hz).
    default_dofs_kp: dict[str, float] = {
        p: raibert_heuristic_kp(s, OLI_NATURAL_FREQ) for p, s in _OLI_DOFS_SPEC.items()
    }
    default_dofs_kd: dict[str, float] = {
        p: raibert_heuristic_kd(s, OLI_NATURAL_FREQ) for p, s in _OLI_DOFS_SPEC.items()
    }
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in _OLI_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in _OLI_DOFS_SPEC.items()}

    metadata: ClassVar[dict[str, Any]] = {
        "robot_height": 1.65,  # nominal standing height [m]
    }


@ROBOT_REGISTRY.register(name="oli_no_hands")
class OliNoHands(Oli):
    """Oli with its moulded rubber hands removed, for welding a real end effector.

    Teleoperating a Dex3 / XHand / Dex1 gripper (or a Wuji Hand2 — see
    :class:`~eden.options.robots.composite.oli_wuji.OliWuji`) means welding that end effector
    onto the wrist, and the stock moulded hand would then sit stacked inside it. This variant
    points at ``HU_D04_01_no_hand.urdf``; every joint and every other link is identical to the
    stock URDF, so DOFs, gains and pose are inherited unchanged from :class:`Oli`. Four things
    differ:

    - ``*_wrist_roll_link`` points at ``*_wrist_roll_link_nohand.STL``. This is what removes the
      hand: it is baked into the stock wrist mesh, not carried on a link, and so is the silver
      end-cap it bolted onto. Dropping both leaves 72 of 322 cm^3 and exposes the real flange.
    - The wrist inertial is **recomputed for the bare metal** — 0.205 kg with its CoM 2.9 cm
      down, against the stock 0.581 kg at 8.9 cm, which was dominated by the rubber hand. So
      the arm carries the end effector you weld on, not that plus a phantom 0.38 kg per wrist.
    - Two ``*_hand_mount`` frames are **added** on the exposed flange (z = -0.058 in
      ``*_wrist_roll_link``, the plane upstream's own stock URDF puts ``*_hand_contact`` on).
      They carry no mass and no geometry; welding an end effector to them needs no offset.
    - The four fixed ``*_hand_{manip,contact}`` links/joints are dropped. Despite the names
      these are not the hand, just 1 cm / 1 g marker boxes for the grasp and contact frames
      LimX's stack targets; they are fixed children Genesis merges away regardless.
    """

    file: str = "humanoids/limx_oli/HU_D04_01_no_hand.urdf"

    # Genesis merges fixed childless links, so the two flange frames must be named here or they
    # vanish at build and every mount lookup fails. Extends ``OLI_FEET_LINKS_NAME`` rather than
    # restating it, so a foot link added there still reaches this variant.
    links_to_keep: list[str] = [*OLI_FEET_LINKS_NAME, "left_hand_mount", "right_hand_mount"]


# Per-joint action scale (0.25 * effort / kp) at OLI_NATURAL_FREQ; for ActionTerm scale.
OLI_ACTION_SCALE: dict[str, float] = {
    p: raibert_heuristic_action_scale(s, OLI_NATURAL_FREQ) for p, s in _OLI_DOFS_SPEC.items()
}
