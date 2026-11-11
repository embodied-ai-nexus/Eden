"""Unitree G1 humanoid configurations (23/29 DOF variants)."""

from __future__ import annotations
from typing import ClassVar, Any

from genesis.typing import Vec3FType

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.actuators import (
    ActuatorSpecOptions,
    Unitree4010_25,
    Unitree5020_16,
    Unitree5020_16_DOUBLE,
    Unitree7520_14,
    Unitree7520_22,
)
from eden.utils.motion_reference.bipedal_ik import BipedalIKProfile


G1_DOFS23_NAME: list[str] = [
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]


G1_UCSD_NO_WRIST_DOFS_NAME: list[str] = [
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


# The ankle-roll links are the foot frames: what the robot stands on, what locomotion terms
# measure against, and what Genesis must not merge away (hence ``links_to_keep`` too).
G1_FEET_LINKS_NAME: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")


@ROBOT_REGISTRY.register()
class G1Dofs23(RobotOptions):
    """Unitree 23 DOF G1 Configuration."""

    file: str = "humanoids/unitree_g1/g1_23dof.xml"
    links_to_keep: list[str] = list(G1_FEET_LINKS_NAME)
    feet_links_name: ClassVar[tuple[str, ...]] = G1_FEET_LINKS_NAME

    dofs_name: list[str] = G1_DOFS23_NAME
    default_root_pos: Vec3FType = (0.0, 0.0, 0.783675)  # x,y,z [m]
    metadata: ClassVar[dict[str, Any]] = {
        "robot_height": 1.32,
    }


@ROBOT_REGISTRY.register()
class G1Dofs29NoWristDof(G1Dofs23):
    """23-active-DOF G1 built from the 29-DOF G1 with wrists welded (the AMO / GMT training robot).

    Same active-DOF count as :class:`G1Dofs23` but a different joint set: the default ``g1_23dof.xml`` is the
    official ``g1_23dof_rev_1_0`` (1-DOF waist + ``wrist_roll``), whereas this keeps the full 3-DOF waist
    (yaw/roll/pitch) and drops both wrists. ``file`` and ``dofs_name`` are bound here so they can't drift.
    """

    file: str = "humanoids/unitree_g1/g1_29dof_wrist_disabled.xml"
    dofs_name: list[str] = G1_UCSD_NO_WRIST_DOFS_NAME


G1_DOFS29_NAME: list[str] = [
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
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# Link order used as the motion-tracking target set by G1 motion datasets
# (PHUMA, LAFAN1, SEED, AMASS retargets). Datasets reference this directly
# rather than carrying their own copy, so the order stays in lockstep with
# whatever the retargeting pipeline emitted.
G1_DOFS29_TRACKED_LINKS_NAME: list[str] = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


# Single source of truth for per-joint actuator assignments on the 29DoF G1.
# Waist roll/pitch and ankle roll/pitch use ``Unitree5020_16_DOUBLE`` to model
# the parallel 4-bar linkage driven by two 5020 motors (effort and reflected
# inertia are summed at the joint; speed is preserved at the assumed 1:1
# linkage ratio). All Raibert-heuristic kp/kd/action-scale values below are
# derived from this dict, so changing actuator assignments here propagates
# automatically to control gains and action scales.
_G1Dofs29_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    "*_knee_joint": Unitree7520_22,
    "*_hip_roll_joint": Unitree7520_22,
    "*_hip_pitch_joint": Unitree7520_14,
    "*_hip_yaw_joint": Unitree7520_14,
    "*_ankle_*_joint": Unitree5020_16_DOUBLE,
    "waist_yaw_joint": Unitree7520_14,
    "waist_roll_joint": Unitree5020_16_DOUBLE,
    "waist_pitch_joint": Unitree5020_16_DOUBLE,
    "*_shoulder_*_joint": Unitree5020_16,
    "*_elbow_joint": Unitree5020_16,
    "*_wrist_roll_joint": Unitree5020_16,
    "*_wrist_pitch_joint": Unitree4010_25,
    "*_wrist_yaw_joint": Unitree4010_25,
}


@ROBOT_REGISTRY.register()
class G1Dofs29(RobotOptions):
    """Unitree G1 29DoF Configuration."""

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = _G1Dofs29_DOFS_SPEC

    file: str = "humanoids/unitree_g1/g1_29dof_rev_1_0.xml"
    links_to_keep: list[str] = list(G1_FEET_LINKS_NAME)
    feet_links_name: ClassVar[tuple[str, ...]] = G1_FEET_LINKS_NAME

    dofs_name: list[str] = G1_DOFS29_NAME
    default_root_pos: Vec3FType = (0.0, 0.0, 0.783675)  # x,y,z [m]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "*_hip_pitch_joint": -0.1,
        "*_hip_roll_joint": 0.0,
        "*_hip_yaw_joint": 0.0,
        "*_knee_joint": 0.3,
        "*_ankle_pitch_joint": -0.2,
        "*_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "*_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.15,
        "*_shoulder_yaw_joint": 0.0,
        "*_elbow_joint": 1.5,
        "*_wrist_roll_joint": 0.0,
        "*_wrist_pitch_joint": 0.0,
        "*_wrist_yaw_joint": 0.0,
        "right_shoulder_roll_joint": -0.15,
    }
    # Raibert helpers below are evaluated at class-definition time using the
    # default ``natural_freq=NATURAL_FREQ`` (10 Hz). If you tune NATURAL_FREQ
    # globally these dicts won't follow — rebuild them on the subclass.
    default_dofs_kp: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KP() for p, s in _G1Dofs29_DOFS_SPEC.items()}
    default_dofs_kd: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KD() for p, s in _G1Dofs29_DOFS_SPEC.items()}
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in _G1Dofs29_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in _G1Dofs29_DOFS_SPEC.items()}

    metadata: ClassVar[dict[str, Any]] = {
        "robot_height": 1.32,
    }


@ROBOT_REGISTRY.register()
class G1Dofs29NoHands(G1Dofs29):
    """Unitree G1 29DoF Configuration with no hands."""

    file: str = "humanoids/unitree_g1/g1_29dof_rev_1_0_no_hand.xml"


# Dex3 hand joints per side, in the MJCF document order of the GEAR-SONIC
# ``g1_29dof_with_hand`` model. The order is load-bearing: the reference sim
# maps Dex3 motor command i onto the i-th hand joint in document order, and
# both sides follow the same thumb(3)/middle(2)/index(2) pattern in that
# model (the deploy repo's variant of the same file orders the right hand
# differently — do not swap the asset).
G1_DEX3_LEFT_DOFS_NAME: list[str] = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
G1_DEX3_RIGHT_DOFS_NAME: list[str] = [name.replace("left_", "right_", 1) for name in G1_DEX3_LEFT_DOFS_NAME]


@ROBOT_REGISTRY.register()
class G1Dofs29Dex3(G1Dofs29):
    """Unitree G1 29DoF + two 7-DOF Dex3 hands (43 actuated DOFs) — the SONIC teleop sim body.

    Body joints keep the :class:`G1Dofs29` actuator spec. Hand gains are the
    Dex3 controller's values (kp 1.5 / kd 0.1, ``dex3_hands.hpp``); hand
    armature and force limits come from the source MJCF (armature 0.01,
    ``actuatorfrcrange`` ±2.45 thumb base / ±1.4 other fingers).

    The MJCF is the GEAR-SONIC model, kept separate from ``g1_29dof_rev_1_0``:
    the SONIC calibration offsets are pinned to this model's ``torso_link``
    frame (10 mm lower than rev_1_0's). ``file`` is the single source of truth
    for the asset path — the operator-side FK and visualizer read it too.
    """

    file: str = "humanoids/unitree_g1/g1_29dof_with_hand_sonic.xml"

    dofs_name: list[str] = G1_DOFS29_NAME + G1_DEX3_LEFT_DOFS_NAME + G1_DEX3_RIGHT_DOFS_NAME
    default_dofs_pos: dict[str, float] = {**G1Dofs29().default_dofs_pos, "*_hand_*_joint": 0.0}
    default_dofs_kp: dict[str, float] = {**G1Dofs29().default_dofs_kp, "*_hand_*_joint": 1.5}
    default_dofs_kd: dict[str, float] = {**G1Dofs29().default_dofs_kd, "*_hand_*_joint": 0.1}
    default_dofs_armature: dict[str, float] = {**G1Dofs29().default_dofs_armature, "*_hand_*_joint": 0.01}
    # Patterns must be disjoint (a joint matching two patterns is an error).
    default_dofs_force_limits: dict[str, float] = {
        **G1Dofs29().default_dofs_force_limits,
        "*_hand_thumb_0_joint": 2.45,
        "*_hand_thumb_[12]_joint": 1.4,
        "*_hand_middle_*_joint": 1.4,
        "*_hand_index_*_joint": 1.4,
    }


# Per-joint action scale = ``0.25 * EFFORT_LIMIT / kp``; same source-of-truth
# spec dict, same Raibert heuristic. Use this for ``ActionTerm.configure(scale=...)``.
G1_ACTION_SCALE: dict[str, float] = {p: s.RAIBERT_HEURISTIC_ACTION_SCALE() for p, s in _G1Dofs29_DOFS_SPEC.items()}


# Robot-agnostic IK spec used by the smooth-start-transition modifier
# (``eden.datasets.modifiers.smooth_start_end``). Stance qpos and per-DOF
# bias weights are derived at apply-time from ``robot_options`` and the
# fnmatch patterns below — see ``BipedalIKProfile``'s resolvers.
#
# Foot contact-point offsets are the ``pos`` values from the G1 MJCF's
# ``left_foot`` / ``right_foot`` site declarations. Carrying them on the
# profile (instead of resolving from MJCF sites at runtime) keeps the IK
# Genesis-native and lets the same machinery serve URDF-based robots
# without a sites concept.
G1_IK_PROFILE: BipedalIKProfile = BipedalIKProfile(
    robot_options=G1Dofs29(),
    left_foot_link="left_ankle_roll_link",
    right_foot_link="right_ankle_roll_link",
    left_foot_local_offset=(0.04, 0.0, -0.037),
    right_foot_local_offset=(0.04, 0.0, -0.037),
    upper_body_joint_patterns=(
        "*_shoulder_*_joint",
        "*_elbow_joint",
        "*_wrist_*_joint",
    ),
    # OpenTrack's W_pos profile, expressed in joint-name patterns so it
    # survives MJCF refactors. Base yaw is the only base override; legs free
    # roll/yaw to be zeroed out; pitch joints stay at the default 1.0.
    base_lin_bias_weights=(1.0, 1.0, 1.0),
    base_ang_bias_weights=(1.0, 1.0, 2.5),
    joint_bias_patterns={
        "*_hip_roll_joint": 0.0,
        "*_hip_yaw_joint": 0.0,
        "*_ankle_roll_joint": 0.0,
    },
    qpos_bias_scale=0.5,
    ee_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    damping=1e-3,
    dq_bound=10.0,
)
