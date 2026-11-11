"""Frozen I/O contract of the HUSKY skateboarding checkpoint (G1 23-DOF).

Everything here mirrors the mjlab training stack of `TeleHuman/humanoid_skateboarding
<https://github.com/TeleHuman/humanoid_skateboarding>`_ (CC BY-NC 4.0) and was
cross-checked against three sources that all agree:

- the compiled training scene ``test_scene/mjlab_scene.xml`` (joint order,
  armature, actuator gainprm/biasprm, force ranges, init keyframe),
- the metadata embedded in the released ``ckpts/test.onnx`` (``joint_names``,
  ``joint_stiffness``, ``joint_damping``, ``default_joint_pos``,
  ``action_scale``, ``observation_names``),
- the standalone deployment script ``test_scene/sim.py`` (observation layout,
  scales, history handling, action transform).

The joint order is the MJCF declaration order of the skater G1 — identical to
Eden's ``G1_UCSD_NO_WRIST_DOFS_NAME`` (3-DOF waist, wrists welded) — so no
permutation is needed anywhere. mjlab derives kp/kd/armature from the same
Unitree motor specs and Raibert heuristic (10 Hz natural frequency, damping
ratio 2) that Eden's actuator spec classes encode, and the derived numbers are
equal to the ONNX metadata to all published digits (pinned by
``tests/extensions/policies/husky/test_constants.py``), so the gains below are built from the Eden
spec classes rather than baked floats.
"""

from __future__ import annotations

import fnmatch
import math

from eden.options.actuators import (
    ActuatorSpecOptions,
    Unitree5020_16,
    Unitree5020_16_DOUBLE,
    Unitree7520_14,
    Unitree7520_22,
)

# MJCF declaration order of the 23 actuated joints — the order of the policy's
# joint_pos / joint_vel / action vectors.
POLICY_JOINT_NAMES: tuple[str, ...] = (
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
)

NUM_JOINTS: int = len(POLICY_JOINT_NAMES)

# Per-joint actuator assignment (the 29-DOF spec minus the welded wrists).
HUSKY_G1_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    "*_hip_pitch_joint": Unitree7520_14,
    "*_hip_roll_joint": Unitree7520_22,
    "*_hip_yaw_joint": Unitree7520_14,
    "*_knee_joint": Unitree7520_22,
    "*_ankle_*_joint": Unitree5020_16_DOUBLE,
    "waist_yaw_joint": Unitree7520_14,
    "waist_roll_joint": Unitree5020_16_DOUBLE,
    "waist_pitch_joint": Unitree5020_16_DOUBLE,
    "*_shoulder_*_joint": Unitree5020_16,
    "*_elbow_joint": Unitree5020_16,
}

# The training init keyframe ("push" stance: left foot planted on the deck,
# right leg cocked to push, arms posed). It doubles as the policy's default
# joint offset: joint_pos observations subtract it and PD targets add it back.
PUSH_KEYFRAME_DOFS_POS: dict[str, float] = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.23,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.7,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 1.17,
    "right_ankle_pitch_joint": -0.45,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": -0.03,
    "left_shoulder_roll_joint": 0.45,
    "left_shoulder_yaw_joint": -0.21,
    "left_elbow_joint": 1.32,
    "right_shoulder_pitch_joint": -0.7,
    "right_shoulder_roll_joint": -0.845,
    "right_shoulder_yaw_joint": 0.83,
    "right_elbow_joint": 1.19,
}

# Training placement: pelvis root relative to the deck root. The deck sits at
# z=0.10 (wheels touching the ground) and the pelvis starts 0.10 m to the
# board's left so the left foot lands on the deck's rear marker.
SKATEBOARD_DEFAULT_ROOT_POS: tuple[float, float, float] = (0.0, 0.0, 0.10)
ROBOT_DEFAULT_ROOT_POS: tuple[float, float, float] = (-0.03, 0.1, 0.78)


def _resolve(patterns: dict[str, type[ActuatorSpecOptions]], attr: str) -> tuple[float, ...]:
    """Resolve a per-pattern actuator attribute to a per-joint tuple (first match wins)."""
    values: list[float] = []
    for joint in POLICY_JOINT_NAMES:
        for pattern, spec in patterns.items():
            if fnmatch.fnmatch(joint, pattern):
                value = getattr(spec, attr)
                values.append(value() if callable(value) else value)
                break
        else:
            raise KeyError(f"No actuator spec pattern matches joint {joint!r}")
    return tuple(values)


# Resolved per-joint vectors in POLICY_JOINT_NAMES order.
DEFAULT_DOFS_POS: tuple[float, ...] = tuple(PUSH_KEYFRAME_DOFS_POS[name] for name in POLICY_JOINT_NAMES)
ACTION_SCALE: tuple[float, ...] = _resolve(HUSKY_G1_DOFS_SPEC, "RAIBERT_HEURISTIC_ACTION_SCALE")
DOFS_KP: tuple[float, ...] = _resolve(HUSKY_G1_DOFS_SPEC, "RAIBERT_HEURISTIC_KP")
DOFS_KD: tuple[float, ...] = _resolve(HUSKY_G1_DOFS_SPEC, "RAIBERT_HEURISTIC_KD")

# --- Observation contract ------------------------------------------------
# One frame is [command(2) heading(1) base_ang_vel(3) projected_gravity(3)
# joint_pos_rel(23) joint_vel(23) last_action(23) phase(1)] = 79 dims. The
# actor input stacks a 5-frame history PER TERM: each term's history is
# flattened time-major (oldest first) and the flattened blocks are then
# concatenated in term order — NOT frame-major. 5 x 79 = 395.
OBS_BLOCK_DIMS: tuple[int, ...] = (2, 1, 3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS, 1)
FRAME_DIM: int = sum(OBS_BLOCK_DIMS)
HISTORY_LENGTH: int = 5
ACTOR_OBS_DIM: int = FRAME_DIM * HISTORY_LENGTH

COMMAND_SCALE: tuple[float, float] = (2.0, 1.0)  # (lin_vel_x, relative heading)
HEADING_SCALE: float = 1.0 / math.pi
ANG_VEL_SCALE: float = 0.25
DOFS_VEL_SCALE: float = 0.05

# --- Gait phase clock -----------------------------------------------------
# The policy is phase-conditioned: phase = (step * 0.02 / 6.0) % 1 sweeps one
# push→transition→steer→transition cycle every 6 s. While the velocity command
# is ~zero the training env pins the clock at phase 0 ("still" envs).
CYCLE_TIME_S: float = 6.0
STILL_VX_THRESHOLD: float = 0.1
