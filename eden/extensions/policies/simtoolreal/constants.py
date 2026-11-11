"""Frozen I/O contract for the SimToolReal pretrained policy.

Source of truth: ``tylerlum/simtoolreal``
``isaacgymenvs/utils/observation_action_utils_sharpa.py`` (the deploy-time
numpy reference) and ``isaacsimenvs/cfg/task/SimToolReal.yaml``.

Robot: KUKA iiwa14 (7 DOF) + SharPa Wave left hand (22 DOF) = 29 DOF.
Policy runs at 60 Hz (sim 120 Hz, decimation 2).

DOF ORDERING GOTCHA
-------------------
``JOINT_NAMES_CANON`` is IsaacGym's BFS-then-alphabetical actuated-joint order
(the ``left_{1..5}_`` prefixes were added upstream precisely to force this
order). Genesis loads the same 29 joints in a *different* native order, so the
port indexes every read/write by name (``find_named_dofs_idx_local(ARM_DOFS / HAND_DOFS,
preserve_order=True)`` on the arm / hand entity) and never assumes the native order.
The policy obs/action are all expressed in ``JOINT_NAMES_CANON`` order.
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# Joints (canonical / policy order)
# ---------------------------------------------------------------------------

ARM_JOINT_NAMES: list[str] = [f"iiwa14_joint_{i}" for i in range(1, 8)]

HAND_JOINT_NAMES: list[str] = [
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

JOINT_NAMES_CANON: list[str] = ARM_JOINT_NAMES + HAND_JOINT_NAMES
NUM_JOINTS: int = 29
NUM_ARM_JOINTS: int = 7
NUM_HAND_JOINTS: int = 22
assert len(JOINT_NAMES_CANON) == NUM_JOINTS

# The names above are upstream's, from the merged URDF, and stay the labels for the frozen
# obs/action order (the limit arrays below are indexed by them). The composite's stock assets
# drop the ``iiwa14_`` prefix and the ordering digit upstream injected into five proximal hand
# joints. Derived, not restated, so the two lists cannot drift out of the canonical order.
ARM_DOFS: list[str] = [n.replace("iiwa14_joint_", "joint") for n in ARM_JOINT_NAMES]
HAND_DOFS: list[str] = [re.sub(r"^left_\d_", "left_", n) for n in HAND_JOINT_NAMES]
assert len(ARM_DOFS) == NUM_ARM_JOINTS and len(HAND_DOFS) == NUM_HAND_JOINTS

# Palm frame = the iiwa flange + local offset; 5 fingertip distal-phalanx links (order is the
# fingertip_pos_rel_palm obs block). ``link7`` is the composite's name for the merged URDF's
# ``iiwa14_link_7`` — the same frame, so PALM_CENTER_OFFSET transfers unchanged.
PALM_LINK: str = "link7"
FINGERTIP_LINKS: list[str] = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

# ---------------------------------------------------------------------------
# Joint limits (canonical order). The hardware/deploy code uses the
# "restricted" arm limits (first 7 shrunk by 10 deg) for the final clamp, but
# the *observation* unscale and the hand scale use the UNrestricted limits.
# ---------------------------------------------------------------------------

Q_LOWER_LIMITS = np.array(
    [
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -3.0543,
        -0.1745,
        -0.3491,
        -0.5236,
        -0.3491,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
    ],
    dtype=np.float32,
)

Q_UPPER_LIMITS = np.array(
    [
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        3.0543,
        1.9199,
        0.1309,
        1.3963,
        0.3491,
        1.7453,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        0.2618,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
    ],
    dtype=np.float32,
)
assert Q_LOWER_LIMITS.shape == (NUM_JOINTS,) == Q_UPPER_LIMITS.shape

_ARM_RESTRICT = np.deg2rad(10.0)
Q_LOWER_LIMITS_RESTRICTED = Q_LOWER_LIMITS.copy()
Q_LOWER_LIMITS_RESTRICTED[:NUM_ARM_JOINTS] += _ARM_RESTRICT
Q_UPPER_LIMITS_RESTRICTED = Q_UPPER_LIMITS.copy()
Q_UPPER_LIMITS_RESTRICTED[:NUM_ARM_JOINTS] -= _ARM_RESTRICT

# ---------------------------------------------------------------------------
# Geometry offsets (metres, local frames)
# ---------------------------------------------------------------------------

PALM_CENTER_OFFSET = np.array([-0.00, -0.02, 0.16], dtype=np.float32)  # in iiwa14_link_7
FINGERTIP_OFFSET = np.array([0.02, 0.002, 0.0], dtype=np.float32)  # per fingertip DP frame

# 4 object-frame keypoint corners, scaled at runtime by
# OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2 * object_scales.
KEYPOINT_CORNERS = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float32)
OBJECT_BASE_SIZE: float = 0.04
KEYPOINT_SCALE: float = 1.5

# ---------------------------------------------------------------------------
# Action transform (see action_utils.compute_joint_pos_targets)
# ---------------------------------------------------------------------------

ARM_MOVING_AVERAGE: float = 0.1
HAND_MOVING_AVERAGE: float = 0.1
DOF_SPEED_SCALE: float = 1.5  # arm velocity-delta gain (named hand_dof_speed_scale upstream)
CONTROL_DT: float = 1.0 / 60.0  # policy step
SIM_DT: float = 1.0 / 120.0
DECIMATION: int = 2

# ---------------------------------------------------------------------------
# Observation layout (actor obs_list, total 140). Order is load-bearing.
# ---------------------------------------------------------------------------

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,  # 29, unscaled to [-1, 1] via Q limits
    "joint_vel": NUM_JOINTS,  # 29, raw
    "prev_action_targets": NUM_JOINTS,  # 29, raw (canonical order)
    "palm_pos": 3,  # world frame, robot-origin relative
    "palm_rot": 4,  # xyzw (legacy IsaacGym convention)
    "object_rot": 4,  # xyzw
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
}
OBS_LIST: list[str] = list(OBS_FIELD_SIZES.keys())
NUM_OBS: int = sum(OBS_FIELD_SIZES.values())
assert NUM_OBS == 140, NUM_OBS

# ---------------------------------------------------------------------------
# Network shape (rl_games SimToolRealLSTMAsymmetricPPO actor, from checkpoint)
# ---------------------------------------------------------------------------

SAPG_EXTRA_DIM: int = 32  # extra_params row width; deploy selects row 0
RNN_INPUT_DIM: int = NUM_OBS + SAPG_EXTRA_DIM  # 172
RNN_HIDDEN: int = 1024
MLP_UNITS: tuple[int, ...] = (1024, 1024, 512, 512)
NORM_CLAMP: float = 5.0
NORM_EPS: float = 1e-5

# Robot base world position, matching the IsaacSim training scene: robot at (0, 0.8, 0) with
# the table/object/goal near the world origin in front of it. Equals the deploy-time T_W_R
# translation, so obs can be read in world coordinates with no separate R->W lift.
ROBOT_BASE_POS = (0.0, 0.8, 0.0)
T_W_R_TRANSLATION = np.array(ROBOT_BASE_POS, dtype=np.float32)

# Provenance only: upstream trained against the merged
# `kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`, which is not on the
# assets hub; this port builds the equivalent rig from stock unit robots (`KukaSharpa`).

# ---------------------------------------------------------------------------
# PD gains (Isaac Lab actuator config: scene_utils.ARM/HAND_JOINT_*).
# Used as the Genesis built-in PD stiffness (kp) / damping (kd).
# ---------------------------------------------------------------------------

ARM_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}
ARM_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}
HAND_STIFFNESS: dict[str, float] = {
    "left_1_thumb_CMC_FE": 6.95,
    "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62,
    "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76,
    "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9,
    "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76,
    "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9,
    "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76,
    "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9,
    "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38,
    "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9,
    "left_pinky_DIP": 0.9,
}
HAND_DAMPING: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.28676845,
    "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083,
    "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_2_index_MCP_FE": 0.20859232,
    "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185,
    "left_index_DIP": 0.03504461,
    "left_3_middle_MCP_FE": 0.2085923,
    "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185,
    "left_middle_DIP": 0.03504461,
    "left_4_ring_MCP_FE": 0.20859226,
    "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183,
    "left_ring_DIP": 0.0350446,
    "left_5_pinky_CMC": 0.02782345,
    "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528,
    "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}
ARM_KP = ARM_STIFFNESS
ARM_KD = ARM_DAMPING
HAND_KP = HAND_STIFFNESS
HAND_KD = HAND_DAMPING

# Joint armature (rotor inertia). The arm has none in the original; the hand uses
# these per-joint values (scene_utils.HAND_JOINT_ARMATURE, applied via the
# IsaacSim ImplicitActuatorCfg). Eden's asset loading fills armature=0.1 on every zero-armature
# URDF joint (``default_armature``) — ~30x too much on the fingers — so these MUST be set explicitly.
ARM_ARMATURE: dict[str, float] = {n: 0.0 for n in ARM_JOINT_NAMES}
HAND_ARMATURE: dict[str, float] = {
    "left_1_thumb_CMC_FE": 0.0032,
    "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265,
    "left_thumb_MCP_AA": 0.00265,
    "left_thumb_IP": 0.0006,
    "left_2_index_MCP_FE": 0.00265,
    "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006,
    "left_index_DIP": 0.00042,
    "left_3_middle_MCP_FE": 0.00265,
    "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006,
    "left_middle_DIP": 0.00042,
    "left_4_ring_MCP_FE": 0.00265,
    "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006,
    "left_ring_DIP": 0.00042,
    "left_5_pinky_CMC": 0.00012,
    "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265,
    "left_pinky_PIP": 0.0006,
    "left_pinky_DIP": 0.00042,
}

# ---------------------------------------------------------------------------
# Contact frictions (SimToolReal.yaml `modify_asset_frictions`). The robot's
# bulk friction is ROBOT_FRICTION; the fingertip (DP) links are overridden to
# the higher FINGER_TIP_FRICTION for grip. Object and table use their own.
# ---------------------------------------------------------------------------

ROBOT_FRICTION: float = 0.5
FINGER_TIP_FRICTION: float = 1.5
OBJECT_FRICTION: float = 0.5
TABLE_FRICTION: float = 0.5

# ---------------------------------------------------------------------------
# Home / reset pose (deployment/mujoco INIT_JOINT_POS, canonical order).
# Arm posed over the table; hand open (zeros).
# ---------------------------------------------------------------------------

ARM_INIT_POS = [-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308]
HAND_INIT_POS = [0.0] * NUM_HAND_JOINTS
INIT_DOFS_POS: dict[str, float] = {
    **{n: v for n, v in zip(ARM_JOINT_NAMES, ARM_INIT_POS)},
    **{n: 0.0 for n in HAND_JOINT_NAMES},
}

# ---------------------------------------------------------------------------
# Default sim2sim scene layout (deployment/mujoco MujocoSimConfig), in WORLD
# coordinates. The robot base sits at ROBOT_BASE_POS=(0,0.8,0); the table/object/
# goal are near the world origin, ~0.8 m in front of the base (so they do NOT
# overlap the arm). Table is a box; object rests just above the table top; goal
# is a lifted target pose.
# ---------------------------------------------------------------------------

TABLE_SIZE = (0.475, 0.4, 0.3)  # full box extents (m)
TABLE_POS = (0.0, 0.0, 0.38)  # box center; top at 0.53
OBJECT_START_POS = (0.0, 0.0, 0.58)
OBJECT_START_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
GOAL_START_POS = (0.0, 0.0, 0.78)  # lift goal: +0.2 above object
GOAL_START_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

# Goal-sampling volume (SimToolReal.yaml reset.target_volume_*), world frame —
# all above the table top (0.53). The demo resamples random reachable goal poses
# inside this box.
TARGET_VOLUME_MINS = (-0.35, -0.2, 0.6)
TARGET_VOLUME_MAXS = (0.35, 0.2, 0.95)

# The ``object_scales`` observation is a tool's grasp bounding box in metres times 25 — NOT a
# metric scale factor. The multiplier is upstream's convention (dextoolbench/objects.py), so it
# lives here with the policy rather than on the object options, which record the plain metric
# bbox under ``metadata.grasp_bbox``.
OBJECT_SCALES_PER_METRE: float = 25.0


def object_scales_for(grasp_bbox: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert a tool's metric grasp bbox into the policy's ``object_scales`` observation.

    Pairs with :mod:`eden.options.objects.dextoolbench`, whose tools carry the bbox as
    ``metadata.grasp_bbox``.
    """
    return tuple(float(v) * OBJECT_SCALES_PER_METRE for v in grasp_bbox)


# Default for the sim2sim scene: the long_screwdriver.
DEFAULT_OBJECT_SCALES = object_scales_for((0.10, 0.030, 0.030))
