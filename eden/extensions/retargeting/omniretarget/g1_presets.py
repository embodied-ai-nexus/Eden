"""G1 (29-DOF) OmniRetarget setup + human-motion preprocessing.

Single source of truth shared by the ``eden retarget omniretarget`` CLI and
``examples/recipes/retargeting/omniretarget/retarget_robot_only.py``: the foot-sticking / link-offset
/ manual-bound tables that configure :class:`OmniRetargeter` for the Unitree G1,
plus the two motion-preprocessing helpers (height/scale normalization and
foot-sticking detection).
"""

from __future__ import annotations

import numpy as np

# Eden's G1 model (g1_29dof_rev_1_0.xml) has no foot sphere bodies, so these virtual contact points
# sit at ``LINK_OFFSETS`` from the ankle roll links, emulating the 4 sphere bodies per foot of
# OmniRetarget's MuJoCo G1 model.
FOOT_STICKING_LINKS = {
    "left_sphere_1": "left_ankle_roll_link",
    "left_sphere_2": "left_ankle_roll_link",
    "left_sphere_3": "left_ankle_roll_link",
    "left_sphere_4": "left_ankle_roll_link",
    "right_sphere_1": "right_ankle_roll_link",
    "right_sphere_2": "right_ankle_roll_link",
    "right_sphere_3": "right_ankle_roll_link",
    "right_sphere_4": "right_ankle_roll_link",
}

LINK_OFFSETS = {
    "L_Toe": np.array([0.14, 0.0, -0.03]),
    "R_Toe": np.array([0.14, 0.0, -0.03]),
    "left_sphere_1": np.array([-0.05, 0.025, -0.03]),
    "left_sphere_2": np.array([-0.05, -0.025, -0.03]),
    "left_sphere_3": np.array([0.12, 0.03, -0.03]),
    "left_sphere_4": np.array([0.12, -0.03, -0.03]),
    "right_sphere_1": np.array([-0.05, 0.025, -0.03]),
    "right_sphere_2": np.array([-0.05, -0.025, -0.03]),
    "right_sphere_3": np.array([0.12, 0.03, -0.03]),
    "right_sphere_4": np.array([0.12, -0.03, -0.03]),
}

MANUAL_LB = {
    "waist_roll_joint": -0.3,
    "waist_pitch_joint": -0.1,
    "left_wrist_roll_joint": -0.1,
    "left_wrist_pitch_joint": -0.1,
    "left_wrist_yaw_joint": -0.05,
    "right_wrist_roll_joint": -0.1,
    "right_wrist_pitch_joint": -0.1,
    "right_wrist_yaw_joint": -0.05,
}

MANUAL_UB = {
    "waist_roll_joint": 0.3,
    "left_elbow_joint": 1.4,
    "left_wrist_roll_joint": 0.2,
    "left_wrist_pitch_joint": 0.3,
    "left_wrist_yaw_joint": 0.05,
    "right_elbow_joint": 1.4,
    "right_wrist_roll_joint": 0.2,
    "right_wrist_pitch_joint": 0.3,
    "right_wrist_yaw_joint": 0.05,
}

MANUAL_COST = {"waist_yaw_joint": 0.2, "waist_roll_joint": 0.2}

NOMINAL_TRACKING_DOFS = [
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
]

FOOT_STICKING_MAP = {
    "L_Toe": ["left_sphere_1", "left_sphere_2", "left_sphere_3", "left_sphere_4"],
    "R_Toe": ["right_sphere_1", "right_sphere_2", "right_sphere_3", "right_sphere_4"],
}


def preprocess_human_motion(
    human_joints: np.ndarray,
    demo_joints: list[str],
    toe_names: list[str],
    scale: float,
    mat_height: float = 0.1,
) -> np.ndarray:
    """Lower the human motion so its lowest toe sits at ``mat_height`` (at 0 when it starts below
    ``mat_height``), then scale every coordinate by ``scale`` to robot size.

    Returns a new array; ``human_joints`` is left as it was.
    """
    toe_indices = [demo_joints.index(toe_names[0]), demo_joints.index(toe_names[1])]
    z_min = human_joints[:, toe_indices, 2].min()
    if z_min >= mat_height:
        z_min -= mat_height
    lowered = human_joints.copy()
    lowered[:, :, 2] -= z_min
    return lowered * scale


def extract_foot_sticking_velocity(
    human_joints: np.ndarray,
    demo_joints: list[str],
    toe_names: list[str],
    velocity_threshold: float = 0.01,
) -> list[dict[str, bool]]:
    """Flag, per frame, whether each toe's planar displacement since the previous frame is at most
    ``velocity_threshold`` (metres per frame, not m/s). Frame 0 is never flagged.
    """
    left_idx = demo_joints.index(toe_names[0])
    right_idx = demo_joints.index(toe_names[1])
    left_vel = np.linalg.norm(np.diff(human_joints[:, left_idx, :2], axis=0), axis=1)
    right_vel = np.linalg.norm(np.diff(human_joints[:, right_idx, :2], axis=0), axis=1)
    left_vel = np.concatenate([[velocity_threshold + 1], left_vel])
    right_vel = np.concatenate([[velocity_threshold + 1], right_vel])
    return [
        {toe_names[0]: left_vel[i] <= velocity_threshold, toe_names[1]: right_vel[i] <= velocity_threshold}
        for i in range(len(human_joints))
    ]
