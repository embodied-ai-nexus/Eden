"""SMPL elbow/wrist rotations -> G1 wrist joint targets (POSE mode).

The G1 elbow has a single pitch DOF, so the SMPL elbow's roll/yaw content is
decomposed off (twist about the elbow axis vs. swing) and folded into the
wrist joints, preserving the SMPL wrist's own rotation. Fills the six wrist
entries of the 29-dim IsaacLab joint vector; everything else stays zero on
the pose stream.
"""

from __future__ import annotations

import numpy as np
from genesis.utils import geom as gu
from scipy.spatial.transform import Rotation as sRot

from eden.utils.geom import quat_swing_twist

# body_pose (21, 3) rows for the arm joints (SMPL joint index - 1).
SMPL_L_ELBOW = 17
SMPL_R_ELBOW = 18
SMPL_L_WRIST = 19
SMPL_R_WRIST = 20

# IsaacLab wrist joint indices.
G1_L_WRIST_ROLL, G1_R_WRIST_ROLL = 23, 24
G1_L_WRIST_PITCH, G1_R_WRIST_PITCH = 25, 26
G1_L_WRIST_YAW, G1_R_WRIST_YAW = 27, 28

_ELBOW_AXIS = np.array([0.0, 1.0, 0.0])  # the G1 elbow's pitch axis


def decompose_rotation_aa(rotation_aa: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split axis-angle rotations into twist about ``axis`` and the residual swing.

    Returns wxyz quaternions ``(q_twist, q_swing)`` with ``q = q_twist ⊗ q_swing``
    (:func:`eden.utils.geom.quat_swing_twist` on the rotations' quaternions). A zero
    rotation decomposes into identities (the reference divides by the zero angle there).
    """
    return quat_swing_twist(gu.rotvec_to_quat(np.atleast_2d(np.asarray(rotation_aa, dtype=np.float64))), axis)


def wrist_joints_from_body_pose(body_pose: np.ndarray) -> np.ndarray:
    """The 29-dim IsaacLab joint vector for a pose-stream frame (wrists only).

    ``body_pose`` is the frame's ``(21, 3)`` (or flat 63-dim) local axis-angle
    pose. All Euler readings are intrinsic XYZ; the per-side signs follow the
    reference mapping verbatim.
    """
    pose = np.asarray(body_pose, dtype=np.float64).reshape(21, 3)
    joint_pos = np.zeros(29)

    _, l_swing = decompose_rotation_aa(pose[SMPL_L_ELBOW][None], _ELBOW_AXIS)
    _, r_swing = decompose_rotation_aa(pose[SMPL_R_ELBOW][None], _ELBOW_AXIS)
    l_swing_euler = sRot.from_quat(l_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")[0]
    r_swing_euler = sRot.from_quat(r_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")[0]
    l_wrist_euler = sRot.from_rotvec(pose[SMPL_L_WRIST]).as_euler("XYZ")
    r_wrist_euler = sRot.from_rotvec(pose[SMPL_R_WRIST]).as_euler("XYZ")

    l_roll = l_swing_euler[0] + l_wrist_euler[0]
    l_pitch = -l_wrist_euler[1]
    l_yaw = l_swing_euler[2] + l_wrist_euler[2]
    r_roll = -(r_swing_euler[0] + r_wrist_euler[0])
    r_pitch = -r_wrist_euler[1]
    r_yaw = r_swing_euler[2] + r_wrist_euler[2]

    joint_pos[G1_L_WRIST_ROLL] = l_roll
    joint_pos[G1_L_WRIST_PITCH] = -l_pitch  # double negation kept verbatim from the reference
    joint_pos[G1_L_WRIST_YAW] = l_yaw
    joint_pos[G1_R_WRIST_ROLL] = r_roll
    joint_pos[G1_R_WRIST_PITCH] = r_pitch
    joint_pos[G1_R_WRIST_YAW] = r_yaw
    return joint_pos
