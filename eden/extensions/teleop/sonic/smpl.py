"""PICO full-body tracking -> SMPL parameters and root-local joints (POSE mode).

Port of the reference streamer's SMPL path:

1. The service's 24 global joint rotations (scalar-last, service frame) become
   parent-local axis-angle after a per-joint ``Ry(180°)`` frame fix.
2. The root orientation is re-based: pre-multiply ``Rx(+90°)`` (service Y-up ->
   robot Z-up), then post-multiply away SMPL's rest orientation (the 120°
   rotation about (1,1,1)).
3. Joint positions come from an SMPL-X forward kinematics over the rest
   skeleton in ``sonic/smplx_joints_info.npz`` (fetched from the
   ``Kashu7100/eden_assets`` HF dataset; 55 joints; output = the 22 body joints
   + the two thumb tips, SMPL-X indices 39 and 54 — NOT the SMPL
   left_hand/right_hand joints), expressed in the root-local frame.

No body-shape (betas) or operator-height scaling exists anywhere in this
pipeline — the reference relies on calibration offsets and the policy itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from eden.utils.assets import get_asset_path
from eden.utils.geom import quat_mul

# SMPL 24-joint parent tree (the reference literal carries a harmless off-by-one
# at joint 23 that is never read on the wire; this is the correct tree).
SMPL_PARENTS: tuple[int, ...] = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)

# SMPL-X output gather: 22 body joints + left/right thumb tips.
_OUTPUT_JOINTS = np.concatenate([np.arange(22), np.array([39, 54])])

# Rest-skeleton asset on the Kashu7100/eden_assets HF dataset (downloaded + cached
# by ``get_asset_path`` on first use — kinematic data, not committed to the repo).
_SMPLX_INFO_ASSET = "sonic/smplx_joints_info.npz"

# SMPL's rest orientation: 120° about (1, 1, 1), removed from the root quat.
_SMPL_BASE_ROT_CONJ = np.array([0.5, -0.5, -0.5, -0.5])
_RX_90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # Rx(+90°), wxyz


def smpl_root_ytoz_up(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rebase the SMPL root quaternion from Y-up to Z-up (pre-multiply Rx(+90°))."""
    return quat_mul(_RX_90, np.asarray(quat_wxyz, dtype=np.float64))


def remove_smpl_base_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    """Remove SMPL's rest orientation (post-multiply by conj([.5, .5, .5, .5]))."""
    return quat_mul(np.asarray(quat_wxyz, dtype=np.float64), _SMPL_BASE_ROT_CONJ)


def global_quats_to_local_aa(body_poses: np.ndarray) -> np.ndarray:
    """Service-frame global joint rotations -> parent-local axis-angle ``(24, 3)``.

    ``body_poses`` is the raw ``(24, 7)`` service array (scalar-last quats).
    Every joint is post-multiplied by ``Ry(180°)`` first (the service's joint
    frames are flipped relative to SMPL's).
    """
    global_quats = np.asarray(body_poses, dtype=np.float64)[:, [6, 3, 4, 5]]
    global_rots = sRot.from_quat(global_quats, scalar_first=True) * sRot.from_euler("y", 180, degrees=True)
    local = []
    for i in range(24):
        parent = SMPL_PARENTS[i]
        rot = global_rots[i] if parent == -1 else global_rots[parent].inv() * global_rots[i]
        local.append(rot.as_rotvec())
    return np.asarray(local)


class SmplFk:
    """SMPL-X forward kinematics over the HF-hosted rest skeleton (positions only)."""

    def __init__(self, info_path: str | None = None) -> None:
        info = np.load(info_path or get_asset_path(_SMPLX_INFO_ASSET))
        self._rest_joints = np.asarray(info["J"], dtype=np.float64)  # (55, 3)
        self._parents = np.asarray(info["parents"], dtype=np.int64)  # (55,)
        if self._rest_joints.shape != (55, 3) or self._parents.shape != (55,):
            raise ValueError(f"unexpected SMPL-X skeleton shapes in {info_path}")

    def joints(self, global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
        """Posed joint positions ``(24, 3)`` for a root axis-angle + 63-dim body pose.

        The full 55-joint pose is ``[global_orient, body_pose, zeros(99)]``;
        the output gathers the 22 body joints plus the two thumb tips.
        """
        body_pose = np.asarray(body_pose, dtype=np.float64).reshape(-1)[:63]
        full_pose = np.concatenate([np.asarray(global_orient, dtype=np.float64).reshape(3), body_pose, np.zeros(99)])
        rot_mats = sRot.from_rotvec(full_pose.reshape(55, 3)).as_matrix()  # (55, 3, 3)

        rel_joints = self._rest_joints.copy()
        rel_joints[1:] -= self._rest_joints[self._parents[1:]]

        transforms = np.zeros((55, 4, 4))
        transforms[:, :3, :3] = rot_mats
        transforms[:, :3, 3] = rel_joints
        transforms[:, 3, 3] = 1.0

        chain = [transforms[0]]
        for i in range(1, 55):
            chain.append(chain[self._parents[i]] @ transforms[i])
        positions = np.stack(chain)[:, :3, 3]
        return positions[_OUTPUT_JOINTS]


@dataclass
class SmplFrame:
    """One processed SMPL frame for the pose stream."""

    body_pose: np.ndarray  # (69,) parent-local axis-angle, joints 1..23
    global_orient_quat: np.ndarray  # (4,) wxyz, Z-up, rest orientation removed
    smpl_joints_local: np.ndarray  # (24, 3) root-local positions (22 body + thumb tips)


def process_body_poses(body_poses: np.ndarray, fk: SmplFk) -> SmplFrame:
    """Full POSE-mode conversion of one raw ``(24, 7)`` service frame."""
    pose_aa = global_quats_to_local_aa(body_poses)
    body_pose = pose_aa[1:].reshape(-1)  # (69,)

    root_quat = sRot.from_rotvec(pose_aa[0]).as_quat(scalar_first=True)
    root_quat = smpl_root_ytoz_up(root_quat)
    global_orient_new = sRot.from_quat(root_quat, scalar_first=True).as_rotvec()

    joints = fk.joints(global_orient_new, body_pose[:63])

    root_quat = remove_smpl_base_rot(root_quat)
    joints_local = sRot.from_quat(root_quat, scalar_first=True).inv().apply(joints)
    return SmplFrame(body_pose=body_pose, global_orient_quat=root_quat, smpl_joints_local=joints_local)
