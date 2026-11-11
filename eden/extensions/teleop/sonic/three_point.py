"""VR 3-point pose extraction and calibration (VR_3PT mode).

Port of the reference streamer's 3-point path. From the raw ``(24, 7)``
service body frame it takes root (0), left hand (22), right hand (23) and
neck (12 — deliberately not the head, which moves too freely), converts them
to the robot frame, applies per-joint alignment offsets, and expresses the
three keypoints relative to the root. :class:`ThreePointPose` then calibrates
them against G1 forward kinematics so the streamed targets live in the same
pelvis-local frame the policy's teleop encoder was trained on:

* full calibration (every engage): operator stands in the neutral pose ==
  the G1's all-zero joint configuration; the neck orientation and both wrist
  offsets are captured against FK of ``q = 0``.
* wrist recalibration (every VR_3PT entry): only the wrist offsets are
  re-captured, against FK of the robot's measured joints; the neck
  calibration is preserved to avoid jumps from tracking noise.

The neck *position* is never measured: it is reconstructed from the
calibrated neck orientation through a fixed two-link chain.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# Service [x, y, z] -> robot [-x, z, y] (proper rotation; the handedness flip
# is absorbed by the axis swap applied as a similarity transform).
Q_ROBOT_SERVICE = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# Per-keypoint frame alignment, post-multiplied: root/neck yaw -90°,
# left wrist roll +90°, right wrist roll -90° then yaw 180°.
_OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
]

_KEYPOINT_JOINTS = (0, 22, 23, 12)  # root, left hand, right hand, neck

# Kinematic chain for the synthesized neck position.
TORSO_LINK_OFFSET_Z = 0.05  # m, root -> torso_link
NECK_LINK_LENGTH = 0.35  # m, torso_link -> neck along the neck's local Z

# Calibration check (an Eden diagnostic, not in the reference). The neck orientation captured
# at the full calibration rotates every later 3-point target. Standing upright, the tracked neck
# is aligned with the pelvis (the body model's rest convention), so a neck far off the pelvis
# there — head down at the controllers, torso twisted — skews the whole mapping.
NECK_TILT_WARN_DEG = 20.0


def _service_pose_to_robot(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One ``[x,y,z,qx,qy,qz,qw]`` service pose -> robot-frame (pos, wxyz quat)."""
    pos = Q_ROBOT_SERVICE @ np.asarray(pose[:3], dtype=np.float64)
    rot = sRot.from_quat(pose[3:], scalar_first=False).as_matrix()
    rel = sRot.from_matrix(Q_ROBOT_SERVICE @ rot @ Q_ROBOT_SERVICE.T)
    return pos, rel.as_quat(scalar_first=True)


def process_3pt_pose(body_poses: np.ndarray) -> np.ndarray:
    """Raw ``(24, 7)`` service frame -> ``(3, 7)`` root-relative keypoints.

    Rows are [left wrist, right wrist, neck], each ``[x, y, z, qw, qx, qy, qz]``.
    """
    body_poses = np.asarray(body_poses, dtype=np.float64)
    kp = np.zeros((4, 7))
    for rel_i, joint in enumerate(_KEYPOINT_JOINTS):
        pos, quat = _service_pose_to_robot(body_poses[joint])
        rot = sRot.from_quat(quat, scalar_first=True) * _OFFSETS[rel_i]
        kp[rel_i, :3] = pos
        kp[rel_i, 3:] = rot.as_quat(scalar_first=True)

    root_pos = kp[0, :3].copy()
    root_rot = sRot.from_quat(kp[0, 3:], scalar_first=True)
    for i in range(1, 4):
        kp[i, :3] = root_rot.inv().apply(kp[i, :3] - root_pos)
        kp[i, 3:] = (root_rot.inv() * sRot.from_quat(kp[i, 3:], scalar_first=True)).as_quat(scalar_first=True)
    return kp[1:]


class ThreePointPose:
    """Calibrated 3-point targets from raw PICO body frames (in-memory state only)."""

    def __init__(self, fk) -> None:
        """``fk`` provides ``key_frame_poses(q_mj29) -> {left_wrist, right_wrist, torso}``
        (see :class:`~eden.extensions.teleop.sonic.g1_fk.G1KeyFrameFk`).
        """
        self._fk = fk
        self._calibration_pending = False
        self._neck_quat_inv: np.ndarray | None = None  # wxyz
        self._lwrist_offset: np.ndarray | None = None
        self._rwrist_offset: np.ndarray | None = None
        self._lwrist_rot_offset: sRot | None = None
        self._rwrist_rot_offset: sRot | None = None
        self._override_robot_q: np.ndarray | None = None
        #: (left, right) degrees between the operator's and the robot's wrist at the last calibration.
        self.wrist_rot_offset_deg: tuple[float, float] | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._neck_quat_inv is not None

    def process(self, body_poses: np.ndarray) -> np.ndarray:
        """Extract + calibrate the 3-point pose for one raw body frame."""
        raw = process_3pt_pose(body_poses)
        if self._calibration_pending:
            self._capture_calibration(raw)
        return self._apply_calibration(raw)

    def calibrate_now(self, body_poses: np.ndarray) -> None:
        """Full calibration against FK of the all-zero joint configuration.

        The operator's neutral pose (upright, feet together, forearms bent 90°
        forward) *is* the G1's zero configuration.
        """
        raw = process_3pt_pose(body_poses)
        # Full means the neck too: it is otherwise captured once and kept across
        # wrist recalibrations, so a re-engage after a fall auto-reset (operator
        # re-donned the headset, or re-ran body calibration) would reuse the first one.
        self._neck_quat_inv = None
        self._override_robot_q = np.zeros(29)
        self._capture_calibration(raw)
        print("[ThreePointPose] calibration completed (zero-pose reference)")

    def reset_with_measured_q(self, body_q_measured: np.ndarray) -> None:
        """Queue a wrist-only recalibration against FK of the measured joints."""
        self._lwrist_offset = self._rwrist_offset = None
        self._lwrist_rot_offset = self._rwrist_rot_offset = None
        self._override_robot_q = np.asarray(body_q_measured, dtype=np.float64).copy()
        self._calibration_pending = True
        print("[ThreePointPose] wrist recalibration pending (neck preserved, measured q)")

    def _capture_calibration(self, raw: np.ndarray) -> None:
        # Neck orientation: captured once and preserved across recalibrations.
        neck_tilt_deg = None
        if self._neck_quat_inv is None:
            neck_rot = sRot.from_quat(raw[2, 3:], scalar_first=True)
            self._neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
            neck_tilt_deg = float(np.degrees(neck_rot.magnitude()))
        calib_inv = sRot.from_quat(self._neck_quat_inv, scalar_first=True)

        lw_pos = calib_inv.apply(raw[0, :3])
        rw_pos = calib_inv.apply(raw[1, :3])
        lw_rot = calib_inv * sRot.from_quat(raw[0, 3:], scalar_first=True)
        rw_rot = calib_inv * sRot.from_quat(raw[1, 3:], scalar_first=True)

        q = self._override_robot_q if self._override_robot_q is not None else np.zeros(29)
        g1 = self._fk.key_frame_poses(q)
        g1_lw_rot = sRot.from_quat(g1["left_wrist"]["orientation_wxyz"], scalar_first=True)
        g1_rw_rot = sRot.from_quat(g1["right_wrist"]["orientation_wxyz"], scalar_first=True)

        self._lwrist_offset = lw_pos - g1["left_wrist"]["position"]
        self._rwrist_offset = rw_pos - g1["right_wrist"]["position"]
        self._lwrist_rot_offset = g1_lw_rot * lw_rot.inv()
        self._rwrist_rot_offset = g1_rw_rot * rw_rot.inv()
        self._calibration_pending = False
        self._override_robot_q = None
        self.wrist_rot_offset_deg = tuple(
            float(np.degrees(r.magnitude())) for r in (self._lwrist_rot_offset, self._rwrist_rot_offset)
        )
        print(
            f"[ThreePointPose] calibration captured: L offset {np.round(self._lwrist_offset, 4)}, "
            f"R offset {np.round(self._rwrist_offset, 4)}, wrist rotation offset "
            f"L {self.wrist_rot_offset_deg[0]:.0f}° R {self.wrist_rot_offset_deg[1]:.0f}°"
            + ("" if neck_tilt_deg is None else f", neck {neck_tilt_deg:.0f}° off the pelvis")
        )
        if neck_tilt_deg is not None and neck_tilt_deg > NECK_TILT_WARN_DEG:
            print(
                f"[ThreePointPose] WARNING: your neck was {neck_tilt_deg:.0f}° off your pelvis at calibration "
                "(head down at the controllers, or torso twisted) — every 3-point target is rotated by it. "
                "Stand straight, look ahead, and re-engage (A+B+X+Y) in the neutral pose."
            )

    def _apply_calibration(self, raw: np.ndarray) -> np.ndarray:
        if self._neck_quat_inv is None:
            return raw
        calibrated = raw.copy()
        calib_inv = sRot.from_quat(self._neck_quat_inv, scalar_first=True)

        neck_rot = sRot.from_quat(raw[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv * neck_rot).as_quat(scalar_first=True)

        if self._lwrist_offset is not None:
            calibrated[0, :3] = calib_inv.apply(raw[0, :3]) - self._lwrist_offset
        if self._rwrist_offset is not None:
            calibrated[1, :3] = calib_inv.apply(raw[1, :3]) - self._rwrist_offset
        if self._lwrist_rot_offset is not None:
            lw = calib_inv * sRot.from_quat(raw[0, 3:], scalar_first=True)
            calibrated[0, 3:] = (self._lwrist_rot_offset * lw).as_quat(scalar_first=True)
        if self._rwrist_rot_offset is not None:
            rw = calib_inv * sRot.from_quat(raw[1, 3:], scalar_first=True)
            calibrated[1, 3:] = (self._rwrist_rot_offset * rw).as_quat(scalar_first=True)

        # Neck position: two-link reconstruction from the calibrated orientation.
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0.0, 0.0, 1.0])
        calibrated[2, :3] = np.array([0.0, 0.0, TORSO_LINK_OFFSET_Z]) + NECK_LINK_LENGTH * neck_z
        return calibrated
