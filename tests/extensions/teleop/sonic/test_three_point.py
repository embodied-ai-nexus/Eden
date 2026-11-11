"""``eden.extensions.teleop.sonic.three_point``: the VR 3-point extraction and its calibration.

The calibration tests run against the real G1 key-frame FK (the module-scoped
``g1_key_frame_fk`` fixture in tests/extensions/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from eden.extensions.teleop.sonic.three_point import ThreePointPose, process_3pt_pose


def _frame(joint_poses: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """A raw (24, 7) service frame; identity quats unless overridden."""
    frame = np.zeros((24, 7))
    frame[:, 6] = 1.0  # scalar-last identity
    for joint, pose in (joint_poses or {}).items():
        frame[joint] = pose
    return frame


def _pose(pos, rot: sRot) -> np.ndarray:
    return np.concatenate([np.asarray(pos, dtype=float), rot.as_quat(scalar_first=False)])


def _operator_frame(rng=None):
    """A generic posed operator frame (arbitrary but fixed rotations/positions)."""
    rng = rng or np.random.default_rng(7)
    frame = _frame()
    for joint in (0, 12, 22, 23):
        frame[joint] = _pose(rng.normal(scale=0.3, size=3), sRot.from_quat(rng.normal(size=4)))
    return frame


def _assert_same_rotation(got_wxyz, expected_wxyz, atol=1e-9):
    """Quaternions are sign-ambiguous: q and -q are the same rotation."""
    got, expected = np.asarray(got_wxyz), np.asarray(expected_wxyz)
    assert min(np.linalg.norm(got - expected), np.linalg.norm(got + expected)) < atol, (got, expected)


def _assert_lands_on_wrist_targets(out, targets):
    for row, key in ((0, "left_wrist"), (1, "right_wrist")):
        np.testing.assert_allclose(out[row, :3], targets[key]["position"], atol=1e-9, err_msg=key)
        # The FK quaternion is float32 (Genesis) and not exactly unit; scipy renormalizes it.
        _assert_same_rotation(out[row, 3:], targets[key]["orientation_wxyz"], atol=1e-6)


def test_positions_become_root_relative_in_the_robot_frame():
    frame = _frame(
        {
            0: _pose([0.0, 1.0, 0.0], sRot.identity()),  # root 1 m up (service y)
            22: _pose([0.2, 1.3, 0.1], sRot.identity()),  # left hand
        }
    )
    out = process_3pt_pose(frame)
    # Service offset (0.2, 0.3, 0.1) from the root -> robot [-x, z, y] = (-0.2, 0.1, 0.3), then
    # expressed in the root frame, whose alignment offset is yaw -90°: Rz(+90°) maps it to (-0.1, -0.2, 0.3).
    np.testing.assert_allclose(out[0, :3], [-0.1, -0.2, 0.3], atol=1e-9)


def test_identity_frame_orientations_follow_the_alignment_offsets():
    out = process_3pt_pose(_frame())
    root_off = sRot.from_euler("xyz", [0, 0, -90], degrees=True)
    offsets = [
        sRot.from_euler("xyz", [90, 0, 0], degrees=True),
        sRot.from_euler("xyz", [-90, 0, 180], degrees=True),
        sRot.from_euler("xyz", [0, 0, -90], degrees=True),
    ]
    for row, off in enumerate(offsets):
        _assert_same_rotation(out[row, 3:], (root_off.inv() * off).as_quat(scalar_first=True))


def test_rotation_about_service_up_becomes_robot_yaw():
    theta = 0.4
    frame = _frame({22: _pose([0, 0, 0], sRot.from_euler("y", theta))})
    plain = process_3pt_pose(_frame())
    rotated = process_3pt_pose(frame)
    # The service-frame Y (up) rotation must appear as a robot-frame Z (yaw)
    # rotation applied before the alignment offsets.
    lhs = sRot.from_quat(rotated[0, 3:], scalar_first=True)
    rhs = sRot.from_euler("z", theta) * sRot.from_quat(plain[0, 3:], scalar_first=True)
    assert (lhs.inv() * rhs).magnitude() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.requires_genesis
def test_calibration_closure_reproduces_fk_targets(g1_key_frame_fk):
    tpp = ThreePointPose(g1_key_frame_fk)
    frame = _operator_frame()
    tpp.calibrate_now(frame)
    assert tpp.is_calibrated

    out = tpp.process(frame)
    # Processing the calibration frame itself lands exactly on FK of the zero configuration.
    _assert_lands_on_wrist_targets(out, g1_key_frame_fk.key_frame_poses(np.zeros(29)))
    # Neck: calibrated orientation is identity, position is the synthesized chain
    # (0.05 m root -> torso_link, then the 0.35 m neck link straight up).
    _assert_same_rotation(out[2, 3:], [1, 0, 0, 0])
    np.testing.assert_allclose(out[2, :3], [0, 0, 0.05 + 0.35], atol=1e-9)


@pytest.mark.requires_genesis
def test_neck_position_is_synthesized_from_orientation(g1_key_frame_fk):
    tpp = ThreePointPose(g1_key_frame_fk)
    tpp.calibrate_now(_frame())
    theta = 0.3
    # Tilt the operator's neck about service X: robot -X, i.e. Ry(-theta) once
    # expressed in the yaw -90° neck frame, so the neck's Z axis leans to -x.
    out = tpp.process(_frame({12: _pose([0, 0, 0], sRot.from_euler("x", theta))}))
    _assert_same_rotation(out[2, 3:], sRot.from_euler("y", -theta).as_quat(scalar_first=True))
    np.testing.assert_allclose(out[2, :3], [-0.35 * np.sin(theta), 0.0, 0.05 + 0.35 * np.cos(theta)], atol=1e-9)


@pytest.mark.requires_genesis
def test_wrist_recalibration_uses_measured_joints_and_preserves_neck(g1_key_frame_fk):
    tpp = ThreePointPose(g1_key_frame_fk)
    frame = _operator_frame()
    tpp.calibrate_now(frame)

    measured_q = np.linspace(-0.3, 0.3, 29)
    measured_targets = g1_key_frame_fk.key_frame_poses(measured_q)
    zero_targets = g1_key_frame_fk.key_frame_poses(np.zeros(29))
    assert np.linalg.norm(measured_targets["left_wrist"]["position"] - zero_targets["left_wrist"]["position"]) > 0.05

    tpp.reset_with_measured_q(measured_q)
    # The next processed frame re-captures the wrist offsets against FK(measured q): closure on it.
    frame2 = _operator_frame(np.random.default_rng(9))
    _assert_lands_on_wrist_targets(tpp.process(frame2), measured_targets)
    # The recalibration is one-shot: a later frame is not re-captured.
    later = tpp.process(frame)
    assert not np.allclose(later[0, :3], measured_targets["left_wrist"]["position"], atol=1e-3)
    # The neck kept the first calibration: the first frame's neck still reads as identity,
    # which a neck recaptured from frame2 (a different neck pose) would not.
    _assert_same_rotation(later[2, 3:], [1, 0, 0, 0])


@pytest.mark.requires_genesis
def test_full_recalibration_recaptures_the_neck(g1_key_frame_fk):
    """A re-engage after a fall runs a full calibration; it reused the first neck capture."""
    tpp = ThreePointPose(g1_key_frame_fk)
    tpp.calibrate_now(_operator_frame())
    regrabbed = _operator_frame(np.random.default_rng(11))  # headset re-donned: another neck pose
    tpp.calibrate_now(regrabbed)
    # Closure on the new calibration frame: the neck reads as the calibrated identity.
    _assert_same_rotation(tpp.process(regrabbed)[2, 3:], [1, 0, 0, 0])


@pytest.mark.requires_genesis
def test_uncalibrated_process_returns_raw(g1_key_frame_fk):
    tpp = ThreePointPose(g1_key_frame_fk)
    assert not tpp.is_calibrated
    frame = _operator_frame()
    np.testing.assert_allclose(tpp.process(frame), process_3pt_pose(frame))


def test_an_upright_neck_at_calibration_passes_and_a_pitched_one_is_called_out(g1_key_frame_fk, capsys):
    """Every 3-point target is rotated by the neck captured here, so a skewed one is reported."""
    root = _pose([0, 1, 0], sRot.identity())
    three_point = ThreePointPose(g1_key_frame_fk)

    three_point.calibrate_now(_frame({0: root, 12: _pose([0, 1.5, 0], sRot.identity())}))
    out = capsys.readouterr().out
    assert "neck 0° off the pelvis" in out and "WARNING" not in out
    assert len(three_point.wrist_rot_offset_deg) == 2

    pitched = sRot.from_euler("x", 40, degrees=True)
    three_point.calibrate_now(_frame({0: root, 12: _pose([0, 1.5, 0], pitched)}))
    assert "WARNING: your neck was 40° off your pelvis" in capsys.readouterr().out

    # A wrist-only recalibration keeps the neck, so it neither re-measures nor re-warns about it.
    three_point.reset_with_measured_q(np.zeros(29))
    three_point.process(_frame({0: root, 12: _pose([0, 1.5, 0], pitched)}))
    out = capsys.readouterr().out
    assert "wrist rotation offset" in out and "neck" not in out.split("calibration captured")[1]
