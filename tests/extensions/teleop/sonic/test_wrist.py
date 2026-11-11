"""``eden.extensions.teleop.sonic.wrist``: swing-twist decomposition and the SMPL -> G1 wrist mapping."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from eden.extensions.teleop.sonic.wrist import (
    G1_L_WRIST_PITCH,
    G1_L_WRIST_ROLL,
    G1_L_WRIST_YAW,
    G1_R_WRIST_ROLL,
    decompose_rotation_aa,
    wrist_joints_from_body_pose,
)


def test_decompose_recomposes():
    rng = np.random.default_rng(2)
    aa = rng.normal(scale=0.8, size=(6, 3))
    axis = np.array([0.0, 1.0, 0.0])
    q_twist, q_swing = decompose_rotation_aa(aa, axis)
    for i in range(len(aa)):
        twist = sRot.from_quat(q_twist[i][[1, 2, 3, 0]])
        swing = sRot.from_quat(q_swing[i][[1, 2, 3, 0]])
        recomposed = twist * swing
        expected = sRot.from_rotvec(aa[i])
        assert (recomposed.inv() * expected).magnitude() == pytest.approx(0.0, abs=1e-9)
        # The twist really is about the given axis.
        tv = q_twist[i][1:]
        assert np.linalg.norm(np.cross(tv, axis)) == pytest.approx(0.0, abs=1e-9)


def test_decompose_pure_twist_has_identity_swing():
    aa = np.array([[0.0, 0.7, 0.0]])
    q_twist, q_swing = decompose_rotation_aa(aa, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(q_swing[0], [1, 0, 0, 0], atol=1e-12)
    assert 2 * np.arccos(q_twist[0, 0]) == pytest.approx(0.7, abs=1e-9)


def test_decompose_zero_rotation_is_identity():
    q_twist, q_swing = decompose_rotation_aa(np.zeros((1, 3)), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(q_twist[0], [1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q_swing[0], [1, 0, 0, 0], atol=1e-12)


def test_wrist_mapping_zero_pose_is_zero():
    joints = wrist_joints_from_body_pose(np.zeros((21, 3)))
    np.testing.assert_allclose(joints, 0.0)


def test_only_wrist_slots_are_written():
    rng = np.random.default_rng(4)
    joints = wrist_joints_from_body_pose(rng.normal(scale=0.4, size=(21, 3)))
    wrist_slots = {23, 24, 25, 26, 27, 28}
    for i in range(29):
        if i not in wrist_slots:
            assert joints[i] == 0.0, f"non-wrist slot {i} written"


def test_elbow_twist_stays_out_of_the_wrist():
    # A pure elbow-pitch rotation (about the G1 elbow axis) is all twist:
    # nothing should leak into the wrist joints.
    pose = np.zeros((21, 3))
    pose[17] = [0.0, 0.9, 0.0]  # left elbow, pure pitch
    joints = wrist_joints_from_body_pose(pose)
    np.testing.assert_allclose(joints, 0.0, atol=1e-9)


def test_elbow_roll_moves_into_wrist_roll():
    pose = np.zeros((21, 3))
    pose[17] = [0.3, 0.0, 0.0]  # left elbow roll (not available on the G1 elbow)
    joints = wrist_joints_from_body_pose(pose)
    assert joints[G1_L_WRIST_ROLL] == pytest.approx(0.3, abs=1e-9)
    assert joints[G1_L_WRIST_PITCH] == pytest.approx(0.0, abs=1e-9)
    assert joints[G1_L_WRIST_YAW] == pytest.approx(0.0, abs=1e-9)


def test_wrist_sign_conventions_per_side():
    pose = np.zeros((21, 3))
    pose[19] = [0.2, 0.0, 0.0]  # left wrist roll
    pose[20] = [0.2, 0.0, 0.0]  # right wrist roll
    joints = wrist_joints_from_body_pose(pose)
    assert joints[G1_L_WRIST_ROLL] == pytest.approx(0.2, abs=1e-9)
    assert joints[G1_R_WRIST_ROLL] == pytest.approx(-0.2, abs=1e-9)  # right side negated
    # Left wrist pitch: the reference's double negation makes it +euler_y.
    pose = np.zeros((21, 3))
    pose[19] = [0.0, 0.25, 0.0]
    joints = wrist_joints_from_body_pose(pose)
    assert joints[G1_L_WRIST_PITCH] == pytest.approx(0.25, abs=1e-9)
