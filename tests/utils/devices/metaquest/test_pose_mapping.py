"""``ClutchPoseMapper``: clutch-relative teleop mapping and its slipping-clutch reach limits (vr-teleop-kit's T1-T11)."""

import numpy as np
import pytest

from genesis.utils.geom import inv_quat, quat_to_rotvec, rotvec_to_quat, transform_quat_by_quat

from eden.utils.devices.metaquest.pose_mapping import ClutchPoseMapper

EE_POS = np.array([0.50, 0.00, 0.42])
EE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation
CTRL_POS = np.array([0.0, 1.4, -0.3])
CTRL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _ang_deg(qa, qb):
    # angle of the relative rotation qa ⊗ qb⁻¹ (transform_quat_by_quat(v, u) == u ⊗ v).
    return float(np.degrees(np.linalg.norm(quat_to_rotvec(transform_quat_by_quat(inv_quat(qb), qa)))))


R_Z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # XR +X -> arm-base +Y


@pytest.mark.parametrize(
    "kwargs, ctrl_delta, ee_delta",
    [
        ({}, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ({}, (0.05, 0.0, 0.0), (0.05, 0.0, 0.0)),
        ({"scale": 0.5}, (0.05, 0.0, 0.0), (0.025, 0.0, 0.0)),
        ({"R": R_Z90}, (0.05, 0.0, 0.0), (0.0, 0.05, 0.0)),
    ],
    ids=["no_motion", "one_to_one", "scaled", "frame_rotation"],
)
def test_translation_is_scaled_and_rotated_from_the_engage_pose(kwargs, ctrl_delta, ee_delta):
    m = ClutchPoseMapper(**kwargs)
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    assert m.engaged
    p, q = m.target(CTRL_POS + np.array(ctrl_delta), CTRL_QUAT)
    assert np.allclose(p, EE_POS + np.array(ee_delta)) and np.allclose(q, EE_QUAT)


def test_rotation_passthrough():
    m = ClutchPoseMapper()
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    ctrl_now = rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0]))  # +30° about Y
    _, q = m.target(CTRL_POS, ctrl_now)
    assert np.allclose(q, rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0])))


def test_disengage_returns_none():
    m = ClutchPoseMapper()
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    m.disengage()
    assert m.target(CTRL_POS, CTRL_QUAT) is None


def test_reengage_updates_origin():
    m = ClutchPoseMapper()
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    m.disengage()
    new_ee = np.array([0.55, 0.10, 0.42])
    m.engage(CTRL_POS, CTRL_QUAT, new_ee, EE_QUAT)
    p, _ = m.target(CTRL_POS + np.array([0.05, 0, 0]), CTRL_QUAT)
    assert np.allclose(p, new_ee + np.array([0.05, 0, 0]))


def test_rot_reach_limit_clamps():
    # Twist the controller 120° in 1° steps while the EE stays put; the target
    # must never run further than rot_reach_limit from the current EE.
    m = ClutchPoseMapper(rot_reach_limit=0.5, pos_reach_limit=0.25)
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    worst = 0.0
    for i in range(1, 121):
        q_ctrl = rotvec_to_quat(np.array([0.0, np.radians(i), 0.0]))
        _, tq = m.target(CTRL_POS, q_ctrl, EE_POS, EE_QUAT)
        worst = max(worst, _ang_deg(tq, EE_QUAT))
    assert worst <= np.degrees(0.5) + 0.1


def test_rot_reversal_bites_at_reach_limit():
    # After a 120° push, reversing by just the reach-limit angle brings the
    # target back onto the EE (the absorbed overshoot is never retraced).
    m = ClutchPoseMapper(rot_reach_limit=0.5, pos_reach_limit=0.25)
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    for i in range(1, 121):
        m.target(CTRL_POS, rotvec_to_quat(np.array([0.0, np.radians(i), 0.0])), EE_POS, EE_QUAT)
    back = 120.0 - np.degrees(0.5)
    _, tq = m.target(CTRL_POS, rotvec_to_quat(np.array([0.0, np.radians(back), 0.0])), EE_POS, EE_QUAT)
    assert _ang_deg(tq, EE_QUAT) < 1.0


def test_pos_reach_limit_mouse_style():
    # A 1 m push clamps to pos_reach_limit; the overshoot is absorbed, so a 5 cm
    # reversal moves the target back 5 cm immediately (no retracing the 0.75 m).
    m = ClutchPoseMapper(rot_reach_limit=1.0, pos_reach_limit=0.25)
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    p, _ = m.target(CTRL_POS + np.array([1.0, 0, 0]), CTRL_QUAT, EE_POS, EE_QUAT)
    assert np.allclose(p, EE_POS + np.array([0.25, 0, 0]))
    p, _ = m.target(CTRL_POS + np.array([0.95, 0, 0]), CTRL_QUAT, EE_POS, EE_QUAT)
    assert np.allclose(p, EE_POS + np.array([0.20, 0, 0]))


def test_incremental_equals_absolute_within_reach_limit():
    # With the EE tracking the target and scale_rotation=1, the incremental path
    # reproduces the legacy absolute mapping exactly.
    m_inc = ClutchPoseMapper(rot_reach_limit=3.0, pos_reach_limit=10.0)
    m_abs = ClutchPoseMapper(rot_reach_limit=None, pos_reach_limit=None)
    for mm in (m_inc, m_abs):
        mm.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT)
    ee_q = EE_QUAT
    worst = 0.0
    for i in range(1, 41):
        q_ctrl = rotvec_to_quat(np.radians(i) * np.array([0.5, 0.7, 0.2]))
        _, tq_i = m_inc.target(CTRL_POS, q_ctrl, EE_POS, ee_q)
        _, tq_a = m_abs.target(CTRL_POS, q_ctrl)
        worst = max(worst, _ang_deg(tq_i, tq_a))
        ee_q = tq_i  # EE follows perfectly
    assert worst < 0.01


def test_rotation_pivot_swings_ee():
    # With a pivot 0.2 m below the EE, a pure controller rotation swings the EE
    # in an arc about the pivot: position changes with zero translation input.
    m = ClutchPoseMapper(rot_reach_limit=None, pos_reach_limit=None)
    pivot = EE_POS - np.array([0.0, 0.0, 0.2])  # offset above pivot = (0,0,0.2)
    m.engage(CTRL_POS, CTRL_QUAT, EE_POS, EE_QUAT, pivot_armbase=pivot)
    ctrl_now = rotvec_to_quat(np.array([0.0, np.pi / 2, 0.0]))  # +90° about Y
    p, q = m.target(CTRL_POS, ctrl_now)
    # R_y(90°) maps the (0,0,0.2) offset to (0.2,0,0); EE = pivot + (0.2,0,0).
    assert np.allclose(p, pivot + np.array([0.2, 0.0, 0.0]), atol=1e-9)
    assert _ang_deg(q, rotvec_to_quat(np.array([0.0, np.pi / 2, 0.0]))) < 1e-6
