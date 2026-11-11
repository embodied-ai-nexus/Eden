"""``eden.extensions.teleop.sonic.hands``: trigger -> Dex3 hand targets."""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.teleop.sonic.hands import (
    G1GripperGestureSolver,
    generate_finger_data,
    hand_targets_from_triggers,
)

_MIDDLE_CLOSE_LEFT = np.array([0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5])


def test_trigger_below_threshold_keeps_hand_open():
    left, right = hand_targets_from_triggers(0.4, 0.0)
    np.testing.assert_allclose(left, 0.0)
    np.testing.assert_allclose(right, 0.0)


def test_trigger_closes_middle_gesture():
    left, right = hand_targets_from_triggers(0.9, 0.9)
    np.testing.assert_allclose(left, _MIDDLE_CLOSE_LEFT)  # full close (distance snapped to 0)
    np.testing.assert_allclose(right, -_MIDDLE_CLOSE_LEFT)  # right hand fully negated


def test_partial_grip_interpolates():
    # Middle tip 0.4 away from the thumb -> grip 0.6 -> 60% of the closed pose.
    fingertips = generate_finger_data(0.0, 0.0)
    fingertips[14, 0, 3] = 1.0 - 0.4
    q = G1GripperGestureSolver("left")(fingertips)
    np.testing.assert_allclose(q, 0.6 * _MIDDLE_CLOSE_LEFT, atol=1e-12)


def test_grip_dead_zone():
    fingertips = generate_finger_data(0.0, 0.0)
    fingertips[14, 0, 3] = 0.04  # grip 0.04 < 0.05 threshold
    q = G1GripperGestureSolver("left")(fingertips)
    np.testing.assert_allclose(q, 0.0)


def test_largest_grip_selects_the_gesture():
    fingertips = generate_finger_data(0.0, 0.0)
    fingertips[9, 0, 3] = 0.9  # index grip 0.9
    fingertips[14, 0, 3] = 0.5  # middle grip 0.5
    q = G1GripperGestureSolver("left")(fingertips)
    index_close = np.array([-0.5, 0.7, 0.7, -1.5, -1.5, -0.6, -1.5])
    np.testing.assert_allclose(q, 0.9 * index_close, atol=1e-12)


def test_right_hand_thumb_rotation_keeps_its_sign():
    # The reference flips the thumb rotation (joint 0) per side BEFORE the
    # right hand's full negation, so thumb_0 has the SAME sign on both hands
    # for the index and ring gestures while joints 1..6 mirror.
    for finger, expected_left in (
        (9, np.array([-0.5, 0.7, 0.7, -1.5, -1.5, -0.6, -1.5])),  # index
        (19, np.array([0.5, 0.7, 0.7, -0.6, -1.5, -1.5, -1.5])),  # ring
    ):
        fingertips = generate_finger_data(0.0, 0.0)
        fingertips[finger, 0, 3] = 1.0  # full grip on that finger
        left = G1GripperGestureSolver("left")(fingertips)
        right = G1GripperGestureSolver("right")(fingertips)
        np.testing.assert_allclose(left, expected_left, atol=1e-12)
        assert right[0] == pytest.approx(left[0])  # thumb rotation: same sign
        np.testing.assert_allclose(right[1:], -left[1:], atol=1e-12)  # rest mirrors
