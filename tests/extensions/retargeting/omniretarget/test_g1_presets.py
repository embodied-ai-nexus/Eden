"""Tests for ``eden.extensions.retargeting.omniretarget.g1_presets``: the human-motion preprocessing helpers."""

from __future__ import annotations

import numpy as np

from eden.extensions.retargeting.omniretarget.g1_presets import (
    extract_foot_sticking_velocity,
    preprocess_human_motion,
)

JOINTS = ["Pelvis", "L_Toe", "R_Toe"]
TOES = ["L_Toe", "R_Toe"]


def _motion(toe_z: float) -> np.ndarray:
    """Two frames: pelvis 0.9 above the toes, the left toe dipping 0.02 lower in frame 1."""
    frames = np.array(
        [
            [[0.0, 0.0, 0.9], [0.1, 0.1, 0.0], [0.1, -0.1, 0.0]],
            [[0.0, 0.0, 0.9], [0.1, 0.1, -0.02], [0.1, -0.1, 0.0]],
        ]
    )
    frames[:, :, 2] += toe_z
    return frames


def test_a_motion_whose_toes_dip_below_the_mat_drops_onto_the_floor_and_is_scaled():
    out = preprocess_human_motion(_motion(toe_z=0.05), JOINTS, TOES, scale=0.5)

    # The lowest toe (0.03) goes to z = 0, then every coordinate is halved.
    np.testing.assert_allclose(out[1, 1], [0.05, 0.05, 0.0])
    np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 0.46])


def test_a_motion_whose_toes_never_reach_the_mat_is_lowered_onto_the_mat():
    out = preprocess_human_motion(_motion(toe_z=0.40), JOINTS, TOES, scale=1.0, mat_height=0.1)

    # The lowest toe (0.38) is lowered to the 0.1 mat, not to the floor.
    np.testing.assert_allclose(out[1, 1, 2], 0.1)
    np.testing.assert_allclose(out[0, 0, 2], 1.02)


def test_preprocessing_leaves_the_callers_motion_as_it_was():
    motion = _motion(toe_z=0.05)
    before = motion.copy()

    preprocess_human_motion(motion, JOINTS, TOES, scale=0.5)

    np.testing.assert_array_equal(motion, before)


def test_a_toe_sticks_on_frames_where_its_planar_speed_is_within_the_threshold():
    motion = np.zeros((4, 3, 3))
    motion[:, 1] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.05, 0.0, 0.3], [0.05, 0.01, 0.3]]  # left
    motion[:, 2] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]  # right

    flags = extract_foot_sticking_velocity(motion, JOINTS, TOES, velocity_threshold=0.01)

    # Frame 0 has no previous frame and never sticks. A vertical move (left, frame 1) is not
    # planar speed; a planar step of exactly the threshold (left, frame 3) still sticks.
    assert flags == [
        {"L_Toe": False, "R_Toe": False},
        {"L_Toe": True, "R_Toe": False},
        {"L_Toe": False, "R_Toe": True},
        {"L_Toe": True, "R_Toe": True},
    ]
