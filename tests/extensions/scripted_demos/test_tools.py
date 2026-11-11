"""``tools``: footprints measured from points, and the motions a routine's Tools handle builds."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import numpy as np
import pytest
import torch

from eden.extensions.scripted_demos.tools import (
    CONTACT,
    DONE,
    NO_CONTACT,
    REACHED,
    TIMEOUT,
    MotionKind,
    Outcome,
    Tools,
    aim_rotation,
    footprint_from_points,
)
from eden.utils.geom import transform_by_quat


def _box_points(center, half, yaw, *, tilt_x=0.0, n=8) -> np.ndarray:
    """A dense grid over a box of half extents ``half`` turned by ``yaw``, its top shifted by ``tilt_x`` per metre of height."""
    axes = [np.linspace(-h, h, n) for h in half]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    local[:, 0] += tilt_x * (local[:, 2] + half[2])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ rot.T + np.asarray(center)


def test_a_box_footprint_gives_its_long_axis_extents_and_top():
    footprint = footprint_from_points(_box_points((0.1, -0.2, 0.9), (0.05, 0.016, 0.075), yaw=0.5))
    assert footprint.yaw == pytest.approx(0.5, abs=1e-6)
    assert (footprint.half_length, footprint.half_width) == pytest.approx((0.05, 0.016), abs=1e-6)
    assert (footprint.top_z, footprint.bottom_z) == pytest.approx((0.975, 0.825))
    np.testing.assert_allclose(footprint.center, (0.1, -0.2, 0.9), atol=1e-6)
    np.testing.assert_allclose(footprint.top_center[:2], (0.1, -0.2), atol=1e-6)


def test_the_long_axis_heading_has_no_direction():
    # A box turned by 2.5 rad lies along the same line as one turned by 2.5 - pi.
    footprint = footprint_from_points(_box_points((0, 0, 0), (0.05, 0.02, 0.02), yaw=2.5))
    assert footprint.yaw == pytest.approx(2.5 - math.pi, abs=1e-6)


def test_a_leaning_object_is_met_from_above_off_its_centre():
    footprint = footprint_from_points(_box_points((0, 0, 0.1), (0.02, 0.02, 0.1), yaw=0.0, tilt_x=0.2))
    # The top is shifted 0.2 * 0.2 m = 4 cm along x; the footprint's centre only half that.
    assert footprint.top_center[0] - footprint.center[0] == pytest.approx(0.02, abs=2e-3)


def test_too_few_points_are_refused():
    with pytest.raises(ValueError, match="at least 3 points.*got 2"):
        footprint_from_points(np.zeros((2, 3)))


@pytest.mark.parametrize(
    ("status", "ok"), [(REACHED, True), (CONTACT, True), (DONE, True), (TIMEOUT, False), (NO_CONTACT, False)]
)
def test_an_outcome_is_ok_when_the_motion_did_what_it_was_asked(status, ok):
    assert Outcome("m", status, 1.0, 0.0).ok is ok


def test_motions_are_labelled_and_validated_before_they_reach_the_executor():
    tools = Tools(executor=None, env_id=0, rng=np.random.default_rng(0))  # constructors read no executor state
    guarded = tools.move_until_contact((0, 0, -2), max_distance=0.1, speed=0.02)
    assert guarded.kind is MotionKind.GUARDED
    assert guarded.direction == (0.0, 0.0, -1.0)
    assert guarded.label == "move_until_contact(0.000, 0.000, -1.000, max 0.100 m)"
    assert tools.close_gripper().grasp == 1.0
    assert tools.open_gripper().grasp == 0.0
    with pytest.raises(ValueError, match=r"non-zero direction, got \(0, 0, 0\)"):
        tools.move_until_contact((0, 0, 0))
    with pytest.raises(ValueError, match=r"3 components \(x, y, z\), got 2"):
        tools.move_to((0.1, 0.2), speed=0.1)
    with pytest.raises(ValueError, match="speed must be positive"):
        tools.move_to((0.1, 0.2, 0.3), speed=0.0)


def test_the_long_axis_is_the_one_of_larger_extent_not_of_more_variance():
    # Dense along x (4 cm), two points far out along y (10 cm): x has more variance, y more extent.
    dense = np.stack([np.linspace(-0.02, 0.02, 100), np.zeros(100), np.zeros(100)], axis=-1)
    sparse = np.array([[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]])
    footprint = footprint_from_points(np.concatenate([dense, sparse]))
    assert abs(footprint.yaw) == pytest.approx(math.pi / 2, abs=1e-6)
    assert (footprint.half_length, footprint.half_width) == pytest.approx((0.05, 0.02), abs=1e-6)


def test_a_non_positive_duration_or_tolerance_is_refused_when_the_motion_is_built():
    tools = Tools(executor=None, env_id=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"positive duration_s \[s\], got 0"):
        tools.wait(0)
    with pytest.raises(ValueError, match="tolerance > 0 and timeout_s >= 0, got -0.01"):
        tools.move_to((0, 0, 1), tolerance=-0.01, speed=0.1)


def _turn(quat, vector) -> np.ndarray:
    return transform_by_quat(torch.tensor([vector], dtype=torch.float64), torch.tensor([quat], dtype=torch.float64))[
        0
    ].numpy()


@pytest.mark.parametrize("finger_axis_yaw", [math.pi / 2, 0.3])
def test_aim_points_the_approach_axis_and_the_closing_line_where_asked(finger_axis_yaw):
    straight_down = (math.cos(finger_axis_yaw), math.sin(finger_axis_yaw), 0.0)
    assert aim_rotation((0, 0, -1), straight_down, finger_axis_yaw) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)
    # Pointing sideways at a panel, closing vertically; a closing line off the perpendicular is squared up.
    rot = aim_rotation((0, -2, 0), (0.0, 0.3, 1.0), finger_axis_yaw)
    np.testing.assert_allclose(_turn(rot, (0.0, 0.0, -1.0)), (0.0, -1.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(_turn(rot, straight_down), (0.0, 0.0, 1.0), atol=1e-9)


def test_aim_and_move_to_refuse_what_names_no_rotation():
    with pytest.raises(ValueError, match=r"closing direction \(0, 0, 3\) is parallel to its approach"):
        aim_rotation((0, 0, -1), (0, 0, 3), 0.0)
    tools = Tools(executor=None, env_id=0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match=r"a heading \(yaw=0.5\) or a full rotation"):
        tools.move_to((0, 0, 1), 0.5, rot=(1, 0, 0, 0), speed=0.1)
    tilted = tools.move_to((0, 0, 1), rot=(2, 0, 0, 0), speed=0.1)
    assert tilted.rot == (1.0, 0.0, 0.0, 0.0)
    assert tilted.label == "move_to(0.000, 0.000, 1.000, rot=(1.000, 0.000, 0.000, 0.000))"
