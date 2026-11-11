"""Tests for ``eden.extensions.retargeting.otretarget.supports``: untracked supports read off an object track."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from eden.extensions.retargeting.otretarget.retargeter import InteractionObject
from eden.extensions.retargeting.otretarget.supports import find_untracked_supports
from eden.utils.geom import R_to_quat

T = 12
FAR_HANDS = np.full((T, 4, 3), 5.0)  # hands nowhere near the object


def _still(mesh, pos, rot=np.eye(3)):
    return InteractionObject("thing", mesh, np.tile(np.r_[pos, R_to_quat(rot)], (T, 1)))


def _box_resting_at(height, extents=(0.4, 0.3, 0.2)):
    return _still(trimesh.creation.box(extents=extents), [0.3, -0.2, height + extents[2] / 2])


def test_a_box_resting_still_above_the_floor_gets_a_plate_under_it():
    (support,) = find_untracked_supports(_box_resting_at(0.5), FAR_HANDS)
    assert support.height == pytest.approx(0.5, abs=1e-9)
    np.testing.assert_allclose(support.center, [0.3, -0.2], atol=1e-9)
    # The footprint plus the 2 cm margin on every side, in either axis order (the plate axes come from an SVD).
    np.testing.assert_allclose(sorted(support.extents[:2]), [0.34, 0.44], atol=1e-9)
    plate = support.as_object(T)
    assert plate.fixed
    top = (plate.mesh.vertices @ Rotation.from_quat(plate.poses[0, [4, 5, 6, 3]]).as_matrix().T + plate.poses[0, :3])[
        :, 2
    ]
    assert top.max() == pytest.approx(0.5, abs=1e-9)


def test_a_box_with_a_hand_on_it_is_held_not_resting():
    hands = FAR_HANDS.copy()
    hands[:, 0] = [0.3, -0.2, 0.72]  # 2 cm over the box top: held still in the air, not on a table
    assert find_untracked_supports(_box_resting_at(0.5), hands) == []


def test_a_box_floating_a_few_centimetres_up_is_a_capture_offset():
    assert find_untracked_supports(_box_resting_at(0.08), FAR_HANDS) == []


def test_a_box_touching_down_on_one_edge_rests_on_nothing_level():
    tilted = Rotation.from_euler("x", 35, degrees=True).as_matrix()
    assert (
        find_untracked_supports(_still(trimesh.creation.box(extents=(0.4, 0.3, 0.2)), [0, 0, 0.7], tilted), FAR_HANDS)
        == []
    )


def test_an_upside_down_chair_rests_on_its_seat_and_the_plate_clears_the_hanging_back():
    # Seat 40x40 cm with its underside at 0.6 m; the backrest hangs 35 cm down from one edge, over the table edge.
    seat = trimesh.creation.box(extents=(0.4, 0.4, 0.03)).apply_translation([0.0, 0.0, 0.615])
    back = trimesh.creation.box(extents=(0.4, 0.03, 0.35)).apply_translation([0.0, -0.215, 0.425])
    chair = _still(trimesh.util.concatenate([seat, back]), [0.0, 0.0, 0.0])
    (support,) = find_untracked_supports(chair, FAR_HANDS)
    assert support.height == pytest.approx(0.6, abs=1e-9)
    corners = trimesh.creation.box(extents=support.extents).vertices[:, :2]
    rot = np.array([[np.cos(support.yaw), -np.sin(support.yaw)], [np.sin(support.yaw), np.cos(support.yaw)]])
    assert (corners @ rot.T + support.center)[:, 1].min() > -0.2 + 0.01 - 1e-9  # the back's face, with clearance


def test_hand_points_must_match_the_object_frames():
    with pytest.raises(ValueError, match=r"hand_points has shape \(5, 4, 3\); expected \(T, H, 3\) with T = 12"):
        find_untracked_supports(_box_resting_at(0.5), FAR_HANDS[:5])


def test_the_plate_top_is_no_higher_than_the_object_over_it():
    # A box tilted 2 deg about x rests on its lowest edge; the plate must not start inside the box.
    tilt = Rotation.from_euler("x", 2, degrees=True).as_matrix()
    box = _still(trimesh.creation.box(extents=(0.4, 0.3, 0.2)), [0.0, 0.0, 0.6], tilt)
    (support,) = find_untracked_supports(box, FAR_HANDS)
    lowest = (box.mesh.vertices @ tilt.T + [0.0, 0.0, 0.6])[:, 2].min()
    assert support.height <= lowest + 1e-9
