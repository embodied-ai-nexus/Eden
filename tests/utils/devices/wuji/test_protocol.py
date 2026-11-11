"""``eden.utils.devices.wuji.protocol``: the stream decoders and the vendor wrist -> Eden hand basis change.

The basis change is pinned against the documented axis map, not recomputed from the implementation's matrices.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from genesis.utils.geom import quat_to_R

from eden.utils.devices.wuji import protocol as P

# The vendor's wrist basis, per the Wuji "Coordinate Frames" doc, mapped onto Eden's
# hand basis (x distal / y left / z dorsal). Pinned literally: if R_HAND_WUJI is ever
# re-derived these are the sentences it has to keep satisfying.
#   x_wuji = radial   (toward the thumb)
#   y_wuji = dorsal on a LEFT glove, ventral (palm) on a RIGHT one
#   z_wuji = proximal (toward the elbow)
AXIS_MAP = {
    ("left", (1.0, 0.0, 0.0)): (0.0, -1.0, 0.0),  # thumb side of a left hand is Eden -y
    ("left", (0.0, 1.0, 0.0)): (0.0, 0.0, 1.0),  # back of the hand is Eden +z
    ("left", (0.0, 0.0, 1.0)): (-1.0, 0.0, 0.0),  # toward the elbow is Eden -x
    ("right", (1.0, 0.0, 0.0)): (0.0, 1.0, 0.0),  # thumb side of a right hand is Eden +y
    ("right", (0.0, 1.0, 0.0)): (0.0, 0.0, -1.0),  # palm normal is Eden -z
    ("right", (0.0, 0.0, 1.0)): (-1.0, 0.0, 0.0),
}


@pytest.mark.parametrize("key, expected", sorted(AXIS_MAP.items()))
def test_wrist_axes_map_to_the_eden_hand_convention(key, expected):
    side, axis = key
    assert P.wuji_points_to_hand([axis], side)[0] == pytest.approx(expected)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_matrix_and_quaternion_forms_of_the_basis_change_agree(side):
    """Points rotate by the matrix and orientations conjugate by the quaternion — same rotation."""
    assert np.allclose(quat_to_R(P.Q_HAND_WUJI[side]), P.R_HAND_WUJI[side])
    assert set(P.SIDES) == set(P.R_HAND_WUJI) == set(P.Q_HAND_WUJI)


def test_the_hand_frame_removes_the_vendors_dorsal_ventral_y_flip():
    """The same anatomical placement reads the same on both gloves along x (distal) and z (dorsal).

    Vendor +y is dorsal on a left glove and ventral on a right one, so raw left data is
    the right's mirrored in y. In the hand frame x is always out along the fingers and z
    always out of the back of the hand, so those two components agree. y cannot: it is
    radial on one hand and ulnar on the other, because a left hand *is* a mirrored right
    hand — no right-handed frame can make the two coincide.
    """
    # 2 cm radial (toward the thumb), 1 cm ventral (toward the palm), 8 cm distal.
    right_raw = np.array([[0.02, 0.01, -0.08]])
    left_raw = right_raw * np.array([1.0, -1.0, 1.0])  # the vendor's y flip
    left = P.wuji_points_to_hand(left_raw, "left")[0]
    right = P.wuji_points_to_hand(right_raw, "right")[0]
    assert left[[0, 2]] == pytest.approx(right[[0, 2]])
    assert left[0] == pytest.approx(0.08)  # distal
    assert left[2] == pytest.approx(-0.01)  # 1 cm toward the palm = below the dorsum
    assert left[1] == pytest.approx(-right[1])  # radial on the right, ulnar on the left


def test_pose_orientation_is_conjugated_not_just_relabelled():
    """A rotation about the wrist's proximal axis must come out about Eden's -x."""
    # 90 deg about +z_wuji (proximal) -> 90 deg about -x_eden (distal, reversed).
    half = np.sqrt(0.5)
    pos, quat = P.wuji_pose_to_hand([0.0, 0.0, 0.0], [0.0, 0.0, half, half], "right")
    R = quat_to_R(np.asarray(quat))
    axis_in = np.array([0.0, 1.0, 0.0])  # a vector orthogonal to the rotation axis
    assert np.allclose(pos, 0.0)
    # Rotating +y by +90 deg about -x sends it to -z.
    assert R @ axis_in == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_pose_position_is_rotated_and_quaternion_is_wxyz():
    pos, quat = P.wuji_pose_to_hand([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0], "right")
    assert pos == pytest.approx([-0.3, 0.1, -0.2])
    assert quat == pytest.approx([1.0, 0.0, 0.0, 0.0])  # identity, scalar first


def test_unsolved_pose_reports_identity_orientation():
    """The glove publishes a zero quaternion for a pose it has not solved."""
    _, quat = P.wuji_pose_to_hand([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "left")
    assert quat == pytest.approx([1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "call, bad",
    [
        (lambda v: P.wuji_points_to_hand(v, "right"), [0.0, 0.0, 0.0]),
        (lambda v: P.wuji_pose_to_hand(v, [0, 0, 0, 1], "right"), [0.0, 0.0]),
    ],
)
def test_frame_helpers_validate_shapes(call, bad):
    with pytest.raises(ValueError):
        call(bad)


@pytest.mark.parametrize("bad", ["Left", "both", ""])
def test_unknown_side_names_the_options(bad):
    with pytest.raises(ValueError, match="expected one of"):
        P.check_side(bad)


# ---------------------------------------------------------------------------
# the MANO frame a retargeter consumes
# ---------------------------------------------------------------------------


def _synthetic_hand(side):
    """An anatomically plausible open hand, (21, 3), in Eden's hand frame.

    x distal, y left, z dorsal; the thumb sits radial, which is +y on a right hand and
    -y on a left one. Only the wrist / index-MCP / middle-MCP triangle drives the frame
    estimate, but the whole hand is built so the fixture reads as a hand.
    """
    radial = 1.0 if side == "right" else -1.0
    kp = np.zeros((21, 3))
    mcp_y = {"index_finger": 0.020, "middle_finger": 0.0, "ring_finger": -0.020, "pinky": -0.040}
    mcp_x = {"index_finger": 0.085, "middle_finger": 0.090, "ring_finger": 0.085, "pinky": 0.078}
    lengths = {"index_finger": 0.045, "middle_finger": 0.050, "ring_finger": 0.045, "pinky": 0.035}
    for finger, base in [("index_finger", 5), ("middle_finger", 9), ("ring_finger", 13), ("pinky", 17)]:
        for k in range(4):
            kp[base + k] = [mcp_x[finger] + lengths[finger] * k / 3.0, radial * mcp_y[finger], 0.004 * k]
    for k in range(4):  # thumb: radial and slightly ventral of the palm plane
        kp[1 + k] = [0.030 + 0.022 * k, radial * (0.030 + 0.008 * k), -0.010 - 0.004 * k]
    return kp


def _upstream_wrist_rot(kp):
    """``SingleHandDetector._estimate_frame_from_hand_points``, verbatim, for the pin below."""
    points = kp[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_to_mano_matches_the_upstream_per_frame_estimate(side):
    """The constant replaces dex-retargeting's per-frame SVD, so it must reproduce it.

    A MediaPipe detection arrives in an arbitrary camera frame, so upstream re-derives a
    wrist frame every tick; glove keypoints are already hand-anchored, making that
    estimate a constant. Pin it against the real procedure, not against its derivation.
    """
    from eden.extensions.retargeting.dex_retargeter.constants import OPERATOR2MANO, HandType

    kp = _synthetic_hand(side)
    upstream = (kp - kp[0]) @ _upstream_wrist_rot(kp - kp[0]) @ OPERATOR2MANO[HandType[side]]
    assert P.keypoints_to_mano(kp) == pytest.approx(upstream, abs=1e-6)


def test_hand_to_mano_is_the_same_matrix_for_both_hands():
    """The per-side operator2mano exists to absorb a mirror R_HAND_WUJI already removed."""
    assert P.HAND_TO_MANO.shape == (3, 3)
    assert float(np.linalg.det(P.HAND_TO_MANO)) == pytest.approx(1.0)


def test_keypoints_to_mano_puts_the_wrist_at_the_origin_and_validates_shape():
    kp = _synthetic_hand("right") + np.array([1.0, 2.0, 3.0])  # an arbitrary offset
    mano = P.keypoints_to_mano(kp)
    assert mano[0] == pytest.approx([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"expected \(21, 3\)"):
        P.keypoints_to_mano(kp[:20])


def test_hand_connections_index_real_landmarks():
    flat = [i for pair in P.HAND_CONNECTIONS for i in pair]
    assert max(flat) == P.N_LANDMARKS - 1 and min(flat) == 0
    assert len(P.HAND_CONNECTIONS) == 21  # 6 palm edges + 3 per finger + ... one per bone
    assert set(P.FINGERTIP_LANDMARKS) == {4, 8, 12, 16, 20}


def test_skeleton_decodes_21_landmarks_in_mediapipe_order(frames):
    pos, quat, conf = P.skeleton_to_arrays(frames.skeleton(), "right")
    assert pos.shape == (21, 3) and quat.shape == (21, 4) and conf.shape == (21,)
    assert quat[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert conf[3] == pytest.approx(0.03)
    # landmark 1 is the thumb CMC, at wuji (0.001, 0.002, -0.003)
    assert pos[1] == pytest.approx([0.003, 0.001, -0.002])


def test_skeleton_wuji_frame_passes_positions_through_untouched(frames):
    pos, _, _ = P.skeleton_to_arrays(frames.skeleton(), "right", frame="wuji")
    assert pos[1] == pytest.approx([0.001, 0.002, -0.003])


def test_skeleton_with_the_wrong_joint_count_raises(frames):
    with pytest.raises(ValueError, match="expected 21 skeleton joints"):
        P.skeleton_to_arrays(frames.skeleton(n=20), "right")


def test_fingertips_and_emf_decode_five_poses(frames):
    tips = SimpleNamespace(header=None, poses=[frames.pose((0.0, 0.0, float(i))) for i in range(5)])
    pos, quat, conf = P.fingertips_to_arrays(tips, "left")
    assert pos.shape == (5, 3) and quat.shape == (5, 4) and conf.shape == (5,)
    pos, _, _ = P.emf_poses_to_arrays(tips, "left")
    assert pos.shape == (5, 3)


def test_joint_angles_drop_the_padding_slot(frames):
    packed, conf = P.pack_joint_angles(frames.joint_angles())
    assert packed.shape == (P.N_HAND_DOFS,) == (21,)
    assert conf == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6])
    # thumb keeps all five slots, the index finger only its first four
    assert packed[:5] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert packed[5:9] == pytest.approx([0.5, 0.6, 0.7, 0.8])
    assert 0.9 not in packed  # the index finger's padded fifth slot never reaches the caller


def test_joint_angles_validate_the_5x5_shape(frames):
    frame = frames.joint_angles()
    frame.fingers[2].angles = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="angle slots for the middle"):
        P.pack_joint_angles(frame)


def test_tactile_grid_masks_invalid_taxels_to_nan(frames):
    grid = P.tactile_grid(frames.tactile().data)
    assert grid.shape == (P.TACTILE_ROWS, P.TACTILE_COLS) == (24, 31)
    assert np.isnan(grid[0, :3]).all()
    assert grid[0, 3] == pytest.approx(0.25)
    # the whole point of the mask: a naive reduction must not report a plausible number
    assert np.isnan(grid.mean())
    assert np.nanmean(grid) == pytest.approx(0.25)


def test_tactile_valid_mask_matches_the_sentinel(frames):
    mask = P.tactile_valid(frames.tactile().data)
    assert mask.shape == (24, 31)
    assert mask.sum() == P.N_TAXELS - 3


@pytest.mark.parametrize("decode", [P.tactile_grid, P.tactile_taxels])
@pytest.mark.parametrize("n", [P.N_TAXELS - 1, P.N_TAXELS + 1])
def test_tactile_rejects_a_wrong_sized_sheet(decode, n):
    with pytest.raises(ValueError, match="expected 744 taxels"):
        decode([0.0] * n)


def test_tactile_taxels_lists_only_the_live_nodes(frames):
    """The node-wise view: one entry per real taxel, dead ones gone rather than nan."""
    index, value = P.tactile_taxels(frames.tactile(fill=0.25, invalid=(0, 1, 2)).data)
    assert index.shape == (P.N_TAXELS - 3, 2)
    assert value.shape == (P.N_TAXELS - 3,)
    assert not np.isnan(value).any(), "a live taxel never carries nan"
    assert value == pytest.approx(0.25)
    # the first three cells of row 0 are the dead ones, so node 0 is (0, 3)
    assert index[0].tolist() == [0, 3]
    assert index[-1].tolist() == [P.TACTILE_ROWS - 1, P.TACTILE_COLS - 1]
    # Grid coordinates must never come back as floats a caller could read as pressure.
    assert index.dtype == np.int64 and value.dtype == np.float64


def test_tactile_taxels_flat_ids_are_row_major_into_the_sheet(frames):
    index, value = P.tactile_taxels(frames.tactile(invalid=(0, 1, 2)).data, flat=True)
    assert index.shape == value.shape == (P.N_TAXELS - 3,)
    assert index[0] == 3 and index[-1] == P.N_TAXELS - 1
    rc, _ = P.tactile_taxels(frames.tactile(invalid=(0, 1, 2)).data)
    assert (rc[:, 0] * P.TACTILE_COLS + rc[:, 1] == index).all(), "the two index forms must agree"


def test_tactile_taxels_agree_with_the_masked_grid(frames):
    """Same numbers as tactile_grid, just without the holes — the two must not drift."""
    data = frames.tactile(fill=0.4, invalid=(5, 100, 700)).data
    grid = P.tactile_grid(data)
    index, value = P.tactile_taxels(data)
    assert value == pytest.approx(grid[index[:, 0], index[:, 1]])
    assert len(value) == int(P.tactile_valid(data).sum())


def test_tactile_zones_decode_with_the_documented_lengths(frames):
    zones = P.tactile_zones_to_dict(frames.zones())
    assert set(zones) == set(P.TACTILE_ZONE_SIZES)
    assert {k: v.shape[0] for k, v in zones.items()} == P.TACTILE_ZONE_SIZES
    assert np.isnan(zones["thumb"][-1])


def test_tactile_zones_reject_a_resized_zone(frames):
    zones = frames.zones()
    zones.palm = [0.0] * (P.TACTILE_ZONE_SIZES["palm"] - 1)
    with pytest.raises(ValueError, match="expected 308 taxels in the palm zone"):
        P.tactile_zones_to_dict(zones)


def test_point_cloud_decodes_xyz_and_pressure(frames):
    points, pressure = P.decode_point_cloud(frames.point_cloud())
    assert points == pytest.approx(np.array([[0.0, 0.1, 0.2]]))
    assert pressure == pytest.approx([0.75])


def test_point_cloud_rejects_a_payload_that_disagrees_with_its_header(frames):
    """A truncated frame would otherwise decode into a plausible-looking short cloud."""
    cloud = frames.point_cloud(points=((0.0, 0.1, 0.2),), pressure=(0.75,))
    cloud.point_count = 2
    with pytest.raises(ValueError, match=r"point_count=2 but carries 1 points"):
        P.decode_point_cloud(cloud)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stride": 12},
        {"fields": [SimpleNamespace(name="x", offset=0, type=P.POINT_FIELD_FLOAT32)]},
        {
            "fields": [
                SimpleNamespace(name=n, offset=4 * i, type=2)  # UINT8, not FLOAT32
                for i, n in enumerate(P.POINT_CLOUD_FIELDS)
            ]
        },
    ],
)
def test_point_cloud_rejects_an_unexpected_layout(kwargs, frames):
    """A firmware layout change must raise, not reinterpret the bytes into garbage."""
    with pytest.raises(ValueError, match="unexpected tactile point-cloud layout"):
        P.decode_point_cloud(frames.point_cloud(**kwargs))
