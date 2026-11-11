"""Wuji Glove data layout: stream shapes, landmark tables, and the basis change into Eden.

Everything here is pure numpy and duck-typed over the SDK message objects, so the
whole decode path is testable without a glove (or without ``wuji_sdk`` installed).
:mod:`eden.utils.devices.wuji.glove` owns the connection and the subscriptions.

Frames
------
Wuji publishes hand data in the **wrist frame** (``l_wrist`` / ``r_wrist``), which the
vendor defines anatomically (docs, *Coordinate Frames*)::

    x = radial     (ulnar -> radial, i.e. toward the thumb)
    z = proximal   (distal -> proximal, i.e. along the bone toward the elbow)
    y = right-hand rule, so it MIRRORS between hands:
        left glove  +y = dorsal (back of the hand)
        right glove +y = ventral (palm)

So +y means the *opposite* physical direction on the two gloves. Eden pins every axis
to anatomy instead, in the Genesis basis — x forward along the fingers, y left, z up —
so :data:`R_HAND_WUJI` maps::

    x_eden = -z_wuji                                  (distal, out along the fingers)
    z_eden = +y_wuji (left) / -y_wuji (right)         (dorsal, out of the back of the hand)
    y_eden = z_eden x x_eden = -x_wuji (left) / +x_wuji (right)

Both per-side matrices are proper rotations (det = +1), and the per-side sign flip is exactly
the vendor's own note that "left-hand glove data is mirrored from the right hand, with
the Y direction inverted". Note what this does *not* buy: y is radial on a right hand
and ulnar on a left one, because a left hand is a mirrored right hand and no right-handed
frame can make the two agree. x (distal) and z (dorsal) do agree, which is what a
retargeter and a pose-plotting recipe actually read.

Tactile
-------
The tactile matrix is a fixed ``24 x 31`` row-major grid of 744 taxels, only some of
which are real; the rest are permanently pinned to :data:`TACTILE_INVALID` (``-1.0``).
Two counts describe "real" and they disagree by eight — read :data:`N_ACTIVE_TAXELS` and
:data:`N_UNMASKED_TAXELS` before hard-coding either, here or in prose. That sentinel is a
*number*, so ``grid.mean()`` over a raw frame is silently wrong —
:func:`tactile_grid` therefore turns masked taxels into ``nan``, which fails loudly
instead. Use :func:`tactile_valid` for the boolean mask, or :func:`tactile_taxels` for
the node-wise view: one ``(index, value)`` entry per taxel that actually measured
something, with the dead ones dropped rather than masked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from genesis.utils.geom import R_to_quat, inv_quat, normalize, transform_by_R

from eden.utils.geom import quat_mul, xyzw_to_wxyz

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt

# ----------------------------------------------------------------------------------
# hand pose
# ----------------------------------------------------------------------------------

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# MediaPipe hand-landmark order — the order `hand_skeleton` publishes its 21 joints in,
# and the order `DexRetargeter` / `wuji_sdk.RetargetSession` expect their keypoints in.
HAND_LANDMARKS = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
N_LANDMARKS = len(HAND_LANDMARKS)

# Index of each finger's tip landmark. HAND_LANDMARKS is thumb-first, so this is FINGERS
# order: (4, 8, 12, 16, 20).
FINGERTIP_LANDMARKS = tuple(i for i, name in enumerate(HAND_LANDMARKS) if name.endswith("_tip"))

# MediaPipe bone connectivity (palm arch first, then one chain per finger), for drawing
# the hand as a skeleton rather than a cloud of dots.
HAND_CONNECTIONS = (
    (0, 1),
    (0, 5),
    (0, 17),
    (5, 9),
    (9, 13),
    (13, 17),  # palm
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
)

# `hand_joint_angles` is a fixed 5x5 array (one row per finger, FINGERS order). The
# thumb uses all five slots (CMC's 3 DOF + MCP + IP); every other finger uses the
# first four (MCP's 2 DOF + PIP + DIP) and pads the fifth with 0. 5 + 4*4 = 21 DOF.
JOINT_ANGLE_SLOTS = 5
FINGER_NUM_DOFS = (5, 4, 4, 4, 4)
N_HAND_DOFS = sum(FINGER_NUM_DOFS)

# ----------------------------------------------------------------------------------
# tactile
# ----------------------------------------------------------------------------------

TACTILE_ROWS = 24
TACTILE_COLS = 31
N_TAXELS = TACTILE_ROWS * TACTILE_COLS  # 744 grid cells, most but not all of them real
TACTILE_INVALID = -1.0

#: Taxels the vendor's model covers: the `tactile_point_cloud` point count and the sum of
#: the zones' documented active counts both come to this.
N_ACTIVE_TAXELS = 526

#: Cells a live sheet actually reports as non-sentinel. Measured on firmware 0.11.1 (both a
#: left and a right glove, a fixed mask over 360 consecutive frames) -- deliberately NOT
#: equal to :data:`N_ACTIVE_TAXELS`, so do not derive an active count by counting cells that
#: are not :data:`TACTILE_INVALID`: eight of them read real pressure yet fall outside the
#: 526 the point cloud and the zones account for.
N_UNMASKED_TAXELS = 534

# `tactile_zones` splits the same sheet by hand region. The arrays carry padding too, so a
# zone's length is not its active-taxel count (those sum to N_ACTIVE_TAXELS).
TACTILE_ZONE_SIZES = {"palm": 308, "thumb": 50, "index": 45, "middle": 60, "ring": 54, "pinky": 55}

# `tactile_point_cloud` packs one record per point: x, y, z, pressure as float32 in the
# wrist frame. 7 is the ROS `sensor_msgs/PointField` code for FLOAT32. The stride is
# derived, not asserted separately, so a field added here cannot desynchronise it from the
# offsets `decode_point_cloud` expects.
FLOAT32_BYTES = 4
POINT_CLOUD_FIELDS = ("x", "y", "z", "pressure")
POINT_CLOUD_STRIDE = FLOAT32_BYTES * len(POINT_CLOUD_FIELDS)  # 16
POINT_FIELD_FLOAT32 = 7

# ----------------------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------------------

#: Wuji wrist basis -> Eden hand basis (x distal, y left, z dorsal), per glove side.
R_HAND_WUJI = {
    "left": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "right": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
}

#: The same basis change as a wxyz quaternion. Positions rotate by the matrix above (one
#: matmul for a whole array) and orientations conjugate by this — both on every path, and
#: pinned equal to each other in the tests.
Q_HAND_WUJI = {side: R_to_quat(R) for side, R in R_HAND_WUJI.items()}

#: The two glove sides — the keys of every per-side table above.
SIDES = tuple(R_HAND_WUJI)

#: The frames every pose getter accepts: Eden's hand basis, or the vendor's raw wrist one.
FRAMES = ("hand", "wuji")

#: Serial-number letter -> side. The Wuji firmware encodes handedness in ``sn[3]`` (the
#: convention ``wuji_sdk.Handedness`` documents for the Wuji Hand 2); verified to hold for
#: gloves too on ``WG1J...`` = left and ``WG1K...`` = right. It is only ever a **hint** —
#: :meth:`~eden.utils.devices.wuji.glove.WujiGlove.side` comes from the device's own
#: ``hand_side()``, which needs a connection.
SIDE_FROM_SN_LETTER = {"J": "left", "K": "right"}


def side_from_sn(sn: str) -> str | None:
    """Guess a glove's side from its serial number, or None if the letter is unfamiliar.

    Lets a caller connect to the likely glove first instead of probing every one, without
    ever trusting the guess: whoever uses it must still confirm against ``hand_side()``.
    """
    letter = sn[3].upper() if len(sn) > 3 else ""
    return SIDE_FROM_SN_LETTER.get(letter)


def _one_of(value: str, options: Collection[str], what: str) -> str:
    """Return ``value`` if ``options`` contains it, else raise naming ``what`` and the options."""
    if value not in options:
        raise ValueError(f"unknown {what} {value!r}; expected one of {sorted(options)}")
    return value


def check_side(side: str) -> str:
    """Return ``side`` if it is ``"left"`` or ``"right"``, else raise naming both."""
    return _one_of(side, SIDES, "glove side")


def check_frame(frame: str) -> str:
    """Return ``frame`` if it is ``"hand"`` or ``"wuji"``, else raise naming both."""
    return _one_of(frame, FRAMES, "frame")


#: Below this norm a quaternion carries no rotation: what the glove publishes for a pose
#: it has not solved.
_MIN_QUAT_NORM = 1e-8


def _quat_wxyz(orientation_xyzw: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Normalize an ``(x, y, z, w)`` quaternion and reorder it to Genesis' wxyz.

    A degenerate (zero-norm) input — what the glove reports for a pose it has not solved —
    comes back as identity rather than as a division by zero.
    """
    xyzw = np.asarray(orientation_xyzw, dtype=float).reshape(-1)
    if xyzw.shape != (4,):
        raise ValueError(f"expected a 4-element (x, y, z, w) quaternion, got shape {xyzw.shape}")
    if np.linalg.norm(xyzw) < _MIN_QUAT_NORM:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return normalize(xyzw_to_wxyz(xyzw))


def wuji_points_to_hand(points: npt.ArrayLike, side: str) -> npt.NDArray[np.float64]:
    """Rotate ``(N, 3)`` wrist-frame points into the Eden hand frame for ``side``."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got shape {pts.shape}")
    return transform_by_R(pts, R_HAND_WUJI[check_side(side)])


def wuji_pose_to_hand(
    position: npt.ArrayLike, orientation_xyzw: npt.ArrayLike, side: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert one wrist-frame pose into the Eden hand frame.

    Parameters
    ----------
    position:
        ``(x, y, z)`` in metres, as ``wuji_sdk.Pose.position`` reports it.
    orientation_xyzw:
        The pose's ``wuji_sdk.Quaternion`` as ``(x, y, z, w)`` — the SDK's scalar-last
        order, *not* Genesis' wxyz.
    side:
        ``"left"`` or ``"right"``.

    Returns
    -------
    tuple
        ``(pos (3,), quat (4,))`` in the hand frame, the quaternion **wxyz** as
        everything Genesis-facing expects. A degenerate (zero-norm) input quaternion —
        what the glove reports for a pose it has not solved — comes back as identity.
    """
    side = check_side(side)
    pos = np.asarray(position, dtype=float).reshape(-1)
    if pos.shape != (3,):
        raise ValueError(f"expected a 3-element position, got shape {pos.shape}")
    out_pos, out_quat = _poses_to_hand(pos[None], _quat_wxyz(orientation_xyzw)[None], side)
    return out_pos[0], out_quat[0]


def _poses_to_hand(
    pos: npt.NDArray[np.float64], quat_wxyz: npt.NDArray[np.float64], side: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate ``(N, 3)`` positions and ``(N, 4)`` wxyz orientations into the hand frame.

    The whole array at once: one matmul for the positions, one pair of Hamilton products
    for the orientations. ``side`` must already be checked. It is a change of *basis*, so
    the orientations are conjugated rather than merely relabelled.
    """
    q = np.broadcast_to(Q_HAND_WUJI[side], quat_wxyz.shape)
    return transform_by_R(pos, R_HAND_WUJI[side]), quat_mul(quat_mul(q, quat_wxyz), inv_quat(q))


#: Eden hand frame -> the MANO frame ``dex-retargeting`` optimizes in (right-multiplied,
#: ``kp @ HAND_TO_MANO``, matching upstream's ``keypoints @ wrist_rot @ operator2mano``).
#:
#: Upstream re-estimates a wrist frame from the keypoints themselves every frame (an SVD
#: over the wrist / index-MCP / middle-MCP triangle) because a MediaPipe detection arrives
#: in an arbitrary camera frame. The glove does not: its output is already anchored to the
#: hand, so that estimate is a known constant and this is it. It is the *same* matrix for
#: both hands — upstream's per-side ``operator2mano`` exists to absorb the mirror that
#: :data:`R_HAND_WUJI` has already removed. ``tests/utils/devices/wuji/test_protocol.py`` pins it against
#: the upstream procedure rather than against this derivation.
HAND_TO_MANO = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def keypoints_to_mano(keypoints: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Hand-frame ``(21, 3)`` landmarks -> wrist-origin MANO keypoints for a retargeter.

    Feed the result to ``DexRetargeter.extract_ref_value`` exactly as you would a
    MediaPipe detection's ``joint_pos``. Pass keypoints in Eden's hand frame (the default
    :meth:`~eden.utils.devices.wuji.glove.WujiGlove.keypoints` output), not the vendor's
    wrist frame.
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.shape != (N_LANDMARKS, 3):
        raise ValueError(f"expected ({N_LANDMARKS}, 3) hand-frame keypoints, got shape {kp.shape}")
    return (kp - kp[0]) @ HAND_TO_MANO


def _poses_to_arrays(
    poses: Sequence[Any], side: str, frame: str, expected: int, what: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shared decode for the pose-array streams (skeleton joints, tips, EMF coils).

    Each element must expose ``.pose.position`` / ``.pose.orientation`` / ``.confidence``.
    Returns ``(pos (N, 3), quat (N, 4) wxyz, confidence (N,))``.
    """
    check_frame(frame)
    side = check_side(side)  # checked for either frame: a typo must not wait for frame="hand"
    if len(poses) != expected:
        raise ValueError(f"expected {expected} {what} from the glove, got {len(poses)}")
    pos = np.empty((expected, 3))
    quat = np.empty((expected, 4))
    conf = np.empty(expected)
    for i, entry in enumerate(poses):
        p = np.asarray(entry.pose.position, dtype=float).reshape(-1)
        if p.shape != (3,):
            raise ValueError(f"expected a 3-element position for {what} {i}, got shape {p.shape}")
        o = entry.pose.orientation  # a wuji_sdk.Quaternion: scalar-last x/y/z/w attributes
        pos[i] = p
        quat[i] = _quat_wxyz((o.x, o.y, o.z, o.w))
        conf[i] = float(entry.confidence)
    if frame == "hand":
        pos, quat = _poses_to_hand(pos, quat, side)
    return pos, quat, conf


def skeleton_to_arrays(
    skeleton: Any, side: str, frame: str = "hand"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decode a ``HandSkeleton`` frame into ``(pos (21, 3), quat (21, 4), confidence (21,))``.

    Landmarks come back in :data:`HAND_LANDMARKS` (MediaPipe) order, positions in metres.
    ``frame="hand"`` rotates into Eden's hand basis; ``frame="wuji"`` leaves the vendor's
    wrist basis alone.
    """
    return _poses_to_arrays(skeleton.joints, side, frame, N_LANDMARKS, "skeleton joints")


def fingertips_to_arrays(
    tips: Any, side: str, frame: str = "hand"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decode a ``FingertipPoses`` frame into ``(pos (5, 3), quat (5, 4), confidence (5,))``.

    Fingers are in :data:`FINGERS` order (thumb first).
    """
    return _poses_to_arrays(tips.poses, side, frame, len(FINGERS), "fingertip poses")


def emf_poses_to_arrays(
    emf: Any, side: str, frame: str = "hand"
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decode an ``EmfPoseArray`` frame into ``(pos (5, 3), quat (5, 4), confidence (5,))``.

    These are the raw receiver-coil poses in :data:`FINGERS` order. They are reported in
    the ``emf_tx`` (back-of-hand transmitter) frame, which is a pure translation off the
    wrist frame, so the rotation applied here is the wrist one — the coil offset is *not*
    removed. Confidence below 0.9 means the vendor considers the estimate unreliable.
    """
    return _poses_to_arrays(emf.poses, side, frame, len(FINGERS), "EMF poses")


def pack_joint_angles(angles: Any) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pack a ``HandJointAngles`` frame's 5x5 padded array into the 21 real DOFs.

    Returns ``(angles (21,) in radians, confidence (5,))``. The thumb contributes all
    five of its slots, every other finger the first four (:data:`FINGER_NUM_DOFS`); the
    padding slot is dropped rather than passed on as a fake zero-angle joint.
    """
    fingers = angles.fingers
    if len(fingers) != len(FINGERS):
        raise ValueError(f"expected {len(FINGERS)} fingers of joint angles, got {len(fingers)}")
    packed = []
    conf = np.empty(len(FINGERS))
    for i, (source, finger, n_dofs) in enumerate(zip(fingers, FINGERS, FINGER_NUM_DOFS)):
        row = np.asarray(source.angles, dtype=float).reshape(-1)
        if row.shape != (JOINT_ANGLE_SLOTS,):
            raise ValueError(f"expected {JOINT_ANGLE_SLOTS} angle slots for the {finger}, got {row.shape[0]}")
        packed.append(row[:n_dofs])
        conf[i] = float(source.confidence)
    return np.concatenate(packed), conf  # FINGER_NUM_DOFS sums to N_HAND_DOFS


def _tactile_flat(data: Sequence[float]) -> npt.NDArray[np.float64]:
    """The payload of a sheet-shaped stream as a flat ``(744,)`` array, or raise."""
    flat = np.asarray(data, dtype=float)
    if flat.shape != (N_TAXELS,):
        raise ValueError(f"expected {N_TAXELS} taxels ({TACTILE_ROWS}x{TACTILE_COLS}), got shape {flat.shape}")
    return flat


def tactile_valid(data: Sequence[float]) -> npt.NDArray[np.bool_]:
    """``(24, 31)`` boolean mask of the taxels this frame actually measured."""
    return (_tactile_flat(data) != TACTILE_INVALID).reshape(TACTILE_ROWS, TACTILE_COLS)


def tactile_grid(data: Sequence[float]) -> npt.NDArray[np.float64]:
    """Reshape a flat tactile payload into the ``(24, 31)`` row-major grid, masked to ``nan``.

    Works for every sheet-shaped stream — ``tactile`` (pressure in ``[0, 1]``),
    ``tactile_binary`` (1 = contact), and ``tactile_residual`` (signed, baseline-relative).
    The cells the sheet does not populate arrive as :data:`TACTILE_INVALID` and become
    ``nan`` here, so a stray ``.mean()`` reports ``nan`` instead of a plausible-looking
    number biased toward -1.
    """
    grid = _tactile_flat(data).reshape(TACTILE_ROWS, TACTILE_COLS).copy()
    grid[grid == TACTILE_INVALID] = np.nan
    return grid


def tactile_taxels(data: Sequence[float], flat: bool = False) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """One entry per taxel this frame actually measured: ``(index, value)``.

    The node-wise view of a sheet-shaped stream. :func:`tactile_grid` keeps the full
    ``24 x 31`` layout with the dead taxels as ``nan``; this drops them, so ``value`` is
    every real reading and nothing else — iterate it, reduce it, or feed it somewhere
    without first writing the ``nan`` filter yourself.

    Parameters
    ----------
    data:
        The stream's flat 744-value payload (``TactileFrame.data`` and friends).
    flat:
        ``False`` (default) indexes as ``(N, 2)`` ``(row, col)`` into the grid;
        ``True`` as ``(N,)`` row-major ids in ``[0, 744)``.

    Returns
    -------
    tuple
        ``(index, value)``. ``index`` is an integer array — kept separate from ``value``
        rather than packed into one float array, so a grid coordinate can never be read
        as a pressure. ``N`` is whatever the sheet reports this frame, which on firmware
        0.11.1 is :data:`N_UNMASKED_TAXELS` and is deliberately *not*
        :data:`N_ACTIVE_TAXELS` — see those constants before joining against the point
        cloud, which carries the smaller count.
    """
    values = _tactile_flat(data)
    live = np.flatnonzero(values != TACTILE_INVALID)
    index = live if flat else np.stack(np.unravel_index(live, (TACTILE_ROWS, TACTILE_COLS)), axis=1)
    return index.astype(np.int64), values[live]


def tactile_zones_to_dict(zones: Any) -> dict[str, npt.NDArray[np.float64]]:
    """Decode a ``TactileZones`` frame into ``{zone: (n,) pressures}``, masked to ``nan``.

    Zones are :data:`TACTILE_ZONE_SIZES` (palm plus one per finger); the length of each
    array is fixed by the hardware, and a length mismatch raises rather than silently
    reindexing a caller's per-zone slice.
    """
    out = {}
    for zone, size in TACTILE_ZONE_SIZES.items():
        # `np.array`, not `asarray`: masking is in place, and an SDK frame that already
        # hands back a float64 array would otherwise have its sentinels overwritten —
        # the same frame `WujiGlove.state` publishes raw. `tactile_grid` copies too.
        values = np.array(getattr(zones, zone), dtype=float).reshape(-1)
        if values.shape != (size,):
            raise ValueError(f"expected {size} taxels in the {zone} zone, got {values.shape[0]}")
        values[values == TACTILE_INVALID] = np.nan
        out[zone] = values
    return out


def decode_point_cloud(cloud: Any) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Decode a ``PointCloud`` frame into ``(points (N, 3), pressure (N,))``.

    The payload is a packed byte buffer, so the field descriptors are validated against
    :data:`POINT_CLOUD_FIELDS` / :data:`POINT_CLOUD_STRIDE` first: a firmware change to
    the layout must raise here rather than reinterpret the bytes into plausible garbage.
    Points are in the glove's wrist frame — pass them through :func:`wuji_points_to_hand`
    for the Eden hand frame.
    """
    stride = int(cloud.point_stride)
    layout = [(str(f.name), int(f.offset), int(f.type)) for f in cloud.fields]
    expected = [(name, FLOAT32_BYTES * i, POINT_FIELD_FLOAT32) for i, name in enumerate(POINT_CLOUD_FIELDS)]
    if stride != POINT_CLOUD_STRIDE or layout != expected:
        raise ValueError(
            f"unexpected tactile point-cloud layout: stride={stride} fields={layout}; "
            f"this decoder handles stride={POINT_CLOUD_STRIDE} fields={expected} "
            "(x, y, z, pressure as float32)"
        )
    raw = np.frombuffer(bytes(cloud.data), dtype=np.float32)
    if raw.size % len(POINT_CLOUD_FIELDS):
        raise ValueError(
            f"point-cloud payload of {raw.size * FLOAT32_BYTES} bytes is not a whole number "
            f"of {POINT_CLOUD_STRIDE}-byte points"
        )
    values = raw.reshape(-1, len(POINT_CLOUD_FIELDS)).astype(float)
    # The header counts the points too; disagreeing with the payload means a truncated or
    # over-long frame, which would otherwise decode into a plausible-looking short cloud.
    count = int(cloud.point_count)
    if count != len(values):
        raise ValueError(f"point cloud says point_count={count} but carries {len(values)} points")
    return values[:, :3], values[:, 3]
