"""Fixtures for the Wuji Glove tests: builders for the SDK's stream messages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eden.utils.devices.wuji import protocol as P


def _pose(position, quat_xyzw=(0.0, 0.0, 0.0, 1.0), confidence=1.0):
    x, y, z, w = quat_xyzw
    return SimpleNamespace(
        pose=SimpleNamespace(position=list(position), orientation=SimpleNamespace(x=x, y=y, z=z, w=w)),
        confidence=confidence,
    )


def _skeleton(n=P.N_LANDMARKS):
    joints = [_pose((0.001 * i, 0.002 * i, -0.003 * i), confidence=i / 100.0) for i in range(n)]
    for joint, name in zip(joints, P.HAND_LANDMARKS):
        joint.name = name
    return SimpleNamespace(header=SimpleNamespace(seq=1, timestamp_us=0, frame_id="r_wrist"), joints=joints)


def _joint_angles(confidences=(1.0, 0.9, 0.8, 0.7, 0.6)):
    fingers = [
        SimpleNamespace(angles=[0.1 * (5 * f + s) for s in range(P.JOINT_ANGLE_SLOTS)], confidence=confidences[f])
        for f in range(len(P.FINGERS))
    ]
    # Every finger but the thumb pads its fifth slot with a literal 0.
    for finger in fingers[1:]:
        finger.angles[4] = 0.0
    return SimpleNamespace(header=SimpleNamespace(seq=1), fingers=fingers)


def _tactile(fill=0.25, invalid=(0, 1, 2)):
    data = [fill] * P.N_TAXELS
    for i in invalid:
        data[i] = P.TACTILE_INVALID
    return SimpleNamespace(header=SimpleNamespace(seq=1), data=data)


def _zones():
    zones = SimpleNamespace(header=SimpleNamespace(seq=1))
    for zone, size in P.TACTILE_ZONE_SIZES.items():
        values = [0.5] * size
        values[-1] = P.TACTILE_INVALID
        setattr(zones, zone, values)
    return zones


def _point_cloud(points=((0.0, 0.1, 0.2),), pressure=(0.75,), stride=P.POINT_CLOUD_STRIDE, fields=None):
    if fields is None:
        fields = [
            SimpleNamespace(name=name, offset=4 * i, type=P.POINT_FIELD_FLOAT32)
            for i, name in enumerate(P.POINT_CLOUD_FIELDS)
        ]
    packed = np.array([[*p, q] for p, q in zip(points, pressure)], dtype=np.float32).tobytes()
    return SimpleNamespace(
        header=SimpleNamespace(seq=1),
        frame_id="r_wrist",
        point_count=len(points),
        point_stride=stride,
        fields=fields,
        data=list(packed),
    )


@pytest.fixture(scope="session")
def frames():
    """Builders for the SDK's stream messages: ``frames.skeleton()``, ``frames.tactile(fill=...)``, ..."""
    return SimpleNamespace(
        pose=_pose,
        skeleton=_skeleton,
        joint_angles=_joint_angles,
        tactile=_tactile,
        zones=_zones,
        point_cloud=_point_cloud,
    )
