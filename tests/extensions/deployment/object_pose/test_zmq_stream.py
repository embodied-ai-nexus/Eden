"""The wire format between ``serve_pose_stream`` and ``ZmqPoseSource``."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eden.extensions.deployment.object_pose.zmq_stream import encode_pose_message, parse_pose_message


def test_a_published_pose_parses_back_with_its_capture_latency():
    stamp = time.monotonic() - 0.05
    msg = encode_pose_message("cube", (np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, 0.0, 0.0]), 0.9), stamp)
    pos, quat, latency, score = parse_pose_message(msg, "cube")
    np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])  # normalised
    assert 0.05 <= latency < 1.0
    assert score == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("text", "outcome"),
    [
        (encode_pose_message("pen", None, 0.0), "other"),
        (encode_pose_message("cube", None, 0.0), "untracked"),
        ('{"key": "cube", "pos": [0, 0, "x"], "quat_wxyz": [1, 0, 0, 0]}', "rejected"),
        ('{"key": "cube", "pos": [0, 0, 0], "quat_wxyz": [0, 0, 0, 0]}', "rejected"),
        ("{", "rejected"),
    ],
)
def test_a_message_without_a_usable_pose_says_why(text, outcome):
    assert parse_pose_message(text, "cube") == outcome
