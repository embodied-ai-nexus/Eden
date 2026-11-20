"""The out-of-process wire format, and ``wait_for_pose``'s diagnosis of a source that never delivers."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eden.extensions.deployment.object_pose.sources import (
    PoseSample,
    PoseSource,
    encode_pose_message,
    parse_pose_message,
    wait_for_pose,
)


class _Fixed(PoseSource):
    """A source whose sample and counts the test sets (the camera boundary)."""

    name = "the fixed source"

    def __init__(self, sample: PoseSample | None = None, counts: dict[str, int] | None = None) -> None:
        self.sample, self.counts = sample, counts or {}

    def latest(self) -> PoseSample | None:
        return self.sample

    def message_counts(self) -> dict[str, int]:
        return dict(self.counts)


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


def test_wait_for_pose_returns_on_a_live_pose():
    wait_for_pose(_Fixed(PoseSample(np.zeros(3), np.array([1.0, 0, 0, 0]), time.monotonic())), 0.1, 0.5, "the cube")


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (_Fixed(), "nothing from the fixed source"),
        (_Fixed(counts={"untracked": 3}), r"handled 3 frames .* none with a pose of the cube \(3 untracked\)"),
        (
            _Fixed(PoseSample(np.zeros(3), np.array([1.0, 0, 0, 0]), time.monotonic() - 5.0)),
            "tracked the cube and lost it",
        ),
    ],
)
def test_wait_for_pose_tells_a_dead_source_from_a_lost_object(source, match):
    with pytest.raises(RuntimeError, match=match):
        wait_for_pose(source, 0.05, 0.5, "the cube")
