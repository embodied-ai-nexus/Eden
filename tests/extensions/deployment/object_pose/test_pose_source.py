"""``wait_for_pose``'s diagnosis of a source that never delivers."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eden.extensions.deployment.object_pose.pose_source import PoseSample, PoseSource, wait_for_pose


class _Fixed(PoseSource):
    """A source whose sample and counts the test sets (the camera boundary)."""

    name = "the fixed source"

    def __init__(self, sample: PoseSample | None = None, counts: dict[str, int] | None = None) -> None:
        self.sample, self.counts = sample, counts or {}

    def latest(self) -> PoseSample | None:
        return self.sample

    def message_counts(self) -> dict[str, int]:
        return dict(self.counts)


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
