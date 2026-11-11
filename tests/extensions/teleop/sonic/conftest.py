"""Fixtures and helpers for the SONIC operator tests."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from eden.extensions.policies.sonic.wire import G1DebugPublisher
from eden.extensions.teleop.sonic.planner_streamer import PlannerStreamer
from eden.extensions.teleop.sonic.state_machine import (
    ButtonSnapshot,
    OperatorStateMachine,
    StreamMode,
)
from eden.extensions.teleop.sonic.three_point import ThreePointPose
from tests.support.sonic import bind_debug_publisher, make_body_sample, unpaced


def press(sm: OperatorStateMachine, **buttons) -> object:
    """One tick with the given buttons pressed, then one with all released."""
    t = sm.step(ButtonSnapshot(**buttons))
    sm.step(ButtonSnapshot())
    return t


def engage(sm: OperatorStateMachine) -> None:
    t = press(sm, a=True, b=True, x=True, y=True)
    assert t.mode == StreamMode.PLANNER


def run_planner_streamer(streamer, mode, controllers, ts):
    streamer.run_once(mode, make_body_sample(ts), controllers, timestamp_ns=ts)


def publish_until(publisher: G1DebugPublisher, payload: dict, observed: Callable[[], bool], timeout: float = 5.0):
    """Publish ``payload`` on the robot's g1_debug channel until ``observed()`` holds.

    One publish is not a delivery: PUB drops what it sends before a subscriber has
    joined, and the join is asynchronous. ``observed`` is checked after every send.
    """
    deadline = time.monotonic() + timeout
    while True:
        publisher.publish(payload)
        time.sleep(0.005)
        if observed():
            return
        if time.monotonic() > deadline:
            pytest.fail(f"g1_debug payload {payload!r} was never observed within {timeout} s")


@pytest.fixture
def debug_link():
    """A loopback g1_debug channel: (robot-side ``G1DebugPublisher``, operator-side ``FeedbackReader``).

    Nothing is published until the test does, so the reader deterministically sees no
    feedback until then. Skips without the ``sonic`` extra (pyzmq + msgpack).
    """
    from eden.extensions.teleop.sonic.feedback import FeedbackReader

    publisher, port = bind_debug_publisher()
    reader = FeedbackReader(host="127.0.0.1", port=port)
    yield publisher, reader
    reader.close()
    publisher.close()


@pytest.fixture
def three_point(g1_key_frame_fk) -> ThreePointPose:
    """A fresh, uncalibrated ``ThreePointPose`` over the real G1 key-frame FK."""
    return ThreePointPose(g1_key_frame_fk)


def build_planner_streamer(sent: list, three_point: ThreePointPose, feedback) -> PlannerStreamer:
    """A ``PlannerStreamer`` that appends every packed message to ``sent`` and never sleeps."""
    return unpaced(PlannerStreamer(send=sent.append, three_point=three_point, feedback=feedback))
