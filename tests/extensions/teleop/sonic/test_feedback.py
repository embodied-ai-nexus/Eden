"""``eden.extensions.teleop.sonic.feedback``: the robot-reset watcher on a loopback g1_debug channel."""

from __future__ import annotations

import time

import pytest

from eden.extensions.policies.sonic.wire import G1DebugPoller
from eden.extensions.teleop.sonic.feedback import RobotResetWatcher
from tests.extensions.teleop.sonic.conftest import publish_until
from tests.support.sonic import bind_debug_publisher


@pytest.fixture
def channel():
    """(robot publisher, watcher, deliver): ``deliver(payload)`` leaves ``payload`` pending at the watcher.

    The watcher's own poll is the unit under test, so delivery is confirmed on a probe
    subscriber connected *after* it: once the probe receives the payload, the watcher's
    earlier subscription has joined and the same message is queued (conflated) for it.
    """
    publisher, port = bind_debug_publisher()
    watcher = RobotResetWatcher(host="127.0.0.1", port=port)
    probe = G1DebugPoller(host="127.0.0.1", port=port)

    def deliver(payload: dict) -> None:
        publish_until(publisher, payload, lambda: probe.get_latest() == payload)
        publisher.publish(payload)  # once more, after the probe saw it: the watcher's copy is in flight too
        time.sleep(0.02)

    yield publisher, watcher, deliver
    probe.close()
    watcher.close()
    publisher.close()


def test_watcher_baselines_on_first_observation(channel):
    publisher, watcher, deliver = channel
    assert watcher.poll() is False  # nothing published yet
    deliver({"reset_count": 3})  # operator started after the robot already reset
    assert watcher.poll() is False  # baseline, not a new reset
    # The 3 really was observed as the baseline: the next increment fires.
    publish_until(publisher, {"reset_count": 4}, watcher.poll)


def test_watcher_fires_on_increment_only(channel):
    publisher, watcher, deliver = channel
    deliver({"reset_count": 0})
    assert watcher.poll() is False  # baseline
    publish_until(publisher, {"reset_count": 1}, watcher.poll)  # the increment fires
    deliver({"reset_count": 1})
    assert watcher.poll() is False  # repeat value: fires exactly once per increment
    publish_until(publisher, {"reset_count": 3}, watcher.poll)  # a jump fires too


def test_watcher_tolerates_messages_without_the_key(channel):
    publisher, watcher, deliver = channel
    deliver({"reset_count": 1})
    assert watcher.poll() is False  # baseline
    deliver({"body_q_measured": [0.0] * 29})  # e.g. the reference robot's dict
    assert watcher.poll() is False
    publish_until(publisher, {"reset_count": 2}, watcher.poll)  # the baseline survived the keyless message
