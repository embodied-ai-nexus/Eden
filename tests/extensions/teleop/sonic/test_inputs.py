"""``eden.extensions.teleop.sonic.inputs``: ``BridgeReader`` over a loopback ``eden pico bridge`` stream."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from eden.extensions.teleop.sonic.inputs import BridgeReader
from tests.support.sonic import IDENTITY_BODY, bridge_state

# BridgeReader subscribes over pyzmq (the `sonic` extra).
zmq = pytest.importorskip("zmq")


@pytest.fixture
def bridge(request):
    """A (publish, reader) pair over loopback; publish(state) blocks until observed."""
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    reader = BridgeReader(host="127.0.0.1", port=port)
    reader.start()

    def publish(state: dict, observed=None, timeout: float = 5.0) -> None:
        if observed is None:

            def observed():
                return reader.get_timestamp_ns() == state.get("timestamp_ns")

        deadline = time.monotonic() + timeout
        while not observed():
            if time.monotonic() > deadline:
                pytest.fail(f"reader never observed the published state {state.get('timestamp_ns')}")
            pub.send_string(json.dumps(state))
            time.sleep(0.005)

    yield publish, reader
    reader.stop()
    pub.close(linger=0)


def test_body_round_trip_is_bit_exact(bridge):
    publish, reader = bridge
    rng = np.random.default_rng(0)
    body = rng.standard_normal((24, 7))
    publish(bridge_state(1, body_stamp_ns=100, body=body.tolist()))
    sample = reader.get_latest()
    np.testing.assert_array_equal(sample.body_poses, body)  # JSON round-trips doubles exactly
    assert sample.body_poses.dtype == np.float64
    assert sample.timestamp_ns == 100


def test_dedupes_on_the_body_timestamp_and_tracks_rate(bridge):
    publish, reader = bridge
    publish(bridge_state(1, body_stamp_ns=1_000_000_000))
    first = reader.get_latest()
    assert first.dt == 0.0 and first.fps == 0.0  # first frame, same as PicoReader

    # A publish tick between device frames: new state, same body stamp -> no new sample.
    publish(bridge_state(2, body_stamp_ns=1_000_000_000))
    assert reader.get_latest() is first

    publish(bridge_state(3, body_stamp_ns=1_011_000_000))
    second = reader.get_latest()
    assert second is not first
    assert second.dt == pytest.approx(0.011)
    assert second.fps == pytest.approx(1.0 / 0.011)  # EMA seeds with the first rate

    publish(bridge_state(4, body_stamp_ns=1_031_000_000))
    third = reader.get_latest()
    assert third.dt == pytest.approx(0.020)
    assert third.fps == pytest.approx(0.9 * (1.0 / 0.011) + 0.1 * (1.0 / 0.020))


def test_controller_mapping_crosses_sides_correctly(bridge):
    publish, reader = bridge
    publish(
        bridge_state(
            1,
            left_a_button=True,  # left primary = X
            right_b_button=True,  # right secondary = B
            left_squeeze_value=0.7,
            right_trigger_value=0.4,
            left_menu_button=True,
            right_thumbstick=True,
            right_thumbstick_value=[0.1, -0.2],
        )
    )
    c = reader.controllers()
    assert c.x is True and c.a is False and c.y is False
    assert c.b is True
    assert c.left_grip == pytest.approx(0.7)
    assert c.right_trigger == pytest.approx(0.4)
    assert c.left_menu is True and c.right_menu is False
    assert c.right_axis_click is True and c.left_axis_click is False
    assert c.right_axis == pytest.approx((0.1, -0.2))
    assert c.left_axis == (0.0, 0.0)


def test_neutral_before_the_first_message():
    reader = BridgeReader(host="127.0.0.1", port=1)  # nothing bound; never receives
    c = reader.controllers()
    assert not any([c.a, c.b, c.x, c.y, c.left_menu, c.right_axis_click])
    assert c.left_trigger == 0.0 and c.right_grip == 0.0
    assert reader.get_timestamp_ns() == 0
    assert reader.get_latest() is None
    reader.stop()


def test_message_without_body_updates_controllers_only(bridge):
    publish, reader = bridge
    publish(bridge_state(1, right_a_button=True))
    assert reader.controllers().a is True
    assert reader.get_latest() is None


def test_malformed_body_is_skipped(bridge):
    publish, reader = bridge
    publish(bridge_state(1, body_stamp_ns=100, body=[[0.0] * 7] * 10))  # wrong shape
    assert reader.get_latest() is None
    publish(bridge_state(2, body_stamp_ns=200))
    assert reader.get_latest().timestamp_ns == 200


@pytest.mark.parametrize("bad", ["nan", "zero_quat"])
def test_unusable_body_frame_is_dropped_not_published(bridge, bad):
    """One NaN / zero-norm quaternion used to raise in the pipelines' sRot.from_quat and end the operator."""
    publish, reader = bridge
    body = IDENTITY_BODY.copy()
    if bad == "nan":
        body[5, 4] = np.nan
    else:
        body[5, 3:] = 0.0
    publish(bridge_state(1, body_stamp_ns=100, body=body.tolist()))  # json carries NaN as a bare literal
    assert reader.get_latest() is None
    publish(bridge_state(2, body_stamp_ns=200))
    assert reader.get_latest().timestamp_ns == 200


def test_a_stream_of_bad_frames_still_flags_disconnect(bridge):
    """Dropped frames are not fresh data; the drop path never ran the stale check."""
    publish, reader = bridge
    reader.STALE_TIMEOUT = 0.2
    bad = IDENTITY_BODY.copy()
    bad[:, 3:] = 0.0  # zero-norm quaternions, e.g. tracking lost
    deadline = time.monotonic() + 5.0
    stamp = 100
    while not reader.disconnected:
        assert time.monotonic() < deadline, "a bad-frame stream never flagged disconnect"
        stamp += 1
        publish(bridge_state(stamp, body_stamp_ns=stamp, body=bad.tolist()))  # a new stamp every time
    assert reader.get_latest() is None


def test_staleness_flags_and_recovers(bridge):
    publish, reader = bridge
    reader.STALE_TIMEOUT = 0.2
    publish(bridge_state(1, body_stamp_ns=100))
    assert not reader.disconnected
    deadline = time.monotonic() + 5.0
    while not reader.disconnected:
        assert time.monotonic() < deadline, "staleness never flagged"
        time.sleep(0.02)
    publish(bridge_state(2, body_stamp_ns=200))
    assert not reader.disconnected
