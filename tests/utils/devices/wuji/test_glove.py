"""``WujiGlove`` over a fake ``wuji_sdk``: device selection, subscription wiring, and the getters. No glove needed.

The fake is installed as ``sys.modules["wuji_sdk"]``, where the device's own import finds it.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eden.utils.devices.wuji import glove as G
from eden.utils.devices.wuji import protocol as P
from eden.utils.devices.wuji.glove import WujiGlove


class FakeWujiException(Exception):
    """``wuji_sdk.WujiException``."""


class FakeSubscription:
    """Stands in for both a CallbackSubscription and a rate-controlling Subscription."""

    def __init__(self, owner, stream, callback=None):
        self.owner = owner
        self.stream = stream
        self.callback = callback
        self.closed = False
        self.rate = None

    def set_rate(self, hz):
        if self.owner.no_rate_control:
            raise FakeWujiException(f"device {self.owner.serial_number!r} runs firmware without rate control")
        if self.stream not in G.RATE_SOURCES:
            raise RuntimeError(f"{self.stream} is a derived stream")
        self.rate = hz
        return hz

    def close(self):
        self.closed = True


class FakeResource:
    def __init__(self, owner, stream):
        self._owner = owner
        self._stream = stream

    def subscribe_with_callback(self, callback, on_error=None):
        if self._stream in self._owner.failing_streams:
            raise RuntimeError("device busy")
        sub = FakeSubscription(self._owner, self._stream, callback)
        self._owner.callback_subs[self._stream] = sub
        return sub

    def subscribe(self):
        sub = FakeSubscription(self._owner, self._stream)
        self._owner.rate_subs[self._stream] = sub
        return sub


class FakeGlove:
    def __init__(self, sn, side, failing_streams=(), no_rate_control=False):
        self.serial_number = sn
        self._side = side
        self.failing_streams = set(failing_streams)
        self.no_rate_control = no_rate_control
        self.callback_subs: dict[str, FakeSubscription] = {}
        self.rate_subs: dict[str, FakeSubscription] = {}
        self.model_path = None

    def hand_side(self):
        return SimpleNamespace(get=lambda: self._side)

    def version(self):
        return SimpleNamespace(get=lambda: "1.2.3")

    def hand_model_path(self):
        def _set(value):
            self.model_path = value

        return SimpleNamespace(get=lambda: self.model_path, set=_set)

    def __getattr__(self, name):
        if name in G.STREAMS:
            return lambda: FakeResource(self, name)
        raise AttributeError(name)

    def push(self, stream, frame):
        """Deliver a frame the way the SDK's background thread would."""
        self.callback_subs[stream].callback(frame)


class FakeManager:
    def __init__(self, gloves, other_devices=()):
        self._gloves = list(gloves)
        self._others = list(other_devices)
        self.connected: dict[str, FakeGlove] = {}
        self.disconnects: list[str] = []

    def scan(self):
        found = [SimpleNamespace(sn=g.serial_number, address="tcp://x", device_type="WujiGlove") for g in self._gloves]
        return found + list(self._others)

    def connect(self, *, sn, device_name, options=None):
        for glove in self._gloves:
            if glove.serial_number == sn:
                self.connected[device_name] = glove
                return glove
        raise RuntimeError(f"no such device {sn}")

    def disconnect(self, device_name):
        self.disconnects.append(device_name)
        self.connected.pop(device_name, None)


def fake_sdk(gloves, other_devices=()):
    manager = FakeManager(gloves, other_devices)
    return SimpleNamespace(
        SdkManager=SimpleNamespace(instance=lambda: manager),
        DeviceType=SimpleNamespace(WujiGlove="WujiGlove"),
        WujiException=FakeWujiException,
        ConnectOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        _manager=manager,
    )


@pytest.fixture
def make_glove(monkeypatch):
    """Build a WujiGlove over a fake SDK; returns ``(device, fake_glove, sdk)``."""
    opened: list[WujiGlove] = []

    def make(gloves=None, other_devices=(), **kwargs):
        gloves = list(gloves) if gloves is not None else [FakeGlove("WG1KA0000000R", "right")]
        sdk = fake_sdk(gloves, other_devices)
        monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)
        device = WujiGlove(**kwargs)
        opened.append(device)
        return device, sdk._manager.connected[device._device_name], sdk

    yield make
    for device in opened:
        device.close()


def test_connects_to_the_only_glove_and_reads_its_side(make_glove):
    device, _, _ = make_glove()
    assert device.serial_number == "WG1KA0000000R"
    assert device.side == "right"
    assert device.streams == G.DEFAULT_STREAMS


def test_subscribes_exactly_the_requested_streams(make_glove):
    device, glove, _ = make_glove(streams=("hand_skeleton", "tactile_zones"))
    assert set(glove.callback_subs) == {"hand_skeleton", "tactile_zones"}


def test_unknown_stream_names_the_valid_ones(make_glove):
    with pytest.raises(ValueError, match="unknown glove stream"):
        make_glove(streams=("hand_skeleton", "tactile_raw"))


def test_no_glove_on_the_bus_raises_and_names_what_was_seen(make_glove):
    other = SimpleNamespace(sn="SN-H", address="usb:x", device_type="WujiHand2")
    with pytest.raises(RuntimeError, match="no Wuji Glove found.*SN-H"):
        make_glove(gloves=[], other_devices=[other])


def test_two_gloves_without_a_side_is_ambiguous(make_glove):
    """The message has to be copy-pasteable: both serials, each with its likely side."""
    with pytest.raises(RuntimeError, match="pass sn= or side=") as exc:
        make_glove(gloves=[FakeGlove("WG1JA0000000L", "left"), FakeGlove("WG1KA0000000R", "right")])
    assert "WG1JA0000000L (likely left)" in str(exc.value)
    assert "WG1KA0000000R (likely right)" in str(exc.value)


def test_two_gloves_of_unrecognised_serial_still_report_ambiguity(make_glove):
    """The serial letter is a hint; an unfamiliar one must not swallow the error."""
    with pytest.raises(RuntimeError, match="likely unknown side"):
        make_glove(gloves=[FakeGlove("ZZ-1", "left"), FakeGlove("ZZ-2", "right")])


def test_side_probes_the_serial_that_matches_first(make_glove):
    """sn[3] encodes the side, so the likely glove is tried first — one connect, not two."""
    device, _, sdk = make_glove(
        gloves=[FakeGlove("WG1KA0000000R", "right"), FakeGlove("WG1JA0000000L", "left")], side="left"
    )
    assert device.serial_number == "WG1JA0000000L"
    assert sdk._manager.disconnects == [], "the right glove should never have been opened"


def test_side_falls_back_to_probing_when_the_serial_lies(make_glove):
    """`hand_side` decides, always. A wrong hint costs one extra connect, not correctness."""
    device, _, sdk = make_glove(
        gloves=[FakeGlove("WG1KA0000000X", "left"), FakeGlove("WG1JA0000000Y", "right")], side="right"
    )
    assert device.serial_number == "WG1JA0000000Y"
    assert sdk._manager.disconnects == [device._device_name]  # the mis-labelled one, released


@pytest.mark.parametrize(
    "sn, expected",
    [("WG1JA06260803044", "left"), ("WG1KA06260801526", "right"), ("WG1ZA0000000000", None), ("WG1", None)],
)
def test_side_from_sn_reads_the_firmware_letter(sn, expected):
    """The two real serials are the lab gloves, whose hand_side was read over a connection."""
    assert P.side_from_sn(sn) == expected


def test_requested_side_is_checked_against_the_hardware(make_glove):
    with pytest.raises(RuntimeError, match="reports hand_side 'right', but 'left' was requested"):
        make_glove(side="left")


def test_rate_control_retunes_the_source_streams_not_the_derived_ones(make_glove):
    device, glove, _ = make_glove(streams=("hand_skeleton", "tactile"), rate_hz=30)
    assert {s: sub.rate for s, sub in glove.rate_subs.items()} == {"emf_poses": 30, "tactile": 30}
    assert "hand_skeleton" not in glove.rate_subs  # derived: set_rate would raise
    assert len(device._rate_handles) == 2  # held open, or the device resets the rate


def test_rate_control_reaches_a_derived_only_subscription(make_glove):
    """Subscribing to just tactile_binary must still retune `tactile`, not silently no-op."""
    _, glove, _ = make_glove(streams=("tactile_binary",), rate_hz=30)
    assert {s: sub.rate for s, sub in glove.rate_subs.items()} == {"tactile": 30}


def test_firmware_without_rate_control_warns_and_streams_anyway(make_glove, eden_caplog, frames):
    """Firmware 0.11.1 has no rate control. That is a missing optimisation, not a dead glove.

    Refusing to stream because the device cannot stream *more slowly* would be the wrong
    trade, so the device warns (naming the firmware's own message) and runs at the default.
    """
    device, glove, _ = make_glove(
        gloves=[FakeGlove("WG1KA0000000R", "right", no_rate_control=True)],
        streams=("hand_skeleton", "tactile"),
        rate_hz=30,
    )
    assert "without rate control" in eden_caplog.text
    assert device._rate_handles == []
    assert all(sub.closed for sub in glove.rate_subs.values()), "the probe handle must not leak"
    # ... and the data path is untouched: frames still arrive and decode.
    glove.push("hand_skeleton", frames.skeleton())
    assert device.keypoints().shape == (21, 3)


def test_every_stream_declares_the_source_it_rides_on():
    assert set(G.STREAM_RATE_SOURCE) == set(G.STREAMS)
    assert set(G.STREAM_RATE_SOURCE.values()) <= set(G.RATE_SOURCES)


def test_getters_report_none_until_the_first_frame(make_glove, frames):
    device, glove, _ = make_glove(streams=("hand_skeleton", "tactile"))
    assert device.keypoints() is None
    assert device.tactile() is None
    assert device.tracking is False
    # DeviceBase's tensor surface stays shaped so a caller can stack readings
    assert device.get_command().shape == (21, 3)
    assert torch.count_nonzero(device.get_raw_data()) == 0

    glove.push("hand_skeleton", frames.skeleton())
    glove.push("tactile", frames.tactile())
    assert device.tracking is True
    assert device.keypoints().shape == (21, 3)
    assert device.tactile().shape == (24, 31)


def test_tactile_taxels_getter_matches_the_grid_it_came_from(make_glove, frames):
    device, glove, _ = make_glove(streams=("tactile",))
    assert device.tactile_taxels() is None  # before the first frame
    glove.push("tactile", T := frames.tactile(fill=0.6, invalid=(0, 1)))
    index, value = device.tactile_taxels()
    grid = device.tactile()
    assert index.shape == (P.N_TAXELS - 2, 2) and value.shape == (P.N_TAXELS - 2,)
    assert value == pytest.approx(grid[index[:, 0], index[:, 1]])
    assert torch.equal(device.tactile_taxels(flat=True)[0], index[:, 0] * P.TACTILE_COLS + index[:, 1])
    assert len(T.data) == P.N_TAXELS


def test_tactile_taxels_rejects_a_stream_that_is_not_a_sheet(make_glove):
    """`tactile_zones` and the point cloud are not 24x31 sheets; say so, don't decode garbage."""
    device, _, _ = make_glove(streams=("tactile", "tactile_zones"))
    with pytest.raises(ValueError, match="not a tactile sheet"):
        device.tactile_taxels("tactile_zones")


def test_reading_an_unsubscribed_stream_says_how_to_get_it(make_glove):
    device, _, _ = make_glove(streams=("hand_skeleton",))
    with pytest.raises(ValueError, match="not subscribed.*streams=\\(\\.\\.\\., 'tactile'\\)"):
        device.tactile()


def test_raw_and_command_differ_by_the_basis_change(make_glove, frames):
    device, glove, _ = make_glove(streams=("hand_skeleton",))
    glove.push("hand_skeleton", frames.skeleton())
    raw = device.get_raw_data().numpy()
    command = device.get_command().numpy()
    assert command == pytest.approx(P.wuji_points_to_hand(raw, "right"), abs=1e-6)


def test_user_callbacks_fire_on_the_stream_they_registered_for(make_glove, frames):
    device, glove, _ = make_glove(streams=("hand_skeleton", "tactile"))
    seen = []
    device.add_callback("tactile", lambda frame: seen.append(frame))
    glove.push("hand_skeleton", frames.skeleton())
    assert seen == []
    frame = frames.tactile()
    glove.push("tactile", frame)
    assert seen == [frame]


def test_reset_drops_buffered_frames(make_glove, frames):
    device, glove, _ = make_glove(streams=("hand_skeleton",))
    glove.push("hand_skeleton", frames.skeleton())
    assert device.keypoints() is not None
    device.reset()
    assert device.keypoints() is None


def test_point_cloud_getter_rotates_into_the_hand_frame(make_glove, frames):
    device, glove, _ = make_glove(streams=("tactile_point_cloud",))
    glove.push("tactile_point_cloud", frames.point_cloud())
    points, pressure = device.tactile_points()
    assert points[0] == pytest.approx([-0.2, 0.0, -0.1], abs=1e-6)
    assert pressure == pytest.approx([0.75])
    raw, _ = device.tactile_points(frame="wuji")
    assert raw[0] == pytest.approx([0.0, 0.1, 0.2], abs=1e-6)


def test_state_is_json_safe_and_keeps_the_tactile_sentinel(make_glove, frames):
    import json

    device, glove, _ = make_glove(streams=("hand_skeleton", "hand_joint_angles", "tactile", "tactile_zones"))
    glove.push("hand_skeleton", frames.skeleton())
    glove.push("hand_joint_angles", frames.joint_angles())
    glove.push("tactile", frames.tactile())
    glove.push("tactile_zones", frames.zones())

    state = device.state()
    assert state["device"] == "wuji_glove" and state["sn"] == "WG1KA0000000R" and state["side"] == "right"
    assert np.shape(state["keypoints"]) == (21, 3)
    assert len(state["joint_angles"]) == 21
    # nan is not valid JSON, so the sheets go out raw with the vendor's -1.0 sentinel
    assert state["tactile"][:3] == [P.TACTILE_INVALID] * 3
    assert len(state["tactile_zones"]["palm"]) == P.TACTILE_ZONE_SIZES["palm"]
    assert json.loads(json.dumps(state))["side"] == "right"


def test_state_publishes_every_key_it_declares(make_glove, frames):
    """`STATE_KEYS` is the snapshot's wire format; it must not drift from the loop that fills it."""
    poses = SimpleNamespace(header=None, poses=[frames.pose((0.0, 0.0, float(i))) for i in range(5)])
    device, glove, _ = make_glove(streams=tuple(G.STATE_KEYS))  # also pins them as real stream names
    for stream, frame in (
        ("hand_skeleton", frames.skeleton()),
        ("tip_poses", poses),
        ("emf_poses", poses),
        ("hand_joint_angles", frames.joint_angles()),
        ("tactile_point_cloud", frames.point_cloud()),
    ):
        glove.push(stream, frame)

    state = device.state()
    for stream, keys in G.STATE_KEYS.items():
        assert all(key in state for key in keys), f"{stream} published none of {keys}"


def test_state_omits_streams_that_were_not_subscribed(make_glove, frames):
    device, glove, _ = make_glove(streams=("hand_skeleton",))
    glove.push("hand_skeleton", frames.skeleton())
    state = device.state()
    assert "tactile" not in state and "joint_angles" not in state


def test_state_works_on_a_tactile_only_device(make_glove, frames):
    """A pose-less device must still snapshot, not trip the unsubscribed-stream guard."""
    device, glove, _ = make_glove(streams=("tactile",))
    glove.push("tactile", frames.tactile())
    state = device.state()
    assert "keypoints" not in state and state["ready"] is False
    assert len(state["tactile"]) == P.N_TAXELS


def test_close_releases_every_handle_and_is_idempotent(make_glove):
    device, glove, sdk = make_glove(streams=("hand_skeleton", "tactile"), rate_hz=60)
    name = device._device_name
    device.close()
    device.close()
    assert all(sub.closed for sub in (*glove.callback_subs.values(), *glove.rate_subs.values()))
    assert sdk._manager.disconnects == [name]  # once, despite the second close


def test_two_gloves_on_one_bus_do_not_share_a_connection_handle(monkeypatch, frames):
    """`device_name` is the SDK's connection key, so bimanual use must not share one.

    Built against a *single* manager on purpose: the `make_glove` fixture hands each device
    its own, which cannot show this collision at all. With a shared default both connects
    register under one name, so only one link survives and either close() ends it.
    """
    left_hw, right_hw = FakeGlove("WG1JA0000000L", "left"), FakeGlove("WG1KA0000000R", "right")
    sdk = fake_sdk([left_hw, right_hw])
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)

    left = WujiGlove(side="left", streams=("hand_skeleton",))
    right = WujiGlove(side="right", streams=("hand_skeleton",))
    try:
        assert left._device_name != right._device_name
        assert len(sdk._manager.connected) == 2, "one handle per glove, so both links are live"

        left.close()
        assert sdk._manager.disconnects == [left._device_name]
        assert set(sdk._manager.connected) == {right._device_name}, "the right glove stays connected"
        right_hw.push("hand_skeleton", frames.skeleton())
        assert right.keypoints().shape == (21, 3), "and still reads frames"
    finally:
        right.close()


def test_a_glove_that_drops_mid_probe_does_not_leak_its_connection(monkeypatch):
    """`_connect` can raise *after* opening a link; close() still has to release it."""

    class DropsOnHandSide(FakeGlove):
        def hand_side(self):
            raise RuntimeError("link dropped mid-probe")

    sdk = fake_sdk([DropsOnHandSide("WG1JA0000000L", "left"), FakeGlove("WG1KA0000000R", "right")])
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)
    try:
        WujiGlove(side="left", streams=("hand_skeleton",))
    except RuntimeError as exc:
        assert "link dropped mid-probe" in str(exc)
    else:
        raise AssertionError("expected the probe to propagate the SDK error")

    # try/except, not pytest.raises: the latter holds the traceback and so the half-built
    # device alive, and the release happens in __del__ -> close().
    assert len(sdk._manager.disconnects) == 1, "the connection opened before the failure is released"
    assert sdk._manager.connected == {}


def test_an_explicit_sn_survives_a_scan_that_cannot_enumerate(monkeypatch):
    """The sn= path is the escape hatch when enumeration is broken; its type check is advisory."""

    def exploding_scan():
        raise RuntimeError("bus enumeration unavailable")

    sdk = fake_sdk([FakeGlove("WG1KA0000000R", "right")])
    monkeypatch.setattr(sdk._manager, "scan", exploding_scan)
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)

    device = WujiGlove(sn="WG1KA0000000R", streams=("hand_skeleton",))
    try:
        assert device.side == "right"
    finally:
        device.close()


def test_a_failing_connect_does_not_disconnect_or_warn(monkeypatch, eden_caplog):
    """The no-glove error is the whole diagnostic; a teardown disconnect only adds a warning.

    Plain try/except rather than ``pytest.raises``: the latter holds the traceback, and so the
    half-built device, alive — ``__del__`` -> ``close()`` would not have run by the assertions.
    """
    sdk = fake_sdk([])
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)
    try:
        WujiGlove(streams=("hand_skeleton",))
    except RuntimeError as exc:
        assert "no Wuji Glove found" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError")

    assert sdk._manager.disconnects == [], "there was never a connection to release"
    assert "failed to disconnect" not in eden_caplog.text


def test_an_explicit_sn_still_has_to_be_a_glove(make_glove):
    """Without the check this connects, then dies on `hand_side()` with an AttributeError."""
    hand = SimpleNamespace(sn="SN-H", address="usb:x", device_type="WujiHand2")
    with pytest.raises(RuntimeError, match="SN-H is a WujiHand2, not a Wuji Glove"):
        make_glove(other_devices=[hand], sn="SN-H")


def test_a_failed_subscription_does_not_leak_the_connection(monkeypatch):
    """A stream that fails to subscribe must tear the device down, not leave it half-live."""
    glove = FakeGlove("SN-R", "right", failing_streams={"tactile"})
    sdk = fake_sdk([glove])
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)

    with pytest.raises(RuntimeError, match="device busy"):
        WujiGlove(streams=("hand_skeleton", "tactile"))
    assert glove.callback_subs["hand_skeleton"].closed  # the one that did open is released
    assert [n.startswith(G.DEVICE_NAME_PREFIX) for n in sdk._manager.disconnects] == [True]
