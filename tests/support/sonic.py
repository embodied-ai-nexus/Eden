"""Helpers shared by the SONIC tests in tests/extensions/policies/sonic and tests/extensions/teleop/sonic.

Controller/body samples, the PICO bridge's JSON state, the real SMPL-X FK (skipping when its
asset is unavailable) and a loopback robot ``g1_debug`` publisher.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

import numpy as np
import pytest

from eden.extensions.policies.sonic.wire import G1DebugPublisher
from eden.extensions.teleop.sonic.inputs import BodySample, ControllerSample
from eden.extensions.teleop.sonic.smpl import SmplFk

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def make_controllers(**overrides):
    defaults = {
        "a": False, "b": False, "x": False, "y": False, "left_axis_click": False, "right_axis_click": False,
        "left_menu": False, "right_menu": False, "left_trigger": 0.0, "right_trigger": 0.0,
        "left_grip": 0.0, "right_grip": 0.0, "left_axis": (0.0, 0.0), "right_axis": (0.0, 0.0),
    }  # fmt: skip
    defaults.update(overrides)
    return ControllerSample(**defaults)


def make_body_sample(ts_ns, dt=0.011):
    body = np.zeros((24, 7))
    body[:, 6] = 1.0  # identity, scalar-last
    return BodySample(
        body_poses=body,
        timestamp_realtime=0.0,
        timestamp_monotonic=0.0,
        timestamp_ns=ts_ns,
        dt=dt,
        fps=90.0,
    )


# 24 joints at the origin with identity rotations (xyz + scalar-last quaternion, the SDK layout).
IDENTITY_BODY = np.zeros((24, 7))
IDENTITY_BODY[:, 6] = 1.0


def bridge_state(timestamp_ns: int, body_stamp_ns: int | None = None, body=None, **extra) -> dict:
    state = {"device": "pico", "timestamp_ns": timestamp_ns, **extra}
    if body_stamp_ns is not None:
        state["body_raw_timestamp_ns"] = body_stamp_ns
        state["body_raw"] = body if body is not None else IDENTITY_BODY.tolist()
    return state


def smpl_fk() -> SmplFk:
    """The real SMPL-X FK; skips when its HF-hosted skeleton cannot be fetched (offline, no HF access)."""
    try:
        return SmplFk()
    except FileNotFoundError as e:
        pytest.skip(f"SMPL-X skeleton asset unavailable: {e}")


def resolve_pattern(patterns: dict[str, float], name: str) -> float:
    matches = [v for p, v in patterns.items() if fnmatchcase(name, p)]
    assert len(matches) == 1, f"{name} matched {len(matches)} patterns"
    return matches[0]


def bind_debug_publisher() -> tuple[G1DebugPublisher, int]:
    """A loopback ``G1DebugPublisher`` on a free port.

    It binds a port it is given, so the port is probed first; probe-close-rebind races
    other sockets in the process, hence the retries with fresh random ports. Skips the
    calling test without the ``sonic`` extra (pyzmq + msgpack).
    """
    zmq = pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    for _ in range(10):
        probe = zmq.Context.instance().socket(zmq.PUB)
        port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.close()
        try:
            return G1DebugPublisher(port=port, host="127.0.0.1"), port
        except zmq.ZMQError:
            continue
    raise AssertionError("could not bind a free loopback port for g1_debug")


def unpaced(streamer):
    """Return ``streamer`` with its pacing sleep stubbed out (no 20 / 50 Hz waits in tests).

    ``_sleep`` is the streamers' clock seam: a private attribute, but the only way to stop a
    streamer pacing itself without patching the global ``time.sleep``.
    """
    streamer._sleep = lambda _t: None
    return streamer
