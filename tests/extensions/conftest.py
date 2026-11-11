"""Fixtures shared by the extension test directories."""

from __future__ import annotations

import time

import pytest


@pytest.fixture(scope="module")
def g1_key_frame_fk():
    """The SONIC operator's calibration FK, on a real Genesis scene (one build per module).

    Shared by the operator tests (three_point, planner_streamer, ...) and the robot-side
    operator->robot loopback test. Skips when the GEAR-SONIC G1 asset cannot be fetched.
    """
    from eden.extensions.teleop.sonic.g1_fk import G1KeyFrameFk

    try:
        fk = G1KeyFrameFk()
    except FileNotFoundError as e:  # asset unavailable (offline / no HF access)
        pytest.skip(f"G1 with-hand asset unavailable: {e}")
    yield fk
    fk.close()


@pytest.fixture
def clock(monkeypatch):
    """Return ``advance(seconds)``: moves ``time.monotonic`` forward without sleeping (the clock boundary).

    It patches the process-wide clock, so threads see the jump too — which is the point when a
    test wants a reader thread to find its input stale; a test that wants its threads unaffected
    must not use it.
    """
    real_monotonic = time.monotonic
    offset = 0.0

    def advance(seconds: float) -> None:
        nonlocal offset
        offset += seconds

    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + offset)
    return advance
