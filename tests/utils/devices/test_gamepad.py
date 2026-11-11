"""``eden.utils.devices.gamepad``: the reader thread over a fake ``inputs`` pad (see ``conftest.py``)."""

from __future__ import annotations

import time

import pytest

from eden.utils.devices.gamepad import SE3Gamepad


def test_no_pad_plugged_in_is_refused_at_construction(pad):
    """``get_gamepad`` raising per call would otherwise look like a live pad reading zero for ever."""
    from eden.utils.devices import gamepad as gamepad_module

    gamepad_module.input_devices.gamepads.clear()  # the fixture's device list, with the pad gone
    with pytest.raises(RuntimeError, match="No gamepad found"):
        SE3Gamepad()


def test_losing_the_pad_zeroes_the_axes_but_keeps_a_latched_toggle(pad, eden_caplog):
    """A yanked pad reads as "stop moving"; SE3Gamepad's gripper toggle is operator state and stays."""
    device = SE3Gamepad(scale=1.0, offset=0.0, deadzone=0.0)
    try:
        pad.send(ABS_Y=-0.8, ABS_X=0.4, BTN_SOUTH=1)
        assert pad.drained()
        assert device.get_command().tolist() == pytest.approx([0.8, -0.4, 0.0, 0.0, 0.0, 0.0, 1.0])

        pad.unplug()
        deadline = time.monotonic() + 2.0
        while device.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert device.connected is False
        assert device.get_command().tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        assert "Lost the gamepad" in eden_caplog.text
    finally:
        device.stop()
