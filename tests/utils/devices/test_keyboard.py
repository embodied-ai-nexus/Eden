"""``eden.utils.devices.keyboard``: the SE(2) keyboards' readings, over a fake ``pynput`` (see ``conftest.py``)."""

from __future__ import annotations

import math

import pytest

from eden.utils.devices.keyboard import SE2Keyboard, TeleopSE2Keyboard


@pytest.mark.parametrize(
    "cls, one_press",
    [(TeleopSE2Keyboard, 1.0), (SE2Keyboard, math.tanh(1.0))],
    ids=["teleop_saturates", "se2_ramps"],
)
def test_one_press_of_a_key(pynput, cls, one_press):
    """The teleop keyboard is full deflection from the first press; ``SE2Keyboard`` ramps through tanh."""
    device = cls(key_bindings=SE2Keyboard.wasd_key_bindings())
    try:
        pynput.press("w")
        assert device.get_command().tolist() == pytest.approx([one_press, 0.0, 0.0])
        pynput.release("w")
        assert device.get_command().tolist() == [0.0, 0.0, 0.0]
    finally:
        device.stop()


def test_stop_ends_the_listener_and_zeroes_the_readings(pynput):
    device = TeleopSE2Keyboard(key_bindings=SE2Keyboard.wasd_key_bindings())
    pynput.press("a")
    device.stop()
    assert device.get_command().tolist() == [0.0, 0.0, 0.0]
    pynput.press("a")  # nothing is listening any more
    assert device.get_command().tolist() == [0.0, 0.0, 0.0]
