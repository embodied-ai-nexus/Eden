"""``eden.utils.devices.se2``: building an SE(2) input device by name, and scaling its deflection to a velocity."""

from __future__ import annotations

import pytest
import torch

from eden.extensions.deployment.utils.operator_input import KEYBOARD_COMMANDS
from eden.utils.devices import se2


@pytest.mark.parametrize(
    "device, options, error",
    [("spacemouse", None, r"Unknown SE\(2\) device"), ("gamepad", {"layout": "n64"}, "Unknown gamepad layout")],
)
def test_an_unknown_device_or_layout_is_rejected(device, options, error):
    with pytest.raises(ValueError, match=error):
        se2.make_se2_device(device, options)


def test_teleop_key_bindings_avoid_the_operator_keys(pynput):
    """The listener is global, so a letter/digit binding would also fire on START / slot switches."""
    bindings = se2.teleop_key_bindings()
    bound_chars = {key for key in bindings if isinstance(key, str)}
    assert not bound_chars & set(KEYBOARD_COMMANDS), f"binding collides with an operator key: {bound_chars}"
    assert {b.index for b in bindings.values()} == {0, 1, 2}, "all three SE(2) axes must be reachable"


@pytest.mark.parametrize(
    "deflection, low, high, expected",
    [
        ([0.3, -0.7, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.3, -0.7, 1.0]),  # identity on a unit range
        ([1.0, -1.0, 1.0], [-1.0, -1.0, -0.5], [1.0, 1.0, 0.5], [1.0, -1.0, 0.5]),
        # Full deflection maps to the larger side, then clamps: the short side saturates early.
        ([1.0, -1.0, -0.5], [-2.0, -2.0, -2.0], [3.0, 3.0, 3.0], [3.0, -2.0, -1.5]),
        ([4.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]),  # a miscalibrated stick
    ],
)
def test_scale_se2_deflection(deflection, low, high, expected):
    scaled = se2.scale_se2_deflection(torch.tensor(deflection), torch.tensor(low), torch.tensor(high))
    assert scaled.tolist() == pytest.approx(expected)


def test_a_keyboard_device_is_full_deflection_on_one_press(pynput):
    """One press of Up is the top of the range, however long it is held or the OS repeats it."""
    device = se2.make_se2_device("keyboard")
    try:
        low, high = torch.tensor([-1.0, -1.0, -0.5]), torch.tensor([1.0, 1.0, 0.5])
        pynput.press(pynput.Key.up)
        pynput.press(",")
        assert se2.scale_se2_deflection(device.get_command(), low, high).tolist() == pytest.approx([1.0, 0.0, 0.5])
        for _ in range(6):  # key-repeat accumulates the raw reading ...
            pynput.press(pynput.Key.up)
        assert device.get_command()[0].item() == pytest.approx(1.0)  # ... not the deflection
        pynput.release(pynput.Key.up)
        assert device.get_command()[0].item() == 0.0
    finally:
        device.stop()


def test_a_gamepad_device_emits_a_normalised_deflection_in_its_layouts_scale(pad):
    """Xbox sticks report -32768..32767; pushed fully forward (stick Y negative) is vx = +1."""
    device = se2.make_se2_device("gamepad", {"layout": "xbox"})
    try:
        pad.send(ABS_Y=-32768, ABS_RX=16384)
        assert pad.drained()
        assert device.get_command().tolist() == pytest.approx([1.0, 0.0, -0.5])
    finally:
        device.stop()
