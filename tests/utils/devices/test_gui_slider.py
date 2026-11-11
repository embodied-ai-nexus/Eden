"""``SliderGUI``: slider state, its programmatic command, and the Reset button.

The ImGui viewer is the boundary: a small fake viewer and overlay plugin stand in for Genesis' live
window, and a scripted ``imgui`` drives the panel the plugin would draw each frame.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.utils.devices.gui_slider import SliderGUI


class _Viewer:
    def __init__(self):
        self.plugins = []
        self.alive = True

    def add_plugin(self, plugin):
        self.plugins.append(plugin)
        return plugin

    def is_alive(self):
        return self.alive


class _OverlayPlugin:
    def __init__(self):
        self.panels = []

    def register_panel(self, draw, section):
        self.panels.append((draw, section))


class _ImGui:
    """Scripted ImGui: ``slider_float`` reports the values in ``drag``, ``button`` reports ``press_reset``."""

    def __init__(self, drag=None, press_reset=False):
        self.drag, self.press_reset = drag or {}, press_reset

    def text(self, _):
        pass

    def slider_float(self, label, value, lo, hi, fmt):
        name = label.split("##")[0]
        return (name in self.drag), self.drag.get(name, value)

    def button(self, _):
        return self.press_reset


@pytest.fixture
def slider():
    """Three sliders, one with infinite limits, starting from a pose partly outside them."""
    resets = []
    plugin = _OverlayPlugin()
    slider = SliderGUI(
        _Viewer(),
        ["a", "b", "c"],
        torch.tensor([[-1.0, 1.0], [-0.5, 0.5], [-float("inf"), float("inf")]]),
        reset_callback=lambda: resets.append(True),
        initial_position=torch.tensor([0.2, 3.0, 0.0]),
        plugin=plugin,
    )
    return slider, plugin, resets


def _draw(plugin, imgui):
    ((draw, section),) = plugin.panels
    assert section == "side"
    draw(imgui)


def test_starts_at_the_initial_pose_clipped_into_the_limits(slider):
    s, _, _ = slider
    np.testing.assert_allclose(s.get_command().numpy(), [[0.2, 0.5, 0.0]])


def test_dragging_a_slider_changes_the_command_and_reset_restores_the_start(slider):
    s, plugin, resets = slider
    _draw(plugin, _ImGui(drag={"a": -0.7, "c": 2.0}))
    np.testing.assert_allclose(s.get_command().numpy(), [[-0.7, 0.5, 2.0]], rtol=1e-6)
    _draw(plugin, _ImGui(press_reset=True))
    np.testing.assert_allclose(s.get_command().numpy(), [[0.2, 0.5, 0.0]])
    assert resets == [True]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([0.25, -0.25, 1.5]), [0.25, -0.25, 1.5]),
        (torch.tensor([[0.1, 0.2, 0.3]]), [0.1, 0.2, 0.3]),  # a (1, n) tensor
        (np.array([5.0, -5.0, -5.0]), [1.0, -0.5, -np.pi]),  # clipped; infinite limits become +-pi
    ],
    ids=["array", "batched_tensor", "clipped"],
)
def test_set_command_is_what_get_command_returns(slider, values, expected):
    """The hook a real-robot teleop uses to glue the panel to the measured pose before handing over control."""
    s, _, _ = slider
    s.set_command(values)
    np.testing.assert_allclose(s.get_command().numpy(), [expected], rtol=1e-6)


def test_set_command_does_not_alias_its_input(slider):
    s, _, _ = slider
    values = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    s.set_command(values)
    values[0] = 9.0
    assert s.get_command()[0, 0].item() == pytest.approx(0.1)


def test_set_command_rejects_the_wrong_length(slider):
    s, _, _ = slider
    with pytest.raises(ValueError, match="expects 3 values"):
        s.set_command(np.zeros(4))
