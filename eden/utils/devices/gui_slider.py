"""ImGui slider overlay for manual DOF teleoperation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

import eden as en
from eden.utils.devices.base import DeviceBase

if TYPE_CHECKING:
    from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
    from genesis.vis.viewer import Viewer


class SliderGUI(DeviceBase):
    """
    ImGui-based slider device for teleoperation.

    Attaches an :class:`~genesis.ext.pyrender.overlay.ImGuiOverlayPlugin` to the Genesis
    viewer and registers a custom side panel with one slider per named DoF plus a Reset
    button. The current slider values are polled via :meth:`get_command`, shape ``(1, n_dofs)``.

    Parameters
    ----------
    viewer: genesis.vis.viewer.Viewer
        The viewer to attach the overlay to, typically ``env.scene.viewer``. The viewer
        must already be built — the plugin is registered immediately.
    dofs_name: list[str]
        Labels for each slider (used as the ImGui label text).
    dofs_pos_limits: torch.Tensor
        Tensor of shape ``(n_dofs, 2)`` with ``[lo, hi]`` for each DoF. ``±inf`` is
        clipped to ``±pi`` to keep sliders bounded.
    reset_callback: Callable[[], None] | None
        Optional callback invoked on the viewer thread when the Reset button is pressed.
    initial_position: torch.Tensor | None
        Initial slider values (also used as the Reset target). If ``None``, defaults to
        zero clipped into ``dofs_pos_limits``.
    plugin: ImGuiOverlayPlugin | None
        Existing overlay plugin to register the panel on. If ``None``, the viewer's first
        :class:`ImGuiOverlayPlugin` is reused, and a new one is added only when it has none,
        so several devices share one overlay.
    """

    def __init__(
        self,
        viewer: Viewer,
        dofs_name: list[str],
        dofs_pos_limits: torch.Tensor,
        reset_callback: Callable[[], None] | None = None,
        initial_position: torch.Tensor | None = None,
        plugin: ImGuiOverlayPlugin | None = None,
    ):
        from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

        super().__init__()

        self.dofs_name = list(dofs_name)
        limits = dofs_pos_limits.detach().cpu().numpy().astype(np.float32, copy=True)
        limits[limits == -np.inf] = -np.pi
        limits[limits == +np.inf] = +np.pi
        self.dofs_pos_limits = limits

        if initial_position is None:
            defaults = np.zeros(len(self.dofs_name), dtype=np.float32)
        else:
            defaults = initial_position.detach().cpu().numpy().astype(np.float32, copy=True)
        self._defaults = np.clip(defaults, limits[:, 0], limits[:, 1])
        self._readings = self._defaults.copy()
        self._reset_callback = reset_callback
        self._viewer = viewer

        if plugin is None:
            plugin = next(
                (p for p in viewer.plugins if isinstance(p, ImGuiOverlayPlugin)),
                None,
            )
        if plugin is None:
            plugin = viewer.add_plugin(ImGuiOverlayPlugin())
        self._plugin = plugin
        self._plugin.register_panel(self._draw_panel, section="side")
        en.logger.info("SliderGUI ImGui panel registered.")

    def _draw_panel(self, imgui):
        imgui.text("Teleop Sliders")
        for i, name in enumerate(self.dofs_name):
            lo = float(self.dofs_pos_limits[i, 0])
            hi = float(self.dofs_pos_limits[i, 1])
            changed, new_val = imgui.slider_float(
                f"{name}##teleop_slider_{i}", float(self._readings[i]), lo, hi, "%.3f"
            )
            if changed:
                self._readings[i] = new_val
        if imgui.button("Reset##teleop_slider_reset"):
            self._readings[:] = self._defaults
            if self._reset_callback is not None:
                self._reset_callback()

    def get_raw_data(self) -> torch.Tensor:
        return torch.from_numpy(self._readings.copy()).unsqueeze(0)

    def get_command(self) -> torch.Tensor:
        return torch.from_numpy(self._readings.copy()).unsqueeze(0)

    def set_command(self, values: torch.Tensor | np.ndarray) -> None:
        """Overwrite the slider readings programmatically (clipped into the slider limits).

        For syncing the panel to an externally-known pose — e.g. a real-robot
        teleop keeping the sliders glued to the measured joint positions until
        control is handed over, so the first commanded target equals the
        current pose instead of jumping to wherever the sliders were left.

        Parameters
        ----------
        values: torch.Tensor | np.ndarray
            New slider values, shape ``(n_dofs,)`` or ``(1, n_dofs)``.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(self.dofs_name):
            raise ValueError(f"set_command expects {len(self.dofs_name)} values, got {arr.shape[0]}")
        self._readings[:] = np.clip(arr, self.dofs_pos_limits[:, 0], self.dofs_pos_limits[:, 1])

    def close(self) -> None:
        # The ImGui plugin lifecycle is owned by the viewer; nothing to clean up here.
        pass
