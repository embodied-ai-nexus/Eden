"""Tests for ``eden.tasks.manipulation.lift_box_rgb.config``: the wrist-camera variants' renderer contract."""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY

RGB_TASKS = ("manipulation_lift_box_rgb", "manipulation_lift_ycb_rgb")


@pytest.mark.parametrize("task_name", RGB_TASKS)
def test_renderer_options_accepts_none(task_name):
    """A camera-free / headless build passes ``renderer_options=None``."""
    assert TASK_REGISTRY.get(task_name)(renderer_options=None).renderer_options is None


@pytest.mark.parametrize("task_name", RGB_TASKS)
def test_madrona_lights_survive_reconstruction(task_name):
    """Madrona renders ambient-only without lights, so a reconstruction must keep the light-carrying subclass."""
    cfg = TASK_REGISTRY.build(task_name)
    for label, candidate in (("built", cfg), ("reconstructed", cfg.with_overrides_from_dict({}))):
        renderer = candidate.renderer_options
        assert type(renderer).__name__ == "BatchRendererOptions", f"{label}: {type(renderer).__name__}"
        assert len(renderer.lights) == 2, f"{label}: {len(renderer.lights)} lights"
