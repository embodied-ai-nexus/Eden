"""Tests for the whiteboard task configs: the camera the tasks record with, and that training builds none."""

import pytest

from eden.options import BoxOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.whiteboard import common

TASKS = ["whiteboard_write_text", "whiteboard_write_in_colour", "whiteboard_draw_table"]


@pytest.mark.parametrize("task", TASKS)
def test_training_builds_no_camera_and_a_recording_paints_the_ink(task):
    """A camera in the training config would cost every training run though nothing renders it, so the overlay
    camera comes only with ``eden inference --rec``.
    """
    config = TASK_REGISTRY.build(task)
    assert not list(config.cameras_options.keys())

    config.add_recording_camera()
    (overlay,) = config.cameras_options.rec.overlays
    assert (overlay.name, overlay.metric_name) == ("plane_paint_overlay", "drawing")
    assert overlay.color == common.COLORS[config.scene_options.marker.ink], "the ink is the marker's colour"


def test_a_recording_without_the_whiteboard_marker_names_what_it_needs():
    """The ink's colour is the marker's; a variant that swaps the marker is told so, not handed an AttributeError."""
    config = TASK_REGISTRY.build("whiteboard_write_text")
    config.scene_options.marker = BoxOptions(size=(0.01, 0.01, 0.1))
    with pytest.raises(
        ValueError, match=r"marker_name='marker'\), but that entity is BoxOptions.*WhiteboardMarkerOptions"
    ):
        config.recording_camera()


def test_the_recording_paints_the_ink_of_the_marker_the_drawing_metric_grades():
    """One handle for the marker: the metric's ``marker_name``, not a scene key the camera assumes."""
    config = TASK_REGISTRY.build("whiteboard_write_text")
    config.scene_options.pen = config.scene_options.marker.model_copy(update={"ink": "green"})
    config.metric_options.drawing.marker_name = "pen"
    (overlay,) = config.recording_camera().overlays
    assert overlay.color == common.COLORS["green"]
