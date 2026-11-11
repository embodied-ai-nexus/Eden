"""Tests for the calligraphy task configs: the camera the tasks record with, and that training builds none."""

import pytest

from eden.tasks import TASK_REGISTRY
from eden.tasks.art.calligraphy.grading import INK_COLOR

TASKS = ["art_calligraphy_dai", "art_calligraphy_san", "art_calligraphy_ei", "art_calligraphy_lettering_minimum"]


@pytest.mark.parametrize("task", TASKS)
def test_training_builds_no_camera_and_a_recording_paints_the_ink(task):
    """A camera in the training config would cost every training run though nothing renders it, so the overlay
    camera comes only with ``eden inference --rec``.
    """
    config = TASK_REGISTRY.build(task)
    assert not list(config.cameras_options.keys())

    config.add_recording_camera()
    (overlay,) = config.cameras_options.rec.overlays
    assert (overlay.name, overlay.metric_name, overlay.color) == ("plane_paint_overlay", "writing", INK_COLOR)


def test_the_recording_paints_the_writing_metric_whatever_its_key():
    config = TASK_REGISTRY.build("art_calligraphy_dai")
    writing = config.metric_options.writing
    delattr(config.metric_options, "writing")
    config.metric_options.grade = writing
    (overlay,) = config.recording_camera().overlays
    assert overlay.metric_name == "grade"


def test_a_recording_without_a_writing_metric_names_what_it_needs():
    config = TASK_REGISTRY.build("art_calligraphy_dai")
    delattr(config.metric_options, "writing")
    with pytest.raises(ValueError, match=r"has 0 of them \(metrics: \{\}\)\. Keep exactly one"):
        config.recording_camera()


@pytest.mark.parametrize(("task", "iou"), [("art_calligraphy_dai", 0.8), ("art_calligraphy_lettering_minimum", 0.7)])
def test_each_task_grades_with_its_work_s_rule(task, iou):
    writing = TASK_REGISTRY.build(task).metric_options.writing
    assert (writing.iou, writing.success_threshold) == (iou, iou)
