"""Tests for the painting task configs: the camera the tasks record with, and each task's rules."""

import pytest

from eden.tasks import TASK_REGISTRY
from eden.tasks.art.painting.palette import PaletteOptions

TASKS = ["art_painting_sunset", "art_painting_fuji_study", "art_painting_fuji"]


@pytest.mark.parametrize("task", TASKS)
def test_training_builds_no_camera_and_a_recording_paints_the_paint_in_its_colours(task):
    """A camera in the training config would cost every training run though nothing renders it, so the overlay
    camera comes only with ``eden inference --rec``.
    """
    config = TASK_REGISTRY.build(task)
    assert not list(config.cameras_options.keys())

    config.add_recording_camera()
    (overlay,) = config.cameras_options.rec.overlays
    assert (overlay.name, overlay.metric_name, overlay.color) == ("plane_paint_overlay", "painting", None)


@pytest.mark.parametrize("task", TASKS)
def test_the_palette_holds_the_work_s_paints(task):
    config = TASK_REGISTRY.build(task)
    palette, work = config.scene_options.palette, config.metric_options.painting.work
    assert isinstance(palette, PaletteOptions) and palette.work == work


def test_the_recording_paints_the_painting_metric_whatever_its_key():
    config = TASK_REGISTRY.build("art_painting_sunset")
    painting = config.metric_options.painting
    delattr(config.metric_options, "painting")
    config.metric_options.copy_grade = painting
    (overlay,) = config.recording_camera().overlays
    assert overlay.metric_name == "copy_grade"


def test_a_recording_without_a_painting_metric_names_what_it_needs():
    config = TASK_REGISTRY.build("art_painting_sunset")
    delattr(config.metric_options, "painting")
    with pytest.raises(ValueError, match=r"has 0 of them \(metrics: \{\}\)\. Keep exactly one"):
        config.recording_camera()


@pytest.mark.parametrize(("task", "de_mean"), [("art_painting_sunset", 1.0), ("art_painting_fuji", 2.0)])
def test_each_task_grades_with_its_work_s_rule(task, de_mean):
    painting = TASK_REGISTRY.build(task).metric_options.painting
    assert painting.de_mean == de_mean and painting.success_threshold == painting.ssim_l
