"""Tests for the works the tasks copy: their plans, the targets their tools draw, and their rules."""

import numpy as np
import pytest

from eden.tasks.art.calligraphy.references import render_plan
from eden.tasks.art.calligraphy.works import LETTERING_RULE, WORKS


@pytest.mark.parametrize("name", list(WORKS))
def test_every_stroke_starts_and_ends_in_the_air_and_writes_inside_the_sheet(name):
    work = WORKS[name]
    path, stroke_of = work.plan.path()
    for k, stroke in enumerate(work.plan.strokes):
        assert stroke.points[0][2] < 0 and stroke.points[-1][2] < 0, f"{name} stroke {k} starts or ends on the paper"
        assert (path[stroke_of == k, 2] > 0).any(), f"{name} stroke {k} never touches the paper"
    assert (path[stroke_of < 0, 2] < 0).all()  # travel between strokes is in the air
    width, height = work.sheet
    on_paper = path[path[:, 2] > 0]
    assert (np.abs(on_paper[:, 0]) < width / 2 - 0.01).all() and (np.abs(on_paper[:, 1]) < height / 2 - 0.005).all()


@pytest.mark.parametrize("name", list(WORKS))
def test_every_work_renders_a_target_each_stroke_owns_part_of(name):
    rendering = render_plan(WORKS[name])
    assert rendering.target.any()
    owned = np.bincount(rendering.owner[rendering.target], minlength=len(WORKS[name].plan.strokes))
    assert (owned > 0).all()
    assert ((rendering.owner >= 0) == rendering.target).all()
    assert rendering.path_length > 0 and rendering.mark_width > 0


def test_only_the_lettering_work_departs_from_robopaints_writing_rule():
    assert WORKS["minimum"].rule == LETTERING_RULE
    assert all(WORKS[name].rule == {} for name in ("dai", "san", "ei"))


def test_a_work_is_hashable_and_its_rule_read_only():
    """A frozen record: usable as a cache key, and a task cannot change a work's rule for every other task."""
    assert len({WORKS[name] for name in WORKS}) == len(WORKS)
    with pytest.raises(TypeError):
        WORKS["minimum"].rule["iou"] = 0.5
