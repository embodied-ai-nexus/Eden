"""Tests for the layered stroke planner, on a picture made for the test."""

import pytest
import torch

from eden.tasks.art.painting.grading import srgb_to_lab
from eden.tasks.art.painting.paint import Acrylic
from eden.tasks.art.painting.planner import Layer, load_picture, make_palette, plan_painting
from eden.tasks.art.painting.references import render_painting
from eden.tasks.art.workcell import sheet_grid
from tests.tasks.art.painting.conftest import picture_file

SHEET, CELL = (0.06, 0.06), 0.0005


def _picture() -> torch.Tensor:
    """A dark blue top half over a light sand bottom half, on the sheet's grid."""
    rows, cols = sheet_grid(SHEET, CELL).shape
    picture = torch.empty(rows, cols, 3)
    picture[: rows // 2] = torch.tensor((0.1, 0.2, 0.45))
    picture[rows // 2 :] = torch.tensor((0.85, 0.75, 0.55))
    return picture


def test_the_palette_is_the_picture_s_colours_darkest_first():
    palette = make_palette(_picture(), 2)
    torch.testing.assert_close(palette, torch.tensor([[0.1, 0.2, 0.45], [0.85, 0.75, 0.55]]))


def test_a_paint_no_cell_is_nearest_is_still_a_colour_of_the_picture():
    """Two colours asked for four paints: two centres start on the same colour, and one of them never draws a cell."""
    palette = make_palette(_picture(), 4)
    assert bool(torch.isfinite(palette).all())
    colours = torch.tensor([[0.1, 0.2, 0.45], [0.85, 0.75, 0.55]])
    assert torch.cdist(palette, colours).min(dim=1).values.max() < 1e-5


def test_the_plan_sweeps_each_region_in_its_paint_and_paints_the_picture():
    work = plan_painting(_picture(), name="halves", title="halves", paints=2, layers=(Layer(0.016, 6.0),), sheet=SHEET)
    for item in work.strokes:
        points = item.stroke.points
        assert len({v for _, v, _ in points}) == 1, "a sweep is horizontal"
        assert (points[0][2] < 0.0) and (points[-1][2] < 0.0), "it starts and ends in the air"
        assert (item.paint == 0) == (points[0][1] > 0.0), "the dark paint above, the light below"
    # Painted in paint that dries at once, as the planner plans: what the plan paints, not how its strokes blend.
    target = render_painting(work, medium=Acrylic(open_time=1e-6)).rgb
    delta_e = (srgb_to_lab(target) - srgb_to_lab(_picture())).norm(dim=-1)
    rows, cols = delta_e.shape
    # Away from the sheet's edges and the line between the halves, the target is the picture's colour.
    inner = torch.cat([delta_e[10 : rows // 2 - 10, 10:-10], delta_e[rows // 2 + 10 : -10, 10:-10]])
    assert float(inner.quantile(0.9)) < 1.0


def test_a_picture_is_its_crop_box_on_the_sheet_s_grid(tmp_path):
    picture = load_picture(picture_file(tmp_path / "p.png"), SHEET, CELL, (60, 20, 180, 140))
    assert tuple(picture.shape) == (*sheet_grid(SHEET, CELL).shape, 3)
    torch.testing.assert_close(picture.mean(dim=(0, 1)), torch.tensor([220, 30, 30]) / 255.0, atol=0.01, rtol=0)


@pytest.mark.parametrize(
    ("crop", "message"),
    [
        ((200, 100, 320, 220), r"crop=\(200, 100, 320, 220\) is not a box inside p\.png, which is 300 x 200 pixels"),
        ((0, 0, 200, 100), r"crop=\(0, 0, 200, 100\) is 200 x 100 pixels \(aspect 2\.000\), but the sheet"),
    ],
)
def test_a_crop_box_off_the_picture_or_of_another_aspect_is_refused(tmp_path, crop, message):
    with pytest.raises(ValueError, match=message):
        load_picture(picture_file(tmp_path / "p.png"), SHEET, CELL, crop)


def test_a_plan_and_its_target_are_the_same_whatever_torch_s_default_dtype():
    """At eden.init(precision="64") torch's default dtype is float64; the planner and the canvas stay float32."""
    plan = dict(name="halves", title="halves", paints=2, layers=(Layer(0.016, 6.0),), sheet=SHEET)
    picture = _picture()  # float32, as load_picture's always is
    single = plan_painting(picture, **plan)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        double = plan_painting(picture, **plan)
        target = render_painting(double).rgb
    finally:
        torch.set_default_dtype(default)
    assert double == single
    torch.testing.assert_close(target, render_painting(single).rgb, rtol=0, atol=0)
