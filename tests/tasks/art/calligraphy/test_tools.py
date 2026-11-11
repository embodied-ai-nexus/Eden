"""Tests for the writing tools: Wang et al.'s dynamic brush and the pointed pen."""

import math

import pytest
import torch

from eden.tasks.art.calligraphy.grading import SheetInk
from eden.tasks.art.calligraphy.tools import TOOL_MODELS, DynamicBrush, Footprint, PointedPen, stack_footprints
from eden.tasks.art.workcell import sheet_grid


def _brush_step(brush, u, v, depth):
    return brush.step(torch.tensor([[u, v]]), torch.tensor([depth]), torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize(("kind", "model"), [("brush", DynamicBrush), ("pen", PointedPen)])
def test_each_writing_tool_has_its_model(kind, model):
    assert set(TOOL_MODELS) == {"brush", "pen"}
    assert isinstance(TOOL_MODELS[kind](2, "cpu"), model)


def test_a_brush_held_at_a_depth_settles_to_the_papers_fitted_width_and_drag():
    # Wang et al.'s fit in cm: Width(z) = 1.187 z - 0.03, Drag(z) = 1.075 z + 0.138, at z = 0.5 cm.
    brush = DynamicBrush(1)
    for _ in range(5):
        mark = _brush_step(brush, 0.0, 0.0, 0.005)
    assert float(mark.width[0]) == pytest.approx((1.187 * 0.5 - 0.03) / 100, rel=1e-5)
    assert float(mark.length[0]) == pytest.approx((1.075 * 0.5 + 0.138) / 100, rel=1e-5)
    assert bool(mark.live[0])


def test_the_hairs_root_stays_put_then_is_dragged_after_the_tip():
    # Offset(0.5 cm) = -0.967 * 0.25 + 2.11 * 0.5 - 0.036 = 0.77725 cm. Moving along +u, the root stays where the
    # brush touched down until the tip is that far from it, then trails it at that distance.
    offset = (-0.967 * 0.25 + 2.11 * 0.5 - 0.036) / 100
    brush = DynamicBrush(1)
    _brush_step(brush, 0.0, 0.0, 0.005)
    for k in range(1, 21):  # 5 mm in 0.25 mm steps
        mark = _brush_step(brush, k * 0.00025, 0.0, 0.005)
    torch.testing.assert_close(mark.root, torch.zeros(1, 2), atol=1e-7, rtol=0)
    for k in range(21, 101):  # on to 25 mm
        mark = _brush_step(brush, k * 0.00025, 0.0, 0.005)
    torch.testing.assert_close(mark.root, torch.tensor([[0.025 - offset, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(mark.along, torch.tensor([[-1.0, 0.0]]))  # the hairs trail the way it came


def test_a_brush_lowered_onto_the_paper_trails_its_hairs_the_way_it_came_in():
    brush = DynamicBrush(1)
    for k in range(40):  # in the air, travelling toward -v, slowly: 0.05 mm a sample
        _brush_step(brush, 0.0, -k * 0.00005, -0.002)
    for k in range(1, 6):  # straight down onto the paper
        mark = _brush_step(brush, 0.0, -39 * 0.00005, 0.001 * k)
    torch.testing.assert_close(mark.along, torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=0)


def test_a_brush_in_the_air_lays_nothing_and_a_reset_starts_it_afresh():
    brush = DynamicBrush(2)
    tip, lateral = torch.zeros(2, 2), torch.tensor([[1.0, 0.0]] * 2)
    assert not bool(brush.step(tip, torch.tensor([-0.001, -0.001]), lateral).live.any())
    brush.step(tip + torch.tensor([0.01, 0.0]), torch.tensor([0.005, 0.005]), lateral)
    brush.reset(torch.tensor([0]))
    mark = brush.step(torch.tensor([[0.05, 0.0], [0.05, 0.0]]), torch.tensor([0.005, 0.005]), lateral)
    torch.testing.assert_close(mark.root[0], torch.tensor([0.05, 0.0]))  # restarted at the tip
    assert float(mark.root[1, 0]) < 0.05  # still trailing


def test_the_pen_swells_from_a_hairline_to_its_full_width_with_depth():
    pen = PointedPen()
    widths = pen.width_at(torch.tensor([0.0005, 0.001, 0.00225, 0.0035, 0.006]))
    torch.testing.assert_close(widths, torch.tensor([0.0004, 0.0004, 0.0012, 0.002, 0.002]))
    with pytest.raises(ValueError, match=r"needs 0 < hairline <= max_width"):
        PointedPen(hairline=0.003, max_width=0.002)


@pytest.mark.parametrize(("direction", "expected"), [((0.0, 1.0), 0.002), ((1.0, 0.0), 0.0004)])
def test_a_shaded_pen_stroke_is_wide_along_the_slit_and_a_hairline_across_it(direction, expected):
    # The pen's lateral axis is u, so its slit runs along v: a full-pressure stroke along v is as wide as the tines
    # spread (2 mm), one along u a hairline (0.4 mm).
    pen = PointedPen()
    grid = sheet_grid((0.03, 0.03), 0.0001)
    ink = SheetInk(grid, 1, pen, device="cpu")
    steps = [torch.tensor([[d * (k * 0.0002 - 0.005) for d in direction]]) for k in range(51)]
    marks = [pen.step(tip, torch.tensor([0.0035]), torch.tensor([[1.0, 0.0]])) for tip in steps]
    ink.deposit(stack_footprints(marks))
    across = 1 if direction[1] else 0  # the axis across the stroke: columns (u) for a stroke along v
    width = float(ink.inked[0].any(dim=1 - across).sum()) * grid.cell
    assert width == pytest.approx(expected, abs=2 * grid.cell)


@pytest.mark.parametrize("heading_deg", [0.0, 30.0, 45.0, 135.0])
def test_the_ink_window_holds_every_cell_of_the_deepest_mark_at_any_heading(heading_deg):
    """The flat end's corners lie hypot(drag, width) / 2 from the mark's centre: a window of only the larger of the
    two lost them on a diagonal mark pressed past 11 mm.
    """
    brush = DynamicBrush(1)
    grid = sheet_grid((0.18, 0.18), 0.0005)
    depth = torch.tensor([brush.z_max])
    width, drag = float(brush._width(depth)), float(brush._drag(depth))
    heading = math.radians(heading_deg)
    along = torch.tensor([[[math.cos(heading), math.sin(heading)]]])
    # The root off the cells' lattice, so no cell centre lies on the mark's edge, where rounding decides.
    root = torch.tensor([0.00013, 0.00007])
    live = torch.ones(1, 1, dtype=torch.bool)
    mark = Footprint(root[None, None], along, torch.tensor([[drag]]), torch.tensor([[width]]), live)
    ink = SheetInk(grid, 1, brush, device="cpu")
    ink.deposit(mark)

    centers = torch.as_tensor(grid.centers(), dtype=torch.float32)
    u, v = centers[..., 0] - root[0], centers[..., 1] - root[1]
    s, t = u * along[0, 0, 0] + v * along[0, 0, 1], v * along[0, 0, 0] - u * along[0, 0, 1]
    every_cell = brush.inside(s, t, torch.tensor(drag), torch.tensor(width))
    assert int(every_cell.sum()) > 700, "test premise: the deepest mark covers hundreds of cells"
    assert torch.equal(ink.inked[0], every_cell)
