"""Tests for the paint model: the round brush's mark, the paint load, and wet acrylic laid, mixed and picked up."""

import pytest
import torch

from eden.tasks.art.painting.grading import srgb_to_lab, srgb_to_linear
from eden.tasks.art.painting.paint import Acrylic, Canvas, PaintLoad, RoundBrush, pigment, reflectance
from eden.tasks.art.workcell import PAPER_COLOR, sheet_grid

CELL = 0.0005
RED = torch.tensor([[0.9, 0.1, 0.1]])
BLUE = torch.tensor([[0.1, 0.2, 0.8]])
YELLOW = torch.tensor([[0.95, 0.85, 0.1]])
EVERY = torch.ones(1, dtype=torch.bool)
DEPTH = RoundBrush().depth_for(0.008)  # an 8 mm mark


@pytest.fixture
def canvas():
    return Canvas(sheet_grid((0.06, 0.06), CELL), 1, device="cpu")


def _stroke(canvas, v, *, spacing=0.00025, run=8):
    """A level stroke at ``v`` from ``u = -2`` to ``2`` cm, laid in runs of ``run`` samples ``spacing`` apart, with a
    sample in the air before it and after it.
    """
    u = torch.arange(-0.02 - spacing, 0.02 + 1.5 * spacing, spacing)
    depth = torch.full_like(u, DEPTH)
    depth[0] = depth[-1] = -0.001
    uv = torch.stack([u, torch.full_like(u, v)], dim=-1)[None]
    for a in range(0, len(u), run):
        canvas.lay(uv[:, a : a + run], depth[None, a : a + run])


def _at(canvas, u, v):
    row, col = canvas.grid.cell_of([u, v])
    return canvas.rgb()[0, row, col]


def _across(canvas, u=0.0, half_width=0.003):
    """The colours across a stroke at ``v = 0``, at ``u``, ``half_width`` (m) to either side of its middle."""
    return torch.stack([_at(canvas, u, v) for v in torch.linspace(-half_width, half_width, 13).tolist()])


def _delta_e(a, b):
    return float((srgb_to_lab(a) - srgb_to_lab(b)).norm(dim=-1).max())


def test_the_mark_grows_from_a_touch_to_full_press_and_a_size_has_its_depth():
    brush = RoundBrush()
    diameters = brush.diameter_at(torch.tensor([-0.001, 0.0, 0.0001, brush.full_press, 2 * brush.full_press]))
    assert diameters[0] == 0.0 and diameters[1] == 0.0  # in the air or just at the surface: no mark
    assert diameters[3] == pytest.approx(brush.max_diameter) and diameters[4] == pytest.approx(brush.max_diameter)
    for size in (0.003, 0.0094, 0.016):
        assert float(brush.diameter_at(torch.tensor(brush.depth_for(size)))) == pytest.approx(size)
    with pytest.raises(ValueError, match=r"a 20.0 mm mark is outside the brush's 1.0-18.0 mm range"):
        brush.depth_for(0.02)


def test_the_paint_is_opaque_until_the_brush_runs_dry():
    load = PaintLoad()
    opacity = load.opacity(torch.tensor([1.0, load.reserve, load.dry_load, load.dry_load / 2, 0.0]))
    torch.testing.assert_close(opacity, torch.tensor([1.0, 1.0, 1.0, 0.5, 0.0]))


def test_paints_mix_as_pigments_do_blue_and_yellow_make_green():
    # A paint alone is its own colour.
    for paint in (RED, BLUE, YELLOW):
        torch.testing.assert_close(reflectance(pigment(paint)), srgb_to_linear(paint), atol=1e-5, rtol=1e-4)
    green = reflectance((pigment(BLUE) + pigment(YELLOW)) / 2)[0]
    assert float(green[1]) > 1.5 * float(green[0]) and float(green[1]) > 1.5 * float(green[2])
    # Averaging the colours instead makes a grey with about as much red as green.
    average = srgb_to_linear((BLUE + YELLOW) / 2)[0]
    assert float(average[1]) < 1.2 * float(average[0])


def test_a_full_brush_covers_the_paper_in_its_paint(canvas):
    torch.testing.assert_close(_at(canvas, 0.0, 0.0), torch.tensor(PAPER_COLOR))
    canvas.dip(EVERY, RED)
    torch.testing.assert_close(canvas.brush_rgb(), RED)
    _stroke(canvas, 0.0)
    assert _delta_e(_across(canvas), RED[0]) < 1.0
    torch.testing.assert_close(canvas.brush, pigment(RED))  # nothing wet under it but its own paint


def test_what_is_laid_does_not_depend_on_how_finely_the_brush_is_sampled(canvas):
    fine = Canvas(canvas.grid, 1, device="cpu")
    for target, spacing, run in ((canvas, 0.00025, 8), (fine, 0.0001, 20)):
        target.dip(EVERY, BLUE)
        _stroke(target, 0.0)
        target.dip(EVERY, YELLOW)
        _stroke(target, 0.004, spacing=spacing, run=run)  # half over the blue, still wet
    # The same to within a few per cent; the most at the stroke's lift-off, laid by its last samples alone.
    difference = (canvas.rgb() - fine.rgb()).abs()
    assert float(difference.amax()) < 0.06 and float(difference.mean()) < 1e-3
    torch.testing.assert_close(canvas.brush, fine.brush, atol=0.05, rtol=0.05)


def test_a_stroke_into_wet_paint_mixes_with_it_and_dirties_the_brush(canvas):
    canvas.dip(EVERY, BLUE)
    _stroke(canvas, 0.0)
    canvas.dip(EVERY, YELLOW)
    _stroke(canvas, 0.0)  # straight over the wet blue
    mixed = _at(canvas, 0.01, 0.0)
    assert _delta_e(mixed, YELLOW[0]) > 10.0, "the blue shows through"
    assert float(mixed[1]) > float(mixed[0]) and float(mixed[1]) > float(mixed[2]), "and it is green"
    brush = canvas.brush_rgb()[0]
    assert _delta_e(brush, YELLOW[0]) > 10.0 and float(brush[0]) < float(YELLOW[0, 0]), "the brush is greener"


def test_dry_paint_is_covered_and_not_picked_up(canvas):
    canvas.dip(EVERY, BLUE)
    _stroke(canvas, 0.0)
    canvas.tick(20 * Acrylic().open_time)
    canvas.dip(EVERY, YELLOW)
    _stroke(canvas, 0.0)
    assert _delta_e(_across(canvas), YELLOW[0]) < 1.0
    torch.testing.assert_close(canvas.brush, pigment(YELLOW))


def test_a_stroke_beside_wet_paint_blends_into_it_across_its_side_but_keeps_its_edge_against_dry_paint():
    def beside_blue(medium, wait):  # the colour 1 mm inside a blue stroke from a yellow one's side (ΔE from blue)
        canvas = Canvas(sheet_grid((0.06, 0.06), CELL), 1, medium=medium, device="cpu")
        canvas.dip(EVERY, BLUE)
        _stroke(canvas, 0.0)
        canvas.tick(wait)
        canvas.dip(EVERY, YELLOW)
        _stroke(canvas, 0.007)  # its side at 3 mm, overlapping the blue's by 1 mm
        return _delta_e(_at(canvas, 0.0, 0.002), _at(canvas, 0.0, -0.002))

    assert beside_blue(Acrylic(), 0.0) > 20.0, "the wet blue and the yellow blend across the yellow's side"
    assert beside_blue(Acrylic(blend_rate=0.0), 0.0) < 1.0, "it is the blending: without it the edge is sharp"
    assert beside_blue(Acrylic(), 20 * Acrylic().open_time) < 1.0, "dry paint keeps its edge"
    with pytest.raises(ValueError, match=r"blend_width=0.0 must be positive .* set blend_rate=0.0 instead"):
        Acrylic(blend_width=0.0)


def test_the_bristles_streak_paint_laid_into_wet_paint_but_not_on_paper(canvas):
    def spread(colours):  # how much the colour varies across the stroke (ΔE)
        return float(srgb_to_lab(colours).std(dim=0).norm())

    canvas.dip(EVERY, RED)
    _stroke(canvas, 0.0)
    on_paper = spread(_across(canvas, -0.015))
    canvas.dip(EVERY, YELLOW)
    _stroke(canvas, 0.0)
    # Near its start, before the brush has picked up much red; and in its middle, clear of its sides' blending.
    into_wet = spread(_across(canvas, -0.015, half_width=0.0015))
    assert on_paper < 1.0 and into_wet > 2.0


def test_a_brush_in_the_air_or_off_the_sheet_lays_nothing(canvas):
    canvas.dip(EVERY, RED)
    canvas.lay(torch.tensor([[[0.0, 0.0], [0.001, 0.0]]]), torch.tensor([[-0.001, -0.001]]))
    canvas.lay(torch.tensor([[[0.1, 0.0], [0.101, 0.0]]]), torch.tensor([[DEPTH, DEPTH]]))
    assert not bool(canvas.painted_cells().any())


def test_an_undipped_brush_lays_nothing_and_a_dipped_one_uses_up_the_paper_it_sweeps(canvas):
    _stroke(canvas, 0.0)
    assert not bool(canvas.painted_cells().any()) and float(canvas.load[0]) == 0.0
    canvas.dip(EVERY, RED)
    assert float(canvas.load[0]) == 1.0
    _stroke(canvas, 0.0)
    # 4 cm at 8 mm wide, of a full brush's 3000 mm^2 (and its touch-down and lift-off).
    used = 0.04 * 0.008 / PaintLoad().capacity
    assert float(canvas.load[0]) == pytest.approx(1.0 - used, abs=0.01)


def test_the_sheet_of_some_envs_is_those_envs_of_every_env_s_sheet():
    canvas = Canvas(sheet_grid((0.06, 0.06), CELL), 3, device="cpu")
    canvas.dip(torch.tensor([True, False, True]), torch.cat([RED, RED, BLUE]))
    uv = torch.tensor([[0.0, 0.0], [0.002, 0.0]]).expand(3, -1, -1)
    canvas.lay(uv, torch.full((3, 2), DEPTH))
    envs = torch.tensor([2, 0])
    torch.testing.assert_close(canvas.rgb(envs), canvas.rgb()[envs])
    assert torch.equal(canvas.painted_cells(envs), canvas.painted_cells()[envs])
    assert canvas.rgb(envs).shape == (2, *canvas.grid.shape, 3)


def test_the_first_sample_after_a_reset_owes_nothing_to_the_last_episode(canvas):
    """A dab from a brush running dry is streaked across the way the brush moved: after a reset, from nowhere."""

    def dab(target):
        target.dip(EVERY, RED)
        target.load.fill_(0.1)  # half opacity: the bristles' streaks show
        target.lay(torch.tensor([[[0.0, 0.0]]]), torch.tensor([[DEPTH]]))
        return target.rgb()

    fresh = dab(Canvas(canvas.grid, 1, device="cpu"))
    canvas.dip(EVERY, BLUE)
    canvas.lay(torch.tensor([[[0.0, 0.02], [0.01, 0.025]]]), torch.tensor([[DEPTH, DEPTH]]))
    canvas.reset(EVERY)
    torch.testing.assert_close(dab(canvas), fresh)


def test_a_reset_leaves_bare_paper_and_a_clean_brush(canvas):
    canvas.dip(EVERY, RED)
    _stroke(canvas, 0.0)
    canvas.tick(5.0)
    canvas.reset(slice(None))
    torch.testing.assert_close(_at(canvas, 0.0, 0.0), torch.tensor(PAPER_COLOR))
    assert not bool(canvas.painted_cells().any())
    assert float(canvas.clock[0]) == 0.0 and not bool(canvas.brush.any())
