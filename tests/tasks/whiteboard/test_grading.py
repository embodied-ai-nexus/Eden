"""Tests for the whiteboard ink and verifiers: the original's deposition law, lettering F1 and table checks."""

import math

import numpy as np
import pytest
import torch
from scipy.ndimage import maximum_filter

from eden.tasks.whiteboard.grading import (
    OFF_GRID_PATCH,
    BoardGrid,
    InkLayer,
    StrokeGuide,
    TableJudge,
    TextJudge,
    visible_weight,
)
from eden.tasks.whiteboard.strokes import DRAWINGS, LINE_WIDTH

CELL = 0.001
SIM_DT = 0.005


def _ink(grid, *, semi_axes=(0.0025, 0.001), stray_map=None):
    stray_map = torch.zeros(grid.shape) if stray_map is None else stray_map
    return InkLayer(grid, 1, semi_axes=semi_axes, rate=350.0, force_cap=1.0, stray_map=stray_map, device="cpu")


def _axes(layer, nib_x=(1.0, 0.0), nib_y=(0.0, 1.0), samples=1):
    axes = layer.footprint(torch.tensor([[nib_x]]), torch.tensor([[nib_y]]))  # (1, 1, 2, 2)
    return axes.expand(1, samples, 2, 2)


def _press(layer, uv, force, axes=None, dt=SIM_DT):
    """Lay the samples ``uv (S, 2)`` pressing ``force`` each, on env 0."""
    uv = torch.as_tensor(uv, dtype=torch.float32).reshape(1, -1, 2)
    axes = _axes(layer, samples=uv.shape[1]) if axes is None else axes
    return layer.deposit(uv, torch.full(uv.shape[:2], float(force)), axes, dt)


def _cell(layer, row, col):
    return layer.pigment[0, row * layer.grid.cols + col]


@pytest.fixture
def small_grid():
    return BoardGrid(u0=-0.02, v0=0.02, cell=CELL, rows=40, cols=40)


def test_one_press_lays_the_originals_transfer(small_grid):
    # Under the footprint's full-coverage core the pigment is 1 - exp(-350 dt min(F, 1)): 0.5 N for 10 ms.
    layer = _ink(small_grid)
    centre = [[0.0005, -0.0005]]  # the centre of cell (20, 20)
    _press(layer, centre, 0.5, dt=0.01)
    assert float(_cell(layer, 20, 20)) == pytest.approx(1.0 - math.exp(-350.0 * 0.01 * 0.5), rel=1e-5)
    # Above 1 N the transfer no longer speeds up; and two samples compose like one twice as long.
    capped = _ink(small_grid)
    _press(capped, centre * 2, 5.0, dt=0.005)
    assert float(_cell(capped, 20, 20)) == pytest.approx(1.0 - math.exp(-3.5), rel=1e-5)


def test_no_press_draws_nothing(small_grid):
    layer = _ink(small_grid)
    _press(layer, [[0.0, 0.0], [0.01, 0.0]], 0.0)
    assert float(layer.pigment.sum()) == 0.0 and float(layer.outside_m2[0]) == 0.0


def test_the_chisel_footprint_turns_with_the_nib(small_grid):
    # The 5 x 2 mm chisel: wide along the nib's x. Lying along u it inks 5 mm across the columns, 2 mm down the rows.
    centre = [[0.0005, -0.0005]]
    along_u = _ink(small_grid)
    _press(along_u, centre, 1.0, dt=0.05)
    inked = along_u.weight[0] > 0.5
    assert int(inked[20].sum()) == 5 and int(inked[:, 20].sum()) == 3
    along_v = _ink(small_grid)
    _press(along_v, centre, 1.0, _axes(along_v, nib_x=(0.0, 1.0), nib_y=(1.0, 0.0)), dt=0.05)
    turned = along_v.weight[0] > 0.5
    assert int(turned[:, 20].sum()) == 5 and int(turned[20].sum()) == 3


def test_a_stroke_sampled_every_sim_step_inks_a_gapless_line(small_grid):
    # 12 mm in one 50 ms control step (0.24 m/s, past the arm's reach), sampled at its 10 sim steps.
    layer = _ink(small_grid)
    path = np.c_[np.linspace(-0.006, 0.006, 10), np.full(10, -0.0005)]
    _press(layer, path, 1.0, _axes(layer, nib_x=(0.0, 1.0), nib_y=(1.0, 0.0), samples=10))
    row = layer.weight[0, 20]
    assert bool((row[14:27] > 0.5).all())


def test_ink_off_the_grid_is_counted_not_stored(small_grid):
    layer = _ink(small_grid)
    _press(layer, [[0.10, 0.0]], 1.0, dt=0.05)
    assert float(layer.pigment.sum()) == 0.0
    # The footprint covers ~5 x 2 mm of fully visible ink.
    assert 5e-6 < float(layer.outside_m2[0]) < 1.5e-5


def _visible_area(layer):
    return float(layer.weight.sum()) * CELL**2 + float(layer.outside_m2[0])


def test_a_nib_resting_or_tapping_off_the_grid_counts_its_mark_once(small_grid):
    # A second at 1 N on one spot, on the grid and off it (the same sub-cell position): the same visible area.
    areas = []
    for u in (0.0, 0.10):
        layer = _ink(small_grid)
        for _ in range(20):  # 10 sim steps a control step
            _press(layer, [[u + 0.0005, -0.0005]] * 10, 1.0)
        areas.append(_visible_area(layer))
    on_grid, off_grid = areas
    assert 5e-6 < off_grid == pytest.approx(on_grid, rel=1e-4)
    # Lifted and put down on the same spot, it presses on the same, saturated mark: tapping adds nothing.
    for _ in range(5):
        _press(layer, [[0.1005, -0.0005]], 0.0)
        _press(layer, [[0.1005, -0.0005]], 1.0)
    assert float(layer.outside_m2[0]) == pytest.approx(off_grid, rel=1e-6)


def test_light_taps_off_the_grid_darken_one_mark_as_they_would_on_it(small_grid):
    # Each 0.1 N tap lays a faint footprint; repeated on one spot they darken one mark, not add new ones.
    areas = []
    for u in (0.0, 0.10):
        layer = _ink(small_grid)
        for _ in range(8):
            _press(layer, [[u + 0.0005, -0.0005]] * 2, 0.1)
            _press(layer, [[u + 0.0005, -0.0005]] * 3, 0.0)
        areas.append(_visible_area(layer))
    on_grid, off_grid = areas
    assert on_grid > 2e-6
    assert off_grid == pytest.approx(on_grid, rel=1e-4)


def test_a_fast_stroke_off_the_grid_within_one_control_step_counts_its_area():
    # 40 mm in one control step's 10 samples (0.8 m/s): the patch, re-centred midway along it, holds it all.
    grid = BoardGrid(u0=-0.03, v0=0.01, cell=CELL, rows=20, cols=60)
    path = np.c_[np.linspace(-0.02, 0.02, 10), np.full(10, -0.0005)]
    areas = []
    for v in (0.0, 0.5):
        layer = _ink(grid)
        _press(layer, path + [0.0, v], 1.0, _axes(layer, nib_x=(0.0, 1.0), nib_y=(1.0, 0.0), samples=10), dt=0.05)
        _press(layer, path + [0.0, v], 1.0, _axes(layer, nib_x=(0.0, 1.0), nib_y=(1.0, 0.0), samples=10), dt=0.05)
        areas.append(_visible_area(layer))
    on_grid, off_grid = areas
    assert off_grid == pytest.approx(on_grid, rel=1e-3)


def test_a_chisel_wider_than_the_off_grid_patch_still_counts_its_mark_once():
    # The patch is at least twice the footprint's reach, so even a 10 cm chisel at rest fits in it.
    grid = BoardGrid(u0=-0.06, v0=0.06, cell=CELL, rows=120, cols=120)
    areas = []
    for u in (0.0, 0.5):
        layer = _ink(grid, semi_axes=(0.05, 0.002))
        for _ in range(3):
            _press(layer, [[u + 0.0005, -0.0005]] * 2, 1.0)
        areas.append(_visible_area(layer))
    on_grid, off_grid = areas
    assert off_grid == pytest.approx(on_grid, rel=1e-4)


def test_a_light_press_off_the_grid_darkens_as_it_would_on_it(small_grid):
    # 0.01 N lays too little pigment to show in one sim step (1 - exp(-0.0175) < 0.02); held, it builds up to show.
    areas = []
    for u in (0.0, 0.10):
        layer = _ink(small_grid)
        assert float(_press(layer, [[u + 0.0005, -0.0005]], 0.01).weight.max()) == 0.0
        for _ in range(40):
            _press(layer, [[u + 0.0005, -0.0005]] * 10, 0.01)
        areas.append(_visible_area(layer))
    on_grid, off_grid = areas
    assert on_grid > 5e-6
    assert off_grid == pytest.approx(on_grid, rel=1e-4)


def test_ink_off_the_grid_is_forgotten_once_the_nib_moves_far_from_it(small_grid):
    def two_presses(hover_u=None):
        layer = _ink(small_grid)
        _press(layer, [[0.10, 0.0]], 1.0)
        if hover_u is not None:
            _press(layer, [[hover_u, 0.0]], 0.0)
        _press(layer, [[0.10, 0.0]], 1.0)
        return float(layer.outside_m2[0])

    layer = _ink(small_grid)
    _press(layer, [[0.10, 0.0]], 1.0)
    once = float(layer.outside_m2[0])
    # Hovering within the patch keeps the mark: the second press darkens it as if the nib had stayed.
    assert two_presses(0.10 + 0.5 * OFF_GRID_PATCH) == pytest.approx(two_presses(), rel=1e-6)
    # Beyond it the mark is dropped, and the second press counts a fresh one.
    assert two_presses(0.10 + 2.0 * OFF_GRID_PATCH) == pytest.approx(2 * once, rel=1e-6)


@pytest.mark.parametrize("nib_x", [(1.0, 0.0), (0.0, 1.0), (math.sqrt(0.5), math.sqrt(0.5))])
def test_a_stroke_off_the_grid_counts_the_area_it_would_ink_on_it(nib_x):
    # The same 30 mm stroke at 5 cm/s, once across a grid and once 0.5 m off it: the same law in both, so the same
    # area (up to the float rounding of the footprint's sub-cell position).
    grid = BoardGrid(u0=-0.03, v0=0.02, cell=CELL, rows=40, cols=60)
    nib_y = (-nib_x[1], nib_x[0])
    v = np.linspace(-0.015, 0.015, 121)
    areas = []
    for u in (0.0, 0.5):
        layer = _ink(grid)
        for chunk in np.split(np.c_[np.full(120, u), v[1:]], 12):
            _press(layer, chunk, 1.0, _axes(layer, nib_x=nib_x, nib_y=nib_y, samples=10))
        areas.append(_visible_area(layer))
    on_grid, off_grid = areas
    assert off_grid == pytest.approx(on_grid, rel=0.01)


def test_visible_weight_is_the_originals_ramp():
    torch.testing.assert_close(
        visible_weight(torch.tensor([0.0, 0.02, 0.085, 0.15, 0.9])), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    )


def _scipy_f1(target: np.ndarray, actual: np.ndarray, radius: int) -> float:
    """The original verifier's ``f1`` (``task_set/verifier.py``), on scipy's maximum filter."""
    size = 2 * radius + 1
    precision = (actual * maximum_filter(target.astype(float), size=size)).sum() / max(actual.sum(), 1e-12)
    recall = (maximum_filter(actual, size=size) * target).sum() / target.sum()
    return 2 * precision * recall / max(precision + recall, 1e-12)


@pytest.mark.parametrize("shift", [0, 1, 2, 3, 6])
def test_lettering_f1_matches_the_originals(shift):
    strokes = DRAWINGS["hello"]
    grid = BoardGrid.around(strokes, margin=0.02, cell=CELL)
    judge = TextJudge(grid, strokes, 1, "cpu")
    target = grid.raster(strokes)
    # Ink as drawn a few millimetres low, with a fainter half: weights both inside and outside the 2 mm window.
    actual = np.roll(target, shift, axis=0).astype(np.float32)
    actual[:, : grid.cols // 2] *= 0.5
    ink = _ink(grid, stray_map=judge.stray_map())
    judge.update(ink.paint(torch.as_tensor(actual)[None]))
    assert float(judge.f1(ink)[0]) == pytest.approx(_scipy_f1(target, actual, judge.radius), rel=1e-5)


def test_ink_off_the_grid_lowers_the_lettering_precision():
    strokes = DRAWINGS["hello"]
    grid = BoardGrid.around(strokes, margin=0.02, cell=CELL)
    judge = TextJudge(grid, strokes, 1, "cpu")
    perfect = torch.as_tensor(grid.raster(strokes).astype(np.float32))[None]
    ink = _ink(grid, stray_map=judge.stray_map())
    judge.update(ink.paint(perfect))
    assert float(judge.f1(ink)[0]) == pytest.approx(1.0)
    ink.outside_m2 += float(perfect.sum()) * CELL**2  # as much ink again, off the grid
    assert float(judge.f1(ink)[0]) == pytest.approx(2 * 0.5 / 1.5, rel=1e-4)  # precision halved, recall 1


def test_incremental_grading_matches_a_full_regrade():
    # Random strokes laid sample by sample: the running totals, the F1 and the guide's latches equal a recount.
    strokes = DRAWINGS["hello"]
    grid = BoardGrid.around(strokes, margin=0.02, cell=CELL)
    judge = TextJudge(grid, strokes, 2, "cpu")
    guide = StrokeGuide(grid, strokes, 2, spacing=0.002, device="cpu")
    ink = InkLayer(
        grid, 2, semi_axes=(0.0025, 0.001), rate=350.0, force_cap=1.0, stray_map=judge.stray_map(), device="cpu"
    )
    generator = torch.Generator().manual_seed(0)
    uv = (torch.rand(2, 2, generator=generator) - 0.5) * torch.tensor([0.5, 0.08])
    axes = ink.footprint(torch.tensor([[0.7, 0.7]]).expand(2, 10, 2), torch.tensor([[-0.7, 0.7]]).expand(2, 10, 2))
    for _ in range(60):
        path = uv[:, None] + torch.cumsum(torch.randn(2, 10, 2, generator=generator) * 0.001, dim=1)
        force = (torch.rand(2, 1, generator=generator) > 0.3).float().expand(2, 10)
        change = ink.deposit(path, force, axes, SIM_DT)
        judge.update(change)
        guide.update(change)
        uv = path[:, -1]
    weight = ink.weight
    torch.testing.assert_close(ink.total, weight.sum(dim=(1, 2)), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(ink.stray_cells, (weight * judge.stray_map()).sum(dim=(1, 2)), rtol=1e-4, atol=1e-3)
    target = grid.raster(strokes)
    for env in range(2):
        expected = _scipy_f1(target, weight[env].numpy(), judge.radius)
        assert float(judge.f1(ink)[env]) == pytest.approx(expected, rel=1e-4)
    near = torch.as_tensor(maximum_filter(weight.numpy(), size=(1, 5, 5)) >= 0.5)
    rows_cols = torch.as_tensor(grid.cell_of(guide.points.numpy()))
    torch.testing.assert_close(guide.covered, near[:, rows_cols[:, 0], rows_cols[:, 1]])
    assert bool(guide.covered.any())


@pytest.fixture(scope="module")
def table():
    strokes = DRAWINGS["table"]
    grid = BoardGrid.around(strokes, margin=0.03, cell=CELL)
    return grid, TableJudge(grid, strokes, "cpu"), torch.as_tensor(grid.raster(strokes).astype(np.float32))


def _judge(table, weight):
    grid, judge, _ = table
    ink = _ink(grid, stray_map=judge.stray_map())
    ink.paint(weight[None])
    return judge.judge(ink)


def _erase(grid, weight, u, v, half_u, half_v):
    out = weight.clone()
    rows = slice(int((grid.v0 - v - half_v) / CELL), int((grid.v0 - v + half_v) / CELL) + 1)
    cols = slice(int((u - half_u - grid.u0) / CELL), int((u + half_u - grid.u0) / CELL) + 1)
    out[rows, cols] = 0.0
    return out


def test_the_reference_table_passes(table):
    result = _judge(table, table[2])
    assert bool(result["success"][0]) and float(result["progress"][0]) == 1.0
    assert bool((result["coverage"] >= 0.99).all())


@pytest.mark.parametrize(("gap_mm", "passes"), [(8, True), (14, False)])
def test_a_gap_in_a_line_fails_it_beyond_10_mm(table, gap_mm, passes):
    grid, _, drawn = table
    header_v = 0.12
    skipped = _erase(grid, drawn, 0.0, header_v, gap_mm / 2000 - 0.0005, 0.004)
    result = _judge(table, skipped)
    header = 4
    assert float(result["gap"][0, header]) == pytest.approx(gap_mm / 1000, abs=0.0015)
    assert bool(result["success"][0]) is passes


def test_stray_ink_fails_the_table_beyond_1_cm2(table):
    grid, _, drawn = table
    # A 2 x 1 cm blot in the middle of the lower-left cell, far from every line.
    blotted = drawn.clone()
    r0, c0 = int((grid.v0 + 0.08) / CELL), int((-0.20 - grid.u0) / CELL)
    blotted[r0 : r0 + 10, c0 : c0 + 20] = 1.0
    result = _judge(table, blotted)
    assert float(result["stray_m2"][0]) == pytest.approx(2e-4, rel=0.02)
    assert not bool(result["success"][0]) and 0.9 < float(result["progress"][0]) < 1.0
    small = drawn.clone()
    small[r0 : r0 + 5, c0 : c0 + 10] = 1.0  # 0.5 cm^2
    assert bool(_judge(table, small)["success"][0])


def test_ink_beside_a_line_is_not_that_line(table):
    # The header drawn 8 mm low is beyond the 6 mm reading window: no coverage, and it is stray ink.
    grid, _, drawn = table
    without = _erase(grid, drawn, 0.0, 0.12, 0.29, 0.002)
    rows = slice(int((grid.v0 - 0.112) / CELL) - 1, int((grid.v0 - 0.112) / CELL) + 2)
    cols = slice(int((-0.29 - grid.u0) / CELL), int((0.29 - grid.u0) / CELL))
    without[rows, cols] = 1.0
    result = _judge(table, without)
    assert float(result["coverage"][0, 4]) < 0.05
    assert float(result["stray_m2"][0]) > 1e-4
    assert LINE_WIDTH["normal"] < 0.008


def test_the_guide_latches_inked_points_and_points_at_the_first_uninked():
    strokes = DRAWINGS["table"]
    grid = BoardGrid.around(strokes, margin=0.03, cell=CELL)
    guide = StrokeGuide(grid, strokes, 1, spacing=0.002, device="cpu")
    ink = _ink(grid)
    top = torch.as_tensor(grid.raster(strokes[:1]).astype(np.float32))[None]
    guide.update(ink.paint(top))
    first_uninked = int(guide.next_index()[0])
    assert first_uninked == int((guide.stroke == 0).sum())  # the whole top line is in: the bottom line's start is next
    # The first gain is everything inked: the top line, and the side lines' ends where they meet it.
    assert float(guide.newly_covered[0]) == pytest.approx(float(guide.fraction_covered[0]))
    assert float(guide.fraction_covered[0]) > first_uninked / guide.num_points
    guide.update(ink.paint(torch.zeros_like(top)))  # nothing new: no gain
    assert int(guide.next_index()[0]) == first_uninked and float(guide.newly_covered[0]) == 0.0
