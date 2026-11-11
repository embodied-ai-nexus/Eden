"""Tests for what a painting is: its dips, the tip's path through its plan, its target, and its plan as data."""

from types import MappingProxyType

import numpy as np
import pytest
import torch

from eden.tasks.art.painting.paint import PaintLoad, RoundBrush
from eden.tasks.art.painting.palette import DIP_DEPTH
from eden.tasks.art.painting.references import (
    Painting,
    PaintStroke,
    load_painting,
    render_painting,
    save_painting,
    sweep,
    swept_area,
)
from eden.tasks.art.plans import STOP


def _sweep(u0: float, v: float, length: float, size: float, paint: int) -> PaintStroke:
    return sweep("sweep", paint, v, u0, u0 + length, size)


def _painting(*strokes: PaintStroke) -> Painting:
    return Painting("test", "test", palette=((0.9, 0.1, 0.1), (0.1, 0.1, 0.9)), strokes=strokes, sheet=(0.1, 0.1))


def test_the_brush_is_dipped_for_each_new_paint_and_before_it_would_run_low():
    brush, load = RoundBrush(), PaintLoad()
    painting = _painting(
        _sweep(-0.03, 0.02, 0.06, 0.016, 0),
        _sweep(-0.03, 0.0, 0.06, 0.016, 0),
        _sweep(-0.03, -0.02, 0.01, 0.016, 1),
        _sweep(-0.03, -0.03, 0.01, 0.016, 1),
    )
    # A 16 mm sweep 6 cm long uses over a third of a full load (6 * 1.6 / 30 = 0.32, and its touch-down and lift-off):
    # one leaves the brush above the 0.4 reserve, two would not.
    used = float(swept_area(painting.strokes[0].stroke.path(), brush).sum()) / load.capacity
    assert 1.0 - 2 * used < load.reserve < 1.0 - used
    assert painting.dips() == (True, True, True, False)


def test_a_dip_goes_into_its_paint_s_well_on_the_way_to_the_stroke():
    painting = _painting(_sweep(-0.03, 0.0, 0.01, 0.005, 1))
    path, owner = painting.path()
    dip = path[owner == STOP]
    np.testing.assert_allclose(dip[:, :2], np.broadcast_to(painting.wells[1], dip[:, :2].shape), atol=1e-12)
    assert dip[:, 2].max() == pytest.approx(DIP_DEPTH)
    assert np.flatnonzero(owner == STOP).max() < np.flatnonzero(owner == 0).min()


def test_the_target_is_the_plan_painted_and_painting_it_twice_gives_the_same():
    painting = _painting(_sweep(-0.03, 0.0, 0.04, 0.005, 0))
    a, b = render_painting(painting), render_painting(painting)
    assert torch.equal(a.rgb, b.rgb) and torch.equal(a.painted, b.painted)
    row, col = painting.grid.cell_of([0.0, 0.0])
    torch.testing.assert_close(a.rgb[row, col], torch.tensor(painting.palette[0]))
    # The sweep, and the way its tip comes down onto the paper and lifts off it.
    assert 0.04 <= a.path_length <= 0.045 and a.dips == 1


def test_a_stroke_in_a_paint_the_palette_lacks_is_refused():
    with pytest.raises(ValueError, match=r"strokes use paints \[0, 2\], but the palette has 2 \(indices 0-1\)"):
        _painting(_sweep(0.0, 0.0, 0.01, 0.005, 0), _sweep(0.0, 0.0, 0.01, 0.005, 2))


def test_a_painting_survives_its_data_file(tmp_path):
    painting = _painting(_sweep(-0.03, 0.0, 0.04, 0.005, 0), _sweep(-0.03, 0.01, 0.02, 0.005, 1))
    save_painting(painting, tmp_path / "test.json")
    loaded = load_painting(tmp_path / "test.json", rule=MappingProxyType({"de_mean": 1.0}))
    assert loaded.palette == painting.palette and loaded.sheet == painting.sheet
    assert [(s.paint, s.stroke.points) for s in loaded.strokes] == [
        (s.paint, s.stroke.points) for s in painting.strokes
    ]
    assert loaded.rule["de_mean"] == 1.0
