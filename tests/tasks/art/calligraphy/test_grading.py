"""Tests for the sheet's ink and RoboPaint's picture rules."""

import numpy as np
import pytest
import torch

from eden.tasks.art.calligraphy.grading import SheetInk, WritingJudge, WritingRule
from eden.tasks.art.calligraphy.tools import Footprint, PointedPen
from eden.tasks.art.workcell import sheet_grid

CELL = 0.0005


def _dot(u, v, width=0.002, live=True):
    """A pen-shaped (elliptic) mark ``width`` across, one env and one sample."""
    return Footprint(
        torch.tensor([[[u, v]]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[width]]),
        torch.tensor([[width]]),
        torch.tensor([[live]]),
    )


@pytest.fixture
def ink():
    return SheetInk(sheet_grid((0.02, 0.02), CELL), 1, PointedPen(max_width=0.004), device="cpu")


def test_a_mark_inks_the_cells_it_covers_and_a_dead_one_nothing(ink):
    ink.deposit(_dot(0.0, 0.0, live=False))
    assert int(ink.inked.sum()) == 0
    ink.deposit(_dot(0.0, 0.0))
    inked = ink.inked[0]
    rows, cols = torch.nonzero(inked, as_tuple=True)
    # A 2 mm disc round the sheet's centre: a radius of 2 cells, about 4 pi cells.
    assert int(inked.sum()) == pytest.approx(np.pi * 2**2, rel=0.3)
    assert int(rows.min()) >= 20 - 3 and int(rows.max()) <= 19 + 3 and int(cols.min()) >= 20 - 3


def test_ink_once_laid_stays(ink):
    ink.deposit(_dot(0.0, 0.0))
    before = ink.inked.clone()
    ink.deposit(_dot(0.0, 0.0, width=0.001, live=False))  # a later mark, inked or not, takes nothing away
    ink.deposit(_dot(0.0, 0.0, width=0.001))
    assert torch.equal(ink.inked, before)


def test_a_mark_off_the_sheet_inks_nothing(ink):
    ink.deposit(_dot(0.05, 0.0))
    assert int(ink._ink.sum()) == 0  # the pad (where its cells go) is cleared too


def _square(grid, centre_rc, half):
    image = np.zeros(grid.shape, dtype=bool)
    r, c = centre_rc
    image[r - half : r + half, c - half : c + half] = True
    return image


@pytest.fixture
def judge_setup():
    grid = sheet_grid((0.04, 0.02), CELL)  # 40 x 80 cells
    left, right = _square(grid, (20, 20), 6), _square(grid, (20, 60), 6)
    owner = np.full(grid.shape, -1)
    owner[left], owner[right] = 0, 1
    return grid, left | right, owner


def test_the_target_itself_passes_every_rule(judge_setup):
    grid, target, owner = judge_setup
    judge = WritingJudge(grid, target, owner, WritingRule(), device="cpu")
    out = judge.judge(torch.as_tensor(target)[None])
    assert float(out["iou"][0]) == 1.0 and float(out["boundary_f"][0]) == 1.0 and float(out["stray_frac"][0]) == 0.0
    assert out["stroke_coverage"][0].tolist() == [1.0, 1.0] and bool(out["passed"][0])


def test_a_missing_stroke_fails_on_its_coverage(judge_setup):
    grid, target, owner = judge_setup
    judge = WritingJudge(grid, target, owner, WritingRule(iou=0.4, boundary_f=0.4), device="cpu")
    out = judge.judge(torch.as_tensor(target & (owner == 0))[None])
    assert out["stroke_coverage"][0].tolist() == [1.0, 0.0]
    assert float(out["iou"][0]) == pytest.approx(0.5) and not bool(out["passed"][0])


def test_ink_beyond_the_tolerance_is_stray_and_shifts_the_boundary(judge_setup):
    grid, target, owner = judge_setup
    judge = WritingJudge(grid, target, owner, WritingRule(), device="cpu")
    blot = target.copy()
    blot[2:6, 36:44] = True  # 32 cells, far from both squares
    out = judge.judge(torch.as_tensor(blot)[None])
    assert float(out["stray_frac"][0]) == pytest.approx(32 / target.sum())
    shifted = np.roll(target, 4, axis=1)  # 2 mm: twice the boundary tolerance
    out = judge.judge(torch.as_tensor(np.stack([target, shifted])))
    assert float(out["boundary_f"][1]) < float(out["boundary_f"][0]) == 1.0
    assert not bool(out["passed"][1])


def test_ink_running_on_past_the_crop_has_no_outline_at_its_edge(judge_setup):
    """The boundary F is read in a crop round the target. A stroke that runs on past the crop's edge has its sides
    there but no end, so it must score better than the same stroke that really ends at that edge.
    """
    grid, target, owner = judge_setup
    judge = WritingJudge(grid, target, owner, WritingRule(), device="cpu")
    top = judge._crop[0].start
    assert top > 0, "test premise: the sheet goes on above the crop"
    through, ending = target.copy(), target.copy()
    through[:15, 18:23] = True  # up from the left square, off the top of the crop
    ending[top:15, 18:23] = True  # the same stroke, ending on the crop's first row
    out = judge.judge(torch.as_tensor(np.stack([through, ending])))
    assert float(out["boundary_f"][0]) > float(out["boundary_f"][1])


def test_a_target_with_no_ink_or_a_stroke_without_cells_is_refused(judge_setup):
    grid, target, owner = judge_setup
    with pytest.raises(ValueError, match=r"the target has no inked cell"):
        WritingJudge(grid, np.zeros_like(target), owner, WritingRule(), device="cpu")
    orphan = owner.copy()
    orphan[owner == 1] = 2  # stroke 1 owns nothing
    with pytest.raises(ValueError, match=r"strokes \[1\] own no target cell"):
        WritingJudge(grid, target, orphan, WritingRule(), device="cpu")
