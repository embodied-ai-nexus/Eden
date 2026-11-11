"""Tests for RoboPaint's picture rules for paintings: colour, structure, edges and coverage, in a crop round the
target.
"""

import pytest
import torch

from eden.tasks.art.painting.grading import PaintingJudge, PaintingRule, gaussian_kernel, srgb_to_lab
from eden.tasks.art.workcell import PAPER_COLOR, sheet_grid

GRID = sheet_grid((0.04, 0.04), 0.0005)  # 80 x 80 cells
PAPER = torch.tensor(PAPER_COLOR)
BLUE = torch.tensor((0.1, 0.2, 0.6))


def _target():
    """Bare paper with a blue square in its middle (rows and columns 30-49)."""
    rgb = PAPER.expand(*GRID.shape, 3).clone()
    painted = torch.zeros(GRID.shape, dtype=torch.bool)
    rgb[30:50, 30:50], painted[30:50, 30:50] = BLUE, True
    return rgb, painted


@pytest.fixture
def judge():
    rgb, painted = _target()
    return PaintingJudge(GRID, rgb, painted, PaintingRule(), device="cpu")


def test_lab_is_cielab_d65():
    lab = srgb_to_lab(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    torch.testing.assert_close(lab[0], torch.tensor([100.0, 0.0, 0.0]), atol=0.01, rtol=0)
    torch.testing.assert_close(lab[1], torch.tensor([0.0, 0.0, 0.0]), atol=0.01, rtol=0)
    torch.testing.assert_close(lab[2], torch.tensor([53.24, 80.09, 67.20]), atol=0.05, rtol=0)


def test_the_target_itself_passes_every_rule(judge):
    rgb, painted = _target()
    measures = judge.judge(rgb[None], painted[None])
    assert float(measures["de_mean"]) == pytest.approx(0.0, abs=1e-3)
    assert float(measures["ssim_l"]) == pytest.approx(1.0, abs=1e-4)
    assert float(measures["edge_corr"]) == pytest.approx(1.0, abs=1e-4)
    assert float(measures["paint_iou"]) == 1.0 and bool(measures["passed"])


def test_a_bare_sheet_fails_on_colour_and_coverage(judge):
    rgb, painted = PAPER.expand(1, *GRID.shape, 3), torch.zeros(1, *GRID.shape, dtype=torch.bool)
    measures = judge.judge(rgb, painted)
    assert float(measures["de_mean"]) > 10.0 and float(measures["paint_iou"]) == 0.0
    assert not bool(measures["passed"])


def test_the_colour_is_read_round_the_target_and_coverage_over_the_sheet(judge):
    """A stray stroke far from the target is out of the colour's crop, but costs coverage: paint where the target
    has none.
    """
    assert judge.crop == (slice(20, 60), slice(20, 60))  # the square, 5 mm (10 cells) either way
    rgb, painted = _target()
    rgb, painted = rgb.clone(), painted.clone()
    rgb[0:10, 0:20], painted[0:10, 0:20] = BLUE, True
    measures = judge.judge(rgb[None], painted[None])
    assert float(measures["de_mean"]) == pytest.approx(0.0, abs=1e-3)
    assert float(measures["paint_iou"]) == pytest.approx(400 / 600)
    assert not bool(measures["passed"])


def test_a_copy_off_by_a_little_still_reads_close_and_one_off_by_a_lot_does_not(judge):
    rgb, painted = _target()
    near_rgb, near = PAPER.expand(*GRID.shape, 3).clone(), torch.zeros(GRID.shape, dtype=torch.bool)
    near_rgb[31:51, 30:50], near[31:51, 30:50] = BLUE, True  # half a millimetre down
    far_rgb, far = PAPER.expand(*GRID.shape, 3).clone(), torch.zeros(GRID.shape, dtype=torch.bool)
    far_rgb[40:60, 30:50], far[40:60, 30:50] = BLUE, True  # 5 mm down
    measures = judge.judge(torch.stack([near_rgb, far_rgb]), torch.stack([near, far]))
    assert float(measures["ssim_l"][0]) > float(measures["ssim_l"][1])
    assert float(measures["de_mean"][0]) < float(measures["de_mean"][1])
    assert float(measures["paint_iou"][0]) == pytest.approx(380 / 420)


def test_a_target_with_no_paint_is_refused():
    with pytest.raises(ValueError, match="the target has no painted cell"):
        PaintingJudge(
            GRID, PAPER.expand(*GRID.shape, 3), torch.zeros(GRID.shape, dtype=torch.bool), PaintingRule(), device="cpu"
        )


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"blur": 0.0}, r"PaintingRule: blur=0\.0; it is a standard deviation \(m\) and must be positive"),
        ({"ssim_l": 1.01}, r"PaintingRule: ssim_l=1\.01; it is at most 1 for any copy"),
    ],
)
def test_a_rule_no_copy_could_pass_is_refused(rule, message):
    with pytest.raises(ValueError, match=message):
        PaintingRule(**rule)


def test_a_gaussian_of_no_width_is_refused():
    with pytest.raises(ValueError, match=r"gaussian_kernel: sigma=0\.0 cells; .* must be positive"):
        gaussian_kernel(0.0, "cpu")
