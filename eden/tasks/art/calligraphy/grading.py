"""Ink on the sheet and how a written work is graded, batched in torch.

* :class:`SheetInk` — the ink per env on a grid over the whole sheet: the union of every footprint a tool laid
  (:mod:`~eden.tasks.art.calligraphy.tools`). A cell is inked once a footprint covers its centre. Ink is opaque and the
  brush never runs dry (RoboPaint counts dips into the ink; nothing here makes the brush need one). Ink only exists
  on the sheet: a tool off it marks nothing.
* :class:`WritingJudge` — RoboPaint's picture rules for its writing works, judged at the placement shown (RoboPaint
  first registers the copy to the target within 5 mm; here the placement must match, which is stricter): the IoU of
  the ink with the target, the boundary F-measure, every stroke of the target covered, and little ink off the
  target. The hand rules (strokes, path, mark width) are counted by the metric
  (:class:`~eden.tasks.art.calligraphy.custom_terms.CalligraphyWriting`) as the tool moves.

Unlike the whiteboard's grading, the judge re-reads the sheet each control step: the boundary F-measure needs the
ink's outline, which is not local to the cells a step touched. That is one count over the whole sheet (the ink's
area) and a few boolean passes over a crop round the target (the outline, and three counts); everything about the
target (its cells, its outline's neighbourhoods) is gathered at the target's own cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from eden.tasks.art.calligraphy.tools import Footprint, WritingTool, tool_reach_cells
from eden.utils.plane_grid import PlaneGrid

#: Footprint cells evaluated per chunk of a deposit, bounding its memory.
_CHUNK_CELLS = 1 << 24
#: The ink's colour (RGB in ``[0, 1]``), sumi for the brush and iron gall for the pen alike: what a camera paints the
#: metric's ink in. Grading never reads it.
INK_COLOR = (0.06, 0.055, 0.07)


class SheetInk:
    """Per-env ink on a :class:`~eden.utils.plane_grid.PlaneGrid` over the sheet, laid by ``tool``.

    The ink is stored flat, a byte a cell, with a pad cell at the end (index ``rows * cols``) where a footprint's
    cells off the sheet are addressed; it is cleared after every deposit.
    """

    def __init__(self, grid: PlaneGrid, num_envs: int, tool: WritingTool, *, device) -> None:
        self.grid, self.tool = grid, tool
        self.pad = grid.rows * grid.cols
        self._ink = torch.zeros(num_envs, self.pad + 1, dtype=torch.uint8, device=device)
        reach = tool_reach_cells(tool, grid.cell)
        dr, dc = torch.meshgrid(
            torch.arange(-reach, reach + 1, device=device),
            torch.arange(-reach, reach + 1, device=device),
            indexing="ij",
        )
        self._window = torch.stack([dr.reshape(-1), dc.reshape(-1)], dim=-1)  # (W, 2): row, col offsets

    @property
    def inked(self) -> torch.Tensor:
        """``(num_envs, rows, cols)`` (bool): the cells inked. A view: it changes with the next deposit."""
        return self._ink[:, : self.pad].view(torch.bool).view(-1, *self.grid.shape)

    def deposit(self, footprints: Footprint) -> None:
        """Lay ``footprints`` (every field ``(B, S, ...)``): ink every cell whose centre one of them covers."""
        batch, samples = footprints.live.shape
        chunk = max(1, _CHUNK_CELLS // (batch * self._window.shape[0]))
        for start in range(0, samples, chunk):
            part = Footprint(
                *(getattr(footprints, f)[:, start : start + chunk] for f in Footprint.__dataclass_fields__)
            )
            index, cover = self._cover(part)
            self._ink.scatter_reduce_(1, index.flatten(1), cover.flatten(1), reduce="amax")
        self._ink[:, self.pad] = 0

    def _cover(self, fp: Footprint) -> tuple[torch.Tensor, torch.Tensor]:
        """The cells round each footprint ``(B, S, W)`` (flat, the pad off the sheet) and whether it covers each."""
        g = self.grid
        centre = self.tool.centre(fp)  # (B, S, 2)
        row = torch.floor((g.v0 - centre[..., 1]) / g.cell).long()
        col = torch.floor((centre[..., 0] - g.u0) / g.cell).long()
        rr = row[..., None] + self._window[:, 0]  # (B, S, W)
        cc = col[..., None] + self._window[:, 1]
        on_sheet = (rr >= 0) & (rr < g.rows) & (cc >= 0) & (cc < g.cols)
        # Each cell's centre relative to the footprint's root, in its frame (s along, t across).
        du = g.u0 + (cc.float() + 0.5) * g.cell - fp.root[..., 0, None]
        dv = g.v0 - (rr.float() + 0.5) * g.cell - fp.root[..., 1, None]
        au, av = fp.along[..., 0, None], fp.along[..., 1, None]
        inside = self.tool.inside(du * au + dv * av, dv * au - du * av, fp.length[..., None], fp.width[..., None])
        return torch.where(on_sheet, rr * g.cols + cc, self.pad), (inside & fp.live[..., None]).to(torch.uint8)

    def paint(self, mask: torch.Tensor) -> None:
        """Ink every cell of ``mask (num_envs, rows, cols)`` (bool), with no tool: for setting up a sheet (tests, a
        pre-written start). Ink only grows, so cells already inked stay inked.
        """
        self._ink[:, : self.pad] |= mask.reshape(mask.shape[0], -1).to(torch.uint8)

    def reset(self, envs_idx) -> None:
        self._ink[envs_idx] = 0


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``mask (rows, cols)`` (bool) grown by the ``(2 radius + 1)^2`` square (for the target, once)."""
    if radius <= 0:
        return mask
    flat = mask[None, None].float()
    return F.max_pool2d(flat, 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0.0


def _outline(mask: torch.Tensor) -> torch.Tensor:
    """The cells of ``mask (..., rows, cols)`` (bool) with a 4-neighbour outside it (off the grid counts outside)."""
    interior = mask.clone()
    interior[..., 1:, :] &= mask[..., :-1, :]
    interior[..., :-1, :] &= mask[..., 1:, :]
    interior[..., :, 1:] &= mask[..., :, :-1]
    interior[..., :, :-1] &= mask[..., :, 1:]
    # The edge rows and columns by slice: indexed by a tuple, each write would copy the index to the GPU, a sync.
    interior[..., :1, :] = False
    interior[..., -1:, :] = False
    interior[..., :, :1] = False
    interior[..., :, -1:] = False
    return mask & ~interior


@dataclass(frozen=True)
class WritingRule:
    """RoboPaint's picture thresholds for a writing work (its ``rule``), and the tolerances they are read with."""

    iou: float = 0.8
    boundary_f: float = 0.75
    min_stroke_coverage: float = 0.8
    max_stray_frac: float = 0.04
    #: A boundary cell of the ink and one of the target match within this distance (m).
    boundary_tolerance: float = 0.001
    #: Ink this far (m) or less from the target is not stray.
    stray_tolerance: float = 0.001


def _count(mask: torch.Tensor) -> torch.Tensor:
    """The cells set in each env's ``mask (B, ...)`` (bool), as float: summed as bytes (a bool sum widens to int64)."""
    return mask.reshape(mask.shape[0], -1).view(torch.uint8).sum(dim=1, dtype=torch.int32).float()


class WritingJudge:
    """The picture rules of :class:`WritingRule` for ``target`` on ``grid``, every env at once.

    ``target (rows, cols)`` (bool) is the reference's inked cells and ``owner (rows, cols)`` (int) the stroke each of
    them belongs to (``-1`` elsewhere), from which each stroke's coverage is read.

    The boundary F-measure matches outline cells within ``boundary_tolerance`` (a square of cells): its precision is
    the share of the ink's outline near the target's, its recall the share of the target's outline with ink's
    outline near it. It is read in a crop of the sheet round the target (``crop_margin`` beyond it either way): ink
    farther out is stray ink, which the stray rule charges, and its outline does not count against the precision.
    """

    def __init__(
        self,
        grid: PlaneGrid,
        target: np.ndarray,
        owner: np.ndarray,
        rule: WritingRule,
        *,
        device,
        crop_margin: float = 0.005,
    ) -> None:
        self.grid, self.rule = grid, rule
        target_t = torch.as_tensor(target, dtype=torch.bool, device=device)
        self.target = target_t
        self.target_area = int(target_t.sum())
        if self.target_area == 0:
            raise ValueError("WritingJudge: the target has no inked cell; render the reference on this grid first.")
        self.num_strokes = int(owner.max()) + 1
        flat_target = torch.nonzero(target_t.reshape(-1)).flatten()
        self._target_cells = flat_target  # (m,)
        owner_t = torch.as_tensor(owner, dtype=torch.long, device=device).reshape(-1)[flat_target]
        if bool((owner_t < 0).any()):
            raise ValueError("WritingJudge: every target cell needs an owning stroke (owner >= 0 on the target).")
        self._target_owner = owner_t
        self._stroke_area = torch.bincount(owner_t, minlength=self.num_strokes).float()
        if bool((self._stroke_area == 0).any()):
            missing = torch.nonzero(self._stroke_area == 0).flatten().tolist()
            raise ValueError(f"WritingJudge: strokes {missing} own no target cell; every stroke must leave ink.")
        radius = max(1, round(rule.boundary_tolerance / grid.cell))
        stray_radius = round(rule.stray_tolerance / grid.cell)
        # The crop: the target's rows and columns, and the margin (at least both tolerances) round them.
        margin = max(math.ceil(crop_margin / grid.cell), radius, stray_radius) + 1
        rows_any, cols_any = np.flatnonzero(target.any(axis=1)), np.flatnonzero(target.any(axis=0))
        r0, r1 = max(0, rows_any[0] - margin), min(grid.rows, rows_any[-1] + margin + 1)
        c0, c1 = max(0, cols_any[0] - margin), min(grid.cols, cols_any[-1] + margin + 1)
        self._crop = (slice(r0, r1), slice(c0, c1))
        # The ink's outline is read on the crop and one cell round it (where the sheet has one), then trimmed, so ink
        # running on past the crop's edge has no outline there: only the sheet's own edge is outside.
        pr0, pr1, pc0, pc1 = max(0, r0 - 1), min(grid.rows, r1 + 1), max(0, c0 - 1), min(grid.cols, c1 + 1)
        self._outline_crop = (slice(pr0, pr1), slice(pc0, pc1))
        self._outline_trim = (slice(r0 - pr0, r0 - pr0 + r1 - r0), slice(c0 - pc0, c0 - pc0 + c1 - c0))
        target_crop = target_t[self._crop]
        target_outline = _outline(target_crop)
        self._near_target_outline = _dilate(target_outline, radius)
        self._near_target = _dilate(target_crop, stray_radius)
        # Each target-outline cell's neighbourhood, flat indices into the crop plus a pad slot (always outside).
        rows, cols = target_crop.shape
        r, c = torch.nonzero(target_outline, as_tuple=True)
        d = torch.arange(-radius, radius + 1, device=device)
        rr = r[:, None, None] + d[None, :, None]
        cc = c[:, None, None] + d[None, None, :]
        inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        self._outline_neighbourhood = torch.where(inside, rr * cols + cc, rows * cols).reshape(len(r), -1)

    def judge(self, inked: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every picture measure of ``inked (B, rows, cols)`` (bool), ``(B,)`` each (``stroke_coverage`` ``(B, K)``):
        ``iou``, ``coverage`` (the share of the target inked), ``boundary_f``, ``stroke_coverage``, ``stray_frac``,
        and ``passed``: all four rules met.
        """
        rule, batch = self.rule, inked.shape[0]
        ink_area = _count(inked)
        on_target = inked.reshape(batch, -1)[:, self._target_cells]  # (B, m)
        overlap = on_target.sum(dim=1).float()
        iou = overlap / (ink_area + self.target_area - overlap)
        crop = inked[:, self._crop[0], self._crop[1]]
        rows, cols = self._outline_trim
        outline = _outline(inked[:, self._outline_crop[0], self._outline_crop[1]])[:, rows, cols]
        precision = _count(outline & self._near_target_outline) / _count(outline).clamp_min(1.0)
        padded = torch.cat([outline.reshape(batch, -1), outline.new_zeros(batch, 1)], dim=1)
        recall = padded[:, self._outline_neighbourhood].any(dim=-1).float().mean(dim=1)
        boundary_f = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        covered = torch.zeros(batch, self.num_strokes, device=inked.device).index_add_(
            1, self._target_owner, on_target.float()
        )
        stroke_coverage = covered / self._stroke_area
        stray = (ink_area - _count(crop & self._near_target)) / self.target_area
        passed = (
            (iou >= rule.iou)
            & (boundary_f >= rule.boundary_f)
            & (stroke_coverage.amin(dim=1) >= rule.min_stroke_coverage)
            & (stray <= rule.max_stray_frac)
        )
        return {
            "iou": iou,
            "coverage": overlap / self.target_area,
            "boundary_f": boundary_f,
            "stroke_coverage": stroke_coverage,
            "stray_frac": stray,
            "passed": passed,
        }
