"""Ink on the board and how a drawing is graded: the original's laws and verifiers, batched in torch.

Everything here works on a pigment grid covering one drawing's area of the board (a :class:`BoardGrid`), one
grid per env. Ported from RoboWorld's ``whiteboard`` environment (``gym-anything/robotics``, MIT):

* :class:`InkLayer` — ``whiteboard/ink.py``: pigment transfer from the nib, ``(1 - p) (1 - exp(-350 dt min(F, 1)
  m))`` per sample, under the chisel's elliptical footprint ``m = clip(4 (1 - r), 0, 1)``, and the visible weight
  ``clip((p - 0.02) / 0.13, 0, 1)`` the verifiers read. The original integrates at every 0.5 ms physics tick on
  16 sub-samples per 0.58 mm texel; here one footprint is laid per sim step (the nib's pose and press are sampled
  after each one), on 2 x 2 sub-samples per cell. The original's 1500 uL reservoir is dropped: a task uses at most
  ~25 uL of it.
* :class:`TextJudge` — ``task_set/verifier.py``'s ``f1``: the lettering's F1 against the reference raster within
  the native 2 mm window, success at 0.85.
* :class:`TableJudge` — ``task_set/table_verifier.py``, judged at the placement shown (the original's own
  fallback reading): each line read across its normal for the ink centred on it, junctions excluded, coverage
  >= 90 %, no gap over 10 mm, at most 1 cm^2 of visible stray ink. Not ported: the registration search that
  lets the original accept a copy up to 30 mm / 5 % / 2 degrees off where it is shown (here the placement must
  match, which is stricter), and the skip fill that credits skips under 6 mm between long inked stretches.
* :class:`StrokeGuide` — not in the original: the reference strokes sampled every 2 mm in drawing order, each
  latched *covered* once visible ink lies within the F1 window of it. It feeds the dense rewards and the policy's
  next-points observation.

**Grading is incremental.** Ink only ever grows, and only under the nib, so nothing is regraded over the whole
grid each step: :meth:`InkLayer.deposit` keeps the visible weight, its total and its stray share up to date in the
cells it touches and returns them as an :class:`InkChange`, from which :class:`TextJudge` (the recall's reach)
and :class:`StrokeGuide` (the latches) update the same cells. :class:`TableJudge` reads its lines' neighbourhoods.

Ink outside the grid is not kept for grading; its visible area is added up in ``outside_m2`` and counts as stray
everywhere. Off the grid the pigment is kept, under the same law, only in a patch of the board that follows the
nib (``OFF_GRID_PATCH`` metres either way of it): a stroke, a resting nib, a light press that darkens over time
and a nib tapping one spot all count the visible area they would ink on the grid. Ink the patch leaves behind is
forgotten, so the only overcount is ink the nib comes back to after moving more than that far away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from eden.tasks.whiteboard.strokes import LINE_WIDTH, Stroke, raster, sample_strokes, strokes_bounds
from eden.utils.plane_grid import PlaneGrid

#: The original's visible-ink ramp: pigment density below VISIBLE shows nothing, above DRAWN a full line.
VISIBLE_DENSITY = 0.02
DRAWN_DENSITY = 0.15
#: The F1's neighbourhood: ink and reference count as overlapping within 2 mm (the native window).
F1_WINDOW = 0.002
#: How far either way of the nib the pigment off the grid is kept (m): see :class:`InkLayer`.
OFF_GRID_PATCH = 0.03


def visible_weight(pigment: torch.Tensor) -> torch.Tensor:
    """The original's visible weight of a pigment density, in ``[0, 1]``."""
    return ((pigment - VISIBLE_DENSITY) / (DRAWN_DENSITY - VISIBLE_DENSITY)).clamp(0.0, 1.0)


@dataclass(frozen=True)
class BoardGrid(PlaneGrid):
    """A :class:`~eden.utils.plane_grid.PlaneGrid` over the board (its frame the drawing's), with the drawing's
    footprint and reference raster.
    """

    @classmethod
    def around(cls, strokes: tuple[Stroke, ...], *, margin: float, cell: float) -> BoardGrid:
        """The grid covering ``strokes`` and ``margin`` metres round them."""
        u_min, v_min, u_max, v_max = strokes_bounds(strokes)
        u0, v0 = u_min - margin, v_max + margin
        return cls(u0, v0, cell, math.ceil((v0 - (v_min - margin)) / cell), math.ceil((u_max + margin - u0) / cell))

    def raster(self, strokes: tuple[Stroke, ...]) -> np.ndarray:
        """``strokes`` rasterized on this grid, the way the original draws its reference image."""
        return raster(strokes, origin=(self.u0, self.v0), cell=self.cell, shape=self.shape)


@dataclass(frozen=True)
class InkChange:
    """The cells a deposit inked, ``(num_envs, n)`` each: their flat index into :attr:`InkLayer.pigment` (the
    pad, ``rows * cols``, for a footprint cell off the grid) and their visible weight afterwards. A cell can appear
    more than once, its weight growing; every later entry of a cell is at least its earlier ones.
    """

    index: torch.Tensor
    weight: torch.Tensor


class InkLayer:
    """Per-env pigment on a :class:`BoardGrid`, laid down by the nib under the original's deposition law.

    The pigment is stored flat with one pad cell at the end (index ``rows * cols``), which always holds 0: a
    footprint cell off the grid is addressed there, so a gather needs no bounds mask.

    Off the grid, the pigment is kept in a square patch of cells round the nib (``OFF_GRID_PATCH`` either way, and
    at least twice the footprint's reach), re-centred once per :meth:`deposit` midway between its first and last
    footprints: a footprint's off-grid cells take the same transfer there, and the visible weight they gain is added
    to ``outside_m2``. Cells the patch moves off are dropped (they hold 0 again if the nib comes back). A
    footprint cell beyond the patch counts as fresh ink and is not kept; that takes a nib moving about twice
    ``OFF_GRID_PATCH`` within one deposit (over 1 m/s at the tasks' 50 ms control step).

    Parameters
    ----------
    grid : BoardGrid
    num_envs : int
    semi_axes : tuple[float, float]
        The chisel's footprint semi-axes (m), along the nib's x and y axes.
    rate : float
        Transfer rate (1/s) at a press of ``force_cap`` newtons or more: the original's 350.
    force_cap : float
        Press (N) above which the transfer no longer speeds up: the original's 1 N.
    stray_map : torch.Tensor
        ``(rows, cols)``: how much of a cell's visible ink is stray (the judge's :meth:`~TextJudge.stray_map`).
    device : torch.device | str
    """

    def __init__(
        self,
        grid: BoardGrid,
        num_envs: int,
        *,
        semi_axes,
        rate: float,
        force_cap: float,
        stray_map: torch.Tensor,
        device,
    ) -> None:
        self.grid, self.rate, self.force_cap = grid, rate, force_cap
        self.semi_axes = semi_axes
        self.pad = grid.rows * grid.cols
        self.pigment = torch.zeros(num_envs, self.pad + 1, device=device)
        self._stray_flat = torch.cat([stray_map.reshape(-1).to(device), torch.zeros(1, device=device)])
        #: Sum of the visible weight over the grid, and of its stray share (cells).
        self.total = torch.zeros(num_envs, device=device)
        self.stray_cells = torch.zeros(num_envs, device=device)
        self.outside_m2 = torch.zeros(num_envs, device=device)
        reach = math.ceil(max(semi_axes) / grid.cell) + 1
        dr, dc = torch.meshgrid(
            torch.arange(-reach, reach + 1, device=device),
            torch.arange(-reach, reach + 1, device=device),
            indexing="ij",
        )
        self._window = torch.stack([dr.reshape(-1), dc.reshape(-1)], dim=-1)  # (n_window, 2): row, col offsets
        # The off-grid patch: (2 R + 1)^2 cells, flat, centred on grid cell `_patch_centre` (row, col), plus a pad
        # slot at the end for footprint cells beyond it (only ever written 0). Cell (i, j) of it is grid cell
        # `_patch_centre - R + (i, j)`. At least twice the footprint's reach, so a footprint at rest always fits.
        self._patch_radius = max(math.ceil(OFF_GRID_PATCH / grid.cell), 2 * reach)
        side = 2 * self._patch_radius + 1
        pi, pj = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
        self._patch_ij = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # (side^2, 2)
        self._patch_side = side
        self._patch = torch.zeros(num_envs, side * side + 1, device=device)
        self._patch_centre = torch.zeros(num_envs, 2, dtype=torch.long, device=device)
        quarter = torch.tensor([-0.25, 0.25], device=device)
        sy, sx = torch.meshgrid(quarter, quarter, indexing="ij")
        self._subsamples = torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=-1)  # (4, 2): x (col), y (row)

    @property
    def weight(self) -> torch.Tensor:
        """Visible weight of every cell, ``(num_envs, rows, cols)``: computed over the whole grid, so not per step."""
        return visible_weight(self.pigment[:, : self.pad]).view(-1, *self.grid.shape)

    def weight_of(self, envs_idx: torch.Tensor) -> torch.Tensor:
        """:attr:`weight` of the envs ``envs_idx`` only, ``(len(envs_idx), rows, cols)``: what a camera over a few
        envs paints, without the whole batch's grid.
        """
        return visible_weight(self.pigment[envs_idx, : self.pad]).view(-1, *self.grid.shape)

    @property
    def stray_m2(self) -> torch.Tensor:
        """Visible stray ink (m^2) per env: the grid's, weighted by the stray map, and all of it off the grid."""
        return self.stray_cells * self.grid.cell**2 + self.outside_m2

    def to_grid(self, uv: torch.Tensor) -> torch.Tensor:
        """Continuous grid coordinates ``(x, y)`` (x along the columns, y down the rows, in cells) of ``uv``."""
        g = self.grid
        return torch.stack([(uv[..., 0] - g.u0) / g.cell, (g.v0 - uv[..., 1]) / g.cell], dim=-1)

    def footprint(self, nib_x_uv: torch.Tensor, nib_y_uv: torch.Tensor) -> torch.Tensor:
        """The footprint's semi-axis vectors as the columns of ``(..., 2, 2)``, in grid coordinates.

        ``nib_x_uv`` / ``nib_y_uv``: the nib's x and y axes projected on the board's ``(u, v)``, ``(..., 2)``.
        """
        a, b = self.semi_axes
        to_xy = torch.tensor([1.0, -1.0], device=nib_x_uv.device) / self.grid.cell
        return torch.stack([a * nib_x_uv * to_xy, b * nib_y_uv * to_xy], dim=-1)

    def deposit(self, uv: torch.Tensor, force: torch.Tensor, axes: torch.Tensor, dt: float) -> InkChange:
        """Lay one footprint per sample, in order: at ``uv (B, S, 2)``, pressing ``force (B, S)`` for ``dt`` s each.

        ``force`` is 0 where the nib does not draw; ``axes (B, S, 2, 2)`` is :meth:`footprint`'s matrix. A
        footprint whose projection degenerates (``|det| < 1e-6`` cells^2: the chisel edge-on to the board) draws
        nothing, as in the original. Returns the cells inked.

        A footprint's cells off the grid take the same transfer in the off-grid patch (re-centred midway between
        the first and the last sample) and add the visible weight they gain to ``outside_m2``.
        """
        live = (force > 0.0) & (torch.linalg.det(axes).abs() > 1e-6)  # (B, S)
        inverse = torch.linalg.inv(torch.where(live[..., None, None], axes, torch.eye(2, device=axes.device)))
        exponent = torch.where(live, self.rate * force.clamp(max=self.force_cap) * dt, 0.0)  # (B, S)
        centres = self.to_grid(uv)
        # Every footprint's window at once, (B, S, W): its cells and their transfer.
        rows, cols = self.grid.shape
        cell = torch.floor(centres).long().flip(-1)  # row, col of each centre's cell
        rr = cell[..., 0:1] + self._window[:, 0]
        cc = cell[..., 1:2] + self._window[:, 1]
        inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        # Within one footprint every grid cell appears once; the off-grid ones all address the pad, which takes no
        # transfer and so stays 0. Their transfer goes to the off-grid patch instead.
        index = torch.where(inside, rr * cols + cc, self.pad)
        transfer = exponent[..., None] * self._coverage(centres, inverse, rr, cc)
        transfer_off = torch.where(inside, 0.0, transfer)
        transfer = torch.where(inside, transfer, 0.0)
        self._recentre_patch(torch.div(cell[:, 0] + cell[:, -1], 2, rounding_mode="floor"))
        side, radius = self._patch_side, self._patch_radius
        pr = rr - self._patch_centre[:, None, None, 0] + radius
        pc = cc - self._patch_centre[:, None, None, 1] + radius
        in_patch = (pr >= 0) & (pr < side) & (pc >= 0) & (pc < side)
        patch_index = torch.where(in_patch, pr * side + pc, side * side)  # (B, S, W); beyond it, the pad slot
        # The pigment, one footprint after another: footprints of the same step overlap.
        weights = []
        for k in range(uv.shape[1]):
            pigment = self.pigment.gather(1, index[:, k])
            before = visible_weight(pigment)
            pigment = 1.0 - (1.0 - pigment) * torch.exp(-transfer[:, k])
            after = visible_weight(pigment)
            self.pigment.scatter_(1, index[:, k], pigment)
            gained = after - before
            self.total += gained.sum(dim=1)
            self.stray_cells += (gained * self._stray_flat[index[:, k]]).sum(dim=1)
            weights.append(after)
            # Off the grid, the same in the patch. Cells beyond it (all at the pad slot) count from 0 and keep
            # nothing: they write 0 back, so the pad's duplicate writes all agree.
            off = self._patch.gather(1, patch_index[:, k])
            off_after = 1.0 - (1.0 - off) * torch.exp(-transfer_off[:, k])
            self._patch.scatter_(1, patch_index[:, k], torch.where(in_patch[:, k], off_after, 0.0))
            self.outside_m2 += (visible_weight(off_after) - visible_weight(off)).sum(dim=1) * self.grid.cell**2
        return InkChange(index.flatten(1), torch.cat(weights, dim=1))

    def _recentre_patch(self, centre_cell: torch.Tensor) -> None:
        """Move the off-grid patch to centre on grid cell ``centre_cell (B, 2)`` (row, col), keeping the pigment
        of the cells it still covers and dropping the rest.
        """
        side = self._patch_side
        source = self._patch_ij + (centre_cell - self._patch_centre)[:, None, :]  # (B, side^2, 2) in the old patch
        kept = ((source >= 0) & (source < side)).all(dim=-1)
        source = source.clamp(0, side - 1)
        moved = torch.where(kept, self._patch.gather(1, source[..., 0] * side + source[..., 1]), 0.0)
        self._patch[:, :-1] = moved
        self._patch_centre.copy_(centre_cell)

    def _coverage(
        self, centre: torch.Tensor, inverse: torch.Tensor, rr: torch.Tensor, cc: torch.Tensor
    ) -> torch.Tensor:
        """The footprint's transfer mask ``m = clip(4 (1 - r), 0, 1)`` over cells ``(rr, cc) (..., W)``, averaged over
        2 x 2 sub-samples of each: centred at ``centre (..., 2)`` (grid coordinates), shaped by ``inverse (..., 2, 2)``.
        """
        # Sub-sample points relative to the centre, (..., W, 4) per axis, mapped into the footprint's unit disc.
        x = (cc.float() + 0.5 - centre[..., 0:1])[..., None] + self._subsamples[:, 0]
        y = (rr.float() + 0.5 - centre[..., 1:2])[..., None] + self._subsamples[:, 1]
        m = inverse[..., None, None, :, :]  # (..., 1, 1, 2, 2)
        r = torch.hypot(m[..., 0, 0] * x + m[..., 0, 1] * y, m[..., 1, 0] * x + m[..., 1, 1] * y)
        return (4.0 * (1.0 - r)).clamp(0.0, 1.0).mean(dim=-1)

    def paint(self, weight: torch.Tensor) -> InkChange:
        """Raise every cell's visible weight to at least ``weight (B, rows, cols)``: a board drawn without a nib.

        Ink only grows, so a cell already darker keeps its weight. For setting up a board (tests, a pre-drawn
        start); pass the returned change to the judges and the guide like a deposit's.
        """
        target = torch.cat([weight.reshape(weight.shape[0], -1), torch.zeros_like(weight[:, :1, 0])], dim=1)
        before = visible_weight(self.pigment)
        after = torch.maximum(before, target.clamp(0.0, 1.0))
        gained = after - before
        self.total += gained.sum(dim=1)
        self.stray_cells += (gained * self._stray_flat).sum(dim=1)
        drawn = VISIBLE_DENSITY + after * (DRAWN_DENSITY - VISIBLE_DENSITY)
        self.pigment.copy_(torch.where(gained > 0.0, drawn, self.pigment))
        index = torch.arange(self.pad + 1, device=weight.device).expand_as(after)
        return InkChange(index, after.clone())

    def reset(self, envs_idx) -> None:
        self.pigment[envs_idx] = 0.0
        self.total[envs_idx] = 0.0
        self.stray_cells[envs_idx] = 0.0
        self.outside_m2[envs_idx] = 0.0
        self._patch[envs_idx] = 0.0


def _owners_of_cells(grid: BoardGrid, owner_cells: np.ndarray, radius: int, device) -> torch.Tensor:
    """For every cell of ``grid`` and the pad, the owners whose ``(2 radius + 1)^2`` window holds it, ``(cells + 1, K)``.

    ``owner_cells (n, 2)`` is each owner's ``(row, col)``. Rows are padded with ``n``, a dummy owner. It inverts a
    window gather: ink on cell ``c`` is within reach of exactly the owners in row ``c``.
    """
    n = len(owner_cells)
    d = np.arange(-radius, radius + 1)
    rr = owner_cells[:, None, None, 0] + d[None, :, None]
    cc = owner_cells[:, None, None, 1] + d[None, None, :]
    inside = (rr >= 0) & (rr < grid.rows) & (cc >= 0) & (cc < grid.cols)
    cell = (rr * grid.cols + cc)[inside]
    owner = np.broadcast_to(np.arange(n)[:, None, None], inside.shape)[inside]
    order = np.argsort(cell, kind="stable")
    cell, owner = cell[order], owner[order]
    counts = np.bincount(cell, minlength=grid.rows * grid.cols + 1)
    slot = np.arange(len(cell)) - np.repeat(np.cumsum(counts) - counts, counts)
    table = np.full((grid.rows * grid.cols + 1, max(1, int(counts.max(initial=0)))), n, dtype=np.int64)
    table[cell, slot] = owner
    return torch.as_tensor(table, device=device)


def _dilate(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Max over the ``(2 radius + 1)^2`` square round each cell of ``(..., rows, cols)``."""
    shape = image.shape
    flat = image.reshape(-1, 1, *shape[-2:])
    return F.max_pool2d(flat, 2 * radius + 1, stride=1, padding=radius).reshape(shape)


class TextJudge:
    """The original's lettering F1: ink within 2 mm of the reference over all ink, and the reference reached.

    ``precision = sum(w * dilate(T)) / sum(w)`` (ink off the grid counts in the denominator),
    ``recall = sum(dilate(w) * T) / sum(T)``, both dilations over the native 2 mm window. The precision's sums are
    the ink layer's running totals (its stray map is ``1 - dilate(T)``); the recall keeps, per env and reference
    cell, the darkest ink within the window, raised by :meth:`update` from the cells each deposit inks.
    """

    def __init__(self, grid: BoardGrid, strokes: tuple[Stroke, ...], num_envs: int, device) -> None:
        self.grid = grid
        self.radius = max(1, round(F1_WINDOW / grid.cell))
        target = grid.raster(strokes)
        self.band = _dilate(torch.as_tensor(target.astype(np.float32)), self.radius).to(device)
        target_cells = np.argwhere(target)
        self.num_targets = len(target_cells)
        self._targets_of_cell = _owners_of_cells(grid, target_cells, self.radius, device)
        self.reach = torch.zeros(num_envs, self.num_targets + 1, device=device)  # the last slot is the dummy

    def update(self, change: InkChange) -> None:
        targets = self._targets_of_cell[change.index]  # (B, n, K)
        weight = change.weight[..., None].expand_as(targets)
        self.reach.scatter_reduce_(1, targets.flatten(1), weight.flatten(1), reduce="amax")

    def f1(self, ink: InkLayer) -> torch.Tensor:
        outside = ink.outside_m2 / self.grid.cell**2
        precision = (ink.total - ink.stray_cells) / (ink.total + outside).clamp_min(1e-12)
        recall = self.reach[:, : self.num_targets].sum(dim=1) / max(self.num_targets, 1)
        return 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    def stray_map(self) -> torch.Tensor:
        """Per cell, how much of its visible ink is off the lettering: outside the 2 mm band, 1; inside, 0."""
        return 1.0 - self.band

    def reset(self, envs_idx) -> None:
        self.reach[envs_idx] = 0.0


# --- the table ---------------------------------------------------------------------------------------------------
# The original's constants (task_set/table_verifier.py).
LINE_TOLERANCE = 0.0035
PROXIMITY_LIMIT = 0.006
CENTRE_INK = 0.001
COVERAGE_REQUIRED = 0.90
GAP_ALLOWED = 0.010
GAP_INK_MIN = 0.05
MARKER_HALF_WIDTH = 0.0025
JUNCTION_MARGIN = 0.0005
STRAY_ALLOWANCE_M2 = 1.0e-4
STRAY_RAMP = 0.003
SAMPLE_SPACING = 0.001
OFFSET_STEP = 0.0005


def _segment_distance(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Distance from each of ``points (n, 2)`` to the polyline ``path (m, 2)``."""
    a, b = path[:-1], path[1:]
    d = b - a
    t = np.clip(((points[:, None, :] - a[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-18), 0, 1)
    return np.hypot(*(points[:, None, :] - (a[None] + t[..., None] * d[None])).transpose(2, 0, 1)).min(axis=1)


class TableJudge:
    """The original's table verifier at the placement shown: per-line coverage and gaps, and stray ink.

    Each line is sampled every 1 mm; a sample reads the ink across the line's normal (+-6 mm in 0.5 mm steps) owned
    by its line (the line whose centreline is nearest), finds the reading's centre and credits the ink at that
    centre, ramped off between 3.5 and 6 mm from the centreline. Samples within a marker's half-width of another
    line are excluded (junctions); excluded samples that show ink are transparent to the gap rule. A line passes
    at >= 90 % coverage with no uncovered run over 10 mm; the board is clean with <= 1 cm^2 of visible ink farther
    than the line's band (3.5 mm + its half-width) from every line, ramped in over 3 mm.

    :meth:`judge` returns the original's ``progress`` (fidelity x cleanliness; placement is 1 here) and success.
    """

    def __init__(self, grid: BoardGrid, strokes: tuple[Stroke, ...], device) -> None:
        self.grid = grid
        paths = [np.asarray(path, dtype=np.float64) for _, _, path in strokes]
        half = [LINE_WIDTH[weight] / 2 for _, weight, _ in strokes]
        offsets = np.arange(-PROXIMITY_LIMIT, PROXIMITY_LIMIT + 1e-9, OFFSET_STEP)
        points, normals, included, element, spacing, line_start = [], [], [], [], [], []
        self.line_length = np.array([np.hypot(*np.diff(p, axis=0).T).sum() for p in paths])
        for i, path in enumerate(paths):
            arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
            s = np.linspace(0.0, arc[-1], max(2, math.ceil(arc[-1] / SAMPLE_SPACING) + 1))
            x, y = np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])
            tx, ty = np.gradient(x), np.gradient(y)
            norm = np.hypot(tx, ty)
            pts = np.c_[x, y]
            keep = np.ones(len(s), dtype=bool)
            for j, other in enumerate(paths):
                if j != i:  # one marker colour: every other line is a junction partner
                    keep &= _segment_distance(pts, other) > max(half[j], MARKER_HALF_WIDTH) + JUNCTION_MARGIN
            points.append(pts)
            normals.append(np.c_[-ty / norm, tx / norm])
            included.append(keep)
            element.append(np.full(len(s), i))
            spacing.append(np.full(len(s), s[1] - s[0]))
            line_start.append(np.arange(len(s)) == 0)
        points, normals = np.concatenate(points), np.concatenate(normals)
        element_np = np.concatenate(element)
        reads = points[:, None, :] + offsets[None, :, None] * normals[:, None, :]  # (n, n_offsets, 2)
        cells = grid.cell_of(reads)
        inside = (cells[..., 0] >= 0) & (cells[..., 0] < grid.rows) & (cells[..., 1] >= 0) & (cells[..., 1] < grid.cols)
        pad = grid.rows * grid.cols  # InkLayer's pad cell: always 0
        flat = np.where(
            inside, cells[..., 0].clip(0, grid.rows - 1) * grid.cols + cells[..., 1].clip(0, grid.cols - 1), pad
        )
        # Ownership and stray ramp per cell, from the distance of its centre to every line's centreline.
        centres = grid.centers().reshape(-1, 2)
        distance = np.stack([_segment_distance(centres, p) for p in paths], axis=1)  # (cells, lines)
        owner = distance.argmin(axis=1)
        band = LINE_TOLERANCE + np.maximum(np.asarray(half), MARKER_HALF_WIDTH)
        nearest = distance[np.arange(len(owner)), owner]
        ramp = np.clip((nearest - band[owner]) / STRAY_RAMP, 0.0, 1.0)

        def t(a, dtype=torch.float32):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

        self.num_lines = len(paths)
        self.offsets = t(offsets)
        span = math.ceil(CENTRE_INK / OFFSET_STEP)
        self._near_steps = torch.arange(-span, span + 1, device=device)
        included_np = np.concatenate(included)
        owned = inside & (np.r_[owner, -1][flat] == element_np[:, None])
        # Reads into InkLayer.pigment, (n, n_offsets): a sample's own line's ink, the pad elsewhere; and, for
        # the excluded (junction) samples only, every read, for the gap rule's transparency.
        self.owned_index = t(np.where(owned, flat, pad), torch.long)
        self.excluded = t(np.flatnonzero(~included_np), torch.long)
        self.excluded_index = t(flat[~included_np], torch.long)
        self.included = t(included_np, torch.bool)
        self.element = t(element_np, torch.long)
        self.spacing = t(np.concatenate(spacing))
        self.line_start = t(np.concatenate(line_start), torch.bool)
        self.included_length = torch.zeros(self.num_lines, device=device).index_add_(
            0, self.element, self.spacing * self.included.float()
        )
        self.line_length_t = t(self.line_length)
        self.line_spacing = t([sp[0] for sp in spacing])
        self.stray = t(ramp.reshape(grid.shape))
        self.area_m2 = float(sum(length * LINE_WIDTH[w] for length, (_, w, _) in zip(self.line_length, strokes)))

    def stray_map(self) -> torch.Tensor:
        return self.stray

    def judge(self, ink: InkLayer) -> dict[str, torch.Tensor]:
        """``progress`` (B,), ``success`` (B,) and the per-line ``coverage`` / ``gap`` (B, lines) of ``ink``."""
        pigment = ink.pigment
        batch = pigment.shape[0]
        # visible_weight, in place on the gather's fresh copy: this is the judge's one (B, n, n_offsets) tensor.
        own = pigment[:, self.owned_index].sub_(VISIBLE_DENSITY).div_(DRAWN_DENSITY - VISIBLE_DENSITY).clamp_(0.0, 1.0)
        mass = own.sum(dim=-1)
        centre = torch.where(mass > 0, (own @ self.offsets) / mass.clamp_min(1e-12), 0.0)
        # The ink at the centre: the offsets within CENTRE_INK of it, all within two steps of the nearest one.
        nearest = ((centre + PROXIMITY_LIMIT) / OFFSET_STEP).round().long()
        near = (nearest[..., None] + self._near_steps).clamp(0, self.offsets.numel() - 1)  # (B, n, 5)
        within = (self.offsets[near] - centre[..., None]).abs() <= CENTRE_INK
        at_centre = torch.where(within, own.gather(-1, near), 0.0).amax(dim=-1)
        raw = at_centre * ((PROXIMITY_LIMIT - centre.abs()) / (PROXIMITY_LIMIT - LINE_TOLERANCE)).clamp(0.0, 1.0)
        hit = self.included & (raw >= GAP_INK_MIN)
        transparent = torch.zeros_like(hit)
        transparent[:, self.excluded] = (visible_weight(pigment[:, self.excluded_index]) >= GAP_INK_MIN).any(dim=-1)
        missing = (~hit & ~transparent).float()
        # Longest run of missing samples between two hits, per line: hits and line starts open a new run.
        run = torch.cumsum((hit | self.line_start).long(), dim=1) - 1  # (B, n)
        run_missing = torch.zeros_like(missing).scatter_add_(1, run, missing)
        run_line = torch.zeros_like(run).scatter_(1, run, self.element.expand_as(run))
        longest = (
            torch.zeros(batch, self.num_lines, device=pigment.device).scatter_reduce_(
                1, run_line, run_missing, reduce="amax"
            )
            * self.line_spacing
        )
        covered = torch.zeros(batch, self.num_lines, device=pigment.device).index_add_(
            1, self.element, raw * self.spacing * self.included.float()
        )
        coverage = covered / self.included_length.clamp_min(1e-12)
        gap_excess = (longest - GAP_ALLOWED).clamp_min(0.0)
        credit = torch.minimum((coverage / COVERAGE_REQUIRED).clamp(max=1.0), 1.0 - gap_excess / self.line_length_t)
        fidelity = credit.clamp(0.0, 1.0).mean(dim=-1)
        stray_m2 = ink.stray_m2
        cleanliness = (1.0 - (stray_m2 - STRAY_ALLOWANCE_M2).clamp_min(0.0) / self.area_m2).clamp(0.0, 1.0)
        complete = (coverage >= COVERAGE_REQUIRED).all(dim=-1) & (gap_excess == 0.0).all(dim=-1)
        success = complete & (stray_m2 <= STRAY_ALLOWANCE_M2)
        progress = torch.where(success, 1.0, (fidelity * cleanliness).clamp(max=math.nextafter(1.0, 0.0)))
        return {"progress": progress, "success": success, "coverage": coverage, "gap": longest, "stray_m2": stray_m2}


class StrokeGuide:
    """The reference strokes as points every ``spacing`` metres, in drawing order, latched covered when inked.

    A point is covered once visible ink of weight >= 0.5 lies within the F1 window (2 mm) of it: :meth:`update`
    latches the points within reach of each cell a deposit darkened that far. ``newly_covered`` is the share of
    the points the last update covered, which the dense reward pays.
    """

    def __init__(self, grid: BoardGrid, strokes: tuple[Stroke, ...], num_envs: int, *, spacing: float, device) -> None:
        self.grid = grid
        points, stroke = sample_strokes(strokes, spacing)
        self.points = torch.as_tensor(points, dtype=torch.float32, device=device)  # (n, 2) board uv
        self.stroke = torch.as_tensor(stroke, dtype=torch.long, device=device)
        radius = max(1, round(F1_WINDOW / grid.cell))
        self._points_of_cell = _owners_of_cells(grid, grid.cell_of(points), radius, device)
        self._covered = torch.zeros(num_envs, len(points) + 1, dtype=torch.bool, device=device)  # last: the dummy
        self.newly_covered = torch.zeros(num_envs, device=device)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def covered(self) -> torch.Tensor:
        """``(num_envs, num_points)``: whether each point is covered."""
        return self._covered[:, : self.num_points]

    def update(self, change: InkChange) -> None:
        points = self._points_of_cell[change.index]  # (B, n, K)
        reached = torch.where((change.weight >= 0.5)[..., None], points, self.num_points)
        before = self.covered.sum(dim=-1)
        self._covered.scatter_(1, reached.flatten(1), True)
        self.newly_covered.copy_((self.covered.sum(dim=-1) - before).float() / self.num_points)

    @property
    def fraction_covered(self) -> torch.Tensor:
        return self.covered.float().mean(dim=-1)

    def next_index(self) -> torch.Tensor:
        """Index of each env's first uncovered point in drawing order; the last point once all are covered."""
        covered = self.covered
        order = torch.arange(self.num_points, device=covered.device).expand_as(covered)
        return torch.where(~covered, order, self.num_points - 1).amin(dim=-1)

    def reset(self, envs_idx) -> None:
        self._covered[envs_idx] = False
        self.newly_covered[envs_idx] = 0.0
