"""How a loaded brush lays wet acrylic paint on the sheet, batched in torch.

Like the writing tools (``eden.tasks.art.calligraphy.tools``), the brush is kinematic: what it lays follows from where
its tip is and its *depth*, how far below the paper's surface the undeformed tip has gone (0 or less: in the air). It
is a round brush: its mark is a disc at the tip whose diameter grows with depth (:class:`RoundBrush`), with a soft
edge.

The paint (:class:`Canvas`) is wet acrylic, kept per env on a grid over the sheet. What makes it paint rather than a
painting program's colour:

* **Paints mix as pigments do.** A paint is its Kubelka-Munk pigment (:func:`pigment`): how much it absorbs ``K``
  per colour channel and how much it scatters ``S``, which mix linearly with the amount of each paint; the colour
  shown is the reflectance of the mix's ``K/S`` (:func:`reflectance`). So blue and yellow make green, not grey, and
  a dark paint tints a light one less than its darkness alone would (a light paint is mostly white, which scatters
  strongly). The paper is not paint: it shows through where the paint does not cover it.
* **The brush and the wet paint exchange paint.** For each brush diameter it travels on the paper, the brush lays
  its paint over the cells under it at ``Acrylic.deposit_rate``, and picks up the wet paint under it at
  ``Acrylic.pickup_rate``, so its colour drifts toward what it drags through until the next dip. Wet paint is
  covered less (``Acrylic.wet_resistance``): it mixes with what is laid into it. Both go by the distance the brush
  covers, not by time or by how often it is sampled, and saturate: going over a stroke's own path again changes it
  only toward the brush's paint.
* **Paint dries.** A cell's wetness falls by ``e`` every ``Acrylic.open_time`` seconds of the canvas's clock
  (:meth:`Canvas.tick`): strokes laid one after another blend; a layer laid on a dry one covers it.
* **Wet paint blends across a stroke's sides.** The bristles along the sides of a gliding brush drag the wet paint
  there across them, both ways: within ``Acrylic.blend_width`` of either side, the paint is smoothed with the paint
  beside it at ``Acrylic.blend_rate``, as freely as both are wet. So a stroke laid beside or over wet paint meets it
  in a gradient rather than a seam, and keeps its edge against bare paper and paint that has dried.
* **The brush has bristles.** Laid into wet paint, or from a brush running dry, how much a cell takes varies across
  the brush's width, along its motion (``Acrylic.streaks``): it leaves streaks. A full brush on paper or dry paint
  covers evenly.
* **The brush carries a load of paint** (:class:`PaintLoad`), 1 after a dip (:meth:`Canvas.dip`), which the area
  the brush sweeps on the paper uses up as it lays (:meth:`Canvas.lay`). The paint's opacity stays full while the
  load is above ``dry_load`` and falls to nothing as it runs out; an undipped brush is empty. A dip fills the brush
  with the well's paint, replacing what was on it (the brush is taken as clean: there is no rinse to model, and the
  well is not dirtied). The canvas keeps the brush (its paint and its load), so a target painted along a plan and
  a copy painted through the robot use up paint by the same rule.

RoboPaint paints with libmypaint's acrylic brush; this is a model of its own, made to run every env at once on the
GPU. The mixing is the two-constant Kubelka-Munk model over three channels, as pigment-mixing paint programs use: it
does not know a pigment's spectrum, only its colour, and takes its scattering from its lightness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from eden.tasks.art.painting.grading import linear_to_srgb, srgb_to_linear
from eden.tasks.art.workcell import PAPER_COLOR
from eden.utils.misc import is_env_mask
from eden.utils.plane_grid import PlaneGrid

#: The darkest reflectance a paint is taken to have: keeps a black's ``K/S`` finite.
_MIN_REFLECTANCE = 1.0e-3
#: The scattering of a black paint, a white's being 1: a paint scatters in proportion to its luminance between them.
_BLACK_SCATTERING = 0.1
#: Linear RGB's luminance weights (Rec. 709).
_LUMINANCE = (0.2126, 0.7152, 0.0722)
#: The bristles' streaks: waves of (cycles across the brush's width, phase) whose sum makes the streaks' pattern.
_BRISTLE_WAVES = ((2.3, 0.7), (5.1, 2.9), (8.7, 1.3), (13.9, 4.1), (21.7, 5.6))


def pigment(rgb: torch.Tensor) -> torch.Tensor:
    """The Kubelka-Munk pigment ``(..., 4)`` of a paint whose colour (sRGB in ``[0, 1]``) is ``rgb (..., 3)``: its
    absorption ``K`` per channel and its scattering ``S``, with the ``K/S`` that makes that colour.
    """
    linear = srgb_to_linear(rgb).clamp(_MIN_REFLECTANCE, 1.0)
    # Weighted with Python floats, not a tensor of them: that would be a blocking copy to the GPU on every call.
    luminance = sum(weight * channel for weight, channel in zip(_LUMINANCE, linear.unbind(-1)))
    scattering = (_BLACK_SCATTERING + (1.0 - _BLACK_SCATTERING) * luminance)[..., None]
    return torch.cat([(1.0 - linear) ** 2 / (2.0 * linear) * scattering, scattering], dim=-1)


def reflectance(pigment: torch.Tensor) -> torch.Tensor:
    """The linear reflectance ``(..., 3)`` of an opaque layer of the pigment ``(..., 4)`` (:func:`pigment`)."""
    return km_reflectance(pigment[..., :3] / pigment[..., 3:].clamp_min(1e-12))


def km_reflectance(ratio: torch.Tensor) -> torch.Tensor:
    """The linear reflectance ``(..., 3)`` of an opaque layer of paint whose ``K/S`` is ``ratio (..., 3)``:
    ``1 + K/S - sqrt(K/S (K/S + 2))``, written so a dark paint's does not cancel away in float32.
    """
    return 1.0 / (1.0 + ratio + torch.sqrt(ratio * (ratio + 2.0)))


def bristles(across: torch.Tensor) -> torch.Tensor:
    """How little paint the bristles at ``across`` (``-1`` to ``1``, edge to edge of the brush) carry, in ``[0, 1]``:
    a fixed pattern of streaks, so a stroke made twice streaks the same way.
    """
    waves = sum(torch.sin(math.pi * cycles * across + phase) for cycles, phase in _BRISTLE_WAVES)
    return (0.5 + 0.35 * waves / math.sqrt(len(_BRISTLE_WAVES))).clamp(0.0, 1.0)


@dataclass(frozen=True)
class RoundBrush:
    """A round brush's mark: a disc at the tip, ``diameter_at(depth)`` across, its edge ``softness`` wide.

    The diameter grows linearly from ``min_diameter`` at a touch to ``max_diameter`` at ``full_press`` deep and no
    further. A cell's coverage falls linearly across the edge, from 1 at ``softness / 2`` inside the disc's rim to 0
    at ``softness / 2`` outside it.

    Parameters
    ----------
    min_diameter, max_diameter : float
        The mark's diameter (m) at a touch and fully pressed: 1 mm, and 18 mm, a little over the 16 mm of RoboPaint's
        broadest acrylic brush.
    full_press : float
        Depth (m) at which the tuft is fully pressed: RoboPaint's ``full_press_m``, 15 mm.
    softness : float
        Width (m) of the disc's soft edge.
    """

    min_diameter: float = 0.001
    max_diameter: float = 0.018
    full_press: float = 0.015
    softness: float = 0.0005

    @property
    def reach(self) -> float:
        """How far (m) from the tip any cell the brush covers can lie."""
        return self.max_diameter / 2 + self.softness / 2

    def diameter_at(self, depth: torch.Tensor) -> torch.Tensor:
        """The mark's diameter (m) at ``depth``; 0 in the air."""
        press = (depth / self.full_press).clamp(0.0, 1.0)
        diameter = self.min_diameter + (self.max_diameter - self.min_diameter) * press
        return torch.where(depth > 0.0, diameter, 0.0)

    def depth_for(self, diameter: float) -> float:
        """The depth (m) at which the mark is ``diameter`` across (what a plan presses to for a brush size)."""
        if not self.min_diameter <= diameter <= self.max_diameter:
            raise ValueError(
                f"RoundBrush: a {diameter * 1000:.1f} mm mark is outside the brush's "
                f"{self.min_diameter * 1000:.1f}-{self.max_diameter * 1000:.1f} mm range."
            )
        return self.full_press * (diameter - self.min_diameter) / (self.max_diameter - self.min_diameter)

    def coverage(self, distance: torch.Tensor, diameter: torch.Tensor) -> torch.Tensor:
        """How much of a cell ``distance`` (m) from the tip a ``diameter``-wide mark covers, in ``[0, 1]``."""
        covered = ((diameter / 2 - distance) / self.softness + 0.5).clamp(0.0, 1.0)
        return torch.where(diameter > 0.0, covered, 0.0)


@dataclass(frozen=True)
class PaintLoad:
    """How much paint a full brush carries, and how its opacity fades as it runs out.

    Parameters
    ----------
    capacity : float
        The area (m^2) of paper a full brush sweeps before it is empty: 3000 mm^2, a 25 cm stroke 12 mm wide.
    dry_load : float
        The load below which the paint's opacity falls, linearly to 0 at an empty brush.
    reserve : float
        The load a plan dips before going under (:meth:`~eden.tasks.art.painting.works.Painting.dips`): a copy that
        sweeps more paper than the plan, wandering or pressing harder, still has paint at full opacity.
    """

    capacity: float = 3.0e-3
    dry_load: float = 0.2
    reserve: float = 0.4

    def opacity(self, load: torch.Tensor) -> torch.Tensor:
        return (load / self.dry_load).clamp(0.0, 1.0)


@dataclass(frozen=True)
class Acrylic:
    """How the wet paint behaves (see the module docstring). Rates are per brush diameter the brush travels.

    Parameters
    ----------
    deposit_rate : float
        How fast the brush covers what is under it: a cell a full brush passes over the middle of is covered by all
        but ``exp(-deposit_rate)`` of it, on paper or dry paint.
    wet_resistance : float
        The share by which wet paint slows the covering: what the brush lays into wet paint mixes with it.
    pickup_rate : float
        How fast the brush's paint turns to the wet paint it drags through: over a brush's diameter of nothing but
        wet paint, by ``1 - exp(-pickup_rate)``.
    open_time : float
        The time (s) in which the paint's wetness falls by ``e``.
    streaks : float
        How much the bristles vary what a cell takes in fully wet paint or from an empty brush, 0 (none) to 1 (some
        take nothing).
    blend_width : float
        How far (m) to either side of the brush's sides the bristles drag the wet paint, and how far they drag it.
    blend_rate : float
        How fast the wet paint along the brush's sides is smoothed, per brush diameter it glides: a cell on a side's
        line, in fully wet paint, takes all but ``exp(-blend_rate)`` of the smoothed paint over a diameter.
    """

    deposit_rate: float = 20.0
    wet_resistance: float = 0.85
    pickup_rate: float = 0.25
    open_time: float = 120.0
    streaks: float = 0.85
    blend_width: float = 0.0025
    blend_rate: float = 3.0

    def __post_init__(self) -> None:
        if self.blend_width <= 0.0:
            raise ValueError(
                f"Acrylic: blend_width={self.blend_width} must be positive (it is a width, in m); to turn the blending "
                "across a stroke's sides off, set blend_rate=0.0 instead."
            )


class Canvas:
    """Per-env wet acrylic on a :class:`~eden.utils.plane_grid.PlaneGrid` over the sheet (see the module docstring).

    The canvas is float32 whatever Eden's precision (``eden.init(precision=...)``), as the samples laid on it are.

    Each cell holds two layers: the dry one, as the colour it shows (``under``, linear RGB: the paper, or the paint
    dried on it); and the wet paint over it, as its pigment (``pigment``) and how much of the dry layer it hides
    (``cover``), wet as of the clock's time ``stamp``. All are stored flat with a pad cell at the end (index
    ``rows * cols``) where a mark's cells off the sheet are addressed. The wet layer dries into the dry one when the
    brush next comes to it, by as much as it has dried; how it looks does not change as it dries. The brush holds its
    paint's pigment (``brush``) and its load (``load``, of ``paint_load``). Colours given and shown are sRGB in
    ``[0, 1]``; the bare sheet is ``paper``, the colour the scene draws the paper in.
    """

    def __init__(
        self,
        grid: PlaneGrid,
        num_envs: int,
        *,
        brush: RoundBrush = RoundBrush(),
        load: PaintLoad = PaintLoad(),
        medium: Acrylic = Acrylic(),
        paper=PAPER_COLOR,
        device,
    ) -> None:
        self.grid, self.round_brush, self.paint_load, self.medium = grid, brush, load, medium
        # float32 whatever Eden's precision, so a plan and a target are the same at any precision.
        f32 = {"dtype": torch.float32, "device": device}
        self.pad = grid.rows * grid.cols
        n, cells = num_envs, self.pad + 1
        self.paper = srgb_to_linear(torch.tensor(paper, **f32))
        self.under = self.paper.repeat(n, cells, 1)
        self.pigment = torch.zeros(n, cells, 4, **f32)
        self.cover = torch.zeros(n, cells, **f32)
        self.stamp = torch.zeros(n, cells, **f32)
        #: Whether any paint has been laid on each cell (``(B, pad + 1)``, the pad always False).
        self.painted = torch.zeros(n, cells, dtype=torch.bool, device=device)
        #: Each env's clock (s), advanced by :meth:`tick`.
        self.clock = torch.zeros(n, **f32)
        #: The paint on the brush (its pigment), and its load: 1 after a dip, 0 empty.
        self.brush = torch.zeros(n, 4, **f32)
        self.load = torch.zeros(n, **f32)
        # Where the tip was at the last sample laid, the mark's radius there (m, 0 in the air), the direction the
        # brush last moved in, and a scratch buffer of each cell's exposure to the run being laid.
        self._last_uv = torch.zeros(n, 2, **f32)
        # Whether the brush has laid a sample since the env's reset: the first after it moves from nowhere.
        self._has_last = torch.zeros(n, dtype=torch.bool, device=device)
        self._last_radius = torch.zeros(n, **f32)
        self._start_heading = torch.tensor([1.0, 0.0], **f32)
        self._heading = self._start_heading.repeat(n, 1)
        self._exposure = torch.zeros(n, cells, 2, **f32)
        self._blend = torch.zeros(n, cells, 5, **f32)
        # The smoothing along the sides: a Gaussian, blend_width / 2 its standard deviation, in cells.
        sigma = medium.blend_width / 2 / grid.cell
        taps = torch.arange(-math.ceil(2 * sigma), math.ceil(2 * sigma) + 1, **f32)
        smoothing = torch.exp(-0.5 * (taps / max(sigma, 1e-6)) ** 2)
        self._smoothing = (smoothing / smoothing.sum()).view(1, 1, 1, -1)  # (1, 1, 1, taps): along a row
        # The cells a mark's samples can reach: its disc and soft edge, and the band along its sides (whose outer
        # cells, blended least, are smoothed with what of their neighbours the window holds).
        reach = math.ceil(max(brush.reach, brush.max_diameter / 2 + medium.blend_width) / grid.cell) + 1
        dr, dc = torch.meshgrid(
            torch.arange(-reach, reach + 1, device=device),
            torch.arange(-reach, reach + 1, device=device),
            indexing="ij",
        )
        self._window = torch.stack([dr.reshape(-1), dc.reshape(-1)], dim=-1)  # (W, 2): row, col offsets
        self._window_size = 2 * reach + 1

    @property
    def num_envs(self) -> int:
        return self.cover.shape[0]

    @staticmethod
    def _shown(under: torch.Tensor, paint: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        """The linear colour of cells whose dry layer is ``under``, with wet ``paint`` (its pigment) hiding ``cover``
        of it.
        """
        return under + cover[..., None] * (reflectance(paint) - under)

    def rgb(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, rows, cols, 3)``: what the sheet shows now; only the envs ``envs_idx`` (in that order) if given, so a
        camera over one env does not compute the colours of the rest.
        """
        envs = slice(None) if envs_idx is None else envs_idx
        shown = self._shown(self.under[envs, : self.pad], self.pigment[envs, : self.pad], self.cover[envs, : self.pad])
        return linear_to_srgb(shown).view(-1, *self.grid.shape, 3)

    def painted_cells(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, rows, cols)`` (bool): the cells any paint has been laid on; only the envs ``envs_idx`` if given."""
        envs = slice(None) if envs_idx is None else envs_idx
        return self.painted[envs, : self.pad].reshape(-1, *self.grid.shape)

    def tick(self, dt: float) -> None:
        """Advance every env's clock by ``dt`` (s): the paint dries."""
        self.clock += dt

    def dry(self) -> None:
        """Let all the paint dry at once (as the planner does between strokes)."""
        self.under.copy_(self._shown(self.under, self.pigment, self.cover))
        self.cover.zero_()

    def brush_rgb(self) -> torch.Tensor:
        """``(B, 3)``: the colour (sRGB) of the paint on the brush, as it has picked up wet paint since its dip."""
        return linear_to_srgb(reflectance(self.brush))

    def dip(self, envs: torch.Tensor, rgb: torch.Tensor) -> None:
        """Dip the brush of every env where ``envs (B,)`` is set: fill it with a full load of paint ``rgb (B, 3)``."""
        self.brush.copy_(torch.where(envs[:, None], pigment(rgb), self.brush))
        self.load.masked_fill_(envs, 1.0)

    def lay(self, uv: torch.Tensor, depth: torch.Tensor) -> None:
        """Paint the brush's samples ``uv (B, S, 2)``, ``depth (B, S)``, in order.

        The samples are laid as one run, at the clock's time: the brush lays the paint it holds at the run's start,
        and what it picks up along the run is its paint for the next. Each sample lays at the opacity of the load
        left before it (:meth:`PaintLoad.opacity`), and uses up the paper it sweeps (the way it glided on the paper
        times the mark's diameter) over the load's ``capacity``. A mark that overhangs the sheet's edge uses paint for
        its whole width: the part off the sheet is laid on the desk the sheet lies on, which the canvas does not keep.
        """
        g, medium = self.grid, self.medium
        diameter = self.round_brush.diameter_at(depth)  # (B, S); 0 in the air
        radius = diameter / 2
        # How far the footprint moves at each sample, in brush diameters: the tip's travel plus how much the mark
        # grows or shrinks; at a touch-down, its radius.
        last_uv = torch.where(self._has_last[:, None], self._last_uv, uv[:, 0])
        prev_uv = torch.cat([last_uv[:, None], uv[:, :-1]], dim=1)
        prev_radius = torch.cat([self._last_radius[:, None], radius[:, :-1]], dim=1)
        offset = uv - prev_uv
        step = offset.norm(dim=-1)
        glide = torch.where(prev_radius > 0.0, step, 0.0)
        spread = torch.where(prev_radius > 0.0, (radius - prev_radius).abs(), radius)
        per_diameter = torch.where(diameter > 0.0, 1.0 / diameter.clamp_min(self.round_brush.min_diameter), 0.0)
        diameters = (glide + spread) * per_diameter
        used = glide * diameter / self.paint_load.capacity  # (B, S): the share of a full load each sample uses
        opacity = self.paint_load.opacity((self.load[:, None] - (used.cumsum(dim=1) - used)).clamp_min(0.0))
        self.load.sub_(used.sum(dim=1)).clamp_(min=0.0)
        # The heading at each sample: the way it moved, or the last heading if it did not move.
        moved = step > 1e-7
        heading = torch.cat([self._heading[:, None], offset / step.clamp_min(1e-7)[..., None]], dim=1)
        steps = torch.arange(heading.shape[1], device=heading.device)
        latest = torch.where(torch.cat([moved.new_ones(moved.shape[0], 1), moved], dim=1), steps, 0).cummax(1).values
        heading = heading.gather(1, latest[:, 1:, None].expand(-1, -1, 2))
        self._last_uv.copy_(uv[:, -1])
        self._has_last.fill_(True)
        self._last_radius.copy_(radius[:, -1])
        self._heading.copy_(heading[:, -1])

        # The cells under each sample's footprint, how much of each the disc covers, and how its bristles streak.
        row = torch.floor((g.v0 - uv[..., 1]) / g.cell).long()
        col = torch.floor((uv[..., 0] - g.u0) / g.cell).long()
        rr = row[..., None] + self._window[:, 0]  # (B, S, W)
        cc = col[..., None] + self._window[:, 1]
        on_sheet = (rr >= 0) & (rr < g.rows) & (cc >= 0) & (cc < g.cols)
        du = g.u0 + (cc.float() + 0.5) * g.cell - uv[..., 0, None]
        dv = g.v0 - (rr.float() + 0.5) * g.cell - uv[..., 1, None]
        distance = torch.hypot(du, dv)
        under = self.round_brush.coverage(distance, diameter[..., None]) * on_sheet  # (B, S, W)
        lateral = du * heading[..., 1, None] - dv * heading[..., 0, None]  # (m) right of the brush's motion
        across = lateral / radius.clamp_min(1e-6)[..., None]
        lateral = lateral.abs()
        streaky = medium.streaks * bristles(across.clamp(-1.0, 1.0))
        index = torch.where(on_sheet, rr * g.cols + cc, self.pad).flatten(1)  # (B, S W)
        three, four = index[..., None].expand(-1, -1, 3), index[..., None].expand(-1, -1, 4)

        # The cells as the run finds them, their wet paint dried into the dry layer by as much as it has dried
        # since it was last laid into: what is left of it is wet.
        dry_layer = self.under.gather(1, three)
        paint = self.pigment.gather(1, four)
        cover = self.cover.gather(1, index)
        wetness = torch.exp(-(self.clock[:, None] - self.stamp.gather(1, index)) / medium.open_time)
        wet = cover * wetness  # (B, S W)
        # The dry layer takes the dried paint so the cell looks as it did: the dry layer where the wet paint
        # did not cover it, and the dried paint.
        dried = (cover - wet)[..., None]
        dry_layer = torch.where(
            (wet < 1.0)[..., None],
            ((1.0 - cover)[..., None] * dry_layer + dried * reflectance(paint))
            / (1.0 - wet).clamp_min(1e-12)[..., None],
            dry_layer,
        )

        # What the brush picks up: the wet paint under it, as much as its footprint is wet.
        wet_paint = under.flatten(1) * wet
        wet_share = wet_paint.view_as(under).sum(dim=-1) / under.sum(dim=-1).clamp_min(1e-12)  # (B, S)
        picked = (wet_paint[..., None] * paint).sum(dim=1) / wet_paint.sum(dim=1).clamp_min(1e-12)[:, None]
        pickup = 1.0 - torch.exp(-medium.pickup_rate * (diameters * wet_share).sum(dim=1))

        # What it lays, each cell's exposure summed over the run's samples (so a cell under several of them is
        # covered as by one pass the length of theirs), in two parts: how much of the dry layer it hides, streaked
        # by the bristles as the brush runs dry; and how much of the wet paint it replaces, less as it is wetter
        # (what it lays there mixes in) and streaked as much.
        wet = wet.view_as(under)
        dose = (medium.deposit_rate * diameters * opacity)[..., None] * under  # (B, S, W)
        covering = dose * (1.0 - streaky * (1.0 - opacity[..., None]))
        replacing = dose * (1.0 - streaky * wet) * (1.0 - medium.wet_resistance * wet)
        wet = wet.flatten(1)
        both = index[..., None].expand(-1, -1, 2)
        self._exposure.scatter_add_(1, both, torch.stack([covering.flatten(1), replacing.flatten(1)], dim=-1))
        covered, replaced = (1.0 - torch.exp(-self._exposure.gather(1, both))).unbind(-1)
        self._exposure.scatter_(1, both, 0.0)
        # The wet layer: what was wet, but for the share replaced; and the brush's paint, there and over the dry
        # layer newly hidden.
        new_cover = wet + covered * (1.0 - wet)
        kept = wet * (1.0 - replaced)
        mixed = kept[..., None] * paint + (new_cover - kept)[..., None] * self.brush[:, None]
        new_paint = torch.where((new_cover > 0.0)[..., None], mixed / new_cover.clamp_min(1e-12)[..., None], paint)
        if medium.blend_rate > 0.0:
            # The bristles along the brush's sides drag the wet paint across them as it glides: each cell by how
            # near it is to a side's line.
            radius = radius[..., None]
            near = distance <= radius + medium.blend_width
            sides = (1.0 - (lateral - radius).abs() / medium.blend_width).clamp_min(0.0) * near * on_sheet
            new_paint = self._blend_sides(new_paint, new_cover, index, (glide * per_diameter)[..., None] * sides)
        # A cell under several samples is written with the same values by each.
        self.under.scatter_(1, three, dry_layer)
        self.pigment.scatter_(1, four, new_paint)
        self.cover.scatter_(1, index, new_cover)
        self.stamp.scatter_(1, index, self.clock[:, None].expand_as(new_cover))
        self.painted.scatter_(1, index, self.painted.gather(1, index) | (new_cover > 0.0))
        self.under[:, self.pad] = self.paper
        for buffer in (self.pigment, self.cover, self.stamp, self.painted):
            buffer[:, self.pad] = 0
        self.brush.add_(pickup[:, None] * (picked - self.brush))

    def _blend_sides(
        self, paint: torch.Tensor, cover: torch.Tensor, index: torch.Tensor, exposure: torch.Tensor
    ) -> torch.Tensor:
        """The wet ``paint (B, S W, 4)`` of the cells ``index (B, S W)`` in reach of each sample, ``cover (B, S W)``
        of them wet, after the brush's sides drag it: each cell's paint mixed toward the paint around it as much as
        its ``exposure (B, S, W)`` to the sides, summed over the run, and as freely as it and the paint around it
        move. Paint moves as freely as the square of how wet it is: freely when fresh, hardly at all once mostly
        dry, so a stroke keeps its edge against paint laid minutes before.
        """
        n, samples, size = *exposure.shape[:2], self._window_size
        free = cover**2
        # The paint around each cell of each sample's window, by how freely it moves; and how freely that is.
        weighted = torch.cat([paint * free[..., None], free[..., None]], dim=-1)
        # Each channel of each window smoothed as an image of its own (much faster than a grouped convolution).
        weighted = weighted.view(n * samples, size, size, 5).permute(0, 3, 1, 2).reshape(-1, 1, size, size)
        pad = self._smoothing.shape[-1] // 2
        weighted = torch.nn.functional.conv2d(weighted, self._smoothing, padding=(0, pad))
        weighted = torch.nn.functional.conv2d(weighted, self._smoothing.transpose(2, 3).contiguous(), padding=(pad, 0))
        weighted = weighted.view(n * samples, 5, size, size).permute(0, 2, 3, 1).reshape(n, samples * size * size, 5)
        around = weighted[..., :4] / weighted[..., 4:].clamp_min(1e-12)
        # Summed over the run: the paint around each cell by its exposure, and the exposure.
        exposure = exposure.flatten(1) * free * weighted[..., 4]
        five = index[..., None].expand(-1, -1, 5)
        self._blend.scatter_add_(1, five, torch.cat([around * exposure[..., None], exposure[..., None]], dim=-1))
        summed = self._blend.gather(1, five)
        self._blend.scatter_(1, five, 0.0)
        around = summed[..., :4] / summed[..., 4:].clamp_min(1e-12)
        mixed = 1.0 - torch.exp(-self.medium.blend_rate * summed[..., 4:])
        return paint + mixed * (around - paint)

    def paint(self, rgb: torch.Tensor, painted: torch.Tensor) -> None:
        """Set the sheet to dry paint of ``rgb (B, rows, cols, 3)`` on the cells ``painted (B, rows, cols)``, bare
        paper elsewhere: for setting up a sheet (tests, a pre-painted start).
        """
        painted = painted.reshape(self.num_envs, -1)
        self.under[:, : self.pad] = torch.where(
            painted[..., None], srgb_to_linear(rgb.reshape(self.num_envs, -1, 3)), self.paper
        )
        self.painted[:, : self.pad] = painted
        self.cover.zero_()

    def reset(self, envs_idx) -> None:
        # A bool env mask is applied with torch.where and scalar fills, neither of which syncs; a gather or a tensor
        # written through the mask would (see eden.utils.misc.env_indices).
        if is_env_mask(envs_idx):
            torch.where(envs_idx[:, None, None], self.paper, self.under, out=self.under)
            torch.where(envs_idx[:, None], self._start_heading, self._heading, out=self._heading)
        else:
            self.under[envs_idx] = self.paper
            self._heading[envs_idx] = self._start_heading
        brush = (self.brush, self.load, self._last_radius, self._has_last)
        for buffer in (self.pigment, self.cover, self.stamp, self.painted, self.clock, *brush):
            buffer[envs_idx] = 0
