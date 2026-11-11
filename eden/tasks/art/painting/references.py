"""What a painting to copy is: a plan of the brush, stroke by stroke with its paint, and the target it paints.

RoboPaint's acrylic works are small paintings after public-domain masterpieces, each a target its reference robot
painted and a stroke budget; its stroke plans are not released. The works here are painted for Eden instead: each is
a :class:`Painting`, a palette and a plan of strokes in the art suite's format (:mod:`~eden.tasks.art.plans`), each
stroke with the paint it is made in, and its target is what the paint model (:mod:`~eden.tasks.art.painting.paint`)
lays along that plan (:func:`render_painting`). A copy can therefore match its target, to within how its samples
fall into runs (a target's are :data:`RUN` long, a copy's a control step's: the canvas keeps the difference to a few
hundredths of a channel), and a plan replayed through the robot (the scripted painter) shows the work can be
painted.

The brush is dipped before a stroke (:meth:`Painting.dips`) when the stroke needs another paint than the brush
holds, or when making it would leave the brush under the load's ``reserve``: so the target is painted at full
opacity throughout. A dip goes into the paint's well (:mod:`~eden.tasks.art.painting.palette`) on the way to the
stroke, as a stop of the tip's path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from eden.tasks.art.painting.paint import Acrylic, Canvas, PaintLoad, RoundBrush
from eden.tasks.art.painting.palette import DIP_DEPTH, well_centres
from eden.tasks.art.plans import SAMPLE_SPACING, Stroke, catmull_rom, path_times, tip_path
from eden.tasks.art.workcell import TOOLS, sheet_grid
from eden.utils.plane_grid import PlaneGrid
from eden.utils.procedural_meshes import write_atomically

#: How high (m) above the paper a sweep (:func:`sweep`) starts and ends.
HOVER = -0.003
#: How far (m) past its ends a sweep comes down and lifts off, in the air.
LEAD = 0.002
#: How far (m) a sweep runs level at its depth after it comes down and before it lifts off: the spline through the
#: key points would otherwise overshoot the depth past the touch-down (by 1.2 mm for a 16 mm brush; 0.1 mm so).
SETTLE = 0.001
#: How high (m) above a well the tip comes down from and goes back up to on a dip.
DIP_HOVER = 0.003
#: The pace a target is painted at, the scripted painter's: the paintbrush's draw speed (m/s) on the paper and in a
#: dip, and its travel speed (m/s) between.
DRAW_SPEED = TOOLS["paintbrush"].draw_speed
TRAVEL_SPEED = 0.1
#: The samples of the plan's path a target is laid in at a time: 2 mm of it, about what a copy's brush covers in a
#: control step at the draw speed.
RUN = 8

Rgb = tuple[float, float, float]


@dataclass(frozen=True)
class PaintStroke:
    """One stroke of a painting: the tip's :class:`~eden.tasks.art.plans.Stroke` and the palette index of its
    paint.
    """

    stroke: Stroke
    paint: int


def sweep(name: str, paint: int, v: float, u0: float, u1: float, size: float):
    """A horizontal sweep in ``paint`` at ``v`` from ``u0`` to ``u1`` (m), pressed to make a ``size``-m mark: it comes
    down from :data:`HOVER` :data:`LEAD` before ``u0``, runs level, and lifts off as far past ``u1``. Its key points
    are a ``size`` apart, rounded to 0.01 mm.
    """
    depth = RoundBrush().depth_for(size)
    direction = 1.0 if u1 >= u0 else -1.0
    settle = direction * min(SETTLE, abs(u1 - u0) / 3)
    knots = np.linspace(u0 + settle, u1 - settle, max(2, math.ceil(abs(u1 - u0) / size) + 1))
    points = [(u0 - direction * LEAD, v, HOVER), (u0, v, depth)]
    points += [(float(u), v, depth) for u in knots] + [(u1, v, depth), (u1 + direction * LEAD, v, HOVER)]
    return PaintStroke(Stroke(name, tuple((round(u, 5), round(v, 5), round(z, 5)) for u, v, z in points)), paint)


def swept_area(path: np.ndarray, brush: RoundBrush) -> np.ndarray:
    """The paper each sample of a tip ``path (m, 3)`` sweeps (m^2, ``(m,)``): the way it moved from the sample
    before, both on the paper, times the mark's diameter there. This is what uses up a brush's paint.
    """
    on = path[:, 2] > 0.0
    moved = np.r_[0.0, np.hypot(*np.diff(path[:, :2], axis=0).T)] * (on & np.r_[False, on[:-1]])
    return moved * brush.diameter_at(torch.as_tensor(path[:, 2])).numpy()


@dataclass(frozen=True)
class Painting:
    """A painting to copy: its palette, its strokes in painting order, the sheet and the grid its paint is kept on.

    ``palette`` is each paint's sRGB colour in ``[0, 1]``; well ``k`` of the palette tray holds paint ``k``.
    ``rule`` holds PaintingCopy parameters where this work's differ from the acrylic defaults (read-only, and out of
    the hash, so a work stays hashable). The brush and its load are the paint model's defaults
    (:class:`~eden.tasks.art.painting.paint.RoundBrush`, :class:`~eden.tasks.art.painting.paint.PaintLoad`, as a
    :class:`~eden.tasks.art.painting.paint.Canvas` holds them): its plan, its dips, its target and a copy of it are all
    made with the same brush.
    """

    name: str
    title: str
    palette: tuple[Rgb, ...]
    strokes: tuple[PaintStroke, ...]
    sheet: tuple[float, float] = (0.18, 0.18)
    cell: float = 0.0005
    rule: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    about: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        used = sorted({s.paint for s in self.strokes})
        if used and (used[0] < 0 or used[-1] >= len(self.palette)):
            raise ValueError(
                f"Painting {self.name!r}: strokes use paints {used}, but the palette has {len(self.palette)} "
                f"(indices 0-{len(self.palette) - 1})."
            )

    def painting(self) -> Painting:
        """The painting itself: an authored work is its own, where a planned one
        (:class:`~eden.tasks.art.painting.works.PlannedPainting`) is planned.
        """
        return self

    @property
    def grid(self) -> PlaneGrid:
        return sheet_grid(self.sheet, self.cell)

    @property
    def wells(self) -> np.ndarray:
        """``(len(palette), 2)``: each paint's well centre ``(u, v)`` in the sheet frame."""
        return well_centres(len(self.palette), self.sheet)

    def dips(self) -> tuple[bool, ...]:
        """Whether the brush is dipped before each stroke: before the first, before a change of paint, and before a
        stroke that would leave less than the load's ``reserve`` of a full brush.
        """
        brush, load = RoundBrush(), PaintLoad()
        dips, left, paint = [], 0.0, None
        for item in self.strokes:
            need = float(swept_area(item.stroke.path(), brush).sum()) / load.capacity
            dip = item.paint != paint or left - need < load.reserve
            if dip:
                left, paint = 1.0, item.paint
            left -= need
            dips.append(dip)
        return tuple(dips)

    def dip_path(self, paint: int) -> np.ndarray:
        """The tip's path ``(m, 3)`` of a dip into ``paint``'s well: straight down into it and back up."""
        u, v = self.wells[paint]
        keys = np.array([(u, v, -DIP_HOVER), (u, v, DIP_DEPTH), (u, v, -DIP_HOVER)])
        return catmull_rom(keys, SAMPLE_SPACING)

    def path(self, spacing: float = SAMPLE_SPACING) -> tuple[np.ndarray, np.ndarray]:
        """The tip's whole path ``(m, 3)``, dips and travel included, and what each sample belongs to ``(m,)``:
        :func:`~eden.tasks.art.plans.tip_path` of the strokes, with a dip as the stop before each stroke that
        needs one.
        """
        stops = [self.dip_path(s.paint) if dip else None for s, dip in zip(self.strokes, self.dips())]
        return tip_path([s.stroke for s in self.strokes], stops=stops, spacing=spacing)


@dataclass(frozen=True)
class PaintingRendering:
    """A painting painted along its plan: the target and what the hand rules are scaled from."""

    #: ``(rows, cols, 3)``: the target's colour (sRGB in ``[0, 1]``).
    rgb: torch.Tensor
    #: ``(rows, cols)``: the cells the target has paint on.
    painted: torch.Tensor
    #: The tip's path on the paper (m), the mark's mean width along it (m), and the dips the plan makes.
    path_length: float
    mark_width: float
    dips: int


def render_painting(work: Painting, *, medium: Acrylic = Acrylic(), device="cpu") -> PaintingRendering:
    """Paint ``work``'s plan, dipping as :meth:`Painting.dips` does, on a fresh canvas: its target.

    The plan's path is painted at the scripted painter's pace (:data:`DRAW_SPEED`, :data:`TRAVEL_SPEED`), so its
    paint dries between strokes as a copy's does at that pace, in runs of :data:`RUN` samples, about the brush's
    path in a control step. The canvas uses up the brush's load as a copy's does (:meth:`Canvas.lay`); the plan's
    dips keep it above the load's ``reserve``, so the target is painted at full opacity. ``medium`` is for tests
    that paint a plan as it is planned (dry between strokes); a task's target and copy use the default.
    """
    canvas = Canvas(work.grid, 1, medium=medium, device=device)
    palette = torch.tensor(work.palette, dtype=torch.float32, device=device)
    every = torch.ones(1, dtype=torch.bool, device=device)
    path, owner = work.path()
    times = path_times(path, owner, draw_speed=DRAW_SPEED, travel_speed=TRAVEL_SPEED)
    dips = work.dips()
    length, width_path, clock = 0.0, 0.0, 0.0
    for k, (item, dip) in enumerate(zip(work.strokes, dips)):
        if dip:
            canvas.dip(every, palette[item.paint][None])
        samples = np.flatnonzero(owner == k)
        stroke = path[samples]
        uv = torch.as_tensor(stroke[None, :, :2], dtype=torch.float32, device=device)
        depth = torch.as_tensor(stroke[None, :, 2], dtype=torch.float32, device=device)
        for a in range(0, len(samples), RUN):
            canvas.tick(float(times[samples[a]]) - clock)
            clock = float(times[samples[a]])
            canvas.lay(uv[:, a : a + RUN], depth[:, a : a + RUN])
        on = stroke[:, 2] > 0.0
        moved = np.r_[0.0, np.hypot(*np.diff(stroke[:, :2], axis=0).T)] * (on & np.r_[False, on[:-1]])
        length += float(moved.sum())
        width_path += float(swept_area(stroke, canvas.round_brush).sum())
    return PaintingRendering(
        rgb=canvas.rgb()[0],
        painted=canvas.painted_cells()[0],
        path_length=length,
        mark_width=width_path / max(length, 1e-12),
        dips=sum(dips),
    )


def save_painting(work: Painting, path: str | Path) -> None:
    """Write ``work``'s palette and plan to ``path`` as JSON (its ``rule`` stays with the work's recipe), whole: a
    reader never sees a partial plan.
    """
    data = {
        "name": work.name,
        "title": work.title,
        "about": work.about,
        "sheet": list(work.sheet),
        "cell": work.cell,
        "palette": [list(rgb) for rgb in work.palette],
        "strokes": [{"paint": s.paint, "points": [list(p) for p in s.stroke.points]} for s in work.strokes],
    }
    write_atomically(str(path), json.dumps(data, separators=(",", ":")) + "\n")


def load_painting(path: str | Path, **overrides) -> Painting:
    """The :class:`Painting` :func:`save_painting` wrote to ``path``; ``overrides``
    replace its fields (a work's ``rule``).
    """
    data = json.loads(Path(path).read_text())
    strokes = tuple(
        PaintStroke(Stroke(str(k), tuple(tuple(p) for p in s["points"])), s["paint"])
        for k, s in enumerate(data["strokes"])
    )
    fields = {
        "name": data["name"],
        "title": data["title"],
        "palette": tuple(tuple(rgb) for rgb in data["palette"]),
        "strokes": strokes,
        "sheet": tuple(data["sheet"]),
        "cell": data["cell"],
        "about": data["about"],
    }
    return Painting(**(fields | overrides))
