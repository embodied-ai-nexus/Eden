"""How the planned paintings are made: a layered stroke planner over a source picture, run on the paint model.

The planner paints a picture the way RoboPaint's acrylic works are painted — first the ground in broad strokes,
then the forms, then the details, finer and finer — and records the plan; the work's target is then that plan
painted (``references.render_painting``). It is a picture in and a plan out: which pictures the works are planned
from, and where their plans are kept, is ``works.py``'s.

1. The picture is cropped to the ``crop`` box (pixels, of the sheet's aspect) and resampled onto the sheet's grid.
2. Its palette is ``paints`` colours, k-means in CIELAB over its cells; each paint is the mean colour of the cells
   nearest it.
3. Each layer has a brush size. The picture is blurred in proportion to it, and every cell is labelled with the
   paint nearest its blurred colour (the labels themselves smoothed, so a region is at least about a brush wide).
   Where the canvas so far is further than the layer's ``threshold`` (ΔE) from the blurred picture, each paint's
   region is swept with horizontal strokes ``spacing`` brush widths apart, inset from the region's ends, each at
   most as long as a brush's paint lasts above its reserve. The ground layer paints its regions lightest paint first;
   later layers darkest first, so their light details (clouds, snow) go on top. Within a paint, each stroke starts
   at the one nearest where the last ended.

Every stroke is a :func:`~eden.tasks.art.painting.references.sweep`: it comes down before its start, runs level at the
depth that makes its brush size, and lifts off past its end. Sweeps, never scribbles. The planner lets each stroke
dry before the next (and fills the brush afresh for it): it plans for the picture, and how the plan's strokes blend
wet is the target's (``references.render_painting``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eden.tasks.art.painting.grading import blur, gaussian_kernel, srgb_to_lab
from eden.tasks.art.painting.paint import Canvas
from eden.tasks.art.painting.references import Painting, PaintStroke, sweep
from eden.tasks.art.workcell import sheet_grid


@dataclass(frozen=True)
class Layer:
    """One layer of the plan: its brush ``size`` (m), the ΔE ``threshold`` past which a cell is repainted, the
    strokes' ``spacing`` and ``inset`` (in brush widths), and whether its paints go on ``lightest_first``.
    """

    size: float
    threshold: float
    spacing: float = 0.8
    inset: float = 0.25
    lightest_first: bool = False


#: The acrylic layers: RoboPaint's four acrylic brush sizes, 16, 9.4, 5.2 and 3 mm.
ACRYLIC_LAYERS = (
    Layer(0.016, 6.0, inset=0.15, lightest_first=True),
    Layer(0.0094, 8.0),
    Layer(0.0052, 12.0),
    Layer(0.003, 17.0),
)


def load_picture(
    path: str | Path, sheet: tuple[float, float], cell: float, crop: tuple[int, int, int, int]
) -> torch.Tensor:
    """The ``crop`` box ``(left, top, right, bottom)`` (pixels) of the picture at ``path``, resampled onto the sheet's
    grid: ``(rows, cols, 3)`` sRGB in ``[0, 1]``. The box must lie in the picture and have the sheet's aspect (to
    within 1%), so the picture is not stretched.
    """
    from PIL import Image

    grid = sheet_grid(sheet, cell)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    left, top, right, bottom = crop
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"load_picture: crop={crop} is not a box inside {Path(path).name}, which is {width} x {height} pixels; "
            "give (left, top, right, bottom) with 0 <= left < right <= width and 0 <= top < bottom <= height."
        )
    aspect, box_aspect = sheet[0] / sheet[1], (right - left) / (bottom - top)
    if abs(box_aspect / aspect - 1.0) > 0.01:
        raise ValueError(
            f"load_picture: crop={crop} is {right - left} x {bottom - top} pixels (aspect {box_aspect:.3f}), but the "
            f"sheet {sheet} has aspect {aspect:.3f}: the picture would be stretched. Give a box of the sheet's aspect."
        )
    image = image.crop(crop).resize((grid.cols, grid.rows), Image.LANCZOS)
    return torch.as_tensor(np.array(image), dtype=torch.float32) / 255.0


def make_palette(picture: torch.Tensor, paints: int, *, seed: int = 0, iterations: int = 30) -> torch.Tensor:
    """``(paints, 3)``: the picture's palette (see the module docstring), sRGB, darkest first."""
    lab = srgb_to_lab(picture).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    centres = lab[torch.as_tensor(rng.choice(len(lab), paints, replace=False))]
    for _ in range(iterations):
        nearest = torch.cdist(lab, centres).argmin(dim=1)
        centres = torch.stack(
            [lab[nearest == k].mean(0) if (nearest == k).any() else centres[k] for k in range(paints)]
        )
    nearest = torch.cdist(lab, centres).argmin(dim=1)
    rgb = picture.reshape(-1, 3)
    # A paint no cell is nearest (its centre never drew a cell: the picture has few colours) is the cell nearest it.
    colours = torch.stack(
        [
            rgb[nearest == k].mean(0) if (nearest == k).any() else rgb[(lab - centres[k]).norm(dim=1).argmin()]
            for k in range(paints)
        ]
    )
    return colours[torch.argsort(srgb_to_lab(colours)[:, 0])]  # darkest first


def _labels(lab: torch.Tensor, palette_lab: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Each cell's nearest paint, the votes smoothed by a Gaussian of ``smoothing`` cells."""
    nearest = torch.cdist(lab.reshape(-1, 3), palette_lab).argmin(dim=1).view(lab.shape[:2])
    votes = F.one_hot(nearest, len(palette_lab)).float().permute(2, 0, 1)[None]
    return blur(votes, gaussian_kernel(smoothing, lab.device))[0].argmax(dim=0)


def _sweeps(mask: np.ndarray, *, spacing: float, inset: float, min_run: float, max_run: float):
    """Horizontal runs ``(row, col0, col1)`` (cells) over ``mask``, ``spacing`` rows apart, inset ``inset`` at
    either end, at least ``min_run`` long and split to at most ``max_run``.
    """
    runs = []
    row = spacing / 2
    while row < mask.shape[0]:
        line = mask[int(row)]
        edges = np.flatnonzero(np.diff(np.r_[0, line.astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            a, b = start + inset, stop - 1 - inset
            if b - a >= min_run:
                pieces = max(1, math.ceil((b - a) / max_run))
                cuts = np.linspace(a, b, pieces + 1)
                runs += [(row, cuts[i], cuts[i + 1]) for i in range(pieces)]
        row += spacing
    return runs


def _nearest_first(runs: list, start: tuple[float, float]) -> list:
    """``runs`` reordered so each starts nearest where the one before ended (a sweep may be reversed)."""
    left, ordered, at = list(runs), [], np.asarray(start, dtype=np.float64)
    while left:
        ends = np.array([[(r, c0), (r, c1)] for r, c0, c1 in left], dtype=np.float64)  # (n, 2 ends, 2)
        distance = np.linalg.norm(ends - at, axis=-1)
        index, end = np.unravel_index(int(distance.argmin()), distance.shape)
        row, c0, c1 = left.pop(int(index))
        if end == 1:
            c0, c1 = c1, c0
        ordered.append((row, c0, c1))
        at = np.array([row, c1])
    return ordered


def plan_painting(
    picture: torch.Tensor,
    *,
    name: str,
    title: str,
    paints: int,
    layers: Sequence[Layer] = ACRYLIC_LAYERS,
    sheet: tuple[float, float] = (0.18, 0.18),
    cell: float = 0.0005,
    about: str = "",
) -> Painting:
    """The plan of ``picture`` (``(rows, cols, 3)`` on the sheet's grid, :func:`load_picture`), as a
    :class:`~eden.tasks.art.painting.references.Painting`.
    """
    grid = sheet_grid(sheet, cell)
    if tuple(picture.shape[:2]) != grid.shape:
        raise ValueError(f"plan_painting: the picture is {tuple(picture.shape[:2])}, the sheet's grid {grid.shape}.")
    palette = make_palette(picture, paints)
    palette_lab = srgb_to_lab(palette)
    canvas = Canvas(grid, 1, device="cpu")
    load = canvas.paint_load
    every = torch.ones(1, dtype=torch.bool)
    strokes: list[PaintStroke] = []
    at = (0.0, 0.0)
    for layer in layers:
        radius = layer.size / 2 / cell  # cells
        reference = srgb_to_lab(
            blur(picture.permute(2, 0, 1)[None], gaussian_kernel(0.5 * radius, "cpu"))[0].permute(1, 2, 0)
        )
        needs = (srgb_to_lab(canvas.rgb()[0]) - reference).norm(dim=-1) > layer.threshold
        labels = _labels(reference, palette_lab, radius)
        longest = 0.9 * (1.0 - load.reserve) * load.capacity / layer.size / cell  # cells a load sweeps, with margin
        order = reversed(range(paints)) if layer.lightest_first else range(paints)
        for paint in order:
            mask = ((labels == paint) & needs).numpy()
            runs = _sweeps(
                mask,
                spacing=layer.spacing * 2 * radius,
                inset=layer.inset * 2 * radius,
                min_run=0.5 * radius,
                max_run=longest,
            )
            for row, c0, c1 in _nearest_first(runs, at):
                v = grid.v0 - (row + 0.5) * cell
                u0, u1 = grid.u0 + (c0 + 0.5) * cell, grid.u0 + (c1 + 0.5) * cell
                item = sweep(f"{len(strokes)}", paint, v, u0, u1, layer.size)
                strokes.append(item)
                path = item.stroke.path()
                canvas.dip(every, palette[paint][None])
                uv = torch.as_tensor(path[None, :, :2], dtype=torch.float32)
                canvas.lay(uv, torch.as_tensor(path[None, :, 2], dtype=torch.float32))
                canvas.dry()
                at = (row, c1)
    return Painting(
        name=name,
        title=title,
        palette=tuple(tuple(round(float(c), 4) for c in rgb) for rgb in palette),
        strokes=tuple(strokes),
        sheet=sheet,
        cell=cell,
        about=about,
    )
