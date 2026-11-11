"""The whiteboard tasks' target drawings: a single-stroke alphabet and the strokes each task asks for.

Ported from RoboWorld's ``whiteboard`` environment (``whiteboard/stroke_font.py`` and
``whiteboard/task_set/catalogue.py`` of ``gym-anything/robotics``, MIT). A stroke is
``(name, weight, points)``: ``weight`` is ``"normal"`` (the marker's nominal 3 mm line) or ``"thick"`` (6 mm), and
``points`` is a polyline in the board's writing plane, ``(u, v)`` metres from the drawing's centre — ``u`` to the
right as the robot faces the board, ``v`` up. The original writes in world ``(x, z)`` with ``+x`` to the viewer's
left and mirrors the glyphs to match; ``u`` runs to the right, so here nothing is mirrored.

:func:`raster` draws strokes the way the original rasterizes its reference image (PIL lines with round joints,
``round(width / cell)`` pixels wide), which is what the text tasks' F1 is measured against.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

#: Line width (m) of each stroke weight: the marker's nominal 3 mm line, and twice that for a table's frame.
LINE_WIDTH = {"normal": 0.003, "thick": 0.006}

_O = [(0.4 + 0.4 * math.cos(i * math.tau / 48), 0.5 + 0.5 * math.sin(i * math.tau / 48)) for i in range(49)]

#: The original's single-stroke alphabet: each glyph a list of polylines in a 0.8 x 1 cell, y up.
GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "A": [[(0, 0), (0.4, 1), (0.8, 0)], [(0.18, 0.45), (0.62, 0.45)]],
    "B": [
        [(0, 0), (0, 1), (0.55, 1), (0.8, 0.8), (0.55, 0.55), (0, 0.55)],
        [(0.55, 0.55), (0.8, 0.25), (0.55, 0), (0, 0)],
    ],
    "C": [[(0.8, 0.9), (0.6, 1), (0.2, 1), (0, 0.8), (0, 0.2), (0.2, 0), (0.6, 0), (0.8, 0.1)]],
    "D": [[(0, 0), (0, 1), (0.5, 1), (0.8, 0.75), (0.8, 0.25), (0.5, 0), (0, 0)]],
    "E": [[(0.8, 1), (0, 1), (0, 0), (0.8, 0)], [(0, 0.5), (0.65, 0.5)]],
    "F": [[(0, 0), (0, 1), (0.8, 1)], [(0, 0.5), (0.65, 0.5)]],
    "G": [[(0.8, 0.9), (0.6, 1), (0.2, 1), (0, 0.8), (0, 0.2), (0.2, 0), (0.8, 0), (0.8, 0.45), (0.4, 0.45)]],
    "H": [[(0, 0), (0, 1)], [(0.8, 0), (0.8, 1)], [(0, 0.5), (0.8, 0.5)]],
    "I": [[(0, 1), (0.8, 1)], [(0.4, 1), (0.4, 0)], [(0, 0), (0.8, 0)]],
    "J": [[(0.8, 1), (0.8, 0.2), (0.6, 0), (0.2, 0), (0, 0.2)]],
    "K": [[(0, 0), (0, 1)], [(0.8, 1), (0, 0.5), (0.8, 0)]],
    "L": [[(0, 1), (0, 0), (0.8, 0)]],
    "M": [[(0, 0), (0, 1), (0.4, 0.45), (0.8, 1), (0.8, 0)]],
    "N": [[(0, 0), (0, 1), (0.8, 0), (0.8, 1)]],
    "O": [_O],
    "P": [[(0, 0), (0, 1), (0.6, 1), (0.8, 0.8), (0.6, 0.55), (0, 0.55)]],
    "Q": [
        [(0.1, 0.1), (0, 0.3), (0, 0.8), (0.2, 1), (0.6, 1), (0.8, 0.8), (0.8, 0.2), (0.6, 0), (0.1, 0.1)],
        [(0.5, 0.3), (0.9, -0.05)],
    ],
    "R": [[(0, 0), (0, 1), (0.6, 1), (0.8, 0.8), (0.6, 0.55), (0, 0.55)], [(0.4, 0.55), (0.8, 0)]],
    "S": [
        [(0.8, 0.9), (0.6, 1), (0.2, 1), (0, 0.8), (0.2, 0.55), (0.6, 0.45), (0.8, 0.2), (0.6, 0), (0.2, 0), (0, 0.1)]
    ],
    "T": [[(0, 1), (0.8, 1)], [(0.4, 1), (0.4, 0)]],
    "U": [[(0, 1), (0, 0.2), (0.2, 0), (0.6, 0), (0.8, 0.2), (0.8, 1)]],
    "V": [[(0, 1), (0.4, 0), (0.8, 1)]],
    "W": [[(0, 1), (0.2, 0), (0.4, 0.6), (0.6, 0), (0.8, 1)]],
    "X": [[(0, 0), (0.8, 1)], [(0, 1), (0.8, 0)]],
    "Y": [[(0, 1), (0.4, 0.5), (0.8, 1)], [(0.4, 0.5), (0.4, 0)]],
    "Z": [[(0, 1), (0.8, 1), (0, 0), (0.8, 0)]],
    "0": [[(0, 0.2), (0, 0.8), (0.2, 1), (0.6, 1), (0.8, 0.8), (0.8, 0.2), (0.6, 0), (0.2, 0), (0, 0.2)]],
    "1": [[(0.2, 0.8), (0.4, 1), (0.4, 0)], [(0.1, 0), (0.7, 0)]],
    "2": [[(0, 0.8), (0.2, 1), (0.6, 1), (0.8, 0.8), (0, 0), (0.8, 0)]],
    "3": [[(0, 1), (0.6, 1), (0.8, 0.8), (0.5, 0.55), (0.8, 0.3), (0.6, 0), (0, 0)]],
    "4": [[(0.6, 0), (0.6, 1), (0, 0.35), (0.8, 0.35)]],
    "5": [[(0.8, 1), (0, 1), (0, 0.55), (0.6, 0.55), (0.8, 0.35), (0.8, 0.2), (0.6, 0), (0, 0)]],
    "6": [[(0.7, 1), (0.3, 0.9), (0, 0.5), (0, 0.2), (0.2, 0), (0.6, 0), (0.8, 0.2), (0.6, 0.5), (0, 0.5)]],
    "7": [[(0, 1), (0.8, 1), (0.2, 0)]],
    "8": [
        [
            (0.2, 0.5),
            (0, 0.75),
            (0.2, 1),
            (0.6, 1),
            (0.8, 0.75),
            (0.6, 0.5),
            (0.2, 0.5),
            (0, 0.25),
            (0.2, 0),
            (0.6, 0),
            (0.8, 0.25),
            (0.6, 0.5),
        ]
    ],
    "9": [[(0.8, 0.5), (0.2, 0.5), (0, 0.75), (0.2, 1), (0.6, 1), (0.8, 0.75), (0.8, 0.4), (0.5, 0)]],
    " ": [],
    "-": [[(0.1, 0.5), (0.7, 0.5)]],
    "+": [[(0, 0.5), (0.8, 0.5)], [(0.4, 0.1), (0.4, 0.9)]],
    ".": [[(0.4, 0), (0.4, 0.02)]],
    "!": [[(0.4, 1), (0.4, 0.25)], [(0.4, 0), (0.4, 0.02)]],
}

Stroke = tuple[str, str, tuple[tuple[float, float], ...]]


def text_strokes(text: str, width: float, center: tuple[float, float] = (0.0, 0.0)) -> tuple[Stroke, ...]:
    """``text`` in the alphabet, ``width`` metres from the first glyph's left to the last one's right.

    The original's layout: a glyph advances 1.1 cells, so the line spans ``1.1 * len(text) - 0.3`` cells, and the
    cell height is ``width`` over that span. Each polyline is one stroke named ``letter <i>``.
    """
    unknown = sorted({char for char in text if char not in GLYPHS})
    if unknown:
        raise ValueError(f"text_strokes: no glyph for {unknown} in {text!r}; the alphabet is {''.join(GLYPHS)}.")
    span = len(text) * 1.1 - 0.3
    cu, cv = center
    return tuple(
        (
            f"letter {i}",
            "normal",
            tuple((cu + ((i * 1.1 + x) / span - 0.5) * width, cv + (y - 0.5) / span * width) for x, y in path),
        )
        for i, char in enumerate(text)
        for path in GLYPHS[char]
    )


def table_strokes(width: float, center: tuple[float, float] = (0.0, 0.0)) -> tuple[Stroke, ...]:
    """The original's blank table, ``width`` metres wide: a thick frame, a header line and two column lines."""
    cu, cv = center
    lines = (
        ("top", "thick", ((-0.5, 0.3), (0.5, 0.3))),
        ("bottom", "thick", ((-0.5, -0.3), (0.5, -0.3))),
        ("left", "thick", ((-0.5, -0.3), (-0.5, 0.3))),
        ("right", "thick", ((0.5, -0.3), (0.5, 0.3))),
        ("header", "normal", ((-0.5, 0.2), (0.5, 0.2))),
        ("column one", "normal", ((-1 / 6, -0.3), (-1 / 6, 0.3))),
        ("column two", "normal", ((1 / 6, -0.3), (1 / 6, 0.3))),
    )
    return tuple(
        (name, weight, tuple((cu + x * width, cv + y * width) for x, y in points)) for name, weight, points in lines
    )


def strokes_bounds(strokes: tuple[Stroke, ...]) -> tuple[float, float, float, float]:
    """``(u_min, v_min, u_max, v_max)`` of every stroke's points (m), line widths not included."""
    points = np.asarray([p for _, _, path in strokes for p in path], dtype=np.float64)
    return (*points.min(axis=0).tolist(), *points.max(axis=0).tolist())


def raster(
    strokes: tuple[Stroke, ...], *, origin: tuple[float, float], cell: float, shape: tuple[int, int]
) -> np.ndarray:
    """Boolean ``shape`` image of ``strokes`` on a grid of ``cell``-metre cells whose top-left corner is ``origin``.

    Row 0 is the top (``v`` = ``origin[1]``, decreasing down the rows); column 0 is the left edge (``u`` =
    ``origin[0]``). Lines are drawn as the original draws its reference: ``round(LINE_WIDTH / cell)`` pixels wide
    (at least one), round joints.
    """
    height, width = shape
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    u0, v0 = origin
    for _, weight, path in strokes:
        pixels = [((u - u0) / cell, (v0 - v) / cell) for u, v in path]
        draw.line(pixels, fill=1, width=max(1, round(LINE_WIDTH[weight] / cell)), joint="curve")
    return np.asarray(image, dtype=bool)


def sample_strokes(strokes: tuple[Stroke, ...], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``spacing`` metres along every stroke, in drawing order, and the stroke each belongs to.

    Returns ``(points (n, 2), stroke_index (n,))``; each stroke contributes both its ends.
    """
    points, index = [], []
    for k, (_, _, path) in enumerate(strokes):
        path = np.asarray(path, dtype=np.float64)
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        s = np.linspace(0.0, arc[-1], max(2, math.ceil(arc[-1] / spacing) + 1))
        points.append(np.c_[np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])
        index.append(np.full(len(s), k))
    return np.concatenate(points), np.concatenate(index)


#: The original's three drawing tasks' references, at the original's sizes, centred on the drawing centre:
#: ``write-text`` (HELLO, 0.60 m), ``write-in-colour`` (WELCOME, 0.70 m) and ``draw-table`` (0.60 m).
DRAWINGS: dict[str, tuple[Stroke, ...]] = {
    "hello": text_strokes("HELLO", 0.60),
    "welcome": text_strokes("WELCOME", 0.70),
    "table": table_strokes(0.60),
}
