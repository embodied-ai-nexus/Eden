"""The works the calligraphy tasks copy: each a :class:`~eden.tasks.art.calligraphy.references.Plan` of the tool's tip
(see :mod:`~eden.tasks.art.calligraphy.references` for what a work is and how its target is drawn).

Works (the tasks' :data:`WORKS`):

* ``dai`` 大 (big, 3 strokes), ``san`` 山 (mountain, 3 strokes) and ``ei`` 永 (eternity, 5 strokes, the
  character of the "eight principles" calligraphers practise every basic stroke on), in kaisho (楷書, regular
  script) as Japanese schools teach it, 12 cm, with the Chinese brush. Each is written in the standard Japanese
  stroke order and direction, and its shape is the Klee One font's (Fontworks, SIL Open Font License 1.1), a
  教科書体 (textbook) kaisho: key points authored along each glyph's strokes with a brush's pressure (pressed
  in at the start, lighter through the stroke, pressed to a stop or lifted into a sweep), then moved, at most
  4.8 mm, to fit the ink the brush model lays along them to the glyph (IoU 0.90, 0.86 and 0.81).
* ``minimum`` — the word *minimum*, the copybooks' drill for the pointed pen's shaded downstrokes, in a round
  hand with a 1.4 cm x-height slanted 55 degrees, with the pointed pen.
"""

from __future__ import annotations

import math
from types import MappingProxyType

from eden.tasks.art.calligraphy.references import Plan, Work
from eden.tasks.art.plans import Stroke
from eden.tasks.art.workcell import SLANT_DEG, pen_lateral

# --- the works ---------------------------------------------------------------------------------------------------


def _character(strokes: dict[str, list[tuple[float, float, float]]], size: float) -> tuple[Stroke, ...]:
    """Strokes authored in a unit box (``x`` right, ``y`` up, depth in mm), scaled to a ``size``-m box centred on
    the sheet's origin.
    """
    return tuple(
        Stroke(name, tuple(((x - 0.5) * size, (y - 0.5) * size, z / 1000.0) for x, y, z in points))
        for name, points in strokes.items()
    )


CHARACTER_SIZE = 0.12
HOVER = -3.0  # mm: the tip's height above the paper as a stroke comes in and lifts off

# The kaisho characters, fitted to Klee One SemiBold's glyphs (see the module docstring): each glyph's ink box
# centred in the unit box, whose side is the font's em.

_DAI = {
    "heng": [
        (0.11, 0.615, HOVER),
        (0.14, 0.59, 2.25),
        (0.235, 0.59, 5.5),
        (0.365, 0.6, 6.5),
        (0.525, 0.61, 6.5),
        (0.685, 0.62, 6.5),
        (0.805, 0.625, 6.5),
        (0.91, 0.625, 6.5),
        (0.915, 0.59, HOVER),
    ],
    "pie": [
        (0.45, 0.935, HOVER),
        (0.49, 0.875, 3.5),
        (0.485, 0.77, 7.0),
        (0.475, 0.665, 7.0),
        (0.445, 0.54, 7.0),
        (0.41, 0.41, 7.0),
        (0.32, 0.285, 6.75),
        (0.22, 0.18, 6.0),
        (0.12, 0.12, 4.25),
        (0.02, 0.06, 2.5),
        (0.05, 0.075, HOVER),
    ],
    "na": [
        (0.515, 0.555, HOVER),
        (0.5, 0.55, 3.25),
        (0.57, 0.45, 4.75),
        (0.64, 0.35, 6.0),
        (0.715, 0.255, 6.75),
        (0.79, 0.19, 7.5),
        (0.87, 0.135, 7.25),
        (0.955, 0.085, 7.0),
        (0.99, 0.14, HOVER),
    ],
}

_SAN = {
    "shu": [
        (0.495, 0.92, HOVER),
        (0.495, 0.93, 4.0),
        (0.502, 0.805, 5.55),
        (0.507, 0.595, 7.0),
        (0.502, 0.385, 6.75),
        (0.512, 0.185, 5.75),
        (0.522, 0.195, HOVER),
    ],
    "shu_zhe": [
        (0.155, 0.645, HOVER),
        (0.19, 0.62, 2.5),
        (0.2, 0.515, 5.75),
        (0.198, 0.375, 7.0),
        (0.2, 0.235, 6.75),
        (0.215, 0.135, 6.0),
        (0.335, 0.165, 6.0),
        (0.5, 0.19, 6.25),
        (0.665, 0.205, 6.25),
        (0.805, 0.22, 5.5),
        (0.8, 0.24, HOVER),
    ],
    "shu_right": [
        (0.795, 0.665, HOVER),
        (0.825, 0.63, 3.5),
        (0.83, 0.545, 5.5),
        (0.825, 0.365, 6.75),
        (0.815, 0.185, 7.0),
        (0.82, 0.065, 7.25),
        (0.845, 0.07, HOVER),
    ],
}

_EI = {
    "dian": [
        (0.36, 0.925, HOVER),
        (0.395, 0.915, 4.0),
        (0.47, 0.885, 5.25),
        (0.55, 0.85, 6.0),
        (0.635, 0.81, 6.25),
        (0.65, 0.78, HOVER),
    ],
    "heng_zhe_gou": [
        (0.225, 0.735, HOVER),
        (0.255, 0.71, 2.0),
        (0.35, 0.72, 5.0),
        (0.445, 0.73, 6.25),
        (0.53, 0.73, 6.5),
        (0.555, 0.675, 6.5),
        (0.52, 0.52, 6.75),
        (0.52, 0.33, 7.0),
        (0.52, 0.14, 7.0),
        (0.49, 0.045, 6.0),
        (0.42, 0.095, 5.0),
        (0.36, 0.15, 4.0),
        (0.3, 0.205, 3.0),
        (0.3, 0.2, HOVER),
    ],
    "heng_pie": [
        (0.11, 0.56, HOVER),
        (0.14, 0.545, 2.75),
        (0.235, 0.545, 4.75),
        (0.325, 0.55, 6.0),
        (0.405, 0.545, 6.25),
        (0.435, 0.51, 6.0),
        (0.355, 0.43, 5.75),
        (0.27, 0.34, 5.75),
        (0.185, 0.25, 5.0),
        (0.1, 0.18, 3.75),
        (0.02, 0.12, 2.5),
        (0.055, 0.13, HOVER),
    ],
    "pie": [
        (0.85, 0.675, HOVER),
        (0.84, 0.65, 2.5),
        (0.785, 0.6, 4.0),
        (0.73, 0.55, 5.25),
        (0.68, 0.515, 4.5),
        (0.63, 0.48, 3.75),
        (0.585, 0.5, HOVER),
    ],
    "na": [
        (0.54, 0.585, HOVER),
        (0.545, 0.57, 4.0),
        (0.6, 0.49, 4.5),
        (0.66, 0.41, 5.0),
        (0.73, 0.33, 5.5),
        (0.8, 0.265, 6.0),
        (0.875, 0.21, 6.5),
        (0.95, 0.155, 7.0),
        (1.005, 0.19, HOVER),
    ],
}

# --- round hand, with the pointed pen ------------------------------------------------------------------------------
#: Lettering's rule (the ``minimum`` work's :attr:`~eden.tasks.art.calligraphy.references.Work.rule`): RoboPaint's
#: boundary F (0.8) and stroke coverage (0.75), but an IoU of 0.7 where RoboPaint's is 0.85. A pen's hairline is two
#: 0.2 mm cells wide, so a copy that wanders 0.1 mm either side of it inks half as much again: the scripted writer's
#: copies through the arm's 20 Hz IK, faithful by eye and on every other rule, score 0.80 at its 1.5 cm/s and 0.77
#: at 2 cm/s. RoboPaint also registers a copy to its target (within 5 mm) before scoring it; this grading does not.
LETTERING_RULE = MappingProxyType({"iou": 0.7, "boundary_f": 0.8, "min_stroke_coverage": 0.75})
X_HEIGHT = 0.014
PEN_HOVER = -2.0  # mm
HAIR, SHADE = 0.8, 3.5  # mm: a hairline's depth (the tines closed), a full shade's (the tines fully spread)


def _downstroke(x: float, top: float, bottom: float = 0.0) -> list[tuple[float, float, float]]:
    """A shaded downstroke at ``x`` from ``top`` to ``bottom`` (x-heights): swelling in, full, easing off."""
    span = top - bottom
    return [
        (x, top - 0.1 * span, SHADE * 0.6),
        (x, top - 0.25 * span, SHADE),
        (x, bottom + 0.35 * span, SHADE),
        (x, bottom + 0.15 * span, SHADE * 0.55),
    ]


def _underturn(x: float) -> list[tuple[float, float, float]]:
    """The hairline turn from a downstroke at ``x`` round the baseline and up to the right."""
    return [(x + 0.05, 0.04, 1.0), (x + 0.17, 0.0, HAIR), (x + 0.32, 0.1, HAIR), (x + 0.45, 0.35, HAIR)]


def _overturn(x: float, to: float) -> list[tuple[float, float, float]]:
    """The hairline branch from a downstroke at ``x`` (retraced up from the baseline) over the top to ``to``."""
    return [
        (x, 0.2, HAIR),
        (x + 0.04, 0.55, HAIR),
        (x + 0.16, 0.88, HAIR),
        ((x + to) / 2 + 0.03, 1.0, HAIR),
        (to - 0.05, 0.95, 1.2),
    ]


_ENTRY = [(-0.12, 0.25, PEN_HOVER), (0.0, 0.35, HAIR), (0.1, 0.65, HAIR), (0.2, 0.92, HAIR), (0.25, 1.0, 1.2)]


def _arches(n: int) -> list[tuple[float, float, float]]:
    """``n`` (2 for n, 3 for m): the entry, ``n`` shaded downstrokes joined by overturns, and the underturn out."""
    points = list(_ENTRY)
    for k in range(n):
        x = 0.27 + 0.5 * k
        points += _downstroke(x, 1.0)
        points += [(x, 0.0, 1.0)] if k < n - 1 else []
        points += _overturn(x, x + 0.5) if k < n - 1 else _underturn(x)
    return points + [(0.27 + 0.5 * (n - 1) + 0.55, 0.5, PEN_HOVER)]


def _u() -> list[tuple[float, float, float]]:
    points = list(_ENTRY) + _downstroke(0.27, 1.0) + _underturn(0.27)[:-1]
    points += [(0.68, 0.6, HAIR), (0.75, 0.9, HAIR), (0.78, 1.0, 1.2)] + _downstroke(0.78, 1.0) + _underturn(0.78)
    return points + [(1.3, 0.5, PEN_HOVER)]


def _i() -> list[tuple[float, float, float]]:
    return list(_ENTRY) + _downstroke(0.27, 1.0) + _underturn(0.27) + [(0.8, 0.5, PEN_HOVER)]


def _dot(x: float) -> list[tuple[float, float, float]]:
    return [(x + 0.18, 1.5, PEN_HOVER), (x + 0.24, 1.56, 1.5), (x + 0.3, 1.62, 2.5), (x + 0.36, 1.68, PEN_HOVER)]


#: Each letter's strokes and advance (x-heights).
_LETTERS = {"m": ([_arches(3)], 1.6), "n": ([_arches(2)], 1.1), "u": ([_u()], 1.1), "i": ([_i()], 0.6)}


def _round_hand(word: str, x_height: float) -> tuple[Stroke, ...]:
    """``word`` in the round hand above, ``x_height`` m high, centred on the sheet's origin: each letter one stroke,
    each i's dot one more, written after the word.
    """
    lean = 1.0 / math.tan(math.radians(SLANT_DEG))
    strokes, dots, x = [], [], 0.0
    for i, char in enumerate(word):
        paths, advance = _LETTERS[char]
        strokes += [(f"{char} {i}", [(x + px, py, pz) for px, py, pz in path]) for path in paths]
        if char == "i":
            dots.append((f"dot {i}", _dot(x)))
        x += advance
    width = x + lean
    return tuple(
        Stroke(
            name,
            tuple(((px + py * lean - width / 2) * x_height, (py - 0.5) * x_height, pz / 1000.0) for px, py, pz in path),
        )
        for name, path in strokes + dots
    )


WORKS: dict[str, Work] = {
    "dai": Work(
        "dai",
        "大",
        Plan("brush", _character(_DAI, CHARACTER_SIZE)),
        sheet=(0.18, 0.18),
        cell=0.0005,
        about="大 (big), kaisho, 3 strokes: horizontal, left-falling, right-falling",
    ),
    "san": Work(
        "san",
        "山",
        Plan("brush", _character(_SAN, CHARACTER_SIZE)),
        sheet=(0.18, 0.18),
        cell=0.0005,
        about="山 (mountain), kaisho, 3 strokes: vertical, vertical-turn, vertical",
    ),
    "ei": Work(
        "ei",
        "永",
        Plan("brush", _character(_EI, CHARACTER_SIZE)),
        sheet=(0.18, 0.18),
        cell=0.0005,
        about="永 (eternity), kaisho, 5 strokes: dot, horizontal-turn-hook, horizontal-left-falling, "
        "left-falling, right-falling",
    ),
    "minimum": Work(
        "minimum",
        "minimum",
        Plan("pen", _round_hand("minimum", X_HEIGHT)),
        sheet=(0.16, 0.06),
        cell=0.0002,
        lateral=pen_lateral(),
        rule=LETTERING_RULE,
        about="minimum, the copybooks' drill for shaded downstrokes, in round hand: 7 letters and 2 dots, 9 strokes",
    ),
}
