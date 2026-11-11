"""A plan of the tool's tip, stroke by stroke: the format every art work is written in.

A stroke's ``points`` are key points ``(u, v, depth)`` of the tip in the sheet frame (m; ``u`` to the right, ``v``
up the page, from the sheet's centre), joined by a centripetal Catmull-Rom spline through them; ``depth`` is how far
below the paper the tip goes, negative in the air. Every stroke starts and ends in the air: its first key points
bring the tool down onto the paper (the way it comes in sets the way a brush's hairs trail), its last ones lift it
off. Between strokes the tip rises to :data:`TRAVEL_HEIGHT`, crosses, and comes down to the next stroke's start,
by way of any *stops* the plan makes there (a painting's dip into a paint well).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Spacing (m) of the tip's samples along a plan: RoboPaint's brush sub-step, 0.25 mm.
SAMPLE_SPACING = 0.00025
#: How high (m) above the paper the tip travels between strokes.
TRAVEL_HEIGHT = 0.01
#: :func:`tip_path`'s owner of a sample on the way between strokes.
TRAVEL = -1
#: :func:`tip_path`'s owner of a sample in a stop between strokes.
STOP = -2


@dataclass(frozen=True)
class Stroke:
    """One stroke of a plan: ``points`` ``(n, 3)`` key points ``(u, v, depth)`` of the tip, in the air at both ends."""

    name: str
    points: tuple[tuple[float, float, float], ...]

    def path(self, spacing: float = SAMPLE_SPACING) -> np.ndarray:
        """The tip's path ``(m, 3)``: the Catmull-Rom spline through :attr:`points`, about ``spacing`` apart."""
        return catmull_rom(np.asarray(self.points, dtype=np.float64), spacing)


def segment(a, b, spacing: float) -> np.ndarray:
    """The straight line from ``a`` to ``b`` (both ends included), about ``spacing`` apart."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = max(1, math.ceil(np.linalg.norm(b - a) / spacing))
    return a + (b - a) * np.linspace(0.0, 1.0, n + 1)[:, None]


def catmull_rom(points: np.ndarray, spacing: float) -> np.ndarray:
    """The centripetal Catmull-Rom spline through ``points (n, d)``, sampled about ``spacing`` apart along it.

    The ends are extended by reflecting their neighbours, so the curve starts and ends at the first and last points.
    """
    if len(points) < 2:
        raise ValueError(f"catmull_rom: needs at least 2 points, got {len(points)}.")
    p = np.concatenate([2 * points[:1] - points[1:2], points, 2 * points[-1:] - points[-2:-1]])
    out = [p[1:2]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** 0.5
        n = max(2, math.ceil(np.linalg.norm(p2 - p1) / spacing))
        t = np.linspace(t1, t2, n + 1)[1:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.concatenate(out)


def tip_path(
    strokes: Sequence[Stroke],
    *,
    stops: Sequence[np.ndarray | None] | None = None,
    spacing: float = SAMPLE_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """The tip's whole path ``(m, 3)`` through ``strokes`` in order, travel between them included, and what each
    sample belongs to ``(m,)``: its stroke's index, :data:`TRAVEL` on the way between strokes, or :data:`STOP`.

    ``stops[k]`` (if given and not ``None``) is a path ``(n, 3)`` the tip follows on its way to stroke ``k``, after
    rising from the stroke before (or, for the first stroke, from where the path starts: the stop's own first point):
    it is reached and left at :data:`TRAVEL_HEIGHT`, like a stroke.
    """
    stops = [None] * len(strokes) if stops is None else list(stops)
    if len(stops) != len(strokes):
        raise ValueError(f"tip_path: {len(stops)} stops for {len(strokes)} strokes; give one (or None) per stroke.")
    parts: list[np.ndarray] = []
    owner: list[np.ndarray] = []

    def travel_to(start: np.ndarray) -> None:
        """Rise from where the path is, cross at TRAVEL_HEIGHT and come down to ``start``."""
        end = parts[-1][-1]
        up, over = (*end[:2], -TRAVEL_HEIGHT), (*start[:2], -TRAVEL_HEIGHT)
        legs = [segment(a, b, spacing)[1:] for a, b in ((end, up), (up, over), (over, start))]
        travel = np.concatenate(legs)[:-1]
        parts.append(travel)
        owner.append(np.full(len(travel), TRAVEL))

    for k, stroke in enumerate(strokes):
        stop = stops[k]
        if stop is not None:
            stop = np.asarray(stop, dtype=np.float64)
            if parts:
                travel_to(stop[0])
            parts.append(stop)
            owner.append(np.full(len(stop), STOP))
        path = stroke.path(spacing)
        if parts:
            travel_to(path[0])
        parts.append(path)
        owner.append(np.full(len(path), k))
    return np.concatenate(parts), np.concatenate(owner)


def path_times(
    path: np.ndarray, owner: np.ndarray, *, draw_speed: float, travel_speed: float, turn_radius: float = 0.0
) -> np.ndarray:
    """When ``(m,)`` (s, from the first) the tip is at each sample of ``path (m, 3)``, owned as ``owner (m,)``
    (:func:`tip_path`'s): a stroke's or a stop's samples go at ``draw_speed``, slowed (down
    to a quarter) round turns tighter than ``turn_radius``; the rest at ``travel_speed``.
    """
    careful = owner != TRAVEL
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    slow = careful[1:] & careful[:-1]
    speed = np.where(slow, draw_speed, travel_speed)
    if turn_radius > 0.0:
        # Slower round tight turns: the speed at which the tip's sideways acceleration is that of a turn_radius
        # turn at draw_speed. The turn's radius is read over the segments within turn_radius either way.
        heading = np.arctan2(*np.diff(path[:, :2], axis=0).T[::-1])
        window = max(1, round(turn_radius / max(float(np.median(lengths[slow])), 1e-9))) if slow.any() else 1
        turn = np.abs(np.angle(np.exp(1j * (np.roll(heading, -window) - np.roll(heading, window)))))
        span = np.maximum(2 * window * lengths, 1e-9)
        radius = span / np.maximum(turn, 1e-9)
        speed = np.where(slow, draw_speed * np.clip(np.sqrt(radius / turn_radius), 0.25, 1.0), speed)
    return np.r_[0.0, np.cumsum(lengths / speed)]
