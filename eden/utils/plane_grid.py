"""A grid of square cells over a plane: the layout of a task's per-env marks on a flat surface.

A whiteboard's ink (``eden.tasks.whiteboard``) and a sheet's (``eden.tasks.art.calligraphy``) are kept on one, and a
camera paints them through a :class:`~eden.entities.camera_overlays.PaintedPlane`, which carries the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneGrid:
    """A ``rows x cols`` grid of ``cell``-metre cells over a plane's ``(u, v)`` frame, row 0 at the top.

    Cell ``(r, c)`` covers ``u in [u0 + c cell, u0 + (c + 1) cell]`` and ``v in [v0 - (r + 1) cell, v0 - r cell]``:
    ``(u0, v0)`` is the grid's top-left corner.
    """

    u0: float
    v0: float
    cell: float
    rows: int
    cols: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def cell_of(self, uv: np.ndarray) -> np.ndarray:
        """``(row, col)`` of the cell holding each point ``uv (..., 2)``; may lie outside the grid."""
        uv = np.asarray(uv, dtype=np.float64)
        return np.stack(
            [np.floor((self.v0 - uv[..., 1]) / self.cell), np.floor((uv[..., 0] - self.u0) / self.cell)], -1
        ).astype(np.int64)

    def centers(self) -> np.ndarray:
        """``(u, v)`` of every cell's centre, ``(rows, cols, 2)``."""
        u = self.u0 + (np.arange(self.cols) + 0.5) * self.cell
        v = self.v0 - (np.arange(self.rows) + 0.5) * self.cell
        return np.stack(np.meshgrid(u, v, indexing="xy"), axis=-1)
