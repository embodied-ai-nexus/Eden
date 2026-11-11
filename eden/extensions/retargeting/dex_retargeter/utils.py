"""Helpers for the dex-retargeting port (e.g. low-pass filter)."""

from __future__ import annotations

import numpy as np


class LPFilter:
    """Exponential-moving-average low-pass filter.

    ``alpha`` close to 1 means almost-passthrough; small ``alpha`` produces
    heavier smoothing (and higher latency).
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.y: np.ndarray | None = None
        self.is_init: bool = False

    def next(self, x: np.ndarray) -> np.ndarray:
        if not self.is_init:
            self.y = np.array(x, copy=True)
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self) -> None:
        self.y = None
        self.is_init = False
