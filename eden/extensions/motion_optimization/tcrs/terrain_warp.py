"""Support-aware pelvis-z filtering for terrain reference warping.

Smooths a per-frame absolute pelvis-z target with an asymmetric EMA and a
per-frame rate limit, so the root tracks terrain height changes without step
discontinuities. The caller (the pipeline's pelvis-z retargeting stage)
constructs the target; this filter only smooths it.

Ported from Mondo-Robotics/PMT ``TCRS/stair_mppi/terrain_warp.py`` @ 88838ca
(the ``SupportAwareRootZFilter.step`` path only; upstream's legacy
``TerrainReferenceWarper`` / raw-foot-data filter path is dead code and was
not ported).
"""

from __future__ import annotations


class SupportAwareRootZFilter:
    """Smooth a precomputed absolute pelvis-z target with EMA + rate limit.

    The caller builds ``z_target`` externally; this filter applies an
    asymmetric EMA (faster rise than fall, faster still in flight) followed by
    a per-frame rate clamp.
    """

    def __init__(
        self,
        alpha_up: float = 0.35,
        alpha_down: float = 0.10,
        max_delta: float = 0.02,
        flight_alpha: float = 0.6,
    ):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.max_delta = max_delta
        self.flight_alpha = flight_alpha
        self._ema_z = None

    def reset(self):
        self._ema_z = None

    def step(self, z_target: float, is_flight: bool = False) -> float:
        """Smooth *one* externally-constructed z_target.

        Parameters
        ----------
        z_target : float
            Absolute pelvis z target for this frame.
        is_flight : bool
            If True, track with at least ``flight_alpha``.

        Returns
        -------
        float
            Smoothed pelvis z.
        """
        if self._ema_z is None:
            self._ema_z = float(z_target)
            return self._ema_z

        alpha = self.alpha_up if z_target > self._ema_z else self.alpha_down
        if is_flight:
            alpha = max(alpha, self.flight_alpha)

        ema_target = self._ema_z + alpha * (z_target - self._ema_z)
        dz = min(max(ema_target - self._ema_z, -self.max_delta), self.max_delta)
        self._ema_z = self._ema_z + dz
        return self._ema_z
