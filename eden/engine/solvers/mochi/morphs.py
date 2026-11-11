"""The rod morph: a polyline the MochiSolver simulates as a discrete elastic rod."""

from typing import Any

import genesis as gs
import numpy as np
from genesis.options.morphs import Morph
from pydantic import Field, StrictBool, StrictInt


class Rod(Morph):
    """
    Morph for a rod (cable, rope, wire): a polyline of nodes with a circular cross-section, simulated as a discrete
    elastic rod by the MochiSolver.

    Parameters
    ----------
    points : array-like, shape (n_nodes, 3)
        Positions of the rod nodes in the morph frame, in order along the rod (at least 2).
    radius : float, optional
        Radius of the circular cross-section in meters. Default is 0.01.
    n_cross_section_segments : int, optional
        Number of segments of the tube drawn around the rod. Default is 12.
    is_closed_loop : bool, optional
        Whether the last node connects back to the first one. Default is False.
    pos : tuple, shape (3,), optional
        Position of the morph frame in meters. Default is (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        Orientation of the morph frame in degrees (scipy extrinsic x-y-z). Default is (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        Orientation of the morph frame as a w-x-y-z quaternion; takes precedence over `euler`. Default is None.
    """

    points: Any = None
    radius: float = 0.01
    n_cross_section_segments: StrictInt = Field(default=12, ge=3)
    is_closed_loop: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        try:
            points = np.array(self.points, dtype=np.float64)
        except (TypeError, ValueError):
            gs.raise_exception("`points` should be array-like to be converted to np.ndarray.")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            gs.raise_exception("`points` should be an array of shape (n_nodes, 3) with at least 2 nodes.")
        if not np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.0):
            gs.raise_exception("Consecutive rod nodes must be distinct.")
        if self.radius <= 0.0:
            gs.raise_exception("`radius` must be positive.")
        self.points = points

    def _identifier(self) -> str:
        return "rod"
