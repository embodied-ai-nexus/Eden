"""Shared TCRS fixtures: a synthetic stair height field, and the local G1 MJCF the IK solves against.

The stairs: a flat apron, then 3 steps of 0.1 m rise whose first riser is at x = 0.4, on a 0.02 m grid whose
corner [0, 0] sits at (-1, -1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain
from eden.utils.misc import get_assets_dir

_CELL = 0.02  # height-field cell size (m)
_ORIGIN = (-1.0, -1.0, 0.0)  # world position of grid corner [0, 0]
_STAIRS_START_X = 0.4  # world x of the first riser; flat apron before it
_STEP_RISE = 0.1
_N_STEPS = 3


def _stair_grid(step_depth: float = 0.2) -> np.ndarray:
    """``_N_STEPS`` ascending steps of ``_STEP_RISE`` rise and ``step_depth`` depth on a 5 m x 2 m grid."""
    grid = np.zeros((250, 100), dtype=np.float32)
    for k in range(_N_STEPS):
        grid[int((_STAIRS_START_X - _ORIGIN[0] + step_depth * k) / _CELL) :, :] = _STEP_RISE * (k + 1)
    return grid


@pytest.fixture(scope="session")
def stair_terrain():
    """``stair_terrain(step_depth=0.2, oob_height=0.0)`` -> a ``HeightFieldTerrain`` over :func:`_stair_grid`."""

    def make(step_depth: float = 0.2, oob_height: float = 0.0) -> HeightFieldTerrain:
        return HeightFieldTerrain(_stair_grid(step_depth), _CELL, origin=_ORIGIN, oob_height=oob_height)

    return make


@pytest.fixture(scope="session")
def g1_mjcf() -> Path:
    """The G1 MJCF, probed directly because ``get_asset_path`` would download it on a miss."""
    path = Path(get_assets_dir("humanoids", "unitree_g1", "g1_29dof_rev_1_0.xml"))
    if not path.is_file():
        pytest.skip(f"G1 MJCF asset not downloaded ({path})")
    return path
