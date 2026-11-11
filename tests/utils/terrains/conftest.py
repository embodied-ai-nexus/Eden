"""Shared helpers for the terrain generator tests."""

from __future__ import annotations

import pytest
from genesis.ext.isaacgym import terrain_utils as tu

VS = 0.005


def _subterrain(size_x_m: float, size_y_m: float | None = None, hs: float = 0.25) -> tu.SubTerrain:
    size_y_m = size_x_m if size_y_m is None else size_y_m
    return tu.SubTerrain(
        width=int(size_x_m / hs) + 1, length=int(size_y_m / hs) + 1, vertical_scale=VS, horizontal_scale=hs
    )


@pytest.fixture(scope="session")
def subterrain():
    """``subterrain(size_x_m, size_y_m=None, hs=0.25)``: an empty ``SubTerrain`` of that extent, in cells + 1."""
    return _subterrain
