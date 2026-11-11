"""Tests for ``eden.utils.plane_grid.PlaneGrid``: row 0 at the top, ``(u0, v0)`` the top-left corner."""

import numpy as np

from eden.utils.plane_grid import PlaneGrid


def test_a_point_falls_in_the_cell_whose_centre_is_nearest_it():
    grid = PlaneGrid(u0=-0.02, v0=0.01, cell=0.005, rows=4, cols=8)
    centers = grid.centers()
    assert centers.shape == (4, 8, 2)
    np.testing.assert_allclose(centers[0, 0], (-0.0175, 0.0075))  # top-left: v falls as rows go down
    np.testing.assert_allclose(centers[3, 7], (0.0175, -0.0075))
    assert grid.cell_of(centers).tolist() == np.stack(np.indices(grid.shape), axis=-1).tolist()
    assert grid.cell_of(np.array([-0.021, 0.0])).tolist() == [2, -1]  # off the grid: an index outside it
