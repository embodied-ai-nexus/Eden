"""Tests for ``eden.extensions.retargeting.flashdexretarget.sdf``: the object SDF grid bake."""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.retargeting.flashdexretarget.sdf import bake_object_sdf_grid, cache_path
from tests.extensions.retargeting.flashdexretarget.conftest import write_box

pytest.importorskip("warp", reason="the SDF bake needs warp-lang (the flashdexretarget extra)")

EXTENT, N = 0.2, 41  # grid spacing 1 cm, the origin on a grid point


def test_box_grid_holds_signed_distance_and_outward_gradient(tmp_path):
    mesh = tmp_path / "box" / "visual.obj"
    write_box(mesh, extents=(0.1, 0.1, 0.1))
    grid = bake_object_sdf_grid(str(mesh), 1.0, EXTENT, N, device="cpu").numpy()
    assert grid.shape == (4, N, N, N)
    c = N // 2  # the box centre
    assert grid[0, c, c, c] == pytest.approx(-0.05, abs=1e-5)  # inside: minus the distance to the nearest face
    out = c + 10  # x = +0.10 m, 5 cm outside the +x face
    assert grid[0, out, c, c] == pytest.approx(0.05, abs=1e-5)
    np.testing.assert_allclose(grid[1:, out, c, c], (1.0, 0.0, 0.0), atol=1e-5)
    assert cache_path(str(mesh), EXTENT, N, 1.0).exists()


def test_a_changed_mesh_gets_a_grid_of_its_own(tmp_path):
    """The cache is keyed on the mesh's bytes: a re-imported mesh under the same id must not load the old grid."""
    mesh = tmp_path / "box" / "visual.obj"
    write_box(mesh, extents=(0.1, 0.1, 0.1))
    small = cache_path(str(mesh), EXTENT, N, 1.0)
    write_box(mesh, extents=(0.2, 0.2, 0.2))
    assert cache_path(str(mesh), EXTENT, N, 1.0) != small
    c = N // 2
    assert bake_object_sdf_grid(str(mesh), 1.0, EXTENT, N, device="cpu")[0, c, c, c] == pytest.approx(-0.1, abs=1e-5)
