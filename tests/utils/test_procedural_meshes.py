"""Tests for eden.utils.procedural_meshes: the tet lattice and the sheet the Mochi tasks mesh their bodies with."""

import numpy as np
import pytest

from eden.utils.procedural_meshes import (
    box_tet_lattice,
    box_tet_lattice_file,
    grid_sheet,
    grid_sheet_file,
    swept_tube,
)


def _tet_volumes(verts, tets):
    p = verts[tets]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def test_box_tet_lattice_fills_the_box_with_positive_tets():
    verts, tets = box_tet_lattice((0.075, 0.05, 0.03), (6, 4, 3))
    assert verts.shape == (7 * 5 * 4, 3) and tets.shape == (6 * 4 * 3 * 5, 4)
    volumes = _tet_volumes(verts, tets)
    assert (volumes > 0.0).all()
    np.testing.assert_allclose(volumes.sum(), 0.075 * 0.05 * 0.03, rtol=1e-12)
    np.testing.assert_allclose(verts.min(axis=0), (-0.0375, -0.025, -0.015))
    np.testing.assert_allclose(verts.max(axis=0), (0.0375, 0.025, 0.015))


def test_box_tet_lattice_is_conforming():
    """Every interior triangle is shared by exactly two tets; only the box's surface triangles by one."""
    _, tets = box_tet_lattice((1.0, 1.0, 1.0), (3, 2, 2))
    faces = np.sort(tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1)
    _, count = np.unique(faces, axis=0, return_counts=True)
    assert set(count.tolist()) == {1, 2}
    # 2 triangles per surface quad: 2 * (3*2 + 3*2 + 2*2) quads per box.
    assert (count == 1).sum() == 2 * 2 * (3 * 2 + 3 * 2 + 2 * 2)


def test_grid_sheet_is_flat_and_wound_upward():
    verts, faces = grid_sheet((0.3, 0.2), (3, 2))
    assert verts.shape == (4 * 3, 3) and faces.shape == (2 * 3 * 2, 3)
    assert (verts[:, 2] == 0.0).all()
    normals = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    assert (normals[:, 2] > 0.0).all()
    np.testing.assert_allclose(0.5 * normals[:, 2].sum(), 0.3 * 0.2)


@pytest.mark.parametrize("bad", [(0, 1, 1), (1, -1, 1)])
def test_box_tet_lattice_rejects_an_empty_axis(bad):
    with pytest.raises(ValueError, match="at least one cell per axis"):
        box_tet_lattice((1.0, 1.0, 1.0), bad)


def test_mesh_files_are_written_once_into_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    vtk = box_tet_lattice_file((0.1, 0.1, 0.1), (1, 1, 1))
    obj = grid_sheet_file((0.1, 0.1), (2, 2))
    assert vtk.startswith(str(tmp_path)) and obj.startswith(str(tmp_path))
    text = open(vtk).read()
    assert "POINTS 8 double" in text and "CELLS 5 25" in text
    assert sum(line.startswith("f ") for line in open(obj)) == 8
    # A second call with the same parameters names the same file and leaves it alone.
    stamp = (tmp_path / vtk).stat().st_mtime_ns
    assert box_tet_lattice_file((0.1, 0.1, 0.1), (1, 1, 1)) == vtk
    assert (tmp_path / vtk).stat().st_mtime_ns == stamp


def test_swept_tube_is_closed_wound_outward_and_as_thick_as_its_radius():
    # An elbow: 10 cm straight, a quarter turn of 3 cm radius, 10 cm straight, in the y-z plane.
    turn = [(0.0, 0.1 + 0.03 * np.sin(a), 0.03 - 0.03 * np.cos(a)) for a in np.linspace(0.0, 0.5 * np.pi, 9)]
    path = np.array([(0.0, 0.0, 0.0), *turn, (0.0, 0.13, 0.13)])
    radius = 0.01
    verts, faces = swept_tube(path, radius, sides=32)
    # Closed: every edge is shared by exactly two faces.
    edges = np.sort(np.concatenate([faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]]), axis=1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    assert (count == 2).all()
    # Wound outward, and the volume of a tube of that length (the polygon section is a hair under the circle's).
    p = verts[faces]
    volume = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0
    length = 0.1 + 0.5 * np.pi * 0.03 + 0.1
    assert volume == pytest.approx(np.pi * radius**2 * length, rel=0.02)
    # Every ring vertex lies one radius off the centreline.
    ring = verts[: len(path) * 32].reshape(len(path), 32, 3)
    np.testing.assert_allclose(np.linalg.norm(ring - path[:, None, :], axis=-1), radius, rtol=1e-9)


def test_swept_tube_rejects_a_repeated_point():
    with pytest.raises(ValueError, match="repeats a point at index 1"):
        swept_tube(np.array([(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.1, 0.0)]), 0.01)
