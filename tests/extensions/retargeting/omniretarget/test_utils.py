"""Tests for ``eden.extensions.retargeting.omniretarget.utils``: interaction-mesh adjacency and Laplacians."""

from __future__ import annotations

import numpy as np

from eden.extensions.retargeting.omniretarget.utils import (
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    get_adjacency_list,
)

# A unit right-corner tetrahedron and the one simplex that spans it.
TETRA = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Uniform weights: each vertex minus the mean of its three neighbours.
TETRA_LAPLACIAN = np.array(
    [
        [-1 / 3, -1 / 3, -1 / 3],
        [1.0, -1 / 3, -1 / 3],
        [-1 / 3, 1.0, -1 / 3],
        [-1 / 3, -1 / 3, 1.0],
    ]
)


def test_adjacency_joins_every_pair_of_vertices_that_share_a_tetrahedron():
    # Two tetrahedra glued on the face (1, 2, 3); the apexes 0 and 4 share no simplex.
    adj = get_adjacency_list(np.array([[0, 1, 2, 3], [4, 1, 2, 3]]), num_vertices=6)

    assert adj == [{1, 2, 3}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4}, {1, 2, 3}, set()]


def test_laplacian_coordinates_are_each_vertex_minus_its_neighbours_mean():
    L, coords = calculate_laplacian_coordinates(TETRA, get_adjacency_list(np.array([[0, 1, 2, 3]]), 4))

    np.testing.assert_allclose(coords, TETRA_LAPLACIAN)
    np.testing.assert_allclose(L @ TETRA, TETRA_LAPLACIAN)


def test_laplacian_coordinates_ignore_a_rigid_translation():
    # The retargeter matches Laplacians and pins translation separately through the root term,
    # so a mesh shifted as a whole must keep its Laplacian coordinates.
    L = calculate_laplacian_matrix(TETRA, get_adjacency_list(np.array([[0, 1, 2, 3]]), 4))

    np.testing.assert_allclose(L @ (TETRA + np.array([3.0, -2.0, 0.5])), L @ TETRA, atol=1e-12)
