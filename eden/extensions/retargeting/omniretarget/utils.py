"""Interaction-mesh helpers for OmniRetarget: Delaunay adjacency and Laplacian coordinates."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay


# ------------------------------------------------------------------
# Interaction mesh (Delaunay tetrahedralization)
# ------------------------------------------------------------------


def compute_delaunay_simplices(vertices: np.ndarray) -> np.ndarray:
    """Compute Delaunay tetrahedralization, returning simplex indices."""
    return Delaunay(vertices).simplices


def get_adjacency_list(tetrahedra: np.ndarray, num_vertices: int) -> list[set]:
    """Build an adjacency list from tetrahedra.

    Returns a list of *sets* (one per vertex) containing neighbour indices.
    """
    adj: list[set] = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return adj


# ------------------------------------------------------------------
# Laplacian matrix & coordinates
# ------------------------------------------------------------------


def calculate_laplacian_matrix(vertices: np.ndarray, adj_list: list[set]) -> sp.csr_matrix:
    """Compute the (N x N) Laplacian matrix (uniform weights, sparse CSR) of ``vertices``.

    ``adj_list`` holds each vertex's neighbours, e.g. from :func:`get_adjacency_list`.
    """
    rows, cols, data = [], [], []
    for i, neighbors in enumerate(adj_list):
        for n in neighbors:
            rows.append(i)
            cols.append(n)
            data.append(1)
    adjacency_matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(vertices), len(vertices)))

    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    inv_degree = sp.diags(1.0 / degrees)

    return sp.eye(len(vertices)) - inv_degree @ adjacency_matrix


def calculate_laplacian_coordinates(vertices: np.ndarray, adj_list: list[set]) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return ``(laplacian_matrix, laplacian_coordinates)``.

    Laplacian coordinates = L @ V, capturing per-vertex differential
    structure of the mesh.
    """
    L = calculate_laplacian_matrix(vertices, adj_list)
    return L, L @ vertices
