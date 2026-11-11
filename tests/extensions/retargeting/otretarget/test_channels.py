"""Tests for ``eden.extensions.retargeting.otretarget.channels``: SDF/witness grids and geodesics."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh

from eden.extensions.retargeting.otretarget.channels import (
    MeshChannel,
    PlaneChannel,
    SurfaceGeodesics,
    winding_sign_is_reliable,
)

#: ``MeshChannel`` builds libigl's winding-number tree, 2.6.3+, which the ``ot-retargeter`` extra pins; genesis-world
#: alone installs an older libigl (as the extra-less suites do the ``proxsuite`` tests, these skip there).
requires_winding_tree = pytest.mark.skipif(
    not hasattr(igl, "FastWindingNumberBVH"), reason="MeshChannel needs libigl >= 2.6.3 (the ot-retargeter extra)"
)


@pytest.fixture(scope="module")
def box():
    return MeshChannel("box", trimesh.creation.box((0.4, 0.3, 0.2)), max_distance=0.15, resolution=0.01)


def test_plane_reads_height_and_vertical_projection():
    ch = PlaneChannel("ground", 0.15)
    q = ch.query(np.array([[0.3, -0.2, 0.07], [1.0, 2.0, -0.01]]))
    np.testing.assert_allclose(q.distance, [0.07, -0.01])
    np.testing.assert_allclose(q.witness, [[0.3, -0.2, 0.0], [1.0, 2.0, 0.0]])
    rho, grad = ch.geodesic(np.array([[0.0, 0.0, 0.0]]), np.array([[0.3, 0.4, 0.0]]))
    np.testing.assert_allclose(rho, [0.5])
    np.testing.assert_allclose(grad, [[-0.6, -0.8, 0.0]])


@requires_winding_tree
def test_box_grid_matches_the_box_distance(box):
    # Points facing a box face, where the signed distance is exactly the offset from that face.
    pts = np.array([[0.25, 0.0, 0.0], [0.0, -0.2, 0.03], [0.0, 0.0, 0.05], [0.15, 0.0, 0.0]])
    expected = np.array([0.05, 0.05, -0.05, -0.05])  # faces at x=0.2, y=0.15, z=0.1; the last two inside
    q = box.query(pts)
    assert q.valid.all()
    np.testing.assert_allclose(q.distance, expected, atol=2e-3)
    np.testing.assert_allclose(q.witness[0], [0.2, 0.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(q.distance_grad[0], [1.0, 0.0, 0.0], atol=1e-2)


@requires_winding_tree
def test_box_grid_gradients_match_finite_differences(box):
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.12, 0.12, (100, 3)) + np.array([0.0, 0.0, 0.17])  # above the top face
    q = box.query(x)
    eps = 1e-6
    fd = np.stack([(box.query(x + eps * e).witness - box.query(x - eps * e).witness) / (2 * eps) for e in np.eye(3)], 2)
    np.testing.assert_allclose(q.witness_jac, fd, atol=1e-6)


@requires_winding_tree
def test_points_beyond_the_band_are_invalid(box):
    q = box.query(np.array([[1.0, 0.0, 0.0]]))
    assert not q.valid[0]


def test_geodesic_on_a_sphere_follows_great_circles():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.2)
    geo = SurfaceGeodesics.from_mesh(sphere, density=4000.0)
    rng = np.random.default_rng(1)
    u = rng.normal(size=(20, 3))
    v = rng.normal(size=(20, 3))
    u = 0.2 * u / np.linalg.norm(u, axis=1, keepdims=True)
    v = 0.2 * v / np.linalg.norm(v, axis=1, keepdims=True)
    exact = 0.2 * np.arccos(np.clip((u * v).sum(1) / 0.04, -1, 1))
    rho, _ = geo.distance(u, v)
    # Graph shortest paths over-estimate by a few percent on a sampled surface.
    np.testing.assert_allclose(rho, exact, rtol=0.12, atol=0.01)


def test_geodesic_does_not_cut_through_a_thin_plate():
    plate = trimesh.creation.box((0.3, 0.3, 0.01))
    geo = SurfaceGeodesics.from_mesh(plate, density=8000.0)
    top, bottom = np.array([[0.0, 0.0, 0.005]]), np.array([[0.0, 0.0, -0.005]])
    rho, _ = geo.distance(top, bottom, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]]))
    # Around the edge: at least half the plate width each way, never the 1 cm thickness.
    assert rho[0] > 0.25


def test_geodesic_on_a_plate_thinner_than_the_sample_spacing_keeps_each_end_on_its_face():
    # 1 cm plate sampled ~2.2 cm apart: within the interpolation support of a point on the top face lie samples
    # of the bottom face. Given the outward normals, each end interpolates its own face only, so the distance and
    # its gradient are those of the top face (a gradient in the tens of thousands when both faces blend).
    plate = trimesh.creation.box((0.3, 0.3, 0.01))
    geo = SurfaceGeodesics.from_mesh(plate, density=2000.0)
    assert geo.spacing > 0.02
    rng = np.random.default_rng(3)
    w = np.column_stack([rng.uniform(-0.1, 0.1, (20, 2)), np.full(20, 0.005)])
    w_hat = np.column_stack([rng.uniform(-0.1, 0.1, (20, 2)), np.full(20, 0.005)])
    up = np.tile([0.0, 0.0, 1.0], (20, 1))
    rho, grad = geo.distance(w, w_hat, up, up)
    np.testing.assert_allclose(rho, np.linalg.norm(w - w_hat, axis=1), rtol=0.15, atol=0.01)
    # A distance is 1-Lipschitz; the table interpolated over 2.2 cm samples stays within a small factor of that.
    assert np.linalg.norm(grad[:, :2], axis=1).max() < 3.0
    h = 1e-5
    for e in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        fd = (geo.distance(w + h * e, w_hat, up, up)[0] - geo.distance(w - h * e, w_hat, up, up)[0]) / (2 * h)
        np.testing.assert_allclose(grad @ e, fd, atol=1e-4)


def test_geodesic_at_the_sparsely_sampled_end_of_a_thin_slat_has_a_bounded_gradient():
    # The two sides of a 1 cm slat sampled 1.5 cm apart and its end face only every 6 cm (as on a chair slat at the
    # table's density). Around a point on the end face the sample normals are mostly the two opposite sides', and their
    # interpolation nearly cancels; the agreement of the end normals divided its gradient by that length (343 here, 8e5
    # on a chair). The softened normals bound that division by 1 / soften.
    y, z = np.meshgrid(np.arange(-0.15, 0.151, 0.015), np.arange(-0.05, 0.051, 0.015))
    side = np.column_stack([y.ravel(), z.ravel()])
    end_y = np.arange(-0.15, 0.151, 0.06)
    points = np.concatenate(
        [np.column_stack([np.full(len(side), s), side]) for s in (0.005, -0.005)]
        + [np.column_stack([np.zeros(len(end_y)), end_y, np.full(len(end_y), -0.055)])]
    )
    normals = np.concatenate(
        [np.tile([s, 0.0, 0.0], (len(side), 1)) for s in (1.0, -1.0)] + [np.tile([0.0, 0.0, -1.0], (len(end_y), 1))]
    )
    geo = SurfaceGeodesics(points, normals, spacing=0.015)
    rng = np.random.default_rng(0)
    n = 2000
    w = np.column_stack([rng.uniform(-0.005, 0.005, n), rng.uniform(-0.1, 0.1, n), np.full(n, -0.055)])
    w_hat = w + np.column_stack([rng.uniform(-0.004, 0.004, n), rng.uniform(-0.008, 0.008, n), np.zeros(n)])
    down = np.tile([0.0, 0.0, -1.0], (n, 1))
    _, grad = geo.distance(w, w_hat, down, down)
    _, unsoftened = geo.distance(w, w_hat, down, down, soften=1e-12)
    worst = np.linalg.norm(grad, axis=1).max()
    assert worst < 0.5 * np.linalg.norm(unsoftened, axis=1).max()
    assert np.linalg.norm(geo.distance(w, w_hat, down, down, soften=0.25)[1], axis=1).max() < worst
    # ... and it is the gradient of the distance it reports.
    h = 1e-6
    for e in np.eye(3):
        fd = (geo.distance(w + h * e, w_hat, down, down)[0] - geo.distance(w - h * e, w_hat, down, down)[0]) / (2 * h)
        np.testing.assert_allclose(grad @ e, fd, atol=1e-2)  # a 1e-6 step can straddle a change of neighbour set


@requires_winding_tree
def test_exact_query_matches_libigl_signed_distance(box):
    import igl

    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.3, 0.3, (300, 3))
    mesh = box.mesh
    ref, _, ref_w, _ = igl.signed_distance(
        pts,
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
    )
    q = box.exact_query(pts)
    # Same closest points; the cached winding-number tree decides the same sides.
    np.testing.assert_allclose(q.witness, ref_w, atol=1e-9)
    np.testing.assert_array_equal(np.sign(q.distance), np.sign(ref))
    np.testing.assert_allclose(q.distance, ref, atol=5e-4)


@requires_winding_tree
def test_grid_misreads_the_inside_of_a_feature_a_few_cells_thick():
    # Why the retargeter reads probes near a mesh exactly (``exact_band``): 4 mm inside a 1.6 cm rod the
    # 1 cm grid's distance is off by over a millimetre (the exact distance is -4 mm).
    rod = trimesh.creation.cylinder(radius=0.008, height=0.4, sections=48)
    ch = MeshChannel("rod", rod, max_distance=0.15, resolution=0.01)
    angles = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    pts = np.column_stack([0.004 * np.cos(angles), 0.004 * np.sin(angles), np.linspace(-0.1, 0.1, 40)])
    exact = ch.exact_query(pts).distance
    np.testing.assert_allclose(exact, -0.004, atol=3e-4)
    assert np.abs(ch.query(pts).distance - exact).max() > 1e-3


@requires_winding_tree
def test_geodesic_with_cached_target_weights_is_bit_identical(box):
    # The retargeter computes the w_hat end once per frame and passes it back as `hat`; each row depends only on
    # its own point, so weights computed for a superset of rows, then indexed, must give the same distance.
    rng = np.random.default_rng(3)
    _, w_hat = box.exact_distance(rng.uniform(-0.3, 0.3, (40, 3)))
    _, w = box.exact_distance(rng.uniform(-0.3, 0.3, (40, 3)))
    n_hat = box.exact_query(1.02 * w_hat).distance_grad  # outward: just outside each witness
    rows = np.array([3, 7, 7, 21, 39])
    idx, alpha = box.geodesics.endpoint_weights(w_hat, n_hat)
    rho, grad = box.geodesic(w[rows], w_hat[rows], None, n_hat[rows])
    rho_c, grad_c = box.geodesic(w[rows], w_hat[rows], None, n_hat[rows], hat=(idx[rows], alpha[rows]))
    np.testing.assert_array_equal(rho_c, rho)
    np.testing.assert_array_equal(grad_c, grad)


def _open_bin(radius=0.1, height=0.2):
    """A single-sheet bin: a cylinder wall and a floor, open at the top (as the OMOMO trashcan scan is)."""
    can = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    top = can.face_normals[:, 2] > 0.9
    return trimesh.Trimesh(can.vertices, can.faces[~top], process=False)  # drop the lid


@requires_winding_tree
def test_an_open_bin_reads_a_free_cavity_and_a_continuous_distance_across_its_opening():
    scan = _open_bin()
    raw = MeshChannel("bin", scan, max_distance=0.15, resolution=0.01)
    solid = MeshChannel(
        "bin",
        scan,
        max_distance=0.15,
        resolution=0.01,
        solid_voxel=0.005,
    )
    centre, above = np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.12]])
    # The winding number of the open scan calls the empty cavity solid; the closed solid leaves it free space.
    assert raw.exact_distance(centre)[0][0] < 0.0
    assert solid.exact_distance(centre)[0][0] > 0.05
    # Crossing the opening plane (z = 0.1) the distance changes by the step, not by the 10 cm jump of the raw scan.
    below_open, above_open = np.array([[0.0, 0.0, 0.098]]), np.array([[0.0, 0.0, 0.102]])
    jump = abs(solid.exact_distance(above_open)[0][0] - solid.exact_distance(below_open)[0][0])
    assert jump < 0.01, jump
    assert solid.exact_distance(above)[0][0] > 0.0
    assert (
        solid.mesh.is_watertight
        or not (np.unique(np.sort(solid.mesh.edges, 1), axis=0, return_counts=True)[1] % 2).any()
    )


@requires_winding_tree
def test_a_closed_scan_of_loose_triangles_keeps_its_own_surface():
    # Closed in space but no triangle shares a vertex with another (how the OMOMO scans are stored): not watertight
    # as loaded. Closed into a solid anyway, its surface moved a voxel out and a hand in contact stayed that far off.
    box = trimesh.creation.box((0.2, 0.2, 0.2))
    loose = trimesh.Trimesh(box.vertices[box.faces].reshape(-1, 3), np.arange(36).reshape(-1, 3), process=False)
    assert not loose.is_watertight
    assert winding_sign_is_reliable(loose)
    ch = MeshChannel("box", loose, max_distance=0.15, resolution=0.01, solid_voxel=0.005)
    assert ch.mesh is loose and ch.source_mesh is loose
    assert ch.exact_distance(np.zeros((1, 3)))[0][0] < -0.05
    np.testing.assert_allclose(ch.exact_distance(np.array([[0.1, 0.0, 0.0]]))[0], 0.0, atol=1e-9)


@requires_winding_tree
def test_walls_scanned_as_zero_thickness_sheets_are_closed_into_a_solid():
    # Every face also stored the other way round (a sheet seen from both sides): the winding number is 0 everywhere,
    # so nothing is inside the walls and a probe passes through them unopposed.
    box = trimesh.creation.box((0.2, 0.2, 0.2))
    both = trimesh.Trimesh(box.vertices, np.r_[box.faces, box.faces[:, ::-1]], process=False)
    assert not winding_sign_is_reliable(both)
    ch = MeshChannel("box", both, max_distance=0.15, resolution=0.01, solid_voxel=0.005)
    assert ch.mesh is not both
    assert ch.exact_distance(np.array([[0.1, 0.0, 0.0]]))[0][0] < 0.0  # inside the closed wall


def test_a_mesh_channel_on_a_libigl_without_the_winding_number_tree_names_the_upgrade(monkeypatch):
    """genesis-world installs libigl unpinned; an older one used to fail deep in the build with an AttributeError."""
    monkeypatch.delattr(igl, "FastWindingNumberBVH", raising=False)
    with pytest.raises(
        ImportError, match=r"MeshChannel 'box' needs libigl >= 2\.6\.3 .*uv sync --inexact --extra ot-retargeter"
    ):
        MeshChannel("box", trimesh.creation.box((0.4, 0.3, 0.2)), max_distance=0.15, resolution=0.01)
