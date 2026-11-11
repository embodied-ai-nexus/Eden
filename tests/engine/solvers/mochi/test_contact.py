"""Penalty contact in the MochiSolver: the response and its derivatives, collider distance fields, the contact records
read back for soft bodies, and the self-contact of shells and rods.
"""

import genesis as gs
import numpy as np
import pytest
import trimesh
from genesis.utils.misc import tensor_to_array

from eden.engine.solvers.mochi import MochiOptions, Rod, create_mochi_scene
from eden.engine.solvers.mochi import materials as mochi_materials
from tests.engine.solvers.mochi.conftest import make_static_config, plane_collision_response
from tests.support.asserts import assert_allclose

CONTACT_PARAMS = {
    "coulomb": (1e9, 5e-3, 1e-3, 0.5, 1e-2, 0.0, 0.0, 0.0),
    "viscous": (1e9, 5e-3, 1e-3, 0.0, 1e-2, 100.0, 0.0, 0.0),
    "damping": (1e9, 5e-3, 1e-3, 0.3, 1e-2, 0.0, 50.0, 0.0),
}


@pytest.mark.precision("64")
@pytest.mark.parametrize("params_name", ["coulomb", "viscous", "damping"])
@pytest.mark.parametrize("friction_model", ["c1", "cinf"])
def test_collision_response_is_gradient_of_potential(params_name, friction_model):
    from eden.engine.solvers.mochi.data import FRICTION_MODEL

    params = CONTACT_PARAMS[params_name]
    dt_stage = 0.01
    rng = np.random.default_rng(0)
    normal = np.array([0.3, -0.2, 0.93])
    normal /= np.linalg.norm(normal)
    colliding_normal = -normal
    # Samples over the whole penalty ramp, displaced from their stage-start position along and across the plane.
    distances = np.array([0.003, -0.001, 0.002, -0.005, -0.012, 0.0005])
    points = distances[:, None] * normal[None, :] + rng.normal(scale=0.01, size=(len(distances), 3)) * (1 - normal)
    points_stage_start = points - rng.normal(scale=0.5 * dt_stage, size=points.shape)
    for k, sliding in enumerate((1e-6, 1e-4, 3e-3)):
        points_stage_start[k] = points[k] - sliding * np.array([0.0, 1.0, 0.0])

    for use_fitted in (True, False):
        config = make_static_config(
            friction_model=FRICTION_MODEL.CINF if friction_model == "cinf" else FRICTION_MODEL.C1,
            use_fitted_friction_hessian=use_fitted,
        )
        energy, force, dforce = plane_collision_response(
            points, points_stage_start, normal, colliding_normal, params, dt_stage, config
        )
        assert np.all(energy >= 0.0)
        assert energy[distances > 0.001].max() == 0.0

        # Central finite differences of the energy and of the force with respect to the sample position. The stage
        # displacement moves with the sample, so the dissipation terms are differentiated too.
        eps = 1e-7
        force_fd = np.zeros_like(force)
        dforce_fd = np.zeros_like(dforce)
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = eps
            energy_p, force_p, _ = plane_collision_response(
                points + delta, points_stage_start, normal, colliding_normal, params, dt_stage, config
            )
            energy_m, force_m, _ = plane_collision_response(
                points - delta, points_stage_start, normal, colliding_normal, params, dt_stage, config
            )
            force_fd[:, axis] = -(energy_p - energy_m) / (2 * eps)
            dforce_fd[:, :, axis] = (force_p - force_m) / (2 * eps)
        scale = np.abs(force).max()
        assert_allclose(force, force_fd, atol=1e-5 * scale, rtol=1e-5)
        if not use_fitted:
            # The exact derivative is checked away from the stick-slip regularization width, where the C1 model has a
            # kink in the second derivative.
            hess_scale = np.abs(dforce).max()
            assert_allclose(dforce, dforce_fd, atol=1e-4 * hess_scale, rtol=1e-4)
        # Symmetric and negative semi-definite by construction.
        assert_allclose(dforce, np.swapaxes(dforce, 1, 2), tol=1e-9 * max(np.abs(dforce).max(), 1.0))
        assert np.all(np.linalg.eigvalsh(dforce) <= 1e-6 * max(np.abs(dforce).max(), 1.0))


@pytest.mark.precision("64")
def test_collider_distance_fields(show_viewer):
    scene_mismatched = create_mochi_scene()
    scene_mismatched.add_entity(gs.morphs.Sphere(radius=1.0), material=mochi_materials.Rigid(collider_type="box"))
    with pytest.raises(gs.GenesisException, match="requires a box primitive"):
        scene_mismatched.build()

    scene = create_mochi_scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        show_viewer=show_viewer,
    )
    center = (0.1, 0.2, 0.3)
    box = scene.add_entity(
        gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=center, fixed=True), material=mochi_materials.Rigid()
    )
    grid_box = scene.add_entity(
        gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=(center[0], center[1] + 5.0, center[2]), fixed=True),
        material=mochi_materials.Rigid(collider_type="sdf", sdf_cell_size=0.02),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=1.0, pos=(center[0], center[1] - 5.0, center[2]), fixed=True),
        material=mochi_materials.Rigid(),
    )
    plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, 1.0)), material=mochi_materials.Rigid())
    scene.build()
    solver = scene.mochi_solver

    # Probes along x through the cube center: outside, inside near the face, inside, outside, far outside.
    x_offsets = np.array([-0.6, 0.3, -0.4, 0.9, 1.4])
    expected = np.array([0.1, -0.2, -0.1, 0.4, 0.9])
    expected_grad = np.array([[-1, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    points = np.array(box.get_pos().cpu().numpy())[None, :] + x_offsets[:, None] * np.array([[1.0, 0.0, 0.0]])
    distances, gradients, is_valid = solver.get_collider_distances(box.geoms[0].idx, points)
    assert bool(is_valid.all())
    assert_allclose(distances, expected, tol=1e-9)
    assert_allclose(gradients, expected_grad, tol=1e-9)
    # The grid spans 10% of the size beyond the surface. Beyond it, the field adds the distance to the grid to the
    # value at the closest grid point, which is exact along the face normal.
    grid_offsets = np.array([0.3, -0.4, -0.55, 0.55, 1.4, -0.9])
    grid_expected = np.array([-0.2, -0.1, 0.05, 0.05, 0.9, 0.4])
    grid_expected_grad = np.array([[1, 0, 0], [-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=float)
    points = np.array(grid_box.get_pos().cpu().numpy())[None, :] + grid_offsets[:, None] * np.array([[1.0, 0.0, 0.0]])
    distances, gradients, is_valid = solver.get_collider_distances(grid_box.geoms[0].idx, points)
    assert bool(is_valid.all())
    assert_allclose(distances, grid_expected, atol=5e-4, rtol=0.0)
    assert_allclose(gradients, grid_expected_grad, atol=5e-2, rtol=0.0)

    points = np.array(sphere.get_pos().cpu().numpy())[None, :] + np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    distances, gradients, is_valid = solver.get_collider_distances(sphere.geoms[0].idx, points)
    assert_allclose(distances, [-1.0, 0.0, 1.0, np.sqrt(2.0) - 1.0], tol=1e-9)
    assert_allclose(np.linalg.norm(gradients[1:].cpu().numpy(), axis=-1), 1.0, tol=1e-9)

    points = np.array([[0.0, 0.0, -1.0], [3.0, -2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    distances, gradients, is_valid = solver.get_collider_distances(plane.geoms[0].idx, points)
    assert_allclose(distances, [-2.0, -1.0, 0.0, 1.0], tol=1e-9)
    assert_allclose(gradients, np.tile([[0.0, 0.0, 1.0]], (4, 1)), tol=1e-9)


@pytest.mark.precision("64")
@pytest.mark.parametrize("params_name", ["coulomb", "viscous", "damping"])
@pytest.mark.parametrize("friction_model", ["c1", "cinf"])
def test_collision_response_tangent_is_symmetric(params_name, friction_model):
    # The contact force derivative is built from grad grad^T, I - n n^T and n n^T terms only, so it is symmetric by
    # construction: the solver stores it as six floats (see func_mat3_to_sym6), which is exact only as long as every
    # response path keeps the symmetry.
    from eden.engine.solvers.mochi.data import FRICTION_MODEL

    params = CONTACT_PARAMS[params_name]
    dt_stage = 0.01
    rng = np.random.default_rng(1)
    normal = np.array([0.1, 0.4, 0.911])
    normal /= np.linalg.norm(normal)
    distances = np.array([0.003, -0.001, 0.002, -0.005, -0.012, 0.0005])
    points = distances[:, None] * normal[None, :] + rng.normal(scale=0.01, size=(len(distances), 3)) * (1 - normal)
    points_stage_start = points - rng.normal(scale=0.5 * dt_stage, size=points.shape)
    for use_fitted in (True, False):
        for implicit_normal in (True, False):
            config = make_static_config(
                friction_model=FRICTION_MODEL.CINF if friction_model == "cinf" else FRICTION_MODEL.C1,
                use_fitted_friction_hessian=use_fitted,
                implicit_normal_force_for_dissipation=implicit_normal,
            )
            _, _, dforce = plane_collision_response(
                points, points_stage_start, normal, -normal, params, dt_stage, config
            )
            assert_allclose(dforce, np.swapaxes(dforce, -1, -2), tol=1e-12)


def _mochi_scene(show_viewer, dt, **mochi_kwargs):
    return create_mochi_scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        mochi_options=MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _sheet_obj(path, n_cells, size):
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return str(path)


def _as_arrays(contacts):
    return {key: tensor_to_array(value) for key, value in contacts.items()}


def _scatter(contacts, side, n_vertices, entity_idx):
    """Force on the vertices of a deformable entity from the records where it is on the given side ('a' or 'b')."""
    is_side = contacts[f"entity_{side}"] == entity_idx
    verts = contacts[f"verts_{side}"][is_side]
    bary = contacts[f"bary_{side}"][is_side]
    force = contacts[f"force_{side}"][is_side]
    out = np.zeros((n_vertices, 3))
    for k in range(verts.shape[1]):
        valid = verts[:, k] >= 0
        np.add.at(out, verts[valid, k], bary[valid, k, None] * force[valid])
    return out


@pytest.mark.precision("64")
def test_soft_on_rigid_contacts(show_viewer):
    size, threshold = 0.2, 1e-3
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    plane = scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid(penalty_threshold=threshold))
    cube = scene.add_entity(
        gs.morphs.Box(size=(size, size, size), pos=(0.0, 0.0, 0.5 * size + 0.01), maxvolume=0.0005, nobisect=False),
        material=mochi_materials.Elastic(E=1e5, nu=0.45, rho=1000.0, penalty_threshold=threshold),
    )
    scene.build()
    for _ in range(90):
        scene.step()
    contacts = _as_arrays(cube.get_contacts())
    n_contacts = len(contacts["entity_a"])
    assert n_contacts > 0
    assert np.all(contacts["entity_a"] == cube.idx)
    assert np.all(contacts["entity_b"] == plane.idx)
    assert np.all(contacts["link_a"] == -1)
    assert np.all(contacts["geom_a"] == -1)
    assert np.all(contacts["link_b"] == plane.base_link.idx)
    assert np.all(contacts["geom_b"] == plane.geoms[0].idx)
    # Boundary samples spread over a triangle: the fourth slot is unused.
    assert np.all((contacts["verts_a"][:, :3] >= 0) & (contacts["verts_a"][:, :3] < cube.n_vertices))
    assert np.all(contacts["verts_a"][:, 3] == -1)
    assert_allclose(contacts["bary_a"].sum(axis=1), 1.0, tol=1e-12)
    assert np.all(contacts["verts_b"] == -1)
    assert_allclose(contacts["normal"], np.tile((0.0, 0.0, 1.0), (n_contacts, 1)), tol=1e-12)
    assert np.all(contacts["distance"] < threshold)
    assert np.all(contacts["distance"] > threshold - 5e-3)
    assert_allclose(contacts["position"][:, 2], contacts["distance"], tol=1e-12)
    assert np.all(contacts["weight"] > 0.0)
    # The records account for the whole contact force on the vertices of the body, which balances its weight.
    force_verts = tensor_to_array(cube.get_vertices_contact_force())
    assert_allclose(_scatter(contacts, "a", cube.n_vertices, cube.idx), force_verts, atol=1e-9)
    assert_allclose(contacts["force_a"].sum(axis=0), (0.0, 0.0, cube.mass * 9.8), atol=2e-3)
    # The same records are seen from the collider's side, and with the pair filter.
    for entity, other in ((plane, cube), (cube, plane)):
        view = _as_arrays(entity.get_contacts(with_entity=other))
        assert len(view["entity_a"]) == n_contacts
        assert_allclose(view["force_a"], contacts["force_a"], tol=0.0)
    assert len(plane.get_contacts(with_entity=plane)["entity_a"]) == 0


@pytest.mark.precision("64")
def test_soft_on_soft_contacts(show_viewer):
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    plane = scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    bottom = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05), maxvolume=0.0005, nobisect=False),
        material=mochi_materials.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    top = scene.add_entity(
        gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.2), maxvolume=0.0005, nobisect=False),
        material=mochi_materials.Elastic(E=1e5, nu=0.45, rho=1000.0),
    )
    scene.build()
    assert scene.mochi_solver._has_soft_colliders
    for _ in range(150):
        scene.step()
    contacts = _as_arrays(top.get_contacts(with_entity=bottom))
    n_contacts = len(contacts["entity_a"])
    assert n_contacts > 0
    is_top_a = contacts["entity_a"] == top.idx
    assert np.all(contacts["entity_b"][is_top_a] == bottom.idx)
    assert np.all(contacts["entity_a"][~is_top_a] == bottom.idx)
    assert np.all(contacts["entity_b"][~is_top_a] == top.idx)
    for key in ("link_a", "link_b", "geom_a", "geom_b"):
        assert np.all(contacts[key] == -1)
    # The sample side spreads over a boundary triangle (the fourth slot unused), the collider side over a tetrahedron.
    for side, n_slots in (("a", 3), ("b", 4)):
        n_vertices = np.where(contacts[f"entity_{side}"] == top.idx, top.n_vertices, bottom.n_vertices)
        verts = contacts[f"verts_{side}"][:, :n_slots]
        assert np.all((verts >= 0) & (verts < n_vertices[:, None]))
        assert_allclose(contacts[f"bary_{side}"].sum(axis=1), 1.0, tol=1e-12)
    assert np.all(contacts["verts_a"][:, 3] == -1)
    # A sample point is located inside the collider body (non-negative barycentric weights) and recorded within the
    # smoothing band of the penalty (the rest-shape distance field is slightly positive near the surface).
    assert np.all(contacts["bary_b"] > -1e-12)
    assert np.all(contacts["distance"] < 1e-2)
    assert_allclose(contacts["position"][:, 2], 0.1, atol=1e-2)
    # Records where the upper body is the sample side and where it is the collider side together account for its
    # vertex contact forces, which balance its weight.
    force_top = _scatter(contacts, "a", top.n_vertices, top.idx) + _scatter(contacts, "b", top.n_vertices, top.idx)
    assert_allclose(force_top, tensor_to_array(top.get_vertices_contact_force()), atol=1e-9)
    # Slow frictional creep of the upper body leaves a small lateral component.
    assert_allclose(force_top.sum(axis=0), (0.0, 0.0, top.mass * 9.8), atol=1e-2)
    assert len(bottom.get_contacts(with_entity=top)["entity_a"]) == n_contacts
    # The lower body also touches the ground.
    all_bottom = _as_arrays(bottom.get_contacts())
    assert set(np.unique(all_bottom["entity_b"])) == {plane.idx, bottom.idx} or set(
        np.unique(np.concatenate([all_bottom["entity_a"], all_bottom["entity_b"]]))
    ) == {plane.idx, bottom.idx, top.idx}


@pytest.mark.precision("64")
def test_point_cloud_contacts(tmp_path, show_viewer):
    # A rigid ball and a small cloth patch rest on a corner-held net: the ball is carried by the net's samples
    # colliding against its sphere collider (point-cloud colliders never collide with rigid bodies, as mochi), the
    # patch by the point-cloud contact between the two cloths.
    radius, collider_radius, n_cells, size = 0.05, 0.02, 10, 0.5
    obj_path = _sheet_obj(tmp_path / "sheet.obj", n_cells, size)
    patch_path = _sheet_obj(tmp_path / "patch.obj", 4, 0.15)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 0.5)),
        material=mochi_materials.Shell(
            E=1e6, nu=0.3, rho=1000.0, thickness=2e-3, collider_radius=collider_radius, mass_damping=2.0
        ),
    )
    patch = scene.add_entity(
        gs.morphs.Mesh(file=patch_path, pos=(0.15, 0.15, 0.55)),
        material=mochi_materials.Shell(
            E=1e6, nu=0.3, rho=1000.0, thickness=2e-3, collider_radius=collider_radius, mass_damping=2.0
        ),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=radius, pos=(0.0, 0.0, 0.5 + radius + 0.01)),
        material=mochi_materials.Rigid(rho=500.0),
    )
    scene.build()
    assert scene.mochi_solver._has_pc_colliders
    verts = tensor_to_array(cloth.get_vertices_position())
    corners = np.flatnonzero((np.abs(verts[:, 0]) > 0.5 * size - 1e-6) & (np.abs(verts[:, 1]) > 0.5 * size - 1e-6))
    assert len(corners) == 4
    cloth.set_vertices_fixed(corners)
    for _ in range(240):
        scene.step()
    # The ball's contacts are shell samples of the net against its sphere collider.
    contacts = _as_arrays(ball.get_contacts())
    n_contacts = len(contacts["entity_a"])
    assert n_contacts > 0
    assert np.all(contacts["entity_a"] == cloth.idx)
    assert np.all(contacts["entity_b"] == ball.idx)
    assert np.all(contacts["link_a"] == -1)
    assert np.all(contacts["geom_a"] == -1)
    assert np.all(contacts["link_b"] == ball.base_link.idx)
    assert np.all(contacts["geom_b"] == ball.geoms[0].idx)
    # Boundary samples spread over a triangle: the fourth slot is unused.
    assert np.all((contacts["verts_a"][:, :3] >= 0) & (contacts["verts_a"][:, :3] < cloth.n_vertices))
    assert np.all(contacts["verts_a"][:, 3] == -1)
    assert_allclose(contacts["bary_a"].sum(axis=1), 1.0, tol=1e-12)
    assert np.all(contacts["verts_b"] == -1)
    # Records cover the contact range of the penalty (the threshold).
    assert np.all(contacts["distance"] < 1.1e-3)
    # The collider gradient points away from the ball at every contact.
    ball_pos = tensor_to_array(ball.get_pos())
    assert np.all(np.einsum("ij,ij->i", contacts["normal"], contacts["position"] - ball_pos) > 0.0)
    # Records are consistent with the per-body readbacks: net force on the ball, force on the cloth vertices.
    ball_force = tensor_to_array(ball.get_links_net_contact_force())[0]
    assert_allclose(-contacts["force_a"].sum(axis=0), ball_force, atol=1e-9)
    # The ball still rolls slowly in the sag of the net: only the vertical balance is tight.
    assert_allclose(ball_force[2], float(tensor_to_array(ball.get_mass())) * 9.8, atol=1e-2)
    assert np.linalg.norm(ball_force[:2]) < 0.05
    assert len(cloth.get_contacts(with_entity=ball)["entity_a"]) == n_contacts
    # The patch rests on the net through point-cloud contacts: single-vertex collider records between the two cloths.
    pc_contacts = _as_arrays(patch.get_contacts(with_entity=cloth))
    n_pc = len(pc_contacts["entity_a"])
    assert n_pc > 0
    assert set(np.unique(pc_contacts["entity_a"])) | set(np.unique(pc_contacts["entity_b"])) == {cloth.idx, patch.idx}
    assert np.all(pc_contacts["link_a"] == -1)
    assert np.all(pc_contacts["link_b"] == -1)
    assert np.all(pc_contacts["verts_b"][:, 0] >= 0)
    assert np.all(pc_contacts["verts_b"][:, 1:] == -1)
    assert_allclose(pc_contacts["bary_b"], np.tile((1.0, 0.0, 0.0, 0.0), (n_pc, 1)), tol=0.0)
    # All records involving the cloth scattered onto its vertices match the vertex force readback.
    cloth_contacts = _as_arrays(cloth.get_contacts())
    assert_allclose(
        _scatter(cloth_contacts, "a", cloth.n_vertices, cloth.idx)
        + _scatter(cloth_contacts, "b", cloth.n_vertices, cloth.idx),
        tensor_to_array(cloth.get_vertices_contact_force()),
        atol=1e-9,
    )


def _mochi_scene(show_viewer, dt, gravity=(0.0, 0.0, -9.8), **mochi_kwargs):
    return create_mochi_scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=gravity),
        mochi_options=MochiOptions(**mochi_kwargs),
        show_viewer=show_viewer,
    )


def _strip_obj(path, n_x, n_y, length, width):
    xs = np.linspace(-0.5 * length, 0.5 * length, n_x + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, n_y + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_x):
        for j in range(n_y):
            a = i * (n_y + 1) + j
            faces.append([a, a + 1, a + n_y + 2])
            faces.append([a, a + n_y + 2, a + n_y + 1])
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return str(path)


def _hairpin_points(leg_length, gap, n_leg, n_arc, z_offset=0.0):
    """Two parallel legs joined by a half circle; the second leg is raised by z_offset."""
    x = np.linspace(0.0, leg_length, n_leg + 1)
    leg_a = np.stack([x, np.full_like(x, -0.5 * gap), np.zeros_like(x)], axis=-1)
    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_arc + 1)[1:-1]
    arc = np.stack(
        [
            leg_length + 0.5 * gap * np.cos(angles),
            0.5 * gap * np.sin(angles),
            z_offset * (angles + 0.5 * np.pi) / np.pi,
        ],
        axis=-1,
    )
    leg_b = np.stack([x[::-1], np.full_like(x, 0.5 * gap), np.full_like(x, z_offset)], axis=-1)
    return np.concatenate([leg_a, arc, leg_b])


@pytest.mark.precision("64")
@pytest.mark.parametrize("self_contact", [True, False])
def test_shell_self_contact(tmp_path, show_viewer, self_contact):
    length, width, n_x, n_y, radius = 0.6, 0.15, 12, 3, 0.02
    obj_path = _strip_obj(tmp_path / "strip.obj", n_x, n_y, length, width)
    scene = _mochi_scene(show_viewer, 1.0 / 60.0, n_newton_iterations=8)
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid(penalty_coefficient=1e7))
    cloth = scene.add_entity(
        gs.morphs.Mesh(file=obj_path, pos=(0.0, 0.0, 1e-3)),
        material=mochi_materials.Shell(
            E=1e5,
            nu=0.3,
            rho=200.0,
            thickness=1e-3,
            collider_radius=radius,
            penalty_coefficient=1e7,
            mass_damping=5.0,
            self_contact=self_contact,
        ),
    )
    scene.build()
    rest = tensor_to_array(cloth.get_vertices_position())
    pinned = np.flatnonzero(rest[:, 0] < -0.5 * length + 1e-6)
    driven = np.flatnonzero(rest[:, 0] > 0.5 * length - 1e-6)
    cloth.set_vertices_fixed(pinned)
    # Fold the strip over onto itself: the free edge travels along a half circle about the middle of the strip and
    # lands above the pinned edge.
    n_fold = 90
    for i_step in range(n_fold):
        angle = np.pi * (i_step + 1) / n_fold
        target = rest[driven].copy()
        target[:, 0] = 0.5 * length * np.cos(angle)
        target[:, 2] = 0.5 * length * np.sin(angle) + 1.5 * radius * (i_step + 1) / n_fold + 1e-3
        cloth.set_vertices_target(driven, target)
        scene.step()
    for _ in range(150):
        scene.step()
    pos = tensor_to_array(cloth.get_vertices_position())
    # Interior vertices of the folded half (excluding the driven edge and the fold).
    upper = np.flatnonzero((rest[:, 0] > 0.1 * length) & (rest[:, 0] < 0.5 * length - 1e-6))
    lower = np.flatnonzero(rest[:, 0] < -0.1 * length)
    height = np.median(pos[upper, 2]) - np.median(pos[lower, 2])
    if self_contact:
        # The upper layer rests on the collider spheres of the lower layer.
        assert 0.6 * radius < height < 1.3 * radius
    else:
        # Without self-contact the upper layer falls through onto the ground.
        assert height < 0.3 * radius


def _segments_min_distance(pa, pb):
    """Smallest distance between the segments of two polylines (pa, pb: (n, 3) node arrays)."""
    best = np.inf
    for i in range(len(pa) - 1):
        for j in range(len(pb) - 1):
            p0, u = pa[i], pa[i + 1] - pa[i]
            q0, v = pb[j], pb[j + 1] - pb[j]
            w = p0 - q0
            a, b, c, d, e = u @ u, u @ v, v @ v, u @ w, v @ w
            den = a * c - b * b
            s_ = np.clip((b * e - c * d) / den, 0.0, 1.0) if den > 1e-14 else 0.0
            t_ = (b * s_ + e) / c
            if t_ < 0.0:
                t_, s_ = 0.0, np.clip(-d / a, 0.0, 1.0)
            elif t_ > 1.0:
                t_, s_ = 1.0, np.clip((b - d) / a, 0.0, 1.0)
            best = min(best, np.linalg.norm(w + s_ * u - t_ * v))
    return best


@pytest.mark.precision("64")
@pytest.mark.parametrize("self_contact", [True, False])
def test_rod_self_contact(show_viewer, self_contact):
    # Segments as long as the radius so that the chain of collider spheres has no gaps a rod could slip through; the
    # second leg is slightly raised so that the legs can pass over each other instead of meeting head-on in a plane.
    leg_length, gap, radius, n_leg, n_arc = 0.4, 0.1, 0.01, 40, 16
    points = _hairpin_points(leg_length, gap, n_leg, n_arc, z_offset=0.3 * radius)
    scene = _mochi_scene(show_viewer, 0.01, gravity=(0.0, 0.0, 0.0), n_newton_iterations=8)
    rod = scene.add_entity(
        Rod(points=points, radius=radius),
        material=mochi_materials.Rod(
            E=1e7,
            nu=0.3,
            rho=1000.0,
            mass_damping=1.0,
            self_contact=self_contact,
            self_contact_exclusion_ratio=3.0,
        ),
    )
    scene.build()
    rest = tensor_to_array(rod.get_vertices_position())
    n_nodes = len(points)
    # Swap the sides of the two clamped ends (two nodes each) so that the legs have to pass each other.
    ends = np.array([0, 1, n_nodes - 2, n_nodes - 1])
    # The prescribed end segments cross by construction: measure the free parts of the legs.
    leg_a = np.arange(3, n_leg + 1)
    leg_b = n_nodes - 1 - leg_a
    n_press = 100
    distance = np.inf
    for i_step in range(n_press + 150):
        if i_step < n_press:
            s = (i_step + 1) / n_press
            target = rest[ends].copy()
            target[:, 1] = rest[ends, 1] * (1.0 - 2.0 * s)
            rod.set_vertices_target(ends, target)
        scene.step()
        pos = tensor_to_array(rod.get_vertices_position())
        assert np.all(np.isfinite(pos))
        # Closest approach of the two legs over the whole motion (they end up on swapped sides either way).
        distance = min(distance, _segments_min_distance(pos[leg_a], pos[leg_b]))
    if self_contact:
        # The legs pass over each other, kept apart by the collider spheres of their nodes.
        assert distance > 0.6 * radius
    else:
        # Without self-contact the legs pass through each other's spheres at their initial offset.
        assert distance < 0.45 * radius
