"""The hosting of the MochiSolver in a scene of stock Genesis: what an installed scene does that the fork's did.

Each test builds a real scene. The private Genesis attributes ``hosting`` writes (``Simulator._solvers``,
``Simulator._entities``, ``Scene._desc``) are not read here: what they make the scene do is.
"""

import genesis as gs
import numpy as np
import pytest
import torch
from genesis.engine.scene import SCENE_FORMAT
from genesis.utils.misc import qd_to_torch

from eden.engine.solvers.mochi import (
    MochiOptions,
    MochiRigidSpec,
    MochiSoftSpec,
    MochiSolver,
    add_mochi_entity,
    create_mochi_scene,
    find_mochi_solver,
    install_mochi_solver,
)
from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi.error_codes import MochiErrorCode

SIM_OPTIONS = {"dt": 1.0 / 60.0, "gravity": (0.0, 0.0, -9.8)}


def _scene(**scene_kwargs):
    return create_mochi_scene(
        sim_options=gs.options.SimOptions(**SIM_OPTIONS),
        mochi_options=MochiOptions(n_newton_iterations=8),
        show_viewer=False,
        **scene_kwargs,
    )


def test_an_installed_scene_hands_every_mochi_material_to_the_mochi_solver():
    scene = _scene()
    plane = scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    shell = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Elastic(E=1e5)
    )
    kinematic = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.5)), material=gs.materials.Kinematic()
    )

    solver = scene.mochi_solver
    assert isinstance(solver, MochiSolver) and find_mochi_solver(scene) is solver
    assert plane in solver.entities and shell in solver.soft_entities
    # A material of another solver still goes where upstream sends it.
    assert kinematic in scene.sim.kinematic_solver.entities
    assert not scene.sim.rigid_solver.entities
    # The solver took the scene's gravity, which SceneOptions hands only to the solvers it declares.
    np.testing.assert_allclose(solver._options.gravity, SIM_OPTIONS["gravity"])


def test_a_mochi_spec_reaches_the_mochi_solver_and_a_genesis_spec_of_a_mochi_material_is_refused():
    """The simulator dispatches a spec along the MRO of its class: a `MochiRigidSpec` is a `gs.entities.Rigid`, which
    would otherwise reach the RigidSolver.
    """
    scene = _scene()
    plane = scene.add_entity(spec=MochiRigidSpec(morph=gs.morphs.Plane()))
    cube = scene.add_entity(
        spec=MochiSoftSpec(
            morph=gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Elastic()
        )
    )
    assert plane in scene.mochi_solver.entities and cube in scene.mochi_solver.soft_entities
    assert cube.surface.vis_mode == "visual" and not scene.sim.rigid_solver.entities

    with pytest.raises(gs.GenesisException, match="Pass a spec of `eden.engine.solvers.mochi.specs` instead"):
        scene.add_entity(spec=gs.entities.Rigid(morph=gs.morphs.Plane(), material=mochi_materials.Rigid()))
    with pytest.raises(gs.GenesisException, match="contact parameters of its bodies from their material"):
        MochiRigidSpec(morph=gs.morphs.Plane()).select(gs.indices.Link.named("*"))
    with pytest.raises(gs.GenesisException, match="never both"):
        scene.add_entity(gs.morphs.Plane(), spec=MochiRigidSpec(morph=gs.morphs.Plane()))
    # Heterogeneous entities are refused with the name of the Mochi spec, a sequence of one morph included
    box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
    for morphs, material in (([box, box], mochi_materials.Rigid()), ([box], mochi_materials.Elastic())):
        with pytest.raises(gs.GenesisException, match=r"pass Mochi(Rigid|Soft)Spec one morph, not a sequence of"):
            scene.add_entity(morphs, material=material)


def test_a_mochi_entity_gets_the_surface_defaults_of_its_kind():
    scene = _scene()
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Rigid())
    sheet = scene.add_entity(
        gs.morphs.Box(size=(0.4, 0.4, 0.01), pos=(0.0, 0.0, 1.0)), material=mochi_materials.Shell()
    )

    assert box.surface.vis_mode == "visual" and box.surface.smooth is False and box.surface.double_sided is False
    assert sheet.surface.double_sided is True
    with pytest.raises(gs.GenesisException, match="Unsupported `surface.vis_mode`.*Expected one of: \\['visual'\\]"):
        scene.add_entity(
            gs.morphs.Sphere(radius=0.1, pos=(1.0, 0.0, 0.5)), material=mochi_materials.Elastic(), vis_mode="collision"
        )


def test_a_file_mesh_keeps_its_authored_collision_surface():
    scene = _scene()
    morph = gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.1, pos=(0.0, 0.0, 0.5))
    scene.add_entity(morph, material=mochi_materials.Rigid())
    assert morph.convexify is False


def test_the_hosting_refuses_what_it_cannot_honour(tmp_path):
    plain = gs.Scene(show_viewer=False)
    with pytest.raises(gs.GenesisException, match="No MochiSolver in this scene"):
        add_mochi_entity(plain, gs.morphs.Plane(), mochi_materials.Rigid())

    scene = _scene()
    with pytest.raises(gs.GenesisException, match="already has a MochiSolver"):
        install_mochi_solver(scene)
    with pytest.raises(gs.GenesisException, match="draws the RigidSolver's contacts only"):
        scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid(), visualize_contact=True)
    with pytest.raises(gs.GenesisException, match="cannot be drawn by BatchRenderer"):
        _scene(renderer=gs.renderers.BatchRenderer())

    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    # `gs.Scene.load` would rebuild it without the solver.
    with pytest.raises(gs.GenesisException, match="hosts Eden's MochiSolver"):
        scene.export(tmp_path / f"mochi{SCENE_FORMAT}")
    scene.build()
    with pytest.raises(gs.GenesisException, match="before the scene is built"):
        install_mochi_solver(scene)


def test_a_scene_without_the_solver_runs_the_upstream_code():
    scene = gs.Scene(show_viewer=False)
    assert find_mochi_solver(scene) is None
    assert not hasattr(scene, "mochi_solver")
    # Nothing is bound on the instance: these resolve to the classes' own methods.
    assert "add_entity" not in vars(scene) and "reset" not in vars(scene.sim)
    assert "on_fem" not in vars(scene.visualizer.context)


def test_a_rigid_scene_loads_back_with_the_upstream_rigid_material(tmp_path):
    # Genesis' serializer resolves a material by its class name across every subclass, and the Mochi rigid material
    # (imported by this module) extends `gs.materials.Rigid`: a namesake would load back in its place.
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), material=gs.materials.Rigid())
    exported = tmp_path / f"rigid{SCENE_FORMAT}"
    scene.export(exported)
    assert [type(entity.material) for entity in gs.Scene.load(exported).entities] == [gs.materials.Rigid]


def test_the_rasterizer_draws_the_rigid_and_the_deformable_bodies():
    scene = _scene()
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    scene.add_entity(
        gs.morphs.Sphere(radius=0.15, pos=(0.0, 0.0, 0.5)),
        material=mochi_materials.Rigid(),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
    )
    scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.4, 0.0, 0.3)),
        material=mochi_materials.Elastic(E=1e5),
        surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0)),
    )
    camera = scene.add_camera(res=(160, 120), pos=(1.5, -1.5, 1.0), lookat=(0.0, 0.0, 0.2), GUI=False)
    scene.build()
    rgb_start = np.asarray(camera.render()[0])
    for _ in range(10):
        scene.step()
    rgb = np.asarray(camera.render()[0])

    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)
    blue = (rgb[..., 2] > 150) & (rgb[..., 0] < 80) & (rgb[..., 1] < 80)
    assert red.sum() > 50 and blue.sum() > 50
    # The frame follows the bodies as they fall: the visual geoms are refreshed each step.
    red_start = (rgb_start[..., 0] > 150) & (rgb_start[..., 1] < 80) & (rgb_start[..., 2] < 80)
    assert np.argwhere(red).mean(axis=0)[0] > np.argwhere(red_start).mean(axis=0)[0]


def test_a_frame_drawn_after_a_setter_shows_the_body_where_the_setter_put_it():
    """A setter between two steps syncs the rasterizer with the nodes of the solver it changed alone, as the
    interactive viewer does on every change of a body's geometry; the camera's next frame draws from those nodes.
    """
    scene = _scene()
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    ball = scene.add_entity(
        gs.morphs.Sphere(radius=0.15, pos=(0.0, 0.0, 0.3)),
        material=mochi_materials.Rigid(),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
    )
    camera = scene.add_camera(res=(160, 120), pos=(1.5, -1.5, 1.0), lookat=(0.0, 0.0, 0.2), GUI=False)
    scene.build()
    scene.step()

    def n_red() -> int:
        rgb = np.asarray(camera.render()[0])
        return int(((rgb[..., 0] > 150) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)).sum())

    assert n_red() > 50
    ball.set_pos(torch.tensor([50.0, 0.0, 0.3], dtype=gs.tc_float, device=gs.device))
    scene.visualizer.sync(frozenset((scene.mochi_solver,)))
    assert n_red() == 0


def test_a_reset_restores_a_scene_of_deformable_bodies_alone():
    """``Simulator.reset`` skips a solver whose ``data`` leaves state out (``is_checkpointable``), as the MochiSolver's
    does; the installed reset restores it, deformable vertices included.
    """
    scene = _scene()
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Elastic(E=1e5)
    )
    scene.build(n_envs=2)
    start = cube.get_state().pos.clone()
    for _ in range(5):
        scene.step()
    assert float((cube.get_state().pos - start).abs().max()) > 1e-3

    scene.reset()
    torch.testing.assert_close(cube.get_state().pos, start, rtol=0.0, atol=0.0)


def test_a_saved_state_restores_the_mochi_bodies_of_the_environments_it_is_given():
    """The scene's state holds the MochiSolver's, which `Simulator.get_state` leaves out (it is not checkpointable)."""
    scene = _scene()
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Rigid())
    scene.build(n_envs=2)
    for _ in range(3):
        scene.step()
    state = scene.get_state()
    assert type(scene.mochi_solver).__name__ in state.solvers
    saved = box.get_pos().clone()
    for _ in range(5):
        scene.step()
    moved = box.get_pos().clone()
    assert float((moved - saved).abs().max()) > 1e-4

    scene.reset(state, envs_idx=[1])
    torch.testing.assert_close(box.get_pos(), torch.stack((moved[0], saved[1])), rtol=0.0, atol=0.0)

    state.solvers.pop(type(scene.mochi_solver).__name__)
    with pytest.raises(gs.GenesisException, match="holds no state of the MochiSolver"):
        scene.reset(state)

    other = _scene()
    other.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Rigid())
    other.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.5, 0.0, 0.5)), material=mochi_materials.Rigid())
    other.build(n_envs=2)
    with pytest.raises(gs.GenesisException, match="taken from another scene"):
        scene.reset(other.get_state())


@pytest.mark.parametrize("step_kernel", ["monolith", "pipeline", "graph"])
def test_the_visual_geoms_follow_the_links_after_a_step_of_every_kernel(step_kernel):
    """Genesis places the visual geoms in its own forward kinematics, which the pipelined step ends with; the
    single-kernel steps run Mochi's, and the solver places them itself after the last substep. A geom left behind
    would draw the body frozen where it was built.
    """
    scene = create_mochi_scene(
        sim_options=gs.options.SimOptions(**SIM_OPTIONS, substeps=2),
        mochi_options=MochiOptions(n_newton_iterations=8, step_kernel=step_kernel),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Rigid())
    scene.build(n_envs=2)
    for _ in range(3):
        scene.step()
    solver = scene.mochi_solver
    vgeoms_pos = qd_to_torch(solver.dyn_state.vgeoms.pos)  # (n_vgeoms, n_envs, 3)
    for vgeom in box.vgeoms:
        torch.testing.assert_close(vgeoms_pos[vgeom.idx], box.get_pos(), rtol=0.0, atol=1e-5)
    assert float(box.get_pos()[:, 2].max()) < 0.5 - 1e-3  # it fell: the check is not of the build pose


def test_a_partial_reset_resumes_the_environment_as_an_uninterrupted_run_would():
    """BDF2 extrapolates from the history of the step, which a reset of an env restores with its state. The reset's own
    notification marks the env as changed from outside, which `set_state` must clear once the history is back, or the
    next step would rebuild it and start a backward-Euler step instead.
    """
    scene = create_mochi_scene(
        sim_options=gs.options.SimOptions(**SIM_OPTIONS),
        mochi_options=MochiOptions(n_newton_iterations=8, integrator="bdf2"),
        show_viewer=False,
    )
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 2.0)), material=mochi_materials.Rigid())
    scene.build(n_envs=2)
    box.set_dofs_velocity(torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0, 0.0]] * 2, device=gs.device))
    for _ in range(3):
        scene.step()
    state = scene.get_state()
    for _ in range(4):
        scene.step()
    reference = (box.get_pos().clone(), box.get_quat().clone())
    for _ in range(2):
        scene.step()

    scene.reset(state, envs_idx=[1])
    for _ in range(4):
        scene.step()
    torch.testing.assert_close(box.get_pos()[1], reference[0][1], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(box.get_quat()[1], reference[1][1], rtol=0.0, atol=1e-6)


def test_an_error_raises_at_the_next_check_unless_its_environment_is_reset():
    """The simulator raises the MochiSolver's errors as it raises the rigid solver's, at the start of the step after the
    one that flagged them, and a reset or a setter of an environment clears that environment's.
    """
    scene = _scene()
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Rigid())
    scene.build(n_envs=2)
    solver = scene.mochi_solver
    diverged = np.array([0, int(MochiErrorCode.MOCHI_DIVERGED)], dtype=np.int32)

    # The error field is the only way to make the solver report an error on demand. The step latches what was flagged
    # once it is over, which the mask reads, and a later step raises it: the next one, or the one after on a device
    # whose check does not wait for it (see `Solver.check_errno`).
    solver._errno.from_numpy(diverged)
    assert solver.get_error_envs_mask().tolist() == [False, False]
    scene.step()
    assert solver.get_error_envs_mask().tolist() == [False, True]
    with pytest.raises(gs.GenesisException, match="MochiSolver diverged"):
        for _ in range(2):
            scene.step()

    for clear in (
        lambda: scene.reset(envs_idx=[1]),
        lambda: box.set_qpos(box.get_qpos()[1:], envs_idx=[1]),
        lambda: box.set_dofs_position(box.get_dofs_position()[1:], envs_idx=[1]),
        lambda: box.set_qpos(box.get_qpos(), envs_idx=torch.tensor([False, True], device=gs.device)),
    ):
        solver._errno.from_numpy(diverged)
        solver.update_errno()  # what the end of a step does
        clear()
        assert solver.get_error_envs_mask().tolist() == [False, False]
    for _ in range(10):
        scene.step()


def test_the_arrays_of_the_solver_are_freed_with_its_scene():
    """``Simulator.destroy`` frees the arrays of the solvers whose ``data`` it can list, which this solver refuses: its
    own ``destroy`` frees them.
    """
    if not gs.use_ndarray:
        pytest.skip("only ndarrays are freed with their scene; fields go with `qd.reset`")
    scene = _scene()
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)), material=mochi_materials.Rigid())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.5)), material=mochi_materials.Elastic(E=1e5))
    scene.build(n_envs=2)
    scene.step()
    solver = scene.mochi_solver
    # One array of each kind of struct: the kinematic tree, the collision arrays put in it, the solver's and the soft
    # bodies' own
    arrays = [
        solver.dyn_state.dofs.qf_applied,
        solver.dyn_state.geoms.pos,
        solver.mochi_state.res,
        solver.soft_state.verts_pos,
        solver.geoms_init_AABB,
    ]
    assert all(array._unwrap().arr is not None for array in arrays)

    scene.destroy()
    assert all(array._unwrap().arr is None for array in arrays)
    solver.destroy()  # an array already freed is left alone
