"""The MochiSolver's state goes through Genesis' checkpoint protocol: `MochiSolver.data` declares every array it holds
with its kind (`eden.engine.solvers.mochi.data`), and the simulator saves, restores and frees them as a stock solver's.

A kind the protocol reads wrongly fails silently: an array of the state labelled SCRATCH is left out of a saved state,
and the restored run resumes from another one. The tests here make that loud. They overwrite every scratch and derived
array with garbage after a restore and require the run to match the uninterrupted one bit for bit, and they fail on
an array the solver holds that `data` leaves out.
"""

import dataclasses

import genesis as gs
import numpy as np
import pytest
import quadrants as qd
import torch
import trimesh
from genesis.utils.data import ArrayItem, DataKind

from eden.engine.solvers.mochi import MochiOptions, Rod, create_mochi_scene
from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi.mochi_solver import _CONSTANT_ARRAYS, _SCRATCH_ARRAYS, _STRUCTS

N_STEPS = 6
# A step at which both boxes of the stack in `_scene` have started counting the 4 steps at rest they sleep after, at
# step 23
SLEEP_SAVE_STEP = 20


def _sheet_obj(path, n_cells=2, size=0.3):
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces += [[a, a + 1, a + n_cells + 2], [a, a + n_cells + 2, a + n_cells + 1]]
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return str(path)


def _scene(tmp_path, step_kernel="auto", use_sleeping=True):
    """Every kind of Mochi body, in contact, with the BDF2 history and sleeping on: a sliding and a resting rigid box
    on the ground, an elastic cube, a rod and a shell.
    """
    scene = create_mochi_scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0),
        mochi_options=MochiOptions(
            n_newton_iterations=8,
            integrator="bdf2",
            use_sleeping=use_sleeping,
            sleep_min_steps=4,
            step_kernel=step_kernel,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.15)), material=mochi_materials.Rigid())
    # A stack of two boxes, which settles on the ground and falls asleep (see `SLEEP_SAVE_STEP`): contact between two
    # dynamic bodies is gated on their sleep
    resting = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(-1.0, 0.0, 0.1)), material=mochi_materials.Rigid()
    )
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(-1.0, 0.0, 0.25)), material=mochi_materials.Rigid())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(1.0, 0.0, 0.5)), material=mochi_materials.Elastic(E=1e5)
    )
    points = np.stack([np.linspace(0.0, 0.5, 7), np.zeros(7), np.zeros(7)], axis=-1)
    scene.add_entity(Rod(points=points, radius=0.01, pos=(0.0, 1.0, 0.8)), material=mochi_materials.Rod(E=1e7))
    scene.add_entity(
        gs.morphs.Mesh(file=_sheet_obj(tmp_path / "sheet.obj"), pos=(1.0, 1.0, 0.6)),
        material=mochi_materials.Shell(E=1e5, thickness=2e-3),
    )
    scene.build(n_envs=2)
    box.set_dofs_velocity(torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2, device=gs.device))
    return scene, box, cube, resting


def _own_items(solver, kinds):
    """The arrays of the solver's own structs (not the kinematic tree Genesis declares) of the given kinds."""
    prefixes = tuple(f"{name}" for name in (*_STRUCTS, *_CONSTANT_ARRAYS, *_SCRATCH_ARRAYS, "mochi_config"))
    return [
        item
        for item in solver.data
        if isinstance(item, ArrayItem) and item.kind in kinds and item.name.split(".")[0] in prefixes
    ]


def _poison(solver):
    """Overwrite every scratch and derived array of the solver's own structs, which a step writes before reading."""
    for item in _own_items(solver, (DataKind.SCRATCH, DataKind.DERIVED)):
        dtype = item.array.dtype
        item.array.fill(float("nan") if dtype in (qd.f32, qd.f64) else -1)


def _state_arrays(scene):
    return {name: array.clone() for name, array in scene.mochi_solver.get_state().arrays.items()}


def _assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=0.0, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize("use_sleeping", [True, False])
def test_data_declares_every_array_the_solver_holds(tmp_path, use_sleeping):
    """`Simulator.destroy` frees and a checkpoint saves only what `data` yields. Genesis leaves the kinematic scratch
    of its own tree out, and so does this. A reset of some environments writes the state back along its last axis, so
    every array of the state ends with every environment, or is the 0-d placeholder of a feature switched off.
    """
    scene, _, _, _ = _scene(tmp_path, use_sleeping=use_sleeping)
    solver = scene.mochi_solver
    declared = {
        id(item.array._unwrap() if isinstance(item.array, qd.Tensor) else item.array)
        for item in solver.data
        if isinstance(item, ArrayItem)
    }

    def arrays(value, path):
        if isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
            yield path, value._unwrap() if isinstance(value, qd.Tensor) else value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                yield from arrays(getattr(value, field.name), f"{path}.{field.name}")
        elif isinstance(value, (list, tuple)):
            for i, member in enumerate(value):
                yield from arrays(member, f"{path}[{i}]")
        elif isinstance(value, dict):
            for key, member in value.items():
                yield from arrays(member, f"{path}[{key!r}]")

    held = [
        (path, array)
        for name, value in vars(solver).items()
        if name not in ("data_manager", "kinematics_scratch")
        for path, array in arrays(value, name)
    ]
    assert len(held) > 300
    missing = sorted(path for path, array in held if id(array) not in declared)
    assert not missing, f"MochiSolver.data leaves out {missing}: declare them in their struct or in `_STRUCTS`."
    # A 0-d placeholder of a feature switched off is skipped by the reset
    short = [
        item.name
        for item in _own_items(solver, (DataKind.STATE,))
        if item.array.shape and item.array.shape[-1] != solver.n_envs_
    ]
    assert not short, f"These arrays of the state lack an axis of {solver.n_envs_} environments: {short}"


@pytest.mark.parametrize("step_kernel", ["monolith", "pipeline", "graph"])
def test_a_restored_state_resumes_as_an_uninterrupted_run(tmp_path, step_kernel):
    """The state holds what the next step reads: the configuration, the BDF2 history, the deformable vertices, the
    rod frames and twists, sleeping, and the pinned vertices with their targets (a control input). Everything else is
    rewritten by the step, so garbage there after the restore changes nothing.
    """
    scene, box, cube, resting = _scene(tmp_path, step_kernel)
    cube.set_vertices_fixed(np.array([0, 1]))
    cube.set_vertices_target(np.array([0, 1]), cube.get_state().pos[:, :2] + 0.05)
    island_state = scene.mochi_solver.island_state
    # The rigid entities are the first island nodes, the top box of the stack following the bottom one
    node = scene.mochi_solver.entities.index(resting)
    for _ in range(SLEEP_SAVE_STEP):
        scene.step()
    state = scene.get_state()
    rest_steps_at_save = island_state.nodes_rest_steps.to_numpy()[node : node + 2].copy()
    for _ in range(N_STEPS):
        scene.step()
    expected = _state_arrays(scene)
    # The stack falls asleep during the replay, when the counts of its steps at rest reach 4
    assert (rest_steps_at_save > 0).all() and island_state.nodes_is_asleep.to_numpy()[node : node + 2].all()

    # Change the control inputs and the model the state does not hold, then restore
    cube.set_vertices_fixed(np.array([0, 1]), is_fixed=False)
    scene.reset(state)
    _poison(scene.mochi_solver)
    for _ in range(N_STEPS):
        scene.step()
    _assert_same_state(_state_arrays(scene), expected)


def _per_env(scene):
    """What each environment stands in, the environment first: the rigid configuration and velocity, and the vertices
    of every deformable body.
    """
    solver = scene.mochi_solver
    values = {"qpos": solver.get_qpos(), "dofs_vel": solver.get_dofs_velocity()}
    for entity in solver.soft_entities:
        entity_state = entity.get_state()
        values[f"{entity.name}.pos"], values[f"{entity.name}.vel"] = entity_state.pos, entity_state.vel
    return {name: value.clone() for name, value in values.items()}


def test_a_restore_of_some_environments_leaves_the_others_running(tmp_path):
    """Environment 1 replays from the saved step while environment 0 runs on, each as an uninterrupted run would."""
    scene, _, _, _ = _scene(tmp_path)
    for _ in range(3):
        scene.step()
    state = scene.get_state()
    for _ in range(N_STEPS):
        scene.step()
    at_replay_end = _per_env(scene)
    for _ in range(2 + N_STEPS):
        scene.step()
    at_run_end = _per_env(scene)

    scene.reset(state)
    for _ in range(N_STEPS + 2):
        scene.step()
    scene.reset(state, envs_idx=[1])
    for _ in range(N_STEPS):
        scene.step()
    actual = _per_env(scene)
    for name, value in actual.items():
        torch.testing.assert_close(value[0], at_run_end[name][0], rtol=0.0, atol=0.0, msg=lambda m, n=name: f"{n}: {m}")
        torch.testing.assert_close(
            value[1], at_replay_end[name][1], rtol=0.0, atol=0.0, msg=lambda m, n=name: f"{n}: {m}"
        )


def test_a_simulator_checkpoint_restores_the_model_its_setters_changed(tmp_path):
    """A full checkpoint holds the model parameters the setters write (INFO), which a reset's state does not."""
    scene, box, _, _ = _scene(tmp_path)
    solver = scene.mochi_solver
    for _ in range(3):
        scene.step()
    checkpoint = scene.sim.__getstate__()
    friction = solver.mochi_info.geoms.friction.to_numpy().copy()
    for _ in range(N_STEPS):
        scene.step()
    expected = _state_arrays(scene)

    solver.set_links_contact_params([box.base_link_idx], friction=0.01)
    assert not np.array_equal(solver.mochi_info.geoms.friction.to_numpy(), friction)
    scene.sim.__setstate__(checkpoint)
    np.testing.assert_array_equal(solver.mochi_info.geoms.friction.to_numpy(), friction)
    _poison(solver)
    for _ in range(N_STEPS):
        scene.step()
    _assert_same_state(_state_arrays(scene), expected)
