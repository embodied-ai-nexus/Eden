"""Tests for ``eden.managers.terms.events.motion``: seeding entities from the motion reference at reset.

One mimic env holds a floating-base G1, a fixed-base Panda (the immobile-base branch) and a free box,
each with its reference-state-initialization event; a fixed box and a free one are left untracked. The G1's and the
box's references are tilted and spinning, so a world-frame angular velocity written to the free joint's
body-frame DOFs shows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu
from genesis.utils.geom import inv_transform_by_quat

import eden as en
from eden.managers.terms.events.motion import (
    ObjectReferenceStateInitialization,
    ReferenceStateInitialization,
    ReferenceStateInitializationBase,
)
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from tests.support.motion import (
    OBJECT_LINKS,
    ROBOT_DOFS,
    ROBOT_LINKS,
    build_mimic_env,
    manager_options,
    term_options,
    write_clip,
)

ARM_DOFS = list(FrankaEmikaPanda.model_fields["dofs_name"].default)
SEQUENCES = {"robot": (ROBOT_DOFS, ROBOT_LINKS), "arm": (ARM_DOFS, ["link7"]), "object": ([], OBJECT_LINKS)}
# The reference lists the arm's DOFs in reverse, so writing them by position would cross joints.
ARM_REFERENCE_ORDER = ARM_DOFS[::-1]
RESET = en.EventMode.RESET
TILT = gu.xyz_to_quat(np.array([25.0, -15.0, 40.0]), rpy=True, degrees=True).astype(np.float32)
SPIN_W = np.array([0.5, -0.3, 0.8], dtype=np.float32)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsi")
    for clip, frames in enumerate([20, 30]):
        spinning = {
            f"{key}.{field}": np.tile(np.float32(value), (frames, 1))
            for key in ("robot", "object")
            for field, value in (
                ("base_quat_w", TILT),
                ("base_ang_vel_w", SPIN_W),
                ("base_lin_vel_w", [0.1, 0.2, -0.1]),
            )
        }
        spinning["robot.dofs_vel"] = np.full((frames, len(ROBOT_DOFS)), 0.05, dtype=np.float32)
        spinning["object.dofs_vel"] = np.zeros((frames, 0), dtype=np.float32)
        write_clip(root / f"clip_{clip:02d}.npz", frames, clip=clip, sequences=SEQUENCES, overrides=spinning)
    return root


def _motion_options(library):
    term = term_options(
        library, sequences=SEQUENCES, dofs={"arm": ARM_REFERENCE_ORDER}, dataset={"recalculate_velocities": False}
    )
    return manager_options(terms={"default": term})


def _scene():
    from eden.options.entities import BoxOptions
    from eden.options.robots.unitree_g1 import G1Dofs29

    return {
        "robot": G1Dofs29(),
        "arm": FrankaEmikaPanda(default_root_pos=(0.0, 2.0, 0.0)),
        "object": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 0.0, 0.05)),
        "shelf": BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(3.0, 0.0, 0.5), is_fixed_base=True),
        "crate": BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(3.0, 2.0, 0.1)),
    }


def _rsi_env(library, *, object_rsi: dict | None = None, **config_kwargs):
    """The scene with an RSI reset term on the robot, the arm and the object (``object_rsi``: its extra options)."""
    from eden.options.managers.events import EventManagerOptions

    events = EventManagerOptions(
        reset_robot=ReferenceStateInitialization.configure(mode=RESET, entity_name="robot", dofs_position_range=(0, 0)),
        reset_arm=ReferenceStateInitialization.configure(mode=RESET, entity_name="arm", dofs_position_range=(0, 0)),
        reset_object=ObjectReferenceStateInitialization.configure(
            mode=RESET, entity_name="object", **(object_rsi or {})
        ),
    )
    return build_mimic_env(
        library, scene=_scene(), motion_options=_motion_options(library), event_options=events, **config_kwargs
    )


@pytest.fixture(scope="module")
def env(library):
    return _rsi_env(library)


def _standalone(env, cls, **params):
    term = cls(env, cls.configure(mode=RESET, **params))
    term.build()
    return term


def _init_state(env, name):
    return env.motion_manager.get_init_reference_state()[name]


def _clipped(env, name, dofs_pos: torch.Tensor) -> torch.Tensor:
    limits = env.entities[name].soft_dofs_pos_limits
    return torch.clip(dofs_pos, limits[..., 0], limits[..., 1])


def _arm_expected(env, arm_state_dofs: torch.Tensor) -> torch.Tensor:
    """The arm reference, back in the arm's own DOF order and clipped to its soft limits."""
    return _clipped(env, "arm", arm_state_dofs[:, [ARM_REFERENCE_ORDER.index(n) for n in ARM_DOFS]])


def test_reset_puts_the_floating_base_robot_on_its_reference_with_velocity(env):
    env.reset()

    robot, ref = env.entities["robot"], _init_state(env, "robot")
    qpos = robot.get_qpos()
    torch.testing.assert_close(qpos[:, :3], ref.base_pos_w)
    torch.testing.assert_close(qpos[:, 3:7], ref.base_quat_w)
    # Clipped to the soft limits: the reference puts the knees just below theirs.
    torch.testing.assert_close(qpos[:, 7:], _clipped(env, "robot", ref.dofs_pos))
    assert not torch.equal(qpos[:, 7:], ref.dofs_pos)
    base_vel = robot.get_dofs_vel(dofs_idx_local=robot.find_free_base_dofs_idx_local())
    torch.testing.assert_close(base_vel[:, :3], ref.base_lin_vel_w)
    # The free joint's angular DOFs are body-frame: the world spin rotated by the reset quaternion.
    torch.testing.assert_close(base_vel[:, 3:], inv_transform_by_quat(ref.base_ang_vel_w, ref.base_quat_w))
    assert not torch.allclose(base_vel[:, 3:], ref.base_ang_vel_w, atol=1e-3)
    # The reference velocity survives the qpos write that follows it.
    torch.testing.assert_close(robot.get_dofs_vel(), ref.dofs_vel)
    assert ref.dofs_vel.abs().min() > 0


def test_reset_seeds_an_immobile_base_by_joint_name_clipped_to_its_limits(env):
    env.reset()

    arm = env.entities["arm"]
    expected = _arm_expected(env, _init_state(env, "arm").dofs_pos)
    torch.testing.assert_close(arm.get_dofs_pos(), expected)
    # The reference sits outside joint4's range [-3.07, -0.07]: the clip used that joint's own row.
    assert (expected[:, ARM_DOFS.index("joint4")] < 0).all()
    torch.testing.assert_close(arm.get_pos()[:, :2], torch.tensor([[0.0, 2.0]] * 4, device=env.device))


def test_reset_puts_the_free_object_on_its_reference(env):
    env.reset()

    obj, ref = env.entities["object"], _init_state(env, "object")
    torch.testing.assert_close(obj.get_qpos(), torch.cat([ref.base_pos_w, ref.base_quat_w], dim=1))
    vel = obj.get_dofs_vel()
    torch.testing.assert_close(vel[:, 3:], inv_transform_by_quat(ref.base_ang_vel_w, ref.base_quat_w))
    assert not torch.allclose(vel[:, 3:], ref.base_ang_vel_w, atol=1e-3)
    torch.testing.assert_close(vel[:, :3], ref.base_lin_vel_w)  # linear DOFs stay world-frame


def test_a_partial_reset_reseeds_only_those_envs(env):
    env.reset()
    robot = env.entities["robot"]
    moved = robot.get_qpos().clone()
    moved[:, 0] += 1.0
    robot.set_qpos(moved)

    env.reset(torch.tensor([1, 3], device=env.device))

    qpos = robot.get_qpos()
    torch.testing.assert_close(qpos[[0, 2]], moved[[0, 2]])
    torch.testing.assert_close(qpos[[1, 3], :3], _init_state(env, "robot").base_pos_w[[1, 3]])


def test_overall_noise_scale_scales_every_perturbation(env):
    ranges = dict(
        pose_range={k: (-0.2, 0.2) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        velocity_range={k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        dofs_position_range=(-0.05, 0.05),
    )
    env.reset()
    robot, ref = env.entities["robot"], _init_state(env, "robot")

    _standalone(env, ReferenceStateInitialization, entity_name="robot", overall_noise_scale=0.0, **ranges).compute()
    torch.testing.assert_close(robot.get_qpos()[:, 7:], _clipped(env, "robot", ref.dofs_pos))
    torch.testing.assert_close(robot.get_qpos()[:, :3], ref.base_pos_w)

    _standalone(env, ReferenceStateInitialization, entity_name="robot", overall_noise_scale=1.0, **ranges).compute()
    offset = robot.get_qpos()[:, :3] - ref.base_pos_w
    assert (offset.abs() > 0).any() and (offset.abs() <= 0.2 + 1e-6).all()


def test_object_xy_perturbation_moves_only_x_and_y(env):
    env.reset()
    obj, ref = env.entities["object"], _init_state(env, "object")

    _standalone(env, ObjectReferenceStateInitialization, entity_name="object", xy_range=0.1).compute()

    qpos = obj.get_qpos()
    offset = qpos[:, :2] - ref.base_pos_w[:, :2]
    assert (offset.abs() <= 0.1 + 1e-6).all() and (offset.abs() > 0).any()
    torch.testing.assert_close(qpos[:, 2:], torch.cat([ref.base_pos_w[:, 2:], ref.base_quat_w], dim=1))


def test_a_batched_reset_leaves_what_firing_the_terms_one_by_one_leaves(env):
    """All three flavours with noise on, so the batch must also draw it in the same order."""
    env.reset()
    terms = [
        _standalone(
            env,
            ReferenceStateInitialization,
            entity_name="robot",
            pose_range={"x": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
            velocity_range={"z": (-0.5, 0.5)},
            dofs_position_range=(-0.05, 0.05),
        ),
        _standalone(env, ReferenceStateInitialization, entity_name="arm", dofs_position_range=(-0.05, 0.05)),
        _standalone(env, ObjectReferenceStateInitialization, entity_name="object", xy_range=0.1),
    ]
    entities = [env.entities[name] for name in ("robot", "arm", "object")]
    envs_idx = torch.tensor([0, 2, 3], device=env.device)

    def state():
        return [torch.cat([entity.get_qpos(), entity.get_dofs_vel()], dim=1) for entity in entities]

    torch.manual_seed(0)
    for term in terms:
        term.compute(envs_idx)
    one_by_one = state()
    for entity in entities:  # move everything off, so only what the batch writes can match
        entity.set_qpos(entity.get_qpos() + 0.1)

    torch.manual_seed(0)
    ReferenceStateInitializationBase.compute_batch(terms, envs_idx)

    for batched, expected in zip(state(), one_by_one, strict=True):
        torch.testing.assert_close(batched[envs_idx], expected[envs_idx])
    assert not torch.allclose(state()[0][1], one_by_one[0][1])  # env 1 was not reset


#: The Genesis state setters: each resets the selected envs' collider and constraint state, and the
#: position writers run forward kinematics over every link of them, whichever entity they write.
GENESIS_STATE_WRITES = ("set_dofs_velocity", "set_dofs_position", "set_qpos")


def test_a_reset_seeding_three_entities_makes_one_velocity_and_one_position_write(env, monkeypatch):
    """The cost pin of batching: two Genesis writes and one forward pass for the batch, where one by one made six.

    The arm's joint positions go out as qpos with the others' (``qpos0 + value``), not through
    ``set_dofs_position``, which would re-derive the free bases' quaternions. Counts at the Genesis
    boundary — the solver's own setters, wrapped to pass straight through.
    """
    calls = []
    solver = env.rigid_solver
    for name in GENESIS_STATE_WRITES:
        original = getattr(solver, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(solver, name, counted)

    mask = torch.tensor([True, False, True, True], device=env.device)
    env.event_manager.compute(RESET, envs_idx=mask, global_env_step_count=0)

    assert sorted(calls) == ["set_dofs_velocity", "set_qpos"]


def test_a_reset_seeds_from_the_window_it_fills_and_gathers_no_init_state(env, monkeypatch):
    """The window starts at the current time, so the reset gathers each reference row once.

    One window fill per tracked entity, no separate init-state gather, and the robot lands on the
    window's first frame.
    """
    term = env.motion_manager.get_term("default")
    calls = []
    for name in ("compute", "fill_init_reference_state"):
        original = getattr(term, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(term, name, counted)
    mask = torch.tensor([True, False, True, True], device=env.device)

    env._reset_idx(mask)
    env._dispatch_post_reset()

    assert calls == ["compute"] * len(SEQUENCES)
    ref = env.motion_manager.data["robot"]
    qpos = env.entities["robot"].get_qpos()[mask]
    torch.testing.assert_close(qpos[:, :7], torch.cat([ref.base_pos_w[mask, 0], ref.base_quat_w[mask, 0]], dim=1))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="times the reset on CUDA")
def test_the_reset_cost_grows_slowly_with_the_number_of_seeded_entities():
    """Wall-clock pin of the batched reset: ``_reset_idx`` for a G1 plus 7 boxes vs a G1 plus 1, each with its RSI term.

    The 8-entity / 2-entity ratio at 2048 envs (26 resetting), on an RTX 3080 and on an L40S:

    * batched writes (this code): 1.9x / 2.1x
    * each term writing alone, gathers still per entity (batching turned off): 2.85x / 2.9x
    * before batching, every term regathering every entity: 4.4x / 4.7x

    The bound sits between the first two, so it catches losing the batch, not only the N^2 gathers;
    ``test_a_reset_seeding_three_entities_makes_one_velocity_and_one_position_write`` pins the write count
    exactly. A ratio, so the GPU's speed largely cancels out. Each count runs in its own process
    (``tests/support/reset_bench.py``): this one holds Genesis on gs.cpu.

    Each count is timed twice, interleaved, and the faster run kept: one run times only ~2 s of resets,
    so a burst of host load during it once read 4.45x on a full-suite run (1.77x alone). Losing the
    batch is slower in every run, so the faster one still catches it.
    """
    repo = Path(__file__).resolve().parents[4]

    def reset_ms(num_boxes: int) -> float:
        result = subprocess.run(
            [sys.executable, "-m", "tests.support.reset_bench", str(num_boxes)],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
        assert line is not None, (
            f"reset_bench {num_boxes} printed no RESULT line; stderr tail:\n{result.stderr[-3000:]}"
        )
        return json.loads(line.removeprefix("RESULT "))["reset_idx_ms"]

    runs = [reset_ms(num_boxes) for _ in range(2) for num_boxes in (1, 7)]
    two, eight = min(runs[0::2]), min(runs[1::2])

    assert eight / two < 2.5, (
        f"_reset_idx grew {eight / two:.2f}x from 2 to 8 seeded entities ({two:.2f} -> {eight:.2f} ms); "
        f"batched RSI writes keep it near 2x. Is a term seeding its entity outside the batch?"
    )


def test_teleport_of_an_immobile_base_writes_its_joints_only(env):
    env.reset()
    arm = env.entities["arm"]
    before = arm.get_dofs_pos().clone()
    ref = env.motion_manager.data["arm"]
    wrap = torch.tensor([0, 3], device=env.device)

    _standalone(env, ReferenceStateInitialization, entity_name="arm").teleport(wrap, ref)

    by_name = [ARM_REFERENCE_ORDER.index(n) for n in ARM_DOFS]
    torch.testing.assert_close(arm.get_dofs_pos()[wrap], ref.dofs_pos[wrap, 0][:, by_name])
    torch.testing.assert_close(arm.get_dofs_vel()[wrap], ref.dofs_vel[wrap, 0][:, by_name])
    torch.testing.assert_close(arm.get_dofs_pos()[[1, 2]], before[[1, 2]])


@pytest.mark.parametrize(
    "cls, entity, match",
    [
        (
            ObjectReferenceStateInitialization,
            "shelf",
            "ObjectReferenceStateInitialization on entity 'shelf' writes a base pose, so it requires a "
            "free-floating entity; it is fixed-base",
        ),
        (ReferenceStateInitialization, "shelf", r"'shelf' is not tracked by any motion term; tracked entities: \["),
        (ReferenceStateInitialization, "crate", r"'crate' is not tracked by any motion term; tracked entities: \["),
    ],
)
def test_rsi_on_an_entity_it_cannot_seed_fails_at_build(env, cls, entity, match):
    with pytest.raises(ValueError, match=match):
        _standalone(env, cls, entity_name=entity)


# ── depenetration cap (PhysX max_depenetration_velocity) ──────────────────────────────────────────


def _restoring_callbacks(env):
    """Yield ``env``, then drop the sub-step callbacks registered meanwhile (a test's standalone terms, its probes)."""
    pre, post = list(env._pre_substep_callbacks), list(env._substep_callbacks)
    yield env
    env._pre_substep_callbacks[:], env._substep_callbacks[:] = pre, post


@pytest.fixture
def cap_env(env):
    yield from _restoring_callbacks(env)


def _solver_step(term, entity, dofs_idx, push: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bracket a fake sim step that falls freely and adds ``push``, as a solver resolving an overlap would.

    Returns ``(free, stepped)``: the free-fall velocity, and the velocity the step left before the cap ran.
    """
    term._record_depenetration_vel()
    env = term._env
    free = entity.get_dofs_vel(dofs_idx_local=dofs_idx) + env.rigid_solver.get_gravity() * env.env_options.sim_dt
    entity.set_dofs_vel(free + push, dofs_idx_local=dofs_idx)
    stepped = entity.get_dofs_vel(dofs_idx_local=dofs_idx).clone()
    term._cap_depenetration()
    return free, stepped


def _seeded(term, entity, dofs_idx) -> torch.Tensor:
    """The base linear velocity ``term`` just seeded, which the cap measures from."""
    return entity.get_dofs_vel(dofs_idx_local=dofs_idx).clone()


def test_a_kick_is_scaled_back_along_itself_until_the_base_is_the_cap_from_its_seed(cap_env):
    env = cap_env
    env.reset()
    term = _standalone(env, ObjectReferenceStateInitialization, entity_name="object", max_depenetration_velocity=0.5)
    obj, lin = env.entities["object"], torch.arange(3, device=env.device)
    term.compute()
    seed = _seeded(term, obj, lin)

    free, stepped = _solver_step(term, obj, lin, torch.tensor([3.0, 0.0, 4.0], device=env.device))  # a 5 m/s kick
    vel = obj.get_dofs_vel(dofs_idx_local=lin)
    torch.testing.assert_close((vel - seed).norm(dim=-1), torch.full((env.num_envs,), 0.5, device=env.device))
    # Only the push is scaled: what is left of it points the way the solver pushed (x and z; it had no y).
    kept = ((vel - free) / (stepped - free))[:, [0, 2]]
    torch.testing.assert_close(kept[:, 0], kept[:, 1])
    assert bool(((kept > 0) & (kept < 1)).all())
    torch.testing.assert_close(vel[:, 1], free[:, 1], rtol=0.0, atol=0.0)


def test_a_lasting_overlap_cannot_stack_its_pushes_past_the_cap(cap_env):
    """Each push is under the cap on its own; five of them would carry the base 1.5 m/s off its seed uncapped."""
    env = cap_env
    env.reset()
    term = _standalone(env, ObjectReferenceStateInitialization, entity_name="object", max_depenetration_velocity=0.5)
    obj, lin = env.entities["object"], torch.arange(3, device=env.device)
    term.compute()
    seed = _seeded(term, obj, lin)
    up = torch.tensor([0.0, 0.0, 0.3], device=env.device)

    _, stepped = _solver_step(term, obj, lin, up)
    torch.testing.assert_close(obj.get_dofs_vel(dofs_idx_local=lin), stepped)  # the first, alone, is kept
    for _ in range(4):
        _solver_step(term, obj, lin, up)
    torch.testing.assert_close(
        (obj.get_dofs_vel(dofs_idx_local=lin) - seed).norm(dim=-1), torch.full((env.num_envs,), 0.5, device=env.device)
    )


def test_support_and_a_push_back_towards_the_seed_are_the_solvers_to_keep(cap_env):
    env = cap_env
    env.reset()
    term = _standalone(env, ObjectReferenceStateInitialization, entity_name="object", max_depenetration_velocity=0.5)
    obj, lin = env.entities["object"], torch.arange(3, device=env.device)
    term.compute()
    gravity_step = env.rigid_solver.get_gravity()[0] * env.env_options.sim_dt

    for _ in range(term._depen_steps - 2):  # a floor holding a resting body up, for most of the window
        _, stepped = _solver_step(term, obj, lin, -gravity_step)
        torch.testing.assert_close(obj.get_dofs_vel(dofs_idx_local=lin), stepped, rtol=0.0, atol=0.0)

    _solver_step(term, obj, lin, torch.tensor([0.0, 0.0, 3.0], device=env.device))  # saturate at the cap
    _, stepped = _solver_step(term, obj, lin, torch.tensor([0.0, 0.0, -0.4], device=env.device))  # then slow down
    torch.testing.assert_close(obj.get_dofs_vel(dofs_idx_local=lin), stepped, rtol=0.0, atol=0.0)


def test_the_cap_lets_go_after_its_window_without_touching_the_solvers_velocity(cap_env):
    env = cap_env
    env.reset()
    term = _standalone(env, ObjectReferenceStateInitialization, entity_name="object", max_depenetration_velocity=0.5)
    obj, lin = env.entities["object"], torch.arange(3, device=env.device)
    term.compute()
    for _ in range(term._depen_steps):  # run the window out
        _solver_step(term, obj, lin, torch.zeros(3, device=env.device))

    _, stepped = _solver_step(term, obj, lin, torch.tensor([3.0, 0.0, 4.0], device=env.device))
    torch.testing.assert_close(obj.get_dofs_vel(dofs_idx_local=lin), stepped, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("by_mask", [False, True], ids=["env_ids", "env_mask"])
def test_a_resync_arms_the_cap_for_the_wrapped_envs_only(cap_env, by_mask):
    """The motion-end resync writes through int ids or the wrap mask; either arms exactly the wrapped envs."""
    env = cap_env
    env.reset()
    term = _standalone(env, ReferenceStateInitialization, entity_name="robot", max_depenetration_velocity=1.0)
    robot, ref = env.entities["robot"], env.motion_manager.data["robot"]
    lin = term._depen_lin_idx
    term._depen_left.zero_()
    wrap = torch.arange(env.num_envs, device=env.device) == 0 if by_mask else torch.tensor([0], device=env.device)
    term.teleport(wrap, ref)
    seed = _seeded(term, robot, lin)

    _, stepped = _solver_step(term, robot, lin, torch.tensor([0.0, 0.0, 3.0], device=env.device))
    vel = robot.get_dofs_vel(dofs_idx_local=lin)
    torch.testing.assert_close(vel[0], seed[0] + torch.tensor([0.0, 0.0, 1.0], device=env.device))  # held to 1 m/s
    torch.testing.assert_close(vel[1:], stepped[1:], rtol=0.0, atol=0.0)  # not wrapped: no window, not written


def test_env_step_brackets_every_sim_step_with_the_substep_callbacks(cap_env):
    env = cap_env
    calls = []
    env.register_substep_callback(lambda: calls.append("before"), before=True)
    env.register_substep_callback(lambda: calls.append("after"))
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    assert calls == ["before", "after"] * env.env_options.decimation


#: The cap the end-to-end env's object RSI term sets, and the push per sim step its test drives into it, m/s.
CAP, PUSH = 1.0, 5.0


@pytest.fixture(scope="module")
def _capped_env(library):
    from eden.options.managers.terminations import TerminationManagerOptions
    from eden.terms import DoneTerm

    terms = TerminationManagerOptions(time_out=DoneTerm.configure(func=en.terminations.time_out, time_out=True))
    return _rsi_env(library, object_rsi={"max_depenetration_velocity": CAP}, termination_options=terms)


@pytest.fixture
def capped_env(_capped_env):
    """An env whose own object RSI term caps depenetration at :data:`CAP`, and whose episodes time out."""
    yield from _restoring_callbacks(_capped_env)


def test_an_auto_reset_inside_env_step_arms_the_cap_and_the_next_steps_clamp(capped_env):
    """The event manager's own capped RSI term, reset by a timeout inside ``env.step()``: on the next step a
    :data:`PUSH` every sim step is held to :data:`CAP` off the seed in all, for that env only.
    """
    env = capped_env
    term = next(t for _, name, t in env.event_manager.iter_terms() if name == "reset_object")
    obj, lin = env.entities["object"], torch.arange(3, device=env.device)
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    env.reset()
    for _ in range(term._depen_steps // env.env_options.decimation + 1):  # run the reset's windows out
        env.step(action)
    assert not term._depen_left.any()

    env.episode_length_buf[1] = int(env.max_episode_length) - 1
    env.step(action)
    assert bool(env.reset_buf[1]) and not bool(env.reset_buf[[0, 2, 3]].any())
    assert term._depen_left.tolist() == [0, term._depen_steps, 0, 0]

    def push():  # registered after the term's own before-step read, so it lands as what the step added
        obj.set_dofs_vel(obj.get_dofs_vel(dofs_idx_local=lin) + torch.tensor([0.0, 0.0, PUSH], device=env.device), lin)

    env.register_substep_callback(push, before=True)
    before = obj.get_dofs_vel(dofs_idx_local=lin)[:, 2].clone()
    env.step(action)
    gained = obj.get_dofs_vel(dofs_idx_local=lin)[:, 2] - before
    steps, gravity_loss = env.env_options.decimation, 1.0  # gravity takes ~0.05 m/s per sim step; 1.0 is slack
    assert float(gained[1]) == pytest.approx(CAP, abs=1e-4)  # in the window: the cap off the seed, not CAP per step
    assert bool((gained[[0, 2, 3]] > steps * (PUSH - gravity_loss)).all())  # outside it: the full push


@pytest.mark.parametrize(
    "cls, params, match",
    [
        (
            ObjectReferenceStateInitialization,
            dict(entity_name="object", max_depenetration_velocity=0.0),
            "must both be > 0",
        ),
        (
            ObjectReferenceStateInitialization,
            dict(entity_name="object", max_depenetration_velocity=1.0, depenetration_window_s=0.0),
            "must both be > 0",
        ),
        (ReferenceStateInitialization, dict(entity_name="arm", max_depenetration_velocity=1.0), "has no base to cap"),
    ],
)
def test_a_cap_it_cannot_apply_fails_at_build(cap_env, cls, params, match):
    env = cap_env
    with pytest.raises(ValueError, match=match):
        _standalone(env, cls, **params)
