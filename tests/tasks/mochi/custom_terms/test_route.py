"""Tests for the cable's routed goal, its welded end and a friction grasp, on ``mochi_route_the_cable``."""

import genesis as gs
import numpy as np
import pytest
import torch

from eden.tasks.mochi.route_the_cable import config as route_config
from tests.tasks.mochi.conftest import action, build_task, place_tcp


@pytest.fixture(scope="module")
def env():
    return build_task("mochi_route_the_cable")


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_routed_boxes_count_segments_and_place_the_end(env):
    """Env 0's cable runs through the first spool's seat and ends in the port; env 1's lies where it started."""
    env.reset()
    cable = env.entities["cable"]
    verts = cable.get_vertex_positions().clone()
    seat = route_config.seat_box(0)
    inside = 0.5 * (torch.tensor(seat["lower"]) + torch.tensor(seat["upper"])).to(env.device)
    port = torch.tensor([route_config.PORT[0], route_config.PORT[1], route_config.RADIUS], device=env.device)
    verts[0, 10:15] = inside  # the four segments between nodes 10..14 have their midpoints in the seat
    verts[0, -1] = port
    cable.set_vertex_positions(verts)
    metrics = env.metric_manager
    seat0 = metrics.get_term("seat0").compute()
    assert float(seat0[0]) == pytest.approx(4 / route_config.SEGMENTS)
    assert float(seat0[1]) == 0.0
    assert metrics.get_term("end_in_port").compute().tolist() == [1.0, 0.0]


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_jaws_lift_the_cable_by_friction_and_the_welded_end_holds_through_a_reset(env):
    """Env 0's right hand pinches the cable on the bench and lifts; env 1's closes 10 cm above it."""
    env.reset()
    cable = env.entities["cable"]
    welded = cable.get_vertex_positions()[:, route_config.ANCHORED_NODES].clone()
    k = 25
    node = cable.get_vertex_positions()[:, k].clone()
    pinch = node.clone()
    pinch[:, 2] = 0.0165  # the fingertips on the bench, the cable between the pads
    pinch[1, 2] = 0.12
    place_tcp(env, "right", pinch)
    for _ in range(25):
        env.step(action(env, close={"right_gripper": True}))
    assert float(env.metric_manager.get_value("right_grip")[0]) > 1.0
    up = torch.tensor([0.0, 0.0, 0.5], device=env.device)
    for _ in range(30):
        env.step(action(env, close={"right_gripper": True}, move={"right_hand": up}))
    rise = cable.get_vertex_positions()[:, k, 2] - node[:, 2]
    assert float(rise[0]) > 0.02  # held by the pads' friction alone
    assert abs(float(rise[1])) < 0.002
    torch.testing.assert_close(cable.get_vertex_positions()[:, route_config.ANCHORED_NODES], welded, atol=1e-5, rtol=0)

    env.reset()
    torch.testing.assert_close(cable.get_vertex_positions()[:, route_config.ANCHORED_NODES], welded, atol=1e-5, rtol=0)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_diverged_env_is_reset_while_the_others_go_on(env):
    """One diverged Mochi solve truncates and resets its env rather than raising for the whole scene."""
    from eden.engine.solvers.mochi.error_codes import MochiErrorCode

    env.reset()
    for _ in range(3):
        env.step(action(env))
    length_before = env.episode_length_buf.clone()
    # The flag the step kernel writes in env 1 when its solve diverges, which the end of the next step latches
    env.mochi_solver._errno.from_numpy(np.array([0, int(MochiErrorCode.MOCHI_DIVERGED)], dtype=np.int32))

    env.step(action(env))
    assert env.termination_manager.solver_errors.tolist() == [False, True]
    assert env.episode_length_buf.tolist() == [int(length_before[0]) + 1, 0]  # env 1 reset, env 0 carried on
    assert not env.get_solver_error_envs_mask().any()
    for _ in range(3):  # each starts with Genesis' check of the flags, which finds none
        env.step(action(env))


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_reset_env_reports_no_control_force_until_its_next_step(env):
    """The MochiSolver reports the force the last step applied, which a reset leaves at that of the episode that ended."""
    env.reset()
    for _ in range(3):
        env.step(action(env, close={"right_gripper": True}))
    robot = env.entities["right"]
    assert (robot.get_dofs_control_force()[1] != 0.0).any()

    env.reset(envs_idx=torch.tensor([1], device=env.device))
    assert (robot.get_dofs_control_force()[1] == 0.0).all()
    assert (robot.get_dofs_control_force()[0] != 0.0).any()


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_contact_capacity_overflow_raises_though_a_setter_clears_its_env(env):
    """The MochiSolver's setters clear an env's flags, and a RESET event or a manager's reset sets the state of the
    bodies of the envs it resets: the env keeps the overflow across all of it, and the next control step raises.
    """
    from eden.engine.solvers.mochi.error_codes import MochiErrorCode

    env.reset()
    env.step(action(env))
    env.mochi_solver._errno.from_numpy(np.array([0, int(MochiErrorCode.OVERFLOW_MOCHI_CONTACTS)], dtype=np.int32))
    try:
        env.step(action(env))
        assert env.termination_manager.solver_errors.tolist() == [False, True]
        robot = env.entities["right"]
        robot.set_qpos(robot.get_qpos()[1:], envs_idx=torch.tensor([1], device=env.device))  # what a reset event does
        assert not env.get_solver_error_envs_mask().any()  # the solver's flags are cleared, the env keeps its own
        with pytest.raises(gs.GenesisException, match="Exceeding the capacity of a contact list"):
            env.step(action(env))
    finally:
        env.mochi_solver.clear_errno()


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_contact_capacity_overflow_raises_though_a_substep_callback_clears_its_env(env):
    """A substep callback runs between the sim step that flagged an env and the termination manager that reads the
    flag, so a setter it calls would leave nothing to read: the env keeps the overflow, and the next control step raises.
    """
    from eden.engine.solvers.mochi.error_codes import MochiErrorCode

    robot = env.entities["right"]
    env_1 = torch.tensor([1], device=env.device)

    def overwrite_env_1() -> None:
        robot.set_qpos(robot.get_qpos()[1:], envs_idx=env_1)

    env.reset()
    env.step(action(env))
    env.mochi_solver._errno.from_numpy(np.array([0, int(MochiErrorCode.OVERFLOW_MOCHI_CONTACTS)], dtype=np.int32))
    env.register_substep_callback(overwrite_env_1)
    try:
        env.step(action(env))
        assert not env.termination_manager.solver_errors.any()  # the setter cleared the flag before it was read
        with pytest.raises(gs.GenesisException, match="Exceeding the capacity of a contact list"):
            env.step(action(env))
    finally:
        env._substep_callbacks.remove(overwrite_env_1)
        env.mochi_solver.clear_errno()
