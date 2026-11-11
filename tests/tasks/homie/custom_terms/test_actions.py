"""The homie action terms: the upper-body pose curriculum and OpenHomie's motor model."""

from __future__ import annotations

import torch


def _step(env, n: int, action: float = 0.0) -> None:
    actions = torch.full((env.num_envs, env.action_manager.total_action_dim), action, device=env.device)
    for _ in range(n):
        env.step(actions)


def test_upper_body_moves_within_hard_limits_in_training_and_holds_default_in_eval(homie_env):
    term = homie_env.action_manager.get_term("upper_body")
    robot = homie_env.entities["robot"]
    pos = [list(robot.dofs_name).index(n) for n in term.dofs_name]
    lower, upper = (b[:, pos] for b in robot.get_dofs_limit())
    homie_env.reset()
    _step(homie_env, 3)
    target = term.processed_action
    assert (target >= lower - 1e-6).all() and (target <= upper + 1e-6).all()
    assert term.normalized_action.abs().amax() > 0.0

    homie_env.training = False
    try:
        homie_env.reset()
        _step(homie_env, 3)
        assert torch.equal(term.normalized_action, torch.zeros_like(term.normalized_action))
    finally:
        homie_env.training = True
        homie_env.reset()


def test_motor_model_clips_the_leg_torque_to_the_effort_limit(homie_env):
    leg = homie_env.action_manager.get_term("dofs_pos_controller")
    robot = homie_env.entities["robot"]
    pos = [list(robot.dofs_name).index(n) for n in leg.dofs_name]
    homie_env.reset()
    _step(homie_env, 2, action=50.0)  # targets far past every joint limit
    tau = robot.get_dofs_control_force()[:, pos]
    _, effort = robot.get_dofs_force_range()
    assert (tau.abs() <= effort[:, pos] + 1e-4).all()
    assert torch.isclose(tau.abs(), effort[:, pos], rtol=1e-4).any()
    homie_env.reset()
