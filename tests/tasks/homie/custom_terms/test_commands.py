"""``HomieCommand``: the tri-modal squat / stand / walk resample."""

from __future__ import annotations

import torch


def test_resample_never_commands_walking_and_squatting_together(homie_env):
    term = homie_env.command_manager.get_term("homie_command")
    every_env = torch.arange(homie_env.num_envs, device=homie_env.device)
    draws = []
    for _ in range(300):
        term.reset(every_env)
        draws.append(term.command.clone())
    cmd = torch.cat(draws)
    moving = cmd[:, :3].abs().amax(dim=1) > 0.0
    squatting = cmd[:, 3] < term.default_height

    assert not (moving & squatting).any()
    # Squats land in default + [-0.5, 0], i.e. [0.24, 0.74] m.
    assert (cmd[squatting, 3] >= term.default_height - 0.5 - 1e-6).all()
    assert (cmd[~squatting, 3] == term.default_height).all()
    # Regime probabilities 1/2 walk, 1/3 squat, 1/6 stand (1200 draws; ~5 sigma bands).
    n = cmd.shape[0]
    assert abs(moving.sum().item() / n - 1 / 2) < 0.08
    assert abs(squatting.sum().item() / n - 1 / 3) < 0.07
