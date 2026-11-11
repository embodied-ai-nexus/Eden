"""The gait phase command in ``eden/managers/terms/commands/gait.py``."""

from __future__ import annotations

import math

import pytest
import torch

from eden.managers.command_manager import CommandManager
from eden.managers.terms.commands.gait import GaitPhaseCommand
from eden.managers.terms.commands.velocity import UniformVelocityCommand
from eden.options.managers.commands import CommandManagerOptions

pytestmark = pytest.mark.requires_genesis

STEP = 2.0 * math.pi * 1.0 * 0.02  # one update at 1 Hz and the scene's control dt of 0.02 s


@pytest.fixture
def velocity_command(box_env, install_manager):
    """``velocity_command(rows)``: a real ``base_velocity`` command term holding ``rows``."""

    def install(rows):
        options = CommandManagerOptions(base_velocity=UniformVelocityCommand.configure(entity_name="robot"))
        manager = install_manager(box_env, "command_manager", CommandManager(box_env, options))
        manager.get_term("base_velocity").vel_command_b[:] = torch.tensor(rows)

    return install


def _gait(env, make_term, **params):
    return make_term(GaitPhaseCommand, env, gait_frequency=1.0, **params)


def test_gait_phase_advances_each_foot_from_its_offset(box_env, make_term):
    term = _gait(box_env, make_term)
    term._update_command()
    torch.testing.assert_close(term.command, torch.tensor([[STEP, STEP - math.pi]] * 4))  # right foot wrapped


def test_gait_phase_is_pinned_while_the_velocity_command_says_stand(box_env, make_term, velocity_command):
    """Only env 0 is below the threshold on every axis; the others keep cycling."""
    velocity_command([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.5], [2.0, 0.0, 0.0]])
    term = _gait(box_env, make_term, velocity_command_name="base_velocity", stand_velocity_threshold=0.1)
    for _ in range(5):
        term._update_command()
    torch.testing.assert_close(term.command[0], torch.full((2,), math.pi))
    torch.testing.assert_close(term.command[1:, 0], torch.full((3,), 5 * STEP))


def test_a_zero_stand_threshold_never_pins(box_env, make_term, velocity_command):
    """The check is strict (``<``), so at 0.0 even an exactly-zero command keeps cycling."""
    velocity_command([[0.0, 0.0, 0.0]] * 4)
    term = _gait(box_env, make_term, velocity_command_name="base_velocity", stand_velocity_threshold=0.0)
    term._update_command()
    torch.testing.assert_close(term.command[:, 0], torch.full((4,), STEP))


def test_gait_phase_rejects_a_velocity_command_without_three_columns(box_env, make_term, install_manager):
    """A gait phase is ``(num_envs, 2)``: named as the velocity source it fails loud, not past its last column."""
    options = CommandManagerOptions(base_velocity=GaitPhaseCommand.configure())
    install_manager(box_env, "command_manager", CommandManager(box_env, options))
    term = _gait(box_env, make_term, velocity_command_name="base_velocity")
    with pytest.raises(ValueError, match=r"\(num_envs, >=3\)"):
        term._update_command()


@pytest.mark.parametrize(
    ("num_feet", "phase_offset", "expected"),
    [(2, math.pi, [0.0, math.pi]), (4, math.pi / 2, [0.0, math.pi / 2, math.pi, 3 * math.pi / 2])],
)
def test_foot_i_starts_at_i_times_the_phase_offset(box_env, make_term, num_feet, phase_offset, expected):
    term = _gait(box_env, make_term, phase_offset=phase_offset, num_feet=num_feet)
    torch.testing.assert_close(term.command, torch.tensor([expected] * 4))


def test_reset_restores_the_start_phase_of_the_reset_envs_only(box_env, make_term):
    term = _gait(box_env, make_term)
    for _ in range(5):
        term.compute(box_env.dt)
    advanced = term.command[1].clone()
    term.reset(torch.tensor([True, False, True, False]))
    torch.testing.assert_close(term.command[[0, 2]], torch.tensor([[0.0, math.pi]] * 2))
    torch.testing.assert_close(term.command[[1, 3]], advanced.expand(2, -1))
    assert not torch.allclose(advanced, torch.tensor([0.0, math.pi]))
