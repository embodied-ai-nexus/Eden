"""The oracle pose commands in ``eden/managers/terms/commands/oracle.py``."""

from __future__ import annotations

import pytest
import math

import torch

from eden.managers.terms.commands.oracle import UniformSE3Command, UniformT3Command

pytestmark = pytest.mark.requires_genesis

MASK = torch.tensor([True, False, True, False])


def test_t3_resample_writes_the_draw_into_the_selected_rows_only(box_env, make_term):
    term = make_term(UniformT3Command, box_env, x_range=(1.0, 1.0), y_range=(2.0, 2.0), z_range=(3.0, 3.0))
    term.oracle_pos_b.fill_(5.0)
    term._resample_command(MASK)
    torch.testing.assert_close(term.command[MASK], torch.tensor([[1.0, 2.0, 3.0]] * 2))
    torch.testing.assert_close(term.command[~MASK], torch.full((2, 3), 5.0))


def test_se3_command_starts_at_identity_and_resamples_the_selected_rows(box_env, make_term):
    term = make_term(
        UniformSE3Command,
        box_env,
        x_range=(0.1, 0.1),
        y_range=(0.2, 0.2),
        z_range=(0.3, 0.3),
        roll_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        yaw_range=(math.pi / 2, math.pi / 2),
    )
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 4)
    torch.testing.assert_close(term.command, identity)  # wxyz with a positive real part

    term._resample_command(MASK)

    yawed = [0.1, 0.2, 0.3, math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    torch.testing.assert_close(term.command[MASK], torch.tensor([yawed] * 2))
    torch.testing.assert_close(term.command[~MASK], identity[:2])
