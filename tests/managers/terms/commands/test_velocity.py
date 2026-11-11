"""The SE(2) velocity command terms in ``eden/managers/terms/commands/velocity.py``.

``ConstantVelocityCommand`` holds a value, ``OperatorVelocityCommand`` re-reads it from a keyboard or
gamepad; a deployment swaps either in for a task's trained ``UniformVelocityCommand``. The input
devices need a display or a plugged-in pad, so the operator tests install ``FakeSE2Device`` as the
device ``make_se2_device`` returns.
"""

from __future__ import annotations

import math
import pytest
import torch

from eden.managers.terms.commands.velocity import (
    ConstantVelocityCommand,
    OperatorVelocityCommand,
    UniformVelocityCommand,
)
from eden.utils.devices import se2

pytestmark = pytest.mark.requires_genesis

RANGES = {"lin_vel_x_range": (-1.0, 1.0), "lin_vel_y_range": (-1.0, 1.0), "ang_vel_z_range": (-0.5, 0.5)}
PINNED = {"lin_vel_x_range": (0.5, 0.5), "lin_vel_y_range": (0.0, 0.0), "ang_vel_z_range": (0.0, 0.0)}


class FakeSE2Device:
    """What ``SE2Keyboard`` / ``SE2Gamepad`` give the term: a settable normalized deflection."""

    def __init__(self):
        self.deflection = torch.zeros(3)
        self.stopped = False

    def get_command(self) -> torch.Tensor:
        return self.deflection

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def fake_device(monkeypatch):
    """The input device boundary: every ``make_se2_device`` call returns this fake."""
    device = FakeSE2Device()
    monkeypatch.setattr(se2, "make_se2_device", lambda name, options=None: device)
    return device


@pytest.fixture
def constant(box_env, make_term):
    def make(**overrides):
        return make_term(ConstantVelocityCommand, box_env, **{"entity_name": "robot", **RANGES, **overrides})

    return make


@pytest.fixture
def operator(box_env, make_term, fake_device):
    def make(**overrides):
        return make_term(OperatorVelocityCommand, box_env, **{"entity_name": "robot", **RANGES, **overrides})

    return make


def _rows(values, n=4):
    return torch.tensor([values] * n)


# ---------------------------------------------------------------------------
# ConstantVelocityCommand
# ---------------------------------------------------------------------------


def test_constant_holds_its_value_across_computes(constant):
    """A deployed policy stops being handed a fresh random command."""
    term = constant(command_value=(0.4, -0.2, 0.3))
    for _ in range(5):
        term.compute(dt=0.02)
    torch.testing.assert_close(term.command, _rows([0.4, -0.2, 0.3]))


def test_constant_clamps_into_the_trained_ranges_and_says_so(constant, eden_caplog):
    term = constant(command_value=(5.0, 0.0, -3.0))
    assert term.command[0].tolist() == pytest.approx([1.0, 0.0, -0.5])
    assert "outside the trained ranges" in eden_caplog.text


def test_constant_set_command_rewrites_the_held_value(constant):
    term = constant()
    assert term.set_command((0.6, 0.0, 0.0))[0] == pytest.approx(0.6)
    torch.testing.assert_close(term.command, _rows([0.6, 0.0, 0.0]))
    with pytest.raises(ValueError, match="is \\(vx, vy, wz\\)"):
        term.set_command((0.5, 0.0))


def test_a_degenerate_range_pins_its_axis(constant, eden_caplog):
    """``move_forward`` trains at exactly 0.5 m/s: the clamp holds that value whatever is asked, loudly."""
    term = constant(command_value=(0.0, 0.0, 0.0), **PINNED)
    assert term.command[0].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert "outside the trained ranges" in eden_caplog.text


def test_the_clamp_bounds_are_read_live(constant, eden_caplog):
    """A curriculum widens the range fields in place; the clamp and its message follow."""
    term = constant()
    assert term.set_command((3.0, 0.0, 0.0))[0] == pytest.approx(1.0)
    term.lin_vel_x_range = (-3.0, 3.0)
    assert term.set_command((3.0, 0.0, 0.0))[0] == pytest.approx(3.0)
    term.set_command((5.0, 0.0, 0.0))
    assert "vx (-3.0, 3.0)" in eden_caplog.text and "holding [3.0" in eden_caplog.text


# ---------------------------------------------------------------------------
# OperatorVelocityCommand
# ---------------------------------------------------------------------------


def test_operator_scales_each_device_reading_into_the_trained_range(operator, fake_device):
    """Read every compute, for every env; full deflection is the range, and a stick past it is clamped."""
    term = operator()
    for deflection, expected in [
        ((1.0, 1.0, 1.0), (1.0, 1.0, 0.5)),
        ((-1.0, 0.0, -1.0), (-1.0, 0.0, -0.5)),
        ((0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ((5.0, -5.0, 5.0), (1.0, -1.0, 0.5)),  # a miscalibrated stick must not leave the training distribution
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ]:
        fake_device.deflection = torch.tensor(deflection)
        term.compute(dt=0.02)
        torch.testing.assert_close(term.command, _rows(expected), msg=str(deflection))


def test_operator_reaches_both_ends_of_an_asymmetric_range(operator, fake_device):
    term = operator(lin_vel_x_range=(-2.0, 3.0))
    for deflection, vx in [(1.0, 3.0), (-1.0, -2.0), (-0.5, -1.5)]:  # the gain is 3.0 on the way back too
        fake_device.deflection = torch.tensor([deflection, 0.0, 0.0])
        term.compute(dt=0.02)
        assert term.command[0, 0] == pytest.approx(vx)


def test_operator_seeds_the_buffer_before_the_first_device_read(operator, fake_device):
    """``dep.reset()`` observes before any compute, so the first command must already be in range."""
    fake_device.deflection = torch.tensor([1.0, 0.0, 0.0])
    term = operator(command_value=(0.25, 0.0, 0.0))
    assert term.command[0].tolist() == pytest.approx([0.25, 0.0, 0.0])
    term.compute(dt=0.02)
    assert term.command[0].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_operator_seed_does_not_warn_on_a_zero_excluding_range(operator, eden_caplog):
    """The seed is a placeholder the first read overwrites; an explicit ``set_command`` still reports."""
    term = operator(**PINNED)
    assert term.command[0].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert "outside the trained ranges" not in eden_caplog.text
    term.set_command((0.0, 0.0, 0.0))
    assert "outside the trained ranges" in eden_caplog.text


def test_operator_cannot_move_a_pinned_axis(operator, fake_device):
    term = operator(**PINNED)
    for deflection in ((1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (0.0, 0.0, 0.0)):
        fake_device.deflection = torch.tensor(deflection)
        term.compute(dt=0.02)
        assert term.command[0].tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_operator_follows_a_widened_range(operator, fake_device):
    term = operator()
    fake_device.deflection = torch.tensor([1.0, 0.0, 0.0])
    term.compute(dt=0.02)
    term.lin_vel_x_range = (-2.0, 2.0)
    term.compute(dt=0.02)
    assert term.command[0, 0] == pytest.approx(2.0)


def test_operator_close_stops_the_device(operator, fake_device):
    operator().close()
    assert fake_device.stopped


# ---------------------------------------------------------------------------
# Either substitute for the trained term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ["constant", "operator"])
def test_a_substitute_rejects_an_unknown_entity_at_build(request, factory):
    """The operator term gets this from ``super().build()``; a subclass that skipped it would not."""
    with pytest.raises(KeyError):
        request.getfixturevalue(factory)(entity_name="not_a_robot")


@pytest.mark.parametrize("factory", ["constant", "operator"])
def test_a_substitute_exposes_what_consumers_read_off_the_replaced_term(request, box_env, make_term, factory):
    """What in-tree code reads off a velocity command it did not build.

    Observations and the deploy loop read the command, the error metrics ``command.robot``, and
    ``VelocityCommandCurriculum`` widens the ranges. ``swap_term`` compares only the command tensor.
    """
    replaced = make_term(UniformVelocityCommand, box_env, entity_name="robot", **RANGES)
    substitute = request.getfixturevalue(factory)()
    for name in ("command", "vel_command_b", "robot", "lin_vel_x_range", "lin_vel_y_range", "ang_vel_z_range"):
        assert hasattr(substitute, name), name
    assert substitute.command.shape == replaced.command.shape
    assert substitute.robot is replaced.robot


def test_a_substitute_without_an_entity_reports_none(constant):
    """The task-less deploy path builds a term naming no entity."""
    assert constant(entity_name="").robot is None


# ---------------------------------------------------------------------------
# UniformVelocityCommand
# ---------------------------------------------------------------------------


def test_uniform_resample_writes_only_the_selected_envs(box_env, make_term):
    """A full-batch draw kept through ``torch.where`` leaves every unselected env untouched."""
    term = make_term(
        UniformVelocityCommand,
        box_env,
        entity_name="robot",
        heading_command=True,
        heading_range=(-3.0, 3.0),
        rel_heading_envs=0.0,
        rel_standing_envs=0.0,
        **RANGES,
    )
    term.vel_command_b.fill_(9.0)
    term.heading_target.fill_(9.0)
    term.is_heading_env.fill_(True)
    term.is_standing_env.fill_(True)
    mask = torch.tensor([True, False, True, False])

    term._resample_command(mask)

    torch.testing.assert_close(term.vel_command_b[~mask], torch.full((2, 3), 9.0))
    torch.testing.assert_close(term.heading_target[~mask], torch.full((2,), 9.0))
    assert torch.equal(term.is_heading_env, ~mask) and torch.equal(term.is_standing_env, ~mask)
    low, high = torch.tensor([-1.0, -1.0, -0.5]), torch.tensor([1.0, 1.0, 0.5])
    assert ((term.vel_command_b[mask] >= low) & (term.vel_command_b[mask] <= high)).all()
    assert (term.heading_target[mask].abs() <= 3.0).all()


def test_uniform_heading_steers_heading_envs_and_standing_envs_hold_still(box_env, make_term, restore_state):
    robot = restore_state(box_env.entities["robot"])
    robot.set_quat(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2 + [[math.cos(0.5), 0.0, 0.0, math.sin(0.5)]] + [[1.0, 0.0, 0.0, 0.0]])
    )
    term = make_term(
        UniformVelocityCommand,
        box_env,
        entity_name="robot",
        heading_command=True,
        heading_range=(-3.0, 3.0),
        heading_control_stiffness=0.5,
        **RANGES,
    )
    term.heading_target[:] = torch.tensor([0.4, 0.4, 0.2, 0.4])
    term.is_heading_env[:] = torch.tensor([True, False, True, False])
    term.is_standing_env[:] = torch.tensor([False, False, False, True])
    term.vel_command_b.fill_(0.3)

    term._update_command()

    # wz = clip(0.5 * (target - heading)) on heading envs (env 2 faces yaw 1.0); env 1 keeps its sampled wz.
    torch.testing.assert_close(term.vel_command_b[:, 2], torch.tensor([0.2, 0.3, -0.4, 0.0]))
    torch.testing.assert_close(term.vel_command_b[:3, :2], torch.full((3, 2), 0.3))
    torch.testing.assert_close(term.vel_command_b[3], torch.zeros(3))  # the standing env


def test_uniform_resample_and_update_are_sync_free(cuda_env, count_cuda_syncs):
    """Neither per-step hook blocks on the GPU."""
    options = UniformVelocityCommand.configure(
        entity_name="robot",
        heading_command=True,
        heading_range=(-3.0, 3.0),
        rel_heading_envs=0.5,
        rel_standing_envs=0.5,
        **RANGES,
    )
    term = UniformVelocityCommand(cuda_env, options)
    term.build()
    mask = torch.tensor([True, False, True, False], device="cuda")
    assert count_cuda_syncs(lambda: term._resample_command(mask)) == 0
    assert count_cuda_syncs(term._update_command) == 0
