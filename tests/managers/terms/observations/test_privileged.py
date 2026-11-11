"""The privileged domain-randomization observations in ``eden/managers/terms/observations/privileged.py``.

One Franka hand serves every term: a file morph (Genesis rejects COM writes on a primitive),
``batch_links_info`` / ``batch_dofs_info`` for per-env inertia and gains, and an explicit PD controller whose
modifier chain carries the ``MotorStrength`` that ``MotorStrengthScale`` reads.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.modifiers.actions.actuators import Compose, MotorStrength, TorqueOffset
from eden.managers.terms.actions.joint_actions import ExplicitPDController
from eden.managers.terms.events.domain_rand import RandomizeKpKdGains, RandomizeMotorStrength
from eden.managers.terms.observations.privileged import (
    AddedMass,
    ComOffset,
    FootFriction,
    GainScale,
    MotorStrengthScale,
)
from eden.options import EnvOptions, SceneOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.options.managers.events import EventManagerOptions
from eden.options.robots import FrankaHand
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 4


def _config(**event_terms) -> EdenConfig:
    return EdenConfig(
        env_options=EnvOptions(num_envs=NUM_ENVS, sim_substeps=1, batch_links_info=True, batch_dofs_info=True),
        scene_options=SceneOptions(robot=FrankaHand(default_root_pos=(0.0, 0.0, 0.5))),
        action_options=ActionManagerOptions(
            dofs_pos_controller=ExplicitPDController.configure(
                entity_name="robot",
                dofs_name=FrankaHand().dofs_name,
                modifier=Compose.configure(
                    modifiers=[MotorStrength.configure(), TorqueOffset.configure(torque_offset=0.0)]
                ),
            )
        ),
        event_options=EventManagerOptions(**event_terms),
    )


@pytest.fixture(scope="module")
def env():
    return build_env(_config(), reset=True)


def _make(env, cls, **params):
    term = cls(env, options=cls.configure(**params))
    term.build()
    return term


@pytest.fixture
def robot(env):
    """The hand, with link DR shifts and PD gains restored to nominal after the test."""
    robot = env.entities["robot"]
    yield robot
    n_links = len(robot.links)
    ls = torch.arange(n_links, device=robot.device)
    robot.set_mass_shift(torch.zeros(NUM_ENVS, n_links, device=robot.device), ls_idx_local=ls)
    robot.set_COM_shift(torch.zeros(NUM_ENVS, n_links, 3, device=robot.device), ls_idx_local=ls)
    robot.set_friction_ratio(torch.ones(NUM_ENVS, n_links, device=robot.device), ls_idx_local=ls)
    robot.set_dofs_kp(robot.default_dofs_kp.expand(NUM_ENVS, -1), dofs_idx_local=robot.dofs_idx_local)
    robot.set_dofs_kd(robot.default_dofs_kd.expand(NUM_ENVS, -1), dofs_idx_local=robot.dofs_idx_local)


# ── link DR ──────────────────────────────────────────────────────────────────


def test_added_mass_reads_the_per_env_mass_shift(env, robot):
    ls = torch.tensor([robot.get_link("hand").idx_local], device=robot.device)
    robot.set_mass_shift(torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=robot.device), ls_idx_local=ls)
    out = _make(env, AddedMass, links_name=["hand"]).compute()
    torch.testing.assert_close(out.cpu(), torch.tensor([[1.0], [2.0], [3.0], [4.0]]))


def test_foot_friction_is_the_mean_ratio_over_the_named_links(env, robot):
    pads = ["left_gripper_pad", "right_gripper_pad"]
    ls = torch.tensor([robot.get_link(name).idx_local for name in pads], device=robot.device)
    robot.set_friction_ratio(torch.tensor([[0.6, 0.8]], device=robot.device).expand(NUM_ENVS, 2), ls_idx_local=ls)
    out = _make(env, FootFriction, links_name=pads).compute()
    torch.testing.assert_close(out.cpu(), torch.full((NUM_ENVS, 1), 0.7))


def test_com_offset_flattens_links_and_xyz(env, robot):
    ls = torch.tensor([robot.get_link("hand").idx_local], device=robot.device)
    val = torch.tensor([[[0.01, 0.02, 0.03]]], device=robot.device).expand(NUM_ENVS, 1, 3).contiguous()
    robot.set_COM_shift(val, ls_idx_local=ls)
    out = _make(env, ComOffset, links_name=["hand"]).compute()
    torch.testing.assert_close(out.cpu(), torch.tensor([[0.01, 0.02, 0.03]]).expand(NUM_ENVS, 3))


def test_link_dr_term_rejects_an_unknown_link_pattern(env):
    with pytest.raises((ValueError, AssertionError)):
        _make(env, AddedMass, links_name=["nonexistent"])


# ── actuator DR ──────────────────────────────────────────────────────────────


def test_gain_scale_reports_the_relative_gain_deviation(env, robot):
    """``concat([kp/kp0 - 1, kd/kd0 - 1])``: zero at nominal, the per-env factor minus one once written."""
    term = _make(env, GainScale)
    ndof = len(robot.dofs_name)
    torch.testing.assert_close(term.compute(), torch.zeros(NUM_ENVS, 2 * ndof, device=robot.device), atol=1e-5, rtol=0)

    factor = torch.tensor([[0.8], [0.9], [1.1], [1.2]], device=robot.device)
    robot.set_dofs_kp(robot.default_dofs_kp.expand(NUM_ENVS, -1) * factor, dofs_idx_local=robot.dofs_idx_local)
    robot.set_dofs_kd(robot.default_dofs_kd.expand(NUM_ENVS, -1) * factor, dofs_idx_local=robot.dofs_idx_local)
    torch.testing.assert_close(term.compute(), (factor - 1.0).expand(NUM_ENVS, 2 * ndof), atol=1e-4, rtol=0)


def test_gain_scale_reads_a_zero_baseline_dof_as_nominal_and_warns(env, robot, eden_caplog):
    """A zero nominal kp would make ``kp/kp0`` infinite; the term reports 0 and says why at build."""
    default_kp = robot.default_dofs_kp.clone()
    robot.default_dofs_kp[..., 0] = 0.0
    try:
        term = _make(env, GainScale)
    finally:
        robot.default_dofs_kp.copy_(default_kp)
    # The live gain on DOF 0 is still the non-zero default.
    out = term.compute()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 0].cpu(), torch.zeros(NUM_ENVS))
    assert any("zero nominal" in record.message for record in eden_caplog.records)


def test_gain_scale_rejects_a_gain_event_that_writes_the_controller():
    """With ``action_term_name`` set the event writes the controller caches, so GainScale would read zeros."""
    env = build_env(
        _config(bad_gains=RandomizeKpKdGains.configure(entity_name="robot", action_term_name="dofs_pos_controller"))
    )
    with pytest.raises(ValueError, match="action_term_name"):
        _make(env, GainScale)


def test_motor_strength_scale_reads_the_modifier_deviation(env):
    term = _make(env, MotorStrengthScale, action_term_name="dofs_pos_controller")
    ndof = len(env.entities["robot"].dofs_name)
    torch.testing.assert_close(term.compute().cpu(), torch.zeros(NUM_ENVS, ndof), atol=1e-6, rtol=0)

    event = _make(
        env, RandomizeMotorStrength, action_term_name="dofs_pos_controller", motor_strength_range=(1.15, 1.15)
    )
    event.reset()
    try:
        torch.testing.assert_close(term.compute().cpu(), torch.full((NUM_ENVS, ndof), 0.15), atol=1e-5, rtol=0)
    finally:
        env.action_manager.reset()


def test_motor_strength_scale_rejects_an_unknown_action_term(env):
    with pytest.raises((ValueError, KeyError)):
        _make(env, MotorStrengthScale, action_term_name="does_not_exist")
