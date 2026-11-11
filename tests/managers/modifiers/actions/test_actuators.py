"""Tests for the actuator action modifiers: per-DOF motor specs read off a built G1, ``Compose`` inside a
real ``ExplicitPDController``, ``ActionDelay`` on the physics-step clock, and each modifier's
math driven directly with tensors.

One G1 scene (``sim_dt = 0.005``): the left-leg subset carries the full modifier stack, the right-leg
subset a 30 ms delay. The directly-driven modifiers read only ``sim_dt`` from it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

import eden as en
from eden.managers.modifiers.actions.actuators import (
    ActionDelay,
    Compose,
    ConstantTorqueKick,
    Deadband,
    EffortClip,
    EnvelopeClip,
    FrictionModel,
    GearBacklash,
    MotorStrength,
    TorqueOffset,
)
from eden.managers.modifiers.base import ACTION_MODIFIER_REGISTRY, ActionModifier
from eden.managers.terms.actions.joint_actions import ExplicitPDController, ImplicitPDController
from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.actuators import Unitree5020_16_DOUBLE, Unitree7520_14, Unitree7520_22
from eden.options.managers.modifiers import ActionModifierOptions
from eden.options.robots import FrankaHand
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.configs import EdenConfig, EdenRLConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 4
DECIMATION = 4
# One DOF per motor type; the ankle runs a parallel 4-bar linkage, hence the doubled spec.
LEFT = ["left_knee_joint", "left_hip_pitch_joint", "left_ankle_pitch_joint"]
RIGHT = ["right_knee_joint", "right_hip_pitch_joint", "right_ankle_pitch_joint"]
SPECS = [Unitree7520_22, Unitree7520_14, Unitree5020_16_DOUBLE]

#: Each spec-reading modifier's resolved per-DOF parameters, and the actuator-spec constant each comes from.
SPEC_FIELDS = {
    FrictionModel: {
        "_static_friction": "STATIC_FRICTION",
        "_dynamic_friction": "DYNAMIC_FRICTION",
        "_friction_activation_vel": "FRICTION_ACTIVATION_SPEED",
    },
    EffortClip: {
        "_driving_torque_limit": "DRIVING_TORQUE_LIMIT",
        "_braking_torque_limit": "BRAKING_TORQUE_LIMIT",
        "_full_torque_speed": "FULL_TORQUE_SPEED",
        "_no_load_speed": "NO_LOAD_SPEED",
    },
    EnvelopeClip: {
        "_motor_y1": "DRIVING_TORQUE_LIMIT",
        "_motor_y2": "BRAKING_TORQUE_LIMIT",
        "_motor_x1": "FULL_TORQUE_SPEED",
        "_motor_x2": "NO_LOAD_SPEED",
    },
}


@ACTION_MODIFIER_REGISTRY.register()
class RecordsAppliedTarget(ActionModifier):
    """Passes the applied target through and keeps a copy of each: what the actuator acted on, per physics step."""

    def build(self, num_envs, device, entity=None, dofs_idx_local=None) -> None:
        self.applied: list[torch.Tensor] = []

    def modify_applied_target(self, target: torch.Tensor) -> torch.Tensor:
        self.applied.append(target.clone())
        return target


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS, sim_dt=0.005, decimation=DECIMATION, batch_dofs_info=True),
            scene_options=en.scenes.Empty(robot=G1Dofs29()),
            action_options=ActionManagerOptions(
                stack=ExplicitPDController.configure(
                    entity_name="robot",
                    dofs_name=list(LEFT),
                    modifier=Compose.configure(
                        # Listed in run order: EnvelopeClip per control step, the delay per physics step,
                        # then the torque modifiers (Compose warns otherwise).
                        modifiers=[
                            EnvelopeClip.configure(),
                            ActionDelay.configure(delay_range_steps=(1, 2)),
                            FrictionModel.configure(),
                            EffortClip.configure(),
                            MotorStrength.configure(motor_strength=0.85),
                        ]
                    ),
                ),
                delayed=ExplicitPDController.configure(
                    entity_name="robot",
                    dofs_name=list(RIGHT),
                    # 30 ms = 6 physics steps = 1.5 control steps.
                    modifier=Compose.configure(
                        modifiers=[ActionDelay.configure(delay_range_s=(0.03, 0.03)), RecordsAppliedTarget.configure()]
                    ),
                ),
            ),
        ),
        reset=True,
    )


def _left_idx(env) -> torch.Tensor:
    robot = env.entities["robot"]
    return robot.find_named_dofs(LEFT).dofs_idx_local


def _standalone(env, options) -> ActionModifier:
    """A modifier built on the left-leg DOFs, the way its owning controller builds it."""
    modifier = ACTION_MODIFIER_REGISTRY.get(options.name)(env, options)
    modifier.build(
        num_envs=env.num_envs, device=str(env.device), entity=env.entities["robot"], dofs_idx_local=_left_idx(env)
    )
    return modifier


# ---------------------------------------------------------------------------
# Motor specs read off the entity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["standalone", "in-a-controller-stack"])
@pytest.mark.parametrize("modifier_cls", list(SPEC_FIELDS), ids=lambda cls: cls.__name__)
def test_a_modifier_with_no_explicit_values_reads_each_dofs_motor_spec(env, modifier_cls, where):
    """Row ``i`` must carry DOF ``i``'s motor: three DOFs, three different motors."""
    if where == "standalone":
        modifier = _standalone(env, modifier_cls.configure())
    else:
        modifier = env.action_manager.get_term("stack")._modifier.get(modifier_cls)

    for field, spec_attr in SPEC_FIELDS[modifier_cls].items():
        expected = torch.tensor([[getattr(spec, spec_attr) for spec in SPECS]])
        torch.testing.assert_close(getattr(modifier, field), expected, rtol=1e-5, atol=0.0, msg=field)


@pytest.mark.parametrize(
    "modifier_cls, overrides",
    [
        (
            FrictionModel,
            dict(
                static_friction=[1.0, 2.0, 3.0],
                dynamic_friction=[0.1, 0.2, 0.3],
                friction_activation_vel=[4.0, 5.0, 6.0],
            ),
        ),
        (
            EffortClip,
            dict(
                driving_torque_limit=[10.0, 20.0, 30.0],
                braking_torque_limit=[40.0, 50.0, 60.0],
                full_torque_speed=[1.0, 2.0, 3.0],
                no_load_speed=[4.0, 5.0, 6.0],
            ),
        ),
        (
            EnvelopeClip,
            dict(
                driving_torque_limit=[10.0, 20.0, 30.0],
                braking_torque_limit=[40.0, 50.0, 60.0],
                full_torque_speed=[1.0, 2.0, 3.0],
                no_load_speed=[4.0, 5.0, 6.0],
            ),
        ),
    ],
    ids=["FrictionModel", "EffortClip", "EnvelopeClip"],
)
def test_explicit_per_dof_values_override_the_spec(env, modifier_cls, overrides):
    modifier = _standalone(env, modifier_cls.configure(**overrides))

    for (field, _), value in zip(SPEC_FIELDS[modifier_cls].items(), overrides.values(), strict=True):
        torch.testing.assert_close(getattr(modifier, field), torch.tensor([value]), msg=field)


def test_a_scalar_override_broadcasts_and_the_fields_left_unset_still_come_from_the_spec(env):
    friction = _standalone(
        env, FrictionModel.configure(static_friction=99.0, dynamic_friction=88.0, friction_offset=[0.01, 0.02, 0.03])
    )

    torch.testing.assert_close(friction._static_friction, torch.full((1, 3), 99.0))
    torch.testing.assert_close(friction._dynamic_friction, torch.full((1, 3), 88.0))
    torch.testing.assert_close(friction._friction_offset_tensor, torch.tensor([[0.01, 0.02, 0.03]]))
    torch.testing.assert_close(
        friction._friction_activation_vel,
        torch.tensor([[spec.FRICTION_ACTIVATION_SPEED for spec in SPECS]]),
        rtol=1e-5,
        atol=0.0,
    )


def test_an_envelope_clip_without_an_entity_is_refused(env):
    with pytest.raises(ValueError, match="EnvelopeClip requires entity"):
        EnvelopeClip(env, EnvelopeClip.configure()).build(
            num_envs=NUM_ENVS, device="cpu", entity=None, dofs_idx_local=None
        )


def test_an_envelope_clip_reset_rereads_the_gains_domain_randomization_wrote(env):
    """Gains cached at build go stale when DR writes new ones; ``reset`` re-reads the given envs' rows.

    The cached gains are read directly: the clip they feed is pinned with the action-term tests.
    """
    robot, idx = env.entities["robot"], _left_idx(env)
    original_kp, original_kd = (
        robot.get_dofs_kp(dofs_idx_local=idx).clone(),
        robot.get_dofs_kd(dofs_idx_local=idx).clone(),
    )
    clip = _standalone(env, EnvelopeClip.configure())
    try:
        robot.set_dofs_kp(torch.full((NUM_ENVS, 3), 200.0), dofs_idx_local=idx)
        robot.set_dofs_kd(torch.full((NUM_ENVS, 3), 20.0), dofs_idx_local=idx)
        torch.testing.assert_close(clip._kp, original_kp)  # stale until reset

        clip.reset(envs_idx=torch.tensor([1]))
        assert clip._kp[:, 0].tolist() == pytest.approx(
            [original_kp[0, 0].item(), 200.0] + [original_kp[2, 0].item()] * 2
        )

        clip.reset()
        torch.testing.assert_close(clip._kp, torch.full((NUM_ENVS, 3), 200.0))
        torch.testing.assert_close(clip._kd, torch.full((NUM_ENVS, 3), 20.0))
    finally:
        robot.set_dofs_kp(original_kp, dofs_idx_local=idx)
        robot.set_dofs_kd(original_kd, dofs_idx_local=idx)


@pytest.fixture(scope="module")
def unbatched_env():
    """A Franka hand with ``batch_dofs_info=False``, the default of ten shipped tasks."""
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=3, batch_dofs_info=False),
            scene_options=SceneOptions(robot=FrankaHand(default_root_pos=(0.0, 0.0, 0.5))),
        )
    )


def test_an_envelope_clip_resets_with_one_gain_row_per_env_when_dof_info_is_unbatched(unbatched_env):
    """``reset()`` re-reads the gains with ``envs_idx``, which raised on the first episode reset before the
    getters normalized their shape.
    """
    robot = unbatched_env.entities["robot"]
    clip = EnvelopeClip(unbatched_env, EnvelopeClip.configure())
    clip.build(num_envs=3, device=str(unbatched_env.device), entity=robot, dofs_idx_local=robot.dofs_idx_local)

    clip.reset()
    clip.reset(envs_idx=torch.tensor([0, 2], device=robot.device))

    # Private, but the gain cache is the modifier's only state; it must be (num_envs, n_dofs) to be assigned into.
    assert clip._kp.shape == (3, robot.num_dofs)
    assert torch.allclose(clip._kp, robot.get_dofs_kp(dofs_idx_local=robot.dofs_idx_local))


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def test_compose_get_finds_the_first_match_and_recurses_into_a_nested_compose(env):
    outer = Compose(
        env,
        Compose.configure(modifiers=[EffortClip.configure(), Compose.configure(modifiers=[MotorStrength.configure()])]),
    )

    assert isinstance(outer.get(EffortClip), EffortClip)
    assert isinstance(outer.get(MotorStrength), MotorStrength)  # the only one sits in the nested Compose
    assert outer.get(EnvelopeClip) is None


def test_the_full_stack_stays_finite_through_extreme_actions_steps_and_resets(env):
    """``ActionDelay`` and ``EnvelopeClip`` carry per-env buffers: a half-cleared reset shows up as NaN.

    A modifier that silently emits NaN, not one that raises, is the failure that reaches training.
    """
    manager = env.action_manager
    stack = manager.get_term("stack")
    generator = torch.Generator().manual_seed(0)

    def step(actions):
        env.step(actions)
        assert stack.processed_action.isfinite().all()

    env.reset()
    step(torch.full((NUM_ENVS, 6), 100.0))  # 100x out of range
    for _ in range(5):
        step(torch.randn(NUM_ENVS, 6, generator=generator) * 0.1)
    manager.reset(envs_idx=torch.tensor([0, 2]))
    step(torch.zeros(NUM_ENVS, 6))
    env.reset()
    for _ in range(3):
        step(torch.randn(NUM_ENVS, 6, generator=generator) * 0.1)
    env.reset()


# ---------------------------------------------------------------------------
# ActionDelay: the physics-step clock
# ---------------------------------------------------------------------------


def test_through_env_step_a_step_input_reaches_the_actuator_six_physics_steps_later(env):
    """Mid-way through the next control step. Counted per control step, the same config held the old
    target for 6 control steps (24 physics steps, 120 ms).
    """
    term = env.action_manager.get_term("delayed")
    recorder = term._modifier.get(RecordsAppliedTarget)
    env.reset()
    recorder.applied.clear()
    held = torch.zeros(NUM_ENVS, 6)
    held[:, 3:] = 0.1
    fresh = torch.zeros(NUM_ENVS, 6)
    fresh[:, 3:] = -0.4

    step_input_at = 2
    for control_step in range(5):
        env.step(held if control_step < step_input_at else fresh)
        if control_step == 0:
            held_target = term.processed_action.clone()
    fresh_target = term.processed_action.clone()

    assert len(recorder.applied) == 5 * DECIMATION  # one applied target per physics step
    arrival = step_input_at * DECIMATION + 6
    for i, target in enumerate(recorder.applied):
        torch.testing.assert_close(target, fresh_target if i >= arrival else held_target, msg=f"physics step {i}")

    # reset() clears the history: the next command reaches the actuator with no stale target.
    env.reset()
    env.step(held)
    torch.testing.assert_close(recorder.applied[-1], held_target)
    env.reset()


# ---------------------------------------------------------------------------
# Each modifier's math, driven directly with tensors
# ---------------------------------------------------------------------------


def test_deadband_zeros_torque_inside_deadband(env):
    modifier = Deadband(env, Deadband.configure(deadband_epsilon=0.1))
    # dofs_idx_local has length matching the per-dof tensor size
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    ctrl_torque = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dofs_vel = torch.zeros(2, 2)
    pos_err = torch.tensor([[0.05, 0.2], [0.0, -0.15]])

    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), dofs_vel, pos_err=pos_err)

    # Inside deadband (|pos_err| < 0.1) → zeroed; outside → unchanged
    assert torch.equal(out, torch.tensor([[0.0, 2.0], [0.0, 4.0]]))


def test_deadband_noop_without_pos_err(env):
    modifier = Deadband(env, Deadband.configure(deadband_epsilon=0.5))
    modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    ctrl_torque = torch.tensor([[1.0, -2.0]])
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(1, 2), pos_err=None)
    assert torch.equal(out, ctrl_torque)


def test_deadband_per_env_epsilon(env):
    """Simulates domain randomization by writing per-env epsilons directly."""
    modifier = Deadband(env, Deadband.configure(deadband_epsilon=0.0))
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    modifier._deadband_epsilon[:] = torch.tensor([[0.05, 0.2], [0.5, 0.05]])

    ctrl_torque = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    pos_err = torch.tensor([[0.1, 0.1], [0.1, 0.1]])  # |err|=0.1

    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(2, 2), pos_err=pos_err)

    # env0 dof0: 0.1>=0.05 → keep; dof1: 0.1<0.2 → zero
    # env1 dof0: 0.1<0.5  → zero;  dof1: 0.1>=0.05 → keep
    assert torch.equal(out, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_deadband_per_dof_epsilon_config(env):
    """Per-DOF ``deadband_epsilon`` list must match controlled DOFs and applies per joint."""
    modifier = Deadband(
        env,
        Deadband.configure(deadband_epsilon=[0.05, 0.2]),
    )
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    assert torch.equal(modifier._deadband_epsilon_row, torch.tensor([0.05, 0.2]))
    assert torch.equal(
        modifier._deadband_epsilon,
        torch.tensor([[0.05, 0.2], [0.05, 0.2]]),
    )

    ctrl_torque = torch.ones(2, 2)
    pos_err = torch.full((2, 2), 0.1)
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(2, 2), pos_err=pos_err)

    # Same thresholds as test_deadband_per_env_epsilon env 0; both env rows identical
    assert torch.equal(out, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_gear_backlash_offsets_target_by_last_motion_direction(env):
    modifier = GearBacklash(env, GearBacklash.configure(backlash=0.1))
    modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    first = modifier.modify_processed_action(torch.tensor([[0.0, 0.0]]))
    positive = modifier.modify_processed_action(torch.tensor([[1.0, 2.0]]))
    hold = modifier.modify_processed_action(torch.tensor([[1.0, 2.0]]))
    negative = modifier.modify_processed_action(torch.tensor([[-1.0, -2.0]]))

    assert torch.allclose(first, torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(positive, torch.tensor([[1.1, 2.1]]))
    assert torch.allclose(hold, torch.tensor([[1.1, 2.1]]))
    assert torch.allclose(negative, torch.tensor([[-1.1, -2.1]]))


def test_gear_backlash_reversal_threshold_ignores_small_command_noise(env):
    modifier = GearBacklash(env, GearBacklash.configure(backlash=0.1, reversal_threshold=0.05))
    modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0]))

    modifier.modify_processed_action(torch.tensor([[0.0]]))
    positive = modifier.modify_processed_action(torch.tensor([[1.0]]))
    small_negative = modifier.modify_processed_action(torch.tensor([[0.98]]))

    assert torch.allclose(positive, torch.tensor([[1.1]]))
    assert torch.allclose(small_negative, torch.tensor([[1.08]]))


def test_gear_backlash_takeup_rate_smooths_side_changes(env):
    modifier = GearBacklash(env, GearBacklash.configure(backlash=0.2, takeup_rate=0.5))
    modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0]))

    first = modifier.modify_processed_action(torch.tensor([[0.0]]))
    positive_1 = modifier.modify_processed_action(torch.tensor([[1.0]]))
    positive_2 = modifier.modify_processed_action(torch.tensor([[2.0]]))
    negative = modifier.modify_processed_action(torch.tensor([[-1.0]]))

    assert torch.allclose(first, torch.tensor([[0.0]]))
    assert torch.allclose(positive_1, torch.tensor([[1.1]]))
    assert torch.allclose(positive_2, torch.tensor([[2.15]]))
    assert torch.allclose(negative, torch.tensor([[-1.025]]))


def test_gear_backlash_reset_restores_initial_side(env):
    modifier = GearBacklash(
        env,
        GearBacklash.configure(backlash=[0.1, 0.2], initial_side=[1.0, -1.0]),
    )
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    out = modifier.modify_processed_action(torch.zeros(2, 2))
    assert torch.allclose(out, torch.tensor([[0.1, -0.2], [0.1, -0.2]]))

    modifier.modify_processed_action(torch.ones(2, 2))
    modifier.reset(envs_idx=torch.tensor([0]))
    reset_out = modifier.modify_processed_action(torch.zeros(2, 2))

    assert torch.allclose(reset_out[0], torch.tensor([0.1, -0.2]))
    assert torch.allclose(reset_out[1], torch.tensor([-0.1, -0.2]))


def test_gear_backlash_rejects_negative_backlash(env):
    modifier = GearBacklash(env, GearBacklash.configure(backlash=[0.1, -0.2]))
    with pytest.raises(ValueError, match="nonnegative"):
        modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0, 1]))


def test_motor_strength_per_dof_config(env):
    modifier = MotorStrength(
        env,
        MotorStrength.configure(motor_strength=[2.0, 0.5]),
    )
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    assert torch.equal(modifier._motor_strength_row, torch.tensor([2.0, 0.5]))

    ctrl_torque = torch.ones(2, 2)
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(2, 2), pos_err=None)
    assert torch.equal(out, torch.tensor([[2.0, 0.5], [2.0, 0.5]]))


def test_torque_offset_per_dof_config(env):
    modifier = TorqueOffset(
        env,
        TorqueOffset.configure(torque_offset=[0.25, -0.5]),
    )
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    assert torch.equal(modifier._torque_offset_row, torch.tensor([0.25, -0.5]))

    ctrl_torque = torch.zeros(2, 2)
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(2, 2), pos_err=None)
    assert torch.equal(out, torch.tensor([[0.25, -0.5], [0.25, -0.5]]))


def test_constant_torque_kick_adds_flat_torque_in_pos_err_direction(env):
    modifier = ConstantTorqueKick(env, ConstantTorqueKick.configure(torque_kick=[0.25, 0.5]))
    modifier.build(num_envs=2, device="cpu", dofs_idx_local=torch.tensor([0, 1]))

    ctrl_torque = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pos_err = torch.tensor([[0.1, -0.2], [-0.3, 0.4]])
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(2, 2), pos_err=pos_err)

    assert torch.allclose(out, torch.tensor([[1.25, 1.5], [2.75, 4.5]]))


def test_constant_torque_kick_respects_activation_epsilon(env):
    modifier = ConstantTorqueKick(env, ConstantTorqueKick.configure(torque_kick=0.25, activation_epsilon=0.1))
    modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0, 1, 2]))

    ctrl_torque = torch.tensor([[1.0, 2.0, 3.0]])
    pos_err = torch.tensor([[0.05, -0.1, -0.2]])
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(1, 3), pos_err=pos_err)

    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 2.75]]))


def test_constant_torque_kick_can_follow_ctrl_torque_sign(env):
    modifier = ConstantTorqueKick(env, ConstantTorqueKick.configure(torque_kick=0.2, direction_source="ctrl_torque"))
    modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0, 1, 2]))

    ctrl_torque = torch.tensor([[1.0, -2.0, 0.0]])
    pos_err = torch.tensor([[-1.0, 1.0, 1.0]])
    out = modifier.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(1, 3), pos_err=pos_err)

    assert torch.allclose(out, torch.tensor([[1.2, -2.2, 0.0]]))


def test_constant_torque_kick_rejects_negative_magnitude(env):
    modifier = ConstantTorqueKick(env, ConstantTorqueKick.configure(torque_kick=[0.1, -0.2]))
    with pytest.raises(ValueError, match="nonnegative"):
        modifier.build(num_envs=1, device="cpu", dofs_idx_local=torch.tensor([0, 1]))


def test_compose_forwards_pos_err(env):
    """Compose must forward pos_err so nested Deadband works inside a chain."""
    chain = Compose(env, Compose.configure(modifiers=[Deadband.configure(deadband_epsilon=0.1)]))
    chain.build(num_envs=1, device="cpu", entity=env.entities["robot"], dofs_idx_local=torch.tensor([0, 1]))

    ctrl_torque = torch.tensor([[5.0, 5.0]])
    pos_err = torch.tensor([[0.05, 0.2]])  # first dof inside deadband

    out = chain.modify_ctrl_torque(ctrl_torque.clone(), torch.zeros(1, 2), pos_err=pos_err)
    assert torch.equal(out, torch.tensor([[0.0, 5.0]]))


# --- ActionDelay: the delay counts physics steps and acts inside apply_actions ---


def _built_delay(env, num_envs=1, **delay):
    modifier = ActionDelay(env, ActionDelay.configure(**delay))
    modifier.build(num_envs=num_envs, device="cpu")
    return modifier


def test_action_delay_inside_compose_reaches_the_physics_step_hook(env):
    """``Compose`` forwards the per-physics-step hook, so a nested delay still delays."""
    chain = Compose(env, Compose.configure(modifiers=[ActionDelay.configure(delay_range_steps=(2, 2))]))
    chain.build(num_envs=1, device="cpu")
    outputs = [chain.modify_applied_target(torch.tensor([[value]])).item() for value in (1.0, 2.0, 3.0, 4.0)]
    assert outputs == [1.0, 1.0, 1.0, 2.0]  # backfilled with the first command, then 2 steps behind


def test_action_delay_seconds_resolve_to_physics_steps(env):
    """``delay_range_s`` is converted on the physics-step clock (``sim_dt = 0.005`` here), not ``env.dt``."""
    assert _built_delay(env, delay_range_s=(0.005, 0.03)).resolved_range_steps == (1, 6)
    # The step form gives the same range, so the two units are interchangeable.
    assert _built_delay(env, delay_range_steps=(1, 6)).resolved_range_steps == (1, 6)
    # Any integral type counts steps, not only a Python int.
    assert _built_delay(env, delay_range_steps=(np.int64(1), np.int64(6))).resolved_range_steps == (1, 6)
    # A zero range, or no range at all, is a pass-through that keeps no history.
    command = torch.ones(1, 2)
    for passthrough in (_built_delay(env, delay_range_s=(0.0, 0.0)), _built_delay(env)):
        assert passthrough.delay_steps is None and passthrough.modify_applied_target(command) is command


@pytest.mark.parametrize(
    ("delay", "match"),
    [
        ({"delay_range_s": (0.0, 0.012)}, r"whole multiples of the physics step sim_dt=0\.005.*2\.4 physics steps"),
        ({"delay_range_s": (0.0, 0.01), "delay_range_steps": (0, 2)}, "not both"),
        ({"delay_range_s": (0.02, 0.01)}, r"0 <= min <= max, got \(4, 2\)"),
        ({"delay_range_s": (0.0, float("nan"))}, "must be finite"),
        ({"delay_range_steps": (3, 1)}, r"0 <= min <= max, got \(3, 1\)"),
        ({"delay_range_steps": (-1, 1)}, r"0 <= min <= max"),
        ({"delay_range_steps": (1, 2, 3)}, r"delay_range_steps must be \(min, max\)"),
    ],
)
def test_action_delay_rejects_a_bad_range_at_build(env, delay, match):
    with pytest.raises(ValueError, match=match):
        _built_delay(env, **delay)


def test_action_delay_rejects_a_fractional_step_count_at_build(env):
    """A step count is a whole number; a fraction of a physics step belongs in ``delay_range_s``."""
    with pytest.raises(TypeError, match="delay_range_steps must hold whole physics-step counts.*delay_range_s"):
        _built_delay(env, delay_range_steps=(0, 2.5))


_RETIRED_STEP_FIELDS = r"Retired \['max_delay', 'min_delay'\]: ActionDelay used to count these in control steps.*delay_range_steps.*env_options\.decimation"


def test_action_delay_retired_step_fields_fail_with_migration_directions():
    """``min_delay`` / ``max_delay`` used to count control steps, so they are refused, never reinterpreted.

    Kept under the same name, an old ``min_delay=1`` would now build as one *physics* step — a delay
    ``decimation`` times shorter than the config was written for, with no error. The error has to say
    how to convert, or the obvious fix, ``delay_range_steps=(1, 1)``, makes the same mistake.
    """
    with pytest.raises(ValueError, match=_RETIRED_STEP_FIELDS):
        ActionDelay.configure(min_delay=1, max_delay=1)
    # A saved config never goes through configure(); its options are validated as they load.
    with pytest.raises(ValueError, match=_RETIRED_STEP_FIELDS):
        ActionModifierOptions(name="action_delay", min_delay=1, max_delay=1)


def test_a_config_saved_with_the_retired_step_fields_fails_to_load(tmp_path):
    """The whole saved-config path: a config file holding the retired step fields is refused at load.

    The same file written with ``delay_range_steps`` round-trips, so the refusal is about the
    retired names and not about modifiers in saved configs.
    """
    config = EdenConfig(
        action_options=ActionManagerOptions(
            legs=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=["joint_0"],
                modifier=ActionDelay.configure(delay_range_steps=(4, 4)),
            )
        )
    )
    path = tmp_path / "config.yaml"
    config.save_as_yaml(str(path), lock_config=False)
    reloaded = EdenConfig.load_from_file(str(path))
    assert tuple(reloaded.action_options.legs.modifier.delay_range_steps) == (4, 4)

    saved = yaml.safe_load(path.read_text())
    modifier = saved["config"]["action_options"]["legs"]["modifier"]
    del modifier["delay_range_steps"]
    modifier.update(min_delay=1, max_delay=1)  # how the parkour student wrote its 20 ms under the old fields
    path.write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match=_RETIRED_STEP_FIELDS):
        EdenConfig.load_from_file(str(path))


@pytest.mark.parametrize(
    ("modifiers", "warns"),
    [
        # A control-step modifier listed after the delay runs before it anyway: warn.
        ([ActionDelay.configure(delay_range_steps=(1, 1)), GearBacklash.configure(backlash=0.1)], True),
        # The same, with the delay one Compose deeper.
        (
            [
                Compose.configure(modifiers=[ActionDelay.configure(delay_range_steps=(1, 1))]),
                GearBacklash.configure(backlash=0.1),
            ],
            True,
        ),
        # Listed in the order they run: silent.
        ([GearBacklash.configure(backlash=0.1), ActionDelay.configure(delay_range_steps=(1, 1))], False),
        # A torque modifier after the delay runs after it too: silent.
        ([ActionDelay.configure(delay_range_steps=(1, 1)), MotorStrength.configure(motor_strength=0.9)], False),
    ],
)
def test_compose_warns_when_a_delay_is_listed_before_a_control_step_modifier(env, modifiers, warns, eden_caplog):
    Compose(env, Compose.configure(modifiers=modifiers))
    warnings = [r.getMessage() for r in eden_caplog.records if "runs later" in r.getMessage()]
    if warns:
        assert len(warnings) == 1 and "GearBacklash" in warnings[0]
    else:
        assert warnings == []


def test_action_delay_is_drawn_per_reset_and_held_for_the_episode(env):
    """Each env draws its delay when it resets and keeps it; stepping does not redraw it."""
    num_envs = 64
    torch.manual_seed(0)
    delay = _built_delay(env, num_envs=num_envs, delay_range_steps=(0, 3))
    delay.reset()
    drawn = delay.delay_steps.clone()
    # 64 uniform draws over 4 values miss one with probability ~4 * 0.75**64 ~ 4e-8.
    assert set(drawn.tolist()) == {0, 1, 2, 3}

    for _ in range(10):
        delay.modify_applied_target(torch.randn(num_envs, 2))
    assert torch.equal(delay.delay_steps, drawn)

    # A partial reset by mask (what auto-reset passes) or by index redraws those envs and only those.
    mask = torch.zeros(num_envs, dtype=torch.bool)
    mask[:32] = True
    delay.reset(mask)
    delay.reset(torch.tensor([32, 33]))
    assert torch.equal(delay.delay_steps[34:], drawn[34:])
    # 34 redraws all landing on their old value has probability 4**-34; a reset that skipped the draw always does.
    assert not torch.equal(delay.delay_steps[:34], drawn[:34])
    assert ((delay.delay_steps >= 0) & (delay.delay_steps <= 3)).all()


def test_action_delay_never_replays_a_command_from_before_the_reset(env):
    """A reset env's history is backfilled with its first new command; the others keep lagging."""
    delay = _built_delay(env, num_envs=2, delay_range_steps=(3, 3))
    for value in (1.0, 2.0, 3.0, 4.0):
        delay.modify_applied_target(torch.full((2, 1), value))
    delay.reset(torch.tensor([1]))
    out = delay.modify_applied_target(torch.full((2, 1), 9.0))
    assert out.flatten().tolist() == [2.0, 9.0]


def test_action_delay_reset_and_physics_steps_are_sync_free(env, count_cuda_syncs):
    """Auto-reset hands every modifier a bool mask *every* control step, then the delay runs per physics step.

    Both used to index with a mask (``CircularBuffer.reset`` and its backfill), a GPU sync apiece
    per control step for as long as a delay was configured.
    """
    delay = ActionDelay(env, ActionDelay.configure(delay_range_steps=(1, 3)))
    delay.build(num_envs=8, device="cuda")
    command = torch.randn(8, 2, device="cuda")
    delay.modify_applied_target(command)  # the first append allocates the history
    mask = torch.arange(8, device="cuda") < 3

    def control_step():
        delay.reset(mask)
        for _ in range(DECIMATION):
            delay.modify_applied_target(command)

    assert count_cuda_syncs(control_step) == 0


def test_torque_offset_reset_restores_its_configured_value_in_the_selected_envs(env):
    """The configured value, not a hard-coded zero; a partial reset touches only those envs."""
    modifier = TorqueOffset(env, TorqueOffset.configure(torque_offset=[0.25, -0.5]))
    modifier.build(num_envs=4, device="cpu", dofs_idx_local=torch.tensor([0, 1]))
    modifier._torque_offset[:] = 42.0  # what a randomizer writes
    modifier.reset(envs_idx=torch.tensor([0, 2]))
    torch.testing.assert_close(modifier._torque_offset[[0, 2]], torch.tensor([[0.25, -0.5]] * 2))
    assert (modifier._torque_offset[[1, 3]] == 42.0).all()
    modifier.reset()
    torch.testing.assert_close(modifier._torque_offset, torch.tensor([[0.25, -0.5]] * 4))
