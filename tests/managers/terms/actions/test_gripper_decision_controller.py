"""``GripperDecisionController``: gripper decision rules fused into the env action term.

Each rule is checked against the client-side implementation it replaces (adapters and
setups of the server-backed policies), so the port is behaviour-neutral. Terms are built
against a real WidowX250 env, with Octo's finger targets (open ``±0.037``, closed ``0``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import eden as en
from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.terms.actions.binary_actions import GripperDecisionController
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.options import EnvOptions
from eden.options.managers.actions import ActionManagerOptions, ActionTermOptions
from eden.options.robots.trossen_widowx250 import WidowX250
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

_FINGERS = list(WidowX250.gripper_dofs_name)
_OPEN = {"left_finger": 0.037, "right_finger": -0.037}
_NUM_ENVS = 2


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=_NUM_ENVS),
            scene_options=en.scenes.Empty(robot=WidowX250()),
            action_options=ActionManagerOptions(
                arm_controller=ImplicitPDController.configure(
                    entity_name="robot", dofs_name=list(WidowX250.arm_dofs_name)
                ),
                gripper_controller=GripperDecisionController.configure(
                    entity_name="robot", dofs_name=_FINGERS, open_action=_OPEN, close_action=0.0, input="closedness"
                ),
            ),
        )
    )


def _set_measured_closedness(env, closedness: float) -> None:
    """Teleport both fingers to ``closedness`` of the way from open to closed."""
    robot = env.entities["robot"]
    pos = robot.get_dofs_pos().clone()
    for name, open_pos in _OPEN.items():
        pos[:, robot.dofs_name.index(name)] = open_pos * (1.0 - closedness)
    robot.set_dofs_pos(pos)


@pytest.fixture(autouse=True)
def _open_gripper(env):
    _set_measured_closedness(env, 0.0)


def _term(env, **kw):
    options = ActionTermOptions(entity_name="robot", dofs_name=_FINGERS, open_action=_OPEN, close_action=0.0, **kw)
    term = GripperDecisionController(env, options)
    term.build()
    return term


def _command(term, value):
    term.compute(torch.full((_NUM_ENVS, 1), value, dtype=torch.float32))


def _drive(term, values):
    """Env 0's decided closedness after each command (both envs receive the same stream)."""
    out = []
    for v in values:
        _command(term, v)
        out.append(float(term.last_closedness[0]))
    return out


# ---------------------------------------------------------------------------


def test_registered_and_default_matches_binary_joint_controller(env):
    assert ACTION_TERM_REGISTRY.get("gripper_decision_controller") is GripperDecisionController
    term = _term(env)
    term.compute(torch.tensor([[0.5], [-0.5]]))
    torch.testing.assert_close(term.processed_action, torch.tensor([[0.0, 0.0], [0.037, -0.037]]))
    term.compute(torch.tensor([[True], [False]]))
    torch.testing.assert_close(term.processed_action, torch.tensor([[0.0, 0.0], [0.037, -0.037]]))


def test_open_sign_input_matches_octo_negation(env):
    """Octo / open-pi-zero servers emit +1 = open; the client used to negate."""
    assert _drive(_term(env, input="open_sign"), [1.0, -1.0, 0.0]) == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("value", [0.0, 0.3, 0.5, 0.51, 1.0])
def test_openness_input_closes_unless_strictly_above_the_threshold(env, value):
    """OpenPI Bridge: an openness strictly above 0.5 means open (the rule the former client-side binarize applied)."""
    assert _drive(_term(env, input="openness"), [value]) == [float(not value > 0.5)]


@pytest.mark.parametrize("value", [0.0, 0.49, 0.5, 0.51, 1.0])
def test_closedness_binarize_matches_joint_position_adapter(env, value):
    """DROID-style servers: closedness > threshold closes (JointPositionAdapter rule)."""
    assert _drive(_term(env, input="closedness"), [value]) == [float(value > 0.5)]


def test_hysteresis_and_hold(env):
    """Sequence pinned from the former client-side ``GripperLatch`` (RLDX DROID reference client)."""
    stream = [0.1, 0.55, 0.65, 0.45, 0.35, 0.7, 0.2, 0.3, 0.8, 0.5, 0.39]
    term = _term(env, input="closedness", open_threshold=0.4, close_threshold=0.6, hold_steps=2)
    assert _drive(term, stream) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_equal_thresholds_hysteresis_holds_on_tie(env):
    term = _term(env, input="closedness", open_threshold=0.5, close_threshold=0.5)
    assert _drive(term, [0.6, 0.5, 0.4, 0.5, 0.6]) == [1.0, 1.0, 0.0, 0.0, 1.0]


def _simpler_cogact_google_robot(stream, measured_q=0.0, num_repeat=10):
    """SimplerEnv google_robot gripper, ported line by line, in joint units.

    ``sim_cogact/cogact_policy.py`` turns openness (1 = open) into a relative command, and
    ManiSkill2_real2sim's ``gripper_pd_joint_target_delta_pos_interpolate_by_planner``
    (``PDJointPosController.set_action``: normalize_action to +/-1.31, delta from the last
    target, clip to the URDF range [-0.0001, 1.3] +/- 0.01, repeat the last target when
    |delta| < 1e-3) integrates it. Returns the target closedness (q / 1.3, clamped) per step.
    """
    prev, sticky_on, repeat, sticky_action = None, False, 0, 0.0
    q, out = measured_q, []
    for open_gripper in stream:
        if prev is None:
            relative, prev = 0.0, open_gripper
        else:
            relative = prev - open_gripper
        if abs(relative) > 0.5 and not sticky_on:
            sticky_on, sticky_action, prev = True, relative, open_gripper
        if sticky_on:
            repeat += 1
            relative = sticky_action
        if repeat == num_repeat:
            sticky_on, repeat, sticky_action = False, 0, 0.0
        delta = 1.31 * float(np.clip(relative, -1.0, 1.0))
        if abs(delta) >= 1e-3:
            q = float(np.clip(q + delta, -0.0001 - 0.01, 1.3 + 0.01))
        out.append(min(max(q / 1.3, 0.0), 1.0))
    return out


def test_sticky_latch_matches_simpler_google_robot_delta_controller(env):
    rng = np.random.default_rng(0)
    square = [1.0, 1.0] + [0.0] * 12 + [1.0] * 16 + [0.2] * 3 + [0.9] * 12
    streams = [
        square,
        [1.0] + [0.8] * 8,  # never triggers: the small relative command still closes the gripper
        [1.0, 0.05] + [0.15] * 20,  # latch-close, then hover: reopens once the latch expires
        list(np.clip(np.asarray(square) + rng.normal(0, 0.05, len(square)), 0, 1)),
        list(rng.uniform(0, 1, 80)),
    ]
    for stream in streams:
        got = _drive(_term(env, input="openness", sticky_repeats=10), stream)
        np.testing.assert_allclose(got, _simpler_cogact_google_robot(stream), atol=1e-5)


def test_sticky_latch_starts_from_the_measured_gripper_and_reads_closedness(env):
    # Controller reset sets the target to the current qpos: a half-closed gripper starts half closed.
    _set_measured_closedness(env, 0.5)
    got = _drive(_term(env, input="openness", sticky_repeats=10), [0.5, 0.5])
    np.testing.assert_allclose(got, _simpler_cogact_google_robot([0.5, 0.5], measured_q=0.65), atol=1e-5)
    # The same intent expressed as closedness gives the same trajectory.
    _set_measured_closedness(env, 0.0)
    stream = [1.0, 1.0, 0.0, 0.0, 0.3]
    closed_in = _drive(_term(env, input="closedness", sticky_repeats=10), [1.0 - v for v in stream])
    np.testing.assert_allclose(closed_in, _simpler_cogact_google_robot(stream), atol=1e-5)


def test_zero_holds_keeps_the_previous_sign_decision(env):
    """Robosuite's gripper steps by sign(action): 0 keeps the current target."""
    assert _drive(_term(env, input="sign", zero_holds=True), [1.0, 0.0, -1.0, 0.0]) == [1.0, 1.0, 0.0, 0.0]
    assert _drive(_term(env, input="sign"), [1.0, 0.0]) == [1.0, 0.0]


def test_on_missing_measured_commands_the_measured_closedness_every_missing_step(env):
    _set_measured_closedness(env, 0.3)
    term = _term(env, input="closedness", continuous=True, snap_threshold=0.8, on_missing="measured")
    assert _drive(term, [0.9, float("nan")]) == pytest.approx([1.0, 0.3])
    _set_measured_closedness(env, 0.85)
    assert _drive(term, [float("nan")]) == pytest.approx([1.0])  # snapped like a prediction


def test_nan_command_keeps_the_raw_action_finite(env):
    term = _term(env, input="openness")
    _command(term, 0.2)
    _command(term, float("nan"))
    torch.testing.assert_close(term.action, torch.full((_NUM_ENVS, 1), 0.2))


def test_continuous_and_snap_match_galaxea_snap_rule(env):
    term = _term(env, input="openness", continuous=True, snap_threshold=0.8)
    # openness 0.7 -> closedness 0.3 passes through; openness 0.1 -> closedness 0.9 snaps to 1.0
    assert _drive(term, [0.7, 0.1]) == pytest.approx([0.3, 1.0])


def test_nan_holds_previous_target_and_first_nan_seeds_from_measured_state(env):
    assert _drive(_term(env, input="openness"), [0.1, float("nan"), 0.9, float("nan")]) == [1.0, 1.0, 0.0, 0.0]
    # Never commanded, gripper physically 0.9 of the way closed -> closed.
    _set_measured_closedness(env, 0.9)
    assert _drive(_term(env, input="openness"), [float("nan")]) == [1.0]
    _set_measured_closedness(env, 0.1)
    assert _drive(_term(env, input="openness"), [float("nan")]) == [0.0]


def test_reset_clears_decision_state_per_env(env):
    term = _term(env, input="closedness", open_threshold=0.4, close_threshold=0.6, hold_steps=3)
    _command(term, 0.9)
    term.reset(torch.tensor([1]))
    assert term.last_closedness.tolist() == [1.0, 0.0]
    # A missing command: env 0 holds its decision, env 1 (never commanded since reset) seeds from
    # its measured, open gripper.
    _command(term, float("nan"))
    assert term.last_closedness.tolist() == [1.0, 0.0]
    # 0.5 sits inside the hysteresis band, so each env keeps its own state: env 1 starts over open.
    _command(term, 0.5)
    assert term.last_closedness.tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (dict(input="percent"), "input"),
        (dict(open_threshold=0.5), "both open_threshold and close_threshold"),
        (dict(open_threshold=0.6, close_threshold=0.4), "close_threshold"),
        (dict(hold_steps=-1), "non-negative"),
        (dict(continuous=True), "continuous mode needs"),
        (dict(input="closedness", continuous=True, snap_threshold=1.5), "snap_threshold"),
        (dict(input="closedness", hold_steps=2), "hold_steps only applies"),
        (dict(open_threshold=0.4, close_threshold=0.6), "hysteresis thresholds need"),
        (dict(input="closedness", snap_threshold=0.8), "snap_threshold only applies"),
        (dict(input="closedness", continuous=True, sticky_repeats=5), "continuous mode cannot"),
        (dict(sticky_repeats=5), "sticky_repeats needs"),
        (dict(input="closedness", zero_holds=True), "zero_holds only applies"),
        (dict(on_missing="guess"), "on_missing"),
        (dict(input="closedness", on_missing="measured"), "needs continuous=True"),
    ],
)
def test_invalid_configuration_fails_in_build(env, kw, match):
    with pytest.raises(ValueError, match=match):
        _term(env, **kw)


def test_processed_targets_are_per_dof_open_close_actions(env):
    term = _term(env, input="openness")
    _command(term, 1.0)
    np.testing.assert_allclose(term.processed_action[0].numpy(), [0.037, -0.037])
    _command(term, 0.0)
    np.testing.assert_allclose(term.processed_action[0].numpy(), [0.0, 0.0])


# ---------------------------------------------------------------------------
# The ActionManager stores what the gripper term acted on, never the NaN "no command" sentinel
# ---------------------------------------------------------------------------


def _gripper_column(env) -> int:
    offset = 0
    for name, term in env.action_manager.terms.items():
        if name == "gripper_controller":
            return offset
        offset += term.action_dim
    raise AssertionError("no gripper_controller term")


def test_manager_action_buffers_hold_the_last_gripper_command_on_nan(env):
    manager, col = env.action_manager, _gripper_column(env)
    arm = torch.full((_NUM_ENVS, col), 0.1)
    manager.compute(torch.cat([arm, torch.full((_NUM_ENVS, 1), 0.8)], dim=1))
    manager.compute(torch.cat([arm, torch.full((_NUM_ENVS, 1), float("nan"))], dim=1))
    assert torch.isfinite(manager.action).all() and torch.isfinite(manager.prev_action).all()
    torch.testing.assert_close(manager.action[:, col], torch.full((_NUM_ENVS,), 0.8))
    # The dict form stores the same resolved value.
    manager.compute({"gripper_controller": torch.full((_NUM_ENVS, 1), float("nan"))})
    torch.testing.assert_close(manager.action[:, col], torch.full((_NUM_ENVS,), 0.8))
    torch.testing.assert_close(manager.action[:, :col], arm)


def test_a_missing_command_records_the_measured_seed_it_acted_on(env):
    """A NaN the term fills from the measured gripper is recorded as that value, not a stale or reset one."""
    _set_measured_closedness(env, 0.9)
    # Never commanded since reset: the first miss closes on the measured gripper, and records closedness 0.9 --
    # not the reset value 0.0, which reads as "open".
    hold = _term(env, input="closedness")
    _command(hold, float("nan"))
    assert hold.last_closedness.tolist() == [1.0, 1.0]
    torch.testing.assert_close(hold.action, torch.full((_NUM_ENVS, 1), 0.9))
    # on_missing="measured" acts on the measured closedness at every miss, so that is what each miss records.
    measured = _term(env, input="openness", continuous=True, on_missing="measured")
    _command(measured, 0.8)
    _set_measured_closedness(env, 0.25)
    _command(measured, float("nan"))
    torch.testing.assert_close(measured.action, torch.full((_NUM_ENVS, 1), 0.75))  # openness = 1 - 0.25
    torch.testing.assert_close(measured.prev_action, torch.full((_NUM_ENVS, 1), 0.8))


def test_manager_action_buffers_record_the_measured_seed_after_a_reset(env):
    manager, col = env.action_manager, _gripper_column(env)
    manager.reset()
    _set_measured_closedness(env, 0.9)
    manager.compute({"gripper_controller": torch.full((_NUM_ENVS, 1), float("nan"))})
    torch.testing.assert_close(manager.action[:, col], torch.full((_NUM_ENVS,), 0.9))
