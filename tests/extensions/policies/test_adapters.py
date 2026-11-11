"""Tests for ``eden.extensions.policies.adapters``: the shared robot constants and DOF helpers, the
DROID joint-velocity adapter on a real Droid env, the EE-delta adapter's rotation conventions, and the
row contract every adapter shares.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eden.extensions.policies.adapters import (
    BRIDGE_DEFAULT_ROT,
    DROID_ARM_DOFS,
    ActionAdapter,
    BimanualJointPositionAdapter,
    EEDeltaAdapter,
    JointPositionAdapter,
    JointVelocityAdapter,
    bridge_rpy,
    check_control_rate,
    check_droid_arm_term,
    check_droid_joint_state,
    check_env_gripper_term,
    ee_delta_to_world,
    gripper_dofs_from_name,
    make_droid_joint_position_actions,
    make_droid_joint_state_group,
)
from eden.managers.terms.actions.binary_actions import GripperDecisionController
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.options.robots.trossen_widowx250 import WidowX250
from tests.extensions.policies.conftest import policy_env


def test_gripper_dofs_from_name_expands_robotiq_pairs_only():
    assert gripper_dofs_from_name("left_driver_joint") == ["left_driver_joint", "right_driver_joint"]
    assert gripper_dofs_from_name("right_driver_joint") == ["left_driver_joint", "right_driver_joint"]
    assert gripper_dofs_from_name("left_finger") == ["left_finger"]
    assert len(DROID_ARM_DOFS) == 7


# ---------------------------------------------------------------------------
# JointVelocityAdapter (pi DROID)
# ---------------------------------------------------------------------------


def test_joint_velocity_adapter_official_passthrough_and_gripper_forwarding():
    """Arm velocities are clipped to OpenPI's [-1, 1] safety cap; the gripper scalar reaches the env's
    GripperDecisionController unchanged, and the diagnostics label its intent at the threshold.
    """
    env = policy_env("droid", {})
    adapter = JointVelocityAdapter(
        n_arm=7, arm_dofs_name=DROID_ARM_DOFS, mode="velocity", action_scale=1.0, clip=1.0, gripper_threshold=0.5
    )

    raw = np.array([0.2, -0.4, 0.0, 0.8, -1.2, 0.5, -0.1, 0.49], dtype=np.float32)
    action = adapter(raw, env=env)
    assert torch.allclose(action["ee_controller"], torch.tensor([[0.2, -0.4, 0.0, 0.8, -1.0, 0.5, -0.1]]))
    torch.testing.assert_close(action["gripper_controller"], torch.tensor([[0.49]]))
    assert np.allclose(adapter.last_diagnostics["clipped_arm_action"], [0.2, -0.4, 0.0, 0.8, -1.0, 0.5, -0.1])
    assert np.allclose(adapter.last_diagnostics["target_velocity"], [0.2, -0.4, 0.0, 0.8, -1.0, 0.5, -0.1])
    assert np.isclose(adapter.last_diagnostics["raw_gripper_action"], 0.49)
    assert np.isclose(adapter.last_diagnostics["gripper_command"], 0.49)
    assert adapter.last_diagnostics["gripper_intent"] == "open"

    for value, intent in ((0.50, "open"), (0.51, "close")):
        raw[-1] = value
        torch.testing.assert_close(adapter(raw, env=env)["gripper_controller"], torch.tensor([[value]]))
        assert adapter.last_diagnostics["gripper_intent"] == intent


def test_joint_velocity_adapter_is_rate_agnostic_in_velocity_mode():
    """Velocity commands pass through unscaled regardless of the env's control period.

    In ``velocity`` mode the adapter must not consult ``env.dt`` at all: it emits rad/s, and matching
    DROID's 15 Hz cadence is the *caller's* job (``DroidSetup`` refuses an env whose ``env.dt`` is not
    1/15 s, since a ClientMixin client executes one row per env step; an evaluator that holds each
    action for several env steps keeps its own period instead).
    """
    raw = np.array([0.2, -0.4, 0.0, 0.8, -1.2, 0.5, -0.1, 0.0], dtype=np.float32)
    expected = torch.tensor([[0.2, -0.4, 0.0, 0.8, -1.0, 0.5, -0.1]])

    for dt in (1.0 / 15.0, 0.02):
        adapter = JointVelocityAdapter(
            n_arm=7, arm_dofs_name=DROID_ARM_DOFS, mode="velocity", action_scale=1.0, clip=1.0
        )
        assert torch.allclose(adapter(raw, env=policy_env("droid", {}, dt=dt))["ee_controller"], expected), dt


def test_joint_velocity_adapter_accepts_per_joint_action_scale():
    """A 7-vector scale is applied elementwise, for per-joint tracking calibration."""
    env = policy_env("droid", {})
    scale = [0.873, 0.792, 0.919, 0.661, 0.854, 0.937, 0.675]
    adapter = JointVelocityAdapter(
        n_arm=7,
        arm_dofs_name=DROID_ARM_DOFS,
        mode="velocity",
        action_scale=scale,
        clip=1.0,
    )

    raw = np.array([1.0, 1.0, 1.0, 1.0, -1.5, 0.5, -1.0, 0.0], dtype=np.float32)
    action = adapter(raw, env=env)
    # Clip happens before scaling: joint5's -1.5 saturates to -1.0, then scales.
    expected = np.array([0.873, 0.792, 0.919, 0.661, -0.854, 0.5 * 0.937, -0.675])
    assert np.allclose(action["ee_controller"][0].numpy(), expected, atol=1e-6)
    assert np.allclose(adapter.last_diagnostics["target_velocity"], expected, atol=1e-6)


def test_joint_velocity_adapter_integration_uses_policy_action_period():
    """The integrated target advances from the measured joints by ``velocity * dt`` with the adapter's own
    ``dt``, not ``env.dt``: here 0.064 s, as an evaluator holding each action for several 0.02 s env steps
    would pass. A ClientMixin client executes one row per env step, so its ``dt`` is ``env.dt``.
    """
    env = policy_env("droid", {}, dt=0.02)
    adapter = JointVelocityAdapter(
        n_arm=7,
        arm_dofs_name=DROID_ARM_DOFS,
        mode="velocity_integrate",
        action_scale=1.0,
        clip=1.0,
        dt=0.064,
    )
    adapter.bind(env)

    raw = np.array([0.5, -0.5, 0.25, 0.0, 1.0, -1.0, 0.1, 0.0], dtype=np.float32)
    hold = adapter(np.zeros(8, dtype=np.float32), env=env)["ee_controller"]
    target = adapter(raw, env=env)["ee_controller"]
    torch.testing.assert_close(target - hold, torch.from_numpy(raw[:7] * 0.064).unsqueeze(0))


def test_joint_velocity_adapter_integrates_from_its_joints_in_its_own_order():
    """``q`` is read for ``arm_dofs_name`` in that order (here reversed), the order the arm term takes its columns
    in, not the entity's.
    """
    env = policy_env("droid", {})
    reversed_arm = list(reversed(DROID_ARM_DOFS))
    adapter = JointVelocityAdapter(
        n_arm=7, arm_dofs_name=reversed_arm, mode="velocity_integrate", action_scale=3.0, clip=1.0, dt=1.0 / 15.0
    )
    adapter.bind(env)
    raw = np.array([2.0, 0.5, -0.25, -3.0, 0.1, 0.0, 0.7, 0.0], dtype=np.float32)
    target = adapter(raw, env=env)["ee_controller"]
    robot = env.entities["robot"]
    q = robot.get_dofs_pos(dofs_idx_local=robot.find_named_dofs(reversed_arm).dofs_idx_local)
    assert len(set(q[0].tolist())) == 7  # policy_env's ramp makes every joint distinct, so the order is observable
    torch.testing.assert_close(target, q + torch.from_numpy(0.2 * np.clip(raw[:7], -1.0, 1.0)).unsqueeze(0))


def test_joint_velocity_adapter_lag_recurrence_and_reset():
    env = policy_env("droid", {})
    lag = {"tau_s": [0.2] * 7, "dc_gain": [0.8] * 7}
    adapter = JointVelocityAdapter(
        n_arm=7,
        arm_dofs_name=DROID_ARM_DOFS,
        mode="velocity",
        velocity_lag=lag,
        velocity_lag_dt=0.064,
    )
    raw = np.array([0.5] * 7 + [0.0], dtype=np.float32)
    a = np.exp(-0.064 / 0.2)
    first_expected = 0.8 * (1.0 - a) * 0.5
    second_expected = a * first_expected + 0.8 * (1.0 - a) * 0.5

    first = adapter(raw, env=env)["ee_controller"][0].numpy()
    second = adapter(raw, env=env)["ee_controller"][0].numpy()
    assert np.allclose(first, first_expected)
    assert np.allclose(second, second_expected)
    assert np.allclose(adapter.last_diagnostics["prefilter_target_velocity"], 0.5)

    adapter.reset()
    restarted = adapter(raw, env=env)["ee_controller"][0].numpy()
    assert np.allclose(restarted, first_expected)


def test_joint_velocity_adapter_compensates_real_over_sim_transfer():
    env = policy_env("droid", {})
    compensation = {
        "real_tau_s": [0.25] * 7,
        "real_dc_gain": [0.8] * 7,
        "sim_tau_s": [0.05] * 7,
        "sim_dc_gain": [1.0] * 7,
    }
    adapter = JointVelocityAdapter(
        n_arm=7,
        mode="velocity",
        velocity_compensation=compensation,
        velocity_lag_dt=0.064,
    )
    raw = np.array([0.5] * 7 + [0.0], dtype=np.float32)
    real_a = np.exp(-0.064 / 0.25)
    sim_a = np.exp(-0.064 / 0.05)
    real_b = 0.8 * (1.0 - real_a)
    sim_b = 1.0 * (1.0 - sim_a)

    first_expected = (real_b / sim_b) * 0.5
    second_expected = real_a * first_expected + (real_b / sim_b) * (0.5 - sim_a * 0.5)
    assert np.allclose(adapter(raw, env=env)["ee_controller"][0].numpy(), first_expected)
    assert np.allclose(adapter(raw, env=env)["ee_controller"][0].numpy(), second_expected)
    assert adapter.last_diagnostics["velocity_compensation_enabled"]

    adapter.reset()
    assert np.allclose(adapter(raw, env=env)["ee_controller"][0].numpy(), first_expected)


def test_joint_velocity_adapter_without_lag_is_exact_noop():
    env = policy_env("droid", {})
    adapter = JointVelocityAdapter(
        n_arm=7,
        arm_dofs_name=DROID_ARM_DOFS,
        mode="velocity",
        action_scale=1.0,
        clip=1.0,
    )
    raw = np.array([0.2, -0.4, 0.0, 0.8, -1.2, 0.5, -0.1, 0.0], dtype=np.float32)
    output = adapter(raw, env=env)["ee_controller"][0].numpy()
    assert np.array_equal(output, np.clip(raw[:7], -1.0, 1.0))
    assert np.array_equal(
        adapter.last_diagnostics["prefilter_target_velocity"],
        adapter.last_diagnostics["target_velocity"],
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_arm": 0}, "positive integer"),
        ({"action_scale": [1.0, 1.0]}, "one value per arm joint"),
        ({"action_scale": float("nan")}, "finite"),
        ({"clip": 0.0}, "positive and finite"),
        ({"dt": -0.1}, "positive finite"),
        (
            {"velocity_lag": {"tau_s": [0.2] * 7, "dc_gain": [1.0] * 7}},
            "requires velocity_lag_dt",
        ),
        (
            {
                "velocity_lag": {"tau_s": [float("nan")] * 7, "dc_gain": [1.0] * 7},
                "velocity_lag_dt": 0.064,
            },
            "must all be finite",
        ),
        (
            {
                "velocity_lag": {"tau_s": [0.2] * 7, "dc_gain": [1.0] * 7},
                "velocity_compensation": {
                    "real_tau_s": [0.2] * 7,
                    "real_dc_gain": [1.0] * 7,
                    "sim_tau_s": [0.05] * 7,
                    "sim_dc_gain": [1.0] * 7,
                },
                "velocity_lag_dt": 0.064,
            },
            "mutually exclusive",
        ),
    ],
)
def test_joint_velocity_adapter_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JointVelocityAdapter(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "needs arm_dofs_name"),
        ({"arm_dofs_name": list(DROID_ARM_DOFS[:6])}, r"names 6 joints .*, but n_arm=7"),
    ],
)
def test_an_integrating_joint_velocity_adapter_needs_one_name_per_arm_joint(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JointVelocityAdapter(n_arm=7, mode="velocity_integrate", **kwargs)


@pytest.mark.parametrize(
    ("entity_name", "arm_dofs_name", "match"),
    [
        ("arm", list(DROID_ARM_DOFS[:6]), r"entity_name='arm' is not an entity of the env. Available: \['gripper'"),
        ("robot", [*DROID_ARM_DOFS[:5], "jiont7"], "jiont7"),
        ("robot", ["joint[12]", *DROID_ARM_DOFS[2:]], r"selects 7 joints .*, but n_arm=6"),
    ],
)
def test_binding_an_integrating_adapter_resolves_its_joints_on_the_env(entity_name, arm_dofs_name, match):
    """A missing entity, an unknown joint or a glob selecting more joints than ``n_arm`` raises at bind, not mid-rollout."""
    adapter = JointVelocityAdapter(
        n_arm=6, entity_name=entity_name, arm_dofs_name=arm_dofs_name, mode="velocity_integrate"
    )
    with pytest.raises(ValueError, match=match):
        adapter.bind(policy_env("droid", {}))


def test_an_unbound_integrating_adapter_names_the_missing_bind():
    adapter = JointVelocityAdapter(arm_dofs_name=DROID_ARM_DOFS, mode="velocity_integrate")
    with pytest.raises(RuntimeError, match=r"called before bind\(env\)"):
        adapter(np.zeros(8, dtype=np.float32), env=policy_env("droid", {}))


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros(7, dtype=np.float32),
        np.zeros((1, 8), dtype=np.float32),
        np.array([0.0] * 7 + [float("nan")], dtype=np.float32),
    ],
)
def test_joint_velocity_adapter_rejects_invalid_raw_action(raw):
    adapter = JointVelocityAdapter()
    with pytest.raises(ValueError):
        adapter(raw, env=policy_env("droid", {}))


@pytest.mark.parametrize("raw", [np.zeros(7, dtype=np.float32), np.zeros((1, 8), dtype=np.float32)])
def test_joint_position_adapter_rejects_a_row_of_the_wrong_shape(raw):
    adapter = JointPositionAdapter(n_arm=7)
    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        adapter(raw, env=policy_env("droid", {}))


def test_joint_position_adapter_forwards_a_missing_gripper_as_nan():
    """GalaxeaVLA omits the gripper key; the NaN reaches the env's GripperDecisionController as "no command"."""
    adapter = JointPositionAdapter(n_arm=7)
    action = adapter(np.array([0.0] * 7 + [np.nan], dtype=np.float32), env=policy_env("droid", {}))
    assert torch.isnan(action["gripper_controller"]).all()


def test_droid_joint_state_group_reads_the_gripper_on_the_arm_entity_by_default():
    group = make_droid_joint_state_group("robot", DROID_ARM_DOFS, gripper_dof_name="left_driver_joint")
    assert (group.gripper.entity_name, group.gripper.normalize) == ("robot", "closedness")


def test_droid_joint_state_group_sends_closedness_clipped_to_the_unit_range():
    """DROID's gripper_position is in [0, 1]; a Robotiq driver can travel past its 0.8 close target."""
    import eden as en
    from eden.options import EnvOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots import Droid
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    group = make_droid_joint_state_group(
        "robot", DROID_ARM_DOFS, gripper_entity_name="gripper", gripper_dof_name="left_driver_joint"
    )
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=en.scenes.Empty(robot=Droid()),
            observation_options=ObservationManagerOptions(proprio=group),
        )
    )
    gripper = env.entities["gripper"]
    driver = gripper.dofs_name.index("left_driver_joint")
    for driver_pos, closedness in ((0.4, 0.5), (0.88, 1.0)):
        pos = gripper.get_dofs_pos().clone()
        pos[:, driver] = driver_pos
        gripper.set_dofs_pos(pos)
        proprio = env.observation_manager.compute_group("proprio")
        assert proprio.shape == (1, 8)
        assert proprio[0, -1].item() == pytest.approx(closedness)


# ---------------------------------------------------------------------------
# check_env_gripper_term: policies forwarding a raw gripper scalar need a GripperDecisionController
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def widowx_gripper_env():
    return policy_env(
        "widowx",
        dict(
            arm_controller=ImplicitPDController.configure(entity_name="robot", dofs_name=list(WidowX250.arm_dofs_name)),
            gripper_controller=GripperDecisionController.configure(
                entity_name="robot",
                dofs_name=list(WidowX250.gripper_dofs_name),
                open_action={"left_finger": 0.037, "right_finger": -0.037},
                close_action=0.0,
                input="closedness",
            ),
        ),
    )


def test_check_env_gripper_term_passes_on_the_matching_gripper_decision_term(widowx_gripper_env):
    check_env_gripper_term(widowx_gripper_env, "gripper_controller", input="closedness", threshold=0.5)


@pytest.mark.parametrize(
    ("term_name", "expected", "error", "match"),
    [
        ("missing_gripper", {}, ValueError, "no action term 'missing_gripper'"),
        ("arm_controller", {}, TypeError, "needs a GripperDecisionController"),
        ("gripper_controller", {"thresold": 0.5}, AttributeError, "no field"),
        ("gripper_controller", {"input": "openness"}, ValueError, "configured differently"),
    ],
)
def test_check_env_gripper_term_rejects_missing_foreign_or_mismatched_terms(
    widowx_gripper_env, term_name, expected, error, match
):
    with pytest.raises(error, match=match):
        check_env_gripper_term(widowx_gripper_env, term_name, **expected)


def test_check_env_gripper_term_refuses_a_decision_field_the_policy_leaves_at_its_default():
    # A negative scale swaps open and close under a sign input, so the policy's plain rule no longer holds.
    env = policy_env(
        "widowx",
        dict(
            arm_controller=ImplicitPDController.configure(entity_name="robot", dofs_name=list(WidowX250.arm_dofs_name)),
            gripper_controller=GripperDecisionController.configure(
                entity_name="robot",
                dofs_name=list(WidowX250.gripper_dofs_name),
                open_action={"left_finger": 0.037, "right_finger": -0.037},
                close_action=0.0,
                input="sign",
                scale=-1.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"scale: env term has -1\.0, policy does not use it \(default\) 1\.0"):
        check_env_gripper_term(env, "gripper_controller", input="sign")


# ---------------------------------------------------------------------------
# EEDeltaAdapter: the IK controller's angular rows take a rotation vector
# ---------------------------------------------------------------------------


def test_ee_delta_adapter_passes_an_axis_angle_delta_through_unchanged(widowx_gripper_env):
    """A LIBERO server's axis-angle delta already is the rotation vector DifferentialIKController
    solves for; converting it to Euler angles (the old behaviour) turned [0.3, 0.3, 0.3] into a
    different axis and angle.
    """
    raw = np.array([0.01, -0.02, 0.03, 0.3, 0.3, 0.3, 1.0], dtype=np.float32)
    action = EEDeltaAdapter(rotation="axis_angle")(raw, env=widowx_gripper_env)
    torch.testing.assert_close(action["ee_controller"], torch.tensor([[0.01, -0.02, 0.03, 0.3, 0.3, 0.3]]))
    torch.testing.assert_close(action["gripper_controller"], torch.tensor([[1.0]]))


def test_ee_delta_adapter_turns_an_euler_delta_into_the_same_rotation_vector(widowx_gripper_env):
    """An extrinsic-xyz Euler delta reaches the controller as the rotation vector of the same rotation
    (SimplerEnv's ``euler2axangle``), so both conventions command the same motion for the same
    rotation; the caller's action chunk is left untouched.
    """
    rpy = np.array([0.3, -0.2, 0.5])
    rotvec = Rotation.from_euler("xyz", rpy).as_rotvec()
    raw = np.array([0.01, -0.02, 0.03, *rpy, -1.0], dtype=np.float32)
    raw_before = raw.copy()

    from_euler = EEDeltaAdapter(rotation="euler")(raw, env=widowx_gripper_env)["ee_controller"]
    from_axis_angle = EEDeltaAdapter(rotation="axis_angle")(
        np.array([0.01, -0.02, 0.03, *rotvec, -1.0], dtype=np.float32), env=widowx_gripper_env
    )["ee_controller"]

    torch.testing.assert_close(from_euler[0, 3:], torch.tensor(rotvec, dtype=torch.float32), atol=1e-6, rtol=0)
    torch.testing.assert_close(from_euler, from_axis_angle, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(raw, raw_before)


def test_a_base_frame_ee_delta_reaches_the_world_frame_ik_rotated_by_the_bases_orientation():
    """Bridge-style servers compose deltas in the robot base frame and the IK solves in the world frame:
    on a base yawed 90 degrees, a base +x step and a roll about base x are a world +y step and a turn
    about world y. The base's position plays no part, as both halves are displacements.
    """
    yaw_90 = torch.tensor([[np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]], dtype=torch.float32)
    delta_b = torch.tensor([[0.01, 0.0, 0.02, 0.1, 0.0, 0.0]])
    torch.testing.assert_close(
        ee_delta_to_world(delta_b, yaw_90), torch.tensor([[0.0, 0.01, 0.02, 0.0, 0.1, 0.0]]), atol=1e-7, rtol=0
    )
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(ee_delta_to_world(delta_b, identity), delta_b)


def test_ee_delta_adapter_rejects_an_unknown_rotation_convention():
    with pytest.raises(ValueError, match="rotation must be 'euler' or 'axis_angle', got 'quat'"):
        EEDeltaAdapter(rotation="quat")


def test_bridge_rpy_is_the_rotation_relative_to_the_bridge_top_down_frame_in_xyz_euler():
    """Matches scipy's ``R @ BRIDGE_DEFAULT_ROT.T`` decomposed as extrinsic xyz, for either quaternion sign."""
    rng = np.random.default_rng(0)
    for rotation in Rotation.random(16, random_state=rng):
        quat_wxyz = np.roll(rotation.as_quat(), 1)
        expected = Rotation.from_matrix(rotation.as_matrix() @ BRIDGE_DEFAULT_ROT.T).as_euler("xyz")
        np.testing.assert_allclose(bridge_rpy(quat_wxyz), expected, atol=1e-12)
        np.testing.assert_allclose(bridge_rpy(-quat_wxyz), expected, atol=1e-12)
    np.testing.assert_allclose(bridge_rpy(np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Every adapter: one row in, per-env tensors out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "width"),
    [
        (EEDeltaAdapter(), 7),
        (JointPositionAdapter(n_arm=7), 8),
        (JointVelocityAdapter(n_arm=7), 8),
        (BimanualJointPositionAdapter(left_n=7, right_n=6), 13),
    ],
)
def test_every_adapter_refuses_a_row_of_another_width_and_never_aliases_the_row(widowx_gripper_env, adapter, width):
    """A batched ``(1, D)`` row or a short row raises instead of slicing silently; the returned tensors are copies,
    so the scheduler reusing its chunk buffer cannot change an action already handed to the env.
    """
    for bad in (np.zeros((1, width), dtype=np.float32), np.zeros(width - 1, dtype=np.float32)):
        with pytest.raises(ValueError, match=rf"expects an action row of shape \({width},\), got \({bad.shape[0]}"):
            adapter(bad, env=widowx_gripper_env)

    row = np.full(width, 0.25, dtype=np.float32)
    out = adapter(row, env=widowx_gripper_env)
    before = {key: value.clone() for key, value in out.items()}
    row[:] = 9.0
    for key, value in out.items():
        torch.testing.assert_close(value, before[key])
        assert value.shape[0] == widowx_gripper_env.num_envs


def test_an_adapter_without_a_decoder_cannot_be_built():
    class _NoDecoder(ActionAdapter):
        row_width = 1

    with pytest.raises(TypeError, match="abstract method"):
        _NoDecoder()


# ---------------------------------------------------------------------------
# DROID arm term and joint state checks
# ---------------------------------------------------------------------------

#: What the checks name as the client and the factory that builds the right terms.
_CLIENT = dict(subject="Client", fix="Client.options()")


def _droid_env(observation_options: dict | None = None, **arm_overrides):
    """The Droid rig with the DROID joint-position action terms, ``ee_controller`` updated by ``arm_overrides``."""
    actions = make_droid_joint_position_actions(gripper_entity_name="gripper")
    if arm_overrides:
        actions["ee_controller"] = actions["ee_controller"].model_copy(update=arm_overrides)
    return policy_env("droid", actions, observation_options=observation_options)


def _droid_joint_state(**joint_pos_options):
    """The DROID ``proprio`` group, its ``joint_pos`` term rebuilt with ``joint_pos_options`` when given."""
    from eden.managers.terms.observations.proprio import DofsPos

    group = make_droid_joint_state_group(
        "robot", DROID_ARM_DOFS, gripper_entity_name="gripper", gripper_dof_name="left_driver_joint"
    )
    if joint_pos_options:
        joint_pos = DofsPos.configure(entity_name="robot", dofs_name=list(DROID_ARM_DOFS), **joint_pos_options)
        group = group.model_copy(update={"joint_pos": joint_pos})
    return {"proprio": group}


def test_check_droid_arm_term_accepts_an_all_zero_per_joint_offset():
    """``offset`` may be a per-joint dict; one of zeros is the identity, not an offset."""
    env = _droid_env(offset=dict.fromkeys(DROID_ARM_DOFS, 0.0))
    check_droid_arm_term(env.action_manager.terms["ee_controller"], term_name="ee_controller", **_CLIENT)


def test_check_droid_arm_term_refuses_droids_joints_out_of_order_and_rules_out_reordering():
    """The server reads row ``i`` as DROID's joint ``i``; permuting the reply in the client would hide that."""
    env = _droid_env(dofs_name=list(reversed(DROID_ARM_DOFS)))
    with pytest.raises(
        ValueError,
        match=r"has dofs_name=\['joint7', .*\] \(needs \['joint1', .*'joint7'\], in that order\)\. Fix: build the "
        r"env's action_options from Client\.options\(\)\. Do not undo the difference in the client \(subtracting "
        r"the reference, dividing by the scale or reordering the rows\): the term must execute the reply as is\.$",
    ):
        check_droid_arm_term(env.action_manager.terms["ee_controller"], term_name="ee_controller", **_CLIENT)


def test_check_droid_arm_term_refuses_an_offset_set_after_build():
    """A ``set_dofs_pos_offset()`` (joint-zero domain randomization) shifts every target the server sends."""
    env = _droid_env()
    arm = env.action_manager.terms["ee_controller"]
    arm.set_dofs_pos_offset(torch.full((env.num_envs, len(DROID_ARM_DOFS)), 0.1, device=env.device))
    try:
        with pytest.raises(
            ValueError,
            match=r"Client sends the server's absolute joint targets to 'ee_controller' to be executed as they are, "
            r"but that ImplicitPDController has effective_offset up to 0\.1 in magnitude, shape \(1, 7\) \(row 0: "
            r"\[0\.1, .*\]\), from a set_dofs_pos_offset\(\) call \(needs 0\)\. Fix: build the env's action_options "
            r"from Client\.options\(\)\.",
        ):
            check_droid_arm_term(arm, term_name="ee_controller", **_CLIENT)
    finally:
        arm.set_dofs_pos_offset(torch.zeros(env.num_envs, len(DROID_ARM_DOFS), device=env.device))


def test_check_droid_joint_state_refuses_a_joint_state_of_another_entity_than_the_arm_term():
    """The joint names alone are not enough: the state must read the entity the arm term drives."""
    env = _droid_env(_droid_joint_state())
    check_droid_joint_state(env, env.action_manager.terms["ee_controller"], group="proprio", **_CLIENT)
    with pytest.raises(
        ValueError,
        match=r"'proprio' group does not match: 'joint_pos' reads 'robot' \['joint1', .*\] but the arm term drives "
        r"'gripper' \[.*\], so each joint would be sent another joint's command\. Fix: Client\.options\(\)\.",
    ):
        check_droid_joint_state(env, env.action_manager.terms["gripper_controller"], group="proprio", **_CLIENT)


def test_check_droid_joint_state_refuses_joint_angles_relative_to_the_default_pose():
    env = _droid_env(_droid_joint_state(offset_from_default=True))
    with pytest.raises(ValueError, match=r"'joint_pos' has offset_from_default=True \(the server needs absolute"):
        check_droid_joint_state(env, env.action_manager.terms["ee_controller"], group="proprio", **_CLIENT)


# ---------------------------------------------------------------------------
# check_control_rate
# ---------------------------------------------------------------------------


def test_check_control_rate_accepts_the_upstream_period():
    check_control_rate(policy_env("droid", {}, dt=1.0 / 15.0), control_hz=15.0, subject="Client", source="upstream")


def test_check_control_rate_names_the_speed_and_the_typed_example_timing():
    with pytest.raises(
        ValueError,
        match=r"Client executes one action row per 0\.066667 s \(15 Hz\) control step, as upstream does, but the env "
        r"steps every env\.dt=0\.020000 s .* run each chunk 3\.33x too fast\. Fix: .* EnvOptions\(sim_dt=1/120, "
        r"decimation=8\), with a note\. Do not rescale or resample the actions",
    ):
        check_control_rate(
            policy_env("droid", {}, dt=0.02),
            control_hz=15.0,
            subject="Client",
            source="upstream does",
            timing=("1/120", 8),
            timing_note=", with a note",
        )


def test_check_control_rate_closes_with_the_clients_own_wrong_fix_when_given():
    """A client whose rate is a server-side condition rules out changing the rate, not resampling the actions."""
    with pytest.raises(ValueError, match=r"decimation=8\)\. Do not change the rate\.$") as raised:
        check_control_rate(
            policy_env("droid", {}, dt=0.02),
            control_hz=15.0,
            subject="Client",
            source="upstream does",
            timing=("1/120", 8),
            wrong_fix="Do not change the rate.",
        )
    assert "resample" not in str(raised.value)


@pytest.mark.parametrize(
    ("dt", "speed", "example"),
    [
        # 0.02 s does not divide 1/15 s: the example keeps the decimation nearest the env's and rounds sim_dt to fit.
        (0.02, "3.33x too fast", "EnvOptions(sim_dt=1/45, decimation=3)"),
        # 1/60 s does: the example keeps the env's own sim_dt as typed.
        (1.0 / 60.0, "4.00x too fast", "EnvOptions(sim_dt=0.016666666666666666, decimation=4)"),
        # Slower than the period: one physics step per control step.
        (0.1, "1.50x too slow", "EnvOptions(sim_dt=1/15, decimation=1)"),
    ],
)
def test_check_control_rate_derives_an_example_timing_from_the_envs_sim_dt(dt, speed, example):
    with pytest.raises(ValueError) as excinfo:
        check_control_rate(policy_env("droid", {}, dt=dt), control_hz=15.0, subject="Client", source="upstream")
    message = str(excinfo.value)
    assert speed in message and example in message, message
