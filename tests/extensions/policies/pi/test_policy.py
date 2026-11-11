from __future__ import annotations

import contextlib
import re

import numpy as np
import pytest
import torch

from eden.constants import ReferenceSource
from eden.extensions.policies.adapters import DROID_ARM_DOFS, WIDOWX_DOFS
from eden.extensions.policies.pi.policy import PiPolicy
from eden.extensions.policies.adapters import LIBERO_CAMERA_RESOLUTION
from eden.extensions.policies.pi.setups import DroidSetup
from tests.extensions.policies.conftest import (
    GOLDEN_TASK,
    TILT_QUAT,
    assert_matches_golden,
    chunk,
    ee8,
    joint8,
    policy_env,
    record_golden,
    vec,
)
from tests.extensions.policies.conftest import image as golden_image
from tests.support.policy_servers import WebsocketPolicyServer

_DROID_OBS = {"image": golden_image(), "wrist_image": golden_image(seed=50), "proprio": joint8(0.0)}
#: Eden's Droid rig carries the Robotiq on its own ``gripper`` entity.
_DROID_RIG = dict(gripper_entity_name="gripper")


def _env(
    robot: str,
    setup: str,
    *,
    actions: dict | None = None,
    action_kwargs: dict | None = None,
    observation_options: dict | None = None,
    observation_kwargs: dict | None = None,
    use_wrist_image: bool | None = None,
    dt: float | None = None,
):
    """A real ``robot`` env built, as a run builds it, from ``setup``'s own action and observation options.

    ``actions`` / ``observation_options`` replace the factories' output outright (a refusal test's bad term or
    group). The observation options otherwise take the action options' ``gripper_entity_name``, then
    ``use_wrist_image`` and ``observation_kwargs``. LIBERO's cameras render at openpi's 256x256. The env steps
    every ``dt`` s, by default the setup's own control period.
    """
    if dt is None:
        dt = 1.0 / PiPolicy.SETUP_REGISTRY[setup].control_hz
    action_kwargs = action_kwargs or {}
    if actions is None:
        actions = PiPolicy.action_options(policy_setup=setup, **action_kwargs)
    if observation_options is None:
        kwargs = {key: action_kwargs[key] for key in ("gripper_entity_name",) if key in action_kwargs}
        if use_wrist_image is not None:
            kwargs["use_wrist_image"] = use_wrist_image
        observation_options = PiPolicy.observation_options(policy_setup=setup, **kwargs, **(observation_kwargs or {}))
    resolution = LIBERO_CAMERA_RESOLUTION if setup == "libero" else (16, 16)
    return policy_env(robot, actions, dt=dt, observation_options=observation_options, cam_resolution=resolution)


@contextlib.contextmanager
def _policy(
    robot: str,
    setup: str,
    rows: np.ndarray,
    *,
    action_kwargs: dict | None = None,
    observation_kwargs: dict | None = None,
    task_description: str | None = GOLDEN_TASK,
    dt: float | None = None,
    **configure,
):
    """A ``PiPolicy`` on a real ``robot`` env built from ``setup``'s options (see :func:`_env`), served ``rows``."""
    with WebsocketPolicyServer("msgpack_openpi", lambda call: {"actions": rows}) as server:
        env = _env(
            robot,
            setup,
            action_kwargs=action_kwargs,
            observation_kwargs=observation_kwargs,
            use_wrist_image=configure.get("use_wrist_image"),
            dt=dt,
        )
        options = PiPolicy.configure(
            policy_setup=setup, host=server.host, port=server.port, task_description=task_description, **configure
        )
        policy = PiPolicy(env, options)
        try:
            yield policy, server
        finally:
            policy.close()


def test_ee_delta_setups_declare_their_servers_gripper_convention():
    """Bridge servers emit openness in [0, 1]; the Google-robot checkpoints add the fractal latch."""
    from eden.extensions.policies.pi.setups import GoogleRobotSetup, WidowxBridgeSetup

    assert WidowxBridgeSetup.action_options()["gripper_controller"].input == "openness"
    assert GoogleRobotSetup.action_options()["gripper_controller"].input == "openness"
    assert GoogleRobotSetup.action_options()["gripper_controller"].sticky_repeats == 10


def test_droid_setup_expands_named_robotiq_gripper_dofs():
    actions = DroidSetup.action_options(
        gripper_entity_name="gripper",
        gripper_dof_name="left_driver_joint",
    )
    assert actions["gripper_controller"].entity_name == "gripper"
    assert actions["gripper_controller"].dofs_name == ["left_driver_joint", "right_driver_joint"]
    # The decision (closedness > threshold) lives in the env's GripperDecisionController; targets span
    # [0, gripper_close_pos] exactly like the former ImplicitPD(scale=close_pos) fed with 0/1.
    assert actions["gripper_controller"].name == "gripper_decision_controller"
    assert actions["gripper_controller"].input == "closedness"
    assert actions["gripper_controller"].open_action == 0.0
    assert actions["gripper_controller"].close_action == 0.8


def test_droid_setup_preserves_custom_single_gripper_dof():
    actions = DroidSetup.action_options(
        gripper_entity_name="gripper",
        gripper_dof_name="jaw_joint",
    )
    assert actions["gripper_controller"].dofs_name == ["jaw_joint"]
    assert actions["gripper_controller"].name == "gripper_decision_controller"


# ---------------------------------------------------------------------------
# DROID joint velocity: openpi clips each dim to [-1, 1]; DROID's RobotEnv moves 0.2 rad per unit per
# 15 Hz step (examples/droid/main.py, droid/robot_ik/robot_ik_solver.py joint_velocity_to_delta).
# ---------------------------------------------------------------------------

#: One DROID row: a component above 1 (clipped element-wise, not rescaled), one below -1, and in-range ones.
_DROID_VELOCITY = np.array([2.0, 0.5, -0.25, -3.0, 0.1, 0.0, 0.7], dtype=np.float32)
_DROID_VELOCITY_CLIPPED = np.clip(_DROID_VELOCITY, -1.0, 1.0)


def _droid_rows(velocity: np.ndarray) -> np.ndarray:
    rows = np.zeros((10, 8), dtype=np.float32)
    rows[:, :7] = velocity
    return rows


def test_droid_velocity_mode_commands_three_rad_per_second_per_unit_after_openpis_clip():
    """``velocity``: 0.2 rad per 1/15 s step = 3 rad/s per unit, each dim clipped to [-1, 1] first.

    The element-wise clip is openpi's; it leaves DROID's direction-preserving ``v / max|v|`` rescale
    nothing to do, so [2.0, 0.5, ...] commands [3.0, 1.5, ...], not the rescaled [3.0, 0.75, ...].
    """
    with _policy("droid", "droid", _droid_rows(_DROID_VELOCITY), action_kwargs=dict(gripper_entity_name="gripper")) as (
        policy,
        _,
    ):
        velocity = policy.step(_DROID_OBS)["ee_controller"]
    torch.testing.assert_close(velocity, torch.tensor([3.0 * _DROID_VELOCITY_CLIPPED]))


def test_droid_velocity_integrate_targets_the_arm_terms_joints_plus_point_two_rad_per_unit():
    """``velocity_integrate``: DROID's own position target ``q + 0.2 * clip(v)`` per policy step, ``q`` read
    from the joints the env's arm term drives. Reading them in the term's own order is the adapter's
    (``test_joint_velocity_adapter_integrates_from_its_joints_in_its_own_order``): a DROID arm term must keep
    DROID's order.
    """
    action_kwargs = dict(_DROID_RIG, arm_controller_kind="velocity_integrate")
    with _policy(
        "droid",
        "droid",
        _droid_rows(_DROID_VELOCITY),
        action_kwargs=action_kwargs,
        droid_arm_controller_kind="velocity_integrate",
    ) as (policy, _):
        target = policy.step(_DROID_OBS)["ee_controller"]
        robot = policy.env.entities["robot"]
        q = robot.get_dofs_pos(dofs_idx_local=robot.find_named_dofs(list(DROID_ARM_DOFS)).dofs_idx_local)
    torch.testing.assert_close(target, q + torch.tensor([0.2 * _DROID_VELOCITY_CLIPPED]))


@pytest.mark.parametrize(
    ("setup", "env_kind", "policy_kind", "found", "expected"),
    [
        ("droid", "velocity_integrate", "velocity", "ImplicitPDController", "ImplicitVelocityController"),
        ("droid", "velocity", "velocity_integrate", "ImplicitVelocityController", "ImplicitPDController"),
        ("droid_jointpos", "velocity", "velocity", "ImplicitVelocityController", "ImplicitPDController"),
    ],
)
def test_a_droid_policy_refuses_an_env_whose_arm_term_its_actions_do_not_fit(
    setup, env_kind, policy_kind, found, expected
):
    env = _env("droid", "droid", action_kwargs=dict(_DROID_RIG, arm_controller_kind=env_kind))
    with pytest.raises(
        TypeError,
        match=rf"policy_setup='{setup}' needs action term 'ee_controller' of type {expected}, but the env built "
        rf"{found}\. Fix: build the env's action_options from PiPolicy\.action_options\(policy_setup='{setup}'",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, droid_arm_controller_kind=policy_kind))


@pytest.mark.parametrize(
    ("setup", "kind"), [("droid", "velocity"), ("droid", "velocity_integrate"), ("droid_jointpos", "velocity")]
)
def test_droid_setups_refuse_an_env_not_stepping_at_droids_15_hz(setup, kind):
    """Every DROID row is one 1/15 s step (0.2 rad per unit, or the next joint waypoint); a 20 ms env (Eden's
    PolaRiS tasks) would play the chunk 3.3x too fast.
    """
    action_kwargs = dict(_DROID_RIG, arm_controller_kind=kind) if setup == "droid" else _DROID_RIG
    env = _env("droid", setup, action_kwargs=action_kwargs, dt=0.02)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"PiPolicy.policy_setup={setup!r} executes one action row per 0.066667 s (15 Hz) control step, as DROID's "
            "RobotEnv does (control_hz=15), but the env steps every env.dt=0.020000 s (sim_dt * decimation), which "
            "would run each chunk 3.33x too fast. Fix: build the env with sim_dt * decimation = 1/15 s, e.g. "
            "EnvOptions(sim_dt=1/120, decimation=8) as PolaRiS does, with sim_substeps=9"
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, droid_arm_controller_kind=kind))


@pytest.mark.parametrize(
    ("robot", "setup", "hz", "period", "speed", "timing"),
    [
        ("aloha", "aloha", 50, "0.020000", "3.33x too slow", "EnvOptions(sim_dt=0.002, decimation=10)"),
        ("libero", "libero", 20, "0.050000", "1.33x too slow", "EnvOptions(sim_dt=0.002, decimation=25)"),
        ("widowx", "widowx_bridge", 5, "0.200000", "3.00x too fast", "EnvOptions(sim_dt=0.005, decimation=40)"),
        ("widowx", "google_robot", 3, "0.333333", "5.00x too fast", "EnvOptions(sim_dt=1/300, decimation=100)"),
    ],
)
def test_setups_refuse_an_env_not_stepping_at_their_upstream_control_rate(robot, setup, hz, period, speed, timing):
    """Each row is one upstream control step (gym_aloha 50 Hz, LIBERO's robosuite 20 Hz, SimplerEnv's Bridge 5 Hz
    and Fractal 3 Hz); a 1/15 s env would replay every chunk at another speed.
    """
    env = _env(robot, setup, dt=1.0 / 15.0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"PiPolicy.policy_setup={setup!r} executes one action row per {period} s ({hz} Hz) control step"
        )
        + r".*"
        + re.escape(
            f"but the env steps every env.dt=0.066667 s (sim_dt * decimation), which would run each chunk {speed}. "
            f"Fix: build the env with sim_dt * decimation = 1/{hz} s, e.g. {timing}"
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup))


@pytest.mark.parametrize("setup", list(PiPolicy.SETUP_REGISTRY))
def test_the_timing_each_rate_refusal_suggests_steps_at_the_setups_rate(setup):
    from fractions import Fraction

    setup_class = PiPolicy.SETUP_REGISTRY[setup]
    sim_dt, decimation = setup_class._CONTROL_TIMING
    assert Fraction(sim_dt) * decimation == 1 / Fraction(setup_class.control_hz)


def test_a_droid_joint_position_policy_refuses_an_arm_term_that_adds_the_default_pose():
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    actions = PiPolicy.action_options(policy_setup="droid_jointpos", **_DROID_RIG)
    actions["ee_controller"] = ImplicitPDController.configure(entity_name="robot", dofs_name=list(DROID_ARM_DOFS))
    env = _env("droid", "droid_jointpos", actions=actions, observation_kwargs=_DROID_RIG)
    with pytest.raises(
        ValueError,
        match=r"PiPolicy\.policy_setup='droid_jointpos' sends absolute joint targets to 'ee_controller' to be executed "
        r"as they are, but that ImplicitPDController has reference_source='default'",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="droid_jointpos"))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (dict(scale=0.5), r"effective_scale=\[0\.5\] \(from scale=0\.5, scale_ratio=None, needs 1\)"),
        (dict(scale_ratio=0.5), r"effective_scale=\[.*\] \(from scale=None, scale_ratio=0\.5, needs 1\)"),
        (dict(offset=0.1), r"offset=0\.1 \(needs 0\)"),
    ],
    ids=["scale", "scale_ratio", "offset"],
)
def test_a_droid_velocity_policy_refuses_an_arm_term_that_rescales_or_offsets_its_rad_per_second(override, match):
    """The client already sends DROID's 3 rad/s per unit: a scale changes every joint's speed, and an offset is a
    constant drift while the server commands rest.
    """
    from eden.managers.terms.actions.joint_actions import ImplicitVelocityController

    actions = PiPolicy.action_options(policy_setup="droid", **_DROID_RIG)
    actions["ee_controller"] = ImplicitVelocityController.configure(
        entity_name="robot", dofs_name=list(DROID_ARM_DOFS), **override
    )
    env = _env("droid", "droid", actions=actions, observation_kwargs=_DROID_RIG)
    with pytest.raises(
        ValueError,
        match=r"PiPolicy\.policy_setup='droid' sends joint velocities, already converted to rad/s, to 'ee_controller' "
        r"to be executed as they are, but that ImplicitVelocityController has " + match + r"\. Fix: build the env's "
        r"action_options from PiPolicy\.action_options\(policy_setup='droid', arm_controller_kind='velocity'\)\. "
        r"Do not rescale the server's rows",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="droid"))


_DROID_SETUPS = pytest.mark.parametrize(
    ("setup", "action_kwargs"),
    [("droid", {}), ("droid", dict(arm_controller_kind="velocity_integrate")), ("droid_jointpos", {})],
    ids=["velocity", "velocity_integrate", "jointpos"],
)


@_DROID_SETUPS
def test_a_droid_policy_refuses_an_arm_term_out_of_droids_joint_order(setup, action_kwargs):
    """The server's row i is DROID's joint i; an arm term on the same joints in another order would mix them up."""
    reversed_arm = list(reversed(DROID_ARM_DOFS))
    env = _env("droid", setup, action_kwargs=dict(_DROID_RIG, arm_dofs_name=reversed_arm, **action_kwargs))
    kind = action_kwargs.get("arm_controller_kind", "velocity")
    with pytest.raises(
        ValueError, match=re.escape(f"dofs_name={reversed_arm} (needs {list(DROID_ARM_DOFS)}, in that order)")
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, droid_arm_controller_kind=kind))


@_DROID_SETUPS
def test_a_droid_policy_refuses_a_proprio_whose_joint_order_differs_from_the_arm_terms(setup, action_kwargs):
    """The server's row i is joint i of the state it was sent; a reversed proprio would send each joint another's."""
    reversed_arm = list(reversed(DROID_ARM_DOFS))
    env = _env(
        "droid",
        setup,
        action_kwargs=dict(_DROID_RIG, **action_kwargs),
        observation_kwargs=dict(arm_dofs_name=reversed_arm),
    )
    kind = action_kwargs.get("arm_controller_kind", "velocity")
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"PiPolicy.policy_setup={setup!r} sends proprio[:7] as DROID's joint_position and proprio[7] as its "
            "gripper_position, and the server replies commands for those joints in that order, but the env's "
            f"'proprio' group does not match: 'joint_pos' reads 'robot' {reversed_arm} but the arm term drives "
            f"'robot' {list(DROID_ARM_DOFS)}, so each joint would be sent another joint's command. Fix: pass the "
            f"same arm_entity_name and arm_dofs_name to PiPolicy.observation_options(policy_setup={setup!r}, "
            f"arm_entity_name='robot', arm_dofs_name={list(DROID_ARM_DOFS)!r})"
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, droid_arm_controller_kind=kind))


@pytest.mark.parametrize(
    ("robot", "setup", "edit", "problem"),
    [
        ("droid", "droid", lambda group: group.model_copy(update={"history_length": 2}), "history_length=2"),
        (
            "droid",
            "droid",
            lambda group: PiPolicy.observation_options(
                policy_setup="droid", arm_dofs_name=list(DROID_ARM_DOFS[:6]), **_DROID_RIG
            )["proprio"],
            "its terms ['joint_pos', 'gripper'] are (7,) wide, not (8,)",
        ),
        ("aloha", "aloha", lambda group: group.model_copy(update={"concatenate_terms": False}), "concatenate_terms"),
        # The sign flip belongs to _pack_state; done again in the group, it would cancel out.
        (
            "libero",
            "libero",
            lambda group: group.model_copy(
                update={"gripper_qpos": group.gripper_qpos.model_copy(update={"scale": [1.0, -1.0]})}
            ),
            "gripper_qpos.scale=[1.0, -1.0]",
        ),
    ],
    ids=["droid-history", "droid-six-joints", "aloha-split", "libero-negated-finger"],
)
def test_a_proprio_group_the_setup_cannot_slice_by_column_is_refused_at_construction(robot, setup, edit, problem):
    """``pack_observations`` reads ``proprio`` by column; an older frame first or another width shifts every column."""
    action_kwargs = _DROID_RIG if robot == "droid" else {}
    groups = PiPolicy.observation_options(policy_setup=setup, **action_kwargs)
    groups["proprio"] = edit(groups["proprio"])
    env = _env(robot, setup, action_kwargs=action_kwargs, observation_options=groups)
    with pytest.raises(
        ValueError,
        match=re.escape("Observation group 'proprio' is not the plain current reading the client packs by column: ")
        + rf".*{re.escape(problem)}.*Fix: build the env's 'proprio' group from "
        + re.escape(f"PiPolicy.observation_options(policy_setup={setup!r})."),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup))


def test_a_droid_policy_refuses_a_gripper_state_over_another_stroke_than_the_gripper_term_commands():
    """The proprio's closedness divides by ``gripper_close_pos``; the gripper term closes to its own one."""
    groups = PiPolicy.observation_options(policy_setup="droid", gripper_close_pos=0.7, **_DROID_RIG)
    env = _env("droid", "droid", action_kwargs=_DROID_RIG, observation_options=groups)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Gripper state proprio/gripper and action term 'gripper_controller' describe different grippers: "
            "close_pos=0.7 but 'gripper_controller' has close_action=0.8"
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="droid"))


@pytest.mark.parametrize(
    ("robot", "setup", "configure", "readers"),
    [
        # droid_jointpos sends absolute targets straight to the PD: no controller kind, no velocity filter.
        ("droid", "droid_jointpos", dict(droid_arm_controller_kind="bogus"), "droid_arm_controller_kind: ['droid']"),
        (
            "droid",
            "droid_jointpos",
            dict(droid_velocity_lag={"tau_s": [0.2] * 7, "dc_gain": [1.0] * 7}),
            "droid_velocity_lag: ['droid']",
        ),
        (
            "droid",
            "droid_jointpos",
            dict(droid_velocity_compensation={"real_tau_s": [0.2] * 7}),
            "droid_velocity_compensation: ['droid']",
        ),
        (
            "widowx",
            "widowx_bridge",
            dict(droid_gripper_threshold=0.7),
            "droid_gripper_threshold: ['droid', 'droid_jointpos']",
        ),
        ("widowx", "google_robot", dict(aloha_left_camera_name="cam_wrist"), "aloha_left_camera_name: ['aloha']"),
        ("aloha", "aloha", dict(droid_debug_gripper=True), "droid_debug_gripper: ['droid', 'droid_jointpos']"),
    ],
    ids=[
        "jointpos-kind",
        "jointpos-lag",
        "jointpos-compensation",
        "bridge-threshold",
        "google-aloha-cam",
        "aloha-droid",
    ],
)
def test_a_setup_specific_field_the_selected_setup_does_not_read_is_refused(robot, setup, configure, readers):
    env = _env(robot, setup, action_kwargs=_DROID_RIG if robot == "droid" else {})
    ((name, value),) = configure.items()
    expected = (
        f"PiPolicy {name}={value!r} has no effect under policy_setup={setup!r}, which does not read it; the setups "
        f"that do: {readers}. Fix: drop {name} from PiPolicy.configure(...)"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, **configure))


def test_setup_specific_fields_passed_at_their_defaults_are_accepted():
    """Passing a field's default is not a setting: droid_jointpos takes droid_arm_controller_kind='velocity'."""
    rows = np.zeros((10, 8), dtype=np.float32)
    defaults = dict(
        droid_arm_controller_kind="velocity", droid_velocity_lag=None, aloha_left_camera_name="cam_left_wrist"
    )
    with _policy("droid", "droid_jointpos", rows, action_kwargs=dict(gripper_entity_name="gripper"), **defaults) as (
        policy,
        _,
    ):
        assert policy.droid_arm_controller_kind == "velocity"


@pytest.mark.parametrize(
    ("robot", "setup", "action_kwargs", "reader", "key"),
    [
        ("droid", "droid", dict(gripper_entity_name="gripper"), "DroidInputs", "observation/wrist_image_left"),
        ("droid", "droid_jointpos", dict(gripper_entity_name="gripper"), "DroidInputs", "observation/wrist_image_left"),
        ("libero", "libero", {}, "LiberoInputs", "observation/wrist_image"),
    ],
)
def test_setups_refuse_to_run_without_the_wrist_image_their_server_reads(robot, setup, action_kwargs, reader, key):
    env = _env(robot, setup, action_kwargs=action_kwargs)
    with pytest.raises(
        ValueError,
        match=rf"PiPolicy.policy_setup='{setup}' needs use_wrist_image=True: OpenPI's {reader} reads '{key}' on "
        r"every request",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, use_wrist_image=False))


@pytest.mark.parametrize(("setup", "action_kwargs"), [("droid", dict(gripper_entity_name="gripper")), ("libero", {})])
def test_setups_whose_server_reads_the_wrist_image_send_it_by_default(setup, action_kwargs):
    assert "wrist_image" in PiPolicy.observation_options(policy_setup=setup)
    rows = np.zeros((10, 8 if setup == "droid" else 7), dtype=np.float32)
    with _policy(setup, setup, rows, action_kwargs=action_kwargs) as (policy, _):
        assert policy.use_wrist_image is True


def test_a_droid_observation_without_a_wrist_image_names_the_missing_group():
    with (
        _ramp_policy(15) as (policy, server),
        pytest.raises(KeyError, match=r"no 'wrist_image' group; got groups \['image', 'proprio'\]"),
    ):
        policy.step({"image": golden_image(), "proprio": joint8(0.0)})
    assert server.infer_calls == 0


@pytest.mark.parametrize(("robot", "setup"), [("widowx", "widowx_bridge"), ("aloha", "aloha")])
def test_use_wrist_image_is_refused_at_construction_on_an_env_without_the_wrist_image_group(robot, setup):
    """The setup's default observation options carry no wrist camera, so the first step would have none to send."""
    env = _env(robot, setup)
    with pytest.raises(
        ValueError,
        match=rf"PiPolicy\.policy_setup='{setup}' packs observation groups \['image', 'proprio', 'wrist_image'\] "
        r"into every request \(use_wrist_image=True\), but the env has no \['wrist_image'\]; its groups: \['image', "
        rf"'proprio'\]\. Fix: build the env's observation_options from PiPolicy\.observation_options\(policy_setup="
        rf"'{setup}', use_wrist_image=True\), or PiPolicy\.configure\(use_wrist_image=False\)",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup, use_wrist_image=True))


@pytest.mark.parametrize(
    ("robot", "setup", "action_kwargs", "groups", "missing"),
    [
        ("droid", "droid", _DROID_RIG, ("image", "wrist_image"), "['proprio']"),
        ("libero", "libero", {}, ("proprio", "wrist_image"), "['image']"),
        ("widowx", "google_robot", {}, (), "['image', 'proprio']"),
    ],
    ids=["droid-proprio", "libero-image", "google-none"],
)
def test_an_env_without_a_group_the_setup_packs_is_refused_at_construction(
    robot, setup, action_kwargs, groups, missing
):
    options = PiPolicy.observation_options(policy_setup=setup, **action_kwargs)
    env = _env(
        robot,
        setup,
        action_kwargs=action_kwargs,
        observation_options={group: options[group] for group in groups},
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"but the env has no {missing}; its groups: {sorted(groups)}. Fix: build the env's "
            f"observation_options from PiPolicy.observation_options(policy_setup={setup!r})"
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup))


def test_droid_integrated_position_controller_does_not_add_default_pose_offset():
    actions = DroidSetup.action_options(arm_controller_kind="velocity_integrate")
    assert actions["ee_controller"].reference_source == ReferenceSource.ZERO


def test_droid_action_options_refuse_an_unknown_kind_naming_the_valid_ones():
    with pytest.raises(ValueError, match=r"must be 'velocity' or 'velocity_integrate', got 'position'"):
        DroidSetup.action_options(arm_controller_kind="position")


# ---------------------------------------------------------------------------
# Prompt: sent with every request where the upstream client sends one; ALOHA leaves it to the server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("robot", "setup", "action_kwargs", "obs", "width", "sender"),
    [
        ("droid", "droid", dict(gripper_entity_name="gripper"), _DROID_OBS, 8, "openpi examples/droid/main.py"),
        (
            "droid",
            "droid_jointpos",
            dict(gripper_entity_name="gripper"),
            _DROID_OBS,
            8,
            "openpi examples/droid/main.py",
        ),
        (
            "libero",
            "libero",
            {},
            {"image": golden_image(), "wrist_image": golden_image(seed=1), "proprio": torch.zeros(1, 9)},
            7,
            "openpi examples/libero/main.py",
        ),
        ("widowx", "widowx_bridge", {}, {"image": golden_image(), "proprio": ee8(1.0)}, 7, "the Bridge / Fractal"),
        ("widowx", "google_robot", {}, {"image": golden_image(), "proprio": ee8(1.0)}, 7, "the Bridge / Fractal"),
    ],
)
def test_setups_whose_upstream_client_always_prompts_refuse_to_send_without_a_task(
    robot, setup, action_kwargs, obs, width, sender
):
    rows = np.zeros((10, width), dtype=np.float32)
    with (
        _policy(robot, setup, rows, action_kwargs=action_kwargs, task_description=None) as (policy, server),
        pytest.raises(
            ValueError,
            match=rf"policy_setup='{setup}' sends 'prompt' with every request, as {sender}.* but "
            r"PiPolicy\.task_description is None\. Fix: policy\.set_task\('<instruction>'\)",
        ),
    ):
        policy.step(obs)
    assert server.infer_calls == 0


def test_aloha_omits_the_prompt_until_a_task_is_set_so_the_server_applies_its_default_prompt():
    """The upstream ``examples/aloha_sim`` sends no prompt; ``InjectDefaultPrompt`` only fills a missing one."""
    rows = np.zeros((10, 14), dtype=np.float32)
    obs = {"image": golden_image(), "proprio": torch.zeros(1, 14)}
    with _policy("aloha", "aloha", rows, task_description=None) as (policy, server):
        policy.step(obs)
        policy.set_task("transfer the cube")
        policy.step(obs)
    first, second = (record for record in server.records if "state" in record)
    assert "prompt" not in first
    assert second["prompt"] == "transfer the cube"


def test_the_client_has_no_default_task_description():
    assert PiPolicy.configure().task_description is None


# ---------------------------------------------------------------------------
# LIBERO: the stock task's own action terms, robosuite's world-frame state (examples/libero/main.py)
# ---------------------------------------------------------------------------


def test_libero_setup_drives_the_libero_tasks_own_action_terms():
    """The server row reaches ``osc_pose`` (6) and ``gripper`` (1) unchanged, as openpi steps robosuite."""
    from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
    from eden.tasks.libero.custom_terms.actions import LiberoPandaGripperController

    row = np.array([0.1, -0.2, 0.3, 0.05, -0.04, 0.02, 1.0], dtype=np.float32)
    with _policy("libero", "libero", np.repeat(row[None], 5, axis=0)) as (policy, _):
        env = policy.env
        osc, gripper = env.action_manager.terms["osc_pose"], env.action_manager.terms["gripper"]
        assert type(osc) is OperationalSpaceController and type(gripper) is LiberoPandaGripperController
        assert (osc.position_scale, osc.orientation_scale, osc.action_clip) == (0.05, 0.5, 1.0)
        action = policy.step(
            {"image": golden_image(), "wrist_image": golden_image(seed=1), "proprio": torch.zeros(1, 9)}
        )
        env.action_manager.compute(action)
    assert sorted(action) == ["gripper", "osc_pose"]
    torch.testing.assert_close(action["osc_pose"], torch.tensor([row[:6]]))
    torch.testing.assert_close(action["gripper"], torch.tensor([[1.0]]))


def _libero_payload(obs: dict) -> dict:
    """The one inference payload a LIBERO policy sends for ``obs``."""
    with _policy("libero", "libero", np.zeros((5, 7), dtype=np.float32)) as (policy, server):
        policy.step(obs)
    (payload,) = server.records
    return payload


def test_libero_mirrors_both_cameras_left_right_before_resize_with_pad():
    """The upstream ``[::-1, ::-1]`` on robosuite's bottom-up frame is the upright view mirrored; Eden's is upright."""
    from eden.utils.image import prepare_image_padded

    image, wrist = golden_image(seed=3), golden_image(seed=4)  # random, so asymmetric
    payload = _libero_payload({"image": image, "wrist_image": wrist, "proprio": torch.zeros(1, 9)})
    for key, frame in (("observation/image", image), ("observation/wrist_image", wrist)):
        mirrored = prepare_image_padded(np.ascontiguousarray(frame[0].numpy()[:, ::-1]), (224, 224))
        np.testing.assert_array_equal(payload[key], mirrored)
        assert not np.array_equal(payload[key], prepare_image_padded(frame, (224, 224)))


def test_libero_agentview_marker_on_the_world_plus_y_side_reaches_the_server_on_the_left():
    """The agentview camera's x-axis is world +y, so a +y marker is on the right of Eden's render and on the left
    of the mirrored frame the openpi LIBERO checkpoints were trained on.
    """
    from genesis.options.sensors import RasterizerCameraOptions
    from genesis.utils.geom import quat_to_R, trans_quat_to_T

    from eden.options import EnvOptions
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.entities import SceneOptions, SphereOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.tasks.libero.scenes import _get_libero_family_spec
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    spec = _get_libero_family_spec("LIBERO_Kitchen_Tabletop_Manipulation")
    cam_pos, cam_quat = np.asarray(spec.agentview_pos_w), np.asarray(spec.agentview_quat_w)
    rotation = quat_to_R(cam_quat)
    np.testing.assert_allclose(rotation[:, 0], (0.0, 1.0, 0.0), atol=1e-6)  # camera x-axis = world +y
    marker = cam_pos - 1.2 * rotation[:, 2] + 0.25 * rotation[:, 0]  # 1.2 m ahead, 0.25 m toward world +y
    image_group = PiPolicy.observation_options(policy_setup="libero")["image"]
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                marker=SphereOptions(radius=0.05, default_root_pos=tuple(marker), is_fixed_base=True)
            ),
            cameras_options=CamerasOptions(
                agentview=CameraOptions(
                    cam_extrinsic=np.linalg.inv(trans_quat_to_T(cam_pos, cam_quat)),
                    cam_resolution=LIBERO_CAMERA_RESOLUTION,
                    cam_fov=45.0,
                    sensor_options=RasterizerCameraOptions(),
                )
            ),
            observation_options=ObservationManagerOptions(image=image_group),
        )
    )
    render = env.observation_manager.compute_group("image")

    def marker_column(frame: np.ndarray) -> float:
        brightness = frame.astype(np.float32).mean(axis=-1)
        return float(np.nonzero(np.abs(brightness - np.median(brightness)) > 30)[1].mean()) / frame.shape[1]

    assert marker_column(render[0].cpu().numpy()) > 0.5
    payload = _libero_payload({"image": render, "wrist_image": render, "proprio": torch.zeros(1, 9)})
    assert marker_column(payload["observation/image"]) < 0.5


@pytest.mark.parametrize(
    ("agentview", "eye_in_hand", "refused"),
    [((128, 128), (256, 256), "agentview"), ((256, 256), (128, 128), "eye_in_hand"), ((256, 256), (256, 256), None)],
    ids=["agentview-128", "eye_in_hand-128", "both-256"],
)
def test_libero_policy_refuses_cameras_not_rendering_at_openpis_256(agentview, eye_in_hand, refused):
    """The openpi ``examples/libero/main.py`` render is ``LIBERO_ENV_RESOLUTION = 256``; the stock task's 128x128 render
    would be upsampled off the training distribution.
    """
    from genesis.options.sensors import RasterizerCameraOptions

    import eden as en
    from eden.options import EnvOptions
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.managers.actions import ActionManagerOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots.libero import LiberoPanda
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    groups = PiPolicy.observation_options(policy_setup="libero")
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=0.002, decimation=25),  # robosuite's 20 Hz
            scene_options=en.scenes.Empty(robot=LiberoPanda()),
            cameras_options=CamerasOptions(
                **{
                    name: CameraOptions(cam_resolution=resolution, sensor_options=RasterizerCameraOptions())
                    for name, resolution in (("agentview", agentview), ("eye_in_hand", eye_in_hand))
                }
            ),
            observation_options=ObservationManagerOptions(**groups),
            action_options=ActionManagerOptions(**PiPolicy.action_options(policy_setup="libero")),
        )
    )
    if refused is None:
        with WebsocketPolicyServer("msgpack_openpi", lambda call: {"actions": np.zeros((5, 7), np.float32)}) as server:
            PiPolicy(env, PiPolicy.configure(policy_setup="libero", host=server.host, port=server.port)).close()
        return
    group = "image" if refused == "agentview" else "wrist_image"
    with pytest.raises(
        ValueError,
        match=rf"PiPolicy\.policy_setup='libero' sends camera '{refused}' \(observation term {group}/rgb\), which "
        r"renders at cam_resolution=\(128, 128\), but openpi's LIBERO checkpoints were trained on 256x256 renders "
        rf".* Fix: in the env's cameras_options set the '{refused}' camera's cam_resolution=\(256, 256\), e\.g\. "
        rf"cfg\.cameras_options\.{refused}\.cam_resolution = \(256, 256\)\.",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="libero"))


def test_libero_policy_refuses_an_env_without_the_libero_tasks_action_terms():
    env = _env("libero", "libero", actions={})
    with pytest.raises(
        ValueError,
        match=r"policy_setup='libero' drives action term 'osc_pose', but the env has no such term; its action terms "
        r"are \[\]\. Fix: build the env's action_options from PiPolicy\.action_options\(policy_setup='libero'\)",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="libero"))


def test_libero_state_is_robosuites_world_frame_grip_site_hand_quat_and_signed_fingers():
    """``[robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos]``: the grip site in the *world* frame (the base is
    offset, so a base-frame read would differ), the hand body's orientation, and robosuite's ``[+q1, -q2]``: the
    group reads Eden's two positive fingers and the payload negates the second.
    """
    import eden as en
    from eden.options import EnvOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots.libero import LiberoPanda
    from eden.utils.configs import EdenConfig
    from eden.utils.geom import transform_by_quat
    from tests.support.envs import build_env

    group = PiPolicy.observation_options(policy_setup="libero")["proprio"]
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=en.scenes.Empty(robot=LiberoPanda(default_root_pos=(0.4, -0.3, 0.2))),
            observation_options=ObservationManagerOptions(proprio=group),
        )
    )
    robot = env.entities["robot"]
    fingers = robot.find_named_dofs(list(LiberoPanda.gripper_dofs_name))
    pos = robot.get_dofs_pos().clone()
    pos[:, fingers.columns] = torch.tensor([0.031, 0.012])
    robot.set_dofs_pos(pos)

    proprio = env.observation_manager.compute_group("proprio")

    hand = robot.find_named_links_idx_local(LiberoPanda.ee_links_name)[1]
    hand_pos, hand_quat = robot.get_links_pos(ls_idx_local=hand)[:, 0], robot.get_links_quat(ls_idx_local=hand)[:, 0]
    grip_site = hand_pos + transform_by_quat(torch.tensor([LiberoPanda.grip_site_pos_local]), hand_quat)
    torch.testing.assert_close(proprio[:, :3], grip_site)
    torch.testing.assert_close(proprio[:, 3:7], hand_quat)
    torch.testing.assert_close(proprio[:, 7:], torch.tensor([[0.031, 0.012]]))
    assert proprio[0, 2].item() > 0.2  # world z: the base sits 0.2 m up

    payload = _libero_payload({"image": golden_image(), "wrist_image": golden_image(seed=1), "proprio": proprio})
    np.testing.assert_allclose(payload["observation/state"][:3], proprio[0, :3].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(payload["observation/state"][6:], [0.031, -0.012], rtol=0, atol=1e-6)


@pytest.mark.parametrize("setup", ["widowx_bridge", "google_robot"])
def test_widowx_state_is_simplerenvs_ee_gripper_link_beyond_the_finger_links(setup):
    """SimplerEnv's ``eef_pos`` is ``ee_gripper_link`` in the base frame: ``ee_gripper`` 0.027575 m past the finger
    links' origin along ``gripper_link``'s +x (Interbotix wx250s URDF), not ``gripper_link``, 9.4 cm behind it.
    """
    import eden as en
    from eden.options import EnvOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots.trossen_widowx250 import WidowX250
    from eden.utils.configs import EdenConfig
    from eden.utils.geom import subtract_frame_transforms, transform_by_quat
    from tests.support.envs import build_env

    group = PiPolicy.observation_options(policy_setup=setup)["proprio"]
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=en.scenes.Empty(robot=WidowX250(default_root_pos=(0.3, -0.2, 0.1))),
            observation_options=ObservationManagerOptions(proprio=group),
        )
    )
    robot = env.entities["robot"]
    arm = robot.find_named_dofs(list(WidowX250.arm_dofs_name))
    pos = robot.get_dofs_pos().clone()
    pos[:, arm.columns] = torch.tensor([0.3, -0.4, 0.5, 0.6, 0.7, -0.8])
    robot.set_dofs_pos(pos)

    proprio = env.observation_manager.compute_group("proprio")

    _, links = robot.find_named_links_idx_local(["base_link", "gripper_link", "left_finger_link", "right_finger_link"])
    link_pos, link_quat = robot.get_links_pos(ls_idx_local=links)[0], robot.get_links_quat(ls_idx_local=links)[0]
    gripper_x = transform_by_quat(torch.tensor([[1.0, 0.0, 0.0]]), link_quat[1:2])
    ee_gripper_link = (link_pos[2:3] + link_pos[3:4]) / 2 + 0.027575 * gripper_x
    expected_pos, expected_quat = subtract_frame_transforms(
        link_pos[0:1], link_quat[0:1], ee_gripper_link, link_quat[1:2]
    )
    torch.testing.assert_close(proprio[:, :3], expected_pos)
    torch.testing.assert_close(proprio[:, 3:7], expected_quat)


_WIDOWX_ENVS: dict = {}


@pytest.mark.parametrize(("finger", "openness"), [(0.037, 1.0), (0.026, 0.5), (0.015, 0.0), (0.012, 0.0)])
def test_widowx_gripper_state_is_simplerenvs_openness_over_the_urdf_finger_range(finger, openness):
    """SimplerEnv reports ``1 - get_gripper_closedness() = (q - 0.015) / (0.037 - 0.015)``, clamped (ManiSkill2_real2sim
    wx250s.urdf finger limits); Eden's fingers close further (0.01), so a finger past 0.015 reads 0, not ~0.27.
    """
    import eden as en
    from eden.options import EnvOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots.trossen_widowx250 import WidowX250
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    key = "widowx_gripper_state"
    if key not in _WIDOWX_ENVS:
        _WIDOWX_ENVS[key] = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=1),
                scene_options=en.scenes.Empty(robot=WidowX250()),
                observation_options=ObservationManagerOptions(
                    proprio=PiPolicy.observation_options(policy_setup="widowx_bridge")["proprio"]
                ),
            )
        )
    env = _WIDOWX_ENVS[key]
    robot = env.entities["robot"]
    pos = robot.get_dofs_pos().clone()
    pos[:, robot.find_named_dofs(["left_finger"]).columns] = finger
    robot.set_dofs_pos(pos)
    assert robot.get_dofs_pos()[0, robot.find_named_dofs(["left_finger"]).columns].item() == pytest.approx(finger)

    proprio = env.observation_manager.compute_group("proprio")
    assert proprio[0, 7].item() == pytest.approx(openness, abs=1e-5)


@pytest.mark.parametrize("setup", ["widowx_bridge", "google_robot"])
@pytest.mark.parametrize(
    ("arm", "error", "match"),
    [
        (
            "joint_pd",
            TypeError,
            "needs action term 'ee_controller' of type DifferentialIKController, but the env built "
            "ImplicitPDController. Fix: build the env's action_options from PiPolicy.action_options(policy_setup=",
        ),
        (
            None,
            ValueError,
            "drives action term 'ee_controller', but the env has no such term; its action terms are "
            "['gripper_controller']. Fix: build the env's action_options from PiPolicy.action_options(policy_setup=",
        ),
    ],
    ids=["joint-pd", "missing"],
)
def test_ee_delta_setups_refuse_an_ee_controller_that_is_not_differential_ik(setup, arm, error, match):
    """A 6-DOF joint PD would take the server's [dxyz, drot] as absolute joint targets, silently."""
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    actions = PiPolicy.action_options(policy_setup=setup)
    del actions["ee_controller"]
    if arm == "joint_pd":
        actions["ee_controller"] = ImplicitPDController.configure(entity_name="robot", dofs_name=list(WIDOWX_DOFS))
    env = _env("widowx", setup, actions=actions)
    with pytest.raises(error, match=re.escape(f"PiPolicy.policy_setup={setup!r} ") + ".*" + re.escape(match)):
        PiPolicy(env, PiPolicy.configure(policy_setup=setup))


@pytest.mark.parametrize(
    ("weights", "dropped"),
    [
        (dict(orientation_weight=0.0), "orientation_weight=0.0 (drops the server's rotation rows)"),
        (dict(position_weight=0.0), "position_weight=0.0 (drops the server's translation rows)"),
    ],
    ids=["orientation", "position"],
)
def test_ee_delta_setups_refuse_an_ee_controller_that_ignores_part_of_the_delta(weights, dropped):
    """``orientation_weight=0`` is the term's position-only mode: the server's rotation rows would do nothing."""
    actions = PiPolicy.action_options(policy_setup="widowx_bridge")
    actions["ee_controller"] = actions["ee_controller"].model_copy(update=weights)
    env = _env("widowx", "widowx_bridge", actions=actions)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "PiPolicy.policy_setup='widowx_bridge' sends a 6-D end-effector delta [dx, dy, dz, drx, dry, drz], but "
            f"action term 'ee_controller' has {dropped}, so the arm would ignore part of every action. Fix: build the "
            "env's action_options from PiPolicy.action_options(policy_setup='widowx_bridge') (position_weight=1.0, "
            "orientation_weight=1.0)."
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="widowx_bridge"))


@pytest.mark.parametrize(
    ("scales", "listed"),
    [
        (dict(position_scale=[0.5, 0.5, 0.25]), "position_scale=[0.5, 0.5, 0.25]"),
        (dict(orientation_scale=[0.5, 0.25, 0.5]), "orientation_scale=[0.5, 0.25, 0.5]"),
    ],
    ids=["position", "orientation"],
)
def test_ee_delta_setups_refuse_a_per_axis_ee_controller_scale(scales, listed):
    """The adapter rotates the base-frame delta into the world before the term scales it per world axis."""
    actions = PiPolicy.action_options(policy_setup="widowx_bridge")
    actions["ee_controller"] = actions["ee_controller"].model_copy(update=scales)
    env = _env("widowx", "widowx_bridge", actions=actions)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "PiPolicy.policy_setup='widowx_bridge' rotates the server's base-frame delta into the world frame before "
            f"action term 'ee_controller' scales it per world axis, but the term has {listed}, so on a rotated base "
            "each axis of the server's delta would be scaled by another axis's factor. Fix: build the env's "
            "action_options from PiPolicy.action_options(policy_setup='widowx_bridge') with one scalar scale"
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="widowx_bridge"))


@pytest.mark.parametrize("setup", ["widowx_bridge", "google_robot"])
def test_ee_delta_setups_move_the_point_the_state_reports(setup):
    """SimplerEnv's controller moves and turns ee_gripper_link, so the IK's control point is the proprio's point."""
    from eden.extensions.policies.pi.setups import SIMPLER_WIDOWX_EE_LINK, SIMPLER_WIDOWX_EE_OFFSET_POS

    arm = PiPolicy.action_options(policy_setup=setup)["ee_controller"]
    pose = PiPolicy.observation_options(policy_setup=setup)["proprio"].ee_pose
    driven = (arm.ee_link_name, tuple(arm.ee_local_pos))
    reported = (pose.ee_link_name, tuple(pose.offset_pos))
    assert driven == reported == (SIMPLER_WIDOWX_EE_LINK, SIMPLER_WIDOWX_EE_OFFSET_POS)


#: SimplerEnv's ee_gripper_link, the point the default options both report and move.
_EE_GRIPPER_LINK = "(0.093575, 0.0, 0.0) on 'robot' link 'gripper_link'"


@pytest.mark.parametrize(
    ("observation_kwargs", "action_kwargs", "reported", "driven"),
    [
        (
            {},
            dict(ee_offset_pos=(0.0, 0.0, 0.0)),
            _EE_GRIPPER_LINK,
            "(0.0, 0.0, 0.0) on 'robot' link 'gripper_link'",
        ),
        (
            {},
            dict(ee_link_name="left_finger_link"),
            _EE_GRIPPER_LINK,
            "(0.093575, 0.0, 0.0) on 'robot' link 'left_finger_link'",
        ),
        (
            dict(entity_name="plane", ee_link_name="plane_baselink", gripper_entity_name="robot"),
            {},
            "(0.093575, 0.0, 0.0) on 'plane' link 'plane_baselink'",
            _EE_GRIPPER_LINK,
        ),
    ],
    ids=["link-origin", "other-link", "other-entity"],
)
def test_ee_delta_setups_refuse_an_ee_controller_moving_another_point_than_the_state_reports(
    observation_kwargs, action_kwargs, reported, driven
):
    """The server's delta moves the point it was sent; on gripper_link's origin a rotation shifts it by drot x 9.4 cm."""
    env = _env("widowx", "widowx_bridge", action_kwargs=action_kwargs, observation_kwargs=observation_kwargs)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"PiPolicy.policy_setup='widowx_bridge' sends the pose of the point {reported} (proprio term 'ee_pose'), "
            f"the point the server's [dxyz, drot] moves, but action term 'ee_controller' moves the point {driven} "
            "(ee_local_pos), so the arm would apply every delta to another point than the server computed it for. "
            "Fix: pass the same entity_name, ee_link_name and ee_offset_pos to PiPolicy.observation_options("
            "policy_setup='widowx_bridge', ...) and PiPolicy.action_options(policy_setup='widowx_bridge', ...), or "
            "leave all three at their defaults (entity_name='robot', ee_link_name='gripper_link', "
            "ee_offset_pos=(0.093575, 0.0, 0.0): SimplerEnv's ee_gripper_link)."
        ),
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="widowx_bridge"))


# ---------------------------------------------------------------------------
# ALOHA: gym_aloha's normalised gripper position on both ends
# ---------------------------------------------------------------------------


def test_aloha_gripper_is_normalised_between_the_models_close_and_open_finger_positions():
    """State: (q - close) / (open - close) of the left finger; action: the inverse on both fingers."""
    import eden as en
    from eden.options import EnvOptions
    from eden.options.managers.actions import ActionManagerOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots.aloha_vx300s import Aloha
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    close, open_ = Aloha.close_dofs_pos["left/left_finger"], Aloha.open_dofs_pos["left/left_finger"]
    half = close + 0.5 * (open_ - close)
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=en.scenes.Empty(robot=Aloha()),
            observation_options=ObservationManagerOptions(
                proprio=PiPolicy.observation_options(policy_setup="aloha")["proprio"]
            ),
            action_options=ActionManagerOptions(**PiPolicy.action_options(policy_setup="aloha")),
        )
    )
    robot = env.entities["robot"]
    pos = robot.get_dofs_pos().clone()
    pos[:, robot.find_named_dofs(["left/left_finger", "right/left_finger"]).columns] = torch.tensor([half, open_])
    robot.set_dofs_pos(pos)
    proprio = env.observation_manager.compute_group("proprio")
    assert proprio.shape == (1, 14)
    assert (proprio[0, 6].item(), proprio[0, 13].item()) == (pytest.approx(0.5), pytest.approx(1.0))

    row = np.zeros(14, dtype=np.float32)
    row[6], row[13] = 0.25, 1.0
    with _policy("aloha", "aloha", np.repeat(row[None], 10, axis=0)) as (policy, _):
        action = policy.step({"image": golden_image(), "proprio": proprio})
    env.action_manager.compute(action)
    left = env.action_manager.terms["left_gripper_controller"].processed_action
    right = env.action_manager.terms["right_gripper_controller"].processed_action
    torch.testing.assert_close(left, torch.full((1, 2), close + 0.25 * (open_ - close)))
    torch.testing.assert_close(right, torch.full((1, 2), open_))


def test_aloha_executes_ten_rows_of_each_chunk_as_openpis_action_chunk_broker():
    """``examples/aloha_sim/main.py``: ``ActionChunkBroker(action_horizon=10)`` over pi0_aloha_sim's 50-row chunks."""
    rows = np.repeat(np.arange(50, dtype=np.float32)[:, None], 14, axis=1)
    obs = {"image": golden_image(), "proprio": torch.zeros(1, 14)}
    with _policy("aloha", "aloha", rows) as (policy, server):
        executed = [policy.step(obs)["left_arm_controller"][0, 0].item() for _ in range(10)]
        assert (executed, server.infer_calls) == (list(range(10)), 1)
        assert policy.step(obs)["left_arm_controller"][0, 0].item() == 0.0
        assert server.infer_calls == 2


@pytest.mark.parametrize(
    ("arm_override", "match"),
    [
        (dict(), r"reference_source='default', which adds a reference to every command"),
        (
            dict(reference_source=ReferenceSource.ZERO, scale=0.5),
            r"effective_scale=\[0\.5\] \(from scale=0\.5, scale_ratio=None, needs 1\)",
        ),
        (dict(reference_source=ReferenceSource.ZERO, offset=0.1), r"offset=0\.1 \(needs 0\)"),
    ],
    ids=["default-reference", "scale", "offset"],
)
def test_aloha_refuses_an_arm_term_that_would_not_take_the_servers_absolute_joint_targets(arm_override, match):
    """gym_aloha's actions are absolute joint angles: a default-pose reference, a scale or an offset moves the arm
    somewhere else without any error.
    """
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    actions = PiPolicy.action_options(policy_setup="aloha")
    right_arm = actions["right_arm_controller"]
    actions["right_arm_controller"] = ImplicitPDController.configure(
        entity_name="robot", dofs_name=list(right_arm.dofs_name), **arm_override
    )
    env = _env("aloha", "aloha", actions=actions)
    with pytest.raises(
        ValueError,
        match=r"PiPolicy\.policy_setup='aloha' sends absolute joint targets to 'right_arm_controller' to be executed as "
        r"they are, but that ImplicitPDController has " + match + r".*Fix: build the env's action_options from "
        r"PiPolicy\.action_options\(policy_setup='aloha'\)\. Do not pre-scale the server's rows",
    ):
        PiPolicy(env, PiPolicy.configure(policy_setup="aloha"))


# ---------------------------------------------------------------------------
# Chunk scheduling: the DROID joint-position setup forwards each row's arm targets unchanged, so the
# executed row of a ramp chunk (row i = [i] * 7 + [gripper]) is readable straight off the controller dict.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _ramp_policy(chunk_length: int, pred_action_horizon: int = 15, *, gripper=lambda call: 0.0, **configure):
    def reply(call: int) -> dict:
        rows = np.repeat(np.arange(chunk_length, dtype=np.float32)[:, None], 8, axis=1)
        rows[:, 7] = gripper(call)
        return {"actions": rows}

    with WebsocketPolicyServer("msgpack_openpi", reply) as server:
        env = _env("droid", "droid_jointpos", action_kwargs=_DROID_RIG)
        options = PiPolicy.configure(
            policy_setup="droid_jointpos",
            host=server.host,
            port=server.port,
            use_wrist_image=True,
            task_description=GOLDEN_TASK,
            pred_action_horizon=pred_action_horizon,
            exec_horizon=8,
            **configure,
        )
        policy = PiPolicy(env, options)
        try:
            yield policy, server
        finally:
            policy.close()


def _executed_row(policy: PiPolicy) -> float:
    return float(policy.step(_DROID_OBS)["ee_controller"][0, 0])


def test_pi_policy_executes_first_eight_of_fifteen_before_next_inference():
    with _ramp_policy(15) as (policy, server):
        assert [_executed_row(policy) for _ in range(8)] == list(range(8))
        assert server.infer_calls == 1
        assert policy.last_prediction_chunk_length == 15

        assert _executed_row(policy) == 0
        assert server.infer_calls == 2


def test_pi_policy_temporal_ensembles_old_tail_into_new_head():
    with _ramp_policy(15, chunk_temporal_ensemble_steps=2, chunk_temporal_ensemble_dims=1) as (policy, _):
        for _ in range(8):
            policy.step(_DROID_OBS)

        # Old action 8 and new action 0 receive weights 2/3 and 1/3.
        assert _executed_row(policy) == pytest.approx(16.0 / 3.0)
        # Old action 9 and new action 1 receive weights 1/3 and 2/3.
        assert _executed_row(policy) == pytest.approx(11.0 / 3.0)
        # The remainder of the new chunk is unchanged.
        assert _executed_row(policy) == 2
        assert policy.last_temporal_ensemble_steps == 2


def test_pi_policy_default_does_not_temporal_ensemble():
    with _ramp_policy(15) as (policy, _):
        for _ in range(8):
            policy.step(_DROID_OBS)

        assert _executed_row(policy) == 0
        assert policy.last_temporal_ensemble_steps == 0


def test_pi_policy_temporal_ensemble_preserves_new_gripper_channel():
    """Only the first ``chunk_temporal_ensemble_dims`` columns blend; the new chunk's gripper wins outright."""
    close_on_second_call = lambda call: 0.0 if call == 1 else 1.0  # noqa: E731
    with _ramp_policy(
        15, chunk_temporal_ensemble_steps=2, chunk_temporal_ensemble_dims=7, gripper=close_on_second_call
    ) as (policy, _):
        for _ in range(8):
            policy.step(_DROID_OBS)
        action = policy.step(_DROID_OBS)

    torch.testing.assert_close(action["ee_controller"], torch.full((1, 7), 16.0 / 3.0))
    assert action["gripper_controller"].item() == 1.0


def test_pi_policy_rejects_truncated_prediction_chunk():
    with _ramp_policy(14) as (policy, _), pytest.raises(RuntimeError, match="pred_action_horizon=15"):
        policy.step(_DROID_OBS)


def test_pi0_fast_policy_accepts_ten_and_executes_first_eight():
    with _ramp_policy(10, pred_action_horizon=10) as (policy, server):
        assert [_executed_row(policy) for _ in range(8)] == list(range(8))
        assert server.infer_calls == 1
        assert policy.last_prediction_chunk_length == 10

        assert _executed_row(policy) == 0
        assert server.infer_calls == 2


def test_a_chunk_of_the_wrong_width_is_rejected_before_it_is_queued():
    """A WidowX-bridge client facing a server that returns 6-D rows (another checkpoint) fails on the first chunk."""
    with WebsocketPolicyServer("msgpack_openpi", lambda call: {"actions": np.zeros((5, 6), np.float32)}) as server:
        env = _env("widowx", "widowx_bridge")
        options = PiPolicy.configure(
            policy_setup="widowx_bridge", host=server.host, port=server.port, task_description=GOLDEN_TASK
        )
        policy = PiPolicy(env, options)
        try:
            with pytest.raises(ValueError, match="expects 7-D action rows"):
                policy.step({"image": golden_image(), "proprio": ee8(1.0)})
        finally:
            policy.close()


# ---------------------------------------------------------------------------
# Golden records: wire contract and controller outputs per setup (see the policies conftest)
# ---------------------------------------------------------------------------

#: name -> (robot, setup, action_options kwargs, policy configure kwargs, chunk (t, d, gripper_col), steps, obs)
_GOLDEN_CASES = {
    "pi_widowx_bridge": (
        "widowx",
        "widowx_bridge",
        {},
        dict(use_wrist_image=True),
        (5, 7, 6),
        6,
        lambda s: {
            "image": golden_image(seed=s),
            "proprio": ee8(1.0 - 0.2 * s),
            "wrist_image": golden_image(seed=100 + s),
        },
    ),
    "pi_google_robot": (
        "widowx",
        "google_robot",
        {},
        {},
        (5, 7, 6),
        6,
        lambda s: {"image": golden_image(seed=s), "proprio": ee8(1.0 - 0.2 * s)},
    ),
    # The libero task's own osc_pose + gripper terms; LiberoInputs reads the wrist image on every request.
    "pi_libero": (
        "libero",
        "libero",
        {},
        {},
        (5, 7, 6),
        6,
        lambda s: {
            "image": golden_image(seed=s),
            "wrist_image": golden_image(seed=30 + s),
            "proprio": vec(0.3, -0.1, 0.2, *TILT_QUAT, 0.02, 0.02),
        },
    ),
    # openpi DROID servers return 10-row chunks; the client executes open_loop_horizon=8. Eden's Droid rig
    # carries the Robotiq on its own ``gripper`` entity. One record covers DROID's payload, which every arm mode
    # shares; each mode's actions are pinned by the droid tests above, chunk ensembling by the ramp tests.
    "pi_droid_velocity": (
        "droid",
        "droid",
        dict(gripper_entity_name="gripper"),
        dict(use_wrist_image=True),
        (10, 8, 7),
        10,
        lambda s: {"image": golden_image(seed=s), "wrist_image": golden_image(seed=50 + s), "proprio": joint8(0.1 * s)},
    ),
    "pi_aloha": (
        "aloha",
        "aloha",
        {},
        dict(use_wrist_image=True),
        # pi0_aloha_sim returns 50-row chunks; 12 steps execute 10 of the first and 2 of the second.
        (50, 14, None),
        12,
        lambda s: {
            "image": golden_image(seed=s),
            "wrist_image": golden_image(seed=70 + s),
            "proprio": torch.linspace(-0.5, 0.5, 14).unsqueeze(0),
        },
    ),
}


@pytest.mark.parametrize("name", list(_GOLDEN_CASES))
def test_wire_contract_and_controller_outputs_match_the_golden_record(name):
    robot, setup, action_kwargs, configure, (t, d, gripper_col), steps, obs_fn = _GOLDEN_CASES[name]
    reply = lambda call: {"actions": chunk(t, d, call, gripper_col=gripper_col)}  # noqa: E731
    with WebsocketPolicyServer("msgpack_openpi", reply) as server:
        env = _env(robot, setup, action_kwargs=action_kwargs, use_wrist_image=configure.get("use_wrist_image"))
        options = PiPolicy.configure(policy_setup=setup, host=server.host, port=server.port, **configure)
        policy = PiPolicy(env, options)
        try:
            policy.set_task(GOLDEN_TASK)
            got = record_golden(policy, server, obs_fn, steps)
        finally:
            policy.close()
    assert_matches_golden(name, got)


def test_the_client_declares_exactly_its_official_servers_reset_api():
    # The OpenPI server takes the prompt with every request and has no reset.
    options = PiPolicy.configure()
    assert (options.set_task_endpoint, options.reset_endpoint) == (None, None)
