"""``MolmoAct2Policy`` (DROID, ``json_numpy`` HTTP) against a local fake server."""

from __future__ import annotations

import json_numpy
import numpy as np
import pytest
import torch

from eden.constants import ReferenceSource
from eden.extensions.policies.adapters import DROID_ARM_DOFS
from eden.extensions.policies.molmoact2.policy import MolmoAct2Policy
from tests.extensions.policies.conftest import (
    GOLDEN_TASK,
    assert_matches_golden,
    chunk,
    joint8,
    record_golden,
)
from tests.extensions.policies.conftest import image as golden_image
from tests.support.policy_servers import HttpPolicyServer

_ACTION_OPTIONS = dict(gripper_entity_name="gripper")  # Eden's Droid rig: the Robotiq is its own entity
#: MolmoAct2-DROID's chunk length (norm_stats.json ``franka_droid``: action_horizon 15).
_HORIZON = 15
_ENVS: dict = {}


def _env(
    observation_options: dict | None = None,
    action_options: dict | None = None,
    sim_dt: float = 1.0 / 15.0,
    scene_entities: dict | None = None,
    **action_kwargs,
):
    """Eden's Droid rig with two tiny rasterizer cameras and MolmoAct2's observation and action terms (or the given
    ones), stepping every ``sim_dt`` (DROID's 15 Hz by default), plus any ``scene_entities``; cached per
    configuration.
    """
    import eden as en
    from genesis.options.sensors import RasterizerCameraOptions

    from eden.options import EnvOptions
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.managers.actions import ActionManagerOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots import Droid
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    if observation_options is None:
        observation_options = MolmoAct2Policy.observation_options(**_ACTION_OPTIONS)
    if action_options is None:
        action_options = MolmoAct2Policy.action_options(**_ACTION_OPTIONS, **action_kwargs)
    scene_entities = scene_entities or {}
    key = repr(
        [
            {name: options.model_dump() for name, options in o.items()}
            for o in (observation_options, action_options, scene_entities)
        ]
        + [sim_dt]
    )
    if key not in _ENVS:
        camera = CameraOptions(
            cam_resolution=(16, 12),
            sensor_options=RasterizerCameraOptions(),
            cam_pos=(1.0, 0.0, 1.0),
            cam_lookat=(0.0, 0.0, 0.3),
        )
        _ENVS[key] = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=1, sim_dt=sim_dt, decimation=1),
                scene_options=en.scenes.Empty(robot=Droid(), **scene_entities),
                cameras_options=CamerasOptions(camera=camera, wrist_camera=camera),
                observation_options=ObservationManagerOptions(**observation_options),
                action_options=ActionManagerOptions(**action_options),
            )
        )
    return _ENVS[key]


def test_wire_contract_and_controller_outputs_match_the_golden_record():
    reply = lambda call, record: json_numpy.dumps({"actions": chunk(_HORIZON, 8, call, gripper_col=7), "dt_ms": 1.0})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            obs_fn = lambda s: {  # noqa: E731
                "image": {"external": golden_image(seed=s), "wrist": golden_image(seed=9 + s)},
                "proprio": joint8(0.2),
            }
            # One step past a whole chunk: the 16th step re-queries the server.
            got = record_golden(policy, server, obs_fn, steps=_HORIZON + 1)
        finally:
            policy.close()
    assert_matches_golden("molmoact2", got)


def test_molmoact2_absolute_targets_do_not_add_default_pose_offset():
    actions = MolmoAct2Policy.action_options(gripper_entity_name="gripper")

    assert actions["arm_controller"].reference_source == ReferenceSource.ZERO
    assert actions["gripper_controller"].name == "gripper_decision_controller"
    assert actions["gripper_controller"].input == "closedness"
    assert actions["gripper_controller"].open_action == 0.0


def _step_env_with_gripper(gripper: float, **settings):
    """One policy step on rows whose gripper column is ``gripper``, applied through the env's action manager.

    Returns the policy and the env's gripper term; the env is reset afterwards.
    """
    rows = np.zeros((_HORIZON, 8), dtype=np.float32)
    rows[:, 7] = gripper
    reply = lambda call, record: json_numpy.dumps({"actions": rows})  # noqa: E731
    env = _env(**settings)
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(env, MolmoAct2Policy.configure(host=server.host, port=server.port, **settings))
        try:
            policy.set_task(GOLDEN_TASK)
            env.step(policy.step(_observation()))
        finally:
            policy.close()
    term = env.action_manager.terms["gripper_controller"]
    target, decision = term.processed_action.clone(), term.last_decision
    env.reset()
    return policy, target, decision


def test_molmoact2_continuous_gripper_drives_the_fingers_by_the_raw_closedness():
    """With ``binarize_gripper=False`` the env's gripper term targets closedness x close position, not a 0/1 snap."""
    policy, target, _ = _step_env_with_gripper(0.37, binarize_gripper=False)

    # 0.37 of the 0.8 rad close position on both Robotiq drivers; binarized at 0.5 it would be 0 (open).
    torch.testing.assert_close(target, torch.full_like(target, 0.37 * 0.8))
    assert policy.last_diagnostics["gripper_intent"] == "continuous"


def test_gripper_intent_diagnostic_is_the_env_terms_decision_with_a_flipped_threshold():
    """``flip_gripper`` sends openness, which the term opens on strictly above the threshold: 0.5 > 0.3 opens."""
    policy, _, decision = _step_env_with_gripper(0.5, flip_gripper=True, gripper_threshold=0.3)

    assert policy.last_diagnostics["gripper_intent"] == decision["intent"] == "open"


def test_a_real_observation_reaches_the_server_as_droids_joint_state_and_two_images():
    """The env's own groups, packed: state = the arm joints + the Robotiq's closedness, images = the two cameras."""
    rows = np.zeros((_HORIZON, 8), dtype=np.float32)
    reply = lambda call, record: json_numpy.dumps({"actions": rows})  # noqa: E731
    env = _env()
    robot, gripper = env.entities["robot"], env.entities["gripper"]
    arm = robot.find_named_dofs(list(MolmoAct2Policy.action_options(**_ACTION_OPTIONS)["arm_controller"].dofs_name))
    pos = robot.get_dofs_pos().clone()
    pos[:, arm.columns] = torch.linspace(-0.3, 0.3, 7)
    robot.set_dofs_pos(pos)
    driver = gripper.find_named_dofs(["left_driver_joint"])
    pos = gripper.get_dofs_pos().clone()
    pos[:, driver.columns] = 0.4  # half of the 0.8 rad close position
    gripper.set_dofs_pos(pos)
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(env, MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            policy.step(env.observation_manager.compute())
        finally:
            policy.close()
    sent = server.records[0]["json"]
    np.testing.assert_allclose(sent["state"][:7], np.linspace(-0.3, 0.3, 7), atol=1e-5)
    np.testing.assert_allclose(sent["state"][7], 0.5, atol=1e-5)
    assert sent["external_cam"].shape == sent["wrist_cam"].shape == (12, 16, 3)
    assert sent["external_cam"].dtype == np.uint8


def _observation(seed: int = 0) -> dict:
    return {
        "image": {"external": golden_image(seed=seed), "wrist": golden_image(seed=9 + seed)},
        "proprio": joint8(0.2),
    }


def test_a_chunk_shorter_than_the_checkpoints_15_rows_is_an_error():
    """MolmoAct2-DROID always returns 15 rows; a truncated chunk must not run open-loop as if it were whole."""
    reply = lambda call, record: json_numpy.dumps({"actions": np.zeros((10, 8), dtype=np.float32)})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            with pytest.raises(RuntimeError, match=r"returned 10 actions but pred_action_horizon=15"):
                policy.step(_observation())
        finally:
            policy.close()


def test_a_request_without_a_task_raises_before_anything_is_sent():
    """There is no default instruction: a forgotten ``set_task`` must not reach the server as a generic prompt."""
    reply = lambda call, record: json_numpy.dumps({"actions": np.zeros((_HORIZON, 8), dtype=np.float32)})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            with pytest.raises(
                ValueError,
                match=r"task_description is None\. Fix: policy\.set_task\('<instruction>'\).*"
                r"MolmoAct2Policy\.configure\(task_description='<instruction>'\)",
            ):
                policy.step(_observation())
        finally:
            policy.close()
    assert server.records == []


@pytest.mark.parametrize(
    ("sim_dt", "env_dt", "speed"), [(0.02, r"0\.020000", r"3\.33x too fast"), (0.1, r"0\.100000", r"1\.50x too slow")]
)
def test_construction_refuses_an_env_that_does_not_step_at_droids_15_hz(sim_dt, env_dt, speed):
    """Each row is a waypoint 1/15 s after the last: at 50 Hz a chunk plays 3.33x too fast, at 10 Hz 1.5x too slow."""
    with pytest.raises(
        ValueError,
        match=rf"^MolmoAct2Policy executes one action row per 0\.066667 s \(15 Hz\) control step, as DROID's RobotEnv "
        rf"does \(control_hz=15\) in the data MolmoAct2-DROID was trained on, but the env steps every env\.dt={env_dt} "
        rf"s \(sim_dt \* decimation\), which would run each chunk {speed}\. Fix: build the env with sim_dt \* "
        r"decimation = 1/15 s, e\.g\. EnvOptions\(sim_dt=1/120, decimation=8\) as PolaRiS does, with "
        r"sim_substeps=9 .*\. Do not rescale or resample the actions instead: .*$",
    ):
        MolmoAct2Policy(_env(sim_dt=sim_dt), MolmoAct2Policy.configure())


def _droid_arm(**overrides):
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    arm = MolmoAct2Policy.action_options(**_ACTION_OPTIONS)["arm_controller"]
    return ImplicitPDController.configure(**{**arm.model_dump(include={"entity_name", "dofs_name"}), **overrides})


@pytest.mark.parametrize(
    ("arm_key", "arm_overrides", "error", "match"),
    [
        (
            "ee_controller",
            {},
            ValueError,
            r"to action term 'arm_controller', which must be exactly an ImplicitPDController .* but the env's "
            r"'arm_controller' is missing \(action terms: \['ee_controller', 'gripper_controller'\]\)\. Fix: build the "
            r"env's action_options from MolmoAct2Policy\.action_options\(\)",
        ),
        ("arm_controller", {}, ValueError, r"has reference_source='default', which adds a reference to every command"),
        (
            "arm_controller",
            dict(reference_source=ReferenceSource.ZERO, scale=0.5),
            ValueError,
            r"has effective_scale=\[0\.5\] \(from scale=0\.5, scale_ratio=None, needs 1\)",
        ),
        (
            "arm_controller",
            dict(reference_source=ReferenceSource.ZERO, dofs_name=list(reversed(DROID_ARM_DOFS))),
            ValueError,
            r"^MolmoAct2Policy sends the server's absolute joint targets to 'arm_controller' to be executed as they "
            r"are, but that ImplicitPDController has dofs_name=\['joint7', .*\] \(needs \['joint1', .*'joint7'\], in "
            r"that order\)\. Fix: build the env's action_options from MolmoAct2Policy\.action_options\(\), passing the "
            r"rig's arm / gripper names as keyword arguments\. Do not undo the difference in the client \(subtracting "
            r"the reference, dividing by the scale or reordering the rows\): the term must execute the reply as is\.$",
        ),
    ],
)
def test_construction_refuses_an_arm_term_that_would_not_apply_absolute_targets_as_sent(
    arm_key, arm_overrides, error, match
):
    actions = MolmoAct2Policy.action_options(**_ACTION_OPTIONS)
    actions.pop("arm_controller")
    env = _env(action_options={arm_key: _droid_arm(**arm_overrides), **actions})
    with pytest.raises(error, match=match):
        MolmoAct2Policy(env, MolmoAct2Policy.configure())


def _construct(env) -> None:
    """Construct the client on ``env`` (its construction checks pass) and close it."""
    reply = lambda call, record: json_numpy.dumps({"actions": np.zeros((_HORIZON, 8), dtype=np.float32)})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        MolmoAct2Policy(env, MolmoAct2Policy.configure(host=server.host, port=server.port)).close()


def test_an_arm_term_with_an_all_zero_per_joint_offset_is_accepted():
    """``offset`` may be a per-joint dict; one of zeros is the identity, not an offset."""
    actions = MolmoAct2Policy.action_options(**_ACTION_OPTIONS)
    actions["arm_controller"] = _droid_arm(
        reference_source=ReferenceSource.ZERO, offset=dict.fromkeys(DROID_ARM_DOFS, 0.0)
    )
    _construct(_env(action_options=actions))


def test_an_arm_term_offset_set_after_build_is_refused():
    """A ``set_dofs_pos_offset()`` (joint-zero domain randomization) shifts every absolute target the server sends."""
    env = _env()
    arm = env.action_manager.terms["arm_controller"]
    arm.set_dofs_pos_offset(torch.full((env.num_envs, len(DROID_ARM_DOFS)), 0.1, device=env.device))
    try:
        with pytest.raises(
            ValueError,
            match=r"'arm_controller' to be executed as they are, but that ImplicitPDController has effective_offset "
            r"up to 0\.1 in magnitude, .*from a set_dofs_pos_offset\(\) call \(needs 0\)",
        ):
            MolmoAct2Policy(env, MolmoAct2Policy.configure())
    finally:
        # The env is cached and shared with the other tests.
        arm.set_dofs_pos_offset(torch.zeros(env.num_envs, len(DROID_ARM_DOFS), device=env.device))


def test_a_proprio_naming_droids_joints_by_pattern_is_accepted():
    """The joint-state check compares the joints the ``DofsPos`` reads, resolved, not the names it was given."""
    _construct(
        _env(observation_options=MolmoAct2Policy.observation_options(arm_dofs_name=["joint[1-7]"], **_ACTION_OPTIONS))
    )


def test_construction_refuses_an_end_effector_pose_proprio_of_the_same_width():
    """The EE-delta clients' 8-D ``[ee_pos, ee_quat, openness]`` group would pass a width check as joint state."""
    from eden.extensions.policies.adapters import make_ee_pose_group

    groups = MolmoAct2Policy.observation_options(**_ACTION_OPTIONS)
    groups["proprio"] = make_ee_pose_group(
        "robot", "attachment", gripper_entity_name="gripper", gripper_dof_name="left_driver_joint"
    )
    with pytest.raises(
        ValueError,
        match=r"'proprio' group does not match: it holds the terms \['ee_pose', 'gripper'\], not \['joint_pos', "
        r"'gripper'\] in that order\. Fix: build the env's 'proprio' group from MolmoAct2Policy\.observation_options\("
        r"arm_entity_name='robot', arm_dofs_name=\['joint1', .*\], \.\.\.\)",
    ):
        MolmoAct2Policy(_env(observation_options=groups), MolmoAct2Policy.configure())


def test_construction_refuses_a_joint_state_in_another_order_than_the_arm_term():
    groups = MolmoAct2Policy.observation_options(arm_dofs_name=list(reversed(DROID_ARM_DOFS)), **_ACTION_OPTIONS)
    with pytest.raises(
        ValueError,
        match=r"'joint_pos' reads 'robot' \['joint7', .*\] but the arm term drives 'robot' \['joint1', .*\], so each "
        r"joint would be sent another joint's command\. Fix: .* Do not reorder the server's rows in the client",
    ):
        MolmoAct2Policy(_env(observation_options=groups), MolmoAct2Policy.configure())


def test_droid_proprio_is_closedness_even_with_the_gripper_on_the_arm_entity():
    """DROID's ``gripper_position`` is closedness in [0, 1]; never the raw driver angle."""
    gripper = MolmoAct2Policy.observation_options(gripper_entity_name=None, gripper_close_pos=0.8)["proprio"].gripper
    assert gripper.normalize == "closedness"
    assert gripper.entity_name == "robot" and gripper.close_pos == 0.8


def test_the_client_declares_exactly_its_official_servers_reset_api():
    # The MolmoAct2 server takes the instruction with every request and has no reset.
    options = MolmoAct2Policy.configure()
    assert (options.set_task_endpoint, options.reset_endpoint) == (None, None)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_arm_target_in_a_late_row_refuses_the_whole_chunk_before_its_first_row(bad):
    rows = np.zeros((_HORIZON, 8), dtype=np.float32)
    rows[9, 3] = bad  # rows 0-8 are valid: none of them may drive the arm either
    reply = lambda call, record: json_numpy.dumps({"actions": rows})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            with pytest.raises(
                ValueError,
                match=rf"non-finite arm joint targets for env 0 in row 9 of its 15-row chunk: \[0\.0, 0\.0, 0\.0, "
                rf"{bad}, 0\.0, 0\.0, 0\.0, 0\.0\].*Fix: check the served checkpoint.*Do not replace the bad values",
            ):
                policy.step(_observation())  # the first step: row 0 is never handed to the env
            assert policy._action_queues == ((),)
        finally:
            policy.close()


def test_a_non_finite_arm_target_past_the_15_rows_that_run_still_refuses_the_chunk():
    """A reply longer than the checkpoint's 15 rows runs only its first 15, but a NaN anywhere means broken output."""
    rows = np.zeros((_HORIZON + 1, 8), dtype=np.float32)
    rows[_HORIZON, 0] = np.nan
    reply = lambda call, record: json_numpy.dumps({"actions": rows})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            with pytest.raises(
                ValueError, match=r"non-finite arm joint targets for env 0 in row 15 of its 16-row chunk"
            ):
                policy.step(_observation())
        finally:
            policy.close()


@pytest.mark.parametrize(
    ("obs", "match"),
    [
        (
            {"image": _observation()["image"]},
            r"reads the observation groups \['image', 'proprio'\], but obs has no \['proprio'\] \(its groups are "
            r"\['image'\]\)\. Fix: .* Do not fill a missing group with zeros",
        ),
        (
            {**_observation(), "image": {"external": _observation()["image"]["external"]}},
            r"sends obs\['image'\]\['external'\] and obs\['image'\]\['wrist'\] as its two views, but obs\['image'\] "
            r"holds \['external'\]\. Fix: .*observation_options\(external_camera=\.\.\., wrist_camera=\.\.\.\).* Do not "
            r"send one camera as both views",
        ),
    ],
    ids=["no_proprio", "no_wrist"],
)
def test_a_hand_built_observation_missing_a_group_or_camera_is_refused_with_the_fix(obs, match):
    reply = lambda call, record: json_numpy.dumps({"actions": np.zeros((_HORIZON, 8), dtype=np.float32)})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            with pytest.raises(KeyError, match=match):
                policy.step(obs)
        finally:
            policy.close()
    assert server.records == []


def test_a_nan_gripper_alone_is_a_missing_command_the_env_holds():
    rows = np.zeros((_HORIZON, 8), dtype=np.float32)
    rows[:, 7] = np.nan
    reply = lambda call, record: json_numpy.dumps({"actions": rows})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        policy = MolmoAct2Policy(_env(), MolmoAct2Policy.configure(host=server.host, port=server.port))
        try:
            policy.set_task(GOLDEN_TASK)
            action = policy.step(_observation())
        finally:
            policy.close()
    assert torch.isnan(action["gripper_controller"]).all()
    assert policy.last_diagnostics["gripper_intent"] == "hold"


@pytest.mark.parametrize(
    ("group_settings", "joint_settings", "problem"),
    [
        (dict(history_length=2), {}, r"history_length=2"),
        ({}, dict(scale=0.5), r"joint_pos\.scale=0\.5"),
    ],
)
def test_construction_refuses_a_proprio_that_is_not_the_plain_current_state(group_settings, joint_settings, problem):
    """A history puts older frames in the 8 columns the server reads as q1..q7 + closedness; a scale rescales them."""
    from eden.options.managers.observations import ObservationGroupOptions

    groups = MolmoAct2Policy.observation_options(**_ACTION_OPTIONS)
    proprio = groups["proprio"]
    groups["proprio"] = ObservationGroupOptions(
        **group_settings,
        joint_pos=proprio.joint_pos.model_copy(update=joint_settings),
        gripper=proprio.gripper,
    )
    with pytest.raises(
        ValueError,
        match=rf"Observation group 'proprio' is not the plain current reading .*{problem}.*Fix: build 'proprio' with "
        r"MolmoAct2Policy\.observation_options\(\.\.\.\)\['proprio'\]",
    ):
        MolmoAct2Policy(_env(observation_options=groups), MolmoAct2Policy.configure())


def test_construction_refuses_a_joint_state_read_from_another_entity():
    """Same DOF names, other arm: the state would describe a robot the actions do not move."""
    from eden.options.robots.franka_emika_panda import FrankaEmikaPandaNoHand

    groups = MolmoAct2Policy.observation_options(arm_entity_name="spare_arm", **_ACTION_OPTIONS)
    env = _env(
        observation_options=groups,
        scene_entities=dict(spare_arm=FrankaEmikaPandaNoHand(default_root_pos=(0.0, 1.0, 0.0))),
    )
    with pytest.raises(
        ValueError,
        match=r"'joint_pos' reads 'spare_arm' \[.*\] but the arm term drives 'robot' \[.*\], so each joint would be "
        r"sent another joint's command",
    ):
        MolmoAct2Policy(env, MolmoAct2Policy.configure())


def test_construction_refuses_a_gripper_state_normalised_over_another_close_position():
    """``gripper_close_pos`` passed to observation_options() only would skew the closedness in every request."""
    groups = MolmoAct2Policy.observation_options(gripper_close_pos=0.7, **_ACTION_OPTIONS)
    with pytest.raises(
        ValueError,
        match=r"Gripper state proprio/gripper and action term 'gripper_controller' describe different grippers: "
        r"close_pos=0\.7 but 'gripper_controller' has close_action=0\.8\. Fix: pass the same gripper entity",
    ):
        MolmoAct2Policy(_env(observation_options=groups), MolmoAct2Policy.configure())


def test_construction_refuses_a_concatenated_image_group():
    """The request carries the two cameras as separate frames, read as obs['image']['external'] / ['wrist']."""
    from eden.options.managers.observations import ObservationGroupOptions

    groups = MolmoAct2Policy.observation_options(**_ACTION_OPTIONS)
    image = groups["image"]
    groups["image"] = ObservationGroupOptions(concatenate_terms=True, external=image.external, wrist=image.wrist)
    with pytest.raises(ValueError, match=r"'image' has concatenate_terms=True, which fuses the two cameras"):
        MolmoAct2Policy(_env(observation_options=groups), MolmoAct2Policy.configure())
