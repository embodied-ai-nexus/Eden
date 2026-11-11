"""``Cosmos3Policy`` (DROID world action model over an OpenPI-style WebSocket) against a local fake server."""

from __future__ import annotations

import numpy as np
import pytest

from eden.constants import ReferenceSource
from eden.extensions.policies.adapters import DROID_ARM_DOFS
from eden.extensions.policies.cosmos3.policy import Cosmos3Policy
from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.utils.image import prepare_image_padded
from eden.utils.misc import to_snake_case
from tests.extensions.policies.conftest import (
    GOLDEN_TASK,
    assert_matches_golden,
    chunk,
    joint8,
    record_golden,
)
from tests.extensions.policies.conftest import image as golden_image
from tests.support.policy_servers import WebsocketPolicyServer

_GRIPPER = dict(gripper_entity_name="gripper")  # Eden's Droid rig: the Robotiq is its own entity
#: The scene's cameras; which of them the env renders is up to its observation groups.
_CAMERA_POSES = {
    "camera": (1.2, 0.6, 0.9),
    "wrist_camera": (0.6, 0.0, 1.2),
    "camera_2": (1.2, -0.6, 0.9),
}


def _camera(pos):
    """A small rasterizer sensor camera: renders a batched ``(num_envs, H, W, 3)`` uint8 frame."""
    from genesis.options.sensors import RasterizerCameraOptions

    from eden.options.camera import CameraOptions

    return CameraOptions(
        cam_pos=pos, cam_lookat=(0.5, 0.0, 0.3), cam_resolution=(16, 12), sensor_options=RasterizerCameraOptions()
    )


@pytest.fixture(scope="module")
def renderable():
    """Skip the module's camera envs on a machine without offscreen rendering (as tests/entities/test_camera.py)."""
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CamerasOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(floor=PlaneOptions()),
            cameras_options=CamerasOptions(probe=_camera((1.0, 0.0, 1.0))),
        )
    )
    try:
        env.cameras["probe"].render_rgb()
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        pytest.skip(f"offscreen rendering unavailable: {e}")
    finally:
        env.destroy()


_ENVS: dict = {}


def _env(observation_options: dict | None = None, action_options: dict | None = None, sim_dt: float = 1.0 / 15.0):
    """Eden's Droid rig with the three cameras and Cosmos3's single-exterior observation and action terms (or the
    given ones), stepping every ``sim_dt`` (DROID's 15 Hz by default); cached per configuration.

    Keyed on the dumped options: an ``ObservationGroupOptions`` repr omits its (dynamic) terms.
    """
    if observation_options is None:
        observation_options = Cosmos3Policy.observation_options(**_GRIPPER)
    if action_options is None:
        action_options = Cosmos3Policy.action_options(**_GRIPPER)
    key = repr(
        [{name: options.model_dump() for name, options in o.items()} for o in (observation_options, action_options)]
        + [sim_dt]
    )
    if key not in _ENVS:
        _ENVS[key] = _build_env(observation_options, action_options, sim_dt)
    return _ENVS[key]


def _build_env(observation_options: dict, action_options: dict, sim_dt: float):
    """A new env of :func:`_env`'s layout, for a test that changes its state (the caller destroys it)."""
    import eden as en
    from eden.options import EnvOptions
    from eden.options.camera import CamerasOptions
    from eden.options.managers.actions import ActionManagerOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.options.robots import Droid
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=sim_dt, decimation=1),
            scene_options=en.scenes.Empty(robot=Droid()),
            cameras_options=CamerasOptions(**{name: _camera(pos) for name, pos in _CAMERA_POSES.items()}),
            action_options=ActionManagerOptions(**action_options),
            observation_options=ObservationManagerOptions(**observation_options),
        )
    )


@pytest.fixture(scope="module", autouse=True)
def _destroy_cached_envs():
    yield
    for env in _ENVS.values():
        env.destroy()
    _ENVS.clear()


@pytest.fixture(scope="module")
def one_exterior_env(renderable):
    """Exterior + wrist camera groups: the env for ``duplicate_exterior_view=True``."""
    return _env()


@pytest.fixture(scope="module")
def two_exterior_env(renderable):
    """Two exterior + wrist camera groups: the env for ``secondary_camera_name="camera_2"``."""
    return _env(Cosmos3Policy.observation_options(secondary_camera_name="camera_2", **_GRIPPER))


def _reply(call):
    return {"action": chunk(32, 8, call, gripper_col=7)}


def test_wire_contract_and_controller_outputs_match_the_golden_record(one_exterior_env):
    configure = dict(image_size=(6, 8), exec_horizon=2, pred_action_horizon=32, duplicate_exterior_view=True)

    def obs(step):
        return {
            "image": golden_image(seed=step),
            "wrist_image": golden_image(seed=20 + step),
            "proprio": joint8(0.6 if step else 0.1),
        }

    with WebsocketPolicyServer("msgpack_openpi", _reply) as server:
        policy = Cosmos3Policy(
            one_exterior_env, Cosmos3Policy.configure(host=server.host, port=server.port, **configure)
        )
        try:
            policy.set_task(GOLDEN_TASK)
            got = record_golden(policy, server, obs, steps=3)
        finally:
            policy.close()
    assert_matches_golden("cosmos3_concat_view", got)


def test_the_second_exterior_camera_fills_the_second_exterior_slot(two_exterior_env):
    """``image_2`` goes out as ``exterior_image_2_left``, not a copy of the primary exterior."""
    # Frames already at image_size, so the client's resize-with-pad leaves them untouched.
    exterior, exterior_2, wrist = (golden_image(h=6, w=8, seed=seed) for seed in (1, 2, 3))
    with WebsocketPolicyServer("msgpack_openpi", _reply) as server:
        policy = Cosmos3Policy(
            two_exterior_env,
            Cosmos3Policy.configure(
                host=server.host, port=server.port, image_size=(6, 8), secondary_camera_name="camera_2"
            ),
        )
        try:
            policy.set_task(GOLDEN_TASK)
            policy.step({"image": exterior, "image_2": exterior_2, "wrist_image": wrist, "proprio": joint8(0.1)})
        finally:
            policy.close()
    (sent,) = server.records
    np.testing.assert_array_equal(sent["observation/wrist_image_left"], wrist[0].numpy())
    np.testing.assert_array_equal(sent["observation/exterior_image_1_left"], exterior[0].numpy())
    np.testing.assert_array_equal(sent["observation/exterior_image_2_left"], exterior_2[0].numpy())
    assert "observation/image" not in sent  # the server would read it as an already composed concat view


def test_the_envs_own_observations_go_out_as_the_three_roboarena_views(two_exterior_env):
    """What the env's observation manager produces is what the server receives, view for view."""
    obs, _ = two_exterior_env.reset()  # the observation manager's output, as a driver steps the policy with it
    image_size = (6, 8)
    with WebsocketPolicyServer("msgpack_openpi", _reply) as server:
        policy = Cosmos3Policy(
            two_exterior_env,
            Cosmos3Policy.configure(
                host=server.host, port=server.port, image_size=image_size, secondary_camera_name="camera_2"
            ),
        )
        try:
            policy.set_task(GOLDEN_TASK)
            controls = policy.step(obs)
        finally:
            policy.close()
    (sent,) = server.records
    views = {
        "observation/wrist_image_left": "wrist_image",
        "observation/exterior_image_1_left": "image",
        "observation/exterior_image_2_left": "image_2",
    }
    assert sorted(key for key in sent if "image" in key) == sorted(views)
    # The three cameras look at the robot from different places, so a swapped or duplicated view shows.
    wrist, exterior_1, exterior_2 = (sent[key] for key in views)
    assert not np.array_equal(exterior_1, exterior_2)
    assert not np.array_equal(wrist, exterior_1) and not np.array_equal(wrist, exterior_2)
    for key, group in views.items():
        np.testing.assert_array_equal(sent[key], prepare_image_padded(obs[group], image_size), err_msg=key)
    proprio = obs["proprio"][0].cpu().numpy()
    np.testing.assert_array_equal(sent["observation/joint_position"], proprio[:7])
    np.testing.assert_array_equal(sent["observation/gripper_position"], proprio[7:8])
    assert set(controls) == {"ee_controller", "gripper_controller"}


def test_the_second_exterior_view_must_be_chosen_explicitly(one_exterior_env):
    with pytest.raises(ValueError, match=r"needs a second exterior view.*duplicate_exterior_view=True"):
        Cosmos3Policy(one_exterior_env, Cosmos3Policy.configure())
    with pytest.raises(ValueError, match=r"both secondary_camera_name='camera_2' and duplicate_exterior_view=True"):
        Cosmos3Policy(
            one_exterior_env,
            Cosmos3Policy.configure(secondary_camera_name="camera_2", duplicate_exterior_view=True),
        )


def test_a_policy_with_a_second_exterior_refuses_an_env_without_image_2(one_exterior_env):
    with pytest.raises(ValueError, match=r"reads the observation group\(s\) \['image_2'\]"):
        Cosmos3Policy(one_exterior_env, Cosmos3Policy.configure(secondary_camera_name="camera_2"))


def test_a_duplicating_policy_refuses_an_env_that_renders_a_second_exterior(two_exterior_env):
    with pytest.raises(
        ValueError, match=r"renders a second exterior camera, 'camera_2'.*secondary_camera_name='camera_2'"
    ):
        Cosmos3Policy(two_exterior_env, Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_a_secondary_camera_other_than_the_one_image_2_renders_is_refused(two_exterior_env):
    with pytest.raises(ValueError, match=r"secondary_camera_name='wrist_camera', but .*renders camera 'camera_2'"):
        Cosmos3Policy(two_exterior_env, Cosmos3Policy.configure(secondary_camera_name="wrist_camera"))


def test_an_image_2_that_renders_the_primary_exterior_is_refused(renderable):
    """Two-camera mode with one camera is duplication by another name; it has its own flag."""
    env = _env(Cosmos3Policy.observation_options(secondary_camera_name="camera", **_GRIPPER))
    with pytest.raises(ValueError, match=r"'camera' is also the primary exterior camera"):
        Cosmos3Policy(env, Cosmos3Policy.configure(secondary_camera_name="camera"))


def test_a_joint_state_of_fewer_joints_than_the_arm_term_is_refused(renderable):
    """A 6-joint (7-D) proprio: the server rejects any joint_position width but DROID's 7."""
    env = _env(Cosmos3Policy.observation_options(arm_dofs_name=list(DROID_ARM_DOFS)[:6], **_GRIPPER))
    with pytest.raises(
        ValueError,
        match=r"Cosmos3Policy sends proprio\[:7\] as DROID's joint_position .*'joint_pos' reads 'robot' \['joint1', "
        r".*'joint6'\] but the arm term drives 'robot' \['joint1', .*'joint7'\].*Fix: pass arm_entity_name='robot' "
        r"and arm_dofs_name=\['joint1', .*'joint7'\] to both Cosmos3Policy\.observation_options\(\.\.\.\) and "
        r"Cosmos3Policy\.action_options\(\.\.\.\), keeping the other arguments",
    ):
        Cosmos3Policy(env, Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_an_end_effector_pose_proprio_of_the_same_width_is_refused(renderable):
    """The EE-delta clients' 8-D ``[ee_pos, ee_quat, openness]`` group would pass a width check as joint state."""
    from eden.extensions.policies.adapters import make_ee_pose_group

    groups = Cosmos3Policy.observation_options(**_GRIPPER)
    groups["proprio"] = make_ee_pose_group(
        "robot", "attachment", gripper_entity_name="gripper", gripper_dof_name="left_driver_joint"
    )
    with pytest.raises(
        ValueError,
        match=r"'proprio' group does not match: it holds the terms \['ee_pose', 'gripper'\], not \['joint_pos', "
        r"'gripper'\] in that order\. Fix: pass arm_entity_name='robot'",
    ):
        Cosmos3Policy(_env(observation_options=groups), Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_a_joint_state_in_another_order_than_the_arm_term_is_refused(renderable):
    groups = Cosmos3Policy.observation_options(arm_dofs_name=list(reversed(DROID_ARM_DOFS)), **_GRIPPER)
    with pytest.raises(
        ValueError, match=r"'joint_pos' reads 'robot' \['joint7', .*so each joint would be sent another"
    ):
        Cosmos3Policy(_env(observation_options=groups), Cosmos3Policy.configure(duplicate_exterior_view=True))


@pytest.mark.parametrize(
    ("group_fields", "joint_fields", "match"),
    [
        ({"history_length": 2}, {}, r"'proprio' is not the plain current reading .*: history_length=2\."),
        ({}, {"scale": 0.5}, r"'proprio' is not the plain current reading .*: joint_pos\.scale=0\.5\."),
    ],
    ids=["history", "scale"],
)
def test_a_proprio_with_the_right_terms_behind_a_transform_is_refused(renderable, group_fields, joint_fields, match):
    """``proprio[:7]`` would send the oldest frame, or a rescaled joint state, as the current one."""
    from eden.managers.terms.observations.proprio import DofsPos
    from eden.options.managers.observations import ObservationGroupOptions

    groups = Cosmos3Policy.observation_options(**_GRIPPER)
    joint_pos = DofsPos.configure(entity_name="robot", dofs_name=list(DROID_ARM_DOFS), **joint_fields)
    groups["proprio"] = ObservationGroupOptions(**group_fields, joint_pos=joint_pos, gripper=groups["proprio"].gripper)
    with pytest.raises(ValueError, match=match + r".*Fix: build the env's 'proprio' group from Cosmos3Policy"):
        Cosmos3Policy(_env(observation_options=groups), Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_a_gripper_state_with_another_close_position_than_the_gripper_term_is_refused(renderable):
    """``gripper_close_pos`` is set separately on the two factories; one set alone skews every closedness sent."""
    env = _env(Cosmos3Policy.observation_options(gripper_close_pos=0.7, **_GRIPPER))
    with pytest.raises(
        ValueError,
        match=r"Gripper state proprio/gripper and action term 'gripper_controller' describe different grippers: "
        r"close_pos=0\.7 but 'gripper_controller' has close_action=0\.8\. Fix: pass the same",
    ):
        Cosmos3Policy(env, Cosmos3Policy.configure(duplicate_exterior_view=True))


def _droid_arm(**overrides):
    arm = Cosmos3Policy.action_options(**_GRIPPER)["ee_controller"]
    return ImplicitPDController.configure(**{**arm.model_dump(include={"entity_name", "dofs_name"}), **overrides})


@pytest.mark.parametrize(
    ("arm_key", "arm_overrides", "match"),
    [
        (
            "arm_controller",
            {},
            r"to action term 'ee_controller', which must be exactly an ImplicitPDController .* but the env's "
            r"'ee_controller' is missing \(action terms: \['arm_controller', 'gripper_controller'\]\)",
        ),
        ("ee_controller", {}, r"reference_source='default', which adds a reference to every command"),
        (
            "ee_controller",
            dict(reference_source=ReferenceSource.ZERO, scale=0.5),
            r"effective_scale=\[0\.5\] \(from scale=0\.5, scale_ratio=None, needs 1\)",
        ),
        (
            "ee_controller",
            dict(reference_source=ReferenceSource.ZERO, dofs_name=list(reversed(DROID_ARM_DOFS))),
            r"dofs_name=\['joint7', .*\] \(needs \['joint1', .*'joint7'\], in that order\)",
        ),
    ],
    ids=["missing", "default_reference", "scaled", "reordered"],
)
def test_construction_refuses_an_arm_term_that_would_not_apply_absolute_targets_as_sent(
    renderable, arm_key, arm_overrides, match
):
    actions = Cosmos3Policy.action_options(**_GRIPPER)
    actions.pop("ee_controller")
    env = _env(action_options={arm_key: _droid_arm(**arm_overrides), **actions})
    with pytest.raises(ValueError, match=match):
        Cosmos3Policy(env, Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_an_arm_term_with_an_all_zero_per_joint_offset_is_accepted(renderable):
    """A per-joint dict of zeros is the identity offset; a scalar ``!= 0`` comparison on the dict used to refuse it."""
    actions = Cosmos3Policy.action_options(**_GRIPPER)
    actions["ee_controller"] = _droid_arm(
        reference_source=ReferenceSource.ZERO, offset=dict.fromkeys(DROID_ARM_DOFS, 0.0)
    )
    with WebsocketPolicyServer("msgpack_openpi", _reply) as server:
        configure = Cosmos3Policy.configure(host=server.host, port=server.port, duplicate_exterior_view=True)
        Cosmos3Policy(_env(action_options=actions), configure).close()


@ACTION_TERM_REGISTRY.register()
class Cosmos3TestReinterpretingJointTarget(ImplicitPDController):
    """Every option field as Cosmos3Policy.action_options() builds it, but another class executing the targets."""


def test_a_subclass_of_the_arm_term_is_refused(renderable):
    """A subclass (an EMA-filtered target, ...) can reinterpret the same row with every option field unchanged."""
    actions = Cosmos3Policy.action_options(**_GRIPPER)
    actions["ee_controller"] = actions["ee_controller"].model_copy(
        update={"name": to_snake_case(Cosmos3TestReinterpretingJointTarget.__name__)}
    )
    with pytest.raises(
        ValueError,
        match=r"which must be exactly an ImplicitPDController .* but the env's 'ee_controller' is of type "
        r"Cosmos3TestReinterpretingJointTarget",
    ):
        Cosmos3Policy(_env(action_options=actions), Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_an_arm_term_offset_set_after_build_is_refused(renderable):
    """A ``set_dofs_pos_offset()`` (joint-zero domain randomization) shifts every target the server sends.

    On its own env: the offset is the env's state, which the cached envs share across tests.
    """
    import torch

    env = _build_env(Cosmos3Policy.observation_options(**_GRIPPER), Cosmos3Policy.action_options(**_GRIPPER), 1 / 15)
    try:
        env.action_manager.terms["ee_controller"].set_dofs_pos_offset(
            torch.full((env.num_envs, len(DROID_ARM_DOFS)), 0.1, device=env.device)
        )
        with pytest.raises(
            ValueError, match=r"effective_offset up to 0\.1 in magnitude, .*from a set_dofs_pos_offset\(\) call"
        ):
            Cosmos3Policy(env, Cosmos3Policy.configure(duplicate_exterior_view=True))
    finally:
        env.destroy()


def test_construction_refuses_an_env_that_does_not_step_at_droids_15_hz(renderable):
    """Each row is a waypoint 1/15 s after the last; at 50 Hz the 32-row chunk would play 3.33x too fast."""
    with pytest.raises(
        ValueError,
        match=r"Cosmos3Policy executes one action row per 0\.066667 s \(15 Hz\) control step, as DROID's RobotEnv does "
        r".* env\.dt=0\.020000 s .* 3\.33x too fast\. Fix: .*EnvOptions\(sim_dt=1/120, decimation=8\)"
        r".*sim_substeps=9.*Do not rescale or resample the actions",
    ):
        Cosmos3Policy(_env(sim_dt=0.02), Cosmos3Policy.configure(duplicate_exterior_view=True))


def test_a_request_without_a_task_raises_before_anything_is_sent(one_exterior_env):
    """There is no default prompt: a forgotten ``set_task`` must not reach the server as a generic instruction."""
    obs = {"image": golden_image(h=6, w=8), "wrist_image": golden_image(h=6, w=8, seed=1), "proprio": joint8(0.1)}
    with WebsocketPolicyServer("msgpack_openpi", _reply) as server:
        policy = Cosmos3Policy(
            one_exterior_env,
            Cosmos3Policy.configure(
                host=server.host, port=server.port, image_size=(6, 8), duplicate_exterior_view=True
            ),
        )
        try:
            with pytest.raises(
                ValueError,
                match=r"task_description is None\. Fix: policy\.set_task\('<instruction>'\).*"
                r"Cosmos3Policy\.configure\(task_description='<instruction>'\)",
            ):
                policy.step(obs)
        finally:
            policy.close()
    assert server.records == []


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (Cosmos3Policy.observation_options, dict(primary_camera_name="camera")),
        (Cosmos3Policy.action_options, dict(threshold=0.4)),
    ],
    ids=["observation_options", "action_options"],
)
def test_the_option_factories_reject_unknown_keywords(factory, kwargs):
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        factory(**kwargs)


def test_droid_proprio_is_closedness_even_with_the_gripper_on_the_arm_entity():
    """DROID's ``gripper_position`` is closedness in [0, 1]; never the raw driver angle."""
    gripper = Cosmos3Policy.observation_options(gripper_entity_name=None, gripper_close_pos=0.8)["proprio"].gripper
    assert gripper.normalize == "closedness"
    assert gripper.entity_name == "robot" and gripper.close_pos == 0.8


def test_the_client_declares_exactly_its_official_servers_reset_api():
    # The Cosmos3 server takes the prompt with every request and has no reset.
    options = Cosmos3Policy.configure()
    assert (options.set_task_endpoint, options.reset_endpoint) == (None, None)
