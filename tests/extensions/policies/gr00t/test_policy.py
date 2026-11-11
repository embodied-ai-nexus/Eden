from __future__ import annotations

import contextlib
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eden.constants import ReferenceSource
from eden.extensions.policies.gr00t.policy import Gr00tPolicy
from eden.extensions.policies.adapters import libero_task_action_options
from eden.extensions.policies.gr00t.setups import OxeDroidSetup, droid_eef_correction_quat
from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.modifiers.observations.noise import GaussianNoise
from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
from eden.managers.terms.observations.manipulation import EndEffectorPose
from eden.utils.image import resize_with_pad
from eden.utils.misc import to_snake_case
from eden.utils.policy_client.errors import ServerError
from tests.extensions.policies.conftest import (
    GOLDEN_TASK,
    TILT_QUAT,
    assert_matches_golden,
    core_term,
    ee8,
    joint8,
    keyed_chunk,
    policy_env,
    record_golden,
    vec,
)
from tests.extensions.policies.conftest import image as golden_image
from tests.support.policy_servers import ZmqPolicyServer

_OXE = "oxe_droid_relative_eef_relative_joint"
_EE7 = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
_EE7_KEYS = dict.fromkeys(_EE7, 1)
_OXE_KEYS = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
_OXE_STREAMS = tuple(_OXE_KEYS)
_TASK = "pick up the object"
#: What Gr00tSimPolicyWrapper prefixes each action key with (``--use-sim-policy-wrapper`` servers).
_SIM = "action."


def _modality(keys, delta_indices) -> dict:
    """One modality as Isaac-GR00T's server sends it: a tagged ``ModalityConfig`` record."""
    return {"__ModalityConfig__": True, "as_json": {"modality_keys": list(keys), "delta_indices": list(delta_indices)}}


def _served_config(video, state, action, language, *, action_len, video_delta=(0,)) -> dict:
    return {
        "video": _modality(video, video_delta),
        "state": _modality(state, [0]),
        "action": _modality(action, range(action_len)),
        "language": _modality([language], [0]),
    }


_TASK_KEY = "annotation.human.action.task_description"
_WIDOWX_CONFIG = _served_config(["image_0"], [*_EE7[:6], "pad", "gripper"], _EE7, _TASK_KEY, action_len=8)
_LIBERO_CONFIG = _served_config(["image", "wrist_image"], _EE7, _EE7, _TASK_KEY, action_len=16)
_OXE_CONFIG = _served_config(
    ["exterior_image_1_left", "wrist_image_left"],
    _OXE_STREAMS,
    _OXE_STREAMS,
    "annotation.language.language_instruction",
    action_len=40,
    video_delta=(-15, 0),
)


def _chunk_reply(t: int, keys: dict, *, gripper_key: str, prefix: str = ""):
    """``keyed_chunk`` replies, with each action key prefixed as a sim-wrapper server sends it."""
    return lambda call: {prefix + k: v for k, v in keyed_chunk(t, keys, call, gripper_key=gripper_key).items()}


_oxe_reply = _chunk_reply(40, _OXE_KEYS, gripper_key="gripper_position")

#: GR00T LIBERO's camera resolution (``libero_env.py`` renders 256x256); the setup refuses any other.
_LIBERO_CAMERA = dict(cam_resolution=(256, 256))

#: setup -> (robot, env option kwargs, cameras its image groups read -> CameraOptions kwargs, served modality config,
#: default reply)
_SETUPS = {
    "simpler_env_widowx": (
        "widowx",
        {},
        {"camera": {}},
        _WIDOWX_CONFIG,
        _chunk_reply(8, _EE7_KEYS, gripper_key="gripper", prefix=_SIM),
    ),
    "libero_sim": (
        "libero",
        {},
        dict.fromkeys(("agentview", "eye_in_hand"), _LIBERO_CAMERA),
        _LIBERO_CONFIG,
        _chunk_reply(16, _EE7_KEYS, gripper_key="gripper", prefix=_SIM),
    ),
    # Eden's Droid rig carries the Robotiq on its own ``gripper`` entity.
    _OXE: ("droid", dict(gripper_entity_name="gripper"), {"camera": {}, "wrist_camera": {}}, _OXE_CONFIG, _oxe_reply),
}
#: setup -> EnvOptions kwargs stepping at its checkpoint's control rate (sim_dt * decimation = 1 / control_hz).
_RATES = {
    "simpler_env_widowx": dict(sim_dt=0.005, decimation=40),  # 5 Hz, as Eden's stock SIMPLER tasks
    "libero_sim": dict(sim_dt=0.002, decimation=25),  # 20 Hz, as the stock libero task
    _OXE: dict(sim_dt=1 / 150, decimation=10),  # 15 Hz, DROID_CONTROL_FREQUENCY
}
_ENVS: dict = {}


def _build_env(robot, cameras: dict, observation_options: dict, action_options: dict, *, entities=None, **env_options):
    """A built scene with ``robot`` (plus ``entities``), tiny rasterizer ``cameras`` (name -> CameraOptions kwargs)
    and the given terms.
    """
    import eden as en
    from genesis.options.sensors import RasterizerCameraOptions

    from eden.options import EnvOptions
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.managers.actions import ActionManagerOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    camera = dict(cam_resolution=(16, 12), sensor_options=RasterizerCameraOptions())
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, **env_options),
        scene_options=en.scenes.Empty(robot=robot, **(entities or {})),
        cameras_options=CamerasOptions(
            **{name: CameraOptions(**{**camera, **kwargs}) for name, kwargs in cameras.items()}
        ),
        observation_options=ObservationManagerOptions(**observation_options),
        action_options=ActionManagerOptions(**action_options),
    )
    return build_env(config, reset=True)


def _setup_env(setup: str):
    """The setup's robot with its own ``observation_options()`` and ``action_options()``, built once per module."""
    from eden.options.robots import Droid
    from eden.options.robots.libero import LiberoPanda
    from eden.options.robots.trossen_widowx250 import WidowX250

    if setup not in _ENVS:
        robot, kwargs, cameras, _, _ = _SETUPS[setup]
        robot = {"widowx": WidowX250, "libero": LiberoPanda, "droid": Droid}[robot]()
        _ENVS[setup] = _build_env(
            robot,
            {
                name: dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3), **extra)
                for name, extra in cameras.items()
            },
            Gr00tPolicy.observation_options(policy_setup=setup, **kwargs),
            Gr00tPolicy.action_options(policy_setup=setup, **kwargs),
            **_RATES[setup],
        )
    return _ENVS[setup]


@contextlib.contextmanager
def _gr00t(setup: str = _OXE, reply=None, *, env=None, modality_config=None, info=None, wrap_reply=True, **configure):
    """A real ``Gr00tPolicy`` for ``setup`` on its robot (or ``env``), against a local GR00T ZMQ server."""
    configure.setdefault("task_description", _TASK)
    with ZmqPolicyServer(
        "msgpack_numpy", reply or _SETUPS[setup][4], modality_config=modality_config, info=info, wrap_reply=wrap_reply
    ) as server:
        policy = Gr00tPolicy(
            _setup_env(setup) if env is None else env,
            Gr00tPolicy.configure(policy_setup=setup, host=server.host, port=server.port, **configure),
        )
        try:
            yield policy, server
        finally:
            policy.close()


def _droid_obs(image=None, wrist=True, **overrides) -> dict:
    image = golden_image(180, 320) if image is None else image
    obs = {"image": image, "eef_9d": vec(0.4, 0.1, 0.3, 0, 0, -1, 0, 1, 0), "joint_state": joint8(0.0)}
    if wrist:
        obs["wrist_image"] = image
    return {**obs, **overrides}


def _libero_obs(proprio=None) -> dict:
    proprio = vec(0.3, -0.1, 0.2, *TILT_QUAT, 0.02, -0.02) if proprio is None else proprio
    return {"image": golden_image(), "wrist_image": golden_image(seed=1), "proprio": proprio}


def _sent(server, i: int = 0) -> dict:
    """The observation payload of the server's ``i``-th inference request."""
    return server.records[i]["observation"]


# ---------------------------------------------------------------------------
# Server configuration: each setup checks what its server advertises at construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setup", list(_SETUPS))
def test_each_setup_accepts_the_modality_config_its_server_advertises(setup):
    with _gr00t(setup, modality_config=_SETUPS[setup][3]) as (_, server):
        assert "get_modality_config" in server.control


def test_a_server_launched_for_another_embodiment_is_rejected_at_construction():
    """A WidowX client facing a server launched with the LIBERO embodiment tag fails before the first step."""
    with pytest.raises(ValueError, match="WidowxSimplerSetup.*modality config"):
        with _gr00t("simpler_env_widowx", modality_config=_LIBERO_CONFIG):
            pass


def test_the_client_declares_exactly_its_official_servers_reset_api():
    # The GR00T PolicyServer has a 'reset' endpoint and no set-task API.
    options = Gr00tPolicy.configure()
    assert (options.set_task_endpoint, options.reset_endpoint) == (None, "reset")


# ---------------------------------------------------------------------------
# Request format: --use-sim-policy-wrapper (Gr00tSimPolicyWrapper) vs. a bare Gr00tPolicy server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "expected"),
    [("simpler_env_widowx", True), ("libero_sim", True), (_OXE, False)],
)
def test_the_request_format_defaults_to_the_upstream_documented_server_launch(setup, expected):
    """Upstream's SimplerEnv and LIBERO READMEs launch with --use-sim-policy-wrapper, the DROID one without."""
    with _gr00t(setup) as (policy, _):
        assert policy.use_sim_policy_wrapper is expected


def test_sim_wrapper_request_is_flat_and_its_prefixed_reply_is_decoded():
    """Gr00tSimPolicyWrapper takes ``video.<key>`` / ``state.<key>`` and a ``(B,)`` language batch, replies ``action.<key>``."""
    with _gr00t("simpler_env_widowx", image_size=(6, 8)) as (policy, server):
        action = policy.step({"image": golden_image(), "proprio": ee8(1.0)})
    sent = _sent(server)
    assert set(sent) == {"video.image_0", *(f"state.{k}" for k in (*_EE7[:6], "pad", "gripper")), _TASK_KEY}
    assert sent[_TASK_KEY] == [_TASK]
    assert sent["video.image_0"].shape == (1, 1, 6, 8, 3)
    assert sent["state.x"].shape == (1, 1, 1) and sent["state.x"].dtype == np.float32
    expected = keyed_chunk(8, _EE7_KEYS, 1, gripper_key="gripper")  # the server numbers calls from 1
    assert action["ee_controller"][0, 1].item() == pytest.approx(float(expected["y"][0, 0, 0]))
    assert action["gripper_controller"].item() == float(expected["gripper"][0, 0, 0])


def test_bare_server_request_is_nested_and_its_bare_reply_is_decoded():
    reply = _chunk_reply(8, _EE7_KEYS, gripper_key="gripper")
    with _gr00t("simpler_env_widowx", reply, image_size=(6, 8), use_sim_policy_wrapper=False) as (policy, server):
        policy.step({"image": golden_image(), "proprio": ee8(1.0)})
    sent = _sent(server)
    assert set(sent) == {"video", "state", "language"}
    assert sent["language"] == {_TASK_KEY: [[_TASK]]}
    assert set(sent["video"]) == {"image_0"}


@pytest.mark.parametrize(
    ("use_sim_policy_wrapper", "server_error", "flip"),
    [
        (False, "Video key 'video.image_0' must be in observation", "use_sim_policy_wrapper=True"),
        (True, "Observation must contain a 'video' key", "use_sim_policy_wrapper=False"),
    ],
    ids=["server_has_wrapper", "server_has_no_wrapper"],
)
def test_a_request_in_the_other_format_names_the_flag_to_flip(use_sim_policy_wrapper, server_error, flip):
    """The connect-time modality check passes either way; the first step's server error says what to change."""
    with _gr00t(
        "simpler_env_widowx",
        lambda call: {"error": server_error},
        wrap_reply=False,
        image_size=(6, 8),
        use_sim_policy_wrapper=use_sim_policy_wrapper,
    ) as (policy, _):
        with pytest.raises(ServerError) as excinfo:
            policy.step({"image": golden_image(), "proprio": ee8(1.0)})
    message = str(excinfo.value)
    assert server_error in message  # the server's own text is kept
    assert "--use-sim-policy-wrapper" in message and flip in message


def test_an_unrelated_server_error_passes_through_unchanged():
    with _gr00t("simpler_env_widowx", lambda call: {"error": "CUDA out of memory"}, wrap_reply=False) as (policy, _):
        with pytest.raises(ServerError, match="CUDA out of memory") as excinfo:
            policy.step({"image": golden_image(), "proprio": ee8(1.0)})
    assert "use_sim_policy_wrapper" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Caller mistakes fail at construction or on the first step, never silently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setup", ["simpler_env_widowx", "libero_sim"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_horizon", 1),
        ("video_stride", 3),
        ("debug_gripper", True),
    ],
)
def test_oxe_droid_only_fields_are_rejected_under_the_other_setups(setup, field, value):
    with pytest.raises(ValueError, match=rf"Gr00tPolicy\.{field}=.*oxe_droid_relative_eef_relative_joint"):
        with _gr00t(setup, **{field: value}):
            pass


@pytest.mark.parametrize("field", ["primary_camera_name", "use_wrist_image", "wrist_camera_name", "gripper_n_dofs"])
def test_the_client_has_no_camera_or_gripper_fields_that_nothing_reads(field):
    with pytest.raises(ValueError, match=field):
        Gr00tPolicy.configure(**{field: "camera"})


@pytest.mark.parametrize(
    ("factory", "setup", "kwargs"),
    [
        ("observation_options", "libero_sim", dict(use_wrist_image=True)),
        ("observation_options", "simpler_env_widowx", dict(wrist_camera_name="wrist")),
        # action_options() drives the WidowX fingers on the arm entity, so no separate gripper entity can pair with it.
        ("observation_options", "simpler_env_widowx", dict(gripper_entity_name="gripper")),
        ("observation_options", _OXE, dict(use_wrist_image=False)),
        ("action_options", "libero_sim", dict(scale=0.5)),
        ("action_options", "simpler_env_widowx", dict(gripper_dofs_name=["left_finger"])),
    ],
)
def test_env_option_factories_reject_unknown_kwargs(factory, setup, kwargs):
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        getattr(Gr00tPolicy, factory)(policy_setup=setup, **kwargs)


def test_a_step_without_a_task_raises_instead_of_sending_no_instruction():
    with _gr00t("simpler_env_widowx", task_description=None, image_size=(6, 8)) as (policy, server):
        with pytest.raises(ValueError, match=r"set_task\("):
            policy.step({"image": golden_image(), "proprio": ee8(1.0)})
    assert server.infer_calls == 0


# ---------------------------------------------------------------------------
# OXE-DROID: what the reference client (Isaac-GR00T examples/DROID/main_gr00t.py) sends and executes
# ---------------------------------------------------------------------------


def test_gr00t_droid_absolute_targets_do_not_add_default_pose_offset():
    actions = OxeDroidSetup.action_options(gripper_entity_name="gripper")

    assert actions["ee_controller"].reference_source == ReferenceSource.ZERO
    # Gripper: GripperDecisionController with absolute [0, close_pos] targets (no default-pose offset by design).
    assert actions["gripper_controller"].name == "gripper_decision_controller"
    assert actions["gripper_controller"].input == "closedness"
    assert actions["gripper_controller"].open_action == 0.0


def test_gr00t_forwards_each_executed_rows_own_gripper_value():
    """No chunk averaging or latching: every executed row carries its own ``gripper_position``."""
    per_row = np.resize(np.asarray([0.2, 0.7, 0.4], dtype=np.float32), 40)

    def reply(call: int) -> dict:
        streams = _oxe_reply(call)
        streams["gripper_position"] = per_row.reshape(1, 40, 1)
        return streams

    sent, diagnostics = [], []
    with _gr00t(reply=reply, image_size=(6, 8)) as (policy, _):
        for _ in range(3):
            sent.append(policy.step(_droid_obs(golden_image()))["gripper_controller"].item())
            # JointPositionAdapter's diagnostics, read by evaluators' gripper plots per action.
            diagnostics.append(dict(policy._setup.adapter.last_diagnostics))
    assert sent == pytest.approx([0.2, 0.7, 0.4])
    # The diagnostics describe the row being executed, not the chunk's first row.
    assert [d["raw_gripper_action"] for d in diagnostics] == pytest.approx([0.2, 0.7, 0.4])
    assert [d["gripper_intent"] for d in diagnostics] == ["open", "close", "open"]


def test_gr00t_droid_sends_joint_targets_unchanged_in_the_arm_controllers_order():
    """``joint_position`` (columns 10..16 of the 17-D row) goes to the arm untouched, the closedness to the gripper."""
    with _gr00t(image_size=(6, 8)) as (policy, _):
        action = policy.step(_droid_obs(golden_image()))
    chunk = keyed_chunk(40, _OXE_KEYS, 1, gripper_key="gripper_position")  # the server numbers calls from 1
    np.testing.assert_array_equal(action["ee_controller"][0].numpy(), chunk["joint_position"][0, 0])
    assert action["gripper_controller"].item() == chunk["gripper_position"][0, 0, 0]


def test_gr00t_keeps_the_servers_info_beside_the_action_chunk():
    """The PolicyServer replies ``[action_dict, info]``; ``info`` lands in ``last_response_info``."""
    with _gr00t(info={"inference_ms": 12.5}, image_size=(6, 8)) as (policy, _):
        policy.step(_droid_obs(golden_image()))
    assert policy.last_response_info == {"inference_ms": 12.5}


def test_gr00t_droid_records_a_frame_every_control_step_like_the_reference_client():
    """main_gr00t.py appends to its frame buffer every step and sends [buffer[0], buffer[-1]] (15 steps apart)."""
    frame = lambda step: torch.full((1, 4, 4, 3), step, dtype=torch.uint8)  # noqa: E731
    with _gr00t(image_size=(4, 4)) as (policy, server):
        assert policy.exec_horizon == 15
        for step in range(31):
            policy.step(_droid_obs(frame(step)))
    assert server.infer_calls == 3  # steps 0, 15 and 30
    windows = [_sent(server, i)["video"]["exterior_image_1_left"][0, :, 0, 0, 0].tolist() for i in range(3)]
    assert windows == [[0, 0], [0, 15], [15, 30]]  # the first window pads with the oldest frame


def test_gr00t_droid_letterboxes_images_like_the_reference_client():
    """main_gr00t.py: resize_with_pad(img, 180, 320) — aspect preserved, zero bars, no stretch."""
    assert OxeDroidSetup.default_image_size == (180, 320)
    square = golden_image(240, 240)
    with _gr00t(video_horizon=1) as (policy, server):
        policy.step(_droid_obs(square))
    video = _sent(server)["video"]["exterior_image_1_left"][0, 0]
    assert video.shape == (180, 320, 3)
    np.testing.assert_array_equal(video, resize_with_pad(square[0].numpy(), 180, 320))
    assert not video[:, :70].any() and not video[:, -70:].any()  # pillar-boxed, not stretched


def test_gr00t_droid_refuses_a_hand_built_observation_without_the_wrist_view():
    """Construction checks the env's groups; a dict the caller builds is checked on the step it arrives."""
    with (
        _gr00t(video_horizon=1) as (policy, _),
        pytest.raises(
            KeyError,
            match=r"without \['wrist_image'\], which policy_setup='oxe_droid_relative_eef_relative_joint' reads",
        ),
    ):
        policy.step(_droid_obs(wrist=False))


def test_gr00t_droid_refuses_an_env_without_a_wrist_view_at_construction():
    """The checkpoint's schema has a distinct wrist view: an env publishing none fails before the first step."""
    groups = Gr00tPolicy.observation_options(policy_setup=_OXE, gripper_entity_name="gripper")
    del groups["wrist_image"]
    with pytest.raises(ValueError, match=r"'wrist_image' \(\(H, W, C\) image, missing\)"):
        with _gr00t(env=_env_with_observations(_OXE, groups)):
            pass


def test_gr00t_droid_default_frame_is_the_droid_flange_on_eden_fr3():
    """DROID cartesian_position is panda_link8 (link7 + 0.107 z, no rotation).

    Eden's FR3 ``attachment`` is the same origin rotated by fr3.xml's quat; the default offset must undo it.
    """
    from eden.extensions.policies.gr00t.setups import DROID_FLANGE_FROM_FR3_ATTACHMENT_EULER
    from eden.options.robots.franka_emika_panda import FrankaResearch3
    from eden.utils.assets import get_asset_path

    # The file Eden's FR3 (and so the DROID rig) loads, fetched like any build would.
    xml = Path(get_asset_path(FrankaResearch3().file)).read_text()
    quat = re.search(r'<body name="attachment" pos="0 0 0.107" quat="([^"]+)"', xml).group(1).split()
    r_link7_attachment = Rotation.from_quat(np.roll(np.asarray(quat, dtype=float), -1))  # wxyz -> xyzw
    offset = Rotation.from_euler("xyz", DROID_FLANGE_FROM_FR3_ATTACHMENT_EULER)
    np.testing.assert_allclose((r_link7_attachment * offset).as_matrix(), np.eye(3), atol=1e-6)

    defaults = inspect.signature(OxeDroidSetup.observation_options).parameters
    assert defaults["ee_link_name"].default == "attachment"
    assert defaults["eef_tool_frame_offset_euler"].default == DROID_FLANGE_FROM_FR3_ATTACHMENT_EULER


def test_gr00t_sends_the_eef_9d_it_observes_unchanged():
    """The env's ``eef_9d`` term already is the training-convention state, so the client sends it as is."""
    obs = _droid_obs()
    with _gr00t(video_horizon=1) as (policy, server):
        policy.step(obs)
    np.testing.assert_array_equal(_sent(server)["state"]["eef_9d"][0, 0], obs["eef_9d"][0].numpy())


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-1.0, 0.5, 2.5)])
def test_gr00t_correction_quat_is_the_extrinsic_xyz_tool_offset_then_droids_correction(rpy):
    """``TOOL_OFFSET @ DROID_CORRECT`` with the offset read as extrinsic xyz Euler (scipy lowercase ``"xyz"``)."""
    correct = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
    expected = Rotation.from_euler("xyz", rpy).as_matrix() @ correct
    quat_wxyz = np.asarray(droid_eef_correction_quat(rpy))
    np.testing.assert_allclose(Rotation.from_quat(np.roll(quat_wxyz, -1)).as_matrix(), expected, atol=1e-6)


@pytest.mark.parametrize("tool_offset", [(0.0, 0.0, 0.0), (0.2, -0.1, 0.4)])
def test_gr00t_eef_9d_term_matches_isaac_gr00t_reference_math(tool_offset):
    """The setup's ``eef_9d`` term: rot6d = first two rows of ``R_base_ee @ TOOL_OFFSET @ DROID_CORRECT``."""
    env = policy_env("droid", {})
    out = core_term(
        EndEffectorPose,
        env,
        entity_name="robot",
        ee_link_name="attachment",
        rotation="rot6d",
        frame_correction_quat=droid_eef_correction_quat(tool_offset),
    )
    robot = env.entities["robot"]
    _, (link,) = robot.find_named_links_idx_local(["attachment"])
    base = Rotation.from_quat(np.roll(robot.get_quat()[0].numpy(), -1))
    ee = Rotation.from_quat(np.roll(robot.get_links_quat(ls_idx_local=[link])[0, 0].numpy(), -1))
    correct = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
    r_base = (base.inv() * ee * Rotation.from_euler("xyz", tool_offset)).as_matrix() @ correct
    np.testing.assert_allclose(out[0, 3:].numpy(), r_base[:2].reshape(6), atol=1e-5)


# ---------------------------------------------------------------------------
# LIBERO: what Isaac-GR00T's gr00t/eval/sim/LIBERO/libero_env.py sends and executes
# ---------------------------------------------------------------------------


def _robosuite_quat2axisangle(quat_xyzw):
    """Verbatim robosuite ``quat2axisangle`` (xyzw), as Isaac-GR00T's ``gr00t/eval/sim/LIBERO/libero_env.py`` copies it."""
    quat = np.array(quat_xyzw, dtype=np.float64)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def test_gr00t_libero_state_is_robosuite_axis_angle_like_the_official_client():
    """The LIBERO end-effector sits near 180 deg about x, where w changes sign. robosuite's axis-angle is not
    canonicalised (its angle exceeds pi when w < 0); ``Rotation.as_rotvec`` would fold it back to [0, pi].
    """
    quat_wxyz = vec(-0.2, 0.9, 0.3, -0.24)[0]
    quat_wxyz = quat_wxyz / quat_wxyz.norm()
    with _gr00t("libero_sim", image_size=(6, 8)) as (policy, server):
        policy.step(_libero_obs(vec(0.3, -0.1, 0.2, *quat_wxyz.tolist(), 0.02, -0.02)))
    state = _sent(server)
    sent = [float(state[f"state.{key}"][0, 0, 0]) for key in ("roll", "pitch", "yaw")]
    expected = _robosuite_quat2axisangle(np.roll(quat_wxyz.numpy(), -1))
    assert np.linalg.norm(expected) > np.pi
    np.testing.assert_allclose(sent, expected, rtol=1e-5)


#: robosuite 1.4 Panda: ``right_hand`` sits 0.1065 m along link7, rotated -45 deg about z (robots/panda/robot.xml);
#: the gripper's ``grip_site`` is 0.097 m further along it (grippers/panda_gripper.xml).
_ROBOSUITE_RIGHT_HAND_FROM_LINK7 = Rotation.from_quat([0.0, 0.0, -0.383, 0.924])
_ROBOSUITE_GRIP_SITE_FROM_LINK7 = np.array([0.0, 0.0, 0.1065 + 0.097])
#: Where the LIBERO arena puts the Panda base; anywhere but the origin tells the world frame from the base frame.
_LIBERO_BASE = (-0.66, 0.0, 0.912)
#: A marker on world +y, clear of the arm.
_MARKER_POS = (-0.5, 0.45, 0.9)


@pytest.fixture(scope="module")
def libero_env():
    """A built LIBERO Panda off the origin, with the setup's own observation groups and the stock task's action terms.

    The agentview camera looks back at the robot from +x like LIBERO's, so world +y is on the right of its render;
    a red marker (no collision) sits on that side.
    """
    from eden.options.entities import BoxOptions
    from eden.options.robots.libero import LiberoPanda
    from eden.options.surfaces import EmissionSurfaceOptions

    return _build_env(
        LiberoPanda(default_root_pos=_LIBERO_BASE),
        {
            "agentview": dict(cam_pos=(0.5, 0.0, 1.6), cam_lookat=_LIBERO_BASE, **_LIBERO_CAMERA),
            "eye_in_hand": dict(
                attach_entity_name="robot",
                attach_link_name="hand",
                attach_offset=(0.05, 0.0, 0.0),
                attach_quat=(0.0, 0.7071067811865475, 0.7071067811865475, 0.0),
                **_LIBERO_CAMERA,
            ),
        },
        Gr00tPolicy.observation_options(policy_setup="libero_sim"),
        Gr00tPolicy.action_options(policy_setup="libero_sim"),
        entities={
            "marker": BoxOptions(
                size=(0.3, 0.3, 0.3),
                default_root_pos=_MARKER_POS,
                fixed=True,
                collision=False,
                surface=EmissionSurfaceOptions(color=(1.0, 0.0, 0.0, 1.0)),
            )
        },
        # The stock libero task's 20 Hz control envelope (0.002 s x 25).
        **_RATES["libero_sim"],
    )


def _robosuite_grip_state(robot) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild robosuite's ``robot0_eef_pos`` and ``quat2axisangle(robot0_eef_quat)`` from Eden's link7."""
    _, (link7,) = robot.find_named_links_idx_local(["link7"])
    pos = robot.get_links_pos(ls_idx_local=[link7])[0, 0].double().numpy()
    rot = Rotation.from_quat(np.roll(robot.get_links_quat(ls_idx_local=[link7])[0, 0].double().numpy(), -1))
    right_hand = rot * _ROBOSUITE_RIGHT_HAND_FROM_LINK7
    return pos + rot.apply(_ROBOSUITE_GRIP_SITE_FROM_LINK7), _robosuite_quat2axisangle(right_hand.as_quat())


def test_libero_images_are_mirrored_left_right_as_libero_training_images_are():
    """libero_env.py's ``[::-1, ::-1]`` on robosuite's bottom-up frame is a left-right mirror of the upright view."""
    from eden.utils.image import resize_uint8

    front, wrist = golden_image(12, 16, seed=3), golden_image(12, 16, seed=4)
    with _gr00t("libero_sim", image_size=(6, 8)) as (policy, server):
        policy.step({"image": front, "wrist_image": wrist, "proprio": _libero_obs()["proprio"]})
    sent = _sent(server)
    for key, frame in (("video.image", front), ("video.wrist_image", wrist)):
        upright = frame[0].numpy()
        np.testing.assert_array_equal(sent[key][0, 0], resize_uint8(np.ascontiguousarray(upright[:, ::-1]), (6, 8)))
        assert not np.array_equal(sent[key][0, 0], resize_uint8(upright, (6, 8)))  # the mirror is not a no-op


def test_libero_agentview_shows_world_plus_y_on_the_left_like_the_training_images(libero_env):
    """Eden's agentview renders world +y on the right; the packed image (LIBERO's view) has it on the left."""
    obs = libero_env.observation_manager.compute()
    with _gr00t("libero_sim", env=libero_env, image_size=(24, 32)) as (policy, server):
        policy.step(obs)

    def red(image) -> np.ndarray:
        image = np.asarray(image, dtype=np.int32)
        return (image[..., 0] > 150) & (image[..., 1] < 80) & (image[..., 2] < 80)

    raw, packed = red(obs["image"][0].numpy()), red(_sent(server)["video.image"][0, 0])
    assert raw.shape == (256, 256)  # GR00T LIBERO's render size
    assert raw[:, 128:].sum() > 0 and raw[:, :128].sum() == 0  # the marker is right of centre in Eden's render
    assert packed[:, :16].sum() > 0 and packed[:, 16:].sum() == 0  # and left of centre in what the server gets


def test_libero_state_from_a_real_env_is_robosuites_world_frame_grip_site_state(libero_env):
    """libero_env.py sends robot0_eef_pos (grip site, world), quat2axisangle(robot0_eef_quat) (the right_hand body)
    and robot0_gripper_qpos, whose second finger is negative (the demo stats put it in [-0.041, 0.001]).
    """
    robot = libero_env.entities["robot"]
    columns = robot.find_named_dofs(["finger_joint1", "finger_joint2"]).columns
    pos = robot.get_dofs_pos().clone()
    pos[:, columns] = torch.tensor([0.03, 0.025])  # Eden's Panda: both fingers open positive
    robot.set_dofs_pos(pos)
    with _gr00t("libero_sim", env=libero_env, image_size=(6, 8)) as (policy, server):
        policy.step(libero_env.observation_manager.compute())
    sent = _sent(server)
    grip_pos, axis_angle = _robosuite_grip_state(robot)
    np.testing.assert_allclose([sent[f"state.{k}"][0, 0, 0] for k in "xyz"], grip_pos, atol=1e-4)
    assert sent["state.z"][0, 0, 0] > 1.0  # world frame: the base stands at z = 0.912
    np.testing.assert_allclose([sent[f"state.{k}"][0, 0, 0] for k in ("roll", "pitch", "yaw")], axis_angle, atol=2e-3)
    np.testing.assert_allclose(sent["state.gripper"][0, 0], [0.03, -0.025], atol=1e-6)


def test_widowx_state_sent_is_simplerenvs_ee_gripper_link_beyond_the_finger_links():
    """SimplerEnv's ``eef_pos`` is ``ee_gripper_link`` in the base frame: ``ee_gripper`` 0.027575 m past the finger
    links' origin along ``gripper_link``'s +x (Interbotix wx250s URDF), not ``gripper_link``, 9.4 cm behind it.
    """
    from eden.options.robots.trossen_widowx250 import WidowX250
    from eden.utils.geom import subtract_frame_transforms, transform_by_quat

    setup = "simpler_env_widowx"
    env = _build_env(
        WidowX250(default_root_pos=(0.3, -0.2, 0.1)),
        {"camera": dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3))},
        Gr00tPolicy.observation_options(policy_setup=setup),
        Gr00tPolicy.action_options(policy_setup=setup),
        **_RATES[setup],
    )
    robot = env.entities["robot"]
    arm = robot.find_named_dofs(list(WidowX250.arm_dofs_name))
    pos = robot.get_dofs_pos().clone()
    pos[:, arm.columns] = torch.tensor([0.3, -0.4, 0.5, 0.6, 0.7, -0.8])
    robot.set_dofs_pos(pos)
    with _gr00t(setup, env=env, image_size=(6, 8)) as (policy, server):
        policy.step(env.observation_manager.compute())

    _, links = robot.find_named_links_idx_local(["base_link", "gripper_link", "left_finger_link", "right_finger_link"])
    link_pos, link_quat = robot.get_links_pos(ls_idx_local=links)[0], robot.get_links_quat(ls_idx_local=links)[0]
    gripper_x = transform_by_quat(torch.tensor([[1.0, 0.0, 0.0]]), link_quat[1:2])
    ee_gripper_link = (link_pos[2:3] + link_pos[3:4]) / 2 + 0.027575 * gripper_x
    expected, _ = subtract_frame_transforms(link_pos[0:1], link_quat[0:1], ee_gripper_link, link_quat[1:2])
    sent = _sent(server)
    np.testing.assert_allclose([sent[f"state.{k}"][0, 0, 0] for k in "xyz"], expected[0].numpy(), atol=1e-5)


@pytest.mark.parametrize(("finger", "openness"), [(0.026, 0.5), (0.012, 0.0), (0.037, 1.0)])
def test_widowx_gripper_state_sent_is_simplerenvs_openness_over_the_urdf_finger_range(finger, openness):
    """SimplerEnv sends ``1 - get_gripper_closedness() = (q - 0.015) / (0.037 - 0.015)`` (ManiSkill2_real2sim's
    wx250s.urdf finger limits). Eden's fingers close further, to 0.01, which reads fully closed, not ``q / 0.037``.
    """
    from eden.options.robots.trossen_widowx250 import WidowX250

    setup = "simpler_env_widowx"
    env = _setup_env(setup)
    robot = env.entities["robot"]
    fingers = robot.find_named_dofs(list(WidowX250.gripper_dofs_name))
    saved = robot.get_dofs_pos().clone()
    pos = saved.clone()
    pos[:, fingers.columns] = torch.tensor([finger, -finger])
    robot.set_dofs_pos(pos)
    try:
        with _gr00t(setup, env=env, image_size=(6, 8)) as (policy, server):
            policy.step(env.observation_manager.compute())
    finally:
        robot.set_dofs_pos(saved)
    assert _sent(server)["state.gripper"][0, 0, 0] == pytest.approx(openness, abs=1e-5)


def test_libero_actions_drive_the_stock_tasks_osc_and_gripper_terms(libero_env):
    """Actions are robosuite OSC_POSE inputs; GR00T's [0, 1] gripper becomes -sign(2g - 1) (1 = open -> -1)."""
    grippers = np.asarray([1.0, 0.7, 0.5, 0.3, 0.0], dtype=np.float32)

    def reply(call: int) -> dict:
        rows = np.zeros((16, 7), dtype=np.float32)
        rows[:, 0] = 0.4  # +x: 0.4 * 0.05 m goal offset per step under robosuite's OSC_POSE scaling
        rows[: len(grippers), 6] = grippers
        return {f"{_SIM}{key}": rows[np.newaxis, :, [i]] for i, key in enumerate(_EE7)}

    robot = libero_env.entities["robot"]
    gripper_term = libero_env.action_manager.terms["gripper"]
    obs, _ = libero_env.reset()
    start, _ = _robosuite_grip_state(robot)
    sent_grippers, finger_targets = [], []
    with _gr00t("libero_sim", reply, env=libero_env, image_size=(6, 8)) as (policy, _):
        for _ in grippers:
            action = policy.step(obs)
            assert set(action) == {"osc_pose", "gripper"}
            np.testing.assert_allclose(action["osc_pose"][0].numpy(), [0.4, 0, 0, 0, 0, 0])
            sent_grippers.append(action["gripper"].item())
            obs, *_ = libero_env.step(action)
            finger_targets.append(gripper_term.processed_action[0, 0].item())
    moved = _robosuite_grip_state(robot)[0] - start
    assert sent_grippers == [-1.0, -1.0, 0.0, 1.0, 1.0]  # open, open, hold, close, close
    assert moved[0] > 0.01 and abs(moved[1]) < 0.2 * moved[0]  # the stock OSC servoed the grip site along +x
    # The stateful LIBERO gripper ramps its finger target: wider on -1, held on 0, narrower on +1.
    assert finger_targets[0] < finger_targets[1] == finger_targets[2]
    assert finger_targets[2] > finger_targets[3] > finger_targets[4]


def test_libero_setup_rejects_an_env_without_the_stock_tasks_action_terms():
    env = policy_env("libero", {})
    with pytest.raises(ValueError, match=r"'osc_pose'.*OperationalSpaceController.*missing"):
        with _gr00t("libero_sim", env=env):
            pass


@pytest.mark.parametrize(
    ("term", "update", "message"),
    [
        ("osc_pose", dict(position_scale=0.5), r"position_scale=0.5 \(stock 0.05\)"),
        # Not action units, but the same row still lands elsewhere: the gains and the control point.
        ("osc_pose", dict(kp=300.0), r"kp=300.0 \(stock 150.0\)"),
        (
            "osc_pose",
            dict(ee_local_pos=(0.0, 0.0, 0.0)),
            r"ee_local_pos=\[0.0, 0.0, 0.0\] \(stock \[0.0, 0.0, 0.0965\]\)",
        ),
    ],
)
def test_libero_setup_rejects_an_action_term_that_differs_from_the_stock_tasks(term, update, message):
    """OSC_POSE rows mean robosuite's controller exactly: units, gains and control point are all compared."""
    actions = libero_task_action_options()
    actions[term] = actions[term].model_copy(update=update)
    env = policy_env("libero", actions)
    with pytest.raises(ValueError, match=rf"Action term {term!r} differs from the stock 'libero' task's: {message}"):
        with _gr00t("libero_sim", env=env):
            pass


@ACTION_TERM_REGISTRY.register()
class Gr00tTestReinterpretingOsc(OperationalSpaceController):
    """Every option field at the stock value, but another class executing the OSC_POSE row."""


def test_libero_setup_refuses_a_subclass_of_the_stock_osc_term():
    actions = libero_task_action_options()
    actions["osc_pose"] = actions["osc_pose"].model_copy(
        update={"name": to_snake_case(Gr00tTestReinterpretingOsc.__name__)}
    )
    env = policy_env("libero", actions)
    with pytest.raises(
        ValueError,
        match=r"'osc_pose' as a OperationalSpaceController exactly, but the env's 'osc_pose' is of type "
        r"Gr00tTestReinterpretingOsc",
    ):
        with _gr00t("libero_sim", env=env):
            pass


def test_libero_setup_accepts_the_stock_action_terms_reloaded_from_a_saved_config():
    """A saved config (``eden inference`` replays config.json) reloads ``ee_local_pos`` / ``ee_local_quat`` as lists
    where the task's builder has tuples: still the stock terms, so construction must not refuse them.
    """
    from eden.options.robots.libero import LiberoPanda

    reloaded = {
        name: type(o).model_validate_json(o.model_dump_json()) for name, o in libero_task_action_options().items()
    }
    assert isinstance(reloaded["osc_pose"].model_dump()["ee_local_pos"], list)  # the premise: a reload changes the type
    env = _build_env(
        LiberoPanda(),
        {
            name: dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3), **_LIBERO_CAMERA)
            for name in ("agentview", "eye_in_hand")
        },
        Gr00tPolicy.observation_options(policy_setup="libero_sim"),
        reloaded,
        **_RATES["libero_sim"],
    )
    with _gr00t("libero_sim", env=env):
        pass


def test_libero_action_options_are_exactly_the_stock_tasks_terms():
    """Only the terms: no ``_option_module_`` / ``_option_class_`` serialization markers for callers to trip on."""
    actions = Gr00tPolicy.action_options(policy_setup="libero_sim")
    assert set(actions) == {"osc_pose", "gripper"}
    assert [options.name for options in actions.values()] == [
        "operational_space_controller",
        "libero_panda_gripper_controller",
    ]


def test_an_env_with_the_stock_libero_tasks_observations_is_refused_at_construction():
    """The stock task's ``policy`` group (agentview_rgb, ...) is another layout: refused before the first step."""
    from eden.options.robots.libero import LiberoPanda
    from eden.tasks.libero.config import _make_observation_options

    stock = _make_observation_options()
    env = _build_env(
        LiberoPanda(),
        dict.fromkeys(("agentview", "eye_in_hand"), dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3))),
        {name: getattr(stock, name) for name in stock.term_keys()},
        Gr00tPolicy.action_options(policy_setup="libero_sim"),
        **_RATES["libero_sim"],
    )
    assert set(env.observation_manager.group_obs_dim) == {"policy"}
    with pytest.raises(ValueError) as excinfo, _gr00t("libero_sim", env=env):
        pass
    message = str(excinfo.value)
    assert "'image'" in message and "'proprio' ((9,)" in message and "'policy'" in message
    assert "Gr00tPolicy.observation_options(policy_setup='libero_sim'" in message


def test_a_proprio_group_of_the_wrong_width_is_refused_at_construction():
    """A WidowX env whose proprio lacks the gripper scalar (7-D, not 8-D) fails at construction, naming both widths."""
    from eden.managers.terms.observations.manipulation import EndEffectorPose
    from eden.options.managers.observations import ObservationGroupOptions
    from eden.options.robots.trossen_widowx250 import WidowX250

    groups = Gr00tPolicy.observation_options(policy_setup="simpler_env_widowx")
    groups["proprio"] = ObservationGroupOptions(
        ee_pose=EndEffectorPose.configure(entity_name="robot", ee_link_name="gripper_link")
    )
    env = _build_env(
        WidowX250(),
        {"camera": dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3))},
        groups,
        Gr00tPolicy.action_options(policy_setup="simpler_env_widowx"),
        **_RATES["simpler_env_widowx"],
    )
    with pytest.raises(ValueError, match=r"'proprio' \(\(8,\), got \(7,\)\)"), _gr00t("simpler_env_widowx", env=env):
        pass


def test_libero_cameras_rendering_below_upstreams_256x256_are_refused_at_construction():
    """The stock libero task renders 128x128; GR00T's LIBERO eval renders 256x256, so the env must override it."""
    from eden.options.robots.libero import LiberoPanda

    setup = "libero_sim"
    env = _build_env(
        LiberoPanda(),
        {
            "agentview": dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3), cam_resolution=(128, 128)),
            "eye_in_hand": dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3), **_LIBERO_CAMERA),
        },
        Gr00tPolicy.observation_options(policy_setup=setup),
        Gr00tPolicy.action_options(policy_setup=setup),
        **_RATES[setup],
    )
    with pytest.raises(ValueError) as excinfo, _gr00t(setup, env=env):
        pass
    message = str(excinfo.value)
    assert "camera 'agentview' (observation term image/" in message and "cam_resolution=(128, 128)" in message
    assert "GR00T's LIBERO checkpoints were trained on 256x256 renders" in message
    assert "cfg.cameras_options.agentview.cam_resolution = (256, 256)" in message


def test_an_env_stepping_at_another_rate_than_the_checkpoints_is_refused_at_construction():
    """exec_horizon and video_stride count env steps: a 50 Hz env would play a 5 Hz WidowX policy 10x too fast."""
    from eden.options.robots.trossen_widowx250 import WidowX250

    setup = "simpler_env_widowx"
    env = _build_env(
        WidowX250(),
        {"camera": dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3))},
        Gr00tPolicy.observation_options(policy_setup=setup),
        Gr00tPolicy.action_options(policy_setup=setup),
        sim_dt=0.005,
        decimation=4,
    )
    with pytest.raises(ValueError) as excinfo, _gr00t(setup, env=env):
        pass
    message = str(excinfo.value)
    assert "Gr00tPolicy.policy_setup='simpler_env_widowx' executes one action row per 0.200000 s (5 Hz)" in message
    assert "env.dt=0.020000 s" in message and "10.00x too fast" in message
    assert "EnvOptions(sim_dt=0.005, decimation=40)" in message
    assert "Do not rescale or resample the actions" in message


# ---------------------------------------------------------------------------
# Arm and gripper terms: exactly what each setup's action_options() builds
# ---------------------------------------------------------------------------


def test_droid_refuses_an_arm_term_that_would_not_execute_the_servers_absolute_targets_as_is():
    """The server already converted its relative joint prediction to absolute targets against the sent state, so the
    ImplicitPDController must add no reference, scale nothing, offset nothing, and take the 7 DROID joints in order.
    Every difference is listed at once (one env build covers them all).
    """
    actions = Gr00tPolicy.action_options(policy_setup=_OXE, gripper_entity_name="gripper")
    actions["ee_controller"] = actions["ee_controller"].model_copy(
        update=dict(
            reference_source=ReferenceSource.DEFAULT,
            scale=2.0,
            offset=0.1,
            dofs_name=[f"joint{i}" for i in range(6, 0, -1)],
        )
    )
    env = policy_env("droid", actions)
    with pytest.raises(ValueError) as excinfo, _gr00t(env=env):
        pass
    message = str(excinfo.value)
    for problem in (
        "reference_source='default'",
        "effective_scale=[2.0] (from scale=2.0, scale_ratio=None, needs 1)",
        "offset=0.1 (needs 0)",
        "width 6",
        "dofs_name=['joint6', 'joint5', 'joint4', 'joint3', 'joint2', 'joint1'] (needs ['joint1',",
    ):
        assert problem in message
    assert f"Gr00tPolicy.action_options(policy_setup='{_OXE}')" in message


def test_droid_names_a_reference_source_configured_as_a_plain_string():
    """Term options keep ``reference_source`` as the str they were given (or reloaded from JSON), not the enum.

    A 'default' reference folds default_dofs_pos into effective_offset; only the reference is reported, not a zero offset.
    """
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    actions = Gr00tPolicy.action_options(policy_setup=_OXE, gripper_entity_name="gripper")
    actions["ee_controller"] = ImplicitPDController.configure(
        entity_name="robot", dofs_name=[f"joint{i}" for i in range(1, 8)], reference_source="default"
    )
    env = policy_env("droid", actions)
    assert type(env.action_manager.terms["ee_controller"].reference_source) is str  # the case under test
    with pytest.raises(ValueError, match=r"reference_source='default', which adds a reference") as excinfo:
        with _gr00t(env=env):
            pass
    assert "offset=" not in str(excinfo.value)


def test_droid_refuses_an_arm_term_of_another_type():
    from eden.managers.terms.actions.joint_actions import ExplicitPDController

    actions = Gr00tPolicy.action_options(policy_setup=_OXE, gripper_entity_name="gripper")
    actions["ee_controller"] = ExplicitPDController.configure(
        entity_name="robot", dofs_name=[f"joint{i}" for i in range(1, 8)], reference_source=ReferenceSource.ZERO
    )
    env = policy_env("droid", actions)
    with pytest.raises(ValueError, match=r"'ee_controller' of type ImplicitPDController exactly.*ExplicitPDController"):
        with _gr00t(env=env):
            pass


def test_widowx_refuses_an_arm_term_that_is_not_differential_ik():
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options.robots.trossen_widowx250 import WidowX250

    actions = Gr00tPolicy.action_options(policy_setup="simpler_env_widowx")
    actions["ee_controller"] = ImplicitPDController.configure(entity_name="robot", dofs_name=WidowX250.arm_dofs_name)
    env = policy_env("widowx", actions)
    with pytest.raises(ValueError, match=r"of type DifferentialIKController exactly.*ImplicitPDController") as excinfo:
        with _gr00t("simpler_env_widowx", env=env):
            pass
    assert "Gr00tPolicy.action_options(policy_setup='simpler_env_widowx')" in str(excinfo.value)


@pytest.mark.parametrize(
    ("update", "field"),
    [(dict(threshold=0.3), "threshold"), (dict(sticky_repeats=15), "sticky_repeats")],
)
def test_widowx_refuses_a_gripper_term_deciding_otherwise_than_upstreams_plain_half_threshold(update, field):
    """Upstream's WidowXBridgeEnv opens on ``openness > 0.5``, with no latch: a task's own threshold or latch is refused."""
    actions = Gr00tPolicy.action_options(policy_setup="simpler_env_widowx")
    actions["gripper_controller"] = actions["gripper_controller"].model_copy(update=update)
    env = policy_env("widowx", actions)
    with pytest.raises(ValueError, match=rf"{field}: env term has {update[field]!r}"):
        with _gr00t("simpler_env_widowx", env=env):
            pass


@pytest.mark.parametrize(
    ("update", "field"),
    [(dict(open_threshold=0.4, close_threshold=0.6), "open_threshold"), (dict(continuous=True), "continuous")],
)
def test_droid_refuses_a_gripper_term_deciding_otherwise_than_the_reference_clients_strict_half_threshold(
    update, field
):
    """main_gr00t.py closes on each row's ``gripper_position > 0.5``: hysteresis or continuous targets are refused."""
    actions = Gr00tPolicy.action_options(policy_setup=_OXE, gripper_entity_name="gripper")
    actions["gripper_controller"] = actions["gripper_controller"].model_copy(update=update)
    env = policy_env("droid", actions)
    with pytest.raises(ValueError, match=rf"{field}: env term has {update[field]!r}"), _gr00t(env=env):
        pass


# ---------------------------------------------------------------------------
# State groups: the plain current reading of the arm and gripper the action terms drive
# ---------------------------------------------------------------------------


def _env_with_observations(setup: str, groups: dict):
    """``setup``'s robot with ``groups`` as its observations and the setup's own ``action_options()``."""
    from eden.options.robots import Droid
    from eden.options.robots.libero import LiberoPanda
    from eden.options.robots.trossen_widowx250 import WidowX250

    robot, kwargs, cameras, _, _ = _SETUPS[setup]
    return _build_env(
        {"widowx": WidowX250, "droid": Droid, "libero": LiberoPanda}[robot](),
        {name: dict(cam_pos=(1.0, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3), **extra) for name, extra in cameras.items()},
        groups,
        Gr00tPolicy.action_options(policy_setup=setup, **kwargs),
        **_RATES[setup],
    )


def test_droid_refuses_a_gripper_state_normalised_over_another_stroke_than_the_gripper_term():
    """``gripper_close_pos`` passed to observation_options() only would skew the closedness sent in every request."""
    groups = Gr00tPolicy.observation_options(policy_setup=_OXE, gripper_entity_name="gripper", gripper_close_pos=0.7)
    with pytest.raises(
        ValueError,
        match=r"Gripper state joint_state/gripper and action term 'gripper_controller' describe different grippers: "
        r"close_pos=0\.7 but 'gripper_controller' has close_action=0\.8\. Fix: pass the same gripper entity",
    ):
        with _gr00t(env=_env_with_observations(_OXE, groups)):
            pass


@pytest.mark.parametrize(
    ("group", "term", "settings", "problem"),
    [
        ("eef_9d", "pose", dict(scale=2.0), r"pose\.scale=2\.0"),
        ("joint_state", "joint_pos", dict(noise=GaussianNoise.configure(std=0.01)), r"joint_pos\.noise=NoiseOptions"),
    ],
)
def test_droid_refuses_a_state_group_that_is_not_the_plain_current_reading(group, term, settings, problem):
    """A scale rescales the state the server reads, and noise corrupts it; neither changes the group's width."""
    from eden.options.managers.observations import ObservationGroupOptions

    groups = Gr00tPolicy.observation_options(policy_setup=_OXE, gripper_entity_name="gripper")
    terms = {name: getattr(groups[group], name) for name in groups[group].term_keys()}
    terms[term] = terms[term].model_copy(update=settings)
    groups[group] = ObservationGroupOptions(**terms)
    with pytest.raises(
        ValueError,
        match=rf"Observation group '{group}' is not the plain current reading .*{problem}.*Fix: build '{group}' with "
        rf"Gr00tPolicy\.observation_options\(policy_setup='{_OXE}'\)\['{group}'\]",
    ):
        with _gr00t(env=_env_with_observations(_OXE, groups)):
            pass


def test_droid_refuses_a_joint_state_in_another_order_than_the_arm_term():
    """The server replies targets in the order of the joint state it was sent; the arm term takes DROID's order."""
    reversed_arm = [f"joint{i}" for i in range(7, 0, -1)]
    groups = Gr00tPolicy.observation_options(
        policy_setup=_OXE, gripper_entity_name="gripper", arm_dofs_name=reversed_arm
    )
    with pytest.raises(ValueError) as excinfo, _gr00t(env=_env_with_observations(_OXE, groups)):
        pass
    message = str(excinfo.value)
    assert f"'joint_pos' reads 'robot' {reversed_arm} but the arm term drives 'robot' ['joint1'," in message
    assert "arm_dofs_name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']" in message
    assert "Do not reorder the server's rows in the client" in message


def test_widowx_refuses_a_proprio_that_is_not_the_plain_current_reading():
    """pack_observations slices proprio by column: a scaled pose reaches the server as SimplerEnv's state."""
    from eden.options.managers.observations import ObservationGroupOptions

    groups = Gr00tPolicy.observation_options(policy_setup="simpler_env_widowx")
    proprio = groups["proprio"]
    groups["proprio"] = ObservationGroupOptions(
        ee_pose=proprio.ee_pose.model_copy(update=dict(scale=0.5)), gripper=proprio.gripper
    )
    with pytest.raises(
        ValueError, match=r"Observation group 'proprio' is not the plain current reading .*ee_pose\.scale"
    ):
        with _gr00t("simpler_env_widowx", env=_env_with_observations("simpler_env_widowx", groups)):
            pass


@pytest.mark.parametrize(
    ("term", "update", "message"),
    [
        # pi's LIBERO proprio has the same terms and width but negates q2 in the client, not in the group.
        ("gripper_qpos", dict(scale=1.0), r"gripper_qpos\.scale=1\.0 \(expected \(1\.0, -1\.0\)\)"),
        ("ee_pose", dict(noise=GaussianNoise.configure(std=0.01)), r"ee_pose\.noise=NoiseOptions"),
    ],
)
def test_libero_refuses_a_proprio_that_is_not_the_plain_reading_with_robosuites_finger_sign(term, update, message):
    """pack_observations slices proprio by column and trusts the group for robosuite's [+q1, -q2] gripper layout."""
    from eden.options.managers.observations import ObservationGroupOptions

    groups = Gr00tPolicy.observation_options(policy_setup="libero_sim")
    proprio = groups["proprio"]
    terms = {name: getattr(proprio, name) for name in ("ee_pose", "gripper_qpos")}
    terms[term] = terms[term].model_copy(update=update)
    groups["proprio"] = ObservationGroupOptions(**terms)
    with pytest.raises(ValueError, match=rf"Observation group 'proprio' is not the plain current reading .*{message}"):
        with _gr00t("libero_sim", env=_env_with_observations("libero_sim", groups)):
            pass


# ---------------------------------------------------------------------------
# Golden records: wire contract and controller outputs per setup (see the policies conftest)
# ---------------------------------------------------------------------------

#: name -> (setup, configure kwargs, reply, steps, obs)
_GOLDEN_CASES = {
    "gr00t_simpler_env_widowx": (
        "simpler_env_widowx",
        {},
        _chunk_reply(8, _EE7_KEYS, gripper_key="gripper", prefix=_SIM),
        6,
        lambda s: {"image": golden_image(seed=s), "proprio": ee8(1.0 - 0.2 * s)},
    ),
    "gr00t_libero_sim": (
        "libero_sim",
        {},
        _chunk_reply(16, _EE7_KEYS, gripper_key="gripper", prefix=_SIM),
        6,
        # [x, y, z, quat, q_left, -q_right]: what the setup's proprio group reads off an Eden Panda at 0.02 m.
        lambda s: {
            "image": golden_image(seed=s),
            "wrist_image": golden_image(seed=30 + s),
            "proprio": vec(0.3, -0.1, 0.2, *TILT_QUAT, 0.02, -0.02),
        },
    ),
    "gr00t_oxe_droid": (
        _OXE,
        dict(video_horizon=2, video_stride=3),
        _oxe_reply,
        10,
        lambda s: {
            "image": golden_image(seed=s),
            "wrist_image": golden_image(seed=40 + s),
            "eef_9d": vec(0.3, -0.1, 0.2, 1, 0, 0, 0, 1, 0),
            "joint_state": joint8(0.05 * s),
        },
    ),
}


@pytest.mark.parametrize("name", list(_GOLDEN_CASES))
def test_wire_contract_and_controller_outputs_match_the_golden_record(name):
    setup, configure, reply, steps, obs_fn = _GOLDEN_CASES[name]
    with ZmqPolicyServer("msgpack_numpy", reply) as server:
        env = _setup_env(setup)
        options = Gr00tPolicy.configure(
            policy_setup=setup, image_size=(6, 8), host=server.host, port=server.port, **configure
        )
        policy = Gr00tPolicy(env, options)
        try:
            policy.set_task(GOLDEN_TASK)
            got = record_golden(policy, server, obs_fn, steps)
        finally:
            policy.close()
    assert_matches_golden(name, got)
