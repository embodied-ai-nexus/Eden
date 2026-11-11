from __future__ import annotations

import contextlib
import re

import numpy as np
import pytest
import torch

import eden as en
from eden.constants import ReferenceSource
from eden.extensions.policies.adapters import DROID_ARM_DOFS
from eden.extensions.policies.galaxeavla.policy import GalaxeaVlaPolicy
from eden.extensions.policies.galaxeavla.setups import GalaxeaDroidFrankaSetup
from eden.options import EnvOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions
from eden.options.robots import Droid
from eden.options.robots.aloha_vx300s import Aloha
from eden.utils.configs import EdenConfig
from tests.extensions.policies.conftest import (
    GOLDEN_TASK,
    assert_matches_golden,
    chunk,
    joint8,
    policy_env,
    record_golden,
)
from tests.extensions.policies.conftest import image as golden_image
from tests.support.envs import build_env
from tests.support.policy_servers import WebsocketPolicyServer

# ---------------------------------------------------------------------------
# DROID Franka gripper: the setup forwards the server's openness as closedness (1 - openness; NaN when
# the AR head omits the key); the env's GripperDecisionController, built from action_options(), decides.
# Each test drives a real policy against a server replaying a scripted openness per reply, and feeds
# the policy's command through the env's own action manager.
# ---------------------------------------------------------------------------

_CLOSE_POS = 0.8  # Robotiq driver angle at full closure

_ENVS: dict[str, object] = {}


def _env(robot: type, actions: dict, proprio) -> object:
    """``robot`` with the ``actions`` terms and the one observation group ``proprio``, stepping at 15 Hz; cached.

    No cameras: the tests feed the images.
    """
    # model_dump, not repr: an options repr elides nested term values (swapped gripper positions would collide).
    key = repr((robot.__name__, {name: term.model_dump() for name, term in actions.items()}, proprio.model_dump()))
    if key not in _ENVS:
        _ENVS[key] = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 15.0, decimation=1),
                scene_options=en.scenes.Empty(robot=robot()),
                action_options=ActionManagerOptions(**actions),
                observation_options=ObservationManagerOptions(proprio=proprio),
            )
        )
    return _ENVS[key]


def _droid_env(actions: dict, proprio=None):
    """Eden's Droid rig (the Robotiq on its own ``gripper`` entity) with ``actions`` and the setup's ``proprio``."""
    if proprio is None:
        proprio = GalaxeaVlaPolicy.observation_options(policy_setup="droid_franka", gripper_entity_name="gripper")
        proprio = proprio["proprio"]
    return _env(Droid, actions, proprio)


def _droid_obs(s: int = 0) -> dict:
    return {"image": golden_image(seed=s), "wrist_image": golden_image(seed=5 + s), "proprio": joint8(0.3)}


def _set_measured_closedness(env, closedness: float) -> None:
    """Teleport both Robotiq drivers ``closedness`` of the way to the closed angle."""
    gripper = env.entities["gripper"]
    _, idx = gripper.find_named_dofs_idx_local(["left_driver_joint", "right_driver_joint"], preserve_order=True)
    gripper.set_dofs_pos(torch.full((1, 2), _CLOSE_POS * closedness), dofs_idx_local=idx)


class _DroidFranka:
    """One ``droid_franka`` policy, its env and the env's gripper term."""

    def __init__(self, policy: GalaxeaVlaPolicy, server: WebsocketPolicyServer):
        self.policy, self.server, self.env = policy, server, policy.env
        self.term = self.env.action_manager.terms["gripper_controller"]

    @property
    def diagnostics(self) -> dict:
        return self.policy._setup.adapter.last_diagnostics  # what evaluators' gripper plots read

    def command(self) -> torch.Tensor:
        """One policy step: the gripper command the policy hands the action manager."""
        return self.policy.step(_droid_obs())["gripper_controller"]

    def step(self) -> torch.Tensor:
        """One policy step through the env's action manager: the gripper term's finger targets."""
        self.env.action_manager.compute({"gripper_controller": self.command()})
        return self.term.processed_action


@contextlib.contextmanager
def _droid_rig(reply, *, measured_closedness: float = 0.0, **gripper_options):
    """A ``droid_franka`` policy against a server answering ``reply(call)`` (calls count from 1)."""
    gripper_options.setdefault("gripper_decode_mode", "binary")
    with WebsocketPolicyServer("msgpack_galaxea", reply, control_keys=("__reset__",)) as server:
        env = _droid_env(
            GalaxeaVlaPolicy.action_options(
                policy_setup="droid_franka", gripper_entity_name="gripper", **gripper_options
            )
        )
        env.action_manager.reset()  # the env is shared: drop the previous test's gripper state
        _set_measured_closedness(env, measured_closedness)
        options = GalaxeaVlaPolicy.configure(
            policy_setup="droid_franka", host=server.host, port=server.port, **gripper_options
        )
        policy = GalaxeaVlaPolicy(env, options)
        try:
            yield _DroidFranka(policy, server)
        finally:
            policy.close()


@contextlib.contextmanager
def _droid_franka(script, *, measured_closedness: float = 0.0, **gripper_options):
    """``script``: the gripper openness of each successive reply; ``None`` omits the key."""
    replies = iter(script)

    def reply(call: int) -> dict:
        openness = next(replies)
        action = {"right_arm": np.zeros(7, dtype=np.float32)}
        if openness is not None:
            action["right_gripper"] = np.asarray([openness], dtype=np.float32)
        return {"action": action, "need_obs": True}

    with _droid_rig(reply, measured_closedness=measured_closedness, **gripper_options) as rig:
        yield rig


def _fingers(closedness: float) -> torch.Tensor:
    return torch.full((1, 2), _CLOSE_POS * closedness)


def test_unpack_forwards_closedness_and_nan_for_a_missing_gripper():
    with _droid_franka([0.3, None]) as rig:
        torch.testing.assert_close(rig.command(), torch.full((1, 1), 0.7))
        assert rig.diagnostics["gripper_source"] == "server_prediction"

        held = rig.command()
        assert held.shape == (1, 1) and torch.isnan(held).all()
        assert rig.diagnostics["gripper_source"] == "omitted"


@pytest.mark.parametrize(
    ("closedness", "expected"),
    [(0.2, 0.2), (0.8, 0.8), (0.81, 1.0), (1.0, 1.0)],
)
def test_continuous_snap_gripper_rule(closedness: float, expected: float):
    with _droid_franka([1.0 - closedness], gripper_decode_mode="continuous-snap", gripper_snap_threshold=0.8) as rig:
        torch.testing.assert_close(rig.step(), _fingers(expected))
        assert rig.diagnostics["gripper_decode_mode"] == "continuous-snap"


def test_continuous_snap_missing_gripper_commands_the_measured_position():
    """droid-franka-client eval/main.py: an omitted gripper key sends clip(gripper_position), then snaps > 0.8."""
    with _droid_franka([0.3, None, None], gripper_decode_mode="continuous-snap") as rig:
        rig.step()

        _set_measured_closedness(rig.env, 0.4)  # fingers are part-way to the commanded 0.7
        torch.testing.assert_close(rig.step(), _fingers(0.4))
        _set_measured_closedness(rig.env, 0.9)  # above the snap threshold
        torch.testing.assert_close(rig.step(), _fingers(1.0))


def test_droid_franka_defaults_follow_the_reference_client():
    assert GalaxeaVlaPolicy.gripper_decode_mode == "continuous-snap"
    gripper = GalaxeaDroidFrankaSetup.action_options(gripper_entity_name="gripper")["gripper_controller"]
    assert (gripper.continuous, gripper.snap_threshold, gripper.on_missing) == (True, 0.8, "measured")


def test_droid_gripper_controller_drives_both_robotiq_drivers():
    actions = GalaxeaDroidFrankaSetup.action_options(
        gripper_entity_name="gripper",
        gripper_dof_name="left_driver_joint",
        gripper_close_pos=0.8,
    )
    gripper = actions["gripper_controller"]

    assert gripper.name == "gripper_decision_controller"
    assert gripper.entity_name == "gripper"
    assert gripper.dofs_name == ["left_driver_joint", "right_driver_joint"]
    assert gripper.input == "closedness"
    assert gripper.open_action == 0.0
    assert gripper.close_action == 0.8
    assert gripper.continuous is True


@pytest.mark.parametrize(
    ("threshold", "openness", "closed"),
    # gripper_close_threshold is a closedness threshold: close iff 1 - openness > threshold.
    [(0.5, 0.2, True), (0.5, 0.5, False), (0.5, 0.8, False), (0.7, 0.5, False), (0.7, 0.25, True), (0.3, 0.6, True)],
)
def test_binary_gripper_thresholds_closedness(threshold: float, openness: float, closed: bool):
    with _droid_franka([openness], gripper_close_threshold=threshold) as rig:
        torch.testing.assert_close(rig.step(), _fingers(float(closed)))


@pytest.mark.parametrize(("threshold", "measured_closedness", "closed"), [(0.5, 0.5, False), (0.7, 0.4, False)])
def test_first_missing_gripper_seed_thresholds_measured_closedness(
    threshold: float, measured_closedness: float, closed: bool
):
    with _droid_franka([None], measured_closedness=measured_closedness, gripper_close_threshold=threshold) as rig:
        torch.testing.assert_close(rig.step(), _fingers(float(closed)))


def test_missing_gripper_holds_explicit_close_while_physical_gripper_is_still_open():
    with _droid_franka([0.0, None], measured_closedness=0.0) as rig:
        torch.testing.assert_close(rig.step(), _fingers(1.0))

        _set_measured_closedness(rig.env, 0.2)  # moving, but not past the 0.5 threshold yet
        torch.testing.assert_close(rig.step(), _fingers(1.0))


def test_missing_gripper_holds_explicit_open_while_physical_gripper_is_still_closed():
    with _droid_franka([1.0, None], measured_closedness=1.0) as rig:
        torch.testing.assert_close(rig.step(), _fingers(0.0))

        _set_measured_closedness(rig.env, 0.8)
        torch.testing.assert_close(rig.step(), _fingers(0.0))


@pytest.mark.parametrize(("measured_closedness", "expected_closedness"), [(0.2, 0.0), (0.8, 1.0)])
def test_first_missing_gripper_after_reset_initializes_from_measured_state(
    measured_closedness: float, expected_closedness: float
):
    with _droid_franka([None], measured_closedness=measured_closedness) as rig:
        torch.testing.assert_close(rig.step(), _fingers(expected_closedness))
        assert rig.diagnostics["gripper_source"] == "omitted"


def test_reset_clears_explicit_gripper_command_latch():
    with _droid_franka([0.0, None], measured_closedness=0.0) as rig:
        rig.step()  # explicit close

        rig.env.action_manager.reset(torch.tensor([0]))
        torch.testing.assert_close(rig.step(), _fingers(0.0))


@pytest.mark.parametrize(
    ("sim_closedness", "galaxea_openness"),
    [(-0.1, 1.0), (0.0, 1.0), (0.25, 0.75), (1.0, 0.0), (1.1, 0.0)],
)
def test_droid_observation_wiring_converts_closedness_to_galaxea_openness(
    sim_closedness: float, galaxea_openness: float
):
    obs = {**_droid_obs(), "proprio": torch.tensor([[0.0] * 7 + [sim_closedness]])}
    with _droid_franka([0.5], gripper_decode_mode="continuous-snap") as rig:
        rig.policy.step(obs)
        sent = rig.server.records[0]["state"]["right_gripper"]
    np.testing.assert_allclose(sent, [galaxea_openness])


def test_droid_proprio_is_closedness_even_with_the_gripper_on_the_arm_entity():
    """The reference sends DROID gripper_position in [0, 1]; never the raw driver angle."""
    groups = GalaxeaDroidFrankaSetup.observation_options(gripper_entity_name=None, gripper_close_pos=0.8)
    gripper = groups["proprio"].gripper
    assert gripper.normalize == "closedness"
    assert gripper.entity_name == "robot" and gripper.close_pos == 0.8


# ---------------------------------------------------------------------------
# Golden records: wire contract and controller outputs per setup (see the policies conftest)
# ---------------------------------------------------------------------------

_ARM = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
#: ALOHA's finger joint: 0.04 m fully open, 0 closed (its URDF limits).
_ALOHA_GRIPPER = dict(gripper_open_pos=0.04, gripper_close_pos=0.0)
#: GalaxeaVLA's R1-Lite (two 6-DOF arms + a gripper each) mapped onto Eden's two-arm ALOHA rig by name.
_R1_ON_ALOHA = {
    **_ALOHA_GRIPPER,
    **{
        f"{side}_{part}": value
        for side in ("left", "right")
        for part, value in (
            ("arm_dofs_name", [f"{side}/{name}" for name in _ARM]),
            ("gripper_dof_name", f"{side}/left_finger"),
        )
    },
}


def _r1_env(setup: str, dofs: dict, *, actions: dict | None = None, observation_dofs: dict | None = None, proprio=None):
    """ALOHA with the R1 action terms (``actions``, else ``action_options(**dofs)``) and the ``proprio`` group
    (``proprio``, else that of ``observation_options(**(observation_dofs or dofs))``); cached.
    """
    if actions is None:
        actions = GalaxeaVlaPolicy.action_options(policy_setup=setup, **dofs)
    if proprio is None:
        proprio = GalaxeaVlaPolicy.observation_options(policy_setup=setup, **(observation_dofs or dofs))["proprio"]
    return _env(Aloha, actions, proprio)


def _r1_reply(call: int) -> dict:
    row = chunk(1, 14, call)[0]
    return {
        "action": {
            "left_arm": row[0:6],
            "left_gripper": np.asarray([0.0 if call % 2 else 100.0], np.float32),  # HDAS stroke
            "right_arm": row[7:13],
            "right_gripper": np.asarray([20.0], np.float32),
        },
        "need_obs": True,
    }


def _droid_franka_reply(call: int) -> dict:
    row = chunk(1, 8, call)[0]
    action = {"right_arm": row[:7]}
    if call % 3 != 0:  # every third reply omits the gripper key (AR head drop)
        action["right_gripper"] = np.asarray([0.1 if call % 2 else 0.95], np.float32)
    return {"action": action, "need_obs": True}


#: name -> (env(setup, action_options kwargs), setup, action_options kwargs, configure kwargs, reply, steps, obs)
_GOLDEN_CASES = {
    "galaxea_r1lite": (
        _r1_env,
        "galaxea_r1lite",
        _R1_ON_ALOHA,
        {},
        _r1_reply,
        4,
        lambda s: {
            "head_image": golden_image(seed=s),
            "left_wrist_image": golden_image(seed=10 + s),
            "right_wrist_image": golden_image(seed=20 + s),
            "proprio": torch.linspace(-0.4, 0.4, 14).unsqueeze(0),
        },
    ),
    # Eden's Droid rig carries the Robotiq on its own ``gripper`` entity. The client forwards the same
    # closedness / NaN stream in either gripper_decode_mode (the env term decides), so one record covers both.
    "galaxea_droid_franka_snap": (
        lambda setup, kwargs: _droid_env(GalaxeaVlaPolicy.action_options(policy_setup=setup, **kwargs)),
        "droid_franka",
        dict(gripper_entity_name="gripper", gripper_decode_mode="continuous-snap"),
        dict(gripper_decode_mode="continuous-snap"),
        _droid_franka_reply,
        6,
        _droid_obs,
    ),
}


@pytest.mark.parametrize("name", list(_GOLDEN_CASES))
def test_wire_contract_and_controller_outputs_match_the_golden_record(name):
    make_env, setup, action_kwargs, configure, reply, steps, obs_fn = _GOLDEN_CASES[name]
    with WebsocketPolicyServer("msgpack_galaxea", reply, control_keys=("__reset__",)) as server:
        env = make_env(setup, action_kwargs)
        options = GalaxeaVlaPolicy.configure(policy_setup=setup, host=server.host, port=server.port, **configure)
        policy = GalaxeaVlaPolicy(env, options)
        try:
            policy.set_task(GOLDEN_TASK)
            got = record_golden(policy, server, obs_fn, steps)
        finally:
            policy.close()
    assert_matches_golden(name, got)


def test_the_client_declares_exactly_its_official_servers_reset_api():
    # GalaxeaVLA's serve_policy.py resets on '__reset__' and has no set-task API.
    options = GalaxeaVlaPolicy.configure()
    assert (options.set_task_endpoint, options.reset_endpoint) == (None, "__reset__")


# ---------------------------------------------------------------------------
# DROID Franka request protocol: the droid-franka-client's need_obs rule
# ---------------------------------------------------------------------------


def _need_obs_reply(need_obs):
    """A DROID reply per call whose ``need_obs`` is ``need_obs(call)``; ``None`` leaves the key out."""

    def reply(call: int) -> dict:
        out = {"action": {"right_arm": np.zeros(7, np.float32), "right_gripper": np.asarray([0.5], np.float32)}}
        if (flag := need_obs(call)) is not None:
            out["need_obs"] = flag
        return out

    return reply


def _sent_observation(record: dict) -> bool:
    if record == {}:
        return False
    assert set(record) == {"images", "state", "task", "frequency", "embodiment_type"}, record
    return True


def test_droid_sends_an_empty_request_while_the_server_serves_its_cached_chunk():
    """serve_policy.py with action_steps=3: need_obs is true only on the reply that exhausts the chunk."""
    with _droid_rig(_need_obs_reply(lambda call: call % 3 == 0)) as rig:
        for _ in range(5):
            rig.command()
        assert [_sent_observation(r) for r in rig.server.records] == [True, False, False, True, False]


def test_droid_reset_and_set_task_rearm_the_full_observation():
    with _droid_rig(_need_obs_reply(lambda call: False)) as rig:
        rig.command()
        rig.command()
        rig.policy.reset()
        rig.command()
        rig.policy.set_task("another task")
        rig.command()
        assert [_sent_observation(r) for r in rig.server.records] == [True, False, True, True]
        assert rig.server.control == [{"__reset__": True}, {"__reset__": True}]


def test_droid_reply_without_need_obs_asks_for_the_full_observation():
    """The reference reads ``response.get("need_obs", True)``."""
    with _droid_rig(_need_obs_reply(lambda call: None)) as rig:
        rig.command()
        rig.command()
        assert [_sent_observation(r) for r in rig.server.records] == [True, True]


# ---------------------------------------------------------------------------
# Reply parts: aliases, omitted parts, non-finite values
# ---------------------------------------------------------------------------


def test_droid_alias_keys_decode_like_the_canonical_ones():
    arm = np.linspace(-0.3, 0.3, 7, dtype=np.float32)

    def canonical(call):
        return {"action": {"right_arm": arm, "right_gripper": np.asarray([0.2], np.float32)}, "need_obs": True}

    def aliased(call):
        return {"action": {"joint_position": arm, "gripper": np.asarray([0.2], np.float32)}, "need_obs": True}

    commands = []
    for reply in (canonical, aliased):
        with _droid_rig(reply) as rig:
            commands.append(rig.policy.step(_droid_obs()))
    assert commands[0].keys() == commands[1].keys() == {"ee_controller", "gripper_controller"}
    for key in commands[0]:
        torch.testing.assert_close(commands[0][key], commands[1][key])
    torch.testing.assert_close(commands[1]["ee_controller"], torch.from_numpy(arm).unsqueeze(0))
    torch.testing.assert_close(commands[1]["gripper_controller"], torch.full((1, 1), 0.8))


def test_droid_reply_without_an_arm_raises_naming_the_missing_part():
    with _droid_rig(lambda call: {"action": {"gripper": np.asarray([0.2], np.float32)}, "need_obs": True}) as rig:
        with pytest.raises(
            KeyError,
            match=r"no 'right_arm' action \(accepted keys: \['right_arm', 'joint_position'\]\); its action holds \['gripper'\]",
        ):
            rig.command()


@pytest.mark.parametrize("gripper", [[float("nan")], [float("inf")]])
def test_droid_non_finite_gripper_raises_instead_of_reading_as_omitted(gripper):
    def reply(call):
        return {"action": {"right_arm": np.zeros(7, np.float32), "right_gripper": np.asarray(gripper, np.float32)}}

    with _droid_rig(reply, gripper_decode_mode="continuous-snap") as rig:
        with pytest.raises(ValueError, match="'right_gripper' is not finite"):
            rig.command()


# ---------------------------------------------------------------------------
# R1 setups: per-setup part layout, arm widths, omitted / non-finite parts
# ---------------------------------------------------------------------------

#: R1 Pro's 7-DOF arms on ALOHA: each side's 6 arm joints plus its second finger joint (7 distinct DOFs),
#: the gripper on the first finger joint.
_R1PRO_ON_ALOHA = {
    **_ALOHA_GRIPPER,
    **{
        f"{side}_{part}": value
        for side in ("left", "right")
        for part, value in (
            ("arm_dofs_name", [f"{side}/{name}" for name in _ARM] + [f"{side}/right_finger"]),
            ("gripper_dof_name", f"{side}/left_finger"),
        )
    },
}


def _r1_obs(width: int):
    return lambda s: {
        "head_image": golden_image(seed=s),
        "left_wrist_image": golden_image(seed=10 + s),
        "right_wrist_image": golden_image(seed=20 + s),
        "proprio": torch.linspace(-0.4, 0.4, width).unsqueeze(0),
    }


@contextlib.contextmanager
def _r1_rig(setup: str, dofs: dict, reply):
    with WebsocketPolicyServer("msgpack_galaxea", reply, control_keys=("__reset__",)) as server:
        env = _r1_env(setup, dofs)
        env.action_manager.reset()  # the env is shared between tests
        policy = GalaxeaVlaPolicy(
            env, GalaxeaVlaPolicy.configure(policy_setup=setup, host=server.host, port=server.port)
        )
        try:
            yield policy, server
        finally:
            policy.close()


def _r1_parts(arm: int, call: int, *, drop: tuple[str, ...] = (), **override) -> dict:
    row = chunk(1, 2 * arm + 2, call)[0]
    parts = {
        "left_arm": row[:arm],
        "left_gripper": np.asarray([10.0], np.float32),  # HDAS stroke: 0 closed .. 100 open
        "right_arm": row[arm + 1 : 2 * arm + 1],
        "right_gripper": np.asarray([90.0], np.float32),
    }
    parts.update({key: np.asarray(value, np.float32) for key, value in override.items()})
    return {"action": {k: v for k, v in parts.items() if k not in drop}, "need_obs": False}


def test_setups_declare_their_layout_and_derive_the_action_width():
    from eden.extensions.policies.galaxeavla.setups import SETUP_REGISTRY

    widths = {key: (dict(cls.part_layout), cls.action_dim) for key, cls in SETUP_REGISTRY.items()}
    assert widths == {
        "galaxea_r1lite": ({"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}, 14),
        "galaxea_r1pro": ({"left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1}, 16),
        "droid_franka": ({"right_arm": 7, "right_gripper": 1}, 8),
    }


def test_r1pro_packs_and_decodes_7_dof_arms_and_sends_the_observation_every_tick():
    with _r1_rig("galaxea_r1pro", _R1PRO_ON_ALOHA, lambda call: _r1_parts(7, call)) as (policy, server):
        commands = [policy.step(_r1_obs(16)(s)) for s in range(2)]

        # need_obs=False does not change the R1 requests: upstream's R1-Lite scheduler always sends obs.
        assert all("state" in record for record in server.records)
        state = server.records[0]["state"]
        proprio = _r1_obs(16)(0)["proprio"][0].numpy()
        np.testing.assert_array_equal(state["left_arm"], proprio[0:7])
        np.testing.assert_array_equal(state["right_arm"], proprio[8:15])
        # Gripper columns are openness, sent as the HDAS stroke (0 closed .. 100 open).
        np.testing.assert_allclose(state["left_gripper"], 100.0 * np.clip(proprio[7:8], 0.0, 1.0), rtol=1e-6)
        np.testing.assert_allclose(state["right_gripper"], 100.0 * np.clip(proprio[15:16], 0.0, 1.0), rtol=1e-6)
        assert server.records[0]["embodiment_type"] == "galaxea_r1pro"

        sent = _r1_parts(7, 1)["action"]
        torch.testing.assert_close(commands[0]["left_arm_controller"], torch.from_numpy(sent["left_arm"])[None])
        torch.testing.assert_close(commands[0]["right_arm_controller"], torch.from_numpy(sent["right_arm"])[None])
        torch.testing.assert_close(commands[0]["left_gripper_controller"], torch.full((1, 1), 0.9))  # stroke 10
        policy.env.action_manager.compute(commands[0])  # the env's own terms take the 7-D arm targets

        with pytest.raises(ValueError, match=r"16-D obs\['proprio'\].*got 14 values"):
            policy.step(_r1_obs(14)(0))


@pytest.mark.parametrize(
    ("setup", "factory", "kwargs", "match"),
    [
        ("galaxea_r1lite", "observation_options", _R1PRO_ON_ALOHA, "left_arm_dofs_name has 7 names.*is 6 joints"),
        ("galaxea_r1lite", "action_options", _R1PRO_ON_ALOHA, "left_arm_dofs_name has 7 names.*is 6 joints"),
        ("galaxea_r1pro", "action_options", _R1_ON_ALOHA, "left_arm_dofs_name has 6 names.*is 7 joints"),
        (
            "galaxea_r1lite",
            "action_options",
            {**_ALOHA_GRIPPER, "right_arm_dofs_name": ["j"] * 5},
            "right_arm_dofs_name has 5 names",
        ),
        ("droid_franka", "observation_options", {"arm_dofs_name": ["j"] * 6}, "arm_dofs_name has 6 names.*is 7"),
        ("droid_franka", "action_options", {"arm_dofs_name": ["j"] * 8}, "arm_dofs_name has 8 names.*is 7"),
    ],
)
def test_arm_dof_lists_must_match_the_setup_layout(setup, factory, kwargs, match):
    with pytest.raises(ValueError, match=match):
        getattr(GalaxeaVlaPolicy, factory)(policy_setup=setup, **kwargs)


def test_r1pro_policy_on_an_env_built_with_6_dof_arms_raises_at_construction():
    env = policy_env("aloha", GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA))
    with pytest.raises(ValueError, match="'left_arm_controller' commands 6 DOFs.*7-joint left_arm"):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1pro"))


def _replace_term(actions: dict, name: str, **update) -> dict:
    """``actions`` with term ``name``'s options updated by ``update``."""
    return {**actions, name: actions[name].model_copy(update=update)}


def _explicit_pd(term) -> object:
    """An ExplicitPDController on the same DOFs as the ImplicitPDController options ``term``, zero reference."""
    from eden.managers.terms.actions.joint_actions import ExplicitPDController

    return ExplicitPDController.configure(
        entity_name=term.entity_name, dofs_name=term.dofs_name, reference_source=term.reference_source
    )


_ARM_TERM_REFUSALS = [
    # A task's own arm term: the default reference adds the default pose to every absolute target.
    (
        dict(reference_source=ReferenceSource.DEFAULT),
        r"ImplicitPDController has reference_source='default', which adds a reference to every command",
    ),
    # A config reloaded from JSON holds the plain str: the message must still render.
    (dict(reference_source="default"), r"ImplicitPDController has reference_source='default', which adds"),
    (dict(scale=0.5), r"effective_scale=\[0\.5\] \(from scale=0\.5, scale_ratio=None, needs 1\)"),
    (dict(offset=0.1), r"offset=0\.1 \(needs 0\)"),
    (None, r"must be exactly an ImplicitPDController .* is of type ExplicitPDController"),
]


@pytest.mark.parametrize(("update", "match"), _ARM_TERM_REFUSALS)
def test_r1_refuses_an_arm_term_that_would_not_apply_absolute_targets(update, match):
    actions = GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA)
    if update is None:
        actions = {**actions, "right_arm_controller": _explicit_pd(actions["right_arm_controller"])}
    else:
        actions = _replace_term(actions, "right_arm_controller", **update)
    env = _r1_env("galaxea_r1lite", _R1_ON_ALOHA, actions=actions)
    with pytest.raises(ValueError, match=match) as raised:
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1lite"))
    assert "'right_arm_controller'" in str(raised.value)
    assert "GalaxeaVlaPolicy.action_options(policy_setup='galaxea_r1lite', ...)" in str(raised.value)


@pytest.mark.parametrize(("update", "match"), _ARM_TERM_REFUSALS)
def test_droid_franka_refuses_an_arm_term_that_would_not_apply_absolute_targets(update, match):
    actions = GalaxeaVlaPolicy.action_options(policy_setup="droid_franka", gripper_entity_name="gripper")
    if update is None:
        actions = {**actions, "ee_controller": _explicit_pd(actions["ee_controller"])}
    else:
        actions = _replace_term(actions, "ee_controller", **update)
    env = policy_env("droid", actions)
    with pytest.raises(ValueError, match=match) as raised:
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="droid_franka"))
    assert "'ee_controller'" in str(raised.value)
    assert "GalaxeaVlaPolicy.action_options(policy_setup='droid_franka', ...)" in str(raised.value)


def test_droid_franka_refuses_an_env_without_its_arm_term_or_with_another_width():
    actions = GalaxeaVlaPolicy.action_options(policy_setup="droid_franka", gripper_entity_name="gripper")
    renamed = {"arm_controller" if name == "ee_controller" else name: term for name, term in actions.items()}
    with pytest.raises(ValueError, match=r"'ee_controller' is missing \(action terms: \['arm_controller'"):
        GalaxeaVlaPolicy(policy_env("droid", renamed), GalaxeaVlaPolicy.configure(policy_setup="droid_franka"))

    six = _replace_term(actions, "ee_controller", dofs_name=actions["ee_controller"].dofs_name[:6])
    with pytest.raises(ValueError, match=r"has width 6 \(DROID's arm has 7 joints\).*arm_dofs_name="):
        GalaxeaVlaPolicy(policy_env("droid", six), GalaxeaVlaPolicy.configure(policy_setup="droid_franka"))


def test_droid_franka_refuses_droids_joints_out_of_droids_order_even_with_a_matching_proprio():
    """The checkpoint reads row ``i`` as DROID's joint ``i``: a consistently reversed rig permutes state and commands."""
    reversed_dofs = list(DROID_ARM_DOFS)[::-1]
    actions = GalaxeaVlaPolicy.action_options(
        policy_setup="droid_franka", gripper_entity_name="gripper", arm_dofs_name=reversed_dofs
    )
    env = _droid_env(actions, _droid_proprio(arm_dofs_name=reversed_dofs))
    with pytest.raises(
        ValueError,
        match=r"dofs_name=\['joint7', .*\] \(needs \['joint1', .*'joint7'\], in that order\)\. Fix: .* Do not undo "
        r"the difference in the client \(.*reordering the rows\)",
    ):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="droid_franka"))


@pytest.mark.parametrize("setup", ["galaxea_r1lite", "droid_franka"])
def test_an_arm_term_offset_set_after_build_is_refused(setup):
    """A ``set_dofs_pos_offset()`` (joint-zero domain randomization) shifts every absolute target the server sends.

    On its own env: the offset is the env's state, which the cached envs share across tests.
    """
    if setup == "droid_franka":
        robot, dofs, arm_name = Droid, dict(gripper_entity_name="gripper"), "ee_controller"
    else:
        robot, dofs, arm_name = Aloha, _R1_ON_ALOHA, "right_arm_controller"
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 15.0, decimation=1),
            scene_options=en.scenes.Empty(robot=robot()),
            action_options=ActionManagerOptions(**GalaxeaVlaPolicy.action_options(policy_setup=setup, **dofs)),
            observation_options=ObservationManagerOptions(
                proprio=GalaxeaVlaPolicy.observation_options(policy_setup=setup, **dofs)["proprio"]
            ),
        )
    )
    try:
        arm = env.action_manager.terms[arm_name]
        arm.set_dofs_pos_offset(torch.full((env.num_envs, arm.action_dim), 0.1, device=env.device))
        with pytest.raises(
            ValueError,
            match=rf"{arm_name!r} to be executed as they are, but that ImplicitPDController has effective_offset up "
            r"to 0\.1 in magnitude, .*from a set_dofs_pos_offset\(\) call",
        ):
            GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup=setup))
    finally:
        env.destroy()


def test_r1_refuses_proprio_and_action_grippers_with_different_open_close_positions():
    """Swapped open / close positions between the two factories would invert the gripper on the wire."""
    swapped = {**_R1_ON_ALOHA, "gripper_open_pos": 0.0, "gripper_close_pos": 0.04}
    env = _r1_env("galaxea_r1lite", _R1_ON_ALOHA, observation_dofs=swapped)
    with pytest.raises(
        ValueError,
        match=r"Gripper state proprio/left_gripper and action term 'left_gripper_controller' describe different "
        r"grippers: open_pos=0\.0 but 'left_gripper_controller' has open_action=0\.04; close_pos=0\.04 but "
        r"'left_gripper_controller' has close_action=0\.0\. Fix: pass the same gripper entity",
    ):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1lite"))


def test_r1_refuses_an_env_without_the_proprio_it_sends():
    # Actions only: pack_observations sends the arms' and both grippers' state, so they have no source.
    env = policy_env("aloha", GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA))
    with pytest.raises(ValueError, match=r"no observation group 'proprio'; available: \[\]") as raised:
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1lite"))
    assert "GalaxeaVlaPolicy.observation_options(policy_setup='galaxea_r1lite', ...)" in str(raised.value)

    groups = GalaxeaVlaPolicy.observation_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA)["proprio"]
    flipped = groups.model_copy(
        update={"left_gripper": groups.left_gripper.model_copy(update={"normalize": "closedness"})}
    )
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 15.0, decimation=1),
            scene_options=en.scenes.Empty(robot=Aloha()),
            action_options=ActionManagerOptions(
                **GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA)
            ),
            observation_options=ObservationManagerOptions(proprio=flipped),
        )
    )
    with pytest.raises(ValueError, match=r"has a GripperState with normalize='closedness' there"):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1lite"))


def test_droid_franka_refuses_a_proprio_that_is_not_8_wide():
    with _droid_franka([0.5]) as rig:
        with pytest.raises(
            ValueError,
            match=r"8-D obs\['proprio'\] \[right_arm\(7\), right_gripper\(1\)\], got 7 values\. Fix: build the env's "
            r"observations with GalaxeaVlaPolicy\.observation_options\(policy_setup='droid_franka', \.\.\.\)",
        ):
            rig.policy.step({**_droid_obs(), "proprio": torch.zeros((1, 7))})


def _r1_obs_at(offset: float) -> dict:
    """``_r1_obs(14)(0)`` with every measured joint shifted by ``offset``."""
    obs = _r1_obs(14)(0)
    return {**obs, "proprio": obs["proprio"] + offset}


def _arm_target(command: dict, side: str) -> np.ndarray:
    return command[f"{side}_arm_controller"][0].numpy()


def test_r1_omitted_arm_holds_the_measured_joints_before_its_first_target():
    """serve_policy.py drops any part the AR head did not predict, arms included; upstream's client holds it."""
    reply = lambda call: _r1_parts(6, call, drop=("left_arm",))  # noqa: E731
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, reply) as (policy, _):
        obs = _r1_obs_at(0.0)
        command = policy.step(obs)
        np.testing.assert_array_equal(_arm_target(command, "left"), obs["proprio"][0, 0:6].numpy())
        np.testing.assert_array_equal(_arm_target(command, "right"), reply(1)["action"]["right_arm"])
        manager = policy.env.action_manager
        manager.compute(command)  # the arm term gets a finite target, never the row's NaN
        torch.testing.assert_close(manager.terms["left_arm_controller"].processed_action, obs["proprio"][:, 0:6])


def test_r1_omitted_arm_holds_its_previous_target_while_the_other_arm_moves():
    replies = {1: _r1_parts(6, 1), 2: _r1_parts(6, 2, drop=("left_arm",)), 3: _r1_parts(6, 3, drop=("left_arm",))}
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, replies.__getitem__) as (policy, _):
        commands = [policy.step(_r1_obs_at(0.1 * s)) for s in range(3)]  # the arm moved: the hold is not re-measured
        for call, command in enumerate(commands, start=1):
            np.testing.assert_array_equal(_arm_target(command, "left"), replies[1]["action"]["left_arm"])
            np.testing.assert_array_equal(_arm_target(command, "right"), replies[call]["action"]["right_arm"])


def test_r1_arm_hold_restarts_from_the_measured_joints_after_a_reset():
    """The held target belongs to its episode: a reset never replays the previous episode's arm target."""
    replies = {1: _r1_parts(6, 1), 2: _r1_parts(6, 2, drop=("right_arm",))}
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, replies.__getitem__) as (policy, _):
        policy.step(_r1_obs_at(0.0))
        policy.reset()  # episode boundary
        obs = _r1_obs_at(0.2)
        command = policy.step(obs)
        np.testing.assert_array_equal(_arm_target(command, "right"), obs["proprio"][0, 7:13].numpy())
        np.testing.assert_array_equal(_arm_target(command, "left"), replies[2]["action"]["left_arm"])


@pytest.mark.parametrize(
    "action",
    [{}, {"joint_position": np.zeros(6, np.float32)}],
    ids=["empty", "only-foreign-keys"],
)
def test_r1_reply_with_none_of_its_parts_raises_instead_of_freezing_the_robot(action):
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, lambda call: {"action": action, "need_obs": True}) as (policy, _):
        with pytest.raises(KeyError, match=r"reply carries none of its action parts \(accepted keys: \['left_arm', "):
            policy.step(_r1_obs(14)(0))


def test_r1_omitted_gripper_holds_within_an_episode_and_reads_the_fingers_after_a_reset():
    """An omitted gripper is NaN: hold the previous target, but never replay the previous episode's."""
    omit = lambda call: _r1_parts(6, call, drop=("left_gripper",))  # noqa: E731
    replies = {1: _r1_parts(6, 1, left_gripper=[0.0]), 2: omit(2), 3: omit(3)}
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, replies.__getitem__) as (policy, _):
        manager, robot = policy.env.action_manager, policy.env.entities["robot"]
        term = manager.terms["left_gripper_controller"]
        _, finger = robot.find_named_dofs_idx_local(["left/left_finger"], preserve_order=True)
        robot.set_dofs_pos(torch.full((1, 1), 0.04), dofs_idx_local=finger)  # fingers measured fully open

        manager.compute(policy.step(_r1_obs(14)(0)))  # stroke 0: close
        torch.testing.assert_close(term.processed_action, torch.full((1, 1), 0.0))
        command = policy.step(_r1_obs(14)(1))
        assert torch.isnan(command["left_gripper_controller"]).all()
        manager.compute(command)
        torch.testing.assert_close(term.processed_action, torch.full((1, 1), 0.0))  # held, fingers still open

        manager.reset()  # episode boundary
        policy.reset()
        manager.compute(policy.step(_r1_obs(14)(2)))
        torch.testing.assert_close(term.processed_action, torch.full((1, 1), 0.04))  # the measured fingers


def test_r1_gripper_wire_is_the_hdas_stroke_both_ways():
    """GalaxeaVLA R1 data: gripper position 0 = closed, 100 = open (R1_GRIPPER_*_STROKE)."""
    reply = lambda call: _r1_parts(6, call, left_gripper=[25.0], right_gripper=[100.0])  # noqa: E731
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, reply) as (policy, server):
        obs = _r1_obs(14)(0)
        obs["proprio"][0, 6], obs["proprio"][0, 13] = 0.75, 1.2  # openness of the Eden fingers (1.2 overshoots)
        command = policy.step(obs)
        state = server.records[0]["state"]
        np.testing.assert_allclose(state["left_gripper"], [75.0])
        np.testing.assert_allclose(state["right_gripper"], [100.0])
        torch.testing.assert_close(command["left_gripper_controller"], torch.full((1, 1), 0.75))  # closedness
        torch.testing.assert_close(command["right_gripper_controller"], torch.full((1, 1), 0.0))
        manager = policy.env.action_manager
        manager.compute(command)  # continuous: open_pos + (close_pos - open_pos) * closedness
        torch.testing.assert_close(manager.terms["left_gripper_controller"].processed_action, torch.full((1, 1), 0.01))


def test_r1_gripper_dof_positions_are_required():
    with pytest.raises(TypeError, match="gripper_open_pos"):
        GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", gripper_close_pos=0.0)
    groups = GalaxeaVlaPolicy.observation_options(policy_setup="galaxea_r1lite", **_ALOHA_GRIPPER)
    gripper = groups["proprio"].left_gripper
    assert (gripper.normalize, gripper.open_pos, gripper.close_pos) == ("openness", 0.04, 0.0)


@pytest.mark.parametrize(
    "override",
    [{"left_gripper": [float("nan")]}, {"right_arm": [0.1, 0.2, float("nan"), 0.0, 0.0, 0.0]}],
)
def test_r1_non_finite_part_raises_instead_of_reading_as_omitted(override):
    (part,) = override
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, lambda call: _r1_parts(6, call, **override)) as (policy, _):
        with pytest.raises(ValueError, match=f"'{part}' is not finite"):
            policy.step(_r1_obs(14)(0))


@pytest.mark.parametrize(
    "override",
    [
        {"left_arm": np.arange(12) * 0.01},
        {"right_arm": np.zeros((1, 12))},
        {"left_gripper": [10.0, 20.0]},
        {"left_arm": np.zeros((0, 6))},
    ],
    ids=["flat", "batched", "gripper", "empty_chunk"],
)
def test_r1_part_of_another_width_raises_instead_of_executing_part_of_it(override):
    """``serve_policy.py`` serves one step per reply: 12 values for a 6-joint arm are another width, not two steps."""
    (part,) = override
    shape = re.escape(str(np.shape(override[part])))
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, lambda call: _r1_parts(6, call, **override)) as (policy, _):
        with pytest.raises(
            ValueError,
            match=rf"'galaxea_r1lite' reply part '{part}' has shape {shape}, but one step of the part is \d wide .* "
            r"Do NOT trim or pad the part to fit",
        ):
            policy.step(_r1_obs(14)(0))


def test_r1_part_with_a_leading_chunk_dim_executes_its_first_row():
    rows = (np.arange(12, dtype=np.float32) * 0.01).reshape(2, 6)
    with _r1_rig("galaxea_r1lite", _R1_ON_ALOHA, lambda call: _r1_parts(6, call, left_arm=rows)) as (policy, _):
        command = policy.step(_r1_obs(14)(0))["left_arm_controller"]
    torch.testing.assert_close(command, torch.from_numpy(rows[:1]).to(command))


def test_droid_arm_alias_of_twice_its_width_raises_instead_of_executing_its_first_7():
    """The reference client cuts 14 values to their first 7; this client refuses them (another width, not 2 steps)."""

    def reply(call: int) -> dict:
        return {"action": {"joint_position": np.zeros(14, np.float32)}, "need_obs": True}

    with _droid_rig(reply) as rig:
        with pytest.raises(ValueError, match=r"'Droid_Franka' reply part 'right_arm' has shape \(14,\), but one step"):
            rig.command()


# ---------------------------------------------------------------------------
# Configuration: no field or kwarg a setup would ignore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "entity_name",
        "arm_dofs_name",
        "gripper_entity_name",
        "gripper_dof_name",
        "gripper_close_pos",
        "gripper_n_dofs",
        "primary_camera_name",
        "wrist_camera_name",
        "gripper_high_is_close",
    ],
)
def test_removed_policy_fields_are_rejected(field):
    with pytest.raises(ValueError, match=f"invalid: \\['{field}'\\]"):
        GalaxeaVlaPolicy.configure(**{field: None})


@pytest.mark.parametrize(
    ("setup", "factory", "kwarg"),
    [
        ("galaxea_r1lite", "action_options", "arm_dofs_name"),  # a DROID kwarg under R1
        ("galaxea_r1lite", "observation_options", "gripper_scale"),  # removed: the stroke conversion is fixed
        ("droid_franka", "observation_options", "head_camera_name"),  # an R1 kwarg under DROID
        ("droid_franka", "action_options", "gripper_decode_mod"),  # a misspelling
    ],
)
def test_setup_factories_reject_kwargs_they_do_not_take(setup, factory, kwarg):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{kwarg}'"):
        getattr(GalaxeaVlaPolicy, factory)(policy_setup=setup, **{kwarg: None})


@pytest.mark.parametrize(
    ("setup", "field", "value"),
    [
        ("galaxea_r1lite", "gripper_decode_mode", "binary"),
        ("galaxea_r1pro", "gripper_snap_threshold", 0.6),
        ("galaxea_r1lite", "debug_gripper", True),
        ("galaxea_r1lite", "gripper_close_threshold", 0.3),
    ],
)
def test_fields_the_selected_setup_ignores_raise_when_changed(setup, field, value):
    dofs = _R1PRO_ON_ALOHA if setup == "galaxea_r1pro" else _R1_ON_ALOHA
    env = policy_env("aloha", GalaxeaVlaPolicy.action_options(policy_setup=setup, **dofs))
    with pytest.raises(ValueError, match=f"{field}={value!r} has no effect under policy_setup='{setup}'"):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup=setup, **{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pred_action_horizon", 16),
        ("exec_horizon", 4),
        ("chunk_temporal_ensemble_steps", 2),
        ("chunk_temporal_ensemble_dims", 7),
        ("image_size", (224, 224)),
        ("response_action_key", "actions"),
    ],
)
def test_client_knobs_no_setup_reads_raise_when_changed(field, value):
    """The server serves one step per request at native image size; these ClientMixin knobs would do nothing."""
    env = policy_env("aloha", GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA))
    with pytest.raises(ValueError, match=re.escape(f"{field}={value!r} has no effect under any policy_setup")):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1lite", **{field: value}))


_UNREAD_THRESHOLDS = [
    (
        dict(gripper_decode_mode="binary", gripper_snap_threshold=0.6),
        "gripper_snap_threshold=0.6 has no effect under gripper_decode_mode='binary'; only "
        "gripper_decode_mode='continuous-snap' reads it",
    ),
    (
        dict(gripper_decode_mode="continuous-snap", gripper_close_threshold=0.3),
        "gripper_close_threshold=0.3 has no effect under gripper_decode_mode='continuous-snap'; only "
        "gripper_decode_mode='binary' reads it",
    ),
]


@pytest.mark.parametrize(("options", "message"), _UNREAD_THRESHOLDS)
def test_droid_action_options_refuse_the_threshold_the_decode_mode_ignores(options, message):
    where = "GalaxeaVlaPolicy.action_options(policy_setup='droid_franka', ...)"
    with pytest.raises(ValueError, match=re.escape(f"{where}: {message}")):
        GalaxeaVlaPolicy.action_options(policy_setup="droid_franka", gripper_entity_name="gripper", **options)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (
            dict(gripper_decode_mode="snap"),
            r"GalaxeaVlaPolicy\.policy_setup='droid_franka': gripper_decode_mode='snap' is not one of \['binary', "
            r"'continuous-snap'\]\. Fix: configure\(gripper_decode_mode='continuous-snap'\)",
        ),
        (
            dict(gripper_snap_threshold=1.5),
            r"GalaxeaVlaPolicy\.policy_setup='droid_franka': gripper_snap_threshold=1\.5 is not a closedness in "
            r"\[0, 1\]\. Fix: configure\(gripper_snap_threshold=0\.8\)",
        ),
    ],
)
def test_droid_policy_refuses_an_unknown_decode_mode_or_a_snap_threshold_outside_0_1(options, match):
    env = policy_env(
        "droid", GalaxeaVlaPolicy.action_options(policy_setup="droid_franka", gripper_entity_name="gripper")
    )
    with pytest.raises(ValueError, match=match):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="droid_franka", **options))


@pytest.mark.parametrize(("options", "message"), _UNREAD_THRESHOLDS)
def test_droid_policy_refuses_the_threshold_the_decode_mode_ignores(options, message):
    env = _droid_env(
        GalaxeaVlaPolicy.action_options(
            policy_setup="droid_franka",
            gripper_entity_name="gripper",
            gripper_decode_mode=options["gripper_decode_mode"],
        )
    )
    with pytest.raises(ValueError, match=re.escape(f"GalaxeaVlaPolicy.configure(): {message}")):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="droid_franka", **options))


# ---------------------------------------------------------------------------
# Control rate: the client sends one request, and the server serves one step, per env.step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "dt", "speed"),
    [("galaxea_r1lite", 1.0 / 50.0, "3.33x too fast"), ("droid_franka", 1.0 / 7.5, "2.00x too slow")],
)
def test_refuses_an_env_stepping_at_another_rate_than_control_frequency(setup, dt, speed):
    if setup == "droid_franka":
        env = policy_env(
            "droid", GalaxeaVlaPolicy.action_options(policy_setup=setup, gripper_entity_name="gripper"), dt=dt
        )
    else:
        env = policy_env("aloha", GalaxeaVlaPolicy.action_options(policy_setup=setup, **_R1_ON_ALOHA), dt=dt)
    with pytest.raises(
        ValueError,
        match=rf"GalaxeaVlaPolicy\.policy_setup='{setup}' executes one action row per 0\.066667 s \(15 Hz\) "
        rf"control step, as control_frequency=15\.0 tells the server, .* env\.dt={dt:.6f} s .* {speed}\. "
        r"Fix: build the env with sim_dt \* decimation = 1/15 s.* Do not change control_frequency to match the env "
        r"instead: .* \(GalaxeaVLA's R1-Lite config\.toml and droid-franka-client both run at 15 Hz\)\.$",
    ):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup=setup))


@pytest.mark.parametrize(
    ("setup", "frequency", "match"),
    [
        ("galaxea_r1lite", 0.0, r"control_frequency must be a positive number of Hz, got 0\.0"),
        ("galaxea_r1lite", float("nan"), r"control_frequency must be a positive number of Hz, got nan"),
        # droid-franka-client sends an int: 15.5 would reach the server as 15 Hz.
        (
            "droid_franka",
            15.5,
            r"GalaxeaVlaPolicy\.policy_setup='droid_franka' sends 'frequency' as an int.*control_frequency=15\.5 is "
            r"not a whole number",
        ),
    ],
)
def test_refuses_a_control_frequency_the_wire_cannot_carry(setup, frequency, match):
    if setup == "droid_franka":
        env = policy_env("droid", GalaxeaVlaPolicy.action_options(policy_setup=setup, gripper_entity_name="gripper"))
    else:
        env = policy_env("aloha", GalaxeaVlaPolicy.action_options(policy_setup=setup, **_R1_ON_ALOHA))
    with pytest.raises(ValueError, match=match):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup=setup, control_frequency=frequency))


# ---------------------------------------------------------------------------
# proprio <-> actions: pack_observations cuts obs['proprio'] by column
# ---------------------------------------------------------------------------


def _regroup(group: ObservationGroupOptions, order: list[str]) -> ObservationGroupOptions:
    """``group``'s terms declared in ``order``."""
    return ObservationGroupOptions(**{name: getattr(group, name) for name in order})


def _r1_proprio(**dofs) -> ObservationGroupOptions:
    return GalaxeaVlaPolicy.observation_options(policy_setup="galaxea_r1lite", **{**_R1_ON_ALOHA, **dofs})["proprio"]


_LEFT_ARM, _RIGHT_ARM = _R1_ON_ALOHA["left_arm_dofs_name"], _R1_ON_ALOHA["right_arm_dofs_name"]


@pytest.mark.parametrize(
    ("proprio", "match"),
    [
        # Same terms, same 14 columns, another order: split_parts would read the right arm as the left gripper.
        (
            lambda: _regroup(_r1_proprio(), ["left_arm", "right_arm", "left_gripper", "right_gripper"]),
            r"GalaxeaVlaPolicy\.policy_setup='galaxea_r1lite' cuts obs\['proprio'\] by column into \[left_arm\(6\), "
            r"left_gripper\(1\), right_arm\(6\), "
            r"right_gripper\(1\)\], but the env's 'proprio' group is \[left_arm\(6\), right_arm\(6\), "
            r"left_gripper\(1\), right_gripper\(1\)\]",
        ),
        (
            lambda: _r1_proprio(left_arm_dofs_name=_RIGHT_ARM, right_arm_dofs_name=_LEFT_ARM),
            r"(?s)'proprio/left_arm' reads 'robot' \['right/waist', .*\] while 'left_arm_controller' drives 'robot' "
            r"\['left/waist', .*Fix: pass the same left_arm_entity_name= / left_arm_dofs_name=, "
            r"right_arm_entity_name= / right_arm_dofs_name= to",
        ),
        (
            lambda: _r1_proprio(left_arm_dofs_name=_LEFT_ARM[::-1]),
            r"'proprio/left_arm' reads 'robot' \['left/wrist_rotate', .*\] while 'left_arm_controller' drives",
        ),
        (
            lambda: _r1_proprio().model_copy(update={"history_length": 2}),
            r"Observation group 'proprio' is not the plain current reading the client packs by column: "
            r"history_length=2",
        ),
    ],
    ids=["term_order", "arms_swapped", "arm_reordered", "history"],
)
def test_r1_refuses_a_proprio_group_pack_observations_would_cut_wrong(proprio, match):
    env = _r1_env("galaxea_r1lite", _R1_ON_ALOHA, proprio=proprio())
    with pytest.raises(ValueError, match=match):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="galaxea_r1lite"))


def _droid_proprio(**kwargs) -> ObservationGroupOptions:
    return GalaxeaVlaPolicy.observation_options(policy_setup="droid_franka", gripper_entity_name="gripper", **kwargs)[
        "proprio"
    ]


def _droid_gripper_as_openness() -> ObservationGroupOptions:
    group = _droid_proprio()
    return group.model_copy(update={"gripper": group.gripper.model_copy(update={"normalize": "openness"})})


@pytest.mark.parametrize(
    ("proprio", "match"),
    [
        (None, r"no observation group 'proprio'; available: \[\]"),
        (
            # pack_observations sends 1 - closedness as the openness the checkpoint was trained on.
            _droid_gripper_as_openness,
            r"GalaxeaVlaPolicy\.policy_setup='droid_franka' sends the gripper's measured closedness, read from a "
            r"GripperState\(normalize='closedness'\) term at "
            r"'proprio/gripper', but the env has a GripperState with normalize='openness' there",
        ),
        (
            lambda: _droid_proprio(gripper_close_pos=0.7),
            r"Gripper state proprio/gripper and action term 'gripper_controller' describe different grippers: "
            r"close_pos=0\.7 but 'gripper_controller' has close_action=0\.8",
        ),
        (
            lambda: _droid_proprio(arm_dofs_name=list(DROID_ARM_DOFS)[::-1]),
            r"(?s)'proprio/joint_pos' reads 'robot' \[.*\] while 'ee_controller' drives 'robot' .*Fix: pass the same "
            r"arm_entity_name= / arm_dofs_name= to",
        ),
        (
            lambda: _regroup(_droid_proprio(), ["gripper", "joint_pos"]),
            r"into \[joint_pos\(7\), gripper\(1\)\], but the env's 'proprio' group is \[gripper\(1\), "
            r"joint_pos\(7\)\]",
        ),
    ],
    ids=["missing", "openness", "close_pos", "arm_reordered", "term_order"],
)
def test_droid_refuses_a_proprio_group_pack_observations_would_misread(proprio, match):
    actions = GalaxeaVlaPolicy.action_options(policy_setup="droid_franka", gripper_entity_name="gripper")
    env = policy_env("droid", actions) if proprio is None else _droid_env(actions, proprio())
    with pytest.raises(ValueError, match=match):
        GalaxeaVlaPolicy(env, GalaxeaVlaPolicy.configure(policy_setup="droid_franka"))


def _construct(env, setup: str) -> None:
    """Build (and close) a ``setup`` policy on ``env``: construction connects, so a server answers the handshake."""
    with WebsocketPolicyServer("msgpack_galaxea", lambda call: {}, control_keys=("__reset__",)) as server:
        env.action_manager.reset()  # the env is shared between tests
        GalaxeaVlaPolicy(
            env, GalaxeaVlaPolicy.configure(policy_setup=setup, host=server.host, port=server.port)
        ).close()


def test_droid_accepts_a_proprio_naming_droids_joints_by_pattern():
    """The proprio check compares the joints the ``DofsPos`` reads, resolved, not the names it was configured with."""
    from eden.managers.terms.observations.proprio import DofsPos

    joint_pos = DofsPos.configure(entity_name="robot", dofs_name=["joint[1-7]"])
    proprio = _droid_proprio().model_copy(update={"joint_pos": joint_pos})
    _construct(
        _droid_env(
            GalaxeaVlaPolicy.action_options(policy_setup="droid_franka", gripper_entity_name="gripper"), proprio
        ),
        "droid_franka",
    )


def test_r1_accepts_an_arm_term_with_an_all_zero_per_joint_offset():
    """``offset`` may be a per-joint dict; one of zeros is the identity, not an offset."""
    actions = GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **_R1_ON_ALOHA)
    actions = _replace_term(actions, "right_arm_controller", offset=dict.fromkeys(_RIGHT_ARM, 0.0))
    _construct(_r1_env("galaxea_r1lite", _R1_ON_ALOHA, actions=actions), "galaxea_r1lite")


# ---------------------------------------------------------------------------
# Gripper DOF positions: GripperDecisionController takes a float, never an int
# ---------------------------------------------------------------------------


def test_r1_gripper_positions_given_as_ints_drive_the_env():
    ints = {**_R1_ON_ALOHA, "gripper_open_pos": 0.04, "gripper_close_pos": 0}
    with _r1_rig("galaxea_r1lite", ints, lambda call: _r1_parts(6, call, left_gripper=[0.0])) as (policy, _):
        manager = policy.env.action_manager
        manager.compute(policy.step(_r1_obs(14)(0)))  # stroke 0: fully closed
        torch.testing.assert_close(manager.terms["left_gripper_controller"].processed_action, torch.zeros((1, 1)))


@pytest.mark.parametrize("value", [1, np.float32(0.5)])
def test_factories_hand_gripper_positions_on_as_python_floats(value):
    r1 = {**_R1_ON_ALOHA, "gripper_open_pos": value, "gripper_close_pos": value}
    droid = dict(policy_setup="droid_franka", gripper_entity_name="gripper", gripper_close_pos=value)
    positions = [
        GalaxeaVlaPolicy.action_options(policy_setup="galaxea_r1lite", **r1)["left_gripper_controller"].close_action,
        GalaxeaVlaPolicy.observation_options(policy_setup="galaxea_r1lite", **r1)["proprio"].left_gripper.close_pos,
        GalaxeaVlaPolicy.action_options(**droid)["gripper_controller"].close_action,
        GalaxeaVlaPolicy.observation_options(**droid)["proprio"].gripper.close_pos,
    ]
    assert [(type(v), v) for v in positions] == [(float, float(value))] * 4
