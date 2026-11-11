"""``CompositeDeployment`` against a real arm + welded hand env: each part on its own wire, driven as one robot."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Union
import threading

import numpy as np
import pytest
import torch

import eden.extensions.deployment.base as deployment_base
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY, bind_deployment, resolve_deployment_for
from eden.extensions.deployment.base import DeploymentBase
from eden.extensions.deployment.composite import PART_HOOKS, CompositeDeployment
from eden.extensions.deployment.utils.operator_input import CMD_INIT, CMD_START
from eden.extensions.deployment.utils.record import load_deploy_record
from eden.extensions.deployment.utils.state import RobotStateEntity
from eden.options.entities import RobotOptions
from eden.options.robots.composite.base import DexManipulatorOptions, HumanoidOptions
from eden.utils.cli.tasks import (
    _check_composite_deploy,
    _DeploySlot,
    _parse_part_options,
    _repoint_swapped_slot_controllers,
    _route_backend_flag,
    _warn_on_unswitchable_slot_gains,
)
from eden.utils.geom import quat_error_magnitude
from tests.extensions.deployment.conftest import TWO_LINK_MJCF, CachedEntityQuat
from tests.support.deploy import (
    FirmwareGainsLoopbackDeployment,
    LoopbackDeployment,
    ScriptedOperator,
    deployments,
    robot_state,
)

#: A two-finger hand whose palm is a free root body, so it welds onto a parent link by that body.
WELDED_HAND_MJCF = """
<mujoco model="welded_hand">
  <worldbody>
    <body name="palm">
      <freejoint name="root"/>
      <geom type="box" size="0.03 0.03 0.03" mass="0.2"/>
      <body name="f0" pos="0.05 0 0">
        <joint name="finger_0" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01" mass="0.05"/>
        <body name="f1" pos="0.05 0 0">
          <joint name="finger_1" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ARM_DOFS = ["j1", "j2"]
HAND_DOFS = ["finger_0", "finger_1"]
HAND_DEFAULT = 0.1


class _Arm(RobotOptions):
    pass


class _Hand(RobotOptions):
    pass


# One backend per part, each its own in-memory wire, as the Franka and the XHand are separate SDKs.
@DEPLOYMENT_REGISTRY.register()
class CompositeArmLoopback(LoopbackDeployment):
    pass


@DEPLOYMENT_REGISTRY.register()
class CompositeHandLoopback(LoopbackDeployment):
    pass


@DEPLOYMENT_REGISTRY.register()
class CompositeFirmwareHandLoopback(FirmwareGainsLoopbackDeployment):
    """A hand that runs its own gains, as the XHand does."""


@DEPLOYMENT_REGISTRY.register()
class CompositeRemoteHandLoopback(LoopbackDeployment):
    """A part whose robot carries a remote (as the Oli and the G1 do): :attr:`operator` presses its buttons."""

    def __init__(self, env, options) -> None:
        super().__init__(env, options)
        self.operator = ScriptedOperator()

    def _remote_inputs(self):
        return [self.operator]


@DEPLOYMENT_REGISTRY.register()
class CompositeClockedHandLoopback(LoopbackDeployment):
    """A hand whose loop, like the Wuji's deployed alone, reads before the commands publish and keeps the episode
    clock.
    """

    sync_before_commands = True
    advances_episode_clock = True


@DEPLOYMENT_REGISTRY.register()
class CompositePacedHandLoopback(LoopbackDeployment):
    """A hand whose send waits out the control period in its default stream mode, as a Wuji's ``"sync"`` does."""

    stream_mode: str = "sync"

    @property
    def blocks_for_the_period(self) -> bool:
        return self.stream_mode == "sync"


@DEPLOYMENT_REGISTRY.register()
class CompositeScaleRefusingHandLoopback(LoopbackDeployment):
    """A hand that, like the XHand, runs firmware gains and refuses a command-gain scale, with its *standalone* fix."""

    def _resolve_options(self) -> None:
        super()._resolve_options()
        if self.deploy_kp_kd_scale != 1.0:
            raise ValueError(f"deploy_kp_kd_scale={self.deploy_kp_kd_scale:g} cannot reach the hand's firmware gains.")


@DEPLOYMENT_REGISTRY.register()
class CompositeAddressedHandLoopback(LoopbackDeployment):
    """A hand addressed by a serial number (text, whatever it looks like) and a bus id (a number)."""

    serial_number: str = ""
    hand_id: int = 0
    homes_on_connect: bool = True
    alias: Union[str, None] = None  # noqa: UP007 -- the spelling a text match on annotations misread
    fingers: tuple[str, ...] = ()


@DEPLOYMENT_REGISTRY.register()
class CompositePublishingHandLoopback(LoopbackDeployment):
    """A hand that re-sends its held command at a rate of its own, as the Oli and the G1 do."""

    publish_rate: float | None = 500.0


@DEPLOYMENT_REGISTRY.register()
class CompositeHoldingHandLoopback(LoopbackDeployment):
    """A backend with a hold of its own, which a composite never asks a part for."""

    def _hold_payload(self):
        return super()._hold_payload()


@DEPLOYMENT_REGISTRY.register()
class CompositeSteppingHandLoopback(LoopbackDeployment):
    """A backend that replaces the base loop, as the Wuji hands do."""

    def step(self, action=None):
        return super().step(action)


def _rig(assets: Path) -> DexManipulatorOptions:
    (assets / "arm.xml").write_text(TWO_LINK_MJCF)
    (assets / "hand.xml").write_text(WELDED_HAND_MJCF)
    return DexManipulatorOptions(
        arm=_Arm(
            file=str(assets / "arm.xml"),
            dofs_name=ARM_DOFS,
            default_dofs_pos={".*": 0.0},
            default_dofs_kp={".*": 50.0},
            default_dofs_kd={".*": 2.0},
            is_fixed_base=True,
        ),
        hand=_Hand(
            file=str(assets / "hand.xml"),
            dofs_name=HAND_DOFS,
            default_dofs_pos={".*": HAND_DEFAULT},
            default_dofs_kp={".*": 5.0},
            default_dofs_kd={".*": 0.1},
            default_root_pos=(0.3, 0.0, 0.0),  # the end of link2
            is_fixed_base=False,
        ),
        ee_link="link2",
    )


#: The welded hand of :data:`WELDED_HAND_MJCF` with its joints renamed, for the second wrist of a humanoid rig.
RIGHT_HAND_MJCF = WELDED_HAND_MJCF.replace('"finger_', '"rfinger_')


class _Body(_Arm):
    """The humanoid-shaped rig's root: the arm, declaring a foot of its own."""

    feet_links_name: ClassVar[tuple[str, ...]] = ("link1",)


def _humanoid_config(assets: Path):
    """A humanoid-shaped rig: the arm as its root, a hand welded to each of its links, the rig's own feet."""
    from eden import actions, observations
    from eden.options import ObservationGroupOptions
    from eden.utils.configs import (
        ActionManagerOptions,
        EdenRLConfig,
        EnvOptions,
        ObservationManagerOptions,
        SceneOptions,
    )

    arm = _rig(assets).arm
    (assets / "right_hand.xml").write_text(RIGHT_HAND_MJCF)
    hand = dict(default_dofs_kp={".*": 5.0}, default_dofs_kd={".*": 0.1}, is_fixed_base=False)
    rig = HumanoidOptions(
        robot=_Body(**arm.model_dump(exclude={"class_name"})),
        left_hand=_Hand(file=str(assets / "hand.xml"), dofs_name=HAND_DOFS, **hand),
        right_hand=_Hand(file=str(assets / "right_hand.xml"), dofs_name=["rfinger_0", "rfinger_1"], **hand),
        left_ee_link="link1",
        right_ee_link="link2",
        feet_links_name=("link2",),
    )
    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=rig),
        action_options=ActionManagerOptions(
            arm=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=ARM_DOFS)
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(arm_pos=observations.DofsPos.configure(entity_name="robot"))
        ),
    )


def _config(assets: Path):
    from eden import actions, observations
    from eden.options import ObservationGroupOptions
    from eden.utils.configs import (
        ActionManagerOptions,
        EdenRLConfig,
        EnvOptions,
        ObservationManagerOptions,
        SceneOptions,
    )

    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=_rig(assets)),
        action_options=ActionManagerOptions(
            arm=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=ARM_DOFS),
            # Reversed: the payload must place targets by name, not by position.
            hand=actions.ImplicitPDController.configure(entity_name="hand", dofs_name=HAND_DOFS[::-1]),
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                arm_pos=observations.DofsPos.configure(entity_name="robot"),
                hand_pos=observations.DofsPos.configure(entity_name="hand"),
                hand_quat=CachedEntityQuat.configure(entity_name="hand"),
            )
        ),
    )


@pytest.fixture(scope="module")
def rig_env(tmp_path_factory):
    from tests.support.envs import build_env

    return build_env(_config(tmp_path_factory.mktemp("composite")), reset=True)


@pytest.fixture(scope="module")
def humanoid_env(tmp_path_factory):
    from tests.support.envs import build_env

    return build_env(_humanoid_config(tmp_path_factory.mktemp("humanoid")), reset=True)


@pytest.fixture
def bindings():
    """Bind the arm and hand to their loopback backends; restore the global map afterwards."""
    snapshot = dict(deployment_base._ROBOT_DEPLOYMENT_BINDINGS)
    bind_deployment(_Arm, "composite_arm_loopback")
    bind_deployment(_Hand, "composite_hand_loopback")
    yield
    deployment_base._ROBOT_DEPLOYMENT_BINDINGS.clear()
    deployment_base._ROBOT_DEPLOYMENT_BINDINGS.update(snapshot)


@pytest.fixture
def deploy(rig_env, bindings):
    with deployments(rig_env) as make:
        yield lambda cls=CompositeDeployment, **options: make(cls, **options)


def _step(dep, action) -> dict:
    return dep.step({"arm": torch.tensor([action[:2]]), "hand": torch.tensor([action[2:]])})


def _policy(env, cls):
    return next(term for term in env.observation_manager._group_obs_terms["policy"] if isinstance(term, cls))


# ---------------------------------------------------------------------------
# Resolution and construction
# ---------------------------------------------------------------------------


def test_a_composite_resolves_to_the_composite_backend_and_each_part_to_its_own(tmp_path, bindings):
    rig = _rig(tmp_path)
    assert resolve_deployment_for(rig) is CompositeDeployment
    assert resolve_deployment_for(rig.arm) is CompositeArmLoopback
    assert resolve_deployment_for(rig.hand) is CompositeHandLoopback


def test_the_franka_xhand_rig_resolves_to_the_franka_and_the_xhand():
    from eden.extensions.deployment.franka_panda import FrankaPandaDeployment
    from eden.extensions.deployment.robotera_xhand import RoboTeraXHandDeployment
    from eden.options.robots import FrankaXHand

    rig = FrankaXHand()
    assert resolve_deployment_for(rig) is CompositeDeployment
    assert resolve_deployment_for(rig.arm) is FrankaPandaDeployment
    assert resolve_deployment_for(rig.hand) is RoboTeraXHandDeployment


def test_each_part_drives_its_own_entity_and_the_rig_is_one_robot(rig_env, deploy):
    sim_arm, sim_hand = rig_env.entities["robot"], rig_env.entities["hand"]
    dep = deploy()

    assert dep.dofs_name == ARM_DOFS + HAND_DOFS
    assert [type(part) for part in dep.parts.values()] == [CompositeArmLoopback, CompositeHandLoopback]
    for name, sim in (("robot", sim_arm), ("hand", sim_hand)):
        wrapper = rig_env.entities[name]
        assert isinstance(wrapper, RobotStateEntity) and wrapper._sim_entity is sim
    assert _policy(rig_env, CachedEntityQuat).robot is rig_env.entities["hand"], "a cached handle is repointed"
    np.testing.assert_array_equal(dep.default_dof_pos, [0.0, 0.0, HAND_DEFAULT, HAND_DEFAULT])
    np.testing.assert_array_equal(dep.default_dof_kp, [50.0, 50.0, 5.0, 5.0])
    assert not any(part._rate for part in dep.parts.values()), "the composite paces the loop, not each part"


def test_one_action_reaches_each_part_as_its_own_targets(rig_env, deploy):
    dep = deploy()
    dep.request_start()
    _step(dep, [0.0] * 4)  # the START tick holds
    _step(dep, [0.2, -0.3, 0.4, 0.5])

    terms = rig_env.action_manager.terms
    arm, hand = dep.parts["robot"], dep.parts["hand"]
    np.testing.assert_allclose(arm.sent[-1].dofs_pos, terms["arm"].processed_action[0].numpy(), atol=1e-6)
    # The hand term drives (finger_1, finger_0); the hand's wire is (finger_0, finger_1).
    np.testing.assert_allclose(hand.sent[-1].dofs_pos, terms["hand"].processed_action[0].numpy()[::-1], atol=1e-6)
    np.testing.assert_array_equal(arm.sent[-1].dofs_kp, [50.0, 50.0])
    np.testing.assert_array_equal(hand.sent[-1].dofs_kp, [5.0, 5.0])


def test_observations_read_every_parts_reading(rig_env, deploy):
    dep = deploy()
    dep.parts["robot"].readings.append(robot_state(2, dofs_pos=[0.3, -0.2]))
    dep.parts["hand"].readings.append(robot_state(2, dofs_pos=[0.6, 0.7]))

    obs = _step(dep, [0.0] * 4)

    policy = obs["policy"][0].numpy()
    np.testing.assert_allclose(policy[:4], [0.3, -0.2, 0.6, 0.7], atol=1e-6)
    np.testing.assert_allclose(dep.state.dofs_pos, [0.3, -0.2, 0.6, 0.7])


def test_the_welded_hands_velocity_is_the_arms_link_not_its_own_placeholder_reading(rig_env, deploy):
    """Both arm joints turn about y, so the hand on link2 turns about its own y at their summed rate."""
    dep = deploy()
    hand = rig_env.entities["hand"]

    dep.parts["robot"].readings.append(robot_state(2, dofs_pos=[0.5, 0.0], dofs_vel=np.array([0.4, 0.3])))
    _step(dep, [0.0] * 4)

    ang = hand.get_ang(frame="body")[0]
    torch.testing.assert_close(ang, torch.tensor([0.0, 0.7, 0.0]).to(ang), atol=1e-4, rtol=0)
    sim_vel = hand._sim_entity.get_vel()[0]
    torch.testing.assert_close(hand.get_vel()[0], sim_vel.to(hand.get_vel()), atol=1e-6, rtol=0)
    assert sim_vel.norm() > 0.1, "the hand rides on a turning arm"


def test_the_welded_hands_base_follows_the_arm_not_its_own_placeholder_reading(rig_env, deploy):
    """The hand has no IMU and reports identity; its base must still turn with the arm's joints."""
    dep = deploy()
    hand = rig_env.entities["hand"]

    dep.parts["robot"].readings.append(robot_state(2, dofs_pos=[0.0, 0.0]))
    _step(dep, [0.0] * 4)
    at_rest = hand.get_quat().clone()
    dep.parts["robot"].readings.append(robot_state(2, dofs_pos=[0.5, 0.0]))
    _step(dep, [0.0] * 4)

    turned = hand.get_quat()
    assert quat_error_magnitude(turned, at_rest).item() == pytest.approx(0.5, abs=1e-4)
    torch.testing.assert_close(turned, hand._sim_entity.get_quat().to(turned), atol=1e-6, rtol=0)
    torch.testing.assert_close(_policy(rig_env, CachedEntityQuat).compute(), turned)


@pytest.mark.parametrize(("entity_name", "dofs"), [("robot", ARM_DOFS), ("hand", HAND_DOFS)])
def test_a_single_robot_backend_deploys_the_part_its_entity_name_names(rig_env, bindings, entity_name, dofs):
    """``robot`` is the scene key the whole rig sits under, and also the env entity of its arm."""
    with deployments(rig_env) as make:
        dep = make(LoopbackDeployment, entity_name=entity_name)
        assert dep.dofs_name == dofs
        assert [isinstance(rig_env.entities[name], RobotStateEntity) for name in ("robot", "hand")] == [
            name == entity_name for name in ("robot", "hand")
        ]


def test_the_composite_backend_refuses_a_part_that_is_not_a_rig(rig_env, bindings):
    with deployments(rig_env) as make, pytest.raises(TypeError, match="'hand' is a single _Hand"):
        make(CompositeDeployment, entity_name="hand")


# ---------------------------------------------------------------------------
# The wire fans out
# ---------------------------------------------------------------------------


def test_each_part_low_passes_its_own_targets_and_the_record_holds_what_was_sent(rig_env, deploy, tmp_path):
    path = tmp_path / "rec.npz"
    dep = deploy(parts_options={"hand": {"target_lpf": 0.5}}, debug_record=True, debug_record_path=str(path))
    dep.request_start()
    _step(dep, [0.0] * 4)  # warm-starts both filters at the default targets
    _step(dep, [1.0, 1.0, 1.0, 1.0])

    arm_target = rig_env.action_manager.terms["arm"].processed_action[0].numpy()
    hand_target = rig_env.action_manager.terms["hand"].processed_action[0].numpy()[::-1]
    np.testing.assert_allclose(dep.parts["robot"].sent[-1].dofs_pos, arm_target, atol=1e-6)
    np.testing.assert_allclose(
        dep.parts["hand"].sent[-1].dofs_pos, 0.5 * hand_target + 0.5 * dep.parts["hand"].sent[-2].dofs_pos, atol=1e-6
    )
    dep.close()
    record = load_deploy_record(str(path))
    assert list(record["dofs_name"]) == ARM_DOFS + HAND_DOFS
    np.testing.assert_allclose(
        record["cmd_pos"][-1],
        np.concatenate([dep.parts["robot"].sent[-1].dofs_pos, dep.parts["hand"].sent[-1].dofs_pos]),
    )


def test_a_part_that_fails_to_connect_disconnects_the_ones_already_up(deploy, monkeypatch):
    dep = deploy()
    closed = []
    monkeypatch.setattr(dep.parts["robot"], "close", lambda: closed.append("robot"))

    def refuse() -> None:
        raise ConnectionError("hand not found on /dev/ttyUSB0")

    monkeypatch.setattr(dep.parts["hand"], "connect", refuse)
    with pytest.raises(ConnectionError, match="hand not found"):
        dep.connect()
    assert closed == ["robot"]


def test_close_closes_every_part_even_when_one_raises(deploy, monkeypatch, eden_caplog):
    dep = deploy()
    closed = []
    monkeypatch.setattr(dep.parts["robot"], "close", lambda: closed.append("robot"))

    def broken() -> None:
        closed.append("hand")
        raise OSError("bus error")

    monkeypatch.setattr(dep.parts["hand"], "close", broken)
    with pytest.raises(OSError, match="bus error"):
        dep.close()
    assert sorted(closed) == ["hand", "robot"], "the arm still closed"
    assert "Closing part 'hand'" in eden_caplog.text


def test_each_part_reads_and_sends_on_its_own_thread_at_the_same_time(deploy, monkeypatch):
    """One after the other, a tick's I/O was the sum of the parts'; now the arm's and the hand's overlap.

    Each call returns only once every part is inside its own, so parts driven in turn would never get there.
    """
    dep = deploy(connect=True)
    seen = {name: set() for name in dep.parts}
    together = threading.Barrier(len(dep.parts), timeout=2.0)
    for name, part in dep.parts.items():
        for method in ("read_state", "send_payload"):

            def overlapping(*args, wire=getattr(part, method), name=name):
                seen[name].add(threading.current_thread())
                together.wait()
                return wire(*args)

            monkeypatch.setattr(part, method, overlapping)

    dep.request_start()
    for _ in range(3):
        _step(dep, [0.2] * 4)

    assert all(len(threads) == 1 for threads in seen.values()), f"one thread per part, every tick: {seen}"
    arm, hand = (next(iter(threads)) for threads in seen.values())
    assert arm is not hand and threading.main_thread() not in (arm, hand)
    np.testing.assert_allclose(dep.parts["hand"].sent[-1].dofs_pos, dep.state.dofs_pos[2:], atol=1e-6)


def test_a_part_that_fails_on_its_thread_raises_in_the_loop_by_name_once_the_others_are_done(deploy, monkeypatch):
    dep = deploy()

    def unplugged(payload) -> None:
        raise OSError("write failed: device disconnected")

    monkeypatch.setattr(dep.parts["hand"], "send_payload", unplugged)
    with pytest.raises(OSError, match="device disconnected") as raised:
        _step(dep, [0.0] * 4)  # the boot hold

    assert any("part 'hand'" in note for note in raised.value.__notes__)
    assert len(dep.parts["robot"].sent) == 1, "the arm's send finished before the loop raised"
    assert not any(part.sending_hold for part in dep.parts.values())


# ---------------------------------------------------------------------------
# Refusals, before anything is left swapped
# ---------------------------------------------------------------------------


def _assert_env_as_built(env) -> None:
    assert not any(isinstance(env.entities[name], RobotStateEntity) for name in ("robot", "hand"))


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"parts_options": {"gripper": {}}}, ValueError, r"names \['gripper'\].*Parts: \['robot', 'hand'\]"),
        (
            {"parts_options": {"hand": {"debug_record": True}}},
            ValueError,
            r"sets \['debug_record'\], which the composite sets for the whole rig",
        ),
        ({"parts_options": {"hand": {"no_such_option": 1}}}, ValueError, "no_such_option"),
        ({"disable_joints": ("nope",)}, ValueError, "matched no DOF"),
    ],
    ids=["unknown-part", "composite-owned-option", "unknown-backend-option", "disable-typo"],
)
def test_a_bad_option_leaves_the_env_as_built(rig_env, deploy, options, error, match):
    with pytest.raises(error, match=match):
        deploy(**options)
    _assert_env_as_built(rig_env)


@pytest.mark.parametrize(
    ("backend", "match"),
    [
        ("composite_stepping_hand_loopback", r"replaces DeploymentBase\.step\(\)"),
        ("composite_holding_hand_loopback", r"replaces DeploymentBase\._hold_payload\(\)"),
    ],
    ids=["own-loop", "own-hold"],
)
def test_a_part_whose_backend_cannot_run_inside_a_composite_is_refused(rig_env, deploy, backend, match):
    bind_deployment(_Hand, backend)
    with pytest.raises(NotImplementedError, match=match):
        deploy()
    _assert_env_as_built(rig_env)


def test_a_part_no_backend_is_bound_to_is_named(rig_env, deploy):
    del deployment_base._ROBOT_DEPLOYMENT_BINDINGS[_Hand]
    with pytest.raises(
        ValueError,
        match=r"Part 'hand' of DexManipulatorOptions \(_Hand\) has no deployment backend.*`--entity_name hand`",
    ):
        deploy()
    _assert_env_as_built(rig_env)


def test_disabling_a_joint_whose_part_ignores_command_gains_is_refused(rig_env, deploy):
    bind_deployment(_Hand, "composite_firmware_hand_loopback")
    with pytest.raises(ValueError, match=r"matched \['finger_0'\] on part 'hand'.*would keep servoing"):
        deploy(disable_joints=("finger_0",))
    _assert_env_as_built(rig_env)


def test_the_gain_scale_softens_the_commands_and_reaches_every_part(deploy):
    """Each backend gives the scale its own meaning (a Wuji Hand 2's MIT gains at connect), so each is handed it."""
    dep = deploy(deploy_kp_kd_scale=0.5)
    np.testing.assert_array_equal(dep.default_dof_kp, [25.0, 25.0, 2.5, 2.5])
    assert [part.deploy_kp_kd_scale for part in dep.parts.values()] == [0.5, 0.5]


def test_a_body_is_commissioned_at_a_low_scale_next_to_a_hand_that_refuses_one(rig_env, deploy):
    """The body's 0.2 -> 1.0 ramp-up must not be blocked by a hand whose firmware gains cannot be scaled."""
    bind_deployment(_Hand, "composite_scale_refusing_hand_loopback")
    with pytest.raises(ValueError, match="cannot reach the hand's firmware gains") as refused:
        deploy(deploy_kp_kd_scale=0.2)
    assert any("--part_option hand.deploy_kp_kd_scale=1.0" in note for note in refused.value.__notes__)
    _assert_env_as_built(rig_env)

    dep = deploy(deploy_kp_kd_scale=0.2, parts_options={"hand": {"deploy_kp_kd_scale": 1.0}})
    np.testing.assert_allclose(dep.default_dof_kp, [10.0, 10.0, 5.0, 5.0])
    assert [part.deploy_kp_kd_scale for part in dep.parts.values()] == [0.2, 1.0]


def test_a_part_option_typed_on_the_command_line_takes_its_declared_type(deploy):
    """A serial number that looks like a number stays text; a bus id becomes the int it is declared as."""
    bind_deployment(_Hand, "composite_addressed_hand_loopback")
    typed = {
        "serial_number": "12345678",
        "hand_id": "2",
        "target_lpf": "0.5",
        "homes_on_connect": "false",
        "alias": "12345678",
        "fingers": "('finger_0',)",
    }
    hand = deploy(parts_options={"hand": typed}).parts["hand"]
    assert (hand.serial_number, hand.hand_id, hand.target_lpf) == ("12345678", 2, 0.5)
    assert hand.homes_on_connect is False, "the text 'false' is truthy; the option is declared a bool"
    assert (hand.alias, hand.fingers) == ("12345678", ("finger_0",))


@pytest.mark.parametrize(
    ("option", "value", "declared"),
    [
        ("homes_on_connect", "maybe", "bool"),
        ("hand_id", "two", "int"),
        ("fingers", "finger_0", r"tuple\[str, \.\.\.\]"),
    ],
)
def test_a_part_option_that_is_not_its_declared_type_is_refused_by_name(rig_env, deploy, option, value, declared):
    bind_deployment(_Hand, "composite_addressed_hand_loopback")
    with pytest.raises(ValueError, match=rf"parts_options\['hand'\]\['{option}'\] = '{value}' is not a {declared}"):
        deploy(parts_options={"hand": {option: value}})
    _assert_env_as_built(rig_env)


def test_softening_the_init_ramp_is_refused_on_a_rig_with_a_part_that_runs_its_own_gains(rig_env, deploy):
    """That part would ramp at full gains while the rest ramp softly, as a Wuji hand refuses it standalone."""
    bind_deployment(_Hand, "composite_firmware_hand_loopback")
    with pytest.raises(ValueError, match=r"init_dofs_kp_scale=0\.3 .*\{'hand': 'CompositeFirmwareHandLoopback'\}"):
        deploy(init_dofs_kp_scale=0.3)
    _assert_env_as_built(rig_env)

    bind_deployment(_Hand, "composite_hand_loopback")
    assert deploy(init_dofs_kp_scale=0.3, init_dofs_kd_scale=0.5).init_dofs_kp_scale == 0.3


def test_slot_gains_are_compared_only_on_the_parts_that_run_their_own(deploy, eden_caplog):
    """A switch brings the new slot's gains to the arm; only the hand runs one set whichever slot is active."""
    bind_deployment(_Hand, "composite_firmware_hand_loopback")
    dep = deploy()
    np.testing.assert_array_equal(dep.command_gain_dofs(), [True, True, False, False])
    kp, kd = dep.wire_gains()
    slots = [
        _DeploySlot("boot", None, gains=(kp, kd)),
        _DeploySlot("stiff-arm", None, gains=(kp * [2.0, 2.0, 1.0, 1.0], kd)),
        _DeploySlot("stiff-hand", None, gains=(kp * [1.0, 1.0, 2.0, 2.0], kd)),
    ]
    _warn_on_unswitchable_slot_gains(dep, slots)
    warned = [record.getMessage() for record in eden_caplog.records if "was trained at another" in record.getMessage()]
    assert len(warned) == 1 and "'stiff-hand'" in warned[0] and "finger_0" in warned[0]


# ---------------------------------------------------------------------------
# The rig's feet
# ---------------------------------------------------------------------------


def test_a_humanoid_rigs_own_feet_feed_the_root_part(humanoid_env, bindings):
    with deployments(humanoid_env) as make:
        dep = make(CompositeDeployment)
        assert dep.feet_links == ("link2",) and dep.parts["robot"].feet_links == ("link2",)


def test_a_rig_that_declares_no_feet_has_none_and_refuses_the_estimator(humanoid_env, bindings, monkeypatch):
    """``feet_links_name=()`` is how a rig says its humanoid's feet are not what it stands on."""
    monkeypatch.setattr(humanoid_env.config.scene_options.robot, "feet_links_name", ())
    with deployments(humanoid_env) as make:
        assert make(CompositeDeployment).feet_links == (), "not the root robot's own link1"
    with (
        deployments(humanoid_env) as make,
        pytest.raises(ValueError, match=r"declares it has none \(feet_links_name=\(\)\)"),
    ):
        make(CompositeDeployment, estimate_lin_vel=True)


# ---------------------------------------------------------------------------
# The operator state machine, once for the whole rig
# ---------------------------------------------------------------------------


def test_the_rig_boots_holding_every_part_until_it_is_started(deploy):
    dep = deploy()
    dep.parts["robot"].readings.append(robot_state(2, dofs_pos=[0.3, -0.2]))
    dep.parts["hand"].readings.append(robot_state(2, dofs_pos=[0.6, 0.7]))
    dep.reset()

    _step(dep, [1.0] * 4)

    assert not dep.policy_active
    np.testing.assert_allclose(dep.parts["robot"].sent[-1].dofs_pos, [0.3, -0.2])
    np.testing.assert_allclose(dep.parts["hand"].sent[-1].dofs_pos, [0.6, 0.7])


def test_a_parts_remote_drives_the_whole_rig_through_init_and_start(rig_env, deploy):
    """The Oli's joystick on an Oli + Wuji rig: one START hands every part to the policy."""
    bind_deployment(_Hand, "composite_remote_hand_loopback")
    dep = deploy(init_steps=2)
    dep.reset()
    remote = dep.parts["hand"].operator
    assert remote in dep._inputs

    remote.press(CMD_INIT)
    for _ in range(3):
        _step(dep, [1.0] * 4)
    np.testing.assert_allclose(dep.parts["robot"].sent[-1].dofs_pos, [0.0, 0.0], err_msg="INIT reached the stance")
    np.testing.assert_allclose(dep.parts["hand"].sent[-1].dofs_pos, [HAND_DEFAULT] * 2)

    remote.press(CMD_START)
    _step(dep, [0.9] * 4)  # computed before START: the tick repeats the INIT target instead
    assert dep.policy_active
    np.testing.assert_allclose(dep.parts["robot"].sent[-1].dofs_pos, [0.0, 0.0])
    np.testing.assert_allclose(dep.parts["hand"].sent[-1].dofs_pos, [HAND_DEFAULT] * 2)
    _step(dep, [0.2, -0.3, 0.4, 0.5])
    terms = rig_env.action_manager.terms
    np.testing.assert_allclose(
        dep.parts["robot"].sent[-1].dofs_pos, terms["arm"].processed_action[0].numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        dep.parts["hand"].sent[-1].dofs_pos, terms["hand"].processed_action[0].numpy()[::-1], atol=1e-6
    )


def test_a_hold_of_the_rig_is_a_hold_of_every_part(deploy):
    """A Franka part leaves its gripper alone on a hold; inside a rig it must still know it is one."""
    dep = deploy(init_steps=1)
    seen = {name: [] for name in dep.parts}
    for name, part in dep.parts.items():
        wire = part.send_payload

        def recording(payload, name=name, part=part, wire=wire):
            seen[name].append(part.sending_hold)
            wire(payload)

        part.send_payload = recording

    _step(dep, [0.0] * 4)  # STOP
    dep.request_start()
    _step(dep, [0.0] * 4)  # the START tick
    _step(dep, [0.0] * 4)  # the policy

    assert seen == {name: [True, True, False] for name in dep.parts}


def test_a_loop_setting_every_part_runs_alone_is_the_rigs(rig_env, deploy):
    """Two parts that both keep the episode clock deployed alone keep it in the rig: once per policy tick."""
    bind_deployment(_Arm, "composite_clocked_hand_loopback")
    bind_deployment(_Hand, "composite_clocked_hand_loopback")
    clock = int(rig_env.episode_length_buf[0])
    dep = deploy()
    assert dep.sync_before_commands and dep.advances_episode_clock
    _step(dep, [0.0] * 4)
    assert int(rig_env.episode_length_buf[0]) == clock, "the policy's clock: it stands still before START"
    dep.request_start()
    _step(dep, [0.0] * 4)
    assert int(rig_env.episode_length_buf[0]) == 0, "restarted at START; the START tick reads the policy's first"
    _step(dep, [0.0] * 4)
    _step(dep, [0.0] * 4)
    assert int(rig_env.episode_length_buf[0]) == 2, "advanced once per policy tick, not once per part"


def test_a_loop_setting_only_some_parts_run_is_not_handed_to_the_others(rig_env, deploy, eden_caplog):
    """An Oli body next to Wuji hands: the body must run as it does deployed alone, not in the hands' order and clock."""
    bind_deployment(_Hand, "composite_clocked_hand_loopback")
    clock = int(rig_env.episode_length_buf[0])
    dep = deploy()

    assert not dep.sync_before_commands and not dep.advances_episode_clock, "the arm's own loop, as it runs alone"
    for setting in ("sync_before_commands", "advances_episode_clock"):
        assert f"{setting} is set by part(s) {{'hand': 'CompositeClockedHandLoopback'}}" in eden_caplog.text
    dep.request_start()
    for _ in range(3):
        _step(dep, [0.0] * 4)
    assert int(rig_env.episode_length_buf[0]) == clock, "the arm's clock, which nothing advances alone"


def test_a_part_that_waits_out_the_period_in_its_send_paces_the_rig_as_it_paces_itself(humanoid_env, bindings):
    """A Wuji hand keeps its own ``stream_mode="sync"``: its send is the period's wait, on its own thread.

    The two hands' waits overlap, so the rig still takes one period, and no rate limiter adds to it.
    """
    bind_deployment(_Hand, "composite_paced_hand_loopback")
    with deployments(humanoid_env) as make:
        dep = make(CompositeDeployment, sync=True)
        hands = {name: part for name, part in dep.parts.items() if name != "robot"}
        assert {part.stream_mode for part in hands.values()} == {"sync"}, "each hand's own default, as it runs alone"
        assert dep._rate is None, "the hands pace the rig"

        together = threading.Barrier(len(hands), timeout=2.0)
        for part in hands.values():
            part._idle_tick = together.wait  # a hand's hold for the period returns only once both are holding
        dep._idle_tick()

    with deployments(humanoid_env) as make:
        dep = make(CompositeDeployment, sync=True, parts_options={name: {"stream_mode": "thread"} for name in hands})
        assert dep._rate is not None, "no part waits out the period: the rig's rate limiter paces it"


def test_a_rig_whose_parts_share_a_joint_name_is_refused(tmp_path):
    rig = _rig(tmp_path)
    rig.hand.dofs_name = ["j1", "finger_1"]
    with pytest.raises(NotImplementedError, match=r"share joint names \{'j1': \['robot', 'hand'\]\}"):
        CompositeDeployment._check_unique_dofs(rig, dict(rig.entity_options()))


# ---------------------------------------------------------------------------
# The CLI flag
# ---------------------------------------------------------------------------


def test_part_options_keep_every_value_as_typed_for_the_backend_to_read():
    """Only the part's backend knows an option's type (see the typed-option test above)."""
    parsed = _parse_part_options(["robot.robot_ip=172.16.0.2", "hand.hand_id=1", "hand.serial_number=0123"])
    assert parsed == {"robot": {"robot_ip": "172.16.0.2"}, "hand": {"hand_id": "1", "serial_number": "0123"}}


@pytest.mark.parametrize("spec", ["hand.protocol", "protocol=EtherCAT", ".protocol=EtherCAT", "hand.=1"])
def test_a_malformed_part_option_names_the_expected_form(spec):
    with pytest.raises(ValueError, match="PART.OPTION=VALUE"):
        _parse_part_options([spec])


def test_a_slot_controller_on_any_part_follows_that_parts_swap(rig_env, deploy):
    """Only ``--entity_name``'s entity was repointed: a slot driving the hand kept reading the simulated hand."""
    from eden import actions

    sim_entities = dict(rig_env.entities)
    controllers = {}
    for name, dofs in (("robot", ARM_DOFS), ("hand", HAND_DOFS)):
        options = actions.ImplicitPDController.configure(entity_name=name, dofs_name=dofs)
        controllers[name] = actions.ImplicitPDController(env=rig_env, options=options)
        controllers[name].build()  # against the sim entity, before the deployment swaps it
    deploy()

    _repoint_swapped_slot_controllers([("slot", None, controllers)], sim_entities, rig_env.entities)

    for name, controller in controllers.items():
        held = list(vars(controller).values())
        assert not any(value is sim_entities[name] for value in held), f"the {name} slot controller reads the sim"
        assert any(value is rig_env.entities[name] for value in held), f"the {name} slot controller reads the robot"


# ---------------------------------------------------------------------------
# Part order and the parts' backends
# ---------------------------------------------------------------------------


class _HandFirstRig(DexManipulatorOptions):
    """The rig of :func:`_rig`, declaring its welded hand before the arm it hangs off."""

    def entity_parts(self):
        return super().entity_parts()[::-1]


def test_a_welded_part_comes_after_its_parent_whatever_order_the_rig_declares(tmp_path, bindings):
    rig = _rig(tmp_path)
    hand_first = _HandFirstRig(arm=rig.arm, hand=rig.hand, ee_link="link2")
    assert hand_first.entity_names == ["hand", "robot"]
    assert list(CompositeDeployment.part_backends(hand_first)) == ["robot", "hand"]


def test_the_real_rigs_parts_run_inside_a_composite():
    """Every hook the shipped backends override is one the composite calls on a part."""
    from eden.options.robots import FrankaXHand, OliWuji

    assert PART_HOOKS <= set(vars(DeploymentBase)), sorted(PART_HOOKS - set(vars(DeploymentBase)))
    backends = [
        cls.__name__ for rig in (FrankaXHand(), OliWuji()) for cls in CompositeDeployment.part_backends(rig).values()
    ]
    assert set(backends) == {
        "FrankaPandaDeployment",
        "RoboTeraXHandDeployment",
        "LimxOliDeployment",
        "WujiHandDeployment",
    }


# ---------------------------------------------------------------------------
# `eden deploy` on a rig
# ---------------------------------------------------------------------------


def test_a_backend_flag_reaches_the_parts_that_declare_it(eden_caplog):
    """--publish_rate 0 means one packet per tick (None) on the part that republishes; an explicit part option wins."""
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    backends = {"robot": CompositeArmLoopback, "hand": CompositePublishingHandLoopback}
    configure_kwargs: dict = {}
    parts: dict = {}
    _route_backend_flag(
        "publish_rate", publish_rate_configure_kwargs(0), CompositeDeployment, backends, configure_kwargs, parts
    )
    assert (configure_kwargs, parts) == ({}, {"hand": {"publish_rate": None}})

    parts = {"hand": {"publish_rate": "250"}}
    _route_backend_flag("publish_rate", {"publish_rate": 100.0}, CompositeDeployment, backends, configure_kwargs, parts)
    assert parts == {"hand": {"publish_rate": "250"}}

    _route_backend_flag(
        "power_protect_factor", {"power_protect_factor": 5}, CompositeDeployment, backends, configure_kwargs, parts
    )
    assert configure_kwargs == {}
    assert "not supported by CompositeDeployment / CompositeArmLoopback / CompositePublishingHandLoopback" in (
        eden_caplog.text
    )


def test_a_rig_under_mujoco_refuses_the_single_robot_mujoco_flags_and_part_options_refuse_a_single_robot(tmp_path):
    """The rig's MuJoCo backend does not run them, and a flag no backend declares would be dropped in silence.

    Checked on the env's own scene too, so a --policy env (no task config) is refused as a --task one is.
    """
    from eden.utils.configs import SceneOptions

    scene = SceneOptions(robot=_rig(tmp_path))
    _check_composite_deploy(scene, "robot", mujoco=True, parts_options=None, mujoco_flags={"--menagerie": False})
    with pytest.raises(
        SystemExit, match=r"--menagerie / --async are not supported by --mujoco on 'robot', a composite"
    ):
        _check_composite_deploy(
            scene, "robot", mujoco=True, parts_options=None, mujoco_flags={"--menagerie": True, "--async": True}
        )
    with pytest.raises(SystemExit, match="'hand' is a single robot"):
        _check_composite_deploy(scene, "hand", mujoco=False, parts_options={"hand": {"protocol": "EtherCAT"}})
    _check_composite_deploy(scene, "robot", mujoco=False, parts_options={"hand": {"protocol": "EtherCAT"}})
    _check_composite_deploy(scene, "hand", mujoco=True, parts_options=None, mujoco_flags={"--menagerie": True})
