"""``DeploymentBase`` against a real G1 env: construction, the entity and command swaps, the payload, and the loop."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

import eden.extensions.deployment.base as deployment_base
from eden import observations
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    bind_deployment,
    build_term_action_maps,
    resolve_deployment_for,
    resolve_dofs_values,
    with_owned_modifiers,
)
from eden.extensions.deployment.utils.operator_input import CMD_INIT, CMD_START, CMD_STOP
from eden.extensions.deployment.utils.record import load_deploy_record
from eden.extensions.deployment.utils.state import RobotStateEntity
from eden.managers import COMMAND_TERM_REGISTRY, OBSERVATION_TERM_REGISTRY
from eden.managers.modifiers.actions.actuators import ActionDelay, Compose, EnvelopeClip
from eden.managers.modifiers.observations import ExponentialMovingAverage, ObservationDelay
from eden.managers.observation_manager import ObservationTerm
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.managers.terms.commands.velocity import ConstantVelocityCommand
from eden.managers.terms.observations.locomotion import ScaledCommands, VelocityTrackingErrorIntegral
from eden.options.entities import RobotOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1Dofs23, G1Dofs29
from eden.terms import ObsTerm
from tests.extensions.deployment.conftest import (
    COMMAND_CLIP_FRAMES,
    COMMAND_CLIP_STEP,
    MANAGER_CLIP_JOINT_STEP,
    CachedEntityModifier,
    CachedEntityQuat,
    two_link_robot,
)
from tests.support.deploy import (
    FirmwareGainsLoopbackDeployment,
    LoopbackDeployment,
    OperatorLoopbackDeployment,
    UnattendedLoopbackDeployment,
    deployments,
    robot_env,
    robot_state,
    start_policy,
)

_RANGES = {"lin_vel_x_range": (-1.0, 1.0), "lin_vel_y_range": (-1.0, 1.0), "ang_vel_z_range": (-0.5, 0.5)}


def _policy_term(env, cls):
    return next(term for term in env.observation_manager._group_obs_terms["policy"] if isinstance(term, cls))


def _held(**overrides):
    return ConstantVelocityCommand.configure(entity_name="robot", **{**_RANGES, **overrides})


def _steps(dep, n: int, targets=None) -> None:
    """``n`` control ticks of an extension-policy slot writing ``targets`` (the default pose if unset)."""
    for _ in range(n):
        targets = dep.default_dof_pos if targets is None else targets
        dep.write_dofs_targets(torch.as_tensor(targets, dtype=torch.float32), dep.dofs_name)
        dep.step(None)


@COMMAND_TERM_REGISTRY.register()
class DeviceOwningVelocityCommand(ConstantVelocityCommand):
    """A held command that owns an input device, as ``OperatorVelocityCommand`` does. Counts its closes."""

    _closes = 0

    def close(self) -> None:
        type(self)._closes += 1


@COMMAND_TERM_REGISTRY.register()
class BrokenDriverVelocityCommand(ConstantVelocityCommand):
    """A held command whose input driver fails on the way out."""

    def close(self) -> None:
        raise RuntimeError("the input driver blew up")


# ---------------------------------------------------------------------------
# Per-DOF values and the robot -> backend binding
# ---------------------------------------------------------------------------


def test_wildcard_values_land_on_the_joints_their_patterns_name(eden_caplog):
    """Values follow ``dofs_name`` order, not pattern order: the scramble that sent a Holosoma G1 the wrong knee kp."""
    np.testing.assert_array_equal(
        resolve_dofs_values({"d|e": 1.0, "a|b": 2.0, "c": 3.0}, ["a", "b", "c", "d", "e"]), [2.0, 2.0, 3.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(resolve_dofs_values({"a": 5.0}, ["a", "b"]), [5.0, 0.0])
    assert "No value matched for DOFs ['b']" in eden_caplog.text


def test_holosoma_g1_gains_match_the_released_onnx_metadata():
    """The kp/kd embedded in the released ``ppo_g1_29dof.onnx``, in its ``dof_names`` order."""
    pytest.importorskip("onnxruntime", exc_type=ImportError)
    from eden.extensions.policies.holosoma.constants import G1_LOCO_DEFAULT_DOF_POS
    from eden.extensions.policies.holosoma.policy import holosoma_g1_robot

    leg_kp = (40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196, 28.501246196)
    leg_kd = (2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687, 1.814445687)
    arm_kp = (14.250623098,) * 5 + (16.778327481,) * 2
    arm_kd = (0.907222843,) * 5 + (1.068141502,) * 2
    onnx_kp = leg_kp * 2 + (40.179238471, 28.501246196, 28.501246196) + arm_kp * 2
    onnx_kd = leg_kd * 2 + (2.557889765, 1.814445687, 1.814445687) + arm_kd * 2

    robot = holosoma_g1_robot()
    dofs_name = list(robot.dofs_name)
    assert resolve_dofs_values(robot.default_dofs_kp, dofs_name) == pytest.approx(onnx_kp)
    assert resolve_dofs_values(robot.default_dofs_kd, dofs_name) == pytest.approx(onnx_kd)
    assert resolve_dofs_values(robot.default_dofs_pos, dofs_name) == pytest.approx(G1_LOCO_DEFAULT_DOF_POS)


class _ArmRobot(RobotOptions):
    pass


class _ArmRobotMk2(_ArmRobot):
    pass


class _UnboundRobot(RobotOptions):
    pass


@pytest.fixture
def scoped_bindings():
    """Snapshot and restore the global robot -> backend map around a test that binds."""
    snapshot = dict(deployment_base._ROBOT_DEPLOYMENT_BINDINGS)
    yield
    deployment_base._ROBOT_DEPLOYMENT_BINDINGS.clear()
    deployment_base._ROBOT_DEPLOYMENT_BINDINGS.update(snapshot)


def test_a_binding_resolves_through_the_robots_mro(scoped_bindings):
    from eden.extensions.deployment.mujoco import MujocoDeployment
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment

    bind_deployment(_ArmRobot, "mujoco_deployment")
    assert resolve_deployment_for(_ArmRobot(file="arm.urdf")) is MujocoDeployment
    assert resolve_deployment_for(_ArmRobotMk2(file="arm.urdf")) is MujocoDeployment, "inherited from the parent"

    bind_deployment(_ArmRobotMk2, "unitree_g1_deployment")
    assert resolve_deployment_for(_ArmRobotMk2(file="arm.urdf")) is UnitreeG1Deployment, "the most derived wins"


def test_an_unbound_robot_raises_listing_the_bindings(scoped_bindings):
    bind_deployment(_ArmRobot, "mujoco_deployment")
    with pytest.raises(KeyError, match=r"No deployment registered for robot _UnboundRobot.*_ArmRobot"):
        resolve_deployment_for(_UnboundRobot(file="arm.urdf"))


def test_every_g1_variant_resolves_to_the_unitree_g1_backend():
    import eden.extensions.deployment  # noqa: F401 -- each backend module binds its robots on import
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.options.robots.unitree_g1 import G1Dofs23, G1Dofs29NoHands

    for robot_cls in (G1Dofs29, G1Dofs23, G1Dofs29NoHands):
        assert resolve_deployment_for(robot_cls()) is UnitreeG1Deployment, robot_cls.__name__


def test_backends_take_the_feet_from_the_robot_not_from_themselves():
    """A backend that redeclared ``feet_links`` would leave the robot-agnostic MuJoCo backend with none."""
    import eden.extensions.deployment  # noqa: F401

    assert [name for name, backend in DEPLOYMENT_REGISTRY if "feet_links" in backend.__dict__] == []


# ---------------------------------------------------------------------------
# Construction fails fast, before the env is touched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"target_lpf": 0.0}, ValueError, "target_lpf must be in"),
        ({"target_lpf": 1.5}, ValueError, "target_lpf must be in"),
        ({"target_lpf": float("nan")}, ValueError, "target_lpf must be in"),
        ({"on_motion_end": "stan"}, ValueError, "on_motion_end must be"),
        ({"on_motion_end": "switch", "motion_end_slot": 0}, ValueError, "1-based"),
        ({"init_target": "frame_zero"}, ValueError, "must be 'default', 'motion' or 'motion_rsi'"),
        ({"init_target": "motion_rsi"}, ValueError, "drives a real robot"),
        ({"debug_record": True, "debug_record_flush_steps": 0}, ValueError, "debug_record_flush_steps"),
        ({"disable_joints": ("left_wirst_*",)}, ValueError, "matched no DOF"),
        ({"gain_ramp_steps": -1}, ValueError, "gain_ramp_steps must be >= 0"),
        ({"policy_seconds": 0.0}, ValueError, "policy_seconds must be > 0"),
        ({"policy_seconds": 6.0}, ValueError, "only a simulator may.*the operator ends"),
        ({"feet_links": "l_foot"}, TypeError, "bare string"),
        ({"ignore_action_terms": "hand"}, TypeError, "bare string"),
        ({"disable_joints": "left_wrist_yaw_joint"}, TypeError, "bare string"),
        ({"ignore_action_terms": ("hnad",)}, ValueError, r"\['hnad'\], which are not action terms"),
        ({"ignore_action_terms": ("legs",)}, ValueError, "whose joint targets the payload does carry"),
        ({"command_overrides": {"velocity_commmand": _held()}}, ValueError, "does not have"),
        ({"entity_name": "hand", "estimate_lin_vel": True}, ValueError, r"feet_links_name.*--feet_links"),
    ],
)
def test_a_bad_option_raises_before_the_entity_swap(g1_env, deploy, eden_caplog, options, error, match):
    built = dict(g1_env.entities)
    trained_command = g1_env.command_manager.get_term("velocity_command")

    with pytest.raises(error, match=match):
        deploy(**options)

    assert g1_env.entities == built
    assert g1_env.command_manager.get_term("velocity_command") is trained_command
    assert "rolled back" not in eden_caplog.text, "an options error must not get as far as the swap"


def test_a_state_to_observation_override_without_the_keyword_is_refused_at_construction(deploy):
    """``step`` passes ``post_physics_refresh=True``; an override without it fails here, not on the first tick."""

    class Stale(LoopbackDeployment):
        def state_to_observation(self, state):
            return {}

    class Forwarding(LoopbackDeployment):
        def state_to_observation(self, state, **kwargs):
            return super().state_to_observation(state, **kwargs)

    with pytest.raises(TypeError, match="post_physics_refresh"):
        deploy(Stale)
    assert set(_steps_obs(deploy(Forwarding))) == {"policy"}


def _steps_obs(dep):
    dep.write_dofs_targets(torch.as_tensor(dep.default_dof_pos, dtype=torch.float32), dep.dofs_name)
    return dep.step(None)


def test_a_backends_own_options_check_runs_before_the_swap(g1_env, deploy, eden_caplog):
    """``_resolve_options`` is where a backend validates, so a raise there leaves the env as built."""

    class PickyDeployment(LoopbackDeployment):
        def _resolve_options(self) -> None:
            super()._resolve_options()
            raise ValueError("this backend does not like this config")

    sim_entity = g1_env.entities["robot"]
    with pytest.raises(ValueError, match="does not like"):
        deploy(PickyDeployment)

    assert g1_env.entities["robot"] is sim_entity
    assert "rolled back" not in eden_caplog.text


@pytest.mark.parametrize(
    ("options", "feet"),
    [
        ({}, G1Dofs29.feet_links_name),
        (
            {"feet_links": ("left_ankle_pitch_link", "right_ankle_pitch_link")},
            ("left_ankle_pitch_link", "right_ankle_pitch_link"),
        ),
        ({"entity_name": "hand"}, ()),
    ],
    ids=["from-the-robot", "option-overrides", "footless"],
)
def test_the_feet_come_from_the_robot_unless_the_option_names_them(deploy, options, feet):
    assert deploy(**options).feet_links == tuple(feet)


def test_the_estimator_builds_on_a_footed_robot_with_no_flag(deploy):
    """``eden deploy --mujoco --estimate_lin_vel`` on a G1 needs no ``--feet_links``."""
    assert deploy(estimate_lin_vel=True).provides_base_lin_vel


def test_without_the_estimator_a_hardware_backend_provides_no_base_velocity(deploy_two_link):
    assert not deploy_two_link().provides_base_lin_vel


# ---------------------------------------------------------------------------
# The entity swap
# ---------------------------------------------------------------------------


def test_the_swap_repoints_every_handle_cached_on_the_robot(g1_env, deploy):
    """Terms and modifiers that cached the sim entity read the real robot's state on hardware."""
    sim_robot, sim_hand = g1_env.entities["robot"], g1_env.entities["hand"]
    terms = g1_env.action_manager.terms
    clips = [m for name in ("legs", "arms") for m in with_owned_modifiers([terms[name]]) if isinstance(m, EnvelopeClip)]
    [probe_modifier] = [
        modifier
        for term in g1_env.observation_manager._group_obs_terms["policy"]
        for modifier in getattr(term, "_modifiers", ())
        if isinstance(modifier, CachedEntityModifier)
    ]

    dep = deploy()

    wrapper = g1_env.entities["robot"]
    assert isinstance(wrapper, RobotStateEntity) and wrapper is not sim_robot
    assert terms["legs"].entity is wrapper and terms["arms"].entity is wrapper
    assert len(clips) == 2 and all(clip._entity is wrapper for clip in clips), "the modifiers a term owns"
    assert _policy_term(g1_env, CachedEntityQuat).robot is wrapper
    assert probe_modifier._robot is wrapper, "an observation term's modifiers"
    assert terms["hand"].entity is sim_hand and g1_env.entities["hand"] is sim_hand, "another entity stays put"
    assert dep._state_entity is wrapper
    robot_side = with_owned_modifiers([terms["legs"], terms["arms"]])
    stale = [type(obj).__name__ for obj in robot_side if any(value is sim_robot for value in vars(obj).values())]
    assert stale == [], "nothing on the robot's action path still reads the simulation"


def test_the_scan_recurses_into_the_modifiers_a_compose_holds(g1_env):
    """``Compose`` keeps its children in a list, one level below the term's ``_modifier``."""
    scanned = with_owned_modifiers([g1_env.action_manager.terms["legs"]])
    assert [type(obj) for obj in scanned] == [ImplicitPDController, Compose, EnvelopeClip, ActionDelay]


def test_observations_read_the_measured_state_through_a_cached_handle(g1_env, deploy):
    rolled = np.array([np.cos(0.2), np.sin(0.2), 0.0, 0.0])
    dep = deploy()
    dep.readings.append(robot_state(dep.num_dofs, base_quat=rolled, dofs_pos=np.full(dep.num_dofs, 0.3)))

    _steps(dep, 1)

    np.testing.assert_allclose(_policy_term(g1_env, CachedEntityQuat).compute()[0].numpy(), rolled, atol=1e-6)
    np.testing.assert_allclose(g1_env.entities["robot"].get_dofs_pos()[0].numpy(), 0.3, atol=1e-6)


@pytest.mark.parametrize(
    "form",
    [
        lambda entity: entity.get_quat,
        lambda entity: [entity],
        lambda entity: (entity,),
        lambda entity: {entity},
        lambda entity: {"robot": entity},
    ],
    ids=["bound-method", "list", "tuple", "set", "dict"],
)
def test_a_handle_the_swap_cannot_repoint_fails_construction_and_leaves_the_env_as_built(
    g1_env, deploy, monkeypatch, form
):
    """Loud at deploy init, not a stale read on the robot; and the scan finishes before anything is written."""
    sim_entity = g1_env.entities["robot"]
    monkeypatch.setattr(_policy_term(g1_env, CachedEntityQuat), "handles", form(sim_entity), raising=False)

    with pytest.raises(RuntimeError, match="cannot repoint"):
        deploy()

    assert g1_env.entities["robot"] is sim_entity
    assert g1_env.action_manager.terms["legs"].entity is sim_entity, "scanned, but never written"
    assert _policy_term(g1_env, CachedEntityQuat).robot is sim_entity


def test_a_failure_after_the_swap_rolls_it_back_and_a_retry_fails_the_same_way(g1_env, deploy, eden_caplog):
    """Without the rollback, a retry would wrap the wrapper and fail somewhere else entirely."""
    sim_entity = g1_env.entities["robot"]
    messages = []
    for _ in range(2):
        with pytest.raises(Exception, match="Link not found for name: l_foot") as exc_info:
            deploy(estimate_lin_vel=True, feet_links=("l_foot", "r_foot"))
        messages.append(str(exc_info.value))

    assert messages[0] == messages[1]
    assert g1_env.entities["robot"] is sim_entity
    assert g1_env.action_manager.terms["legs"].entity is sim_entity, "the repoint is undone too"
    assert _policy_term(g1_env, CachedEntityQuat).robot is sim_entity
    assert "rolled back the 'robot' entity swap" in eden_caplog.text


# ---------------------------------------------------------------------------
# Command overrides
# ---------------------------------------------------------------------------


def test_an_override_replaces_the_command_everything_reads(g1_env, deploy):
    """Including an observation term that cached the trained generator in its ``build()``."""
    trained = g1_env.command_manager.get_term("velocity_command")

    deploy(command_overrides={"velocity_command": _held(command_value=(0.5, 0.0, 0.2))})

    held = g1_env.command_manager.get_term("velocity_command")
    assert isinstance(held, ConstantVelocityCommand) and held is not trained
    np.testing.assert_allclose(_policy_term(g1_env, ScaledCommands).compute()[0].numpy(), [0.5, 0.0, 0.2])


def test_close_closes_the_substitute_once_and_restores_the_trained_command(g1_env, deploy, monkeypatch):
    """A substitute outlives the deploy as a dead term; the next deployment on this env must find the trained one."""
    monkeypatch.setattr(DeviceOwningVelocityCommand, "_closes", 0)
    trained = g1_env.command_manager.get_term("velocity_command")
    options = DeviceOwningVelocityCommand.configure(entity_name="robot", **_RANGES)
    dep = deploy(command_overrides={"velocity_command": options})

    dep.close()
    dep.close()

    assert DeviceOwningVelocityCommand._closes == 1
    assert g1_env.command_manager.get_term("velocity_command") is trained
    torch.testing.assert_close(_policy_term(g1_env, ScaledCommands).compute(), trained.command, msg="repoint undone")


def test_a_substitute_whose_close_raises_is_still_replaced(g1_env, deploy, eden_caplog):
    trained = g1_env.command_manager.get_term("velocity_command")
    dep = deploy(
        command_overrides={"velocity_command": BrokenDriverVelocityCommand.configure(entity_name="robot", **_RANGES)}
    )

    dep.close()

    assert g1_env.command_manager.get_term("velocity_command") is trained
    assert "Could not close the substituted command term 'velocity_command'" in eden_caplog.text


def test_a_failure_after_the_override_unwinds_it_without_displacing_the_error(g1_env, deploy, monkeypatch):
    """The rollback runs inside ``__init__``'s ``except``: a throwing ``close`` there must not replace the real error."""
    sim_entity = g1_env.entities["robot"]
    trained = g1_env.command_manager.get_term("velocity_command")
    # A term holding the command where the swap cannot repoint it: the override is applied, then refused.
    monkeypatch.setattr(_policy_term(g1_env, CachedEntityQuat), "commands", [trained], raising=False)
    options = BrokenDriverVelocityCommand.configure(entity_name="robot", **_RANGES)

    with pytest.raises(RuntimeError, match="replaced command term inside a list"):
        deploy(command_overrides={"velocity_command": options})

    assert g1_env.command_manager.get_term("velocity_command") is trained
    assert g1_env.entities["robot"] is sim_entity


def test_an_override_on_an_env_without_commands_raises(two_link_env, deploy_two_link):
    sim_entity = two_link_env.entities["robot"]
    with pytest.raises(ValueError, match="no command manager"):
        deploy_two_link(command_overrides={"velocity_command": _held()})
    assert two_link_env.entities["robot"] is sim_entity


def test_every_backend_close_runs_the_shared_teardown():
    """``close`` is a backend's only teardown hook; one that skips ``_teardown_common`` loses ``--debug_record``.

    A source check because the hardware backends cannot be closed here without their SDKs.
    """
    import importlib
    import inspect
    import pkgutil

    import eden.extensions.deployment as deployment_pkg

    for module in pkgutil.iter_modules(deployment_pkg.__path__):
        if not module.ispkg and module.name != "base":
            importlib.import_module(f"eden.extensions.deployment.{module.name}")

    def tears_down(cls) -> bool:
        """Whether ``cls.close`` runs the teardown itself, or through a ``super().close()`` that does."""
        if cls.close is DeploymentBase.close:
            return False
        source = inspect.getsource(cls.close)
        if "_teardown_common" in source:
            return True
        owner = next(klass for klass in cls.__mro__ if "close" in vars(klass))
        parent = cls.__mro__[cls.__mro__.index(owner) + 1]
        return "super().close()" in source and tears_down(parent)

    missing = [
        cls.__name__ for _, cls in DEPLOYMENT_REGISTRY if cls.close is not DeploymentBase.close and not tears_down(cls)
    ]
    assert missing == []


# ---------------------------------------------------------------------------
# Which action terms reach the payload
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unsendable_terms(g1_env):
    """Built terms whose ``processed_action`` a joint-position payload cannot gather from."""
    from eden import actions
    from eden.managers.terms.actions.welding_actions import ParallelJawWelding, SuctionCupWelding

    def built(cls, **options):
        term = cls(g1_env, cls.configure(**options))
        term.build()
        return term

    return {
        "suction": built(SuctionCupWelding, entity_name="robot", ee_link_name="left_wrist_yaw_link"),
        "hand_suction": built(SuctionCupWelding, entity_name="hand", ee_link_name="f1"),
        "weld": built(
            ParallelJawWelding,
            entity_name="robot",
            ee_link_name="left_wrist_yaw_link",
            left_gripper_link_name="left_wrist_yaw_link",
            right_gripper_link_name="left_wrist_pitch_link",
        ),
        "vel_ff": built(
            actions.VelocityFeedforwardPDController,
            entity_name="robot",
            dofs_name=["left_knee_joint", "right_knee_joint"],
        ),
        "unbuilt": actions.ImplicitPDController(
            g1_env, actions.ImplicitPDController.configure(entity_name="robot", dofs_name=["left_knee_joint"])
        ),
    }


def _maps(env, names, dofs_name, extra=None, ignore=()):
    terms = {name: env.action_manager.terms[name] for name in names} | dict(extra or {})
    return build_term_action_maps(terms, dofs_name, env.device, ignore, entity_name="robot")


def test_the_payload_carries_each_terms_targets_in_the_robots_dof_order(g1_env, deploy):
    """``legs`` lists its joints in reverse; the payload is in wire order, and the hand's term is not in it."""
    dep = deploy()
    terms = g1_env.action_manager.terms
    action = torch.linspace(-0.5, 0.5, sum(term.action_dim for term in terms.values())).unsqueeze(0)

    payload = dep.action_to_payload(action)

    for name in ("legs", "arms"):
        for column, dof in enumerate(terms[name].dofs_name):
            assert payload.dofs_pos[dep.dofs_name.index(dof)] == pytest.approx(
                float(terms[name].processed_action[0, column])
            ), dof
    assert len(payload.dofs_pos) == len(G1_DOFS29_NAME)


def test_a_fully_mapped_term_logs_nothing(g1_env, eden_caplog):
    legs = g1_env.action_manager.terms["legs"]
    [(term, _, _)] = _maps(g1_env, ["legs"], legs.dofs_name)
    assert term is legs
    assert eden_caplog.records == []


@pytest.mark.parametrize(
    ("names", "dofs", "level", "message"),
    [
        (
            ["legs", "arms", "hand"],
            "robot",
            "WARNING",
            "check this is a second robot and not a mis-declared entity_name",
        ),
        (["legs", "hand"], "legs", "INFO", "drives entity 'hand', not the deployed 'robot', so its 2 DOF(s)"),
        (["legs"], "legs[:-2]", "WARNING", "2 DOF(s) missing from the declared dofs_name of the deployed 'robot'"),
        (["legs", "arms"], "legs", "WARNING", "none of the 17 DOF(s) it controls are in that robot's declared"),
        (["legs"], "robot", "WARNING", "17 DOF(s) of the deployed 'robot' are reached by no action term"),
    ],
    ids=["other-entity-shared-name", "other-entity", "term-overhangs-robot", "term-misses-robot", "robot-unreached"],
)
def test_each_kind_of_mismatch_is_logged_once_at_its_level(g1_env, eden_caplog, names, dofs, level, message):
    terms = g1_env.action_manager.terms
    legs = terms["legs"].dofs_name
    dofs_name = {"robot": list(G1_DOFS29_NAME), "legs": legs, "legs[:-2]": legs[:-2]}[dofs]

    maps = _maps(g1_env, names, dofs_name)

    reached = ["legs", "arms"] if dofs == "robot" and "arms" in names else ["legs"]
    assert [term for term, _, _ in maps] == [terms[name] for name in reached]
    assert [record.levelname for record in eden_caplog.records] == [level]
    assert message in eden_caplog.text


def test_a_long_dof_list_is_truncated_in_the_log(g1_env, eden_caplog):
    _maps(g1_env, ["legs", "arms"], g1_env.action_manager.terms["legs"].dofs_name)
    assert "(+12 more)" in eden_caplog.text


def test_another_entitys_term_may_be_acknowledged(g1_env):
    """Acknowledging a term the function drops anyway must not be rejected as one it carries."""
    maps = _maps(g1_env, ["legs", "arms", "hand"], list(G1_DOFS29_NAME), ignore=("hand",))
    assert [term for term, _, _ in maps] == [g1_env.action_manager.terms[name] for name in ("legs", "arms")]


@pytest.mark.parametrize(
    ("extra", "ignore", "outcome"),
    [
        ("suction", (), "raises"),
        ("suction", ("suction",), "excluded"),
        ("hand_suction", (), "raises"),
        ("weld", (), "excluded"),
        ("vel_ff", (), r"processed_action is \(1, 4\), not \(num_envs, 2\)"),
        ("unbuilt", (), "never built"),
    ],
    ids=["suction", "suction-acknowledged", "another-entitys-suction", "surrogate-weld", "wrong-width", "unbuilt"],
)
def test_a_term_the_payload_cannot_carry_raises_unless_its_effect_is_accounted_for(
    g1_env, unsendable_terms, extra, ignore, outcome
):
    """A welding term's 1-D buffer is refused at init instead of reaching the first hardware step and dying there.

    The gather check runs before the entity filter, so a vacuum on another entity is still refused.
    """
    terms = {extra: unsendable_terms[extra]}
    if outcome == "excluded":
        maps = _maps(g1_env, ["legs", "arms"], list(G1_DOFS29_NAME), terms, ignore)
        assert [term for term, _, _ in maps] == [g1_env.action_manager.terms[name] for name in ("legs", "arms")]
        return
    match = f"'{extra}'.*cannot feed a joint-position payload" if outcome == "raises" else outcome
    with pytest.raises(ValueError, match=match):
        _maps(g1_env, ["legs", "arms"], list(G1_DOFS29_NAME), terms, ignore)


def test_action_delay_is_named_only_on_terms_the_payload_carries(deploy, eden_caplog):
    """No deployment applies an ``ActionDelay``. ``legs`` holds one inside a ``Compose``; ``hand`` holds one but is
    never sent.
    """
    deploy()
    assert "ActionDelay on action term(s) ['legs'] is not applied" in eden_caplog.text


def test_a_task_without_action_delay_says_nothing_about_it(deploy_two_link, eden_caplog):
    deploy_two_link()
    assert "ActionDelay" not in eden_caplog.text


def test_gains_are_the_robots_scaled_with_disabled_joints_limp(deploy):
    dep = deploy(deploy_kp_kd_scale=0.5, disable_joints=("left_wrist_*",))
    _steps(dep, 1)

    kp = 0.5 * resolve_dofs_values(G1Dofs29().default_dofs_kp, dep.dofs_name)
    wrist = [i for i, name in enumerate(dep.dofs_name) if name.startswith("left_wrist_")]
    kp[wrist] = 0.0
    np.testing.assert_allclose(dep.sent[-1].dofs_kp, kp)
    assert dep.sent[-1].dofs_kd[wrist].tolist() == [0.0, 0.0, 0.0]
    assert dep.disabled_dofs == ("left_wrist_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint")


# ---------------------------------------------------------------------------
# What a step sends
# ---------------------------------------------------------------------------


def test_nothing_is_sent_until_a_slot_writes_its_targets(deploy):
    """An all-zeros command drives every joint to 0 rad; it must never go out as a stand-in for "nothing yet"."""
    dep = deploy()
    start_policy(dep)  # the operator hands control to the policy

    dep.step(None)

    assert dep.sent == []
    with pytest.raises(RuntimeError, match="before anything wrote joint targets"):
        dep.action_to_payload(None)


def test_an_action_does_not_arm_the_direct_target_path(g1_env, deploy):
    """``None`` after an action-manager step means "nothing to send", not "resend the buffer"."""
    dep = deploy()
    dep.action_to_payload(torch.zeros(1, sum(term.action_dim for term in g1_env.action_manager.terms.values())))
    with pytest.raises(RuntimeError, match="before anything wrote joint targets"):
        dep.action_to_payload(None)


def test_written_targets_are_sent_every_step_in_wire_order(deploy):
    """The MuJoCo sim froze for a whole session when ``step(None)`` skipped the send. Zero is a target like any other."""
    dep = deploy()
    start_policy(dep)  # the operator hands control to the policy
    first, third = dep.dofs_name[0], dep.dofs_name[2]

    for value in (0.1, 0.2, 0.0):
        dep.write_dofs_targets(torch.tensor([value, -value]), [third, first])
        dep.step(None)

    assert len(dep.sent) == 3
    np.testing.assert_allclose([s.dofs_pos[[0, 2]] for s in dep.sent], [[-0.1, 0.1], [-0.2, 0.2], [0.0, 0.0]])
    np.testing.assert_allclose(dep.action_to_payload(None).dofs_pos[[0, 2]], [0.0, 0.0], err_msg="the latch holds")


def test_target_lpf_smooths_every_command_it_is_sent(deploy):
    dep = deploy(target_lpf=0.25)
    dep.request_start()  # the operator hands control to the policy
    hold, a, b = dep.default_dof_pos, np.full(dep.num_dofs, 0.4), np.full(dep.num_dofs, -0.4)

    for target in (a, a, b):  # the START tick holds; the policy's targets follow
        _steps(dep, 1, target)

    np.testing.assert_allclose(dep.sent[0].dofs_pos, hold, err_msg="warm-started: no startup transient")
    first = 0.25 * a + 0.75 * hold
    np.testing.assert_allclose(
        dep.sent[1].dofs_pos, first, rtol=1e-6, err_msg="the policy's first command, from the hold"
    )
    np.testing.assert_allclose(dep.sent[2].dofs_pos, 0.25 * b + 0.75 * first, rtol=1e-6)


def test_target_lpf_hands_back_a_copy_and_rewarms_on_a_new_width(deploy, deploy_two_link):
    """Backends write ``payload.dofs_pos = _apply_target_lpf(...)``: an alias would let the payload corrupt the filter."""
    dep = deploy(target_lpf=0.5)
    dep._apply_target_lpf(np.zeros(3))
    out = dep._apply_target_lpf(np.full(3, 2.0))
    out[:] = 999.0
    np.testing.assert_allclose(dep._apply_target_lpf(np.full(3, 2.0)), 1.5)
    np.testing.assert_allclose(dep._apply_target_lpf(np.array([5.0, 6.0])), [5.0, 6.0])

    unfiltered = deploy_two_link(target_lpf=1.0)
    x = np.array([0.1, -0.2])
    assert unfiltered._apply_target_lpf(x) is x


# ---------------------------------------------------------------------------
# Each policy slot's own PD gains
# ---------------------------------------------------------------------------


def _stiff_hips(cls=G1Dofs29, kp: float = 200.0, kd: float = 8.0):
    """``cls`` on the stock G1 gains with its hip pitch replaced: a policy trained on another motor, as SONIC's is."""
    stock = G1Dofs29()
    return cls(
        default_dofs_kp={**stock.default_dofs_kp, "*_hip_pitch_joint": kp},
        default_dofs_kd={**stock.default_dofs_kd, "*_hip_pitch_joint": kd},
    )


def test_a_slots_gains_are_its_robots_under_the_runs_scale_and_disabled_joints(deploy):
    dep = deploy(deploy_kp_kd_scale=0.5, disable_joints=("left_hip_pitch_joint",))
    left, right = dep.dofs_name.index("left_hip_pitch_joint"), dep.dofs_name.index("right_hip_pitch_joint")
    rest = [i for i in range(dep.num_dofs) if i not in (left, right)]

    kp, kd = dep.wire_gains(_stiff_hips())

    assert (kp[right], kd[right]) == pytest.approx((100.0, 4.0))
    assert (kp[left], kd[left]) == (0.0, 0.0), "--disable_joints holds for every slot, not only the boot one"
    np.testing.assert_allclose(kp[rest], dep.default_dof_kp[rest])
    np.testing.assert_allclose(kd[rest], dep.default_dof_kd[rest])


def test_a_slot_falls_back_to_the_deployed_robots_gains_on_every_dof_its_robot_does_not_name(deploy):
    """No robot at all (a checkpoint, a policy pinning none), one declaring no gains, or a G1Dofs23 on the 29-DOF G1."""
    dep = deploy()
    np.testing.assert_array_equal(dep.wire_gains(None)[0], dep.default_dof_kp)
    assert G1Dofs23().default_dofs_kp == {}, "trains on the asset's gains, which a deployment cannot read"
    np.testing.assert_array_equal(dep.wire_gains(G1Dofs23())[0], dep.default_dof_kp)

    full_kp, full_kd = dep.wire_gains(_stiff_hips())
    subset = G1Dofs23().dofs_name
    kp, _ = dep.wire_gains(
        G1Dofs23(
            default_dofs_kp={name: full_kp[dep.dofs_name.index(name)] for name in subset},
            default_dofs_kd={name: full_kd[dep.dofs_name.index(name)] for name in subset},
        )
    )

    unnamed = [dep.dofs_name.index(name) for name in sorted(set(dep.dofs_name) - set(G1Dofs23().dofs_name))]
    named = [i for i in range(dep.num_dofs) if i not in unnamed]
    assert unnamed, "G1Dofs23 has no wrist pitch/yaw"
    np.testing.assert_allclose(kp[unnamed], dep.default_dof_kp[unnamed])
    np.testing.assert_allclose(kp[named], full_kp[named], err_msg="none of them limp")


def test_a_pinned_robot_whose_gain_map_leaves_out_a_driven_joint_is_refused(deploy):
    """Pydantic replaces a dict field whole: an override-only map would have sent kp = 0 to every other joint."""
    dep = deploy()
    with pytest.raises(ValueError, match=r"G1Dofs29's default_dofs_kp gives no value for \['left_hip_roll_joint'"):
        dep.wire_gains(G1Dofs29(default_dofs_kp={"*_hip_pitch_joint": 200.0}))


@pytest.mark.parametrize("cls", [LoopbackDeployment, OperatorLoopbackDeployment], ids=["policy", "stop_hold"])
def test_set_gains_ramps_every_command_linearly_over_gain_ramp_steps(deploy, cls):
    """The policy's command and the STOP hold alike: no payload may jump to the new stiffness."""
    dep = deploy(cls, connect=True, gain_ramp_steps=4)
    kp0, kd0 = dep.default_dof_kp.copy(), dep.default_dof_kd.copy()

    dep.set_gains(3 * kp0, 3 * kd0)
    _steps(dep, 5)

    for payload, fraction in zip(dep.sent, (0.25, 0.5, 0.75, 1.0, 1.0), strict=True):
        np.testing.assert_allclose(payload.dofs_kp, kp0 * (1 + 2 * fraction))
        np.testing.assert_allclose(payload.dofs_kd, kd0 * (1 + 2 * fraction))


def test_a_switch_mid_ramp_ramps_from_the_gains_on_the_wire(deploy):
    dep = deploy(gain_ramp_steps=2)
    kp0, kd0 = dep.default_dof_kp.copy(), dep.default_dof_kd.copy()

    dep.set_gains(3 * kp0, kd0)
    _steps(dep, 1)  # halfway: 2 * kp0
    dep.set_gains(kp0, kd0)
    _steps(dep, 2)

    np.testing.assert_allclose([p.dofs_kp for p in dep.sent], [2 * kp0, 1.5 * kp0, kp0])


def test_zero_ramp_steps_sends_the_new_gains_on_the_next_command(deploy):
    dep = deploy()
    kp, kd = dep.wire_gains(_stiff_hips())

    dep.set_gains(kp, kd, ramp_steps=0)
    _steps(dep, 1)

    np.testing.assert_array_equal(dep.sent[0].dofs_kp, kp)
    np.testing.assert_array_equal(dep.sent[0].dofs_kd, kd)


def test_set_gains_refuses_gains_that_are_not_one_per_deployment_dof(deploy):
    with pytest.raises(ValueError, match=r"got 3 kp and 3 kd\. Build them with wire_gains"):
        deploy().set_gains(np.ones(3), np.ones(3))


def test_set_gains_refuses_a_backend_whose_robot_ignores_command_gains(deploy):
    """The Wuji/XHand hands and the Franka: a ramp there would be a no-op the caller never hears about."""
    dep = deploy(FirmwareGainsLoopbackDeployment)
    with pytest.raises(RuntimeError, match="would change nothing on the robot"):
        dep.set_gains(dep.default_dof_kp, dep.default_dof_kd)


# ---------------------------------------------------------------------------
# --debug_record through the loop
# ---------------------------------------------------------------------------


def _record_run(dep, n: int, **extra) -> None:
    for i in range(n):
        dep.readings.append(robot_state(dep.num_dofs, stamp=float(i), extra={k: v(i) for k, v in extra.items()}))
        _steps(dep, 1)


def test_the_recording_holds_one_row_per_step_across_chunks(deploy, tmp_path):
    path = tmp_path / "rec.npz"
    dep = deploy(debug_record=True, debug_record_path=str(path), debug_record_flush_steps=5)

    _record_run(dep, 12, motor_temp=lambda i: np.full(2, 40.0 + i))
    dep.close()

    rec = load_deploy_record(str(path))
    assert rec["stamp"] == pytest.approx(np.arange(12, dtype=float))
    for key in ("cmd_pos", "cmd_kp", "cmd_kd", "q", "qd", "tau", "base_quat", "base_ang_vel", "base_lin_acc"):
        assert rec[key].shape[0] == 12, key
    assert list(rec["dofs_name"]) == dep.dofs_name
    np.testing.assert_allclose(rec["x_motor_temp"][[0, -1]], [[40.0, 40.0], [51.0, 51.0]])


def test_a_killed_run_keeps_every_flushed_chunk(deploy, tmp_path):
    """No ``close()`` at all, as a hard kill leaves it: 23 steps at 5 a chunk is four chunks on disk."""
    path = tmp_path / "rec.npz"
    dep = deploy(debug_record=True, debug_record_path=str(path), debug_record_flush_steps=5)

    _record_run(dep, 23)

    deadline = time.monotonic() + 5.0
    stamps = np.zeros(0)
    while time.monotonic() < deadline and len(stamps) < 20:
        try:
            stamps = load_deploy_record(str(path))["stamp"]
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    assert stamps == pytest.approx(np.arange(20, dtype=float))


def test_the_recording_carries_the_estimators_contacts_and_forces(deploy, tmp_path):
    """The record is taken after the observation pass, or the estimate never reaches it."""
    path = tmp_path / "rec.npz"
    dep = deploy(estimate_lin_vel=True, debug_record=True, debug_record_path=str(path))

    _record_run(dep, 3)
    dep.close()

    rec = load_deploy_record(str(path))
    assert rec["x_base_lin_vel_estimate"].shape == (3, 3)
    assert rec["x_contacts"].shape == (3, 2)
    assert rec["x_forces"].shape == (3, 6)


def test_a_tick_whose_observation_raises_is_still_recorded(deploy, tmp_path):
    """The crashing tick is the one a post-mortem wants. What it never got to is NaN, not a dropped row."""
    path = tmp_path / "rec.npz"
    dep = deploy(estimate_lin_vel=True, debug_record=True, debug_record_path=str(path))
    _record_run(dep, 2)
    dep.readings.append(robot_state(dep.num_dofs, stamp=2.0, base_quat=None))  # the IMU dropped out

    with pytest.raises((TypeError, RuntimeError)):
        _steps(dep, 1)
    dep.close()

    rec = load_deploy_record(str(path))
    assert rec["stamp"] == pytest.approx([0.0, 1.0, 2.0])
    assert np.isfinite(rec["x_base_lin_vel_estimate"][:2]).all()
    assert np.isnan(rec["x_base_lin_vel_estimate"][2]).all()


# ---------------------------------------------------------------------------
# The operator state machine
# ---------------------------------------------------------------------------


def test_a_backend_with_no_operator_holds_instead_of_starting_itself(deploy, eden_caplog):
    """Only a simulator may run INIT and START unattended; anything else holds, and says so once."""
    dep = deploy(UnattendedLoopbackDeployment, connect=True)

    _steps(dep, 3, np.full(dep.num_dofs, 0.5))

    for sent in dep.sent:
        np.testing.assert_allclose(sent.dofs_pos, dep.default_dof_pos, err_msg="the hold, not the policy's 0.5")
    assert eden_caplog.text.count("no operator input is open") == 1


@pytest.mark.parametrize("cls", [LoopbackDeployment, UnattendedLoopbackDeployment])
def test_every_backend_boots_holding_until_it_is_told_to_start(deploy, cls):
    """No backend hands control to the policy on its own: an arm and a hand boot in STOP like a humanoid."""
    dep = deploy(cls, connect=True)
    dep.reset()

    _steps(dep, 2, np.full(dep.num_dofs, 0.5))

    assert not dep.policy_active
    for sent in dep.sent:
        np.testing.assert_allclose(sent.dofs_pos, dep.default_dof_pos, err_msg="the hold, not the policy's 0.5")


def test_a_script_drives_the_state_machine_without_an_operator(deploy, eden_caplog):
    """``request_init`` / ``request_start`` are the keys a script presses on purpose; nothing warns about them."""
    dep = deploy(UnattendedLoopbackDeployment, init_steps=2, connect=True)
    start = np.full(dep.num_dofs, 0.4)
    dep.readings.append(robot_state(dep.num_dofs, dofs_pos=start))
    dep.reset()

    dep.request_init()
    _steps(dep, 3, np.full(dep.num_dofs, 0.5))
    assert dep.init_done and not dep.policy_active
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, dep.default_dof_pos, err_msg="INIT ramped to the stance")

    dep.request_start()
    _steps(dep, 2, np.full(dep.num_dofs, 0.5))
    np.testing.assert_allclose(dep.sent[-2].dofs_pos, dep.default_dof_pos, err_msg="the START tick holds")
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, 0.5)

    dep.request_stop()
    _steps(dep, 1, np.full(dep.num_dofs, 0.9))
    assert not dep.policy_active
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, 0.5, err_msg="STOP holds where the robot is")
    assert "no operator input is open" not in eden_caplog.text


def test_a_simulator_with_no_operator_runs_init_then_start(deploy):
    class Simulator(UnattendedLoopbackDeployment):
        allow_unattended_start = True

    dep = deploy(Simulator, init_steps=2, connect=True)

    _steps(dep, 5, np.full(dep.num_dofs, 0.5))

    np.testing.assert_allclose(dep.sent[-1].dofs_pos, 0.5, err_msg="START handed control to the policy")


def test_init_ramps_from_the_measured_pose_to_the_default_stance(deploy):
    dep = deploy(OperatorLoopbackDeployment, init_steps=4, init_dofs_kp_scale=0.5, connect=True)
    start = np.full(dep.num_dofs, 0.4)
    dep.readings.append(robot_state(dep.num_dofs, dofs_pos=start))
    _steps(dep, 1)  # boots in STOP, holding
    dep.operator.press(CMD_INIT)

    _steps(dep, 6, np.full(dep.num_dofs, 9.0))

    ramp = [sent.dofs_pos for sent in dep.sent[1:]]
    for k, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0, 1.0]):
        np.testing.assert_allclose(ramp[k], start + alpha * (dep.default_dof_pos - start), err_msg=f"tick {k}")
    np.testing.assert_allclose(dep.sent[-1].dofs_kp, 0.5 * dep.default_dof_kp)


@pytest.mark.parametrize(
    ("entity_name", "init_target", "expected"),
    [
        ("robot", "default", "default"),
        ("robot", "motion", "clip"),
        ("robot", "motion_rsi", "clip"),
        ("hand", "motion", "default"),
    ],
    ids=["default", "motion", "motion-rsi-on-a-simulator", "nothing-tracks-the-entity"],
)
def test_init_ramps_to_the_init_target(deploy, eden_caplog, entity_name, init_target, expected):
    class Simulator(OperatorLoopbackDeployment):
        supports_base_placement = True

    dep = deploy(Simulator, entity_name=entity_name, init_target=init_target, init_steps=1, connect=True)
    dep.operator.press(CMD_INIT)

    _steps(dep, 2)

    if expected == "clip":
        target = MANAGER_CLIP_JOINT_STEP * np.array([G1_DOFS29_NAME.index(name) for name in dep.dofs_name])
    else:
        target = dep.default_dof_pos
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, target, atol=1e-6)
    assert ("tracks no motion" in eden_caplog.text) == (entity_name == "hand")


def test_a_script_that_reads_the_keyboard_itself_can_close_the_terminal_reader(deploy, monkeypatch, eden_caplog):
    """A pynput teleop sees the terminal's keys too: ``s`` would be a teleop axis and START on the same line."""
    from eden.extensions.deployment.utils import operator_input

    opened = []
    monkeypatch.setattr(operator_input.KeyboardInput, "start", lambda self: opened.append(self) or False)
    deploy(connect=True)
    assert len(opened) == 1, "the default opens it"

    dep = deploy(operator_keyboard=False, connect=True)
    dep.reset()
    _steps(dep, 2)
    assert len(opened) == 1 and dep._inputs == []
    assert "Holding in STOP" not in eden_caplog.text, "the script relays the commands"


def test_the_action_handed_to_the_start_tick_is_never_sent(deploy):
    """The loop computed it before START: off the hold's observation, in the old yaw frame, by a slot not yet re-armed.

    So the START tick holds, and its read -- the windows cleared, the clip at frame 0 -- is what the
    re-armed slot acts on: the policy's first command goes out on the next tick.
    """
    dep = deploy(OperatorLoopbackDeployment, init_steps=1, connect=True)
    _steps(dep, 1)
    dep.operator.press(CMD_START)

    _steps(dep, 1, np.full(dep.num_dofs, 0.5))

    assert dep.policy_active and dep.consume_start_events() == 1
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, dep.default_dof_pos, err_msg="the hold, not the stale 0.5")
    _steps(dep, 1, np.full(dep.num_dofs, 0.7))
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, 0.7, err_msg="the first action computed after START")


def test_only_the_policys_commands_are_sent_as_the_policys(deploy):
    """``sending_hold`` is how a backend tells the state machine's own command from the policy's.

    The Franka leaves its gripper alone on a hold: a measured-width target would stop a
    force-controlled grasp, and INIT's default width would open it.
    """

    class Recording(OperatorLoopbackDeployment):
        def send_payload(self, payload) -> None:
            self.holds.append(self.sending_hold)
            super().send_payload(payload)

    dep = deploy(Recording, init_steps=1, connect=True)
    dep.holds = []
    _steps(dep, 1)  # STOP
    dep.operator.press(CMD_INIT)
    _steps(dep, 2)  # the ramp, then its hold
    dep.operator.press(CMD_START)
    _steps(dep, 2)  # the START tick, then the policy
    dep.operator.press(CMD_STOP)
    _steps(dep, 1)

    assert dep.holds == [True, True, True, True, False, True]
    dep.send_payload(dep._hold_payload())
    assert dep.holds[-1] is False, "a command sent from outside step() is its sender's, as a sysid excitation is"


def test_policy_active_follows_the_operator_gate(deploy):
    assert not deploy(LoopbackDeployment).policy_active, "every backend boots in STOP"
    dep = deploy(OperatorLoopbackDeployment, init_steps=1, connect=True)
    assert not dep.policy_active, "boots in STOP"
    dep.operator.press(CMD_INIT)
    _steps(dep, 1)
    assert not dep.policy_active
    dep.operator.press(CMD_START)
    _steps(dep, 1)
    assert dep.policy_active
    dep.operator.press(CMD_STOP)
    _steps(dep, 1)
    assert not dep.policy_active


def test_init_and_start_rewind_the_clip_wherever_it_was_left(g1_env, deploy):
    """STOP at frame 2 then INIT must ramp to frame 0; and the policy is handed the clip from its start."""
    motion = g1_env.command_manager.get_term("motion")
    dep = deploy(OperatorLoopbackDeployment, init_steps=1, connect=True)
    dep.operator.press(CMD_INIT)
    _steps(dep, 2)
    dep.operator.press(CMD_START)
    _steps(dep, 5)  # the START tick holds the clip at frame 0; four policy ticks play it out
    assert int(motion.time_steps[0]) == COMMAND_CLIP_FRAMES - 1 and dep.motion_finished()

    dep.operator.press(CMD_STOP)
    _steps(dep, 3)
    assert int(motion.time_steps[0]) == COMMAND_CLIP_FRAMES - 1, "the clip holds while the policy is off"
    dep.operator.press(CMD_INIT)
    _steps(dep, 1)
    assert int(motion.time_steps[0]) == 0
    assert float(g1_env.motion_manager.get_motion_phase("robot").max()) == 0.0
    dep.operator.press(CMD_START)
    _steps(dep, 1)
    assert int(motion.time_steps[0]) == 0, "the START tick reads the policy's first frame"
    _steps(dep, 1)
    assert int(motion.time_steps[0]) == 1


def _frozen_joint_run(dep, *, freeze: bool, ramp: bool) -> None:
    """Hold, then START; with ``ramp``, the robot reads 0.6 rad off its stance and INIT ramps it back first."""
    offset = 0.6 if ramp else 0.0
    if freeze:  # bit-constant at the pose it was left in, as the G1's stuck wrist was
        dep.frozen[dep.dofs_name[0]] = float(dep.default_dof_pos[0]) + offset
    if ramp:
        dep.readings.append(robot_state(dep.num_dofs, dofs_pos=dep.default_dof_pos + 0.6))
    _steps(dep, 2)
    if ramp:
        dep.operator.press(CMD_INIT)
        _steps(dep, dep.init_steps + 1)
    dep.operator.press(CMD_START)
    _steps(dep, 1)
    dep.frozen.clear()


def test_a_joint_that_never_reports_is_named_at_start(deploy, eden_caplog):
    """On hardware a joint can read bit-constant for a whole run while its command moves 1.5 rad."""
    dep = deploy(OperatorLoopbackDeployment, init_steps=5, connect=True)

    _frozen_joint_run(dep, freeze=True, ramp=True)

    [warning] = [r.getMessage() for r in eden_caplog.records if "bit-identical" in r.getMessage()]
    assert f"'{dep.dofs_name[0]} (commanded over 0.60 rad)'" in warning, warning
    assert dep.dofs_name[1] not in warning, "a joint that follows its command is not named"
    assert "--disable_joints does not help" in warning


@pytest.mark.parametrize(("freeze", "ramp"), [(False, True), (True, False)], ids=["tracking", "command-never-moved"])
def test_no_frozen_joint_warning_without_evidence(deploy, eden_caplog, freeze, ramp):
    dep = deploy(OperatorLoopbackDeployment, init_steps=5, connect=True)
    _frozen_joint_run(dep, freeze=freeze, ramp=ramp)
    assert "bit-identical" not in eden_caplog.text


def test_each_start_checks_only_its_own_window(deploy, eden_caplog):
    dep = deploy(OperatorLoopbackDeployment, init_steps=5, connect=True)
    _frozen_joint_run(dep, freeze=True, ramp=True)
    eden_caplog.clear()
    dep.operator.press(CMD_STOP)

    _frozen_joint_run(dep, freeze=False, ramp=True)

    assert "bit-identical" not in eden_caplog.text


def test_reset_boots_in_stop_with_the_clip_rewound_wherever_it_was_left(g1_env, deploy):
    """INIT with ``init_target="motion"`` reads frame 0 off the clip, so reset must leave it there."""
    motion = g1_env.command_manager.get_term("motion")
    dep = deploy()
    dep.request_start()
    _steps(dep, 2)
    assert int(motion.time_steps[0]) > 0

    obs = dep.reset()

    assert not dep.policy_active and int(motion.time_steps[0]) == 0
    assert set(obs) == {"policy"}
    _steps(dep, 2, np.full(dep.num_dofs, 9.0))
    assert int(motion.time_steps[0]) == 0, "the clip holds through STOP"
    assert not np.isclose(dep.sent[-1].dofs_pos, 9.0).any(), "the hold, not the policy"


def test_a_step_ticks_stateful_observations_once_and_a_reset_never(g1_env, deploy):
    """The deploy loop is phase 7 for observation terms; ``reset`` reads without advancing, as ``EnvBase.reset`` does."""
    integral = _policy_term(g1_env, VelocityTrackingErrorIntegral)
    command = g1_env.command_manager.get_command("velocity_command")[0].clone()
    dep = deploy()

    dep.reset()
    torch.testing.assert_close(integral.compute()[0], torch.zeros(3))

    for tick in range(1, 4):
        _steps(dep, 1)
        # The loopback robot stands still, so each tick adds the whole command times the control dt.
        torch.testing.assert_close(integral.compute()[0], tick * g1_env.dt * command[:3])


# ---------------------------------------------------------------------------
# The STOP hold latches its pose
# ---------------------------------------------------------------------------


def _booted_in_stop(dep, *readings: float):
    """An operator backend booted in STOP after reading ``readings[0]``; the robot then reads the rest, in turn."""
    dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=np.full(dep.num_dofs, pos)) for pos in readings)
    dep.state = dep.read_state()  # the read every backend's reset ends with
    return dep


def test_the_hold_latches_once_and_ignores_the_robot_afterwards(deploy_two_link):
    """The whole bug: the hold rebuilt its target from the last reading, so a sagging joint dragged it down.

    That made the PD a damper of ``kd + kp*control_dt`` (8 N.m.s/rad on a G1 leg against a ``kp`` of
    99), and the G1 folded to the floor in 2.5 s under STOP, the operator's abort.
    """
    dep = _booted_in_stop(deploy_two_link(OperatorLoopbackDeployment, connect=True), 0.2, 0.7, 1.2, 2.2)

    _steps(dep, 4, np.full(dep.num_dofs, 9.0))

    for sent in dep.sent:
        np.testing.assert_allclose(sent.dofs_pos, 0.2, err_msg="the pose the hold began at, not the sag")
        np.testing.assert_allclose(sent.dofs_kp, dep.default_dof_kp)
        np.testing.assert_allclose(sent.dofs_kd, dep.default_dof_kd)


def test_stop_latches_where_the_policy_left_the_robot_and_keeps_it(deploy_two_link):
    """The sequence an operator drives: START, the robot moves, STOP catches it there and arrests the drift."""
    dep = _booted_in_stop(deploy_two_link(OperatorLoopbackDeployment, init_steps=1, connect=True), 0.0)
    dep.operator.press(CMD_START)
    _steps(dep, 3, np.full(dep.num_dofs, 0.9))  # the loopback robot tracks it: it now reads 0.9
    dep.operator.press(CMD_STOP)
    _steps(dep, 1)
    dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=np.full(dep.num_dofs, pos)) for pos in (1.1, 1.4))

    _steps(dep, 2)

    for sent in dep.sent[-3:]:
        np.testing.assert_allclose(sent.dofs_pos, 0.9)


def test_a_second_stop_relatches_where_the_robot_then_is(deploy_two_link):
    dep = _booted_in_stop(deploy_two_link(OperatorLoopbackDeployment, init_steps=1, connect=True), 0.1)
    _steps(dep, 1)
    dep.operator.press(CMD_START)
    _steps(dep, 2, np.full(dep.num_dofs, -0.7))
    dep.operator.press(CMD_STOP)

    _steps(dep, 1)

    np.testing.assert_allclose(dep.sent[0].dofs_pos, 0.1)
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, -0.7, err_msg="not the first hold's pose")


def test_start_again_under_the_policy_holds_where_the_robot_is_not_at_the_boot_hold(deploy_two_link):
    """An operator restarting the clip: the START tick holds, and a latch from before the first START is 0.7 rad off.

    Sent at the full default gains for a control period, that latch was a torque spike mid-motion.
    """
    dep = _booted_in_stop(deploy_two_link(OperatorLoopbackDeployment, init_steps=1, connect=True), 0.1)
    _steps(dep, 1)  # the boot hold latches 0.1
    dep.operator.press(CMD_START)
    _steps(dep, 3, np.full(dep.num_dofs, 0.8))  # the START tick, then the policy: the robot now reads 0.8
    dep.operator.press(CMD_START)

    _steps(dep, 1, np.full(dep.num_dofs, 0.5))  # computed before this START, so never sent
    _steps(dep, 1, np.full(dep.num_dofs, 0.6))

    assert dep.consume_start_events() == 2
    np.testing.assert_allclose(dep.sent[-2].dofs_pos, 0.8, err_msg="where the robot was, not the boot hold's 0.1")
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, 0.6, err_msg="the policy's next command")


def test_start_after_the_end_of_clip_stand_holds_the_stance_not_the_boot_hold(deploy):
    dep = _booted_in_stop(deploy(on_motion_end="stand", stand_steps=2), 0.2)
    start_policy(dep)  # the START tick holds the boot latch at 0.2
    _steps(dep, 7, np.full(dep.num_dofs, 0.7))  # the clip ends and the stand ramps to the default stance

    dep.request_start()
    _steps(dep, 1, np.full(dep.num_dofs, 0.7))

    np.testing.assert_allclose(dep.sent[-1].dofs_pos, dep.default_dof_pos, err_msg="the stance, not 0.2")


def test_a_backend_editing_the_payload_in_place_does_not_edit_the_latch(deploy_two_link):
    """``send_payload`` rewrites ``dofs_pos`` in place on some backends (the Oli's torque clamp)."""

    class InPlaceScaling(OperatorLoopbackDeployment):
        def send_payload(self, payload) -> None:
            payload.dofs_pos *= 0.5
            super().send_payload(payload)

    dep = _booted_in_stop(deploy_two_link(InPlaceScaling, connect=True), 0.4)

    _steps(dep, 3)

    for sent in dep.sent:
        np.testing.assert_allclose(sent.dofs_pos, 0.2, err_msg="halved once per tick, not compounding")


# ---------------------------------------------------------------------------
# Each episode's first observation comes from the robot, not the simulated spawn pose
# ---------------------------------------------------------------------------

#: What ``dofs_pos`` reads off the simulated two-link arm at its spawn pose, and off the real one.
SPAWN, ROBOT = 0.0, 0.5


@OBSERVATION_TERM_REGISTRY.register()
class ControlTicks(ObservationTerm):
    """A stateful term: a counter advanced in the phase-7 hook and zeroed by ``reset``."""

    def build(self) -> None:
        self._ticks = torch.zeros(self.num_envs, 1, device=self.device)

    def post_physics_refresh(self) -> None:
        self._ticks.add_(1.0)

    def reset(self, envs_idx=None) -> None:
        self._ticks[slice(None) if envs_idx is None else envs_idx] = 0.0

    def compute(self) -> torch.Tensor:
        return self._ticks


@OBSERVATION_TERM_REGISTRY.register()
class PoseAtReset(ObservationTerm):
    """Zeroed by ``reset`` and rebuilt from the current frame by ``post_reset``, as ``ParkourProprio``'s history is."""

    def build(self) -> None:
        self._pose = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, envs_idx=None) -> None:
        self._pose[slice(None) if envs_idx is None else envs_idx] = 0.0

    def post_reset(self, envs_idx) -> None:
        self._pose[envs_idx] = self._env.entities["robot"].get_dofs_pos()[envs_idx]

    def compute(self) -> torch.Tensor:
        return self._pose


def _dofs_pos(**options):
    return observations.DofsPos.configure(entity_name="robot", **options)


@pytest.fixture(scope="module")
def stateful_obs_env(two_link_asset):
    """The two-link arm with one group per kind of per-step observation state an episode starts afresh.

    Built and reset as the training wrapper leaves it: every window, average and delay line holds
    the frame computed off the simulated arm at ``SPAWN``. That frame is the contamination under test.
    """
    return robot_env(
        two_link_robot(two_link_asset),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(dofs_pos=_dofs_pos()),
            history=ObservationGroupOptions(dofs_pos=_dofs_pos(history_length=5, flatten_history_dim=True)),
            ema=ObservationGroupOptions(dofs_pos=_dofs_pos(modifiers=[ExponentialMovingAverage.configure(alpha=0.2)])),
            delay=ObservationGroupOptions(dofs_pos=_dofs_pos(modifiers=[ObservationDelay.configure(control_steps=3)])),
            ticks=ObservationGroupOptions(ticks=ControlTicks.configure()),
            at_reset=ObservationGroupOptions(pose=PoseAtReset.configure()),
            last_action=ObservationGroupOptions(action=ObsTerm.configure(func=observations.last_action)),
        ),
    )


@pytest.fixture
def deploy_stateful(stateful_obs_env, monkeypatch):
    """``deploy_stateful(**options)``: a loopback deployment against :func:`stateful_obs_env` whose arm reads ``ROBOT``."""
    # The operator presses Enter once the INIT loop has ticked.
    monkeypatch.setattr("builtins.input", lambda prompt="": time.sleep(0.02))
    with deployments(stateful_obs_env) as make:

        def deploy(**options) -> LoopbackDeployment:
            dep = make(**options)
            dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=[ROBOT] * dep.num_dofs) for _ in range(16))
            return dep

        yield deploy


def _tick(dep) -> dict:
    """One control tick holding the default pose; returns the observation it read."""
    dep.write_dofs_targets(torch.as_tensor(dep.default_dof_pos, dtype=torch.float32), dep.dofs_name)
    return dep.step(None)


def _delivered(obs, group: str) -> list[float]:
    return [round(float(value), 6) for value in obs[group][0]]


def test_reset_delivers_every_window_modifier_and_term_from_the_robot(deploy_stateful):
    """Uncleared, the history window was 4/5 spawn frames, the moving average 80% one, the delay line all of it."""
    obs = deploy_stateful().reset()

    assert _delivered(obs, "policy") == [ROBOT] * 2
    assert _delivered(obs, "history") == [ROBOT] * 10
    assert _delivered(obs, "ema") == [ROBOT] * 2
    assert _delivered(obs, "delay") == [ROBOT] * 2
    assert _delivered(obs, "ticks") == [0.0]
    assert _delivered(obs, "at_reset") == [ROBOT] * 2  # reset() alone would leave it zero: post_reset rebuilds it


def test_no_spawn_frame_surfaces_on_the_ticks_after_reset(deploy_stateful):
    """The delay is the worse half: uncleared, ``control_steps=3`` hands the policy the spawn frame whole for three ticks."""
    dep = deploy_stateful()
    dep.reset()

    for tick in range(1, 5):
        obs = _tick(dep)
        assert _delivered(obs, "ema") == [ROBOT] * 2
        assert _delivered(obs, "delay") == [ROBOT] * 2
        assert _delivered(obs, "ticks") == [float(tick)]  # once per episode, not once per tick


def test_a_loop_with_no_reset_starts_afresh_on_its_first_tick(deploy_stateful):
    """Driving ``step`` by hand is supported; its first read is the episode's first, as an env reset's is."""
    obs = _tick(deploy_stateful())

    assert _delivered(obs, "history") == [ROBOT] * 10
    assert _delivered(obs, "delay") == [ROBOT] * 2
    assert _delivered(obs, "ticks") == [0.0]  # this tick's phase 7, then the reset: an env reset's first frame


def test_a_second_reset_starts_afresh_again(deploy_stateful):
    """A re-run, or a MuJoCo rollout's next episode, refills from the robot as it is *then*."""
    dep = deploy_stateful()
    dep.reset()
    for _ in range(3):
        _tick(dep)
    dep.readings.clear()
    dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=[0.9] * dep.num_dofs) for _ in range(4))

    obs = dep.reset()

    assert _delivered(obs, "history") == [0.9] * 10
    assert _delivered(obs, "ticks") == [0.0]


def test_a_read_that_starts_no_episode_clears_nothing(deploy_stateful):
    """A mid-run read (Wuji's ``observe()``) is a read: it neither ticks nor restarts the episode."""
    dep = deploy_stateful()
    dep.reset()
    for _ in range(3):
        _tick(dep)

    assert _delivered(dep.state_to_observation(dep.state), "ticks") == [3.0]


def _operator_run(deploy_stateful, *, before_start: float, after_start: float):
    """An operator backend reset, held in STOP for 4 ticks reading ``before_start``, then reading ``after_start``."""
    dep = deploy_stateful(cls=OperatorLoopbackDeployment, init_steps=1, connect=True)
    dep.readings.clear()
    dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=[before_start] * dep.num_dofs) for _ in range(5))
    dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=[after_start] * dep.num_dofs) for _ in range(8))
    dep.reset()
    for _ in range(4):
        _tick(dep)
    return dep


def test_start_refills_every_window_from_the_first_post_start_frame(deploy_stateful):
    """START re-captures the yaw offset, so every frame from STOP/INIT is in the old world frame.

    Uncleared, the START tick hands the policy a history window 4/5 STOP frames, a moving average 80%
    one and a delay line that is all STOP frame, and a term state that counted the hold.
    """
    dep = _operator_run(deploy_stateful, before_start=ROBOT, after_start=0.9)

    dep.operator.press(CMD_START)
    obs = _tick(dep)

    assert _delivered(obs, "history") == [0.9] * 10
    assert _delivered(obs, "ema") == [0.9] * 2
    assert _delivered(obs, "delay") == [0.9] * 2
    assert _delivered(obs, "ticks") == [0.0]
    assert _delivered(obs, "at_reset") == [0.9] * 2


def _fed_back(env, value: float) -> None:
    """The policy's raw action, as the action manager holds it after the policy last acted."""
    manager = env.action_manager
    manager.compute(torch.full((1, manager.total_action_dim), value, device=env.device))


def test_start_zeroes_the_last_action_the_policy_fed_back(deploy_stateful, stateful_obs_env):
    """A hold tick runs no action manager, so the action from before STOP would otherwise open the next episode."""
    dep = _operator_run(deploy_stateful, before_start=ROBOT, after_start=ROBOT)
    _fed_back(stateful_obs_env, 0.3)
    assert _delivered(_tick(dep), "last_action") == [0.3] * 2, "a hold tick clears nothing"

    dep.operator.press(CMD_START)

    assert _delivered(_tick(dep), "last_action") == [0.0] * 2


def test_restart_observations_starts_every_window_afresh_from_the_last_reading(deploy_stateful, stateful_obs_env):
    """What a policy switch hands the slot switched in: an env reset's first frame, not another slot's."""
    dep = deploy_stateful()
    dep.reset()
    for _ in range(2):
        _tick(dep)
    dep.readings.clear()
    dep.readings.extend(robot_state(dep.num_dofs, dofs_pos=[0.9] * dep.num_dofs) for _ in range(4))
    obs = _tick(dep)
    assert _delivered(obs, "history") == [ROBOT] * 8 + [0.9] * 2, "the window still holds the frames before"
    _fed_back(stateful_obs_env, 0.3)

    obs = dep.restart_observations()

    assert _delivered(obs, "history") == [0.9] * 10
    assert _delivered(obs, "ticks") == [0.0]
    assert _delivered(obs, "at_reset") == [0.9] * 2
    assert _delivered(obs, "last_action") == [0.0] * 2
    assert _delivered(_tick(dep), "ticks") == [1.0], "the next tick continues the restarted episode"


def test_stop_and_init_clear_nothing(deploy_stateful):
    """Only the handover to the policy starts afresh: STOP and INIT are frames the policy is not acting on."""
    dep = _operator_run(deploy_stateful, before_start=ROBOT, after_start=ROBOT)
    dep.operator.press(CMD_START)
    _tick(dep)
    _tick(dep)

    dep.operator.press(CMD_STOP)
    _tick(dep)
    dep.operator.press(CMD_INIT)

    assert _delivered(_tick(dep), "ticks") == [3.0]  # 0 on the START tick, then 1 per tick through STOP and INIT


# ---------------------------------------------------------------------------
# The motion reference in the deploy loop
# ---------------------------------------------------------------------------


def test_each_step_advances_both_clips_then_publishes_the_new_frame(g1_env, deploy):
    """The command clip must not freeze on a deploy, and the command must publish the frame the step advanced to."""
    motion = g1_env.command_manager.get_term("motion")
    dep = deploy()
    start_policy(dep)  # the operator hands control to the policy

    published = []
    for _ in range(4):
        _steps(dep, 1)
        published.append(float(g1_env.command_manager.get_command("motion")[0, 0]))

    assert published == pytest.approx([COMMAND_CLIP_STEP * k for k in (1, 2, 2, 2)]), "held at the last frame"
    assert int(motion.time_steps[0]) == COMMAND_CLIP_FRAMES - 1
    assert float(g1_env.motion_manager.get_motion_phase("robot").max()) > 0.0


def test_a_held_policy_leaves_the_clip_at_frame_zero(g1_env, deploy):
    dep = deploy(OperatorLoopbackDeployment, connect=True)
    _steps(dep, 3)
    assert int(g1_env.command_manager.get_term("motion").time_steps[0]) == 0
    assert float(g1_env.motion_manager.get_motion_phase("robot").max()) == 0.0


def test_a_rewind_restarts_both_clips_and_disarms_the_resampling_timer(g1_env, deploy):
    """Clock only: a resample would teleport and perturb a robot already standing where it is."""
    motion = g1_env.command_manager.get_term("motion")
    dep = deploy()
    _steps(dep, 3)
    motion.time_left.zero_()  # the training timer has run out

    dep.reset_motion_reference()

    assert int(motion.time_steps[0]) == 0
    assert float(g1_env.motion_manager.get_motion_phase("robot").max()) == 0.0
    assert torch.isinf(motion.time_left).all()


def test_motion_finished_waits_for_every_clip(g1_env, deploy):
    """Reading only the manager left ``on_motion_end`` inert on every command-path deploy (SONIC, beyondmimic_moves)."""
    dep = deploy()
    start_policy(dep)  # the operator hands control to the policy
    finished = []
    for _ in range(5):
        _steps(dep, 1)
        finished.append(dep.motion_finished())

    # The 3-frame command clip ends after 2 steps; the manager's clip after 4.
    assert finished == [False, False, False, True, True]


def test_a_command_only_task_finishes_and_rewinds_its_clip(g1_command_env, deploy_command_only):
    """No motion manager: the path SONIC and beyondmimic_moves deploy with.

    Reading only the manager left ``motion_finished`` false for ever there, and an early return on the
    missing manager skipped the command clip's rewind.
    """
    motion = g1_command_env.command_manager.get_term("motion")
    dep = deploy_command_only()
    start_policy(dep)  # the operator hands control to the policy
    finished = []
    for _ in range(3):
        _steps(dep, 1)
        finished.append(dep.motion_finished())
    assert finished == [False, True, True]

    dep.reset_motion_reference()

    assert int(motion.time_steps[0]) == 0 and not dep.motion_finished()


def test_motion_finished_is_false_with_no_clip_at_all(deploy_two_link):
    dep = deploy_two_link()
    _steps(dep, 3)
    assert not dep.motion_finished()


def test_on_motion_end_stand_ramps_to_the_default_stance_and_holds(deploy):
    dep = deploy(on_motion_end="stand", stand_steps=2)
    start_policy(dep)  # the operator hands control to the policy
    policy = np.full(dep.num_dofs, 0.7)

    _steps(dep, 7, policy)

    # Policy until the end fires (after the 4th step), then from the measured 0.7 to the default in 2 ticks.
    sent = [s.dofs_pos for s in dep.sent]
    for k in range(4):
        np.testing.assert_allclose(sent[k], policy)
    np.testing.assert_allclose(sent[4], policy + 0.5 * (dep.default_dof_pos - policy))
    np.testing.assert_allclose(sent[5], dep.default_dof_pos)
    np.testing.assert_allclose(sent[6], dep.default_dof_pos)

    dep.reset_motion_reference()  # a START or a slot switch re-arms the clip
    _steps(dep, 1, policy)
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, policy)


def test_on_motion_end_switch_queues_one_handover(deploy):
    dep = deploy(on_motion_end="switch", motion_end_slot=2)
    start_policy(dep)  # the operator hands control to the policy

    _steps(dep, 5)
    assert dep.consume_policy_switches() == [1], "1-based for the operator, 0-based for the loop"
    _steps(dep, 3)
    assert dep.consume_policy_switches() == []


def test_a_plain_rl_env_steps_with_neither_motion_nor_commands(deploy_two_link):
    dep = deploy_two_link()
    dep.write_dofs_targets(torch.tensor([0.1, -0.1]), ["j1", "j2"])
    assert set(dep.step(None)) == {"policy"}


def test_only_the_base_drives_the_command_manager():
    """``_update_commands`` is the one call site: a copy per backend is how one frozen clip reached all three."""
    package = Path(deployment_base.__file__).parent
    markers = ("command_manager.compute(", "command_manager.draw_vis(", "_advance_motion_reference()")
    offenders = {
        path.name: [marker for marker in markers if marker in path.read_text()]
        for path in sorted(package.glob("*.py"))
        if path.name != "base.py"
    }
    assert {name: found for name, found in offenders.items() if found} == {}
