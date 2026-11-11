"""``eden.extensions.policies.sonic.plant``: the bodies the SONIC teleop loop drives, on real built envs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import eden as en  # noqa: E402
from eden.extensions.deployment.mujoco import MujocoDeployment  # noqa: E402
from eden.extensions.deployment.utils.operator_input import CMD_INIT, CMD_NEXT_POLICY, CMD_START, CMD_STOP  # noqa: E402
from eden.extensions.policies.sonic.joint_maps import DEFAULT_ANGLES_MJ  # noqa: E402
from eden.extensions.policies.sonic.plant import (  # noqa: E402
    NUM_BODY_DOFS,
    NUM_HAND_DOFS,
    DeploymentPlant,
    GantryBand,
    GenesisPlant,
    SonicTeleopEnv,
)
from eden.extensions.policies.sonic.policy import SonicPolicy  # noqa: E402
from eden.extensions.policies.sonic.teleop_policy import SonicTeleopPolicy  # noqa: E402
from eden.options import ActionManagerOptions, EnvOptions, ObservationManagerOptions  # noqa: E402
from tests.support.deploy import OperatorLoopbackDeployment, deployments, robot_state  # noqa: E402

pytestmark = pytest.mark.requires_genesis

STAND = np.asarray(DEFAULT_ANGLES_MJ, dtype=np.float64)


def _build(spec) -> SonicTeleopEnv:
    """The single-env scene ``eden teleop sonic robot`` builds from a policy spec."""
    env = SonicTeleopEnv(
        env_options=EnvOptions(num_envs=1, episode_length_s=1e9, **spec.env_options()),
        scene_options=en.scenes.Empty(robot=spec.robot_options()),
        action_options=ActionManagerOptions(**spec.action_options()),
        observation_options=ObservationManagerOptions(**spec.observation_options()),
        eval_mode=True,
    )
    env.build()
    env.reset()
    return env


@pytest.fixture(scope="module")
def body_env():
    """The body-only G1 a deployment plant drives: ``eden deploy --policy sonic``'s robot."""
    return _build(SonicPolicy)


@pytest.fixture
def deploy(body_env):
    with deployments(body_env) as make:
        yield make


def _action(q_body: np.ndarray) -> dict[str, torch.Tensor]:
    return {"dofs_pos_controller": torch.as_tensor(q_body, dtype=torch.float32).reshape(1, -1)}


# ---------------------------------------------------------------------------
# Deployment plant over an in-memory wire (the hardware shape)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [({"sync": True}, "sync=False"), ({"auto_yaw_align": True}, "auto_yaw_align=False")],
)
def test_a_deployment_the_loop_cannot_drive_is_refused(deploy, options, message):
    dep = deploy(OperatorLoopbackDeployment, **{"auto_yaw_align": False, **options})
    with pytest.raises(ValueError, match=message):
        DeploymentPlant(dep, is_hardware=True)


def test_a_hardware_reading_is_the_wire_state_with_no_base_position_or_hands(deploy):
    dep = deploy(OperatorLoopbackDeployment, connect=True, auto_yaw_align=False)
    plant = DeploymentPlant(dep, is_hardware=True)
    pose = np.linspace(-0.5, 0.5, NUM_BODY_DOFS)
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    dep.readings.append(robot_state(NUM_BODY_DOFS, dofs_pos=pose, base_quat=quat, base_ang_vel=np.array([1.0, 2, 3])))

    reading = plant.read()

    assert plant.is_hardware and not plant.has_hands and plant.band is None
    np.testing.assert_array_equal(reading.q_body, pose)
    np.testing.assert_array_equal(reading.base_quat, quat)
    np.testing.assert_array_equal(reading.base_ang, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(reading.base_pos, np.zeros(3))
    np.testing.assert_array_equal(reading.q_hands, np.zeros(NUM_HAND_DOFS))
    snapshot = reading.snapshot("cpu")
    assert snapshot.q_mj.shape == (1, NUM_BODY_DOFS) and snapshot.q_mj.dtype == torch.float32
    assert snapshot.pelvis_quat_wxyz.shape == (1, 4)


def test_targets_reach_the_robot_only_between_start_and_stop(deploy):
    dep = deploy(OperatorLoopbackDeployment, connect=True, auto_yaw_align=False, init_steps=2)
    plant = DeploymentPlant(dep, is_hardware=True)
    target = STAND + 0.1

    plant.step(_action(target))
    assert not plant.policy_active and plant.consume_start_events() == 0
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, dep.default_dof_pos)  # STOP holds the measured pose

    dep.operator.press(CMD_INIT)
    plant.step(_action(target))
    dep.operator.press(CMD_START)
    plant.step(_action(target))
    assert plant.policy_active and plant.consume_start_events() == 1 and plant.consume_start_events() == 0
    # The START tick repeats the INIT target: a target computed before START is never sent.
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, dep.default_dof_pos, atol=1e-6)
    plant.step(_action(target))
    np.testing.assert_allclose(dep.sent[-1].dofs_pos, target, atol=1e-6)

    dep.operator.press(CMD_STOP)
    plant.step(_action(STAND))
    assert not plant.policy_active


def test_a_policy_switch_request_is_drained_not_left_to_pile_up(deploy, capsys):
    dep = deploy(OperatorLoopbackDeployment, connect=True, auto_yaw_align=False)
    plant = DeploymentPlant(dep, is_hardware=True)
    dep.operator.press(CMD_NEXT_POLICY)

    plant.step(_action(STAND))

    assert "policy switch request 'next' ignored" in capsys.readouterr().out
    assert dep.consume_policy_switches() == []


# ---------------------------------------------------------------------------
# Deployment plant over MuJoCo (sim-to-sim)
# ---------------------------------------------------------------------------


def _mujoco_plant(deploy, **options) -> tuple[DeploymentPlant, MujocoDeployment]:
    dep = deploy(MujocoDeployment, connect=True, auto_yaw_align=False, gantry=True, **options)
    plant = DeploymentPlant(dep, is_hardware=False, band=GantryBand(dep))
    plant.reset()
    return plant, dep


def test_the_mujoco_body_reports_its_true_base_and_can_be_reset(deploy):
    plant, dep = _mujoco_plant(deploy, init_steps=2)

    reading = plant.read()

    assert not plant.is_hardware
    assert reading.base_pos[2] == pytest.approx(dep.robot_options.default_root_pos[2], abs=0.02)
    np.testing.assert_allclose(reading.q_body, STAND, atol=1e-3)
    np.testing.assert_allclose(reading.base_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-3)


def test_a_headless_mujoco_body_starts_itself_and_holds_the_stand_on_the_gantry(deploy):
    plant, dep = _mujoco_plant(deploy, init_steps=2)

    for _ in range(50):
        plant.step(_action(STAND))

    reading = plant.read()
    assert plant.policy_active, "no TTY under pytest: the backend ran INIT, then START"
    assert plant.consume_start_events() == 1
    upright = 1.0 - 2.0 * (reading.base_quat[1] ** 2 + reading.base_quat[2] ** 2)
    assert upright > 0.99 and reading.base_pos[2] > 0.7


def test_the_band_key_releases_and_re_rigs_the_gantry_between_physics_steps(deploy):
    plant, dep = _mujoco_plant(deploy, init_steps=2)
    band = plant.band
    assert band.enable and dep._sim.gantry_anchor_height is not None

    assert band.handle_key("9") and not band.handle_key("7")
    assert dep._sim.gantry_anchor_height is not None, "a key press alone must not touch the sim (keyboard thread)"
    plant.step(_action(STAND))
    assert not dep.gantry and dep._sim.gantry_anchor_height is None

    band.enable = True
    plant.reset()
    assert dep.gantry and dep._sim.gantry_anchor_height is not None


# ---------------------------------------------------------------------------
# Genesis plant
# ---------------------------------------------------------------------------


def test_the_genesis_body_reads_body_and_hands_and_steps_under_the_band():
    plant = GenesisPlant(_build(SonicTeleopPolicy))
    plant.reset()

    reading = plant.read()
    action = _action(STAND)
    action["hands_pos_controller"] = torch.zeros(1, NUM_HAND_DOFS)
    plant.step(action)  # the band is on: its wrench goes through Genesis' external-wrench API

    assert plant.has_hands and not plant.is_hardware and plant.band.enable
    assert reading.q_hands.shape == (NUM_HAND_DOFS,)
    np.testing.assert_allclose(reading.q_body, STAND, atol=1e-5)
    np.testing.assert_allclose(plant.read().q_body, STAND, atol=0.05)
