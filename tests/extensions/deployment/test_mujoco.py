"""``MujocoDeployment``: the sim-to-sim backend, driving real MuJoCo physics from a real env."""

from __future__ import annotations

import re
import time

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from eden.extensions.deployment.mujoco import MujocoDeployment  # noqa: E402
from eden.extensions.deployment.utils.state import RobotCommand  # noqa: E402
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME  # noqa: E402
from tests.extensions.deployment.conftest import (  # noqa: E402
    MANAGER_CLIP_BASE_POS,
    MANAGER_CLIP_JOINT_STEP,
    TWO_LINK_MJCF,
    two_link_robot,
)
from tests.support.deploy import deployments, robot_env  # noqa: E402


@pytest.fixture(scope="module")
def two_link_env(two_link_asset):
    """The two-link arm, with an ``ActionDelay`` of 15 ms on its one action term.

    ``test_a_task_action_delay_never_reaches_the_mujoco_robot`` is the test it serves.
    """
    from eden.managers.modifiers.actions.actuators import ActionDelay

    return robot_env(two_link_robot(two_link_asset), modifier=ActionDelay.configure(delay_range_s=(0.015, 0.015)))


@pytest.fixture
def unreachable_hub(monkeypatch, tmp_path):
    """No Hugging Face Hub: the one function through which Eden downloads raises.

    Menagerie models also resolve against an empty cache: they share the asset directory with Eden's own, so a copy any
    earlier test downloaded would otherwise answer without the Hub.
    """
    from eden.extensions.deployment.utils.menagerie import MENAGERIE_DATASET
    from eden.utils.assets import get_asset_path

    def offline(**kwargs):
        raise RuntimeError(f"hub unreachable: {kwargs['repo_id']}")

    def uncached_menagerie(file, *, registry, dataset, local_dir=None):
        if dataset == MENAGERIE_DATASET:
            local_dir = str(tmp_path / "menagerie")
        return get_asset_path(file, registry=registry, dataset=dataset, local_dir=local_dir)

    monkeypatch.setattr("eden.utils.assets._snapshot_download_locked", offline)
    monkeypatch.setattr("eden.extensions.deployment.mujoco.get_asset_path", uncached_menagerie)


def _connected(make, **options):
    return make(MujocoDeployment, connect=True, **options)


# ---------------------------------------------------------------------------
# Which model MuJoCo loads
# ---------------------------------------------------------------------------


def test_the_default_model_is_the_training_asset_and_needs_no_hub(g1_env, deploy, unreachable_hub):
    """``eden deploy --mujoco`` sim-to-sims against the asset the policy trained on, whatever the network does."""
    dep = deploy(MujocoDeployment)
    assert dep._resolve_asset().endswith(g1_env.config.scene_options.robot.file)


def test_auto_falls_back_to_the_training_asset_when_menagerie_is_unreachable(
    g1_env, deploy, unreachable_hub, eden_caplog
):
    dep = deploy(MujocoDeployment, mjcf_source="auto")
    assert dep._resolve_asset().endswith(g1_env.config.scene_options.robot.file)
    assert "Menagerie model 'unitree_g1/g1.xml' unavailable" in eden_caplog.text


def test_menagerie_raises_when_its_model_is_unreachable(deploy, unreachable_hub):
    with pytest.raises(RuntimeError, match="hub unreachable"):
        deploy(MujocoDeployment, mjcf_source="menagerie")._resolve_asset()


@pytest.mark.parametrize(
    ("options", "outcome"),
    [
        ({"mjcf_source": "menagerie"}, "No Menagerie mapping"),
        ({"mjcf_source": "bogus"}, "mjcf_source must be"),
        ({"mjcf_source": "menagerie", "mjcf_path": "explicit"}, None),
    ],
    ids=["menagerie-unmapped", "unknown-source", "explicit-path-wins"],
)
def test_model_resolution_on_an_unmapped_robot(deploy_two_link, two_link_asset, options, outcome):
    if options.get("mjcf_path") == "explicit":
        options = {**options, "mjcf_path": str(two_link_asset)}
    dep = deploy_two_link(MujocoDeployment, **options)
    if outcome is None:
        assert dep._resolve_asset() == str(two_link_asset)
    else:
        with pytest.raises(ValueError, match=outcome):
            dep._resolve_asset()


# ---------------------------------------------------------------------------
# What the simulated robot reports
# ---------------------------------------------------------------------------


def test_a_floating_base_reports_its_ground_truth_pose_and_velocity(deploy):
    """Dropping ``extra`` fed every ``base_lin_vel`` policy a constant zero in sim-to-sim."""
    dep = _connected(deploy)
    dep.reset()
    truth = dep._sim.read()

    state = dep.read_state()

    np.testing.assert_allclose(state.extra["base_pos"], truth["base_pos"])
    np.testing.assert_allclose(state.extra["base_lin_vel"], truth["base_lin_vel"])


@pytest.mark.parametrize("env_name", ["g1", "two_link"], ids=["opted-out", "fixed-base"])
def test_no_ground_truth_base_when_opted_out_or_fixed(deploy, deploy_two_link, env_name):
    make = deploy if env_name == "g1" else deploy_two_link
    dep = _connected(make, provide_ground_truth_base=env_name != "g1")
    assert "base_lin_vel" not in dep.read_state().extra
    assert "base_pos" not in dep.read_state().extra


@pytest.mark.parametrize(
    ("env_name", "options", "provides"),
    [
        ("g1", {}, True),
        ("g1", {"provide_ground_truth_base": False}, False),
        ("g1", {"provide_ground_truth_base": False, "estimate_lin_vel": True}, True),
        ("two_link", {}, False),
    ],
    ids=["ground-truth", "blinded", "estimator", "fixed-base"],
)
def test_the_base_velocity_comes_from_ground_truth_or_the_estimator(
    deploy, deploy_two_link, env_name, options, provides
):
    """What ``eden deploy`` checks before letting a ``base_lin_vel`` policy run."""
    make = deploy if env_name == "g1" else deploy_two_link
    assert make(MujocoDeployment, **options).provides_base_lin_vel is provides


def test_the_imu_is_exact_by_default_and_noisy_on_request(g1_env):
    """An exact accelerometer hides how an IMU-driven estimate drifts on hardware."""
    bias_acc, bias_gyro = np.array([0.1, -0.2, 0.3]), np.array([0.01, 0.0, -0.01])
    with deployments(g1_env) as make:
        exact = _connected(make)
        truth = exact._sim.read()
        state = exact.read_state()
        np.testing.assert_array_equal(state.base_lin_acc, truth["base_lin_acc"])
        np.testing.assert_array_equal(state.base_ang_vel, truth["base_ang_vel"])
    with deployments(g1_env) as make:
        noisy = _connected(
            make, imu_acc_noise=0.3, imu_gyro_noise=0.05, imu_acc_bias=tuple(bias_acc), imu_gyro_bias=tuple(bias_gyro)
        )
        readings = [noisy.read_state() for _ in range(4000)]
    acc = np.array([r.base_lin_acc for r in readings]) - truth["base_lin_acc"]
    gyro = np.array([r.base_ang_vel for r in readings]) - truth["base_ang_vel"]
    np.testing.assert_allclose(acc.mean(0), bias_acc, atol=0.03)
    np.testing.assert_allclose(acc.std(0), 0.3, rtol=0.1)
    np.testing.assert_allclose(gyro.mean(0), bias_gyro, atol=0.005)
    np.testing.assert_allclose(gyro.std(0), 0.05, rtol=0.1)


def test_imu_noise_repeats_under_a_seed(g1_env):
    draws = []
    for _ in range(2):
        with deployments(g1_env) as make:
            draws.append(_connected(make, imu_acc_noise=0.3, imu_noise_seed=7).read_state().base_lin_acc)
    np.testing.assert_array_equal(draws[0], draws[1])


def test_negative_imu_noise_is_rejected(deploy):
    with pytest.raises(ValueError, match="imu_acc_noise is a standard deviation"):
        deploy(MujocoDeployment, imu_acc_noise=-0.1)


# ---------------------------------------------------------------------------
# Lockstep and asynchronous physics
# ---------------------------------------------------------------------------


def _hold(dep, pos) -> RobotCommand:
    n = dep.num_dofs
    return RobotCommand(
        dofs_pos=np.asarray(pos, dtype=np.float64),
        dofs_vel=np.zeros(n),
        dofs_torque=np.zeros(n),
        dofs_kp=dep.default_dof_kp,
        dofs_kd=dep.default_dof_kd,
    )


def test_lockstep_physics_advances_only_when_commanded(deploy_two_link):
    dep = _connected(deploy_two_link)
    dep.reset()
    t_sim = dep._sim.data.time
    time.sleep(0.05)
    dep.read_state()
    assert dep._sim.data.time == t_sim, "no command, no physics"

    dep.send_payload(_hold(dep, dep.default_dof_pos))

    assert dep._sim.data.time == pytest.approx(t_sim + dep._n_substeps * dep._sim.sim_dt)


def test_mujoco_runs_the_timestep_given_and_refuses_one_off_the_control_period_in_lockstep(two_link_env):
    """``--mujoco_sim_dt`` reaches the model. Rounding a fraction of steps a tick would drift the physics clock off the
    policy's, so lockstep refuses a given one off the control period, naming one that fits; async keeps its own clock.
    """
    with deployments(two_link_env) as make:
        control_dt = make(MujocoDeployment).control_dt
    with deployments(two_link_env) as make:
        dep = _connected(make, sim_dt=control_dt / 4)
        assert dep._sim.model.opt.timestep == pytest.approx(control_dt / 4)
        assert dep._n_substeps == 4

    fit = re.escape(f"--mujoco_sim_dt {control_dt / 3:g}")
    with deployments(two_link_env) as make, pytest.raises(ValueError, match=f"\\(sim_dt\\) does not divide.*{fit}"):
        make(MujocoDeployment, sim_dt=control_dt / 2.5)
    with deployments(two_link_env) as make:
        assert _connected(make, sim_dt=control_dt / 2.5, async_physics=True)._sim.sim_dt == pytest.approx(
            control_dt / 2.5
        )


def test_an_assets_timestep_off_the_control_period_gives_way_in_lockstep_to_the_largest_under_it_that_fits(
    two_link_env, tmp_path, eden_caplog
):
    """The asset's own timestep is no one's mistake: lockstep runs the largest step under it that divides the control
    period, as a composite rig's default does, and says so; async keeps the asset's.
    """
    with deployments(two_link_env) as make:
        control_dt = make(MujocoDeployment).control_dt
    asset = tmp_path / "two_link_off.xml"
    asset.write_text(TWO_LINK_MJCF.replace("<worldbody>", f'<option timestep="{control_dt / 2.5!r}"/><worldbody>', 1))

    with deployments(two_link_env) as make:
        dep = _connected(make, mjcf_path=str(asset))
        assert dep._sim.model.opt.timestep == pytest.approx(control_dt / 3)
        assert dep._n_substeps == 3
    assert f"running at {control_dt / 3:g} s" in eden_caplog.text
    with deployments(two_link_env) as make:
        assert _connected(make, mjcf_path=str(asset), async_physics=True)._sim.sim_dt == pytest.approx(control_dt / 2.5)


def test_the_floor_stands_where_the_ground_plane_does_with_its_friction_and_mujocos_solver_parameters(
    two_link_asset,
):
    """The ground plane's pose and friction reach the floor; its solver parameters do not, as the robot's own geoms
    keep its asset's, and a contact mixing a Genesis value with a MuJoCo one would be neither engine's.
    """
    from eden.options.entities import PlaneOptions
    from eden.utils.configs import SceneOptions
    from tests.support.deploy import robot_env_config
    from tests.support.envs import build_env

    normal = (0.0, -np.sin(0.1), np.cos(0.1))
    plane = PlaneOptions(
        normal=normal,
        default_root_pos=(0.0, 0.0, 0.05),
        default_geoms_friction={"*": 0.6},
        default_geoms_sol_params={"*": (0.03, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)},
    )
    config = robot_env_config(two_link_robot(two_link_asset))
    config.scene_options = SceneOptions(robot=config.scene_options.robot, plane=plane)
    with deployments(build_env(config, reset=True)) as make:
        model = _connected(make)._sim.model
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "__eden_floor__")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[floor])
    np.testing.assert_allclose(model.geom_pos[floor], (0.0, 0.0, 0.05), atol=1e-6)
    np.testing.assert_allclose(rotation.reshape(3, 3)[:, 2], normal, atol=1e-6)
    assert model.geom_friction[floor, 0] == pytest.approx(0.6)
    assert tuple(model.geom_solref[floor]) == (0.02, 1.0), "MuJoCo's, not the ground's 0.03"


def test_on_a_terrain_a_single_robots_floor_is_flat_with_the_terrains_friction_alone(two_link_asset, eden_caplog):
    """A terrain's shape is no plane's, and its solver parameters stay out as a plane's do: friction alone, said aloud."""
    from eden.options.entities import TerrainOptions
    from eden.utils.configs import SceneOptions
    from tests.support.deploy import robot_env_config
    from tests.support.envs import build_env

    terrain = TerrainOptions(
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.25,
        default_geoms_friction={"*": 0.6},
        default_geoms_sol_params={"*": (0.03, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)},
    )
    config = robot_env_config(two_link_robot(two_link_asset))
    config.scene_options = SceneOptions(robot=config.scene_options.robot, terrain=terrain)
    with deployments(build_env(config, reset=True)) as make:
        model = _connected(make)._sim.model
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "__eden_floor__")
    np.testing.assert_allclose(model.geom_pos[floor], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(model.geom_quat[floor], (1.0, 0.0, 0.0, 0.0))
    assert model.geom_friction[floor, 0] == pytest.approx(0.6)
    assert tuple(model.geom_solref[floor]) == (0.02, 1.0), "MuJoCo's, not the terrain's 0.03"
    assert "[deploy] MuJoCo: the ground 'terrain' is a terrain" in eden_caplog.text


def test_the_floors_caveat_is_said_only_when_the_floor_is_in_the_model(two_link_asset, tmp_path, eden_caplog):
    """An asset with a plane of its own keeps that one, so the scene's grounds it stands in for go unsaid."""
    from eden.options.entities import PlaneOptions
    from eden.utils.configs import SceneOptions
    from tests.support.deploy import robot_env_config
    from tests.support.envs import build_env

    config = robot_env_config(two_link_robot(two_link_asset))
    config.scene_options = SceneOptions(
        robot=config.scene_options.robot,
        plane=PlaneOptions(),
        basement=PlaneOptions(default_root_pos=(0.0, 0.0, -1.0)),
    )
    env = build_env(config, reset=True)
    own_plane = tmp_path / "own_plane.xml"
    own_plane.write_text(
        TWO_LINK_MJCF.replace("<worldbody>", '<worldbody><geom name="ground" type="plane" size="0 0 0.05"/>', 1)
    )
    with deployments(env) as make:
        assert _connected(make, mjcf_path=str(own_plane))._sim.floor is None
    assert "grounds" not in eden_caplog.text
    with deployments(env) as make:
        assert _connected(make)._sim.floor is not None
    assert "the scene has 2 grounds" in eden_caplog.text


def test_policy_seconds_ends_a_single_robots_run_too(deploy_two_link):
    """``--mujoco_seconds`` is a base option: the single robot ends after the policy drove it that long."""
    dep = _connected(deploy_two_link, policy_seconds=0.1, init_steps=2)
    ticks = round(0.1 / dep.control_dt)
    dep.reset()
    policy = torch.tensor([0.3, -0.3])
    driven = 0
    with pytest.raises(KeyboardInterrupt):
        for _ in range(ticks + 20):  # headless: the backend presses INIT, holds its two cycles, then START
            dep.write_dofs_targets(policy, ["j1", "j2"])
            dep.step(None)
            driven = dep._policy_ticks
    assert driven == ticks


def test_the_rollout_is_recorded_from_start_on_in_lockstep_only(two_link_env, tmp_path):
    with deployments(two_link_env) as make, pytest.raises(ValueError, match="record_video.*async_physics"):
        make(MujocoDeployment, record_video=str(tmp_path / "async.mp4"), async_physics=True)

    video = tmp_path / "rollout.mp4"
    with deployments(two_link_env) as make:
        try:
            dep = _connected(make, record_video=str(video), init_steps=2)
        except RuntimeError as e:
            if "offscreen renderer" in str(e):
                pytest.skip(f"no OpenGL context for MuJoCo here: {e}")
            raise
        dep.reset()
        policy = torch.tensor([0.3, -0.3])
        for _ in range(8):  # headless: the backend presses INIT, holds its two cycles, then START
            dep.write_dofs_targets(policy, ["j1", "j2"])
            dep.step(None)
            if not dep.policy_active:
                assert dep._video.frames == 0, "nothing is recorded before START"
        assert dep._video.frames > 0
        dep.close()
    assert video.stat().st_size > 0


def test_target_lpf_low_passes_the_targets_the_mujoco_robot_is_driven_to(deploy_two_link):
    """Every hardware backend low-passes its targets on the wire; the MuJoCo one ignored ``--target_lpf``."""
    dep = _connected(deploy_two_link, target_lpf=0.5)
    dep.reset()
    start = np.asarray(dep.default_dof_pos, dtype=np.float64)
    step = start + 0.4
    first, second = _hold(dep, start), _hold(dep, step)
    dep.send_payload(first)
    dep.send_payload(second)

    np.testing.assert_allclose(second.dofs_pos, 0.5 * step + 0.5 * start, err_msg="what the PD is driven to")


@pytest.mark.parametrize("async_physics", [False, True], ids=["lockstep", "async"])
def test_a_slot_switchs_gains_drive_the_mujoco_robot_damping_included(deploy_two_link, async_physics):
    """After ``set_gains`` the MuJoCo PD holds at the new stiffness, and the kd it folds into the armature follows."""
    dep = _connected(deploy_two_link, async_physics=async_physics)
    dep.reset()
    sim = dep._sim
    asset_armature = sim.model.dof_armature[sim._v_idx] - dep.default_dof_kd * sim.sim_dt

    def gravity_droop() -> float:
        """How far the arm sags under a hold of its default pose: gravity torque / kp once settled."""
        for _ in range(150):
            dep.send_payload(_hold(dep, dep.default_dof_pos))
            if async_physics:
                time.sleep(dep.control_dt)
        return float(np.abs(dep.read_state().dofs_pos - dep.default_dof_pos).max())

    soft = gravity_droop()
    dep.set_gains(4 * dep.default_dof_kp, 4 * dep.default_dof_kd, ramp_steps=0)
    stiff = gravity_droop()

    assert soft > 1e-3, "the arm sags at its own gains, so a stiffer PD has something to show"
    assert stiff == pytest.approx(soft / 4, rel=0.25)
    np.testing.assert_allclose(sim.model.dof_armature[sim._v_idx], asset_armature + dep.default_dof_kd * sim.sim_dt)


@pytest.mark.parametrize("async_physics", [False, True], ids=["lockstep", "async"])
def test_a_commands_own_kd_leaves_the_armature_at_the_active_slots(deploy_two_link, async_physics):
    """The INIT ramp's ``init_dofs_kd_scale`` is no training robot's damping: only a slot's gains move the armature."""
    dep = _connected(deploy_two_link, async_physics=async_physics)
    dep.reset()
    sim = dep._sim
    armature = sim.model.dof_armature[sim._v_idx].copy()

    soft = _hold(dep, dep.default_dof_pos)
    soft.dofs_kd = 0.25 * dep.default_dof_kd
    dep.send_payload(soft)

    np.testing.assert_array_equal(sim.model.dof_armature[sim._v_idx], armature)


def test_async_physics_runs_on_wall_time_and_picks_up_each_command(deploy_two_link):
    dep = _connected(deploy_two_link, async_physics=True)
    dep.reset()
    first = dep.read_state()
    time.sleep(0.1)
    assert dep.read_state().stamp > first.stamp, "the robot's clock ran while the control thread slept"
    assert dep._physics.substeps >= 20  # 0.1 s at 500 Hz, with slack for a loaded host

    dep.send_payload(_hold(dep, [0.4, -0.4]))
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and not np.allclose(dep.read_state().dofs_pos, [0.4, -0.4], atol=0.1):
        time.sleep(0.02)

    np.testing.assert_allclose(dep.read_state().dofs_pos, [0.4, -0.4], atol=0.1)
    assert dep._physics.reanchors == 0
    dep.close()
    assert dep._physics is None


def test_an_async_reset_returns_to_the_default_pose_and_holds_it(deploy_two_link):
    dep = _connected(deploy_two_link, async_physics=True)
    dep.reset()
    dep.send_payload(_hold(dep, [0.5, 0.5]))
    time.sleep(0.15)
    assert np.abs(dep.read_state().dofs_pos).max() > 0.1

    assert set(dep.reset()) == {"policy"}

    # Back at the default pose, not at 0.5. The physics thread keeps stepping in wall time through the
    # rest of reset() (it rewinds the clip and starts the observation history afresh), so a loaded host
    # can land a tick or two of the hold's gravity droop before this read; exactly zero is not the claim.
    np.testing.assert_allclose(dep.read_state().dofs_pos, dep.default_dof_pos, atol=1e-2)
    time.sleep(0.1)
    # The hold is back in force: near the default pose (a plain PD's gravity droop), not resuming the 0.5 target.
    np.testing.assert_allclose(dep.read_state().dofs_pos, dep.default_dof_pos, atol=0.1)


def test_a_second_headless_reset_runs_init_and_start_again(deploy_two_link):
    """No TTY under pytest, so the backend presses INIT and START itself; a reset must re-arm that."""
    dep = _connected(deploy_two_link, init_steps=2)
    policy = torch.tensor([0.3, -0.3])
    for _ in range(2):
        dep.reset()
        before = dep._sim.data.time
        for _ in range(6):
            dep.write_dofs_targets(policy, ["j1", "j2"])
            dep.step(None)
        assert dep._sim.data.time > before, "every tick commanded the sim"
        assert dep._use_policy, "INIT ran its two cycles, then START"


_HISTORY_LENGTH = 5


@pytest.fixture(scope="module")
def two_link_history_env(two_link_asset):
    """The two-link arm with a five-frame ``dofs_pos`` history window: per-step state an episode starts afresh."""
    from eden import observations
    from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions

    dofs_pos = observations.DofsPos.configure(
        entity_name="robot", history_length=_HISTORY_LENGTH, flatten_history_dim=True
    )
    return robot_env(
        two_link_robot(two_link_asset),
        observation_options=ObservationManagerOptions(policy=ObservationGroupOptions(dofs_pos=dofs_pos)),
    )


def test_a_second_reset_starts_the_observation_history_afresh(two_link_history_env):
    """On the MuJoCo backend too: a ``reset`` replacing the base's would keep the last episode's window."""
    with deployments(two_link_history_env) as make:
        dep = _connected(make, init_steps=2)
        dep.reset()
        for _ in range(8):
            dep.write_dofs_targets(torch.tensor([0.4, -0.4]), ["j1", "j2"])
            dep.step(None)
        assert np.abs(dep.read_state().dofs_pos).max() > 0.1, "the arm left the reset pose"

        obs = dep.reset()

        reset_pose = dep.read_state().dofs_pos
    np.testing.assert_allclose(obs["policy"][0].numpy(), np.tile(reset_pose, _HISTORY_LENGTH), atol=1e-6)


def test_the_async_gantry_is_rigged_and_cranked_while_the_physics_runs(deploy_two_link):
    dep = _connected(deploy_two_link, async_physics=True, gantry=True, gantry_draw=False)
    dep.reset()
    before, steps_before = dep._sim.gantry_anchor_height, dep._physics.substeps
    assert before is not None

    dep._adjust_gantry(+1.0)

    assert dep._sim.gantry_anchor_height == pytest.approx(before + dep.gantry_step)
    time.sleep(0.1)
    assert dep._physics.substeps > steps_before and dep._physics.running
    assert dep._physics.reanchors == 0


def test_the_gantry_can_be_released_and_rigged_again_mid_run(deploy_two_link):
    dep = _connected(deploy_two_link, gantry=True, gantry_draw=False)
    dep.reset()
    assert dep._sim.gantry_anchor_height is not None

    dep.rig_gantry(False)
    assert not dep.gantry and dep._sim.gantry_anchor_height is None

    dep.rig_gantry(True)
    dep.reset()  # a reset keeps whichever way the gantry was last left
    assert dep.gantry and dep._sim.gantry_anchor_height is not None


def test_a_task_action_delay_never_reaches_the_mujoco_robot(two_link_env, deploy_two_link, eden_caplog):
    """The robot acts on each command from the substep it arrives on, and the deploy says the delay is off.

    15 ms is 7.5 MuJoCo substeps: the latency the MuJoCo robot used to replay, which made ``connect()`` refuse it.
    """
    dep = _connected(deploy_two_link, control_dt=0.002)
    assert "ActionDelay on action term(s) ['joint_pos'] is not applied" in eden_caplog.text
    assert dep._n_substeps == 1
    dep.reset()
    term = two_link_env.action_manager.terms["joint_pos"]
    for raw in (0.2, -0.3):
        payload = dep.action_to_payload(torch.full((1, 2), raw))
        np.testing.assert_allclose(payload.dofs_pos, term.processed_action[0].numpy(), err_msg="sent undelayed")
        before = dep._sim.read()
        dep.send_payload(payload)
        # tau = kp * (target - q) + kd * (0 - dq), solved for the target MuJoCo acted on.
        tau = dep._sim.read()["dofs_torque"]
        acted = (tau + payload.dofs_kd * before["dofs_vel"]) / payload.dofs_kp + before["dofs_pos"]
        np.testing.assert_allclose(acted, payload.dofs_pos, atol=1e-9)


# ---------------------------------------------------------------------------
# init_target="motion_rsi": the one sim-only start
# ---------------------------------------------------------------------------


def _clip_frame_zero(dofs_name) -> np.ndarray:
    return MANAGER_CLIP_JOINT_STEP * np.array([G1_DOFS29_NAME.index(name) for name in dofs_name])


@pytest.mark.parametrize("async_physics", [False, True], ids=["lockstep", "async"])
def test_motion_rsi_places_joints_and_base_at_the_clips_first_frame(deploy, async_physics):
    """With ``--async`` the placement goes through the physics thread, which then holds frame 0."""
    dep = _connected(deploy, init_target="motion_rsi", async_physics=async_physics)

    dep.reset()

    state = dep._physics.read() if async_physics else dep._sim.read()
    tolerance = 0.05 if async_physics else 1e-6  # the thread may have run a few substeps since
    np.testing.assert_allclose(state["dofs_pos"], _clip_frame_zero(dep.dofs_name), atol=tolerance)
    np.testing.assert_allclose(state["base_pos"], MANAGER_CLIP_BASE_POS, atol=tolerance)
    if not async_physics:
        assert dep._sim.data.time == 0.0, "a teleport, not a ramp: nothing was commanded"


def test_motion_rsi_without_a_clip_keeps_the_reset_pose(deploy_two_link, eden_caplog):
    dep = _connected(deploy_two_link, init_target="motion_rsi")
    dep.reset()
    np.testing.assert_allclose(dep._sim.read()["dofs_pos"], dep.default_dof_pos, atol=1e-9)
    assert "tracks no motion; keeping the reset pose" in eden_caplog.text
