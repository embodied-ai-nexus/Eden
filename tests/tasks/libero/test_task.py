"""LIBERO end to end on a real built scene: fixed-state reset, stove indicator, goal metric, gripper, Panda dynamics.

Every ``libero_goal`` task shares one scene (cabinet, stove, rack, bowl, plate, ...), so one build of
``turn_on_the_stove`` serves the goal metric of several of its sibling tasks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# isort: off
import eden  # isolate the DDP rank before importing Genesis and Torch
# isort: on

import numpy as np
import pytest
import torch
from genesis.utils.geom import transform_by_quat

from eden.options.managers.events import EventManagerOptions
from eden.options.robots.libero import LIBERO_ROBOT_ENTITY_NAME, LiberoPanda
from eden.tasks import TASK_REGISTRY
from eden.tasks.libero.catalog import get_task_spec
from eden.tasks.libero.custom_terms.events import LiberoFixedStateReset, LiberoStoveBurnerVisualization
from eden.tasks.libero.custom_terms.metrics import LiberoGoalMetric
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

TASK_NAME = "turn_on_the_stove"
_ROBOT_JOINTS = (*LiberoPanda.arm_dofs_name, *LiberoPanda.gripper_dofs_name)
_BUNDLE_INIT_IDS = (10, 11, 12)


@pytest.fixture(scope="module")
def built_env():
    """The task's scene, OSC and gripper, and its stove events; no cameras and no bundle-backed reset."""
    config = TASK_REGISTRY.build("libero", task_name=TASK_NAME, init_ids="0,1")
    events = {
        name: getattr(config.event_options, name)
        for name in config.event_options.term_keys()
        if name != "libero_fixed_state"
    }
    return build_env(
        EdenConfig(
            env_options=config.env_options,
            scene_options=config.scene_options,
            action_options=config.action_options,
            event_options=EventManagerOptions(**events),
        ),
        reset=True,
    )


@pytest.fixture
def env(built_env):
    """The module's env, reset to its built state after the test."""
    yield built_env
    built_env.reset()


def _dofs(entity, names):
    _, idx = entity.find_named_dofs_idx_local(list(names), preserve_order=True)
    return torch.as_tensor(idx, dtype=torch.int32)


def _set_dofs(entity, names, values):
    entity.set_dofs_pos(torch.as_tensor(values, dtype=torch.float32), dofs_idx_local=_dofs(entity, names))


# --------------------------------------------------------------------------------------------------------------
# LiberoFixedStateReset
# --------------------------------------------------------------------------------------------------------------


def _bowl_state(row):
    """A distinct free-body state per bundle row: row 0 translates only, rows 1 and 2 also spin."""
    quat = [[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5], [0.0, 0.0, 0.0, 1.0]][row]
    ang_vel_body = [[0.0, 0.0, 0.0], [0.2, 0.3, 0.4], [-0.5, 0.1, 0.2]][row]
    return [0.1 * row, 0.2, 1.0], quat, [0.1, 0.2 + row, 0.3], ang_vel_body


def _write_bundle(env, tmp_path, *, init_ids=_BUNDLE_INIT_IDS, drop_fixture_joint=None):
    """Write a bundle for the live scene: each entity keeps its pose, except the moved robot, bowl and joints."""
    task_spec = get_task_spec(TASK_NAME)
    rows = len(init_ids)
    descriptors, arrays = [], {}

    def put(key, values):
        arrays[key] = np.asarray(values, dtype=np.float64).reshape(rows, -1)

    for name in (LIBERO_ROBOT_ENTITY_NAME, *task_spec.entity_categories):
        entity = env.entities[name]
        prefix = f"entity/{name}"
        if name == LIBERO_ROBOT_ENTITY_NAME:
            descriptor = {"semantic_name": name, "joint_names": list(_ROBOT_JOINTS)}
            pos, quat = entity.get_pos(relative=True)[0], entity.get_quat(relative=True)[0]
            joint_names = list(_ROBOT_JOINTS)
        else:
            joint_names = [
                joint for joint in entity.joints_name if joint != "root_joint" and _dofs(entity, [joint]).numel()
            ]
            descriptor = {"semantic_name": name, "root_link_name": entity.base_link_name, "joint_names": joint_names}
            link0 = torch.tensor([0], dtype=torch.int32)
            pos = entity.get_links_pos(ls_idx_local=link0, relative=True)[0, 0]
            quat = entity.get_links_quat(ls_idx_local=link0, relative=True)[0, 0]
        if name == drop_fixture_joint:
            descriptor["joint_names"] = joint_names = joint_names[:-1]
        descriptors.append(descriptor)

        if name == "akita_black_bowl_1":
            states = [_bowl_state(row) for row in range(rows)]
            put(f"{prefix}/root_pos_world", [s[0] for s in states])
            quats = np.asarray([s[1] for s in states])
            put(f"{prefix}/root_quat_wxyz_world", quats / np.linalg.norm(quats, axis=1, keepdims=True))
            put(f"{prefix}/root_lin_vel_world", [s[2] for s in states])
            put(f"{prefix}/root_ang_vel_body", [s[3] for s in states])
        else:
            quat = quat.double().cpu().numpy()
            put(f"{prefix}/root_pos_world", [pos.tolist()] * rows)
            put(f"{prefix}/root_quat_wxyz_world", [(quat / np.linalg.norm(quat)).tolist()] * rows)
            put(f"{prefix}/root_lin_vel_world", np.zeros((rows, 3)))
            put(f"{prefix}/root_ang_vel_body", np.zeros((rows, 3)))
        for index, joint in enumerate(joint_names):
            # In range, distinct per row and per joint; nonzero velocities catch a pose setter running
            # after the velocity write and zeroing it.
            lower, upper = (
                bound[0, 0].item() for bound in entity.get_dofs_limit(dofs_idx_local=_dofs(entity, [joint]))
            )
            put(f"{prefix}/joint/{joint}/qpos", [[lower + (upper - lower) * (0.3 + 0.1 * row)] for row in range(rows)])
            put(f"{prefix}/joint/{joint}/qvel", [[0.05 * (index + 1) + 0.1 * row] for row in range(rows)])

    manifest = {"task_name": TASK_NAME, "init_ids": list(init_ids), "entities": descriptors}
    (tmp_path / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(tmp_path / "states.npz", **arrays)
    return tmp_path / "task.json", arrays


def _fixed_state_term(env, bundle_path, init_ids):
    options = LiberoFixedStateReset.configure(task_name=TASK_NAME, bundle_path=str(bundle_path), init_ids=init_ids)
    return LiberoFixedStateReset(env=env, options=options)


@pytest.mark.parametrize(
    ("init_ids", "expected_rows"),
    [([], [0, 1]), ([12, 10], [2, 0])],
    ids=["implicit-first-rows", "explicit-subset"],
)
def test_fixed_state_restores_poses_joints_and_nonzero_velocities(env, tmp_path, init_ids, expected_rows):
    bundle_path, arrays = _write_bundle(env, tmp_path)
    term = _fixed_state_term(env, bundle_path, init_ids)
    term.build()
    term.compute()  # raises if its own read-back of the restored state disagrees with the bundle

    rows = torch.tensor(expected_rows)

    def expected(key):
        return torch.as_tensor(arrays[key], dtype=torch.float32)[rows]

    robot = env.entities[LIBERO_ROBOT_ENTITY_NAME]
    for joint in _ROBOT_JOINTS:
        idx = _dofs(robot, [joint])
        torch.testing.assert_close(robot.get_dofs_pos(dofs_idx_local=idx), expected(f"entity/robot/joint/{joint}/qpos"))
        torch.testing.assert_close(robot.get_dofs_vel(dofs_idx_local=idx), expected(f"entity/robot/joint/{joint}/qvel"))
    cabinet = env.entities["wooden_cabinet_1"]
    for joint in ("top_level", "middle_level", "bottom_level"):
        idx = _dofs(cabinet, [joint])
        prefix = f"entity/wooden_cabinet_1/joint/{joint}"
        torch.testing.assert_close(cabinet.get_dofs_pos(dofs_idx_local=idx), expected(f"{prefix}/qpos"))
        torch.testing.assert_close(cabinet.get_dofs_vel(dofs_idx_local=idx), expected(f"{prefix}/qvel"))

    bowl, link0 = env.entities["akita_black_bowl_1"], torch.tensor([0], dtype=torch.int32)
    source_quat = expected("entity/akita_black_bowl_1/root_quat_wxyz_world")
    torch.testing.assert_close(bowl.get_pos(relative=True), expected("entity/akita_black_bowl_1/root_pos_world"))
    torch.testing.assert_close(bowl.get_quat(relative=True), source_quat, atol=1e-6, rtol=0.0)
    # The source records the linear velocity of its root point and the body-frame angular velocity.
    torch.testing.assert_close(
        bowl.get_links_vel(ls_idx_local=link0, relative=True)[:, 0],
        expected("entity/akita_black_bowl_1/root_lin_vel_world"),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        bowl.get_links_ang(ls_idx_local=link0)[:, 0],
        transform_by_quat(expected("entity/akita_black_bowl_1/root_ang_vel_body"), source_quat),
        atol=1e-5,
        rtol=0.0,
    )


def test_fixed_state_init_id_count_mismatch_names_num_envs_and_the_fix(env, tmp_path):
    bundle_path, _ = _write_bundle(env, tmp_path)
    term = _fixed_state_term(env, bundle_path, [10])

    with pytest.raises(ValueError, match=r"selects 1 fixed state\(s\) but the environment has num_envs=2") as excinfo:
        term.build()
    message = str(excinfo.value)
    assert "init_ids=[10] -> num_envs=1" in message
    assert "--num_envs" in message


def test_fixed_state_rejects_more_envs_than_implicit_bundle_rows(env, tmp_path):
    bundle_path, _ = _write_bundle(env, tmp_path, init_ids=(10,))

    with pytest.raises(ValueError, match="1 init states, fewer than num_envs=2"):
        _fixed_state_term(env, bundle_path, []).build()


def test_fixed_state_rejects_a_fixture_descriptor_missing_a_joint(env, tmp_path):
    bundle_path, _ = _write_bundle(env, tmp_path, drop_fixture_joint="wooden_cabinet_1")

    with pytest.raises(ValueError, match=r"fixture joint coverage mismatch for 'wooden_cabinet_1'.*bottom_level"):
        _fixed_state_term(env, bundle_path, []).build()


# --------------------------------------------------------------------------------------------------------------
# LiberoStoveBurnerVisualization
# --------------------------------------------------------------------------------------------------------------


def test_stove_indicator_shows_at_the_burner_only_when_the_button_is_on(env):
    stove, indicator = env.entities["flat_stove_1"], env.entities["flat_stove_1_burner_indicator"]
    term = LiberoStoveBurnerVisualization(
        env=env,
        options=LiberoStoveBurnerVisualization.configure(
            stove_entity_name="flat_stove_1", indicator_entity_name="flat_stove_1_burner_indicator"
        ),
    )
    term.build()
    burner = torch.as_tensor(stove.find_named_links_idx_local(["burner"])[1], dtype=torch.int32)

    _set_dofs(stove, ["button"], [[0.5], [0.5]])  # the on-threshold itself counts as on
    term.compute()
    visible = indicator.get_pos()
    burner_pos, burner_quat = (
        stove.get_links_pos(ls_idx_local=burner)[:, 0],
        stove.get_links_quat(ls_idx_local=burner)[:, 0],
    )
    torch.testing.assert_close(torch.linalg.vector_norm(visible - burner_pos, dim=1), torch.full((2,), 0.022))
    torch.testing.assert_close(indicator.get_quat(), burner_quat)

    _set_dofs(stove, ["button"], [[0.49], [0.49]])
    term.compute(envs_idx=torch.tensor([1]))
    torch.testing.assert_close(indicator.get_pos()[0], visible[0])  # env 0 was not selected
    torch.testing.assert_close(indicator.get_pos()[1], visible[1] + torch.tensor([0.0, 0.0, -10.0]))


# --------------------------------------------------------------------------------------------------------------
# LiberoGoalMetric
# --------------------------------------------------------------------------------------------------------------


def _press_stove_button(env):
    _set_dofs(env.entities["flat_stove_1"], ["button"], [[0.5], [0.49]])  # Turnon is ``>= 0.5``


def _pull_middle_drawer(env):
    _set_dofs(env.entities["wooden_cabinet_1"], ["middle_level"], [[-0.15], [-0.14]])  # Open is strictly ``< -0.14``


def _slide_plate_to_the_stove_front(env):
    # BDDL region x in [-0.09, -0.01], y in [0.17, 0.25] on the table top at z=0.895; env 1 misses it in x.
    env.entities["plate_1"].set_pos(torch.tensor([[-0.05, 0.21, 0.905], [0.05, 0.21, 0.905]]), relative=True)


def _rest_bowl_on_the_plate(env):
    # Contact-dependent, so it settles under physics; env 1 rests beside the plate, beyond the 3 cm xy window.
    env.entities["plate_1"].set_pos(torch.tensor([[0.0, 0.2, 0.92], [0.0, 0.2, 0.92]]), relative=True)
    env.entities["akita_black_bowl_1"].set_pos(torch.tensor([[0.0, 0.2, 0.96], [0.1, 0.2, 0.96]]), relative=True)
    for _ in range(10):
        env.step({"osc_pose": torch.zeros((2, 6)), "gripper": torch.zeros((2, 1))})


@pytest.mark.parametrize(
    ("task_name", "arrange"),
    [
        ("turn_on_the_stove", _press_stove_button),
        ("open_the_middle_drawer_of_the_cabinet", _pull_middle_drawer),
        ("push_the_plate_to_the_front_of_the_stove", _slide_plate_to_the_stove_front),
        ("put_the_bowl_on_the_plate", _rest_bowl_on_the_plate),
    ],
    ids=["turnon", "open-region", "on-workspace-region", "on-object-with-contact"],
)
def test_goal_metric_scores_success_only_where_the_goal_holds(env, task_name, arrange):
    arrange(env)
    metric = LiberoGoalMetric(env=env, options=LiberoGoalMetric.configure(task_name=task_name))
    metric.build()

    assert metric.compute().tolist() == [1.0, 0.0]


# --------------------------------------------------------------------------------------------------------------
# LiberoPandaGripperController and the built Panda
# --------------------------------------------------------------------------------------------------------------


def test_gripper_close_command_ramps_both_fingers_to_positive_targets(env):
    env.reset()
    env.step({"osc_pose": torch.zeros((2, 6)), "gripper": torch.ones((2, 1))})

    # One policy step is 25 physics steps of +0.01 from the half-open midpoint (0.02) toward closed (0.0).
    torch.testing.assert_close(env.action_manager.get_term("gripper").processed_action, torch.full((2, 2), 0.015))
    robot = env.entities[LIBERO_ROBOT_ENTITY_NAME]
    assert (robot.get_dofs_pos(dofs_idx_local=_dofs(robot, LiberoPanda.gripper_dofs_name)) > 0.0).all()


def test_built_panda_carries_the_libero_dynamics(env):
    robot = env.entities[LIBERO_ROBOT_ENTITY_NAME]
    idx = _dofs(robot, _ROBOT_JOINTS)
    arm_armature = [5.0 / index for index in range(1, 8)]

    torch.testing.assert_close(robot.get_dofs_armature(dofs_idx_local=idx)[0], torch.tensor([*arm_armature, 1.0, 1.0]))
    torch.testing.assert_close(
        robot.get_dofs_damping(dofs_idx_local=idx)[0], torch.tensor([0.1] * 5 + [0.01] * 2 + [100.0] * 2)
    )
    torch.testing.assert_close(robot.get_dofs_frictionloss(dofs_idx_local=idx)[0], torch.tensor([0.1] * 7 + [1.0] * 2))
    gripper = _dofs(robot, LiberoPanda.gripper_dofs_name)
    torch.testing.assert_close(robot.get_dofs_kp(dofs_idx_local=gripper)[0], torch.tensor([1000.0, 1000.0]))
    torch.testing.assert_close(robot.get_dofs_kd(dofs_idx_local=gripper)[0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(robot.get_dofs_force_range(dofs_idx_local=gripper)[1][0], torch.tensor([20.0, 20.0]))


# --------------------------------------------------------------------------------------------------------------
# Mounted-Panda OSC on the GPU backend
# --------------------------------------------------------------------------------------------------------------


def _run_mounted_osc_probe() -> None:
    import genesis as gs

    eden.init(backend=gs.gpu, debug=True)
    config = TASK_REGISTRY.build(
        "libero", task_name="KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet", init_ids="0,1,2"
    )
    env = build_env(
        EdenConfig(
            env_options=config.env_options, scene_options=config.scene_options, action_options=config.action_options
        )
    )
    assert env.device.type == "cuda"
    robot = env.entities[LIBERO_ROBOT_ENTITY_NAME]
    arm = _dofs(robot, LiberoPanda.arm_dofs_name).to(env.device)
    initial_arm_pos = robot.get_dofs_pos(dofs_idx_local=arm).clone()
    osc = env.action_manager.get_term("osc_pose")
    ee_link = torch.as_tensor(
        robot.find_named_links_idx_local(osc.ee_link_name)[1], dtype=torch.int32, device=env.device
    )
    ee_local_pos = torch.tensor(osc.ee_local_pos, device=env.device)

    def grip_site() -> torch.Tensor:
        link_pos = robot.get_links_pos(ls_idx_local=ee_link).squeeze(1)
        link_quat = robot.get_links_quat(ls_idx_local=ee_link).squeeze(1)
        return link_pos + transform_by_quat(ee_local_pos.expand_as(link_pos), link_quat)

    initial_site = grip_site()
    osc_action = torch.zeros((3, 6), device=env.device)
    osc_action[0, 0], osc_action[1, 0] = 0.1, -0.1
    env.step({"osc_pose": osc_action, "gripper": torch.zeros((3, 1), device=env.device)})

    site_delta = grip_site() - initial_site
    assert torch.isfinite(site_delta).all()
    assert (
        torch.linalg.vector_norm(robot.get_dofs_pos(dofs_idx_local=arm)[:2] - initial_arm_pos[:2], dim=1) > 1e-8
    ).all()
    # Opposite x commands move the grip site in opposite x directions, relative to zero-action settling.
    signed_x_delta = (site_delta[:2, 0] - site_delta[2, 0]) * torch.tensor((1.0, -1.0), device=env.device)
    assert (signed_x_delta > 1e-6).all(), site_delta.tolist()


@pytest.mark.slow
def test_mounted_panda_executes_a_nonzero_osc_step_on_gpu():
    """The suite runs on ``gs.cpu`` and Genesis cannot switch backend in-process, so the GPU run is a subprocess."""
    repository_root = Path(__file__).parents[3]
    # Prepended, not replaced: a run that puts its own packages on PYTHONPATH (a worktree's sitecustomize, a Genesis
    # overlay) must import the same ones in the child, or the probe runs another Genesis than the suite.
    python_path = os.pathsep.join([str(repository_root), *filter(None, [os.environ.get("PYTHONPATH")])])
    process_env = {**os.environ, "PYTHONPATH": python_path, "PYTHONFAULTHANDLER": "1"}
    result = subprocess.run(
        [sys.executable, __file__, "--run-mounted-osc-probe"],
        cwd=repository_root,
        capture_output=True,
        env=process_env,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"exit {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


if __name__ == "__main__":
    if sys.argv[1:] != ["--run-mounted-osc-probe"]:
        raise SystemExit("expected --run-mounted-osc-probe")
    _run_mounted_osc_probe()
