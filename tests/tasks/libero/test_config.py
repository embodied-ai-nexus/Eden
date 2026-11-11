"""The registered ``libero`` task: what ``LiberoTaskMod`` derives from a catalog task and an init-id selection."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import numpy as np
import pytest

from eden.constants import EventMode
from eden.options.robots.libero import LIBERO_ROBOT_ENTITY_NAME, LiberoPanda
from eden.tasks import TASK_REGISTRY
from eden.tasks.libero.catalog import get_suite_task_names, get_task_spec
from eden.tasks.libero.config import (
    LIBERO_POLICY_HORIZON_STEPS,
    LIBERO_RESET_WARMUP_STEPS,
    LiberoConfig,
    LiberoTaskMod,
)
from eden.utils.configs import EdenRLConfig
from tests.support.asserts import assert_allclose

TASK_NAME = get_suite_task_names("libero_object")[0]
STOVE_TASK_NAME = "turn_on_the_stove"
INDICATOR = "flat_stove_1_burner_indicator"


def test_the_default_bundle_root_is_read_from_the_asset_directory_when_the_task_is_built(monkeypatch, tmp_path):
    monkeypatch.setenv("EDEN_ASSETS_DIR", str(tmp_path / "assets"))

    config = TASK_REGISTRY.build("libero", task_name=TASK_NAME, init_ids="0")

    expected = tmp_path / "assets" / "libero" / "fixed_states" / "libero_object" / TASK_NAME / "state-bundle"
    assert config.event_options.libero_fixed_state.bundle_path == str(expected)


def test_init_ids_select_the_bundle_rows_and_set_num_envs():
    config = TASK_REGISTRY.build("libero", task_name=TASK_NAME, bundle_root="/bundles", init_ids="0, 10, 19")

    assert config.env_options.num_envs == 3
    reset = config.event_options.libero_fixed_state
    assert reset.task_name == TASK_NAME
    assert reset.bundle_path == f"/bundles/libero_object/{TASK_NAME}/state-bundle"
    assert reset.init_ids == [0, 10, 19]
    assert config.metric_options.goal.task_name == TASK_NAME
    assert config.language_instructions == [get_task_spec(TASK_NAME).language_instruction]


def test_episode_covers_the_warmup_and_the_policy_horizon():
    env_options = TASK_REGISTRY.build("libero").env_options
    policy_dt = env_options.sim_dt * env_options.decimation

    assert env_options.episode_length_s / policy_dt == pytest.approx(
        LIBERO_RESET_WARMUP_STEPS + LIBERO_POLICY_HORIZON_STEPS
    )


def test_without_init_ids_num_envs_leads_and_the_bundle_defaults_to_eden_assets():
    config = TASK_REGISTRY.build("libero")

    assert config.env_options.num_envs == 20
    assert config.event_options.libero_fixed_state.init_ids == []
    assert config.event_options.libero_fixed_state.bundle_path.endswith(
        f"/libero/fixed_states/libero_object/{TASK_NAME}/state-bundle"
    )


def test_malformed_init_ids_are_rejected():
    with pytest.raises(ValueError, match="comma-separated list of integers"):
        TASK_REGISTRY.build("libero", init_ids="7,,9")


def test_policy_observes_both_cameras_and_split_arm_and_gripper_joints():
    policy = TASK_REGISTRY.build("libero").observation_options.policy

    assert {name for name in policy.model_extra if not name.startswith("_")} == {
        "agentview_rgb",
        "eye_in_hand_rgb",
        "arm_joint_pos",
        "gripper_joint_pos",
    }
    assert policy.arm_joint_pos.dofs_name == [f"joint{index}" for index in range(1, 8)]
    assert policy.gripper_joint_pos.dofs_name == ["finger_joint1", "finger_joint2"]


def test_cameras_use_the_family_agentview_pose_and_the_wrist_link():
    cameras = TASK_REGISTRY.build("libero", task_name=TASK_NAME, init_ids="7").cameras_options

    # The extrinsic is world-to-camera: its inverse puts the camera at the floor family's agentview.
    camera_to_world = np.linalg.inv(cameras.agentview.cam_extrinsic)
    assert_allclose(camera_to_world[:3, 3], (0.8965773716836134, 5.216182733499864e-07, 0.65), atol=1e-12)
    wrist = cameras.eye_in_hand
    assert wrist.attach_entity_name == LIBERO_ROBOT_ENTITY_NAME
    assert wrist.attach_link_name == LiberoPanda.ee_links_name[0]


def test_flat_stove_adds_an_indicator_entity_and_its_events():
    config = TASK_REGISTRY.build("libero", task_name=STOVE_TASK_NAME, init_ids="7")

    assert getattr(config.scene_options, INDICATOR).collision is False
    reset = getattr(config.event_options, f"{INDICATOR}_reset")
    assert reset.mode == EventMode.RESET
    assert reset.priority > config.event_options.libero_fixed_state.priority  # after the fixed state lands
    assert getattr(config.event_options, f"{INDICATOR}_interval").mode == EventMode.INTERVAL


def test_stove_indicator_interval_follows_the_callers_policy_rate():
    config = LiberoConfig()
    config.env_options.sim_dt = 0.004
    config.env_options.decimation = 10

    LiberoTaskMod(task_name=STOVE_TASK_NAME, bundle_root="/unused").apply(config)

    assert getattr(config.event_options, f"{INDICATOR}_interval").interval_range_s == (0.04, 0.04)


def test_config_round_trips_through_eden_serialization(tmp_path):
    config = TASK_REGISTRY.build("libero", task_name=TASK_NAME, init_ids="3,9")
    path = tmp_path / "libero_config.json"

    config.save_as_json(str(path), lock_config=False)
    loaded = EdenRLConfig.load_from_file(str(path))

    assert type(loaded) is LiberoConfig
    assert loaded.env_options.num_envs == 2
    assert loaded.event_options.libero_fixed_state.init_ids == [3, 9]
    assert loaded.metric_options.goal.task_name == TASK_NAME
