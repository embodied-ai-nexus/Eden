"""Tests for ``eden.tasks.simpler.base_config``: every Bridge task shares the sysid'd WidowX250 and 5 Hz control.

The gain tables are external BridgeV2 sysid results, written out literally.
"""

from __future__ import annotations

import pytest

from eden.tasks.registry import TASK_REGISTRY
from eden.utils.configs import EdenConfig
from eden.tasks.simpler.base_config import DEFAULT_ROBOT

# Three tasks from bridge_table, one from bridge_sink: the split the base config closed.
BRIDGE_TASKS = (
    "simpler_put_spoon_on_table_cloth",
    "simpler_put_carrot_on_plate",
    "simpler_stack_green_cube_on_yellow_cube",
    "simpler_put_eggplant_in_basket",
)

SYSID_KP = {
    "waist": 1169.7891719504198,
    "shoulder": 730.0,
    "elbow": 808.4601346394447,
    "forearm_roll": 1229.1299089624076,
    "wrist_angle": 1272.2760456418862,
    "wrist_rotate": 1056.3326605132252,
    "left_finger": 100,
    "right_finger": 100,
}
SYSID_KD = {
    "waist": 330.0,
    "shoulder": 180.0,
    "elbow": 152.12036565582588,
    "forearm_roll": 309.6215302722146,
    "wrist_angle": 201.04998711007383,
    "wrist_rotate": 269.51458932695414,
    "left_finger": 20,
    "right_finger": 20,
}


@pytest.fixture(scope="module")
def bridge_configs():
    return {name: TASK_REGISTRY.get(name)() for name in BRIDGE_TASKS}


@pytest.mark.parametrize("field", ["env_options", "action_options"])
def test_bridge_tasks_agree_on_every_base_owned_field(bridge_configs, field):
    dumps = {name: getattr(cfg, field).model_dump() for name, cfg in bridge_configs.items()}
    reference_name, reference = next(iter(dumps.items()))
    for name, dump in dumps.items():
        assert dump == reference, f"{name}.{field} drifted from {reference_name}.{field}"


def test_bridge_tasks_all_use_the_sysid_widowx250_at_5hz(bridge_configs):
    """BridgeV2 demonstrations are 5 Hz, and the arm inside each per-task scene is the fitted one."""
    for name, cfg in bridge_configs.items():
        robot = cfg.scene_options.robot
        assert robot.default_dofs_kp == SYSID_KP, f"{name} lost the sysid kp table"
        assert robot.default_dofs_kd == SYSID_KD, f"{name} lost the sysid kd table"
        assert robot.material.gravity_compensation == 1.0, f"{name} lost gravity compensation"
        assert robot.default_root_pos == (0.0, 0.0, 0.0), f"{name} moved the arm base off SIMPLER's base_link origin"
        rate = 1.0 / (cfg.env_options.sim_dt * cfg.env_options.decimation)
        assert rate == pytest.approx(5.0), f"{name} runs at {rate} Hz"


def test_tasks_do_not_share_mutable_robot_state(bridge_configs):
    """One module-level ``DEFAULT_ROBOT`` feeds every task: editing one task's gains must not retune the rest."""
    robots = [cfg.scene_options.robot for cfg in bridge_configs.values()]
    robots[0].default_dofs_kp["waist"] = -1.0
    try:
        for other in robots[1:]:
            assert other.default_dofs_kp["waist"] == SYSID_KP["waist"]
        assert DEFAULT_ROBOT.default_dofs_kp["waist"] == SYSID_KP["waist"]
    finally:
        robots[0].default_dofs_kp["waist"] = SYSID_KP["waist"]


def test_the_real_image_backdrop_survives_a_saved_config(bridge_configs, tmp_path):
    """``eden inference`` replays the run's ``config.json``: each camera's backdrop must come back as declared."""
    for name, cfg in bridge_configs.items():
        path = tmp_path / f"{name}.json"
        cfg.save_as_json(str(path), lock_config=False)
        loaded = EdenConfig.load_from_file(str(path))
        (backdrop,) = cfg.cameras_options.camera.overlays
        (reloaded,) = loaded.cameras_options.camera.overlays
        assert backdrop.name == "background_image_composite" and backdrop.entities, f"{name} lost its backdrop"
        assert reloaded == backdrop, f"{name}'s backdrop changed through config.json"
