"""The supermimic task config and its data flags."""

import pytest

from eden.tasks import TASK_REGISTRY


@pytest.mark.parametrize("flag", ["raw_data_path", "processed_data_path", "clip_list"])
def test_an_empty_data_flag_raises(flag):
    with pytest.raises(ValueError, match=f"--{flag} was given an empty path"):
        TASK_REGISTRY.build("supermimic", **{flag: ""})


@pytest.mark.parametrize(
    ("dataset", "name"), [("lafan1", "lafan1_root_joints_g1_dataset"), ("bones_seed", "seed_root_joints_g1_dataset")]
)
def test_the_dataset_flag_selects_a_root_and_joints_library_at_the_control_rate(dataset, name):
    cfg = TASK_REGISTRY.build("supermimic", dataset=dataset, clip_list="clips.txt")
    options = cfg.motion_options.default.dataset_options
    assert options.name == name
    assert options.include_list_file == "clips.txt"
    assert options.target_framerate == pytest.approx(50.0)
