"""``generate_demos``: refuses what it cannot do before building anything, and writes labelled, aligned episodes."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

from pathlib import Path

import h5py
import pytest

from eden.constants import DatasetExportMode
from eden.extensions.scripted_demos.generate import generate_demos
from eden.tasks import TASK_REGISTRY


def test_a_task_without_a_demonstrator_is_refused_naming_the_ones_that_have_one(tmp_path):
    config = TASK_REGISTRY.build("robosuite_two_arm_lift")
    with pytest.raises(KeyError, match=r"'robosuite_two_arm_lift' has no scripted demonstrator.*robosuite_stack"):
        generate_demos(config, "robosuite_two_arm_lift", num_episodes=1, output_dir=str(tmp_path))


def test_exporting_nothing_is_refused(tmp_path):
    config = TASK_REGISTRY.build("robosuite_lift")
    with pytest.raises(ValueError, match="EXPORT_NONE"):
        generate_demos(
            config,
            "robosuite_lift",
            num_episodes=1,
            output_dir=str(tmp_path),
            export_mode=DatasetExportMode.EXPORT_NONE,
        )


def test_generated_episodes_are_complete_labelled_and_row_aligned(tmp_path):
    config = TASK_REGISTRY.build("robosuite_lift")
    config.env_options.num_envs = 2
    report = generate_demos(config, "robosuite_lift", num_episodes=2, output_dir=str(tmp_path), filename="lift")

    assert report.episodes >= 2
    assert Path(report.dataset_path).exists()
    with h5py.File(report.dataset_path, "r") as f:
        demos = [f["data"][name] for name in f["data"]]
        # Every ended episode is exported (EXPORT_ALL); the ones still running at the end are not.
        assert len(demos) == report.episodes
        assert sum(bool(demo.attrs["success"]) for demo in demos) == report.successes
        for demo in demos:
            steps = demo["action"].shape[0]
            assert demo["action"].shape[1] == 7  # 6-D IK delta + gripper
            for key in ("obs/policy", "obs/critic", "state/qpos", "timestamp", "reward"):
                assert demo[key].shape[0] == steps, key
            assert "initial_state" in demo
    # The scripted Panda lifts the cube.
    assert report.successes == report.episodes


def test_camera_only_is_refused_for_a_demonstrator_that_reads_simulator_state(tmp_path):
    config = TASK_REGISTRY.build("robosuite_lift")
    with pytest.raises(
        ValueError,
        match=r"'robosuite_lift''s demonstrator.*no camera-only mode; tasks that do: \[[^]]*'robosuite_pick_place'[^]]*\]",
    ):
        generate_demos(config, "robosuite_lift", num_episodes=1, output_dir=str(tmp_path), camera_only=True)


def test_an_episode_the_task_ends_on_success_is_not_reported_as_a_failure_or_cut_short(tmp_path):
    """Wipe terminates on success, so the env resets the episode — and clears its success flag — inside the step.

    The report must take the episode's success from what was exported, not from the cleared flag:
    reading the flag logged every Wipe episode's trace as "the first unsuccessful episode".
    """
    config = TASK_REGISTRY.build("robosuite_wipe")
    config.env_options.num_envs = 2
    report = generate_demos(config, "robosuite_wipe", num_episodes=2, output_dir=str(tmp_path), p_mistake=0.0)

    assert report.successes == 2
    assert report.cut_short == 0
    assert report.failure_trace == []


def test_a_two_arm_task_records_both_arms_state_side_by_side(tmp_path):
    config = TASK_REGISTRY.build("robosuite_two_arm_transport")
    config.env_options.num_envs = 2
    # Episodes cut by the generator's limit (the demonstrator needs ~2 min): exported, unsuccessful.
    report = generate_demos(
        config, "robosuite_two_arm_transport", num_episodes=2, output_dir=str(tmp_path), episode_length_s=1.0
    )

    assert report.episodes >= 2
    assert report.successes == 0
    with h5py.File(report.dataset_path, "r") as f:
        demo = f["data"][next(iter(f["data"]))]
        steps = demo["action"].shape[0]
        assert demo["action"].shape[1] == 14  # each arm's 6-D IK delta + gripper
        # robot0's then robot1's state: two Panda arms of 7 joints, two root poses.
        assert demo["state/qpos"].shape == (steps, 14)
        assert demo["state/pos"].shape == (steps, 6)
        assert demo["state/quat"].shape == (steps, 8)
