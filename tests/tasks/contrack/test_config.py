"""The ConTrack task config: how its data mod turns one clip into the scene, motion and curriculum options."""

import os

import numpy as np
import pytest

from eden.tasks import TASK_REGISTRY
from eden.tasks.contrack.config import _export_object_mesh
from eden.utils.assets import SAMPLE_DATA_REPO
from eden.utils.misc import get_cache_dir


@pytest.fixture(scope="module")
def cfg():
    return TASK_REGISTRY.build("contrack")


def test_the_default_clip_is_fetched_into_the_user_cache_not_read_from_the_checkout(cfg):
    """The default must not point inside the source tree, which a pip install does not have."""
    raw = cfg.motion_options.default.dataset_options.raw_data_path
    assert raw.startswith(get_cache_dir(SAMPLE_DATA_REPO, "contrack")), raw


def test_an_empty_data_flag_raises_instead_of_falling_back_to_the_default_clip():
    with pytest.raises(FileNotFoundError, match="must point at a single ConTrack clip"):
        TASK_REGISTRY.build("contrack", data="")


def test_the_target_framerate_must_match_the_control_rate():
    with pytest.raises(ValueError, match="does not match the control rate"):
        TASK_REGISTRY.build("contrack", target_framerate=60.0)


def test_handedness_is_read_from_the_clip(cfg):
    """The bundled sample clip records a left hand, so the robot and the dataset are both left-handed."""
    assert type(cfg.scene_options.robot).__name__ == "XArm7XHand1_L"
    assert cfg.motion_options.default.dataset_options.side == "left"
    assert type(cfg.runner_options).__name__ == "RslRlContrackRunnerOptions"  # CMDP-PPO through `eden train`


def test_the_pose_break_gate_starts_at_the_curriculum_lenient_end(cfg):
    gate = cfg.termination_options.object_pose_break
    curriculum = cfg.curriculum_options.object_pose_threshold
    assert curriculum.termination_term_name == "object_pose_break"
    assert (gate.pos_threshold, gate.rot_threshold) == (curriculum.pos_threshold_max, curriculum.rot_threshold_max)


def test_the_object_mesh_cache_is_content_addressed_and_written_atomically(tmp_path):
    """Every DDP rank runs the mod, so no partial file may appear under the final name."""
    clip = tmp_path / "clip.h5"
    faces = np.array([[0, 1, 2]], np.uint32)
    first = _export_object_mesh(clip, np.zeros((3, 3), np.float32), faces)
    again = _export_object_mesh(clip, np.zeros((3, 3), np.float32), faces)
    edited = _export_object_mesh(clip, np.ones((3, 3), np.float32), faces)

    assert first == again
    assert edited != first
    cache = tmp_path / ".cache"
    assert sorted(p.name for p in cache.iterdir()) == sorted([os.path.basename(first), os.path.basename(edited)])
