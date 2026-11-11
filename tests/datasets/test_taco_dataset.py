"""Tests for ``eden.datasets.taco_dataset.TacoDataset`` over a synthetic TACO release."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.datasets.taco_dataset import TacoDataset
from tests.support.hand_object_releases import write_taco

TRIPLET = "(dust, brush, plate)"


def _load(raw, mano_dir, **options):
    configured = TacoDataset.configure(
        raw_data_path=str(raw), body_model_dir=str(mano_dir), target_framerate=TacoDataset.source_framerate, **options
    )
    return TacoDataset(options=configured, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp("taco")
    write_taco(raw, TRIPLET, "20230101_001", tool_pos=(0.3, 0.0, 0.1), target_pos=(-0.3, 0.0, 0.1))
    return raw


def test_each_hand_holds_its_nearest_object_in_the_turned_world(raw, mano_dir):
    dataset = _load(raw, mano_dir)
    assert dataset.motion_sequence_keys == ("right_hand", "object_right", "left_hand", "object_left")
    assert dataset.metadata["clip_names"] == ["taco_dust__brush__plate_20230101_001"]
    assert (dataset.metadata["objects_right"], dataset.metadata["objects_left"]) == (["brush01"], ["plate02"])
    # TACO's z-up world turned 180 degrees about z: the tool at (0.3, 0, 0.1) lands at (-0.3, 0, 0.1).
    np.testing.assert_allclose(dataset.motion_sequences["object_right"].base_pos_w[0][0], (-0.3, 0.0, 0.1), atol=1e-6)
    hand = dataset.motion_sequences["right_hand"]
    np.testing.assert_allclose(hand.base_pos_w[0][0], (-0.3, -0.05, 0.2), atol=1e-6)  # the wrist
    torch.testing.assert_close(hand.links_pos_w[0][:, 0], hand.base_pos_w[0])  # keypoint 0 is the wrist
    assert hand.body_pose_quat is None


def test_sides_loads_only_those_hands_and_body_model_params_their_mano(raw, mano_dir):
    dataset = _load(raw, mano_dir, sides="left", body_model_params=True)
    assert dataset.motion_sequence_keys == ("left_hand", "object_left")
    hand = dataset.motion_sequences["left_hand"]
    assert hand.body_model == ("mano_left",) and hand.body_pose_quat._data_shape == (15, 4)
    assert dataset.metadata["objects_right"] == [""]


def test_meshes_are_in_metres_and_a_wrong_directory_names_the_layout(raw, mano_dir, tmp_path):
    np.testing.assert_allclose(_load(raw, mano_dir).object_mesh("brush01").extents, (0.1, 0.1, 0.1), atol=1e-6)
    with pytest.raises(FileNotFoundError, match=r"not a TACO release: \['Hand_Poses', 'Object_Poses'"):
        _load(tmp_path, mano_dir)


def test_the_release_needs_the_mano_files(raw):
    configured = TacoDataset.configure(raw_data_path=str(raw), target_framerate=30.0)
    with pytest.raises(ValueError, match="ships MANO parameters, not keypoints.*set body_model_dir"):
        TacoDataset(options=configured, device=torch.device("cpu"))


def test_a_file_filter_selects_the_same_clips_built_and_loaded_from_the_cache(tmp_path, mano_dir):
    """The filters match clip names (``taco_<action>__...``), which the release's directories do not carry."""
    raw = tmp_path / "raw"
    write_taco(raw, TRIPLET, "20230101_001", tool_pos=(0.3, 0.0, 0.1), target_pos=(-0.3, 0.0, 0.1))
    write_taco(raw, "(stir, spoon, bowl)", "20230101_002", tool_pos=(0.3, 0.0, 0.1), target_pos=(-0.3, 0.0, 0.1))
    cache = str(tmp_path / "cache.pt")

    built = _load(raw, mano_dir, include_pattern="^taco_stir__", processed_data_path=cache)
    assert built.metadata["clip_names"] == ["taco_stir__spoon__bowl_20230101_002"]

    _load(raw, mano_dir, processed_data_path=str(tmp_path / "all.pt"))  # a cache of both clips
    loaded = _load(raw, mano_dir, include_pattern="^taco_stir__", processed_data_path=str(tmp_path / "all.pt"))
    assert loaded.metadata["clip_names"] == built.metadata["clip_names"]
    assert int(loaded.motion_sequences["right_hand"].buffer_length.shape[0]) == 1
