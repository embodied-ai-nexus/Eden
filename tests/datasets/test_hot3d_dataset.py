"""Tests for ``eden.datasets.hot3d_dataset.Hot3dDataset`` over a synthetic HOT3D recording."""

from __future__ import annotations

import pytest
import torch

from eden.datasets.hot3d_dataset import Hot3dDataset
from tests.support.hand_object_releases import HOT3D_RECORDING, write_hot3d


@pytest.fixture(scope="module")
def raw(tmp_path_factory, mano_dir):
    raw = tmp_path_factory.mktemp("hot3d")
    write_hot3d(raw, mano_dir, T=100, touch=(30, 80))
    return raw


def _load(raw, mano_dir, **options):
    configured = Hot3dDataset.configure(
        raw_data_path=str(raw), body_model_dir=str(mano_dir), target_framerate=Hot3dDataset.source_framerate, **options
    )
    return Hot3dDataset(options=configured, device=torch.device("cpu"))


def test_both_hands_on_one_object_is_one_padded_segment_both_hands_holding_it(raw, mano_dir):
    dataset = _load(raw, mano_dir)
    assert dataset.metadata["clip_names"] == [f"{HOT3D_RECORDING}_seg000"]
    assert dataset.metadata["objects_right"] == dataset.metadata["objects_left"] == ["111"]
    # Frames 30-79 in contact, plus 10 still frames at each end.
    assert int(dataset.motion_sequences["right_hand"].buffer_length[0]) == 50 + 2 * 10
    pose = dataset.motion_sequences["object_right"].base_pos_w[0]
    torch.testing.assert_close(pose[:10], pose[10:11].expand(10, 3))  # the padding repeats the first frame


def test_a_segment_of_a_mode_not_kept_is_skipped_by_name(raw, mano_dir):
    with pytest.raises(RuntimeError, match=rf"first: \[\('{HOT3D_RECORDING}_seg000', 'mode B'\)\]"):
        _load(raw, mano_dir, modes=("A",))


def test_in_a_one_hand_segment_only_the_active_hand_holds_the_object(tmp_path, mano_dir):
    """Mode A: the left hand hovers 1 m above everything, so it holds nothing (FlashDexRetarget handed it the object)."""
    write_hot3d(tmp_path, mano_dir, T=100, touch=(30, 80), idle="left")
    dataset = _load(tmp_path, mano_dir, modes=("A",), sides="right")
    assert dataset.metadata["clip_names"] == [f"{HOT3D_RECORDING}_seg000"]
    assert dataset.metadata["objects_right"] == ["111"]
    with pytest.raises(RuntimeError, match=r"'no hand with an object \(left\)'"):
        _load(tmp_path, mano_dir, modes=("A",), sides="left")


def test_the_recording_gives_every_object_over_its_merged_frames(raw, mano_dir):
    """A second camera stream's row 2 ms after frame 5 is the same frame: 100 frames, not 101."""
    context = _load(raw, mano_dir).recording_object_poses(HOT3D_RECORDING)
    assert sorted(context) == ["111", "222"] and context["111"].shape == (100, 4, 4)


def test_the_clips_a_recording_gives_keep_keypoints_not_vertices(raw, mano_dir):
    """The segmenting runs MANO over the whole recording; its 778 vertices a frame must not outlive the read, or a
    release's worth of them sits in memory until it is packed.
    """
    from dataclasses import fields

    import numpy as np

    dataset = _load(raw, mano_dir)
    (clip,) = dataset.read_unit(str(raw / "dataset" / HOT3D_RECORDING))
    for hand in clip.hands.values():
        assert hand.keypoints.shape == (len(hand), 21, 3) and hand.keypoints.dtype == np.float32
        arrays = [getattr(hand, f.name) for f in fields(hand) if isinstance(getattr(hand, f.name), np.ndarray)]
        assert max(a.nbytes / len(hand) for a in arrays if a.ndim > 1) <= 45 * 8  # the finger pose is the widest


def test_a_segment_across_a_missing_frame_is_skipped_rather_than_packed_with_a_jump(tmp_path, mano_dir):
    """Object 222 has no pose on frame 50, so the reader keeps no frame 50: frames 49 and 51 would sit side by side."""
    write_hot3d(tmp_path, mano_dir, T=100, touch=(30, 80), no_object_at=(50,))
    stamp_49 = 1_000_000_000 + 33_333_333 * 49  # the release's timestamp, not an index into the kept frames
    with pytest.raises(RuntimeError, match=rf"'missing frames \(1 gap\(s\), first after timestamp_ns {stamp_49}\)'"):
        _load(tmp_path, mano_dir)
