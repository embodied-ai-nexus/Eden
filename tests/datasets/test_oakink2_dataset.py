"""Tests for ``eden.datasets.oakink2_dataset.OakInk2Dataset`` over a synthetic OakInk2 release."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.datasets.oakink2_dataset import YUP_TO_ZUP, OakInk2Dataset
from tests.support.hand_object_releases import OAKINK2_SEQ, write_oakink2


@pytest.fixture(scope="module")
def release(tmp_path_factory, mano_dir):
    raw = tmp_path_factory.mktemp("oakink2")
    return raw, write_oakink2(raw, mano_dir)


def _load(raw, mano_dir, **options):
    configured = OakInk2Dataset.configure(
        raw_data_path=str(raw),
        body_model_dir=str(mano_dir),
        target_framerate=OakInk2Dataset.source_framerate,
        **options,
    )
    return OakInk2Dataset(options=configured, device=torch.device("cpu"))


def test_a_hand_holds_the_object_its_links_touch_not_the_one_the_program_names(release, mano_dir):
    raw, positions = release
    dataset = _load(raw, mano_dir, sides="right")
    assert dataset.metadata["clip_names"] == [f"{OAKINK2_SEQ}__st00"]
    assert dataset.metadata["objects_right"] == ["obj_b"]  # the program says obj_a
    assert int(dataset.motion_sequences["right_hand"].buffer_length[0]) == 20  # mocap frames 100-119
    # y-up turned z-up.
    np.testing.assert_allclose(
        dataset.motion_sequences["right_hand"].base_pos_w[0][0], YUP_TO_ZUP @ positions["right_wrist"], atol=1e-6
    )


def test_stages_it_cannot_use_are_skipped_with_their_reason(release, mano_dir):
    raw, _ = release
    skipped = dict(_load(raw, mano_dir, sides="right").metadata["skipped"])
    assert skipped == {
        f"{OAKINK2_SEQ}__st01": "several objects in one hand",
        f"{OAKINK2_SEQ}__st02": "no stage interval",
        f"{OAKINK2_SEQ}__st03": "no mocap frame in the stage interval (frames 500-520)",
    }
    # Both hands: the left one holds nothing in stage 0, so nothing is left.
    with pytest.raises(RuntimeError, match=r"every clip .* was skipped .*sides='both' needs each of those hands"):
        _load(raw, mano_dir)


def test_the_recording_gives_every_object_over_all_its_frames(release, mano_dir):
    raw, positions = release
    context = _load(raw, mano_dir, sides="right").recording_object_poses(OAKINK2_SEQ)
    assert sorted(context) == ["obj_a", "obj_b"] and context["obj_a"].shape == (40, 4, 4)
    np.testing.assert_allclose(context["obj_a"][0, :3, 3], YUP_TO_ZUP @ positions["obj_a"])


def test_a_stage_keeps_its_hands_keypoints_not_their_vertices(release, mano_dir):
    from dataclasses import fields

    raw, _ = release
    dataset = _load(raw, mano_dir, sides="right")
    clip = dataset.read_unit(str(raw / "anno_preview" / f"{OAKINK2_SEQ}.pkl"))[0]
    for hand in clip.hands.values():
        arrays = [getattr(hand, f.name) for f in fields(hand) if isinstance(getattr(hand, f.name), np.ndarray)]
        assert max(a.nbytes / len(hand) for a in arrays if a.ndim > 1) <= 45 * 8  # no (T, 778, 3) vertices


def test_a_stage_whose_held_object_has_no_mesh_is_skipped_by_name(tmp_path, mano_dir):
    """Nothing with a mesh to touch, so the hand gets the program's object, which has none either: the stage is
    skipped when the release is read, not accepted for a consumer to trip over.
    """
    raw = tmp_path / "raw"
    write_oakink2(raw, mano_dir)
    for mesh in (raw / "object_repair" / "align_ds").glob("*/model.obj"):
        mesh.unlink()
    with pytest.raises(RuntimeError, match=r"held object has no mesh"):
        _load(raw, mano_dir, sides="right")


def test_the_unloaded_hand_does_not_cost_the_loaded_one_its_stage(tmp_path, mano_dir):
    """The program names two objects for the left hand of stage 0: only a load that includes that hand skips it."""
    write_oakink2(tmp_path, mano_dir, left_objects=("obj_a", "obj_b"))
    assert _load(tmp_path, mano_dir, sides="right").metadata["clip_names"] == [f"{OAKINK2_SEQ}__st00"]
    with pytest.raises(RuntimeError, match=r"every clip .* was skipped .*__st00', 'several objects in one hand'"):
        _load(tmp_path, mano_dir, sides="left")


def test_a_stage_missing_a_mocap_frame_is_skipped_rather_than_packed_with_a_jump(tmp_path, mano_dir):
    write_oakink2(tmp_path, mano_dir, missing=(110,))
    with pytest.raises(RuntimeError, match=r"missing frames \(1 gap\(s\), first after mocap frame 109\)"):
        _load(tmp_path, mano_dir, sides="right")


def test_drop_first_and_last_n_frames_trim_every_clip(release, mano_dir):
    raw, _ = release
    dataset = _load(raw, mano_dir, sides="right", drop_first_n_frames=2, drop_last_n_frames=3)
    hand = dataset.motion_sequences["right_hand"]
    assert int(hand.buffer_length[0]) == 20 - 2 - 3
    untrimmed = _load(raw, mano_dir, sides="right").motion_sequences["right_hand"]
    torch.testing.assert_close(hand.links_pos_w[0][:15], untrimmed.links_pos_w[0][2:17])
    with pytest.raises(RuntimeError, match=r"too short \(20 frames, 0 after drop_first_n_frames=10 / "):
        _load(raw, mano_dir, sides="right", drop_first_n_frames=10, drop_last_n_frames=10)


def test_a_recording_whose_mocap_ids_step_by_two_is_skipped_rather_than_packed_too_fast(tmp_path, mano_dir):
    """The stride is the recording's most common step, so a uniform 2 is not a missing frame every other frame; but
    packed at 120 Hz its clips would play twice too fast.
    """
    write_oakink2(tmp_path, mano_dir, stride=2)
    with pytest.raises(RuntimeError, match=rf"'{OAKINK2_SEQ}', 'mocap frame ids step by 2, not 1 \(120 Hz\)'"):
        _load(tmp_path, mano_dir, sides="right")
