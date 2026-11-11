"""``DexterHandDataset``, the ConTrack ``.h5`` loader, over clips written to ``tmp_path``."""

import pathlib

import h5py
import numpy as np
import pytest
import torch

import eden
from eden.tasks.contrack.dataset import DexterHandDataset, _motion_dofs_name, _motion_links_name

_SEGMENT_NAMES = [
    "index_proximal",
    "index_middle",
    "index_distal",
    "middle_proximal",
    "middle_middle",
    "middle_distal",
    "pinky_proximal",
    "pinky_middle",
    "pinky_distal",
    "ring_proximal",
    "ring_middle",
    "ring_distal",
    "thumb_proximal",
    "thumb_middle",
    "thumb_distal",
]


def _write_clip(path, *, num_frames=8, is_rhand=False, fps=20.0):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["qpos"] = rng.uniform(-0.5, 0.5, size=(num_frames, 1, 19)).astype(np.float32)
        f["fps"] = np.array([fps], np.float32)
        f["is_rhand"] = np.array([1 if is_rhand else 0], np.uint8)
        f["urdf_name"] = np.array([b"xarm7_xhand.urdf"])
        f["base_translation"] = np.zeros((1, 3), np.float32)
        g = f.create_group("object_tracks/0")
        # Object slides along +x with identity orientation so the object-local
        # -> world contact-point composition is easy to verify.
        trans = np.zeros((num_frames, 3), np.float32)
        trans[:, 0] = np.arange(num_frames, dtype=np.float32) * 0.1
        g["translations"] = trans
        quat = np.zeros((num_frames, 4), np.float32)
        quat[:, 3] = 1.0  # xyzw identity
        g["orientations_xyzw"] = quat
        g["vertices"] = rng.uniform(-0.05, 0.05, size=(8, 3)).astype(np.float32)
        g["faces"] = np.array([[0, 1, 2]], np.uint32)
        c = f.create_group("contacts")
        c["segment_names"] = np.array([s.encode() for s in _SEGMENT_NAMES])
        is_contact = np.zeros((num_frames, 1, 15), np.uint8)
        points = np.full((num_frames, 1, 15, 3), np.nan, np.float32)
        # Frame 0: index_middle + index_distal both in contact (merged into
        # index_rota_link2 = row 1) with distinct points -> weighted mean.
        is_contact[0, 0, 1] = 1
        is_contact[0, 0, 2] = 1
        points[0, 0, 1] = (0.1, 0.0, 0.0)
        points[0, 0, 2] = (0.3, 0.0, 0.0)
        # Frame 1: thumb_proximal only (row 8).
        is_contact[1, 0, 12] = 1
        points[1, 0, 12] = (0.0, 0.2, 0.0)
        c["0/is_contact"] = is_contact
        c["0/points"] = points
    return path


def _load(tmp_path, side="left", target_framerate=20.0, **clip_kwargs):
    clip = _write_clip(tmp_path / "clip.h5", **clip_kwargs)
    options = DexterHandDataset.configure(
        raw_data_path=str(clip),
        processed_data_path=str(tmp_path / "cache.pt"),
        target_framerate=target_framerate,
        side=side,
    )
    return DexterHandDataset(options, device=torch.device("cpu"))


def test_the_flat_qpos_is_split_in_order_across_arm_and_hand(tmp_path):
    """Columns 0:7 drive the arm and 7:19 the hand; contacts ride with the hand, whose links they name."""
    ds = _load(tmp_path)  # target_framerate == the clip fps, so no retiming blends the columns
    with h5py.File(tmp_path / "clip.h5", "r") as f:
        qpos = torch.from_numpy(np.asarray(f["qpos"], np.float32)[:, 0])
    arm, hand, obj = (ds.motion_sequences[key] for key in ("robot", "hand", "object"))

    assert torch.equal(arm.dofs_pos.contiguous(), qpos[:, :7])
    assert torch.equal(hand.dofs_pos.contiguous(), qpos[:, 7:])
    assert hand.contact_points_w.shape[1:] == (10, 1, 3)
    assert hand.contact_points_valid.shape[1:] == (10, 1)
    assert arm.contact_points_w is None and obj.contact_points_w is None
    assert (arm.is_free_floating, hand.is_free_floating, obj.is_free_floating) == (False, False, True)
    assert ds.motion_dofs_name["robot"] == [f"joint{i}" for i in range(1, 8)]
    assert ds.motion_dofs_name["hand"][0] == "left_hand_thumb_bend_joint"
    assert ds.motion_links_name["robot"] == []
    assert ds.motion_links_name["hand"][0] == "left_hand_index_rota_link1"


def test_contact_merge_and_world_composition(tmp_path):
    """Segments merged onto one link average their points, then compose with the object pose of that frame."""
    ds = _load(tmp_path)
    hand = ds.motion_sequences["hand"]
    valid = hand.contact_points_valid.contiguous()
    points = hand.contact_points_w.contiguous()
    # Frame 0: only index_rota_link2 (row 1); merged point = mean of the two
    # segment points (0.2, 0, 0) composed with object pose at frame 0 (origin).
    assert valid[0].sum() == 1 and bool(valid[0, 1, 0])
    assert torch.allclose(points[0, 1, 0], torch.tensor([0.2, 0.0, 0.0]), atol=1e-6)
    # Frame 1: thumb_rota_link1 (row 8); object has moved to x=0.1, identity
    # rotation -> world point = local + (0.1, 0, 0).
    assert valid[1].sum() == 1 and bool(valid[1, 8, 0])
    assert torch.allclose(points[1, 8, 0], torch.tensor([0.1, 0.2, 0.0]), atol=1e-6)


def test_side_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="left-hand recording"):
        _load(tmp_path, side="right")


def test_the_declared_side_default_matches_the_published_clips():
    """``side`` is fixed before any file is opened, so its default is a claim about the published clips."""
    sample_dir = pathlib.Path(eden.__file__).resolve().parents[1] / "examples/projects/mimic/contrack/sample"
    clips = sorted(sample_dir.glob("*.h5"))
    assert clips, f"no sample clips under {sample_dir}"

    for clip in clips:
        with h5py.File(clip, "r") as f:
            side = "right" if bool(np.asarray(f["is_rhand"], np.uint8)[0]) else "left"
        assert side == DexterHandDataset.side, (
            f"{clip.name} is a {side}-hand clip but DexterHandDataset.side defaults to "
            f"{DexterHandDataset.side!r}, so configuring this class over the published clips "
            f"without naming a side would refuse them. Change the default, or stop publishing "
            f"mixed handedness under contrack/."
        )
    # The name axes are derived from the same default, so they have to move with it.
    assert DexterHandDataset.motion_dofs_name == _motion_dofs_name(DexterHandDataset.side)
    assert DexterHandDataset.motion_links_name == _motion_links_name(DexterHandDataset.side)


def test_invalid_side_raises_before_any_file_is_read(tmp_path):
    with pytest.raises(ValueError, match="must be 'left' or 'right'"):
        DexterHandDataset.configure(raw_data_path=str(tmp_path / "nope.h5"), side="ambidextrous")


@pytest.mark.parametrize("side", ["left", "right"])
def test_configured_options_carry_the_requested_side_name_axes(side):
    """Filled from the class attribute, a saved config would name the default side's joints for either hand."""
    options = DexterHandDataset.configure(raw_data_path="clip.h5", side=side)

    assert options.side == side
    assert options.motion_dofs_name["hand"][0] == f"{side}_hand_thumb_bend_joint"
    assert options.motion_links_name["hand"][0] == f"{side}_hand_index_rota_link1"
    assert options.motion_dofs_name["robot"] == [f"joint{i}" for i in range(1, 8)]


def test_instance_axes_follow_side_even_when_the_options_carry_the_class_defaults(tmp_path):
    """A hand-built options object carrying the default side's tables must not override ``side``.

    Derived from the class default, so it models a stale options object on the non-default side, and it
    is this file's only end-to-end right-hand clip while the published clips keep the default at left.
    """
    other = "right" if DexterHandDataset.side == "left" else "left"
    clip = _write_clip(tmp_path / "clip.h5", is_rhand=other == "right")
    options = DexterHandDataset.configure(
        raw_data_path=str(clip),
        processed_data_path=str(tmp_path / "cache.pt"),
        target_framerate=20.0,
        side=other,
    )
    # Exactly what ``_build_config_dict`` fills in for an options object that never went
    # through the deriving ``configure``: the class attributes, i.e. the default side's tables.
    options.motion_dofs_name = _motion_dofs_name(DexterHandDataset.side)
    options.motion_links_name = _motion_links_name(DexterHandDataset.side)

    ds = DexterHandDataset(options, device=torch.device("cpu"))

    assert ds.motion_dofs_name["hand"][0] == f"{other}_hand_thumb_bend_joint"
    assert ds.motion_links_name["hand"][0] == f"{other}_hand_index_rota_link1"


def test_missing_contact_segment_names_fail_loudly(tmp_path):
    """The merge maps 15 annotated segments onto 10 links by name; a missing one used to be a bare ``KeyError``."""
    clip = _write_clip(tmp_path / "clip.h5")
    with h5py.File(clip, "r+") as f:
        del f["contacts/segment_names"]
        f["contacts/segment_names"] = np.array([s.encode() for s in _SEGMENT_NAMES[:-1] + ["typo_distal"]])
    options = DexterHandDataset.configure(
        raw_data_path=str(clip),
        processed_data_path=str(tmp_path / "cache.pt"),
        target_framerate=20.0,
        side="left",
    )
    with pytest.raises(ValueError, match="segment_names is missing"):
        DexterHandDataset(options, device=torch.device("cpu"))


def test_retiming_keeps_contact_alignment(tmp_path):
    ds = _load(tmp_path, target_framerate=50.0)
    arm = ds.motion_sequences["robot"]
    hand = ds.motion_sequences["hand"]
    # Every per-frame field shares the retimed length, across entities too: arm and hand are retimed separately.
    assert hand.contact_points_valid.contiguous().shape[0] == hand.dofs_pos.contiguous().shape[0]
    assert arm.dofs_pos.contiguous().shape[0] == hand.dofs_pos.contiguous().shape[0]
    assert hand.contact_points_valid.dtype == torch.bool


def test_a_missing_raw_path_does_not_suggest_a_download_this_dataset_cannot_do(tmp_path):
    """``raw_data_path=None`` would only lead to "No data source found" here."""
    options = DexterHandDataset.configure(raw_data_path="/no/such/dir", processed_data_path=str(tmp_path / "x.pt"))
    with pytest.raises(FileNotFoundError, match="DexterHandDataset has no source of its own") as raised:
        DexterHandDataset(options=options, device="cpu")
    assert "Pass raw_data_path=None" not in str(raised.value)


def test_a_frame_window_is_refused_before_anything_is_read(tmp_path):
    """The reader never trims, so ``drop_*_n_frames`` would otherwise be ignored yet still key the cache."""
    options = DexterHandDataset.configure(raw_data_path=str(tmp_path / "absent.h5"), drop_first_n_frames=2)
    with pytest.raises(ValueError, match="DexterHandDataset does not trim its clips"):
        DexterHandDataset(options, device=torch.device("cpu"))
