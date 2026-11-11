"""Tests for ``eden.datasets.dex_ycb_dataset.DexYCBDataset`` over a minimal DexYCB tree written to ``tmp_path``.

Pins the opt-in ``retargeted_label`` motion key: ``<seq>/retargeted_<label>.npz``, as written by
``examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py``, becomes a ``robot`` key.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from eden.datasets.dex_ycb_dataset import DexYCBDataset
from tests.datasets.conftest import write_box
from tests.support.hand_object_releases import TEXTURE_RGB, write_textured_box

NUM_FRAMES = 6
NUM_DOFS = 12
LABEL = "xhand1_right"


def _write_sequence(seq_dir, *, retargeted: dict | None):
    """One grasp clip, valid from frame 0, with per-frame MANO keypoints from a master camera ``cam0``."""
    seq_dir.mkdir(parents=True)
    pose_y = np.zeros((NUM_FRAMES, 1, 7), np.float32)
    pose_y[..., 3] = 1.0  # xyzw identity
    pose_y[..., 6] = np.linspace(0.1, 0.2, NUM_FRAMES)[:, None]
    pose_m = np.full((NUM_FRAMES, 1, 51), 0.01, np.float32)
    np.savez(seq_dir / "pose.npz", pose_y=pose_y, pose_m=pose_m)
    (seq_dir / "meta.yml").write_text(
        yaml.safe_dump({"ycb_ids": [2], "ycb_grasp_ind": 0, "mano_sides": ["right"], "extrinsics": "ext1"})
    )
    (seq_dir / "cam0").mkdir()
    for frame in range(NUM_FRAMES):
        np.savez(seq_dir / "cam0" / f"labels_{frame:06d}.npz", joint_3d=np.zeros((1, 21, 3), np.float32))
    if retargeted is not None:
        np.savez(seq_dir / f"retargeted_{LABEL}.npz", **retargeted)


def _retargeted(**overrides):
    payload = {
        "dofs_pos": np.zeros((NUM_FRAMES, NUM_DOFS), np.float32),
        "dofs_name": np.array([f"joint_{i}" for i in range(NUM_DOFS)]),
        "base_pos_w": np.zeros((NUM_FRAMES, 3), np.float32),
        "base_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (NUM_FRAMES, 1)).astype(np.float32),
        "framerate": np.float32(30.0),
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not None}


@pytest.fixture
def dex_ycb(tmp_path):
    """``build(*retargeted_payloads)``: a tree with one clip per payload (``None`` = no retargeted file), loaded."""
    calibration = tmp_path / "calibration" / "extrinsics_ext1"
    calibration.mkdir(parents=True)
    identity_3x4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    (calibration / "extrinsics.yml").write_text(
        yaml.safe_dump({"master": "cam0", "extrinsics": {"apriltag": identity_3x4}})
    )

    def build(*payloads, **options):
        root = tmp_path / "data"
        for i, payload in enumerate(payloads):
            _write_sequence(root / "subject" / f"seq{i}", retargeted=payload)
        options = DexYCBDataset.configure(
            raw_data_path=str(root),
            calibration_dir=str(tmp_path / "calibration"),
            retargeted_label=LABEL,
            ground_initial_pose=False,
            **options,
        )
        return DexYCBDataset(options=options, device=torch.device("cpu"))

    return build


def test_an_over_trimmed_clip_is_reported_as_too_short_by_name(dex_ycb):
    """Not as a clip without valid hand frames: the trim, not the clip, is what is wrong.

    It is the only clip, so nothing is left to load and the short-clip message is the error.
    """
    with pytest.raises(ValueError, match=r"seq0/pose\.npz' has 0 frame\(s\) at \d+ Hz; every clip needs at least 2"):
        dex_ycb(_retargeted(), drop_last_n_frames=NUM_FRAMES)


def test_the_retargeted_robot_becomes_a_third_motion_key(dex_ycb):
    dataset = dex_ycb(_retargeted())

    assert dataset.motion_sequence_keys == ("hand", "object", "robot")
    assert dataset.motion_dofs_name["robot"] == [f"joint_{i}" for i in range(NUM_DOFS)]
    robot, hand = dataset.motion_sequences["robot"], dataset.motion_sequences["hand"]
    assert robot.dofs_pos.shape[1:] == (NUM_DOFS,)
    assert torch.equal(robot.buffer_length, hand.buffer_length)


def test_a_clip_without_the_retargeted_file_is_dropped_with_a_warning(dex_ycb, eden_caplog):
    dataset = dex_ycb(_retargeted(), None)

    assert len(dataset) == 1
    assert dataset.metadata["sequence_ids"] == ["subject/seq0"]
    assert "dropping 1 clip(s)" in eden_caplog.text


def test_no_retargeted_file_anywhere_says_how_to_make_one(dex_ycb):
    with pytest.raises(RuntimeError, match="dex_ycb_position_retargeting.py"):
        dex_ycb(None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_pos_w": np.zeros((NUM_FRAMES - 1, 3), np.float32)}, "base pose shapes"),
        ({"dofs_name": np.array(["only", "three", "names"])}, "dofs_name length"),
    ],
    ids=["frame-count", "name-count"],
)
def test_a_malformed_retargeted_file_is_refused(dex_ycb, overrides, message):
    with pytest.raises(ValueError, match=message):
        dex_ycb(_retargeted(**overrides))


def test_a_file_without_a_framerate_is_read_at_the_dex_ycb_rate(dex_ycb):
    """Older files carry none; any other default would retime the robot out of step with the hand."""
    dataset = dex_ycb(_retargeted(framerate=None))
    assert torch.equal(dataset.motion_sequences["robot"].buffer_length, dataset.motion_sequences["hand"].buffer_length)


def _write_mano_labelled_release(root, mano_dir, *, side="right", frames=6):
    """One clip whose labelled keypoints are the synthetic MANO of its ``pose_m`` (45 PCA coefficients on the mean
    hand, a non-zero shape and translation), in a master camera whose frame is the AprilTag's, as DexYCB labels them.
    """
    import smplx

    from eden.datasets.constants import MANO_KEYPOINTS_NAME
    from eden.datasets.dex_ycb_dataset import _DEX_YCB_TO_EDEN_KEYPOINTS
    from eden.datasets.utils.body_model import mano_keypoints

    rng = np.random.default_rng(3)
    betas = rng.normal(0.0, 1.0, 10)
    pose_m = np.zeros((frames, 1, 51), np.float32)
    pose_m[:, 0, :3] = rng.normal(0.0, 0.5, (frames, 3))
    pose_m[:, 0, 3:48] = rng.normal(0.0, 0.5, (frames, 45))
    pose_m[:, 0, 48:51] = rng.normal(0.0, 0.2, (frames, 3))
    layer = smplx.create(
        str(mano_dir / f"MANO_{side.upper()}.pkl"),
        model_type="mano",
        is_rhand=side == "right",
        use_pca=False,
        flat_hand_mean=False,
    )
    hands_components = np.eye(45)[::-1]  # the synthetic model's PCA basis
    out = layer(
        global_orient=torch.tensor(pose_m[:, 0, :3]),
        hand_pose=torch.tensor(pose_m[:, 0, 3:48] @ hands_components, dtype=torch.float32),
        betas=torch.tensor(np.tile(betas, (frames, 1)), dtype=torch.float32),
        transl=torch.tensor(pose_m[:, 0, 48:51]),
    )
    eden_order = mano_keypoints(out.joints.detach(), out.vertices.detach()).numpy()
    labels = np.zeros_like(eden_order)
    labels[:, list(_DEX_YCB_TO_EDEN_KEYPOINTS)] = eden_order
    assert len(MANO_KEYPOINTS_NAME) == labels.shape[1]

    seq = root / "data" / "subject" / "seq0"
    seq.mkdir(parents=True)
    pose_y = np.zeros((frames, 1, 7), np.float32)
    pose_y[..., 3] = 1.0
    np.savez(seq / "pose.npz", pose_y=pose_y, pose_m=pose_m)
    meta = {"ycb_ids": [2], "ycb_grasp_ind": 0, "mano_sides": [side], "extrinsics": "ext1", "mano_calib": ["subj"]}
    (seq / "meta.yml").write_text(yaml.safe_dump(meta))
    (seq / "cam0").mkdir()
    for frame in range(frames):
        np.savez(seq / "cam0" / f"labels_{frame:06d}.npz", joint_3d=labels[frame][None].astype(np.float32))
    calibration = root / "calibration"
    (calibration / "extrinsics_ext1").mkdir(parents=True)
    identity_3x4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    (calibration / "extrinsics_ext1" / "extrinsics.yml").write_text(
        yaml.safe_dump({"master": "cam0", "extrinsics": {"apriltag": identity_3x4}})
    )
    (calibration / "mano_subj").mkdir()
    (calibration / "mano_subj" / "mano.yml").write_text(yaml.safe_dump({"betas": betas.tolist()}))
    return eden_order, pose_m


def _load(root, **options):
    configured = DexYCBDataset.configure(
        raw_data_path=str(root / "data"),
        calibration_dir=str(root / "calibration"),
        ground_initial_pose=False,
        target_framerate=30.0,
        **options,
    )
    return DexYCBDataset(options=configured, device=torch.device("cpu"))


def test_the_hand_root_is_the_labelled_wrist_not_manos_translation(tmp_path, mano_dir):
    """MANO's translation sits a shape-dependent offset off the wrist (~9 cm on the real model)."""
    keypoints, pose_m = _write_mano_labelled_release(tmp_path, mano_dir)
    hand = _load(tmp_path).motion_sequences["hand"]
    root = hand.base_pos_w.concatenated_tensor.numpy()
    np.testing.assert_allclose(root, keypoints[:, 0], atol=1e-6)
    assert np.linalg.norm(root - pose_m[:, 0, 48:51], axis=1).min() > 1e-3
    assert hand.body_pose_quat is None  # body_model_params is off by default


def test_the_mano_parameters_reproduce_the_labelled_keypoints(tmp_path, mano_dir):
    from eden.datasets.utils.body_model import body_model_forward, mano_keypoints

    keypoints, _ = _write_mano_labelled_release(tmp_path, mano_dir, side="left")
    hand = _load(tmp_path, body_model_params=True, body_model_dir=str(mano_dir)).motion_sequences["hand"]
    assert hand.body_model == ("mano_left",)
    joints, verts = body_model_forward(
        "mano_left",
        mano_dir,
        hand.base_pos_w.concatenated_tensor,
        hand.base_quat_w.concatenated_tensor,
        hand.body_pose_quat.concatenated_tensor,
        hand.body_shape.concatenated_tensor,
    )
    np.testing.assert_allclose(mano_keypoints(joints, verts).numpy(), keypoints, atol=1e-5)


def test_mano_parameters_need_the_model_files(tmp_path, mano_dir):
    _write_mano_labelled_release(tmp_path, mano_dir)
    with pytest.raises(ValueError, match="no body-model directory was given. Set body_model_dir"):
        _load(tmp_path, body_model_params=True)


def test_a_cache_from_before_the_hand_root_fix_is_refused_whatever_the_identity_strictness(tmp_path, mano_dir):
    """Its base_pos_w is MANO's translation, ~9 cm off the wrist: no setting can make it right."""
    _write_mano_labelled_release(tmp_path, mano_dir)
    cache = tmp_path / "cache.pt"
    _load(tmp_path, processed_data_path=str(cache))
    payload = torch.load(cache, weights_only=False)
    del payload["_meta"]["cache_semantics"]  # what a cache written before the fix carries
    torch.save(payload, cache)
    with pytest.raises(
        ValueError, match=r"prepared by an earlier DexYCBDataset .*strict_cache_identity=False does not"
    ):
        _load(tmp_path, processed_data_path=str(cache), strict_cache_identity=False)


def test_an_object_mesh_is_read_once_per_dataset(tmp_path, mano_dir):
    """The import asks for a clip's object mesh once per clip (its floor); the release's file is read the first time."""
    _write_mano_labelled_release(tmp_path, mano_dir)
    model = tmp_path / "models" / "002_master_chef_can" / "textured_simple.obj"
    write_box(model)
    dataset = _load(tmp_path, apply_mesh_correction=False)
    mesh = dataset.object_mesh("002_master_chef_can")
    model.unlink()
    assert dataset.object_mesh("002_master_chef_can") is mesh


def test_an_object_mesh_holds_its_texture_only_when_asked_and_then_uncached(tmp_path, mano_dir):
    """The cached mesh is geometry only, since trimesh decodes a texture as it loads it; the textured one, which only
    an export needs, is read afresh each time and not kept.
    """
    _write_mano_labelled_release(tmp_path, mano_dir)
    write_textured_box(tmp_path / "models" / "002_master_chef_can" / "textured_simple.obj")
    dataset = _load(tmp_path, apply_mesh_correction=False)
    textured = dataset.object_mesh("002_master_chef_can", materials=True)
    assert textured.visual.material.image.convert("RGB").getpixel((0, 0)) == TEXTURE_RGB
    assert dataset.object_mesh("002_master_chef_can", materials=True) is not textured
    plain = dataset.object_mesh("002_master_chef_can")
    image = getattr(plain.visual, "material", None) and plain.visual.material.image
    assert image is None or image.convert("RGB").getpixel((0, 0)) != TEXTURE_RGB
    np.testing.assert_array_equal(plain.vertices, textured.vertices)
    np.testing.assert_array_equal(plain.faces, textured.faces)
