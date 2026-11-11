"""Tests for ``eden.extensions.pose_estimation.gvhmr.convert``: GVHMR's Y-up estimate to an AMASS-style Z-up clip.

Pins the rotation and its ``J0``-corrected translation, the packed ``poses (T, 165)`` layout, and that the
saved npz is what ``eden retarget soma`` reads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from genesis.utils.geom import quat_to_R, transform_by_quat

from eden.extensions.pose_estimation.gvhmr.convert import (
    BODY_POSE_SLICE,
    ROOT_ORIENT_SLICE,
    SMPLX_POSE_DIM,
    save_amass_npz,
    to_amass_smplx_clip,
    y_up_to_z_up_quat,
    y_up_to_z_up_root,
)
from eden.utils.cli.retarget import _load_soma_clip
from eden.utils.geom import quat_from_axis_angle

ATOL = 1e-5
NUM_FRAMES = 7


def _result(seed: int = 0) -> dict:
    """A GVHMR ``estimate()`` result: plausible shapes and SMPL-X's real rest-pelvis offset."""
    rng = np.random.default_rng(seed)
    return {
        "global": {
            "global_orient": rng.normal(size=(NUM_FRAMES, 3)).astype(np.float32),
            "body_pose": rng.normal(size=(NUM_FRAMES, 63)).astype(np.float32) * 0.2,
            "betas": np.tile(rng.normal(size=(1, 10)).astype(np.float32), (NUM_FRAMES, 1)),
            "transl": rng.normal(size=(NUM_FRAMES, 3)).astype(np.float32),
        },
        "incam": {"global_orient": rng.normal(size=(NUM_FRAMES, 3)).astype(np.float32)},
        "pelvis_j0": np.tile(np.array([[-0.002, -0.223, 0.028]], np.float32), (NUM_FRAMES, 1)),
        "K_fullimg": np.eye(3, dtype=np.float32),
        "output_dir": "/tmp/gvhmr/out",
        "video_fps": 30.0,
    }


def test_the_frame_rotation_takes_y_up_to_z_up():
    up = transform_by_quat(torch.tensor([[0.0, 1.0, 0.0]]), y_up_to_z_up_quat()[None])
    torch.testing.assert_close(up, torch.tensor([[0.0, 0.0, 1.0]]), atol=ATOL, rtol=0)


def test_the_root_orientation_is_left_composed_with_the_frame_rotation():
    result = _result()["global"]
    orient_z, _ = y_up_to_z_up_root(result["global_orient"], result["transl"], np.zeros(3, np.float32))

    r_frame = quat_to_R(y_up_to_z_up_quat()[None])[0]
    r_old = quat_to_R(quat_from_axis_angle(torch.as_tensor(result["global_orient"])))
    r_new = quat_to_R(quat_from_axis_angle(torch.as_tensor(orient_z)))
    torch.testing.assert_close(r_new, r_frame @ r_old, atol=ATOL, rtol=0)


def test_the_world_pelvis_position_rotates_rigidly():
    """SMPL-X rotates the body about ``J0``, so ``J0 + transl`` is what the frame change must rotate."""
    result = _result()
    transl, j0 = result["global"]["transl"], result["pelvis_j0"]
    _, trans_z = y_up_to_z_up_root(result["global"]["global_orient"], transl, j0)

    pelvis = transform_by_quat(torch.as_tensor(j0 + transl), y_up_to_z_up_quat()[None].expand(NUM_FRAMES, 4))
    torch.testing.assert_close(torch.as_tensor(j0 + trans_z), pelvis, atol=ATOL, rtol=0)
    steps = [np.linalg.norm(np.diff(t, axis=0), axis=-1) for t in (transl, trans_z)]
    np.testing.assert_allclose(*steps, atol=ATOL)  # the motion itself is unchanged


def test_dropping_the_pelvis_offset_would_shift_the_clip():
    """GVHMR's own ``R @ transl`` helper is decimetres off."""
    result = _result()
    orient, transl = result["global"]["global_orient"], result["global"]["transl"]
    _, exact = y_up_to_z_up_root(orient, transl, result["pelvis_j0"])
    _, naive = y_up_to_z_up_root(orient, transl, np.zeros(3, np.float32))
    assert np.abs(exact - naive).max() > 0.15


def test_the_amass_payload_layout():
    result = _result()
    clip = to_amass_smplx_clip(result, motion_fps=30.0, source_video="tennis.mp4")
    orient_z, trans_z = y_up_to_z_up_root(
        result["global"]["global_orient"], result["global"]["transl"], result["pelvis_j0"]
    )

    assert clip["poses"].shape == (NUM_FRAMES, SMPLX_POSE_DIM) and clip["poses"].dtype == np.float32
    np.testing.assert_allclose(clip["poses"][:, ROOT_ORIENT_SLICE], orient_z)
    np.testing.assert_allclose(clip["poses"][:, BODY_POSE_SLICE], result["global"]["body_pose"])
    assert not clip["poses"][:, BODY_POSE_SLICE.stop :].any()  # jaw, eyes and hands are not estimated
    np.testing.assert_allclose(clip["trans"], trans_z)
    np.testing.assert_allclose(clip["betas"], result["global"]["betas"].mean(axis=0), atol=ATOL)
    assert (str(clip["gender"]), float(clip["mocap_frame_rate"]), str(clip["surface_model_type"])) == (
        "neutral",
        30.0,
        "smplx",
    )
    np.testing.assert_allclose(clip["gvhmr_transl"], result["global"]["transl"])  # the raw estimate, for provenance
    assert str(clip["source_video"]) == "tennis.mp4"


def test_a_malformed_result_is_refused_naming_the_field():
    result = _result()
    result["global"]["body_pose"] = result["global"]["body_pose"][:, :60]  # SMPL, not SMPL-X
    with pytest.raises(ValueError, match="body_pose"):
        to_amass_smplx_clip(result, motion_fps=30.0)


@pytest.mark.parametrize(
    ("drop", "where"),
    [("K_fullimg", "GVHMR result"), ("pelvis_j0", "GVHMR result"), ("global_orient", "GVHMR result\\['global'\\]")],
)
def test_a_missing_field_is_named_not_a_bare_key_error(drop, where):
    result = _result()
    (result["global"] if drop in result["global"] else result).pop(drop)
    with pytest.raises(ValueError, match=f"{where} is missing .*{drop}"):
        to_amass_smplx_clip(result, motion_fps=30.0)


def test_the_saved_npz_is_readable_by_the_soma_retarget_loader(tmp_path):
    """The npz layout is the integration contract with ``eden retarget soma``."""
    clip = to_amass_smplx_clip(_result(), motion_fps=30.0)
    loaded = _load_soma_clip(save_amass_npz(clip, tmp_path / "clip"), "smplx", None)

    assert (loaded["source_fps"], loaded["gender"]) == (30.0, "neutral")
    for key in ("poses", "trans", "betas"):
        np.testing.assert_allclose(loaded[key], clip[key])
