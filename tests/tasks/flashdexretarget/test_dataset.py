"""Tests for ``eden.tasks.flashdexretarget.dataset``: a ``motion.pt`` as Eden motion sequences."""

from __future__ import annotations

import pytest
import torch

from eden.extensions.retargeting.flashdexretarget.clips import MANO_JOINT_NAMES
from eden.tasks.flashdexretarget.dataset import FlashDexRetargetDataset
from eden.utils.geom import quat_to_R


def _load(path) -> FlashDexRetargetDataset:
    return FlashDexRetargetDataset(FlashDexRetargetDataset.configure(raw_data_path=str(path)), torch.device("cpu"))


def test_each_clip_becomes_robot_mano_and_object_sequences_labelled_by_its_object(motion_file):
    packed = torch.load(motion_file, weights_only=True)
    ds = _load(motion_file)
    assert ds.motion_sequence_keys == ("robot", "mano_right", "object_right")
    assert ds.motion_labels("objects") == ["box_small*1", "box_large*1"]
    first = int(packed["motion_num_frames"][0])
    names = list(packed["mano_right_joint_names"])
    order = [names.index(n) for n in MANO_JOINT_NAMES]
    # BatchMotionSequence stores clips back to back; the first clip's rows are the motion file's first rows.
    mano = ds.motion_sequences["mano_right"]
    torch.testing.assert_close(
        mano.links_pos_w.concatenated_tensor[:first], packed["mano_right_joints"][:first][:, order].float()
    )
    torch.testing.assert_close(
        mano.base_pos_w.concatenated_tensor[:first], packed["mano_right_wrist_pos"][:first].float()
    )
    torch.testing.assert_close(
        quat_to_R(mano.base_quat_w.concatenated_tensor[:first]),
        packed["mano_right_wrist_rot"][:first].float(),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        ds.motion_sequences["robot"].dofs_pos.concatenated_tensor[:first], packed["joint_pos"][:first].float()
    )


def test_a_dataset_directory_names_the_motion_file_to_pass(motion_file):
    with pytest.raises(ValueError, match=r"is a directory; pass the motion file inside it, .*motion\.pt"):
        _load(motion_file.parent)
