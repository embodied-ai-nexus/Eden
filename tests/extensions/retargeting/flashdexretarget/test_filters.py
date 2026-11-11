"""Tests for ``eden.extensions.retargeting.flashdexretarget.filters``: the reject rules over imported clips."""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.filters import features, reject_reasons
from eden.options.extensions.flashdexretarget import FDRFilterOptions
from tests.extensions.retargeting.flashdexretarget.conftest import pose_track, write_box

FPS = 60.0


def _clip(T: int, wrist: np.ndarray | None = None, height: float = 0.2) -> dict:
    """A 10 cm box held ``height`` up the whole clip, every fingertip on the middle of its +y face (5 cm from each
    of the face's corners, the box mesh's only vertices), the wrist beside it.
    """
    obj = np.tile((0.0, 0.0, height), (T, 1))
    joints = np.repeat(obj[:, None], len(C.MANO_JOINT_NAMES), 1) + (0.0, 0.05, 0.0)  # on the box's +y face
    return {
        "fps": np.float32(FPS),
        "mano_right_wrist_pos": (obj + (0.0, 0.1, 0.0) if wrist is None else wrist).astype(np.float32),
        "mano_right_joints": joints.astype(np.float32),
        "obj_right_pose": pose_track(obj).astype(np.float32),
        "obj_right_id": np.array("box"),
    }


@pytest.fixture
def meshes(tmp_path):
    write_box(tmp_path / "objects" / "box" / "visual.obj")
    return C.ObjectMeshes(tmp_path)


def test_a_held_clip_of_three_seconds_passes_every_rule(meshes):
    cfg = FDRFilterOptions()
    assert reject_reasons(features(_clip(180), meshes, cfg, FPS), cfg) == []


def test_a_clip_shorter_than_min_sec_is_too_short(meshes):
    cfg = FDRFilterOptions()
    assert reject_reasons(features(_clip(120), meshes, cfg, FPS), cfg) == ["too_short"]


def test_a_fingertip_on_the_middle_of_a_face_touches_the_object(meshes):
    """Held 10 cm up, below obj_offdisk_untouched's 15 cm, by fingertips 7 cm from the nearest mesh vertex: only a
    distance to the surface sees the hold that keeps the floating box from being rejected.
    """
    cfg = FDRFilterOptions()
    assert reject_reasons(features(_clip(180, height=0.1), meshes, cfg, FPS), cfg) == []


def test_a_wrist_jump_is_an_acceleration_spike(meshes):
    wrist = np.tile((0.0, 0.1, 0.2), (180, 1))
    wrist[90:, 0] += 0.1  # 10 cm in one frame at 60 Hz: twice max_acc after the 3-frame average
    cfg = FDRFilterOptions()
    assert reject_reasons(features(_clip(180, wrist), meshes, cfg, FPS), cfg) == ["wrist_acc_spike"]
