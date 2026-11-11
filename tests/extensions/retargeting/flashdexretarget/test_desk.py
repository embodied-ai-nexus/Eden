"""Tests for ``eden.extensions.retargeting.flashdexretarget.desk``: support disks fitted where objects rest."""

from __future__ import annotations

import numpy as np

from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.desk import support_disks
from eden.options.extensions.flashdexretarget import FDRDeskOptions
from tests.extensions.retargeting.flashdexretarget.conftest import pose_track, write_box

FPS = 60.0


def _clip(track: np.ndarray) -> dict:
    T = len(track)
    return {
        "fps": np.float32(FPS),
        "mano_right_wrist_pos": np.zeros((T, 3), np.float32),
        "obj_right_pose": pose_track(track).astype(np.float32),
        "obj_right_id": np.array("box"),
    }


def test_an_object_resting_on_a_table_then_lifted_gets_a_disk_at_the_table_top(tmp_path):
    write_box(tmp_path / "objects" / "box" / "visual.obj", extents=(0.1, 0.06, 0.1))
    rest = np.tile((0.2, 0.1, 0.35), (30, 1))  # bottom face at z = 0.30 for 0.5 s
    lift = np.linspace((0.2, 0.1, 0.35), (0.2, 0.1, 0.6), 30)
    disks = support_disks(_clip(np.concatenate([rest, lift])), C.ObjectMeshes(tmp_path), FDRDeskOptions(), FPS)
    # [x, y, z_top, radius (half the footprint's longer side), height]
    np.testing.assert_allclose(disks, [[0.2, 0.1, 0.30, 0.05, 0.01]], atol=1e-6)


def test_an_object_that_never_rests_gets_no_disk(tmp_path):
    write_box(tmp_path / "objects" / "box" / "visual.obj")
    moving = np.linspace((0.0, 0.0, 0.3), (0.5, 0.0, 0.6), 60)
    disks = support_disks(_clip(moving), C.ObjectMeshes(tmp_path), FDRDeskOptions(), FPS)
    assert disks.shape == (0, 5)
