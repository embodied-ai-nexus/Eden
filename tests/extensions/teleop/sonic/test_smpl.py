"""``eden.extensions.teleop.sonic.smpl``: the SMPL-X tree, frame helpers and forward kinematics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from eden.extensions.teleop.sonic.smpl import (
    SMPL_PARENTS,
    global_quats_to_local_aa,
    process_body_poses,
    remove_smpl_base_rot,
    smpl_root_ytoz_up,
)
from eden.utils.assets import get_asset_path
from tests.support.sonic import IDENTITY, smpl_fk


@pytest.fixture(scope="module")
def rest_joints() -> np.ndarray:
    """The SMPL-X rest skeleton ``(55, 3)``, read from the asset file itself rather than from the FK."""
    smpl_fk()  # skips when the HF-hosted asset cannot be fetched
    return np.load(get_asset_path("sonic/smplx_joints_info.npz"))["J"]


def _service_frame(global_rots: sRot) -> np.ndarray:
    """Build a raw (24, 7) service frame from desired *SMPL* global rotations.

    The pipeline post-multiplies every service rotation by Ry(180°), so the
    service quats are the SMPL rotations pre-corrected by its inverse.
    """
    service = global_rots * sRot.from_euler("y", 180, degrees=True).inv()
    frame = np.zeros((24, 7))
    frame[:, 3:] = service.as_quat(scalar_first=False)  # service is scalar-last
    return frame


def test_parent_tree_is_the_correct_smpl_tree():
    assert SMPL_PARENTS == (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
    assert SMPL_PARENTS[23] == 21  # the reference literal's parent[23]=22 bug is not replicated


def test_root_frame_helpers():
    # Y-up -> Z-up rebases identity onto Rx(+90°).
    rx90 = sRot.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
    np.testing.assert_allclose(smpl_root_ytoz_up(IDENTITY), rx90, atol=1e-12)
    # Removing the base rotation from the base rotation itself yields identity.
    base = np.array([0.5, 0.5, 0.5, 0.5])  # 120° about (1,1,1)
    np.testing.assert_allclose(remove_smpl_base_rot(base), IDENTITY, atol=1e-12)


def test_global_to_local_round_trips():
    rng = np.random.default_rng(11)
    global_rots = sRot.from_quat(rng.normal(size=(24, 4)))  # random unit quats (scipy normalizes)
    frame = _service_frame(global_rots)
    local_aa = global_quats_to_local_aa(frame)
    # Recompose globals from locals down the tree; must reproduce the input.
    recomposed = [sRot.from_rotvec(local_aa[0])]
    for i in range(1, 24):
        recomposed.append(recomposed[SMPL_PARENTS[i]] * sRot.from_rotvec(local_aa[i]))
    for i in range(24):
        diff = recomposed[i].inv() * global_rots[i]
        assert diff.magnitude() == pytest.approx(0.0, abs=1e-9), f"joint {i}"


def test_identity_smpl_pose_maps_to_zero_axis_angle():
    frame = _service_frame(sRot.identity(24))
    np.testing.assert_allclose(global_quats_to_local_aa(frame), 0.0, atol=1e-12)


def test_fk_zero_pose_returns_rest_joints_with_thumb_tips(rest_joints):
    joints = smpl_fk().joints(np.zeros(3), np.zeros(63))
    assert joints.shape == (24, 3)
    rest = rest_joints
    np.testing.assert_allclose(joints[:22], rest[:22], atol=1e-12)
    np.testing.assert_allclose(joints[22], rest[39], atol=1e-12)  # left thumb tip, not left_hand
    np.testing.assert_allclose(joints[23], rest[54], atol=1e-12)


def test_fk_single_joint_rotation_moves_descendants_rigidly(rest_joints):
    fk = smpl_fk()
    rest = rest_joints
    body_pose = np.zeros(63)
    angle_axis = np.array([0.0, np.pi / 2, 0.0])
    body_pose[(18 - 1) * 3 : 18 * 3] = angle_axis  # left elbow (SMPL joint 18)
    joints = fk.joints(np.zeros(3), body_pose)
    # Independent rigid-transform expectation: descendants rotate about joint 18.
    rot = sRot.from_rotvec(angle_axis)
    np.testing.assert_allclose(joints[18], rest[18], atol=1e-12)  # the rotated joint itself stays
    np.testing.assert_allclose(joints[20], rest[18] + rot.apply(rest[20] - rest[18]), atol=1e-9)  # left wrist
    np.testing.assert_allclose(joints[7], rest[7], atol=1e-12)  # unrelated chain untouched


def test_fk_global_orient_rotates_everything(rest_joints):
    fk = smpl_fk()
    rest = rest_joints
    orient = np.array([0.0, 0.0, np.pi / 2])
    joints = fk.joints(orient, np.zeros(63))
    rot = sRot.from_rotvec(orient)
    np.testing.assert_allclose(joints[21], rest[0] + rot.apply(rest[21] - rest[0]), atol=1e-9)


def test_process_body_poses_rebases_the_root_and_expresses_joints_root_locally():
    rng = np.random.default_rng(5)
    global_rots = sRot.from_quat(rng.normal(size=(24, 4)))
    frame = _service_frame(global_rots)
    fk = smpl_fk()
    out = process_body_poses(frame, fk)
    # The SMPL root rotation, rebased Y-up -> Z-up by Rx(+90°) in front; the streamed quat
    # also has SMPL's rest orientation (120° about (1, 1, 1)) removed behind.
    smpl_root = sRot.from_euler("x", 90, degrees=True) * global_rots[0]
    rest_orientation = sRot.from_quat([0.5, 0.5, 0.5, 0.5], scalar_first=True)
    root = sRot.from_quat(out.global_orient_quat, scalar_first=True)
    assert (root.inv() * smpl_root * rest_orientation.inv()).magnitude() == pytest.approx(0.0, abs=1e-9)
    assert out.body_pose.shape == (69,)
    # Root-local joints re-rotated by the streamed root quat equal the world FK output.
    world = fk.joints(smpl_root.as_rotvec(), out.body_pose[:63])
    np.testing.assert_allclose(root.apply(out.smpl_joints_local), world, atol=1e-9)
