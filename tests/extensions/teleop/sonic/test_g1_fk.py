"""Unit tests for the Genesis-backed G1 key-frame FK used by 3-point calibration.

Pinned against a frozen MuJoCo reference (the implementation this replaced):
same model file, root at the origin. The goldens and the random-config parity
sweep catch any frame-convention drift — Genesis's base-link re-origin, link
ordering, or quaternion convention. The FK is the module-scoped ``g1_key_frame_fk``
fixture (tests/extensions/conftest.py): one scene build for the file.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_genesis


def test_zero_pose_key_frames_are_pinned(g1_key_frame_fk):
    poses = g1_key_frame_fk.key_frame_poses(np.zeros(29))
    # Golden values from the GEAR-SONIC model (root at origin, q = 0),
    # computed with MuJoCo; the wrist offsets [0.18, -/+0.025, 0] are folded in.
    # The two wrist goldens mirror each other, so they pin the left/right symmetry too.
    np.testing.assert_allclose(poses["left_wrist"]["position"], [0.379769, 0.123627, 0.095221], atol=1e-5)
    np.testing.assert_allclose(poses["right_wrist"]["position"], [0.379769, -0.123617, 0.095221], atol=1e-5)
    np.testing.assert_allclose(poses["torso"]["position"], [-0.003964, 0.0, 0.404], atol=1e-5)
    np.testing.assert_allclose(poses["torso"]["orientation_wxyz"], [1, 0, 0, 0], atol=1e-4)


def test_joint_motion_moves_the_wrist_frame(g1_key_frame_fk):
    q = np.zeros(29)
    q[15] = 0.5  # left shoulder pitch (MJCF index 15)
    moved = g1_key_frame_fk.key_frame_poses(q)["left_wrist"]["position"]
    np.testing.assert_allclose(moved, [0.2396, 0.167257, -0.0568], atol=1e-4)
    # The right arm is untouched.
    rest = g1_key_frame_fk.key_frame_poses(np.zeros(29))["right_wrist"]["position"]
    np.testing.assert_allclose(g1_key_frame_fk.key_frame_poses(q)["right_wrist"]["position"], rest, atol=1e-6)


# ---------------------------------------------------------------------------
# parity against the frozen MuJoCo implementation
# ---------------------------------------------------------------------------


class _MujocoRefFk:
    """The MuJoCo implementation this module replaced, frozen as the reference."""

    def __init__(self) -> None:
        import mujoco

        from eden.extensions.teleop.sonic.g1_fk import G1_KEY_FRAMES
        from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1Dofs29Dex3
        from eden.utils.assets import get_asset_path

        self._mujoco = mujoco
        self._key_frames = G1_KEY_FRAMES
        self._model = mujoco.MjModel.from_xml_path(get_asset_path(G1Dofs29Dex3().file))
        self._data = mujoco.MjData(self._model)
        self._body_ids = {
            key: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body)
            for key, (body, _) in G1_KEY_FRAMES.items()
        }
        self._joint_qpos_adr = [
            int(self._model.jnt_qposadr[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in G1_DOFS29_NAME
        ]

    def key_frame_poses(self, q_mj29):
        from scipy.spatial.transform import Rotation as sRot

        self._data.qpos[:] = 0.0
        self._data.qpos[3] = 1.0
        for adr, value in zip(self._joint_qpos_adr, np.asarray(q_mj29, dtype=np.float64).reshape(29)):
            self._data.qpos[adr] = value
        self._mujoco.mj_kinematics(self._model, self._data)
        result = {}
        for key, (_, offset) in self._key_frames.items():
            body_id = self._body_ids[key]
            quat = np.asarray(self._data.xquat[body_id])
            rot = sRot.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            result[key] = {
                "position": np.asarray(self._data.xpos[body_id]) + rot @ offset,
                "orientation_wxyz": quat.copy(),
            }
        return result


def test_matches_mujoco_reference_on_random_configs(g1_key_frame_fk):
    pytest.importorskip("mujoco")
    ref = _MujocoRefFk()
    lo = ref._model.jnt_range[1:30, 0]  # joint 0 is the free base
    hi = ref._model.jnt_range[1:30, 1]
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(lo, hi)
        got, want = g1_key_frame_fk.key_frame_poses(q), ref.key_frame_poses(q)
        for key in ("left_wrist", "right_wrist", "torso"):
            np.testing.assert_allclose(
                got[key]["position"], want[key]["position"], atol=1e-5, err_msg=f"{key} position"
            )
            q_got, q_want = got[key]["orientation_wxyz"], want[key]["orientation_wxyz"]
            if float(q_got @ q_want) < 0.0:  # q and -q are the same rotation
                q_got = -q_got
            np.testing.assert_allclose(q_got, q_want, atol=1e-5, err_msg=f"{key} orientation")
