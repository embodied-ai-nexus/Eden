"""Tests for ``eden.datasets.utils.bvh``: parsing and forward kinematics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from eden.datasets.utils.bvh import forward_kinematics, forward_kinematics_matrices, parse_bvh
from eden.utils.geom import quat_to_R

# A root and one child, both with all six channels (as a SOMA-rig BVH has); frame 1 stretches the child's
# bone from its 10 cm offset to 25 cm through its position channels.
_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Spine
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0.0 5.0 0.0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
1.0 2.0 3.0 0.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 0.0
1.0 2.0 3.0 90.0 30.0 0.0 0.0 25.0 0.0 0.0 0.0 45.0
"""


def test_position_channels_replace_the_offset_on_every_joint(tmp_path):
    path = tmp_path / "two.bvh"
    path.write_text(_BVH)
    pos, rot = forward_kinematics_matrices(parse_bvh(str(path)))
    np.testing.assert_allclose(pos[0], [[1.0, 2.0, 3.0], [1.0, 12.0, 3.0]], atol=1e-12)
    # Rotation channels compose in their listed order: Zrotation Yrotation Xrotation is Rz @ Ry @ Rx.
    hips = Rotation.from_euler("z", 90, degrees=True) * Rotation.from_euler("y", 30, degrees=True)
    np.testing.assert_allclose(rot[1, 0], hips.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(pos[1, 1], [1.0, 2.0, 3.0] + hips.apply([0.0, 25.0, 0.0]), atol=1e-12)
    spine = hips * Rotation.from_euler("x", 45, degrees=True)
    np.testing.assert_allclose(rot[1, 1], spine.as_matrix(), atol=1e-12)


def test_forward_kinematics_quaternions_are_the_matrices_in_wxyz(tmp_path):
    path = tmp_path / "two.bvh"
    path.write_text(_BVH)
    bvh = parse_bvh(str(path))
    pos, quat = forward_kinematics(bvh)
    pos_m, rot = forward_kinematics_matrices(bvh)
    np.testing.assert_array_equal(pos, pos_m)
    np.testing.assert_allclose(quat_to_R(quat), rot, atol=1e-12)
