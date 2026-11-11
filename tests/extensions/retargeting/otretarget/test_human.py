"""Tests for ``eden.extensions.retargeting.otretarget.human``: the skinned demonstrator surface."""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from eden.datasets.utils.body_model import SkinnedBodyModel
from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion, smplx_motion

#: SMPL-X model files are licence-gated; point this at the folder with SMPLX_NEUTRAL.npz to run those tests.
SMPLX_DIR = os.environ.get("EDEN_SMPLX_DIR", "")


def _two_bone_rig() -> SkinnedBodyModel:
    # A root at the origin and a child 1 m up +z; vertices rigidly on one bone each. The child's bind
    # frame is rotated 90 deg about z, so bind-relative rotations are not world rotations.
    verts = np.array([[0.1, 0.0, 0.5], [0.1, 0.0, 1.5], [0.0, 0.1, 1.5]])
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    bind_rot = np.stack([np.eye(3), Rotation.from_euler("z", 90, degrees=True).as_matrix()])
    return SkinnedBodyModel(
        verts,
        np.array([[0, 1, 2]]),
        weights,
        np.array([-1, 0]),
        bind_rot,
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        ["root", "child"],
    )


def _motion(n_frames: int = 3, lift: float = 0.0) -> HumanSurfaceMotion:
    rig = _two_bone_rig()
    verts, pos, rot = rig.forward(np.tile(np.eye(3), (n_frames, 2, 1, 1)), np.tile([0.0, 0.0, lift], (n_frames, 1)))
    return HumanSurfaceMotion(
        verts,
        rig.faces,
        rig.skin_weights,
        rig.joint_names,
        pos,
        rot,
        rig.rest_vertices,
        rig.bind_pos,
        rig.bind_rot,
        30.0,
    )


def test_floor_offset_reads_a_constant_float_and_translated_removes_it():
    motion = _motion(lift=0.04)
    offset = motion.floor_offset(("root",))
    np.testing.assert_allclose(offset, 0.54, atol=1e-6)  # lowest root vertex at z = 0.5 + lift
    grounded = motion.translated(np.array([0.0, 0.0, -offset]))
    np.testing.assert_allclose(grounded.floor_offset(("root",)), 0.0, atol=1e-6)
    np.testing.assert_allclose(grounded.joint_pos[:, 0, 2], -0.5, atol=1e-6)


def test_shape_mismatch_names_the_field():
    motion = _motion()
    with pytest.raises(ValueError, match="joint_rot"):
        HumanSurfaceMotion(
            motion.vertices,
            motion.faces,
            motion.skin_weights,
            motion.joint_names,
            motion.joint_pos,
            motion.joint_rot[:, :1],
            motion.rest_vertices,
            motion.rest_joint_pos,
            motion.rest_joint_rot,
            30.0,
        )


@pytest.fixture(scope="module")
def smplx_neutral():
    path = os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz")
    if not SMPLX_DIR or not os.path.isfile(path):
        pytest.skip(f"SMPL-X model not found at {path} (set EDEN_SMPLX_DIR)")
    return path


def test_smplx_rest_pose_is_a_grounded_t_pose_facing_forward(smplx_neutral):
    m = smplx_motion(smplx_neutral, np.zeros((1, 156)), np.zeros((1, 3)), np.zeros(10), 30.0, flat_hand_mean=True)
    rest = m.rest_vertices
    np.testing.assert_allclose(rest[:, 2].min(), 0.0, atol=1e-9)
    # Arms out along the lateral axis (y): wrists far apart at shoulder height, nose toward +x.
    lw, rw = m.rest_joint_pos[m.joint_index("left_wrist")], m.rest_joint_pos[m.joint_index("right_wrist")]
    assert lw[1] > 0.6 and rw[1] < -0.6 and abs(lw[2] - rw[2]) < 0.02
    head = rest[m.vertex_parts == m.joint_index("head")]
    assert head[:, 0].max() > m.rest_joint_pos[m.joint_index("head"), 0] + 0.05


def test_smplx_y_up_world_becomes_z_up(smplx_neutral):
    trans = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])  # 0.5 m up in a y-up world
    m = smplx_motion(smplx_neutral, np.zeros((2, 156)), trans, np.zeros(10), 30.0, flat_hand_mean=True, up_axis="y")
    np.testing.assert_allclose(m.joint_pos[1, 0] - m.joint_pos[0, 0], [0.0, 0.0, 0.5], atol=1e-9)


def _two_armed(n_frames: int = 2) -> tuple[HumanSurfaceMotion, np.ndarray, np.ndarray]:
    # A torso with a left arm (+y) and a right arm (-y), in the rest frame (facing +x, z-up), with a vertex on
    # each arm and one on the torso, plus each arm's mirror partner. The left arm is raised by a turn about an
    # axis that is not in the sagittal plane, so a reflection that forgets to swap sides or to re-handle the
    # rotation shows up.
    names = ["torso", "left_arm", "right_arm"]
    bind_pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.2, 1.4], [0.0, -0.2, 1.4]])
    verts = np.array([[0.05, 0.0, 1.2], [0.0, 0.6, 1.4], [0.0, -0.6, 1.4], [0.02, 0.6, 1.45], [0.02, -0.6, 1.45]])
    weights = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 3], [0, 4, 2]])
    rig = SkinnedBodyModel(verts, faces, weights, np.array([-1, 0, 0]), np.tile(np.eye(3), (3, 1, 1)), bind_pos, names)
    local = np.tile(np.eye(3), (n_frames, 3, 1, 1))
    local[:, 1] = Rotation.from_rotvec([0.6, 0.3, -0.4]).as_matrix()
    v, p, r = rig.forward(local, np.tile([1.0, 2.0, 1.0], (n_frames, 1)))
    motion = HumanSurfaceMotion(v, faces, weights, names, p, r, verts, bind_pos, rig.bind_rot, 30.0)
    return motion, np.array([0, 2, 1, 4, 3]), np.array([0, 2, 1])


def test_mirrored_puts_the_left_arm_motion_on_the_right_arm():
    motion, vmir, jmir = _two_armed()
    n, point = np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 1.0])  # the torso's own sagittal plane
    M = np.diag([1.0, -1.0, 1.0])
    m = motion.mirrored(vmir, jmir, M, n, point)  # bind frames are the rest frame: lateral axis y

    def reflect(x):
        return (x - point) @ M + point

    # The right arm's vertex is where the left arm's was, reflected; the left arm now hangs in its rest pose.
    np.testing.assert_allclose(m.vertices[:, 2], reflect(motion.vertices[:, 1]), atol=1e-6)
    np.testing.assert_allclose(m.vertices[:, 1], reflect(motion.vertices[:, 2]), atol=1e-6)
    np.testing.assert_allclose(m.joint_rot[:, 2], M @ motion.joint_rot[:, 1] @ M, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(m.joint_rot), 1.0, atol=1e-12)
    np.testing.assert_allclose(m.joint_rot[:, 1], np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-12)
    assert m.vertex_parts.tolist() == [0, 1, 2, 1, 2]  # every vertex keeps its anatomical part
    # Outward normals survive: face 0 of the image is the reflection of face 0, and its normal is the reflected
    # normal (a bare reflection would flip it inward).
    np.testing.assert_allclose(
        _normal(m.vertices[0], m.faces[0]), _normal(motion.vertices[0], motion.faces[0]) @ M, atol=1e-6
    )


def _normal(v, f):
    a, b, c = v[f]
    nrm = np.cross(b - a, c - a)
    return nrm / np.linalg.norm(nrm)


def test_mirrored_twice_is_the_original():
    motion, vmir, jmir = _two_armed()
    args = (vmir, jmir, np.diag([1.0, -1.0, 1.0]), np.array([0.3, 1.0, 0.0]), np.array([0.5, 1.0, 0.0]))
    back = motion.mirrored(*args).mirrored(*args)
    np.testing.assert_allclose(back.vertices, motion.vertices, atol=1e-6)
    np.testing.assert_allclose(back.joint_rot, motion.joint_rot, atol=1e-12)
    np.testing.assert_allclose(back.rest_vertices, motion.rest_vertices, atol=1e-12)
    np.testing.assert_array_equal(back.faces, motion.faces)


def test_mirrored_rejects_a_vertex_map_that_is_not_a_permutation():
    motion, vmir, jmir = _two_armed()
    with pytest.raises(ValueError, match="vertex_mirror"):
        motion.mirrored(np.array([0, 2, 2, 4, 3]), jmir, np.diag([1.0, -1.0, 1.0]), [0, 1, 0], [0, 0, 0])


def test_smplx_v_template_replaces_the_mean_shape(smplx_neutral):
    template = np.load(smplx_neutral)["v_template"] * 1.1  # a 10 % taller person
    m = smplx_motion(
        smplx_neutral,
        np.zeros((1, 156)),
        np.zeros((1, 3)),
        np.zeros(10),
        30.0,
        flat_hand_mean=True,
        up_axis="z",
        v_template=template,
    )
    ref = smplx_motion(
        smplx_neutral, np.zeros((1, 156)), np.zeros((1, 3)), np.zeros(10), 30.0, flat_hand_mean=True, up_axis="z"
    )
    np.testing.assert_allclose(
        np.ptp(m.rest_vertices[:, 2]), 1.1 * np.ptp(ref.rest_vertices[:, 2]), rtol=1e-6
    )  # float32 template
    with pytest.raises(ValueError, match="v_template"):
        smplx_motion(smplx_neutral, np.zeros((1, 156)), np.zeros((1, 3)), np.zeros(10), 30.0, v_template=template[:10])


def _toy_smplx_file(path, n_dirs: int) -> str:
    """An SMPL-X-shaped model file small enough to write in a test: one vertex on each of the 55 joints, a chain,
    and ``n_dirs`` blend shapes where shape direction k lifts vertex k (the expression half lifts the last vertex).
    """
    n = 55
    shapedirs = np.zeros((n, 3, n_dirs))
    n_shape = 300 if n_dirs >= 400 else n_dirs // 2
    for k in range(n_dirs):
        shapedirs[k % n if k < n_shape else n - 1, 2, k] = 0.01
    np.savez(
        path,
        v_template=0.1 * np.arange(3 * n, dtype=np.float64).reshape(n, 3),
        f=np.array([[i, i + 1, i + 2] for i in range(n - 2)]),
        weights=np.eye(n),
        kintree_table=np.stack([np.r_[0, np.arange(n - 1)], np.arange(n)]),
        J_regressor=np.eye(n),
        shapedirs=shapedirs,
        posedirs=np.zeros((n, 3, 9 * (n - 1))),
        hands_meanl=np.zeros(45),
        hands_meanr=np.zeros(45),
    )
    return str(path)


def test_smplx_betas_past_a_v1_0_models_shape_block_are_refused(tmp_path):
    # A v1.0 file stacks 10 shape and 10 expression blend shapes: betas 11-16 of an AMASS-style fit read past the
    # shape block drove the expression ones as body shape.
    v1_0 = _toy_smplx_file(tmp_path / "SMPLX_V10.npz", 20)
    betas = np.r_[np.zeros(10), np.full(6, 0.5)]
    with pytest.raises(
        ValueError, match=r"16 betas, but .*SMPLX_V10.npz has 10 shape components .*SMPL-X use the v1.1"
    ):
        smplx_motion(v1_0, np.zeros((1, 156)), np.zeros((1, 3)), betas, 30.0, up_axis="z")
    zero_tail = smplx_motion(v1_0, np.zeros((1, 156)), np.zeros((1, 3)), np.zeros(16), 30.0, up_axis="z")
    v1_1 = smplx_motion(
        _toy_smplx_file(tmp_path / "SMPLX_V11.npz", 400), np.zeros((1, 156)), np.zeros((1, 3)), betas, 30.0, up_axis="z"
    )
    moved = np.linalg.norm(v1_1.rest_vertices - zero_tail.rest_vertices, axis=1)  # the rest pose is re-framed
    np.testing.assert_allclose(moved[10:16], 0.005)  # betas 11-16 shape vertices 10-15, nothing else
    np.testing.assert_allclose(np.delete(moved, np.s_[10:16]), 0.0, atol=1e-12)


def test_smplx_unknown_up_axis_is_refused_before_skinning(tmp_path):
    with pytest.raises(ValueError, match="up_axis='x'; expected 'y' or 'z'"):
        smplx_motion(
            str(tmp_path / "never_read.npz"), np.zeros((1, 156)), np.zeros((1, 3)), np.zeros(10), 30.0, up_axis="x"
        )
