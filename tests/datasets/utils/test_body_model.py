"""Tests for ``eden.datasets.utils.body_model``: a release's MANO parameters <-> the body-model fields <-> joints,
and the linear-blend-skinning rig that runs every model family (checked against the ``smplx`` package).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eden.datasets.utils.body_model import (
    MANO_TIP_VERTICES,
    SkinnedBodyModel,
    axis_angles_to_body_pose_quat,
    body_model_arrays,
    body_model_file,
    body_model_forward,
    mano_keypoints,
    mano_pose_from_release,
    mano_root_offset,
    skinned_body_model,
)
from eden.utils.geom import quat_from_axis_angle

#: SMPL-X model files are licence-gated; point this at the folder with SMPLX_NEUTRAL.npz to run those tests.
SMPLX_DIR = os.environ.get("EDEN_SMPLX_DIR", "")

BETAS = np.linspace(-1.0, 1.0, 10)


def _smplx(mano_dir, **kwargs):
    smplx = pytest.importorskip("smplx", reason="the reference body-model implementation (the dev dependency group)")
    return smplx.create(str(mano_dir / "MANO_RIGHT.pkl"), model_type="mano", is_rhand=True, **kwargs)


def test_a_pca_pose_on_the_mean_hand_expands_to_the_axis_angles_smplx_reads(mano_dir):
    """15 components on the mean hand (HOT3D) -> the same hand as smplx's own PCA layer, relative to the flat hand."""
    coeffs = np.random.default_rng(0).normal(0.0, 0.5, (3, 15))
    pose = mano_pose_from_release(coeffs, mano_dir, "right", num_pca_comps=15, flat_hand_mean=False)
    batch = {"global_orient": torch.zeros(3, 3), "betas": torch.zeros(3, 10)}
    ours = _smplx(mano_dir, use_pca=False, flat_hand_mean=True)(hand_pose=torch.tensor(pose).float(), **batch)
    theirs = _smplx(mano_dir, use_pca=True, num_pca_comps=15, flat_hand_mean=False)(
        hand_pose=torch.tensor(coeffs).float(), **batch
    )
    torch.testing.assert_close(ours.vertices, theirs.vertices)


def test_a_45_component_pca_pose_is_expanded_through_the_basis(mano_dir):
    """The 45 coefficients go through the basis: smplx turns PCA off when asked for all 45 and reads them raw."""
    coeffs = np.linspace(-0.3, 0.3, 45)[None]
    pose = mano_pose_from_release(coeffs, mano_dir, "right", num_pca_comps=45, flat_hand_mean=False)
    basis, mean = np.eye(45)[::-1], np.linspace(-0.2, 0.2, 45)  # the synthetic model's
    np.testing.assert_allclose(pose, coeffs @ basis + mean)


def test_the_root_offset_is_the_shaped_wrist_at_zero_translation(mano_dir):
    out = _smplx(mano_dir, use_pca=False)(betas=torch.tensor(BETAS)[None].float())
    np.testing.assert_allclose(mano_root_offset(mano_dir, "right", BETAS), out.joints[0, 0].detach().numpy(), atol=1e-6)
    assert (
        np.linalg.norm(mano_root_offset(mano_dir, "right", BETAS) - mano_root_offset(mano_dir, "right", 0 * BETAS))
        > 1e-3
    )


def test_the_root_offset_takes_per_frame_shapes(mano_dir):
    betas = np.random.default_rng(4).normal(size=(2, 3, 10))
    offsets = mano_root_offset(mano_dir, "right", betas)
    assert offsets.shape == (2, 3, 3)
    np.testing.assert_allclose(offsets[1, 2], mano_root_offset(mano_dir, "right", betas[1, 2]), atol=1e-6)


def test_forward_puts_the_root_joint_on_the_root_position_and_matches_smplx(mano_dir):
    rng = np.random.default_rng(2)
    orient, fingers = rng.normal(0.0, 0.5, (4, 3)), rng.normal(0.0, 0.3, (4, 45))
    transl = rng.normal(0.0, 0.2, (4, 3))
    reference = _smplx(mano_dir, use_pca=False, flat_hand_mean=True)(
        global_orient=torch.tensor(orient).float(),
        hand_pose=torch.tensor(fingers).float(),
        betas=torch.tensor(np.tile(BETAS, (4, 1))).float(),
        transl=torch.tensor(transl).float(),
    )
    wrist = transl + mano_root_offset(mano_dir, "right", BETAS)
    joints, verts = body_model_forward(
        "mano_right",
        mano_dir,
        torch.tensor(wrist).float(),
        quat_from_axis_angle(torch.tensor(orient)).float(),
        axis_angles_to_body_pose_quat(fingers),
        torch.tensor(np.tile(BETAS, (4, 1))).float(),
    )
    torch.testing.assert_close(joints[:, 0], torch.tensor(wrist).float())
    torch.testing.assert_close(joints, reference.joints, atol=1e-5, rtol=0)
    torch.testing.assert_close(verts, reference.vertices, atol=1e-5, rtol=0)


def test_keypoints_are_the_16_joints_then_the_manopth_fingertips():
    joints, verts = torch.randn(2, 16, 3), torch.randn(2, 778, 3)
    keypoints = mano_keypoints(joints, verts)
    assert keypoints.shape == (2, 21, 3)
    torch.testing.assert_close(keypoints[:, :16], joints)
    torch.testing.assert_close(keypoints[:, 16:], verts[:, list(MANO_TIP_VERTICES)])


def _toy_model(path, n_joints: int, n_dirs: int = 10, shapedirs: np.ndarray | None = None) -> str:
    """A body-model ``.npz`` small enough to write in a test: one vertex on each joint of a chain, no correctives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        v_template=0.1 * np.arange(3 * n_joints, dtype=np.float64).reshape(n_joints, 3),
        f=np.array([[i, i + 1, i + 2] for i in range(n_joints - 2)]),
        weights=np.eye(n_joints),
        kintree_table=np.stack([np.r_[0, np.arange(n_joints - 1)], np.arange(n_joints)]),
        J_regressor=np.eye(n_joints),
        shapedirs=np.zeros((n_joints, 3, n_dirs)) if shapedirs is None else shapedirs,
        posedirs=np.zeros((n_joints, 3, 9 * (n_joints - 1))),
    )
    return str(path)


def test_every_named_family_runs_and_puts_its_root_on_the_root_position(tmp_path):
    """SMPL / SMPL-H / SMPL-X run on the same rig as MANO; only the joint count and the file differ."""
    for name, joints in (("smpl_neutral", 23), ("smplh_male", 51), ("smplx_female", 54)):
        _toy_model(tmp_path / f"{name.split('_')[0].upper()}_{name.split('_')[1].upper()}.npz", joints + 1)
        root = torch.tensor([[0.1, -0.2, 0.9], [0.0, 0.0, 1.0]])
        joints_out, verts = body_model_forward(
            name,
            tmp_path,
            root,
            torch.tensor([[1.0, 0, 0, 0]] * 2),
            torch.tensor([[[1.0, 0, 0, 0]] * joints] * 2),
            torch.zeros(2, 10),
        )
        assert joints_out.shape == (2, joints + 1, 3) and verts.shape == (2, joints + 1, 3)
        torch.testing.assert_close(joints_out[:, 0], root)


def test_a_pose_of_the_wrong_joint_count_and_an_unknown_model_are_refused(tmp_path):
    _toy_model(tmp_path / "SMPL_NEUTRAL.npz", 24)
    args = (torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(1, 21, 4), torch.zeros(1, 10))
    with pytest.raises(ValueError, match=r"smpl_neutral has 23 joints below the root, but body_pose_quat has 21"):
        body_model_forward("smpl_neutral", tmp_path, *args)
    with pytest.raises(ValueError, match="Unknown body_model 'mano'; known: .*'mano_left'"):
        body_model_forward("mano", tmp_path, *args)


@pytest.mark.skipif(not os.path.isfile(os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz")), reason="set EDEN_SMPLX_DIR")
def test_the_smplx_forward_matches_the_smplx_package_on_the_real_model():
    smplx = pytest.importorskip("smplx", reason="the reference body-model implementation (the dev dependency group)")
    rng = np.random.default_rng(5)
    n = 8
    pose, betas, transl = 0.3 * rng.normal(size=(n, 55, 3)), rng.normal(size=10), rng.normal(size=(n, 3))
    ref = smplx.create(
        os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz"),
        model_type="smplx",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=n,
    )
    flat = torch.tensor(pose.reshape(n, -1)).float()
    with torch.no_grad():
        out = ref(
            global_orient=flat[:, :3],
            body_pose=flat[:, 3:66],
            jaw_pose=flat[:, 66:69],
            leye_pose=flat[:, 69:72],
            reye_pose=flat[:, 72:75],
            left_hand_pose=flat[:, 75:120],
            right_hand_pose=flat[:, 120:165],
            betas=torch.tensor(betas).float()[None].expand(n, -1),
            transl=torch.tensor(transl).float(),
        )
    joints, verts = body_model_forward(
        "smplx_neutral",
        SMPLX_DIR,
        out.joints[:, 0],
        quat_from_axis_angle(torch.tensor(pose[:, 0])),
        quat_from_axis_angle(torch.tensor(pose[:, 1:])),
        torch.tensor(np.tile(betas, (n, 1))),
    )
    torch.testing.assert_close(joints, out.joints[:, :55], atol=2e-5, rtol=0)
    torch.testing.assert_close(verts, out.vertices, atol=2e-5, rtol=0)


def test_a_models_npz_is_read_before_its_pkl(tmp_path):
    _toy_model(tmp_path / "SMPL_NEUTRAL.npz", 24)
    (tmp_path / "SMPL_NEUTRAL.pkl").write_bytes(b"not read")
    assert body_model_file(tmp_path, "smpl_neutral") == tmp_path / "SMPL_NEUTRAL.npz"


def test_a_pickle_holding_chumpy_arrays_reads_without_chumpy(tmp_path, monkeypatch):
    """The MANO, SMPL and SMPL-H .pkl files pickle ``shapedirs`` as chumpy objects; no extra is needed to read them."""
    from tests.utils.test_chumpy_pickle import Chumpy, chumpy_pickle

    with np.load(_toy_model(tmp_path / "toy.npz", 3)) as toy:
        arrays = {k: toy[k] for k in toy.files}
    path = tmp_path / "MANO_RIGHT.pkl"
    path.write_bytes(chumpy_pickle(monkeypatch, {**arrays, "shapedirs": Chumpy("Ch", x=arrays["shapedirs"])}))
    data = body_model_arrays(path)
    assert type(data["shapedirs"]) is np.ndarray
    np.testing.assert_array_equal(data["shapedirs"], arrays["shapedirs"])


def test_pickled_sparse_and_numpy_arrays_load_as_plain_arrays(tmp_path):
    from scipy.sparse import csc_matrix

    with np.load(_toy_model(tmp_path / "toy.npz", 3)) as toy:
        arrays = {k: toy[k] for k in toy.files}
    with open(tmp_path / "m.pkl", "wb") as f:
        pickle.dump({**arrays, "J_regressor": csc_matrix(np.eye(3))}, f)
    data = body_model_arrays(tmp_path / "m.pkl")
    assert type(data["J_regressor"]) is np.ndarray
    np.testing.assert_array_equal(data["J_regressor"], np.eye(3))


def test_a_file_without_the_model_arrays_is_refused_naming_them(tmp_path):
    """A converted .npz under other key names is read before the release's .pkl: say what it lacks, not KeyError."""
    np.savez(tmp_path / "MANO_RIGHT.npz", vertices=np.zeros((3, 3)), shape_basis=np.zeros((3, 3, 10)))
    with pytest.raises(ValueError, match=r"MANO_RIGHT.npz lacks \['v_template', 'shapedirs'.*it holds \['shape_basis'"):
        mano_root_offset(tmp_path, "right", np.zeros(10))


def test_a_replaced_model_file_is_read_again(tmp_path):
    path = tmp_path / "SMPL_NEUTRAL.npz"
    _toy_model(path, 24)
    assert not body_model_arrays(path)["shapedirs"].any()
    _toy_model(path, 24, shapedirs=np.ones((24, 3, 10)))
    later = path.stat().st_mtime_ns + 10**9  # the cache keys on (mtime, size); a same-size rewrite keeps the size
    os.utime(path, ns=(later, later))
    assert body_model_arrays(path)["shapedirs"].all()


def test_the_official_and_amass_file_names_are_found(tmp_path):
    _toy_model(tmp_path / "official" / "SMPLH_male.npz", 52)  # the official download's lower-case gender
    assert body_model_file(tmp_path / "official", "smplh_male").name == "SMPLH_male.npz"
    _toy_model(tmp_path / "amass" / "female" / "model.npz", 52)
    assert body_model_file(tmp_path / "amass", "smplh_female") == tmp_path / "amass" / "female" / "model.npz"
    with pytest.raises(
        FileNotFoundError,
        match=r"SMPLH_NEUTRAL\.pkl missing \(also looked for SMPLH_NEUTRAL\.npz, "
        r"SMPLH_neutral\.npz, SMPLH_neutral\.pkl, neutral/model\.npz\)",
    ):
        body_model_file(tmp_path / "amass", "smplh_neutral")


def test_frames_of_different_shapes_pose_as_their_own_calls_would(mano_dir):
    rng = np.random.default_rng(7)
    n = 6
    args = (
        torch.tensor(rng.normal(size=(n, 3))).float(),
        quat_from_axis_angle(torch.tensor(rng.normal(size=(n, 3)))),
        axis_angles_to_body_pose_quat(rng.normal(0, 0.3, (n, 45))),
    )
    shapes = torch.tensor(np.repeat(rng.normal(size=(3, 10)), 2, axis=0)).float()[[0, 2, 1, 4, 3, 5]]
    joints, verts = body_model_forward("mano_right", mano_dir, *args, shapes)
    for i in range(n):
        ji, vi = body_model_forward("mano_right", mano_dir, *(a[i : i + 1] for a in args), shapes[i : i + 1])
        torch.testing.assert_close(joints[i : i + 1], ji)
        torch.testing.assert_close(verts[i : i + 1], vi)


def test_more_betas_than_the_hand_has_shape_components_are_named(mano_dir):
    with pytest.raises(ValueError, match=r"16 betas per hand, but .*MANO_RIGHT.pkl has 10 shape components"):
        mano_root_offset(mano_dir, "right", np.zeros(16))


@pytest.mark.parametrize(
    ("env", "name", "model_type", "n_body"),
    [("EDEN_SMPL_DIR", "smpl_neutral", "smpl", 23), ("EDEN_SMPLH_DIR", "smplh_male", "smplh", 51)],
)
def test_the_smpl_and_smplh_forwards_match_the_smplx_package_on_real_models(env, name, model_type, n_body):
    model_dir = os.environ.get(env, "")
    try:
        path = body_model_file(model_dir or None, name)
    except (ValueError, FileNotFoundError):
        pytest.skip(f"set {env} to a folder holding the {name} model file")
    smplx = pytest.importorskip("smplx", reason="the reference body-model implementation (the dev dependency group)")
    rng = np.random.default_rng(8)
    n = 6
    pose, betas, transl = 0.3 * rng.normal(size=(n, n_body + 1, 3)), rng.normal(size=10), rng.normal(size=(n, 3))
    if model_type == "smplh":
        # The official SMPL-H file carries no hand-PCA basis, which smplx reads even with PCA off: hand it unused
        # identity bases and flat means (flat_hand_mean=True never adds them).
        data = {
            **body_model_arrays(path),
            **{f"hands_{k}{s}": v for s in "lr" for k, v in (("components", np.eye(45)), ("mean", np.zeros(45)))},
        }
        ref = smplx.SMPLH(
            str(path),
            data_struct=smplx.utils.Struct(**data),
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            batch_size=n,
        )
    else:
        # smplx reads a .pkl with a plain pickle.load, which needs chumpy (no longer installed): hand it the arrays
        # Eden read without it. tests/utils/test_chumpy_pickle.py checks smplx's own read, through the stand-ins.
        ref = smplx.SMPL(
            str(path), data_struct=smplx.utils.Struct(**body_model_arrays(path)), num_betas=10, batch_size=n
        )
    flat = torch.tensor(pose.reshape(n, -1)).float()
    kwargs = {"global_orient": flat[:, :3], "body_pose": flat[:, 3:72] if model_type == "smpl" else flat[:, 3:66]}
    if model_type == "smplh":
        kwargs |= {"left_hand_pose": flat[:, 66:111], "right_hand_pose": flat[:, 111:156]}
    with torch.no_grad():
        out = ref(**kwargs, betas=torch.tensor(betas).float()[None].expand(n, -1), transl=torch.tensor(transl).float())
    joints, verts = body_model_forward(
        name,
        model_dir,
        out.joints[:, 0],
        quat_from_axis_angle(torch.tensor(pose[:, 0])),
        quat_from_axis_angle(torch.tensor(pose[:, 1:])),
        torch.tensor(np.tile(betas, (n, 1))),
    )
    torch.testing.assert_close(joints, out.joints[:, : n_body + 1], atol=2e-5, rtol=0)
    torch.testing.assert_close(verts, out.vertices, atol=2e-5, rtol=0)


def test_a_shaped_rig_regresses_its_joints_from_the_shaped_template(tmp_path):
    dirs = np.random.default_rng(6).normal(size=(24, 3, 10))
    path = _toy_model(tmp_path / "SMPL_NEUTRAL.npz", 24, shapedirs=dirs)
    betas = np.linspace(-1.0, 1.0, 10)
    rig = skinned_body_model(path, "smpl", betas)
    expected = 0.1 * np.arange(72.0).reshape(24, 3) + dirs @ betas
    np.testing.assert_allclose(rig.rest_vertices, expected)
    np.testing.assert_allclose(rig.bind_pos, expected)  # J_regressor = I: joint k is vertex k
    with pytest.raises(ValueError, match="joints per frame; the rig has 24"):
        rig.forward(np.tile(np.eye(3), (1, 23, 1, 1)), np.zeros((1, 3)))


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


def test_bind_relative_identity_returns_the_rest_mesh():
    rig = _two_bone_rig()
    verts, pos, rot = rig.forward(np.tile(np.eye(3), (1, 2, 1, 1)), np.zeros((1, 3)))
    np.testing.assert_allclose(verts[0], rig.rest_vertices, atol=1e-6)
    np.testing.assert_allclose(pos[0, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rot[0], rig.bind_rot)


def test_child_rotation_turns_only_its_vertices_about_its_joint():
    rig = _two_bone_rig()
    local = np.tile(np.eye(3), (1, 2, 1, 1))
    # 90 deg about the child's own bind x axis, which is world +y.
    local[0, 1] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    verts, _, _ = rig.forward(local, np.zeros((1, 3)))
    np.testing.assert_allclose(verts[0, 0], [0.1, 0.0, 0.5], atol=1e-6)  # root bone untouched
    # (0.1, 0, 0.5) above the joint, rotated +90 deg about world +y -> (0.5, 0, -0.1) from the joint.
    np.testing.assert_allclose(verts[0, 1], [0.5, 0.0, 0.9], atol=1e-6)


def test_an_axis_angle_pose_on_the_flat_hand_needs_no_model_file():
    pose = np.random.default_rng(1).normal(size=(2, 45))
    np.testing.assert_array_equal(
        mano_pose_from_release(pose, None, "right", num_pca_comps=None, flat_hand_mean=True), pose
    )


def test_missing_model_files_say_where_to_get_them(tmp_path):
    with pytest.raises(ValueError, match="no body-model directory was given. Set body_model_dir"):
        mano_root_offset(None, "right", BETAS)
    with pytest.raises(
        FileNotFoundError, match=r"MANO_LEFT\.pkl missing \(also looked for MANO_LEFT\.npz\).*mano\.is\.tue"
    ):
        mano_root_offset(tmp_path, "left", BETAS)
