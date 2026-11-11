"""Tests for ``eden.extensions.retargeting.otretarget.sources``: the GRAB reader and demonstration mirroring."""

from __future__ import annotations

import os

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from eden.extensions.retargeting.otretarget.retargeter import InteractionObject
from eden.extensions.retargeting.otretarget.sources import (
    SMPLX_FOOT_JOINTS,
    load_grab_sequence,
    mirror_demonstration,
    mirror_object,
    smplx_mirror_maps,
)
from eden.utils.geom import R_to_quat, quat_to_R

#: SMPL-X model files are licence-gated; point this at the folder with SMPLX_NEUTRAL.npz to run those tests.
SMPLX_DIR = os.environ.get("EDEN_SMPLX_DIR", "")


@pytest.fixture(scope="module")
def smplx_neutral():
    path = os.path.join(SMPLX_DIR, "SMPLX_NEUTRAL.npz")
    if not SMPLX_DIR or not os.path.isfile(path):
        pytest.skip(f"SMPL-X model not found at {path} (set EDEN_SMPLX_DIR)")
    return path


def _world(obj: InteractionObject, t: int) -> np.ndarray:
    return np.asarray(obj.mesh.vertices) @ quat_to_R(obj.poses[t, 3:][None])[0].T + obj.poses[t, :3]


def _off_centre_box() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(0.08, 0.05, 0.12))
    box.apply_translation([0.03, -0.01, 0.02])  # not centred on its frame, so a wrong local reflection shows
    return box


def test_mirror_object_reflects_the_world_surface_and_keeps_it_outward():
    rot = Rotation.from_rotvec([[0.4, -0.2, 0.9], [0.1, 0.5, -0.3]]).as_matrix()
    poses = np.concatenate([[[0.5, 0.2, 0.9], [0.6, 0.1, 1.0]], R_to_quat(rot)], axis=1)
    obj = InteractionObject("cup", _off_centre_box(), poses)
    n, point = np.array([0.6, 0.8, 0.0]), np.array([0.4, 0.3, 0.0])
    m = mirror_object(obj, n, point)
    M = np.eye(3) - 2 * np.outer(n, n)
    for t in range(2):
        np.testing.assert_allclose(_world(m, t), (_world(obj, t) - point) @ M + point, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(quat_to_R(m.poses[:, 3:])), 1.0, atol=1e-12)
    assert m.mesh.is_watertight and m.mesh.volume == pytest.approx(obj.mesh.volume, rel=1e-12)


def _write_grab(tmp_path, template, n_frames=8, table_moves=0.0):
    """A minimal raw GRAB file: a standing SMPL-X body, a tilted cup, and GRAB's 5.5 mm table slab."""
    tools = tmp_path / "tools"
    tools.mkdir()
    trimesh.Trimesh(template, process=False).export(tools / "subject.ply")
    _off_centre_box().export(tools / "cup.ply")
    trimesh.creation.box(extents=(0.45, 0.0055, 0.54)).export(tools / "table.ply")
    fullpose = np.zeros((n_frames, 165))
    fullpose[:, 0] = np.pi / 2  # the model's y (up) to the world's z
    obj_aa = np.tile([0.3, 0.2, 0.1], (n_frames, 1))
    obj_t = np.column_stack([np.full(n_frames, 0.5), np.zeros(n_frames), np.linspace(0.85, 1.0, n_frames)])
    tab_aa = np.tile([-np.pi / 2, 0.0, 0.0], (n_frames, 1))
    tab_t = np.column_stack(
        [np.full(n_frames, 0.5) + np.linspace(0, table_moves, n_frames), np.zeros(n_frames), np.full(n_frames, 0.8)]
    )
    path = tmp_path / "cup_lift.npz"
    np.savez(
        path,
        gender=np.array("neutral"),
        framerate=np.array(120.0),
        n_frames=np.array(n_frames),
        obj_name=np.array("cup"),
        body=np.array(
            {"params": {"fullpose": fullpose, "transl": np.zeros((n_frames, 3))}, "vtemp": "tools/subject.ply"}
        ),
        object=np.array({"params": {"global_orient": obj_aa, "transl": obj_t}, "object_mesh": "tools/cup.ply"}),
        table=np.array({"params": {"global_orient": tab_aa, "transl": tab_t}, "table_mesh": "tools/table.ply"}),
    )
    return str(path), obj_aa, obj_t, tab_aa, tab_t


def test_grab_reader_uses_grabs_row_vector_poses_and_a_plate_on_the_table_top(tmp_path, smplx_neutral):
    template = np.load(smplx_neutral)["v_template"]
    path, obj_aa, obj_t, tab_aa, tab_t = _write_grab(tmp_path, template)
    human, (obj, table) = load_grab_sequence(path, str(tmp_path), SMPLX_DIR, step=2, ground=False)
    assert human.num_frames == 4 and human.fps == 60.0
    # GRAB poses objects as v_local @ R(aa) + t.
    for k, raw in enumerate(range(0, 8, 2)):
        expect = np.asarray(obj.mesh.vertices) @ Rotation.from_rotvec(obj_aa[raw]).as_matrix() + obj_t[raw]
        np.testing.assert_allclose(_world(obj, k), expect, atol=1e-12)
    # The plate's top face is the slab's top face; it is 3 cm thick, fixed, and the same every frame.
    slab = np.asarray(trimesh.creation.box(extents=(0.45, 0.0055, 0.54)).vertices)
    slab_w = slab @ Rotation.from_rotvec(tab_aa[0]).as_matrix() + tab_t[0]
    plate_w = _world(table, 0)
    assert table.fixed
    assert plate_w[:, 2].max() == pytest.approx(slab_w[:, 2].max(), abs=1e-7)  # the .ply holds float32
    assert np.ptp(plate_w[:, 2]) == pytest.approx(0.03, abs=1e-12)
    np.testing.assert_allclose(np.ptp(plate_w[:, :2], axis=0), np.ptp(slab_w[:, :2], axis=0), atol=1e-12)
    np.testing.assert_array_equal(table.poses, np.tile(table.poses[0], (4, 1)))
    # The personal template is the body: the standing body is as tall as the template.
    assert np.ptp(human.vertices[0][:, 2]) == pytest.approx(np.ptp(template[:, 1]), rel=1e-6)


def test_grab_grounding_moves_body_object_and_table_together(tmp_path, smplx_neutral):
    path, *_ = _write_grab(tmp_path, np.load(smplx_neutral)["v_template"])
    h0, objs0 = load_grab_sequence(path, str(tmp_path), SMPLX_DIR, ground=False)
    h1, objs1 = load_grab_sequence(path, str(tmp_path), SMPLX_DIR, ground=True)
    lift = h1.vertices[0, 0] - h0.vertices[0, 0]
    assert h1.floor_offset(SMPLX_FOOT_JOINTS) == pytest.approx(0.0, abs=1e-6)
    for a, b in zip(objs0, objs1):
        np.testing.assert_allclose(b.poses[:, :3] - a.poses[:, :3], np.tile(lift, (len(a.poses), 1)), atol=1e-6)


def test_grab_reader_refuses_a_table_that_moves(tmp_path, smplx_neutral):
    path, *_ = _write_grab(tmp_path, np.load(smplx_neutral)["v_template"], table_moves=0.3)
    with pytest.raises(ValueError, match=r"the table of .* moves 17.1 cm over frames \[0, 5\)"):
        load_grab_sequence(path, str(tmp_path), SMPLX_DIR)


def test_smplx_mirror_maps_pair_each_side_with_the_other(smplx_neutral):
    from eden.extensions.retargeting.otretarget.human import SMPLX_JOINT_NAMES

    vmir, jmir = smplx_mirror_maps(smplx_neutral)
    names = list(SMPLX_JOINT_NAMES)
    np.testing.assert_array_equal(vmir[vmir], np.arange(len(vmir)))
    assert names[jmir[names.index("left_thumb3")]] == "right_thumb3"
    assert names[jmir[names.index("pelvis")]] == "pelvis"
    template = np.load(smplx_neutral)["v_template"]
    np.testing.assert_allclose(template[vmir] * [-1, 1, 1], template, atol=1e-4)


def test_mirror_demonstration_gives_the_right_hand_the_left_hands_contact(tmp_path, smplx_neutral):
    from scipy.spatial import cKDTree

    from eden.extensions.retargeting.otretarget.human import smplx_motion

    # A standing body reaching with its left arm toward a cup at its left.
    n_frames = 3
    poses = np.zeros((n_frames, 156))
    poses[:, 0] = np.pi / 2
    poses[:, 3 + 3 * 15 : 3 + 3 * 16] = [0.0, 0.0, -0.9]  # left shoulder (body joint 16 -> body_pose slot 15)
    human = smplx_motion(
        smplx_neutral, poses, np.zeros((n_frames, 3)), np.zeros(10), 30.0, flat_hand_mean=True, up_axis="z"
    )
    wrist = human.joint_pos[0, human.joint_index("left_wrist")]
    cup = InteractionObject(
        "cup", _off_centre_box(), np.tile([*(wrist + [0.0, 0.0, -0.08]), 1, 0, 0, 0], (n_frames, 1))
    )
    mh, (mcup,) = mirror_demonstration(human, [cup], smplx_neutral)

    def hand_gap(h, obj, side):
        parts = [
            h.joint_index(n)
            for n in h.joint_names
            if n.startswith(side)
            and ("wrist" in n or n[len(side) :][:-1] in ("index", "middle", "ring", "pinky", "thumb"))
        ]
        verts = h.vertices[0][np.isin(h.vertex_parts, parts)]
        return cKDTree(_world(obj, 0)).query(verts)[0].min()

    assert hand_gap(mh, mcup, "right_") == pytest.approx(hand_gap(human, cup, "left_"), abs=1e-6)
    assert hand_gap(mh, mcup, "left_") == pytest.approx(hand_gap(human, cup, "right_"), abs=1e-6)
    pelvis = human.joint_index("pelvis")
    np.testing.assert_allclose(mh.joint_pos[0, pelvis], human.joint_pos[0, pelvis], atol=1e-9)  # mirrored in place
    np.testing.assert_allclose(np.linalg.det(mh.joint_rot), 1.0, atol=1e-9)
