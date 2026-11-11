"""Tests for ``eden.extensions.retargeting.otretarget.grasp.HandGrasp`` on a G1 that demonstrates to itself.

The identity rig has no hands, so a link stands in for a palm (the left elbow or wrist link) and the links below it
for its fingers.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import trimesh

pytest.importorskip("proxsuite")  # the QP solver; in the ``ot-retargeter`` extra
pytest.importorskip("threadpoolctl")

from eden.extensions.retargeting.otretarget import InteractionObject, OTRetargeter
from eden.extensions.retargeting.otretarget.retargeter import Targets
from eden.utils.geom import R_to_quat, quat_to_R
from tests.extensions.retargeting.otretarget.conftest import identity_options


def _grasp_options(kin, palm="left_elbow_link", **grasp):
    """The identity setup with ``palm`` as the one 'palm' of a ``hand_grasp`` add-on (the links below it its fingers)."""
    opts = identity_options(kin)
    opts["addons"]["hand_grasp"] = {"palms": {palm: palm}, **grasp}
    return opts


def _box(n, pose=(0.4, 0.0, 0.6), size=0.2):
    return InteractionObject("box", trimesh.creation.box((size,) * 3), np.tile([*pose, 1, 0, 0, 0], (n, 1)))


@pytest.mark.parametrize(
    ("grasp", "match"),
    [
        ({}, "palms is empty"),
        ({"palms": {"no_such_link": "pelvis"}}, "no_such_link"),
        (
            {"palms": {"left_elbow_link": "left_elbow_link"}, "fingertips": {"right_wrist_yaw_link": "x"}},
            "hangs from none",
        ),
        ({"palms": {"left_elbow_link": "left_elbow_link"}, "closure": 0.05}, "closure needs `fingertips`"),
        ({"palms": {"left_elbow_link": "left_elbow_link"}, "rest_palm_normal": (0.0, 0.0, -2.0)}, "rest_palm_normal"),
        ({"palms": {"left_elbow_link": "left_elbow_link"}, "contact": -0.01}, "contact"),
        ({"palms": {"left_elbow_link": "left_elbow_link"}, "grasp_closure": 0.05}, "grasp_closure"),
    ],
)
def test_hand_grasp_misconfiguration_is_refused_at_construction(env, kin, grasp, match):
    opts = identity_options(kin)
    opts["addons"]["hand_grasp"] = grasp
    with pytest.raises(ValueError, match=match):
        OTRetargeter(env, OTRetargeter.configure(**opts))


@pytest.mark.parametrize(
    ("palm_d", "tip_d", "closes"),
    [
        (0.0, 0.03, True),  # palm on the object, the finger 3 cm off: the finger closes
        (0.0, 0.09, False),  # the finger beyond the closure band stays as demonstrated
        (0.05, 0.03, False),  # neither the palm nor two pads touch: no grasp
    ],
)
def test_grasp_closure_closes_the_fingers_of_a_grasping_hand(env, kin, self_retarget, palm_d, tip_d, closes):
    # The demonstrated triples are set by hand: the 'palm' (left knee link) part at palm_d from an object, the
    # 'finger' (left ankle roll link) part at tip_d, every skin facing the object. While the hand grasps, the finger's
    # probes whose sources face the object are marked for closure when the finger is within the band.
    _, human, _, _ = self_retarget
    opts = _grasp_options(
        kin, "left_knee_link", fingertips={"left_ankle_roll_link": "left_ankle_roll_link"}, closure=0.06
    )
    ret = OTRetargeter(env, OTRetargeter.configure(**opts))
    ret._setup(human, [])
    grasp = ret.addon("hand_grasp")
    part = ret.human_probes.part
    knee, ankle = human.joint_index("left_knee_link"), human.joint_index("left_ankle_roll_link")
    dist = np.full(len(part), 1.0)
    dist[part == knee] = palm_d
    dist[part == ankle] = tip_d
    skin = ret.human_probes.normals(human.vertices[0].astype(np.float64))
    h = Targets(dist, np.zeros((len(part), 3)), -skin)  # outward object normal = -skin: every skin faces the object
    src = np.maximum(ret.src, 0)
    robot_t = Targets(np.where(ret.src >= 0, dist[src], np.inf), np.zeros((len(src), 3)), -skin[src])
    grasp._grasp_closure(human, 0, h, robot_t, np.eye(3))
    finger = (ret.src >= 0) & (part[src] == ankle)
    palm = (ret.src >= 0) & (part[src] == knee)
    assert robot_t.closed[finger].all() == closes and robot_t.closed[finger].any() == closes
    # The palm's touching probes are marked whenever the hand grasps (they gate the closure).
    assert robot_t.closed[palm].all() == (palm_d <= grasp.contact)
    assert not robot_t.closed[~(finger | palm)].any()


def test_the_closure_stage_moves_only_the_hand_joints(env, kin, self_retarget):
    # The 'hand' below the left elbow link is the wrist: closing it may move the wrist joints only, never the arm, the
    # root or the object.
    _, human, qs, _ = self_retarget
    n = human.num_frames
    box = _box(n)
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin)))
    ret._setup(human, [box])
    grasp = ret.addon("hand_grasp")
    wrist = {kin.var_names[i] for i in np.flatnonzero(grasp._hand_cols)}
    assert wrist == {f"left_wrist_{j}_joint" for j in ("roll", "pitch", "yaw")}
    pos, rot = box.poses[1:2, :3].copy(), quat_to_R(box.poses[1:2, 3:])
    frame = ret._frame_inputs(human, 1, qs[0], pos, rot)
    q = qs[1].copy()
    q[kin.n_base_q :] += 0.03
    q_c, _ = grasp._close_hands(q, pos, rot, frame, None)
    held = kin.n_base_q + np.flatnonzero(~grasp._hand_cols[kin.n_base_v :])
    np.testing.assert_array_equal(q_c[held], q[held])
    np.testing.assert_array_equal(q_c[: kin.n_base_q], q[: kin.n_base_q])


@pytest.mark.parametrize("robot_side, facing", [(1.0, 1), (-1.0, 0)])
def test_a_cold_start_counts_the_palms_that_hold_the_object_on_the_human_side(
    env, kin, self_retarget, robot_side, facing
):
    # The demonstrated box is 6 cm in front of the human palm along its palm normal; the robot palm counts when its box
    # is in front of it too, not when it is behind its back (the start the first frame must not pick).
    _, human, qs, _ = self_retarget
    n = human.num_frames
    box = _box(n, (0.0, 0.0, 0.0), 0.04)
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin)))
    ret._setup(human, [box])
    grasp = ret.addon("hand_grasp")
    li, jw = kin.link_index("left_elbow_link"), human.joint_index("left_elbow_link")
    normal = np.asarray(grasp.rest_palm_normal)
    centroid = human.vertices[0][human.vertex_parts == jw].astype(np.float64).mean(axis=0)
    local = grasp._palm_probes(li)
    kin.set_configuration(qs[0])
    link_pos, link_rot = kin.link_poses()
    n_h = human.joint_rot[0, jw] @ human.rest_joint_rot[jw].T @ normal
    n_r = link_rot[li] @ ret._tpose_link_rot[li].T @ normal
    eye = np.eye(3)[None]
    demo_pos = (centroid + 0.06 * n_h)[None]
    robot_pos = (link_pos[li] + link_rot[li] @ local + robot_side * 0.06 * n_r)[None]
    palms = [(li, local, centroid)]
    assert grasp._palms_facing(palms, human, demo_pos, eye, qs[0], robot_pos, eye) == facing


def test_a_reapproach_frees_one_arm_and_its_hand_only(env, kin):
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin, side_reapproach=True)))
    cols, finger_q, links = ret.addon("hand_grasp")._side_dofs(kin.link_index("left_elbow_link"))
    names = {kin.var_names[i] for i in np.flatnonzero(cols)}
    assert {"left_shoulder_pitch_joint", "left_elbow_joint", "left_wrist_roll_joint"} <= names
    assert all(n.startswith(("left_shoulder", "left_elbow", "left_wrist")) for n in names), names
    fingers = {kin.var_names[i - kin.n_base_q + kin.n_base_v] for i in finger_q}
    assert fingers == {n for n in names if n.startswith("left_wrist")}
    assert kin.link_index("left_elbow_link") in links and kin.link_index("right_elbow_link") not in links


def test_the_palm_goal_is_the_demonstrated_palm_when_the_object_did_not_move(env, kin, self_retarget):
    # The goal carries the human palm's place and orientation from the demonstrated object pose onto the robot's. With
    # both poses equal it is the human palm itself: its part's centroid, and its orientation through the T-pose offset
    # (the identity rig's link frames).
    _, human, qs, _ = self_retarget
    n = human.num_frames
    box = _box(n)
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin)))
    ret._setup(human, [box])
    grasp = ret.addon("hand_grasp")
    li, jw = kin.link_index("left_elbow_link"), human.joint_index("left_elbow_link")
    pos, rot = box.poses[3:4, :3], quat_to_R(box.poses[3:4, 3:])
    p, R = grasp._palm_goal(li, human, 3, pos, rot, pos, rot)
    np.testing.assert_allclose(
        p, human.vertices[3][human.vertex_parts == jw].astype(np.float64).mean(axis=0), atol=1e-9
    )
    kin.set_configuration(qs[3])
    _, link_rot = kin.link_poses()
    np.testing.assert_allclose(R, link_rot[li], atol=1e-4)
    # Moving the robot's object moves the goal with it, rigidly.
    shift = np.array([[0.1, -0.05, 0.02]])
    p2, R2 = grasp._palm_goal(li, human, 3, pos, rot, pos + shift, rot)
    np.testing.assert_allclose(p2, p + shift[0], atol=1e-9)
    np.testing.assert_allclose(R2, R, atol=1e-12)


def test_the_palm_goal_term_matches_the_cost_it_models(env, kin, self_retarget):
    # The palm goal's gradient is the derivative of its cost, as for every other term (``test_retargeter``).
    _, human, qs, _ = self_retarget
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin, "left_wrist_yaw_link")))
    ret._setup(human, [])
    li = kin.link_index("left_wrist_yaw_link")
    kin.set_configuration(qs[1])
    link_pos, link_rot = kin.link_poses()
    local = ret.addon("hand_grasp")._palm_probes(li)
    goal = (
        li,
        local,
        link_pos[li] + link_rot[li] @ local + 0.03,
        link_rot[li] @ quat_to_R(np.array([0.98, 0.2, 0, 0])),
        1e3,
    )
    frame = ret._frame_inputs(human, 1, qs[0], np.zeros((0, 3)), np.zeros((0, 3, 3)))
    frame.extra["palm_goals"] = [goal]
    q = qs[1].copy()
    base = ret._linearize(q, np.zeros((0, 3)), np.zeros((0, 3, 3)), frame)
    assert base.terms["palm_goal"] > 0
    s = np.random.default_rng(1).normal(size=ret.n_x)
    s /= np.abs(s).max()
    eps = 1e-3  # the test env runs Genesis forward kinematics in float32
    plus, minus = (
        ret._linearize(kin.integrate(q, e * s), np.zeros((0, 3)), np.zeros((0, 3, 3)), frame, frozen=base.frozen)
        for e in (eps, -eps)
    )
    fd = (plus.terms["palm_goal"] - minus.terms["palm_goal"]) / (2 * eps)
    np.testing.assert_allclose(2.0 * base.term_grads["palm_goal"] @ s, fd, rtol=5e-2, atol=1e-3)


def test_the_reapproach_start_puts_the_palm_on_its_goal_with_the_arm_alone(env, kin, self_retarget):
    # The last wrist link as the 'palm': the seven arm joints reach a 6-DoF goal (the elbow link alone has four).
    _, human, qs, _ = self_retarget
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin, "left_wrist_yaw_link")))
    ret._setup(human, [])
    grasp = ret.addon("hand_grasp")
    li = kin.link_index("left_wrist_yaw_link")
    cols, finger_q, links = grasp._side_dofs(li)
    local = grasp._palm_probes(li)
    kin.set_configuration(qs[2])
    link_pos, link_rot = kin.link_poses()
    p_tgt = link_pos[li] + link_rot[li] @ local + np.array([0.03, 0.02, -0.02])
    q = grasp._reapproach_start(
        qs[2], li, local, cols, finger_q, links, p_tgt, link_rot[li], np.zeros(3), np.zeros((0, 3)), np.zeros((0, 3, 3))
    )
    kin.set_configuration(q)
    link_pos, link_rot = kin.link_poses()
    assert np.linalg.norm(link_pos[li] + link_rot[li] @ local - p_tgt) < 0.005
    moved = np.flatnonzero(np.abs(q - qs[2]) > 1e-9)
    allowed = set((kin.n_base_q + np.flatnonzero(cols[kin.n_base_v :])).tolist())
    assert set(moved.tolist()) <= allowed, [kin.var_names[i - kin.n_base_q + kin.n_base_v] for i in moved]


def test_a_palm_sunk_into_a_part_still_reads_the_part_in_front(env, kin, self_retarget):
    # A 2 cm slab in front of the human 'palm' (the left elbow link), with the palm centre 2 mm inside its near face:
    # the nearest surface point is that face, 2 mm behind the centre, but the slab's material is in front. The side
    # reads the material (_palm_offsets), as for a pole pressed into a sunk OMOMO palm.
    _, human, qs, _ = self_retarget
    n = human.num_frames
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin)))
    li, jw = kin.link_index("left_elbow_link"), human.joint_index("left_elbow_link")
    t = 2
    c_h = human.vertices[t][human.vertex_parts == jw].astype(np.float64).mean(axis=0)
    n_h = human.joint_rot[t, jw] @ human.rest_joint_rot[jw].T @ np.asarray(ret.addon("hand_grasp").rest_palm_normal)
    # The slab's thin (x) axis along the palm normal.
    b = np.cross(n_h, [0.0, 0.0, 1.0] if abs(n_h[2]) < 0.9 else [1.0, 0.0, 0.0])
    b /= np.linalg.norm(b)
    R = np.stack([n_h, b, np.cross(n_h, b)], axis=1)
    centre = c_h + 0.008 * R[:, 0]
    slab = InteractionObject(
        "slab", trimesh.creation.box((0.02, 0.2, 0.2)), np.tile(np.r_[centre, R_to_quat(R)], (n, 1))
    )
    ret._setup(human, [slab])
    grasp = ret.addon("hand_grasp")
    pos, rot = slab.poses[t : t + 1, :3], quat_to_R(slab.poses[t : t + 1, 3:])
    kin.set_configuration(qs[t])
    link_pos, link_rot = kin.link_poses()
    h_off, h_depth, _ = grasp._palm_offsets(
        li, grasp._palm_probes(li), c_h, human, t, pos, rot, link_pos, link_rot, pos, rot
    )
    assert -0.003 < h_depth < 0.0  # the centre is 2 mm inside
    assert h_off > 0.005, h_off  # the material is in front


def test_a_failed_clip_leaves_no_palm_goal_on_the_next(env, kin, self_retarget):
    # Per-clip state lives on the add-on and is reset by ``setup``: a clip that raised mid-way (the CLI goes on to the
    # next clip with the same retargeter) hands nothing to the next one.
    _, human, _, _ = self_retarget
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin, side_reapproach=True)))
    ret._setup(human, [])
    grasp = ret.addon("hand_grasp")
    grasp._side_state[kin.link_index("left_elbow_link")] = {"last": 3, "tries": 1, "floor": 3, "goal_from": 3}
    ret._setup(replace(human), [])
    assert grasp._side_state == {}
    frame = ret._frame_inputs(human, 1, None, np.zeros((0, 3)), np.zeros((0, 3, 3)), forward=True)
    assert "palm_goals" not in frame.extra


def test_a_palm_of_several_joints_reads_all_their_skin(env, kin, self_retarget):
    # On the SOMA rig the palm's skin belongs to the hand joint and the four metacarpals: the palm's place is the
    # centroid of all of it, and its frame stays the first joint's.
    _, human, _, _ = self_retarget
    n = human.num_frames
    box = _box(n)
    palm = ["left_elbow_link", "left_wrist_roll_link"]
    ret = OTRetargeter(env, OTRetargeter.configure(**_grasp_options(kin, palm="left_elbow_link")))
    ret.addon("hand_grasp").palms = {"left_elbow_link": palm}
    ret._setup(human, [box])
    grasp = ret.addon("hand_grasp")
    li = kin.link_index("left_elbow_link")
    assert grasp._palm_joint[li] == human.joint_index(palm[0])
    pos, rot = box.poses[3:4, :3], quat_to_R(box.poses[3:4, 3:])
    p, _ = grasp._palm_goal(li, human, 3, pos, rot, pos, rot)
    skin = np.isin(human.vertex_parts, [human.joint_index(j) for j in palm])
    np.testing.assert_allclose(p, human.vertices[3][skin].astype(np.float64).mean(axis=0), atol=1e-9)
