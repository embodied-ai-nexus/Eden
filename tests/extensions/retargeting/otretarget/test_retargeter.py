"""Tests for ``eden.extensions.retargeting.otretarget.retargeter`` (the method) on a G1 that demonstrates to itself."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import trimesh

pytest.importorskip("proxsuite")  # the QP solver; in the ``ot-retargeter`` extra
pytest.importorskip("threadpoolctl")

from scipy.spatial.transform import Rotation

from eden.extensions.retargeting.otretarget import InteractionObject, OTRetargeter
from eden.extensions.retargeting.otretarget.kinematics import RigKinematics
from eden.extensions.retargeting.otretarget.metrics import evaluate_retarget
from eden.utils.geom import quat_to_R
from eden.utils.mesh import get_link_mesh
from tests.extensions.retargeting.otretarget.conftest import (
    FEET,
    LEG_DOFS,
    identity_options,
    q_index,
    robot_demonstration,
    squat,
)


def test_robot_reproduces_its_own_demonstration(kin, self_retarget):
    _, _, qs, result = self_retarget
    cols = [q_index(kin, n) for n in LEG_DOFS]
    err = np.abs(result.qpos[:, cols] - qs[:, cols])
    assert err.mean() < 0.05, err.mean(axis=0)
    np.testing.assert_allclose(result.qpos[:, :3], qs[:, :3], atol=0.02)


def test_feet_stay_on_the_demonstrated_floor_contact(self_retarget):
    ret, human, _, result = self_retarget
    metrics = evaluate_retarget(ret, human, [], result)
    assert metrics["ground/recall"] > 0.9 and metrics["ground/precision"] > 0.9
    assert metrics["ground/depth_err_mm"] < 5.0
    assert metrics["penetration_p95_mm"] < 2.0


def test_a_foot_pivoting_in_place_counts_as_skate(env, kin, self_retarget):
    # The robot yaws 0.3 rad about the lowest vertex of its left sole, so that foot pivots in place. Some vertex of
    # any rigid motion barely moves: the least moving one read the pivot as no skate. The contact vertices sweep.
    _, human, qs, result = self_retarget
    opts = identity_options(kin)
    opts["correspondences"] = [g for g in opts["correspondences"] if not g[0][0].startswith("right_ankle")]
    ret = OTRetargeter(env, OTRetargeter.configure(**opts))
    ret._setup(human, [])  # the channels and groups evaluate_retarget reads; the left foot is the only foot unit
    kin.set_configuration(qs[0])
    pos, rot = kin.link_poses()
    foot = kin.link_index(FEET[0])
    link = next(link for link in kin.entities[0].links if link.name == FEET[0])
    sole = pos[foot] + np.asarray(get_link_mesh(link, use_visual_mesh=True).vertices) @ rot[foot].T
    pivot = sole[np.argmin(sole[:, 2])]
    qpos = np.tile(qs[0], (human.num_frames, 1))
    root_rot = Rotation.from_quat(qs[0, [4, 5, 6, 3]])
    for t, yaw in enumerate(np.linspace(0.0, 0.3, human.num_frames)):
        turn = Rotation.from_euler("z", yaw)
        qpos[t, :3] = turn.apply(qs[0, :3] - pivot) + pivot
        qpos[t, [4, 5, 6, 3]] = (turn * root_rot).as_quat()
    turn_rate = 0.3 / (human.num_frames - 1) * result.fps  # rad/s
    skate = evaluate_retarget(ret, human, [], replace(result, qpos=qpos))["skate_cm_s"]
    assert skate > 100.0 * turn_rate * 0.03, skate  # the sole's contact vertices sit centimetres from the pivot


def test_every_warm_frame_converges_within_its_budget(self_retarget):
    # Each frame stops on a vanishing step, not on the iteration cap. A scalar witness distance (a cone at its
    # target) made Gauss-Newton overshoot and flip back every iterate, so frames ended wherever the budget
    # cut the 2-cycle: frame-to-frame jitter.
    ret, _, _, result = self_retarget
    assert (result.info["iterations"][1:] < ret.iterations).all(), result.info["iterations"]


@pytest.mark.parametrize(("slack", "bounded"), [(1.0, True), (0.0, False)])
def test_the_root_moves_no_faster_than_the_pelvis_plus_its_slack(env, kin, self_retarget, slack, bounded):
    # The demonstrated surface (feet included) jumps 20 cm sideways at frame 4 while the pelvis stays put: the
    # interaction terms pull the whole robot after it. The root may then gain at most `slack / fps` per frame on the
    # pelvis; with the bound off it covers several centimetres more in that frame.
    _, human, _, _ = self_retarget
    jump = np.zeros((human.num_frames, 1, 3))
    jump[4:, 0, 1] = 0.2
    shifted = replace(human, vertices=human.vertices + jump.astype(human.vertices.dtype))
    opts = identity_options(kin, root_speed_slack=slack)
    result = OTRetargeter(env, OTRetargeter.configure(**opts)).retarget_motion(shifted, progress=False)
    pelvis = human.joint_pos[:, human.joint_index("pelvis")]
    gain = np.abs(np.diff(result.qpos[:, :3], axis=0) - np.diff(pelvis, axis=0)).max()
    limit = 1.0 / human.fps + 1e-9
    assert (gain <= limit) == bounded, gain


def test_the_cold_start_is_the_demonstrated_posture_standing_on_the_floor(env, kin):
    # Started from straight legs at the pelvis, a robot in a demonstrated squat began with its legs 30 cm through the
    # floor, and the cold first frame of OMOMO "pick up the fallen tripod" ended kneeling or lying (body orientation
    # 70-100 deg off), a posture no later frame left.
    qs = squat(kin, n_frames=12, depth=0.85)[::-1]
    human = robot_demonstration(kin, qs)
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    ret._setup(human, [])
    q0 = ret._initial_configuration(human, np.zeros((0, 3)), np.zeros((0, 3, 3)))
    cols = [q_index(kin, n) for n in LEG_DOFS]
    np.testing.assert_allclose(q0[cols], qs[0, cols], atol=0.02)
    kin.set_configuration(q0)
    lowest = ret.robot_probes.world(*kin.link_poses())[:, 2].min()
    assert abs(lowest - human.vertices[0][:, 2].min()) < 1e-6
    np.testing.assert_allclose(q0[:3], qs[0, :3], atol=0.01)


def test_a_cold_start_pull_brings_a_link_point_to_its_target(env, kin):
    # ``pulls`` add point targets to the posture fit (the hand-grasp add-on pulls palms onto the human palms); the root
    # may leave the pelvis to reach them.
    qs = squat(kin, n_frames=4, depth=0.3)
    human = robot_demonstration(kin, qs)
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    ret._setup(human, [])
    li = kin.link_index("left_wrist_yaw_link")
    free = ret._initial_configuration(human, np.zeros((0, 3)), np.zeros((0, 3, 3)))
    kin.set_configuration(free)
    target = kin.link_poses()[0][li] + np.array([0.04, 0.0, 0.03])
    pulled = ret._initial_configuration(
        human, np.zeros((0, 3)), np.zeros((0, 3, 3)), pulls=[(li, np.zeros(3), target)], pull_weight=1000.0
    )
    kin.set_configuration(pulled)
    pos = kin.link_poses()[0][li]
    # On the floor again after the fit, so the target's height is not kept exactly.
    assert np.linalg.norm((pos - target)[:2]) < 0.01


def test_joint_limits_narrow_a_joint_range_and_never_widen_it(env, kin):
    # The retargeted motion keeps to the narrower range (Wuji fingers flex 60 deg backwards mechanically; pressed onto an
    # object they did, where the demonstrators' fingers stop at 14 deg).
    mechanical = RigKinematics(env, ["robot"])
    i = mechanical.var_names.index("left_knee_joint") - mechanical.n_base_v
    ret = OTRetargeter(
        env, OTRetargeter.configure(**identity_options(kin, joint_limits={"left_knee_joint": [0.5, 99.0]}))
    )
    assert ret.kin.joint_lower[i] == 0.5
    assert ret.kin.joint_upper[i] == mechanical.joint_upper[i]


def test_result_splits_into_entity_qpos(self_retarget):
    _, _, qs, result = self_retarget
    assert result.entity_qpos["robot"].shape == qs.shape
    np.testing.assert_allclose(np.linalg.norm(result.qpos[:, 3:7], axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (dict(correspondences=[[["pelvis"], ["no_such_link"]]]), "no_such_link"),
        (dict(style_links={"torso_link": "torso_link", "pelvis": "pelvis"}), "rig root 'pelvis'"),
        (dict(tpose={"no_such_joint": 1.0}), "no_such_joint"),
        (dict(kernel_width=0.0), "kernel_width"),
        (dict(kernel_width=0.2), r"kernel_width=0.2 must be in \(0, max_distance=0.15\]"),
        (dict(reweight_every=0), "reweight_every=0 must be >= 1"),
        (dict(trust_region=[0.0, 0.1]), "trust_region"),
        (dict(joint_velocity_limit=0.0), "joint_velocity_limit=0.0 must be > 0"),
        (dict(joint_velocity_limits={"left_knee_joint": -1.0}), "joint_velocity_limits"),
        (dict(smooth_weight=-1.0), "smooth_weight=-1.0 must be >= 0"),
        (dict(interaction_weights=[256.0, -1.0, 2500.0]), "interaction_weights"),
        (dict(step_tolerance=0.01), "step_tolerance=0.01 must be in"),
        (dict(link_weights={"left_(knee": 2.0}), "is not a regex"),
        (dict(link_weights={"chair": 2.0}), r"link_weights entry 'chair' fully matches none of the rig's links"),
        (dict(probe_exclude_links=["no_such_link_.*"]), "probe_exclude_links entry 'no_such_link_.\\*'"),
        (dict(style_link_weights={"no_such_link_.*": 2.0}), "style_link_weights"),
        (dict(first_frame_iterations=0), "first_frame_iterations"),
        (dict(root_speed_slack=-0.5), "root_speed_slack"),
        (dict(joint_limits={"no_such_joint": [0.0, 1.0]}), "joint_limits"),
        (dict(joint_limits={"left_knee_joint": [1.0, 0.0]}), "joint_limits"),
        (dict(addons={"floor_stanse": {}}), r"addons \['floor_stanse'\] are not add-ons.*floor_stance"),
        (dict(addons={"floor_stance": {"stik": False}}), "stik"),
    ],
)
def test_misconfiguration_is_refused_at_construction(env, kin, override, match):
    with pytest.raises(ValueError, match=match):
        OTRetargeter(env, OTRetargeter.configure(**identity_options(kin, **override)))


def test_unknown_human_part_is_refused_before_solving(env, kin, self_retarget):
    _, human, _, _ = self_retarget
    opts = identity_options(kin)
    opts["correspondences"] = [[["left_thigh"], ["pelvis"]]]
    ret = OTRetargeter(env, OTRetargeter.configure(**opts))
    with pytest.raises(ValueError, match="left_thigh"):
        ret.retarget_motion(human, progress=False)


def test_object_poses_must_cover_every_frame(env, kin, self_retarget):
    ret, human, _, _ = self_retarget
    box = InteractionObject("box", trimesh.creation.box((0.3, 0.3, 0.3)), np.tile([2.0, 0, 0.15, 1, 0, 0, 0], (3, 1)))
    with pytest.raises(ValueError, match="box"):
        ret.retarget_motion(human, [box], progress=False)


def test_an_object_carries_its_own_interaction_weight(env, kin, self_retarget):
    # An object's f_l is on the object, not a link_weights regex: a retargeter reused over clips holds no per-clip
    # object names (a regex naming an object absent from a clip either raised or silently weighed 1).
    ret, human, _, _ = self_retarget
    poses = np.tile([2.0, 0, 0.15, 1, 0, 0, 0], (human.num_frames, 1))
    with pytest.raises(ValueError, match=r"InteractionObject 'box': weight=-1.0 must be finite and >= 0"):
        InteractionObject("box", trimesh.creation.box((0.3, 0.3, 0.3)), poses, weight=-1.0)
    heavy = InteractionObject("box", trimesh.creation.box((0.3, 0.3, 0.3)), poses, weight=3.0)
    light = InteractionObject("ball", trimesh.creation.icosphere(radius=0.1), poses)
    ret._setup(human, [heavy, light])
    np.testing.assert_array_equal(ret._object_weight, [3.0, 1.0])


def test_object_quaternions_stay_in_one_hemisphere_through_full_turns(env, kin, self_retarget):
    # A box spinning ~1 rad per frame, well away from the robot: R -> quaternion conversion per frame flipped the
    # sign of some outputs (q -> -q), which finite-difference angular velocity and slerp read as a full turn.
    ret, human, _, _ = self_retarget
    n = human.num_frames
    yaw = np.linspace(0.0, 7.0, n)
    poses = np.column_stack(
        [np.tile([2.0, 0.0, 0.15], (n, 1)), np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)]
    )
    box = InteractionObject("box", trimesh.creation.box((0.3, 0.2, 0.3)), poses)
    out = ret.retarget_motion(human, [box], progress=False).object_poses["box"][:, 3:]
    assert ((out[1:] * out[:-1]).sum(axis=1) > 0.0).all()
    assert out[0] @ poses[0, 3:] > 0.0


def test_an_empty_demonstration_is_refused_before_setup(env, kin, self_retarget):
    ret, human, _, _ = self_retarget
    with pytest.raises(ValueError, match=r"the demonstration has 0 frames at fps=30.0; it needs at least one frame"):
        ret.retarget_motion(human.slice(human.num_frames, human.num_frames), progress=False)


def test_a_larger_substitute_object_is_lifted_to_rest_on_the_floor(env, kin, self_retarget):
    # A 30 cm box rests on the floor in the demonstration, away from the robot; the scene holds a 36 cm box
    # at the same pose, which starts 3 cm into the floor. Its pose is a decision variable: non-penetration
    # and the box-floor contact carried over from the demonstrated box (Sec. III-E) must seat it on z = 0.
    _, human, _, _ = self_retarget
    n = human.num_frames
    poses = np.tile([2.0, 0.5, 0.15, 1.0, 0.0, 0.0, 0.0], (n, 1))
    box = InteractionObject(
        "box", trimesh.creation.box((0.36, 0.36, 0.36)), poses, demo_mesh=trimesh.creation.box((0.3, 0.3, 0.3))
    )
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    result = ret.retarget_motion(human, [box], progress=False)
    final = result.object_poses["box"][-1]
    np.testing.assert_allclose(final[2], 0.18, atol=0.006)
    # Sideways it may adapt a little: the floor-witness targets come from the smaller demonstrated footprint.
    np.testing.assert_allclose(final[:2], [2.0, 0.5], atol=0.03)


def test_a_limb_the_cold_start_puts_inside_an_object_is_drawn_back_and_the_rest_keep_the_posture(env, kin):
    # A box where the left knee of the demonstrated squat is: the left leg starts inside it. Started there, a link in a
    # thin part never got out (the rows on its two faces point in opposite directions). The left leg is drawn back
    # toward its neutral pose until it is clear; the right leg keeps the demonstrated squat; the box does not move.
    qs = squat(kin, n_frames=12, depth=0.85)[::-1]
    human = robot_demonstration(kin, qs)
    kin.set_configuration(qs[0])
    knee = kin.link_poses()[0][kin.link_index("left_knee_link")]
    poses = np.tile([*knee, 1.0, 0.0, 0.0, 0.0], (len(qs), 1))
    box = InteractionObject("box", trimesh.creation.box((0.12, 0.12, 0.12)), poses)
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    ret._setup(human, [box])
    pos, rot = poses[:1, :3].copy(), quat_to_R(poses[:1, 3:])
    q0 = ret._initial_configuration(human, pos, rot)
    assert ret._untangle(q0, pos, rot)[1] is False  # clear now: nothing left to draw back
    kin.set_configuration(q0)
    d, _ = ret.channels[1].exact_distance(ret.robot_probes.world(*kin.link_poses()) - poses[0, :3])
    assert d.min() >= -0.002
    for joint in ("right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"):
        np.testing.assert_allclose(q0[q_index(kin, joint)], qs[0, q_index(kin, joint)], atol=0.02)
    assert 0.0 <= q0[q_index(kin, "left_knee_joint")] < qs[0, q_index(kin, "left_knee_joint")] - 0.1
    np.testing.assert_array_equal(pos, poses[:1, :3])


def test_gauss_newton_gradient_matches_the_cost_it_models(env, kin, self_retarget):
    """Every residual term's gradient is the derivative of the cost evaluated with the same weights.

    The QP steps along that gradient, so a gradient that is not the cost's derivative (a discontinuous
    geodesic interpolation did this) steers every step wrong.
    Private hooks are read because the model is internal to the SQP.
    """
    _, human, qs, _ = self_retarget
    n = human.num_frames
    box = InteractionObject(
        "box", trimesh.creation.box((0.3, 0.3, 0.3)), np.tile([0.32, 0.0, 0.15, 1, 0, 0, 0], (n, 1))
    )
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    ret._setup(human, [box])
    pos, rot = box.poses[1:2, :3].copy(), quat_to_R(box.poses[1:2, 3:])
    frame = ret._frame_inputs(human, 1, qs[0], pos, rot)
    q = qs[1].copy()
    base = ret._linearize(q, pos, rot, frame)
    assert base.terms["robot_witness"] > 0 and base.terms["robot_depth"] > 0

    rng = np.random.default_rng(0)
    s = rng.normal(size=ret.n_x)
    s /= np.abs(s).max()

    def at(eps):
        qn = kin.integrate(q, eps * s[: kin.n_v])
        sl = ret._free_slot[0]
        pn = pos + eps * s[sl : sl + 3]
        rn = Rotation.from_rotvec(eps * s[sl + 3 : sl + 6]).as_matrix()[None] @ rot
        return ret._linearize(qn, pn, rn, frame, frozen=base.frozen)

    # Central differences at 0.2 mm: the geodesic table interpolates over ~1 cm sample spacings, so a 1 mm
    # step already carries a 10% curvature error, while a discontinuous interpolation was 75% off.
    eps = 2e-4
    plus, minus = at(eps), at(-eps)
    for name in base.terms:
        fd = (plus.terms[name] - minus.terms[name]) / (2 * eps)
        model = 2.0 * base.term_grads[name] @ s
        # atol absorbs the one-sided normal term switching on within the 1 mm probe (C1, not a wrong slope).
        np.testing.assert_allclose(model, fd, rtol=3e-2, atol=5e-2, err_msg=name)


def test_a_rejected_step_cuts_off_only_the_probes_it_drove_deeper(env, kin):
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    idx = np.array([3, 5, 7])
    x = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
    grad = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    # Before the step probe 3 was 1 mm deep, probe 5 already 6 mm deep, probe 7 outside; after it, 3 and 7 went
    # deeper (28 and 4 mm) while 5 stayed at 6 mm: only 3 and 7 count, the stuck probe 5 blocks nothing.
    before = [("robot", 1, idx, x, np.array([-0.001, -0.006, 0.010]), grad)]
    after = [("robot", 1, idx, x, np.array([-0.028, -0.006, -0.004]), grad)]
    deeper = ret._deepened(before, after)
    np.testing.assert_array_equal(deeper[("robot", 1)], [3, 7])
    cuts = {}
    assert OTRetargeter._add_cuts(cuts, after, deeper) == 2
    k, x_cut, d_cut, n_cut = cuts[("robot", 1)]
    np.testing.assert_array_equal(k, [3, 7])
    np.testing.assert_allclose(n_cut, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # unit normals
    np.testing.assert_allclose(d_cut, [-0.028, -0.004])
    # The same probes again add nothing: the caller then shrinks the step box instead.
    assert OTRetargeter._add_cuts(cuts, after, deeper) == 0


def test_a_stage_moves_only_its_free_columns(env, kin, self_retarget):
    # ``Frame.free_cols`` (an add-on's stage: the hand joints, one arm) holds every other joint, the root and the object.
    _, human, qs, _ = self_retarget
    n = human.num_frames
    box = InteractionObject("box", trimesh.creation.box((0.2, 0.2, 0.2)), np.tile([0.4, 0.0, 0.6, 1, 0, 0, 0], (n, 1)))
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    ret._setup(human, [box])
    pos, rot = box.poses[1:2, :3].copy(), quat_to_R(box.poses[1:2, 3:])
    cols = np.zeros(kin.n_v, dtype=bool)
    cols[[kin.var_names.index(j) for j in ("left_shoulder_pitch_joint", "left_elbow_joint")]] = True
    frame = replace(ret._frame_inputs(human, 1, qs[0], pos, rot), free_cols=cols)
    q = qs[1].copy()
    q[kin.n_base_q :] += 0.05
    q_s, pos_s, _, _, _ = ret.solve(q, pos, rot, frame, 8)
    held = kin.n_base_q + np.flatnonzero(~cols[kin.n_base_v :])
    np.testing.assert_array_equal(q_s[held], q[held])
    np.testing.assert_array_equal(q_s[: kin.n_base_q], q[: kin.n_base_q])
    np.testing.assert_array_equal(pos_s, pos)
    assert np.abs(q_s - q).max() > 1e-4  # the arm itself does move
