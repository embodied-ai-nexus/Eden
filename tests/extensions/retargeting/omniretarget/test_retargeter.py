"""Tests for ``eden.extensions.retargeting.omniretarget.OmniRetargeter`` on a floating-base Unitree G1 and a fixed XHand1.

The demonstrations are made from the robot itself: a G1 pose is run through forward kinematics and
its keypoints are fed back as the "human" motion. The robot can then match the demonstration
exactly, so where it lands is a known value rather than a recomputation of the optimiser.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("cvxpy")  # OmniRetarget's QP solver; only in the ``all`` extra

import genesis as gs
from genesis.utils.geom import transform_by_quat, xyz_to_quat

import eden as en
from eden.extensions.retargeting.constants import JOINTS_MAP_UNITREE_G1
from eden.extensions.retargeting.omniretarget import OmniRetargeter
from eden.extensions.retargeting.omniretarget.g1_presets import (
    FOOT_STICKING_LINKS,
    FOOT_STICKING_MAP,
    LINK_OFFSETS,
    MANUAL_COST,
    MANUAL_LB,
    MANUAL_UB,
    NOMINAL_TRACKING_DOFS,
)
from eden.options import BoxOptions, EntityOptions, EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

G1_MAP = dict(JOINTS_MAP_UNITREE_G1["smplh"])
# The fixed-base XHand1's wrist and fingertips, as demo names.
HAND_MAP = {
    "wrist": "right_hand_link",
    "thumb": "right_hand_thumb_rota_tip",
    "index": "right_hand_index_rota_tip",
    "middle": "right_hand_mid_tip",
    "ring": "right_hand_ring_tip",
    "pinky": "right_hand_pinky_tip",
}
# A fixed base carrying a ball joint (4 q / 3 DOFs) and a hinge.
BALL_JOINT_MJCF = """<mujoco model="floppy"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="-0.6 -0.6 0.4"><geom type="box" size="0.04 0.04 0.02" density="500"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" density="500"/>
      <body name="spindle" pos="0.08 0 0"><joint name="free_spin_joint" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.01" density="500"/></body></body></body>
</worldbody></mujoco>"""
UPPER_BODY = ["Pelvis", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
NO_STICKING = {"L_Toe": False, "R_Toe": False}
STANDING_HEIGHT = 0.78


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    ball_joint_asset = tmp_path_factory.mktemp("omniretarget") / "floppy.xml"
    ball_joint_asset.write_text(BALL_JOINT_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                robot=en.robots.G1Dofs29(),
                box=BoxOptions(size=(0.3, 0.2, 0.2), default_root_pos=(3.0, 0.0, 0.1)),
                hand=en.robots.XHand1_R(is_fixed_base=True, default_root_pos=(-2.0, 0.0, 0.3)),
                floppy=EntityOptions(
                    file=str(ball_joint_asset),
                    is_fixed_base=True,
                    is_articulated=True,
                    dofs_name=["ball_joint", "free_spin_joint"],
                ),
                floor=PlaneOptions(),
            ),
        ),
        reset=True,
    )


@pytest.fixture(autouse=True)
def _restore_env(env):
    # The retargeter teleports the robot to every candidate it evaluates.
    yield
    env.reset()


def _retargeter(env, **overrides) -> OmniRetargeter:
    params = {
        "entity_name": "robot",
        "foot_links": FOOT_STICKING_LINKS,
        "demo_joints": list(G1_MAP),
        "laplacian_match_links": G1_MAP,
        "link_offsets": LINK_OFFSETS,
        "foot_sticking_map": FOOT_STICKING_MAP,
        # Smoothness pulls frame 0 toward the zero-joint initial guess, which no demonstration
        # here starts from; with it off the robot's own motion is reachable exactly.
        "smooth_weight": 0.0,
    }
    return OmniRetargeter(env=env, options=OmniRetargeter.configure(**{**params, **overrides}))


def _dof(env, name: str) -> int:
    """Column of ``name`` in a retargeted ``(T, 7 + n_joints)`` qpos row."""
    return 7 + env.entities["robot"].dofs_name.index(name)


def _pose(env, base_xyz=(0.0, 0.0, STANDING_HEIGHT), seed: int = 0, **joints: float) -> np.ndarray:
    """A floating-base qpos with every joint a little off its neutral angle, inside its limits."""
    robot = env.entities["robot"]
    lo, hi = (bound[0].cpu().numpy() for bound in robot.get_dofs_limit())
    q_joints = np.clip(0.0, lo, hi) + 0.1 * (hi - lo) * np.random.default_rng(seed).uniform(-1.0, 1.0, len(lo))
    q_joints = np.clip(q_joints, lo, hi)
    for name, value in joints.items():
        q_joints[robot.dofs_name.index(name)] = value
    return np.concatenate([base_xyz, [1.0, 0.0, 0.0, 0.0], q_joints])


def _keypoints(env, q: np.ndarray, links: dict[str, str], entity_name: str = "robot") -> np.ndarray:
    """World positions of ``links`` (with their ``LINK_OFFSETS`` point) at qpos ``q``, via the entity's FK."""
    robot = env.entities[entity_name]
    robot.set_qpos(torch.as_tensor(q, dtype=gs.tc_float, device=gs.device).unsqueeze(0))
    points = []
    for name, link_name in links.items():
        link = robot.get_link(link_name)
        pos = link.get_pos(relative=False)[0]
        if name in LINK_OFFSETS:
            offset = torch.as_tensor(LINK_OFFSETS[name], dtype=pos.dtype, device=pos.device)
            pos = pos + transform_by_quat(offset, link.get_quat(relative=False)[0])
        points.append(pos.cpu().numpy())
    return np.stack(points)


def _demo(env, qs: list[np.ndarray], links: dict[str, str] = G1_MAP, entity_name: str = "robot") -> np.ndarray:
    return np.stack([_keypoints(env, q, links, entity_name) for q in qs])


def test_the_robots_own_motion_is_recovered_keypoint_for_keypoint(env):
    qs = [_pose(env, base_xyz=(0.02 * t, 0.0, STANDING_HEIGHT), left_knee_joint=0.3 + 0.1 * t) for t in range(3)]
    demo = _demo(env, qs)

    out = _retargeter(env).retarget_motion(demo, [NO_STICKING] * 3)

    assert out.shape == (3, 7 + 29)
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(_demo(env, list(out)), demo, atol=3e-3)


def test_the_g1_presets_name_links_and_dofs_the_g1_has(env):
    robot = env.entities["robot"]
    links = {link.name for link in robot.links}
    dofs = set(robot.dofs_name)

    assert set(G1_MAP.values()) | set(FOOT_STICKING_LINKS.values()) <= links
    assert set(MANUAL_LB) | set(MANUAL_UB) | set(MANUAL_COST) | set(NOMINAL_TRACKING_DOFS) <= dofs
    assert set(LINK_OFFSETS) <= set(G1_MAP) | set(FOOT_STICKING_LINKS)
    assert {key for keys in FOOT_STICKING_MAP.values() for key in keys} <= set(FOOT_STICKING_LINKS)


@pytest.mark.parametrize("sticking", [True, False], ids=["sticking", "free"])
def test_a_sticking_foot_stays_planted_while_the_demonstration_slides_it(env, sticking):
    # The whole demonstration walks 3 cm per frame, feet included.
    qs = [_pose(env, base_xyz=(0.03 * t, 0.0, STANDING_HEIGHT)) for t in range(4)]
    flags = [NO_STICKING] + [{"L_Toe": sticking, "R_Toe": False}] * 3

    out = _retargeter(env).retarget_motion(_demo(env, qs), flags)

    left_foot = {key: FOOT_STICKING_LINKS[key] for key in FOOT_STICKING_MAP["L_Toe"]}
    feet = _demo(env, list(out), left_foot)
    slide = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1).max()
    if sticking:
        assert slide < 3e-3
    else:
        assert slide > 2e-2


@pytest.mark.parametrize("activate", [True, False], ids=["constrained", "unconstrained"])
def test_the_foot_ground_constraint_lifts_a_demonstration_sunk_into_the_floor(env, activate):
    demo = _demo(env, [_pose(env)] * 2)
    feet = {**FOOT_STICKING_LINKS}
    sink = _demo(env, [_pose(env)], feet)[0, :, 2].min() + 0.04
    demo[:, :, 2] -= sink  # the lowest foot point now 4 cm under the floor

    out = _retargeter(
        env, activate_foot_ground_constraint=activate, foot_ground_z=0.0, foot_ground_margin=0.005
    ).retarget_motion(demo, [NO_STICKING] * 2)

    lowest = _demo(env, list(out), feet)[:, :, 2].min()
    if activate:
        assert lowest > 0.005 - 1e-3
    else:
        assert lowest < -0.03


@pytest.mark.parametrize(
    "joint, demanded, overrides, bound",
    [
        # The knee's asset limit is [-0.087, 2.88]: a hyperextended demonstration.
        ("left_knee_joint", -0.6, {}, ("lower", -0.0872665)),
        # A manual bound tighter than the asset's own elbow limit.
        ("left_elbow_joint", 1.0, {"manual_ub": {"left_elbow_joint": 0.3}}, ("upper", 0.3)),
    ],
    ids=["asset-limit", "manual-bound"],
)
def test_a_demonstration_past_a_joint_limit_is_held_at_the_limit(env, joint, demanded, overrides, bound):
    demo = _demo(env, [_pose(env, **{joint: demanded})] * 2)
    side, limit = bound

    held = _retargeter(env, **overrides).retarget_motion(demo, [NO_STICKING] * 2)[:, _dof(env, joint)]
    free = _retargeter(env, activate_joint_limits=False, **overrides).retarget_motion(demo, [NO_STICKING] * 2)

    # Without the limits the joint follows the demonstration well past the bound, so the bound
    # is what held it. (Not exactly back to ``demanded``: position keypoints leave an arm redundant.)
    unbounded = free[:, _dof(env, joint)]
    if side == "lower":
        assert held.min() >= limit - 1e-4
        assert unbounded.max() < limit - 0.3
    else:
        assert held.max() <= limit + 1e-4
        assert unbounded.min() > limit + 0.3


@pytest.mark.parametrize("weight", [5.0, 0.0], ids=["tracked", "untracked"])
def test_the_nominal_trajectory_steers_dofs_the_demonstration_leaves_free(env, weight):
    # Only the upper body is demonstrated, so nothing but the nominal term moves the legs.
    upper = {name: G1_MAP[name] for name in UPPER_BODY}
    demo = _demo(env, [_pose(env)] * 3, upper)
    nominal = np.stack([_pose(env, left_knee_joint=0.2 + 0.3 * t, right_knee_joint=0.2 + 0.3 * t) for t in range(3)])

    out = _retargeter(
        env,
        demo_joints=UPPER_BODY,
        laplacian_match_links=upper,
        nominal_tracking_dofs=NOMINAL_TRACKING_DOFS,
        w_nominal_tracking_init=weight,
        smooth_weight=0.2,  # the default: with no keypoint on them, it is what holds untracked legs still
    ).retarget_motion(demo, [NO_STICKING] * 3, q_nominal_list=nominal)

    knee = out[-1, _dof(env, "left_knee_joint")]
    # Untracked, the legs keep the nominal's first frame they started from.
    np.testing.assert_allclose(knee, 0.8 if weight else 0.2, atol=0.03)


def test_the_motion_is_recovered_relative_to_a_turned_and_shifted_object(env):
    # The interaction mesh is built in the object's frame; a box well clear of the robot, turned
    # and lifted, must not change where the robot's own motion lands in the world.
    qs = [_pose(env, base_xyz=(0.02 * t, 0.0, STANDING_HEIGHT)) for t in range(2)]
    demo = _demo(env, qs)
    box_quat = xyz_to_quat(np.array([20.0, 0.0, 60.0]), rpy=True, degrees=True)
    box_poses = np.tile(np.concatenate([[1.2, 0.4, 0.5], box_quat]), (2, 1))

    out = _retargeter(env).retarget_motion(
        demo, [NO_STICKING] * 2, object_name="box", object_poses=box_poses, n_surface_points=24
    )

    np.testing.assert_allclose(_demo(env, list(out)), demo, atol=3e-3)


def test_an_object_without_its_poses_is_refused(env):
    with pytest.raises(ValueError, match="object_poses is required"):
        _retargeter(env).retarget_motion(_demo(env, [_pose(env)]), [NO_STICKING], object_name="box")


def test_a_fixed_base_hands_own_motion_is_recovered(env):
    # A fixed base has no root in its qpos: the rows are just the 12 finger joints.
    hand = env.entities["hand"]
    lo, hi = (bound[0].cpu().numpy() for bound in hand.get_dofs_limit())
    qs = [lo + (0.3 + 0.1 * t) * (hi - lo) for t in range(3)]
    demo = _demo(env, qs, HAND_MAP, "hand")

    out = OmniRetargeter(
        env=env,
        options=OmniRetargeter.configure(
            entity_name="hand", demo_joints=list(HAND_MAP), laplacian_match_links=HAND_MAP, smooth_weight=0.0
        ),
    ).retarget_motion(demo, [{}] * 3)

    assert out.shape == (3, 12)
    assert (out >= lo - 1e-4).all() and (out <= hi + 1e-4).all()
    np.testing.assert_allclose(_demo(env, list(out), HAND_MAP, "hand"), demo, atol=3e-3)


def test_an_initial_pose_is_kept_alongside_a_nominal_trajectory(env):
    # Only the upper body is demonstrated and the nominal term is off, so the legs stay where the
    # motion starts: at ``q_a_init`` when it is given, not at the nominal's first frame.
    upper = {name: G1_MAP[name] for name in UPPER_BODY}
    demo = _demo(env, [_pose(env)] * 2, upper)
    nominal = np.stack([_pose(env, left_knee_joint=0.2)] * 2)

    out = _retargeter(
        env, demo_joints=UPPER_BODY, laplacian_match_links=upper, w_nominal_tracking_init=0.0, smooth_weight=0.2
    ).retarget_motion(demo, [NO_STICKING] * 2, q_a_init=_pose(env, left_knee_joint=0.7), q_nominal_list=nominal)

    np.testing.assert_allclose(out[:, _dof(env, "left_knee_joint")], 0.7, atol=0.03)


@pytest.mark.parametrize(
    "overrides, named",
    [
        ({"laplacian_match_links": {**G1_MAP, "L_Wrist": "left_wrist_typo_link"}}, "left_wrist_typo_link"),
        ({"foot_links": {**FOOT_STICKING_LINKS, "left_sphere_1": "left_ankle_typo_link"}}, "left_ankle_typo_link"),
        ({"demo_joints": [name for name in G1_MAP if name != "L_Knee"]}, "L_Knee"),
        ({"foot_sticking_map": {"L_Toe": ["left_sphere_9"]}}, "left_sphere_9"),
        ({"manual_lb": {"waist_rol_joint": -0.3}}, "waist_rol_joint"),
        ({"manual_ub": {"left_elbow": 1.4}}, "left_elbow"),
        ({"manual_cost": {"waist_yaw": 0.2}}, "waist_yaw"),
        ({"nominal_tracking_dofs": ["left_knee"]}, "left_knee"),
        ({"n_iter": 0}, "n_iter=0"),
    ],
    ids=[
        "link",
        "foot-link",
        "demo-joint",
        "sticking-key",
        "manual-lb",
        "manual-ub",
        "manual-cost",
        "nominal-dof",
        "no-iterations",
    ],
)
def test_a_misconfiguration_is_refused_at_construction_naming_the_entry(env, overrides, named):
    with pytest.raises(ValueError, match=named):
        _retargeter(env, **overrides)


@pytest.mark.parametrize(
    "inputs, named",
    [
        ({"foot_sticking_sequences": [NO_STICKING]}, "foot_sticking_sequences has 1 entries for 2 frames"),
        # LAFAN1 toe names against the SMPL-H-keyed map: the left foot would never stick.
        ({"foot_sticking_sequences": [{"LeftToeBase": True, "R_Toe": False}] * 2}, r"flags \['LeftToeBase'\]"),
        ({"human_joint_motions": np.zeros((2, 3, 3))}, r"must be \(T, 15, 3\)"),
        ({"object_name": "crate", "object_poses": np.zeros((2, 7))}, "'crate' is not an entity"),
        ({"object_name": "box", "object_poses": np.zeros((2, 3))}, r"object_poses has shape \(2, 3\)"),
        ({"q_a_init": np.zeros(29)}, r"q_a_init has shape \(29,\)"),
        ({"q_nominal_list": np.zeros((2, 29))}, r"q_nominal_list has shape \(2, 29\)"),
    ],
    ids=["sticking-length", "sticking-key", "demo-joints", "unknown-object", "object-poses", "initial-pose", "nominal"],
)
def test_a_mismatched_motion_input_is_refused_before_any_frame_is_solved(env, inputs, named):
    args = {"human_joint_motions": _demo(env, [_pose(env)] * 2), "foot_sticking_sequences": [NO_STICKING] * 2}

    with pytest.raises(ValueError, match=named):
        _retargeter(env).retarget_motion(**{**args, **inputs})


def test_an_entity_with_a_multi_dof_joint_is_refused_at_construction(env):
    # A ball joint has 4 q and 3 DOFs; stepping it like a hinge would misplace every slice after it.
    with pytest.raises(ValueError, match=r"n_qs=5 and n_dofs=4.*\['ball_joint'\]"):
        OmniRetargeter(
            env=env,
            options=OmniRetargeter.configure(
                entity_name="floppy", demo_joints=["socket"], laplacian_match_links={"socket": "socket"}
            ),
        )
