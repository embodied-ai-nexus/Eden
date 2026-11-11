"""Tests for ``eden.utils.motion_reference.bipedal_ik``: the IK that walks a G1 from its stance into a clip."""

from __future__ import annotations

import numpy as np
import pytest

from eden.options.robots.unitree_g1 import G1_IK_PROFILE
from eden.utils.motion_reference.bipedal_ik import (
    _resolve_default_qpos,
    _resolve_qpos_bias_weights,
    _resolve_upper_body_addr,
    generate_start_transition,
    integrate_qpos_tangent,
)

TRANSITION = dict(framerate=50.0, default_pose_sec=0.5, step_sec=0.2, transition_sec=0.2, max_ik_iters=80)


@pytest.fixture(scope="module")
def robot():
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.005),
        scene_options=SceneOptions(robot=G1_IK_PROFILE.robot_options),
    )
    env.build()
    yield env.entities["robot"]
    env.destroy()


def _clip_from(first_frame: np.ndarray, num_frames: int) -> np.ndarray:
    return np.tile(first_frame, (num_frames, 1))


def test_tangent_integration_matches_mujoco():
    """The free-joint convention: right-multiply the quaternion by the body-frame angular exponential."""
    mujoco = pytest.importorskip("mujoco")
    from eden.utils.assets import get_asset_path

    model = mujoco.MjModel.from_xml_path(get_asset_path(type(G1_IK_PROFILE.robot_options).model_fields["file"].default))
    qpos = _resolve_default_qpos(G1_IK_PROFILE).copy()
    qpos[3:7] = [np.cos(0.25), 0, 0, np.sin(0.25)]  # a rotated base tells left from right multiplication
    dq = np.random.default_rng(0).standard_normal(model.nv) * 0.2

    expected = qpos.copy()
    mujoco.mj_integratePos(model, expected, dq, 0.7)

    np.testing.assert_allclose(integrate_qpos_tangent(qpos, dq, 0.7), expected, atol=1e-12)


def test_g1_qpos_bias_weights():
    weights = _resolve_qpos_bias_weights(G1_IK_PROFILE)
    dofs = list(G1_IK_PROFILE.robot_options.dofs_name)

    assert _resolve_default_qpos(G1_IK_PROFILE).shape == (36,) and weights.shape == (35,)
    assert weights[5] == 2.5  # base yaw
    for name in ("left_hip_roll_joint", "right_hip_yaw_joint", "left_ankle_roll_joint"):
        assert weights[6 + dofs.index(name)] == 0.0
    for name in ("left_hip_pitch_joint", "right_knee_joint", "left_ankle_pitch_joint", "waist_yaw_joint"):
        assert weights[6 + dofs.index(name)] == 1.0


def test_a_near_stance_clip_takes_one_step_from_the_default_pose(robot):
    default = _resolve_default_qpos(G1_IK_PROFILE)
    first = default.copy()
    first[0] += 0.05  # a small base shift

    out = generate_start_transition(robot, G1_IK_PROFILE, _clip_from(first, 30), **TRANSITION)

    # default hold (25) + one step (11) + merge (10) + the clip (30)
    assert out.shape[1] == default.shape[0] and out.shape[0] < 100
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:11], np.tile(default, (11, 1)))


def test_the_upper_body_does_not_jump_at_the_transition_seam(robot):
    """The merge once restarted from the default arms after the transition had blended them to
    the clip's T-pose: a ~1.5 rad shoulder flip in one frame.
    """
    first = _resolve_default_qpos(G1_IK_PROFILE).copy()
    first[7 + list(robot.dofs_name).index("left_shoulder_roll_joint")] = 1.5
    first[7 + list(robot.dofs_name).index("right_shoulder_roll_joint")] = -1.5

    out = generate_start_transition(robot, G1_IK_PROFILE, _clip_from(first, 20), **TRANSITION)

    upper = _resolve_upper_body_addr(G1_IK_PROFILE, list(robot.dofs_name))
    assert np.abs(np.diff(out[:, upper], axis=0)).max() < 0.4  # rad per frame


def test_a_wide_stance_clip_takes_two_steps(robot):
    first = _resolve_default_qpos(G1_IK_PROFILE).copy()
    first[7 + 1], first[7 + 7] = 0.30, -0.30  # hip rolls open the feet
    clip = _clip_from(first, 30)

    one = generate_start_transition(robot, G1_IK_PROFILE, clip, single_step_threshold_m=1e6, **TRANSITION)
    two = generate_start_transition(robot, G1_IK_PROFILE, clip, single_step_threshold_m=0.01, **TRANSITION)

    assert two.shape[0] > one.shape[0] and np.isfinite(two).all()
