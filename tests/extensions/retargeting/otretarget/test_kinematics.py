"""Tests for ``eden.extensions.retargeting.otretarget.kinematics`` on G1 + Wuji hands (three welded entities)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from eden.extensions.retargeting.otretarget.kinematics import RigKinematics
from eden.options import EnvOptions, SceneOptions
from eden.options.robots.composite import G1Wuji
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def kin():
    env = build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(robot=G1Wuji())))
    return RigKinematics(env, ["robot", "left_hand", "right_hand"])


def _random_q(kin, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros(kin.n_q)
    q[0:3] = [0.3, -0.2, 0.8]
    q[3:7] = Rotation.random(random_state=seed).as_quat()[[3, 0, 1, 2]]
    q[7:] = rng.uniform(np.maximum(kin.joint_lower, -1.5), np.minimum(kin.joint_upper, 1.5))
    return q


def test_rig_spans_the_body_and_both_hands(kin):
    assert kin.n_v == 6 + 29 + 2 * 20
    assert kin.n_q == 7 + 29 + 2 * 20
    # A fingertip hangs below the body's wrist in one tree.
    tip, wrist = kin.link_index("l_index_finger_tip"), kin.link_index("left_wrist_yaw_link")
    chain, cur = [], tip
    while cur >= 0:
        chain.append(cur)
        cur = kin.link_parent[cur]
    assert wrist in chain


def test_link_jacobians_match_finite_differences_of_genesis_kinematics(kin):
    q = _random_q(kin, 0)
    kin.set_configuration(q)
    pos, rot = kin.link_poses()
    jac = kin.link_jacobians(pos, rot)
    # Central differences: the suite runs Genesis in float32, so the step cannot be tiny.
    eps = 1e-3
    for i in range(kin.n_v):
        dv = np.zeros(kin.n_v)
        dv[i] = eps
        kin.set_configuration(kin.integrate(q, dv))
        pos_p, rot_p = kin.link_poses()
        kin.set_configuration(kin.integrate(q, -dv))
        pos_m, rot_m = kin.link_poses()
        np.testing.assert_allclose((pos_p - pos_m) / (2 * eps), jac[:, 0:3, i], atol=2e-3)
        omega = Rotation.from_matrix(np.einsum("lab,lcb->lac", rot_p, rot_m)).as_rotvec() / (2 * eps)
        np.testing.assert_allclose(omega, jac[:, 3:6, i], atol=2e-3)


def test_integrate_and_difference_are_inverse(kin):
    q = _random_q(kin, 1)
    v = np.random.default_rng(2).normal(size=kin.n_v) * 0.2
    np.testing.assert_allclose(kin.difference(kin.integrate(q, v), q), v, atol=1e-10)


def test_integrate_keeps_the_root_quaternion_in_one_hemisphere(kin):
    # Two full turns in small steps: the matrix -> quaternion conversion alone lands in either hemisphere, and a
    # retargeted clip's root quaternion jumped q -> -q between frames.
    q = _random_q(kin, 4)
    step = np.zeros(kin.n_v)
    step[3:6] = [0.3, -0.2, 0.25]
    quats = [q[3:7]]
    for _ in range(60):
        q = kin.integrate(q, step)
        quats.append(q[3:7])
    quats = np.array(quats)
    assert ((quats[1:] * quats[:-1]).sum(axis=1) > 0.9).all()
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)


def test_entity_qpos_splits_root_and_hands(kin):
    q = _random_q(kin, 3)
    robot, left, right = kin.entity_qpos(q)
    assert (len(robot), len(left), len(right)) == (36, 20, 20)
    np.testing.assert_array_equal(np.concatenate([robot, left, right]), q)


def test_a_free_entity_is_refused(kin):
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                a=G1Wuji(base_pos=(0.0, 0.0, 0.8), robot_name="a", left_hand_name="al", right_hand_name="ar"),
            ),
        )
    )
    with pytest.raises(ValueError, match="free root"):
        RigKinematics(env, ["al", "a"])
