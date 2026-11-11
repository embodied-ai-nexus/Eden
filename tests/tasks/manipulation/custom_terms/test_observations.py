"""Tests for ``eden.tasks.manipulation.custom_terms.observations`` and the base-frame helper they share."""

from __future__ import annotations

import math

import pytest
import torch

from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.tasks.manipulation.custom_terms.common import _to_base_frame
from eden.tasks.manipulation.custom_terms.observations import object_orientation_b, object_rot6d_b
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 2
BASE_POS = (0.1, 0.2, 0.05)
# The robot base yawed +90 deg: base x is world y, base y is world -x.
BASE_QUAT = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
CUBE = {"robot_name": "robot", "object_name": "cube"}


@pytest.fixture(scope="module")
def env():
    """A fixed "robot" box at ``BASE_POS`` yawed 90 deg, and a free cube."""
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                robot=BoxOptions(
                    size=(0.1, 0.1, 0.1), default_root_pos=BASE_POS, default_root_quat=BASE_QUAT, is_fixed_base=True
                ),
                cube=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.3, 0.0, 0.025)),
            ),
        ),
        reset=True,
    )


def test_to_base_frame_takes_one_point_or_several_per_env(env):
    """``(N, K, 3)`` points land where each would alone, so a multi-point term need not loop or re-derive the frame."""
    # 1 m along base +x, and 1 m along base +y and up: env 0 gets the first point, env 1 the second.
    world = torch.tensor([[0.1, 1.2, 0.05], [-0.9, 0.2, 1.05]], device=env.device)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], device=env.device)

    one_each = _to_base_frame(env, "robot", world)
    several = _to_base_frame(env, "robot", world.unsqueeze(0).expand(NUM_ENVS, -1, -1))  # both points in each env

    torch.testing.assert_close(one_each, expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(several, expected.unsqueeze(0).expand(NUM_ENVS, -1, -1), atol=1e-5, rtol=0)


def test_object_rot6d_b_is_the_rows_of_the_relative_rotation_and_ignores_the_quaternion_sign(env):
    """A pose estimator may report ``q`` or ``-q`` for one pose; the policy must read the same either way."""
    cube = env.entities["cube"]
    # The cube yawed +180 deg in the world: +90 deg relative to the yawed base.
    yaw_180 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).expand(NUM_ENVS, -1)
    try:
        cube.set_quat(yaw_180)
        positive_q, positive = object_orientation_b(env, **CUBE), object_rot6d_b(env, **CUBE)
        cube.set_quat(-yaw_180)
        negative_q, negative = object_orientation_b(env, **CUBE), object_rot6d_b(env, **CUBE)
    finally:
        env.reset()

    # The quaternion flips its sign with the input; the 6D rotation does not.
    torch.testing.assert_close(negative_q, -positive_q, atol=1e-5, rtol=0)
    torch.testing.assert_close(negative, positive, atol=1e-5, rtol=0)
    # Rz(+90 deg) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]: its first two rows, not its first two columns.
    rows = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0], device=env.device)
    torch.testing.assert_close(positive, rows.expand(NUM_ENVS, -1), atol=1e-5, rtol=0)
