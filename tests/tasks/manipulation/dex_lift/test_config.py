"""Tests for ``eden.tasks.manipulation.dex_lift.config``: what the deployable policy observes, where its goals are,
and the fingertip terms, on the built task.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.custom_terms.common import _base_to_world
from eden.tasks.manipulation.dex_lift.config import (
    _DEX,
    _REST_Z,
    ManipulationDexLiftConfig,
    dex_fingertip_to_object_reward,
    dex_fingertips_pos_b,
    dex_fingertips_to_object_obs,
)
from eden.utils.geom import inv_transform_by_quat
from tests.support.envs import build_env

NUM_ENVS = 2
# What the real rig measures: Franka and XHand encoders, FK of them, the AprilTag cube pose, the goal, the last action.
MEASURED = {
    "arm_dofs_pos",
    "arm_dofs_vel",
    "hand_dofs_pos",
    "object_position",
    "object_orientation",
    "target_object_position",
    "grasp_point",
    "fingertips_pos",
    "last_action",
}
# What only the simulator knows.
PRIVILEGED = {"hand_dofs_vel", "fingertips_to_object", "object_lin_vel", "object_ang_vel"}


def _term_names(group) -> set[str]:
    return {k for k, v in (group.__pydantic_extra__ or {}).items() if not isinstance(v, str)}


def test_the_policy_observes_only_what_the_rig_measures_and_the_critic_also_sees_the_privileged_state():
    obs = ManipulationDexLiftConfig().observation_options
    assert _term_names(obs.policy) == MEASURED
    assert _term_names(obs.critic) == MEASURED | PRIVILEGED


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("manipulation_dex_lift").with_overrides_from_dict(
        {"env_options": {"num_envs": NUM_ENVS}}
    )
    return build_env(config, reset=True)


def _tips_w(env) -> torch.Tensor:
    hand = env.entities[_DEX.hand_name]
    _, idx = hand.find_named_links_idx_local(_DEX.fingertip_links, preserve_order=True)
    return hand.get_links_pos(ls_idx_local=idx)  # (N, 5, 3)


@pytest.mark.requires_genesis
def test_the_first_goal_is_ten_centimetres_above_the_resting_cube_in_the_world(env):
    """The command is in the arm's base frame, which sits ``_ARM_BASE_Z`` above the world origin.

    A world-frame height in the command would put every goal that much higher, out of the arm's reach. At curriculum
    level 0 the goal is the curriculum's easy point.
    """
    goal_w = _base_to_world(env, _DEX.arm_name, env.command_manager.get_term("lift_target").command[..., :3])
    torch.testing.assert_close(goal_w[:, 2].cpu(), torch.full((NUM_ENVS,), _REST_Z + 0.10), atol=1e-5, rtol=0)


@pytest.mark.requires_genesis
def test_fingertips_pos_is_each_fingertip_in_the_arm_base_frame(env):
    robot = env.entities[_DEX.arm_name]
    tips_w = _tips_w(env)
    expected = torch.stack(
        [inv_transform_by_quat(tips_w[:, k] - robot.get_pos(), robot.get_quat()) for k in range(tips_w.shape[1])],
        dim=1,
    )
    actual = dex_fingertips_pos_b(
        env, robot_name=_DEX.arm_name, hand_name=_DEX.hand_name, fingertip_links=_DEX.fingertip_links
    )
    assert actual.shape == (NUM_ENVS, 3 * len(_DEX.fingertip_links))
    torch.testing.assert_close(actual, expected.flatten(start_dim=1), atol=1e-5, rtol=0)


@pytest.mark.requires_genesis
def test_fingertip_terms_measure_from_each_fingertip_to_the_cube(env):
    tips_w = _tips_w(env)
    to_cube = env.entities["cube"].get_pos().unsqueeze(1) - tips_w  # (N, 5, 3)

    tips = {"hand_name": _DEX.hand_name, "object_name": "cube", "fingertip_links": _DEX.fingertip_links}

    observed = dex_fingertips_to_object_obs(env, **tips)
    torch.testing.assert_close(observed, to_cube.flatten(start_dim=1), atol=1e-5, rtol=0)

    expected_reward = torch.exp(-to_cube.square().sum(dim=-1) / 0.05**2).mean(dim=-1)
    torch.testing.assert_close(dex_fingertip_to_object_reward(env, **tips, sigma=0.05), expected_reward)
