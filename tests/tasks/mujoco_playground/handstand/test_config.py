"""The handstand task config's action path."""

from __future__ import annotations

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis


def test_the_controller_integrates_one_step_while_the_manager_keeps_the_raw_delta():
    """Under ``ReferenceSource.ACCUMULATE`` the controller integrates the target itself, so the action manager
    keeps the raw delta rather than an ``env._last_raw_action`` side channel.
    """
    cfg = TASK_REGISTRY.build("mujoco_playground_handstand")
    cfg.env_options.num_envs = 2
    env = build_env(cfg, reset=True)
    controller = env.action_manager.get_term("dofs_pos_controller")
    # Small enough that no joint reaches its soft limit, so the integrated target is unclamped.
    action = torch.empty(2, env.action_manager.total_action_dim, device=env.device).uniform_(-0.05, 0.05)
    env.step(action)
    target = controller.processed_action.clone()

    env.step(action)

    torch.testing.assert_close(env.action_manager.action, action)
    torch.testing.assert_close(controller.processed_action, target + action * controller.effective_scale)
    assert torch.isfinite(env.reward_buf).all()
