"""The getup task config's action path."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.requires_genesis


def test_the_action_manager_holds_the_clamped_raw_delta(env):
    """Under ``ReferenceSource.DELTA`` the raw policy delta, clipped by the runner's ``clip_actions``, is what
    ``last_action`` and the action-rate / stand-still rewards read, not an ``env._raw_action`` side channel.
    """
    clip = env.config.runner_options.clip_actions
    action = torch.empty(env.num_envs, env.action_manager.total_action_dim).uniform_(-3.0, 3.0).clamp(-clip, clip)

    env.step(action.to(env.device))

    torch.testing.assert_close(env.action_manager.action, action.to(env.device))
    assert torch.isfinite(env.reward_buf).all()
