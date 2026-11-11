"""TwoArmTransport: robosuite's success predicate — the payload on the target bin's floor and the trash on the trash bin's."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.two_arm_transport.config import (
    _PAYLOAD_PITCHED_QUAT,
    _PAYLOAD_START_Z,
    _TARGET_BIN_XY,
    _TRASH_BIN_XY,
    _TRASH_REST_Z,
)
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_two_arm_transport")
    config.env_options.num_envs = 2
    return build_env(config, reset=True)


def _settle(env, steps: int = 20) -> None:
    for _ in range(steps):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        assert not env.reset_buf.any(), "an episode ended while the objects settled"


@pytest.mark.requires_genesis
def test_the_task_does_not_start_solved(env):
    env.reset(seed=0)
    _settle(env, 5)
    metrics = env.metric_manager
    assert metrics.get_value("trash_in_trash").tolist() == [0.0, 0.0]
    assert metrics.get_value("payload_in_target").tolist() == [0.0, 0.0]
    assert metrics.get_value("transported").tolist() == [0.0, 0.0]


@pytest.mark.requires_genesis
def test_transported_needs_both_objects_resting_in_their_bins(env):
    """Env 0 gets both objects in their bins, env 1 only the trash: only env 0 is a success."""
    env.reset(seed=1)
    device = env.device
    trash_pos = torch.tensor([[*_TRASH_BIN_XY, _TRASH_REST_Z + 0.002]] * env.num_envs, device=device)
    env.entities["trash"].set_pos(trash_pos, relative=True)
    payload = env.entities["payload"]
    payload.set_quat(
        torch.tensor([_PAYLOAD_PITCHED_QUAT], device=device), envs_idx=torch.tensor([0], device=device), relative=True
    )
    payload.set_pos(
        torch.tensor([[*_TARGET_BIN_XY, _PAYLOAD_START_Z]], device=device),
        envs_idx=torch.tensor([0], device=device),
        relative=True,
    )
    _settle(env)

    metrics = env.metric_manager
    assert metrics.get_value("trash_in_trash").tolist() == [1.0, 1.0]
    assert metrics.get_value("payload_in_target").tolist() == [1.0, 0.0]
    assert metrics.get_value("transported").tolist() == [1.0, 0.0]
    assert metrics.success_buf.tolist() == [True, False]
