"""The getup task, built once for the directory (Go1 on a plane)."""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY
from eden.utils.smoke import build_and_step


def _getup_config(num_envs: int = 2):
    cfg = TASK_REGISTRY.build("mujoco_playground_getup")
    cfg.env_options.num_envs = num_envs
    # Every env takes the sampled branch, so the draw ``DropFromHeight``'s guard protects is exercised.
    cfg.event_options.drop_from_height.drop_prob = 1.0
    return cfg


@pytest.fixture
def getup_config():
    """A fresh, unbuilt getup config with every env dropped from height."""
    return _getup_config()


@pytest.fixture(scope="package")
def env():
    return build_and_step(_getup_config(), num_envs=2, num_steps=1)
