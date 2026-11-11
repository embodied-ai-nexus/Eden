"""A real HUSKY env (G1 23-DOF on the skateboard), shared by the husky tests."""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env


@pytest.fixture(scope="package")
def husky_env():
    cfg = TASK_REGISTRY.build("husky")
    cfg.env_options.num_envs = 4
    return build_env(cfg, reset=True)
