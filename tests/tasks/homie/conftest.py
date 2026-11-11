"""A real homie env (29-DOF G1, the shipped config), shared by the homie tests."""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env

NUM_ENVS = 4


@pytest.fixture(scope="package")
def homie_env():
    cfg = TASK_REGISTRY.build("homie")
    cfg.env_options.num_envs = NUM_ENVS
    return build_env(cfg, reset=True)
