"""Phase predicates for the metric-combinator tests.

``always_true`` / ``always_false`` / ``env_specific`` are metric terms of the tests' own, registered
once here. A predicate that has to change mid-test is a cube's height instead (``lift``).
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY


@METRIC_TERM_REGISTRY.register()
def always_true(env):
    return torch.ones(env.num_envs, device=env.device)


@METRIC_TERM_REGISTRY.register()
def always_false(env):
    return torch.zeros(env.num_envs, device=env.device)


@METRIC_TERM_REGISTRY.register()
def env_specific(env, *, pass_envs):
    """1.0 in the envs listed in ``pass_envs``, 0.0 elsewhere."""
    result = torch.zeros(env.num_envs, device=env.device)
    result[pass_envs] = 1.0
    return result


@pytest.fixture
def lift(panda_env, restore_state):
    """``lift(cube, on)``: raise ``cube`` to 1 m (``on``) or lower it to 0 m, per env when ``on`` is a list."""
    remembered = set()

    def set_height(cube, on):
        entity = panda_env.entities[cube]
        if cube not in remembered:
            restore_state(entity)
            remembered.add(cube)
        on = torch.as_tensor(on if isinstance(on, list) else [on] * panda_env.num_envs)
        pos = entity.get_pos().clone()
        pos[:, 2] = on.float()
        entity.set_pos(pos)

    return set_height
