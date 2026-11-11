"""Tests for ``eden.utils.genesis_runtime``: dropping the rigid solver's per-step query caches."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.utils.configs import EdenRLConfig
from eden.utils.genesis_runtime import clear_rigid_query_caches
from tests.support.envs import build_env


def test_clearing_drops_the_cached_contact_query_so_the_next_query_is_answered_afresh():
    env = build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.04))),
        ),
        reset=True,
    )
    env.scene.step()
    box = env.entities["box"]
    box.get_contacts()
    cache = env.rigid_solver.collider._contact_data_cache
    assert cache  # the query was cached for the rest of the step

    clear_rigid_query_caches(env.rigid_solver)

    assert not cache
    assert not env.rigid_solver.constraint_solver._eq_const_info_cache


def test_a_genesis_without_the_cache_attribute_raises_naming_it():
    """A Genesis bump that renames a cache must not leave the clear silently doing nothing."""
    renamed = SimpleNamespace(collider=SimpleNamespace(_contacts_cache={}), constraint_solver=None)

    with pytest.raises(RuntimeError, match=r"no per-step query cache '_contact_data_cache'.*_RIGID_QUERY_CACHES"):
        clear_rigid_query_caches(renamed)
