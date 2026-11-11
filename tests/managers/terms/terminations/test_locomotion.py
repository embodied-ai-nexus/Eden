"""The locomotion termination terms in ``eden/managers/terms/terminations/locomotion.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.observations.locomotion import base_height
from eden.managers.terms.terminations.locomotion import root_height_below_minimum

pytestmark = pytest.mark.requires_genesis


def test_root_height_below_minimum_is_strict(sloped_env):
    """``height < minimum``: a base resting exactly on the floor survives, one just under it does not."""
    z = sloped_env.entities["box"].get_pos()[0, 2].item()  # the exact float default_root_pos wrote
    at_floor = root_height_below_minimum(sloped_env, entity_name="box", minimum_height=z)
    assert at_floor.dtype == torch.bool and at_floor.tolist() == [False, False]
    assert root_height_below_minimum(sloped_env, entity_name="box", minimum_height=z + 1e-3).all()


def test_root_height_below_minimum_measures_from_the_terrain_when_named(sloped_env):
    """One 3 m floor, opposite verdicts: the box is 2 m up in world z but 3.5 m over the sunken slope."""
    assert root_height_below_minimum(sloped_env, entity_name="box", minimum_height=3.0).all()
    assert not root_height_below_minimum(
        sloped_env, entity_name="box", minimum_height=3.0, terrain_entity_name="terrain"
    ).any()


def test_root_height_below_minimum_thresholds_the_base_height_observation(sloped_env):
    """The termination and the observation read one helper, so they cannot disagree about the number."""
    for terrain_name in (None, "terrain"):
        observed = base_height(sloped_env, entity_name="box", terrain_entity_name=terrain_name)[:, 0]
        floor = observed.min().item() + 0.25
        done = root_height_below_minimum(
            sloped_env, entity_name="box", minimum_height=floor, terrain_entity_name=terrain_name
        )
        assert torch.equal(done, observed < floor), terrain_name
