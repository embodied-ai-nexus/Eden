"""``drop_object``: the single-object PickPlace / NutAssembly variants (``eden/tasks/robosuite/variants.py``)."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest

from eden.options.managers.rewards import RewardGroupOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.variants import drop_object, names_object


def test_drop_object_removes_the_entity_and_every_term_naming_it():
    config = TASK_REGISTRY.build("robosuite_pick_place")
    # A per-object term nested in a reward group must go too, not only the flat top-level ones.
    config.reward_options.shaping = RewardGroupOptions(near_milk=config.reward_options.reach_milk.clone())

    drop_object(config, "milk")

    assert "milk" not in config.scene_options.keys()
    assert config.reward_options.shaping.term_keys() == ()
    for manager in (config.reward_options, config.metric_options, config.observation_options.policy):
        assert not [key for key in manager.term_keys() if names_object(key, "milk")]
    assert [key for key in config.reward_options.term_keys() if names_object(key, "can")]  # the others stay

    with pytest.raises(KeyError, match="Entity 'milk' not found in SceneOptions"):
        drop_object(config, "milk")


def test_drop_object_raises_when_no_term_names_the_object():
    config = TASK_REGISTRY.build("robosuite_pick_place")
    config.scene_options.marker = config.scene_options.milk.clone()

    with pytest.raises(KeyError, match="no manager term names object 'marker'"):
        drop_object(config, "marker")
