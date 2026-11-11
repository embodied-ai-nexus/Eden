"""The SIMPLER table tasks start from SIMPLER's own fixed layouts, never with the two objects overlapping.

One ``simpler_put_spoon_on_table_cloth`` build (eight envs, MPR collision) serves the scene tests; the layout table
itself is checked as plain data.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import itertools

import numpy as np
import pytest

from eden.managers.terms import PlaceFromLayouts
from eden.tasks import TASK_REGISTRY
from eden.tasks.simpler.bridge_table.config import BridgeTableBaseConfig, simpler_layouts
from tests.support.envs import build_env

# SIMPLER's grid centre (-0.16, 0), seen from its robot base at (0.147, 0.028) facing -x: the 15 cm square's corners.
SPOON_CORNERS = {(0.232, -0.047), (0.232, 0.103), (0.382, -0.047), (0.382, 0.103)}


def _xy(pose):
    return (round(float(pose[0]), 3), round(float(pose[1]), 3))


def test_a_layout_puts_source_and_target_on_two_different_corners_of_simplers_square():
    layouts = simpler_layouts((0.075,), (0.0, 90.0))

    assert len(layouts) == 4 * 3 * 2  # ordered corner pairs x source yaws, as SIMPLER enumerates them
    assert {_xy(layout["source"]) for layout in layouts} == SPOON_CORNERS
    assert {_xy(layout["target"]) for layout in layouts} == SPOON_CORNERS
    assert all(_xy(layout["source"]) != _xy(layout["target"]) for layout in layouts)
    # SIMPLER's yaws 0 and 90 deg, seen from a robot turned 180 deg; the target keeps SIMPLER's identity.
    assert {layout["source"][2] for layout in layouts} == {-180.0, -90.0}
    assert {layout["target"][2] for layout in layouts} == {-180.0}


def test_the_cube_layouts_use_both_of_simplers_squares():
    layouts = simpler_layouts((0.05, 0.1), (0.0,))

    assert len(layouts) == 2 * 4 * 3
    gaps = {round(float(np.hypot(*np.subtract(_xy(lay["source"]), _xy(lay["target"])))), 3) for lay in layouts}
    assert gaps == {0.1, 0.141, 0.2, 0.283}


def _table_tasks():
    """Registered tasks whose config is a BridgeTableBaseConfig (only SIMPLER's modules are imported to check)."""
    simpler = [
        n for n in TASK_REGISTRY.list_tasks() if ".simpler." in (TASK_REGISTRY.config_qualname_for_task(n) or "")
    ]
    return sorted(n for n in simpler if issubclass(TASK_REGISTRY.get(n), BridgeTableBaseConfig))


def test_the_registry_has_the_three_simpler_table_tasks():
    assert _table_tasks() == [
        "simpler_put_carrot_on_plate",
        "simpler_put_spoon_on_table_cloth",
        "simpler_stack_green_cube_on_yellow_cube",
    ]


@pytest.mark.parametrize("task", _table_tasks())
def test_every_table_task_places_both_objects_from_layouts(task):
    """The base places nothing, so a table task that forgets its layouts would start every object at its default pose."""
    events = TASK_REGISTRY.build(task).event_options
    terms = [getattr(events, key) for key in events.term_keys()]
    terms = [term for term in terms if term.name == "place_from_layouts"]
    assert len(terms) == 1, f"{task} has {len(terms)} PlaceFromLayouts events"
    assert all(sorted(layout) == ["source", "target"] for layout in terms[0].layouts), task


# --- the spoon scene ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spoon_env():
    config = TASK_REGISTRY.build("simpler_put_spoon_on_table_cloth")
    config.env_options.num_envs = 8
    config.env_options.use_gjk_collision = False
    return build_env(config, reset=True)


def test_every_reset_starts_the_spoon_and_cloth_on_one_of_simplers_layouts(spoon_env):
    pairs = {(_xy(lay["source"]), _xy(lay["target"])) for lay in simpler_layouts((0.075,), (0.0, 90.0))}
    for seed in range(3):
        spoon_env.reset(seed=seed)
        source = spoon_env.entities["source"].get_pos(relative=True).cpu().numpy()
        target = spoon_env.entities["target"].get_pos(relative=True).cpu().numpy()
        for env_idx in range(spoon_env.num_envs):
            assert (_xy(source[env_idx]), _xy(target[env_idx])) in pairs, (seed, env_idx)
            assert np.hypot(*(source[env_idx, :2] - target[env_idx, :2])) > 0.149


def test_layouts_that_place_different_entities_are_refused(spoon_env):
    with pytest.raises(ValueError, match="every layout must place the same entities"):
        PlaceFromLayouts(
            spoon_env,
            PlaceFromLayouts.configure(layouts=[{"source": (0.3, 0.0, 0.0)}, {"target": (0.3, 0.1, 0.0)}]),
        )


def test_no_layouts_are_refused(spoon_env):
    with pytest.raises(ValueError, match="needs at least one layout"):
        PlaceFromLayouts(spoon_env, PlaceFromLayouts.configure(layouts=[]))


def test_a_layout_naming_an_entity_the_scene_lacks_raises_at_build(spoon_env):
    term = PlaceFromLayouts(spoon_env, PlaceFromLayouts.configure(layouts=[{"spon": (0.3, 0.0, 0.0)}]))
    with pytest.raises(KeyError, match="spon"):
        term.build()


def test_the_draw_covers_every_layout(spoon_env):
    """Uniform over the layouts: 64 resets of 8 envs land on all 24 of them."""
    seen = set()
    for seed in itertools.islice(itertools.count(), 64):
        spoon_env.reset(seed=seed)
        source = spoon_env.entities["source"].get_pos(relative=True).cpu().numpy()
        target = spoon_env.entities["target"].get_pos(relative=True).cpu().numpy()
        seen |= {(_xy(source[i]), _xy(target[i])) for i in range(spoon_env.num_envs)}
    assert len(seen) == 12  # 12 corner pairs; the two yaws share positions
