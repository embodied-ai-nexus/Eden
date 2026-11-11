"""The robosuite demonstrator bindings: what each task gets, and the PickPlace routine run on its real scene."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest

from eden.extensions.scripted_demos.block_world import LiftController, TowerController
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.extensions.scripted_demos.routines import RoutineDemonstrator
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.pick_place.config import PickPlaceObjectMod
from eden.tasks.robosuite.pick_place.routine import _quadrant_of
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def bread_env():
    config = PickPlaceObjectMod(object="bread").apply(TASK_REGISTRY.build("robosuite_pick_place"))
    config.env_options.num_envs = 2
    config.env_options.episode_length_s = 60.0  # the routine takes ~25 s; robosuite's RL episodes are shorter
    return build_env(config, reset=True)


def test_each_robosuite_task_with_a_demonstrator_gets_its_kind():
    assert SCRIPTED_DEMO_REGISTRY.get("robosuite_stack").__annotations__["return"] in (
        "TowerController",
        TowerController,
    )
    assert SCRIPTED_DEMO_REGISTRY.get("robosuite_lift").__annotations__["return"] in ("LiftController", LiftController)
    # Sorting four objects takes ~2 min, past the generator's 90 s default.
    assert SCRIPTED_DEMO_REGISTRY.get("robosuite_pick_place").episode_length_s == 200.0


def test_the_pick_place_routine_sorts_the_objects_present(bread_env):
    # Seeded: the reset's draw otherwise depends on the tests run before this one, and some put the
    # bread in bin1's far corner, where the hand pointing down stops 4 cm short of it.
    bread_env.reset(seed=0)
    demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_pick_place")(bread_env, seed=0, p_mistake=0.0)
    assert isinstance(demo, RoutineDemonstrator)
    demo.reset()
    for _ in range(1200):
        if bool(demo.update_subgoals().all()):
            break
        bread_env.step(demo.act())

    assert demo.done.all()
    assert bread_env.metric_manager.success_buf.all()
    for env_id in range(2):
        assert _quadrant_of(demo.object_footprint("bread", env_id)) == "bread"
        assert any("# bread -> goal quadrant" in line for line in demo.trace(env_id))


def test_camera_only_pick_place_asks_for_a_view_per_env(bread_env):
    with pytest.raises(ValueError, match=r"Camera 'agentview' renders 1 view\(s\) for 2 envs.*env_separate_rigid"):
        SCRIPTED_DEMO_REGISTRY.get("robosuite_pick_place")(bread_env, seed=0, privileged=False)
