"""``eden/tasks/polaris/rubric.py``: every Polaris task scores PolaRiS's own rubric.

The criteria, their dependencies, and the names they check are PolaRiS's
(``polaris/environments/__init__.py``), mapped onto this scene's entity names.
"""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY

TASKS = [
    "polaris_block_stack_kitchen",
    "polaris_food_bussing",
    "polaris_move_latte_cup",
    "polaris_organize_tools",
    "polaris_pan_clean",
    "polaris_tape_into_container",
]


def _criteria(task):
    term = TASK_REGISTRY.build(task).metric_options.rubric
    assert term.name == "rubric_metric_term"
    return {criterion.name: criterion.after for criterion in term.criteria}


def _put_in(source, container):
    return {
        f"reach_{source}": [],
        f"lift_{source}": [f"reach_{source}"],
        f"{source}_in_{container}": [f"lift_{source}"],
    }


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("polaris_move_latte_cup", _put_in("latte_art_cup", "cutting_board")),
        ("polaris_organize_tools", _put_in("scissor", "container_00")),
        ("polaris_pan_clean", _put_in("sponge", "frying_pan")),
        ("polaris_tape_into_container", _put_in("tape_00", "container_02")),
        ("polaris_food_bussing", {**_put_in("ice_cream", "yellow_bowl"), **_put_in("grapes", "yellow_bowl")}),
        (
            "polaris_block_stack_kitchen",
            {
                **_put_in("green_cube", "green_tray"),
                **_put_in("wood_cube", "green_tray"),
                "green_cube_in_wood_cube": ["green_cube_in_green_tray", "wood_cube_in_green_tray"],
            },
        ),
    ],
)
def test_each_task_scores_polaris_rubric(task, expected):
    assert _criteria(task) == expected


def test_a_saved_rubric_loads_back_whole(tmp_path):
    """What ``eden inference`` replays: the criteria and their nested AND / function terms survive ``config.json``."""
    config = TASK_REGISTRY.build("polaris_block_stack_kitchen_batched")
    path = str(tmp_path / "config.json")
    config.save_as_json(path, lock_config=False)
    loaded = type(config).load_from_file(path)
    assert loaded.metric_options.rubric.model_dump() == config.metric_options.rubric.model_dump()


@pytest.mark.parametrize("task", TASKS)
def test_the_batched_variant_scores_the_same_rubric(task):
    assert _criteria(task + "_batched") == _criteria(task)


def _named_entities(dumped):
    """Every entity a dumped term option names, its child terms' included."""
    if isinstance(dumped, list):
        return set().union(*map(_named_entities, dumped)) if dumped else set()
    if not isinstance(dumped, dict):
        return set()
    names = {dumped[key] for key in ("entity_name", "entity_a_name", "entity_b_name", "robot_name") if key in dumped}
    return names.union(*(_named_entities(value) for value in dumped.values()))


@pytest.mark.parametrize("task", TASKS)
def test_every_criterion_names_entities_of_the_scene(task):
    """A misspelt entity would only raise at env build; the scene's own names (plus the welded gripper) are the check."""
    config = TASK_REGISTRY.build(task)
    scene = set(config.scene_options.model_dump(exclude_none=True)) | {"gripper"}
    named = _named_entities([criterion.term.model_dump() for criterion in config.metric_options.rubric.criteria])
    assert "gripper" in named and len(named) >= 3
    assert named <= scene, named - scene
