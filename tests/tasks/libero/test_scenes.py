"""``make_libero_scene_options``: each problem family's scene preset, filled with a task's BDDL objects."""

from __future__ import annotations

import dataclasses

import pytest

from eden.options.entities import SceneOptions
from eden.options.objects.libero import LiberoTabletopWorkspace
from eden.options.robots.libero import LiberoMountedPanda, LiberoPanda
from eden.tasks.libero import catalog
from eden.tasks.libero.scenes import make_libero_scene_options


def _empty_task_of_family(problem_name):
    """A catalog spec moved to ``problem_name`` with no BDDL objects (the coffee-table family has no task)."""
    task_spec = catalog.get_task_spec(catalog.get_task_names()[0])
    return dataclasses.replace(task_spec, problem_name=problem_name, entity_categories={}, fixture_names=frozenset())


def test_every_catalog_task_builds_a_scene_holding_all_its_entities():
    for task_name in catalog.get_task_names():
        task_spec = catalog.get_task_spec(task_name)
        scene = make_libero_scene_options(task_spec)
        missing = [name for name in task_spec.entity_categories if not hasattr(scene, name)]
        assert not missing, f"{task_name}: scene lacks {missing}"


@pytest.mark.parametrize(
    ("problem_name", "workspace_field", "robot_field", "robot_type"),
    [
        ("LIBERO_Tabletop_Manipulation", "main_table", "mount", LiberoMountedPanda),
        ("LIBERO_Floor_Manipulation", "floor", "robot", LiberoPanda),
        ("LIBERO_Kitchen_Tabletop_Manipulation", "kitchen_table", "mount", LiberoMountedPanda),
        ("LIBERO_Living_Room_Tabletop_Manipulation", "living_room_table", "robot", LiberoPanda),
        ("LIBERO_Study_Tabletop_Manipulation", "study_table", "mount", LiberoMountedPanda),
        ("LIBERO_Coffee_Table_Manipulation", "coffee_table", "robot", LiberoPanda),
    ],
)
def test_family_presets_own_the_workspace_and_robot_topology(problem_name, workspace_field, robot_field, robot_type):
    scene = make_libero_scene_options(_empty_task_of_family(problem_name))

    assert type(scene) is SceneOptions
    assert hasattr(scene, workspace_field)
    assert isinstance(getattr(scene, robot_field), robot_type)


def test_unknown_problem_family_is_rejected():
    with pytest.raises(KeyError, match="Unsupported LIBERO problem family"):
        make_libero_scene_options(_empty_task_of_family("unknown"))


def test_workspace_comes_from_the_preset_and_fixtures_batch_their_vertices():
    scene = make_libero_scene_options(catalog.get_task_spec("turn_on_the_stove"))

    assert type(scene.main_table) is LiberoTabletopWorkspace  # not a registry "table" object
    assert scene.flat_stove_1.batch_fixed_verts is True  # a BDDL fixture
    assert scene.akita_black_bowl_1.batch_fixed_verts is False  # a movable object
