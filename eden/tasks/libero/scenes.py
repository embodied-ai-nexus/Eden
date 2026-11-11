"""Build task-specific LIBERO scenes from BDDL metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from eden.options.entities import BaseEntityOptions, CompositeEntityOptions, SceneOptions
from eden.options.objects.libero import (
    LIBERO_OBJECT_REGISTRY,
    LiberoCoffeeTableWorkspace,
    LiberoFloorWorkspace,
    LiberoKitchenTabletopWorkspace,
    LiberoLivingRoomTabletopWorkspace,
    LiberoStudyTabletopWorkspace,
    LiberoTabletopWorkspace,
)
from eden.options.scenes.libero import (
    LiberoCoffeeTableSceneOptions,
    LiberoFloorSceneOptions,
    LiberoKitchenTabletopSceneOptions,
    LiberoLivingRoomTabletopSceneOptions,
    LiberoStudyTabletopSceneOptions,
    LiberoTabletopSceneOptions,
)

if TYPE_CHECKING:
    from eden.tasks.libero.catalog import LiberoTaskSpec


@dataclass(frozen=True)
class _LiberoFamilySpec:
    """Task protocol metadata associated with one reusable scene preset."""

    scene_factory: Callable[..., SceneOptions]
    workspace_entity_name: str
    workspace_predicate_origin_w: tuple[float, float, float]
    agentview_pos_w: tuple[float, float, float]
    agentview_quat_w: tuple[float, float, float, float]
    has_directional_light: bool = False


_FAMILY_SPECS_BY_PROBLEM = {
    "LIBERO_Tabletop_Manipulation": _LiberoFamilySpec(
        scene_factory=LiberoTabletopSceneOptions,
        workspace_entity_name=LiberoTabletopWorkspace.workspace_entity_name,
        workspace_predicate_origin_w=(0.0, 0.0, 0.895),
        agentview_pos_w=(0.6586131746834771, 0.0, 1.6103500240372423),
        agentview_quat_w=(0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
    ),
    "LIBERO_Floor_Manipulation": _LiberoFamilySpec(
        scene_factory=LiberoFloorSceneOptions,
        workspace_entity_name=LiberoFloorWorkspace.workspace_entity_name,
        workspace_predicate_origin_w=(0.0, 0.0, 0.02),
        agentview_pos_w=(0.8965773716836134, 5.216182733499864e-07, 0.65),
        agentview_quat_w=(0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979),
    ),
    "LIBERO_Kitchen_Tabletop_Manipulation": _LiberoFamilySpec(
        scene_factory=LiberoKitchenTabletopSceneOptions,
        workspace_entity_name=LiberoKitchenTabletopWorkspace.workspace_entity_name,
        workspace_predicate_origin_w=(0.0, 0.0, 0.875),
        agentview_pos_w=(0.6586131746834771, 0.0, 1.6103500240372423),
        agentview_quat_w=(0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
        has_directional_light=True,
    ),
    "LIBERO_Living_Room_Tabletop_Manipulation": _LiberoFamilySpec(
        scene_factory=LiberoLivingRoomTabletopSceneOptions,
        workspace_entity_name=LiberoLivingRoomTabletopWorkspace.workspace_entity_name,
        workspace_predicate_origin_w=(0.0, 0.0, 0.41),
        agentview_pos_w=(0.6065773716836134, 0.0, 0.96),
        agentview_quat_w=(0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979),
    ),
    "LIBERO_Study_Tabletop_Manipulation": _LiberoFamilySpec(
        scene_factory=LiberoStudyTabletopSceneOptions,
        workspace_entity_name=LiberoStudyTabletopWorkspace.workspace_entity_name,
        workspace_predicate_origin_w=(-0.2, 0.0, 0.867),
        agentview_pos_w=(0.4586131746834771, 0.0, 1.6103500240372423),
        agentview_quat_w=(0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
    ),
    "LIBERO_Coffee_Table_Manipulation": _LiberoFamilySpec(
        scene_factory=LiberoCoffeeTableSceneOptions,
        workspace_entity_name=LiberoCoffeeTableWorkspace.workspace_entity_name,
        workspace_predicate_origin_w=(0.0, 0.0, 0.41),
        agentview_pos_w=(1.5, 0.0, 0.9),
        agentview_quat_w=(0.56, 0.43, 0.43, 0.56),
    ),
}


def _get_libero_family_spec(problem_name: str) -> _LiberoFamilySpec:
    try:
        return _FAMILY_SPECS_BY_PROBLEM[problem_name]
    except KeyError as exc:
        raise KeyError(f"Unsupported LIBERO problem family: {problem_name!r}") from exc


def make_libero_scene_options(
    task_spec: LiberoTaskSpec,
    **extra_entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Build a family preset with the fixtures and movable objects declared by BDDL."""
    family_spec = _get_libero_family_spec(task_spec.problem_name)
    objects: dict[str, BaseEntityOptions] = {}
    staging_origin = (2.5, -0.8, 1.25)
    staging_step = 0.24
    staging_columns = 6
    staging_index = 0
    for entity_name, category_name in task_spec.entity_categories.items():
        if entity_name == family_spec.workspace_entity_name:
            continue
        entity_options = LIBERO_OBJECT_REGISTRY.get(category_name)(
            default_root_pos=(
                staging_origin[0] + (staging_index % staging_columns) * staging_step,
                staging_origin[1] + (staging_index // staging_columns) * staging_step,
                staging_origin[2],
            ),
            batch_fixed_verts=entity_name in task_spec.fixture_names,
        )
        if entity_name in task_spec.fixture_names and not entity_options.is_fixed_base:
            raise ValueError(f"LIBERO fixture {entity_name!r} ({category_name!r}) must use a fixed-base registry entry")
        objects[entity_name] = entity_options
        staging_index += 1
    return family_spec.scene_factory(**objects, **extra_entities)


__all__ = ["make_libero_scene_options"]
