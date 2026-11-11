"""Reusable scene presets for the six LIBERO workspaces."""

from __future__ import annotations

from eden.options.entities import BaseEntityOptions, CompositeEntityOptions, SceneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.libero import (
    LiberoCoffeeTableWorkspace,
    LiberoFloorWorkspace,
    LiberoKitchenTabletopWorkspace,
    LiberoLivingRoomTabletopWorkspace,
    LiberoStudyTabletopWorkspace,
    LiberoTabletopWorkspace,
)
from eden.options.objects.robosuite import RethinkMount
from eden.options.robots.libero import (
    LIBERO_MOUNT_ENTITY_NAME,
    LIBERO_ROBOT_ENTITY_NAME,
    LiberoMountedPanda,
    LiberoPanda,
)


def _mounted_panda(root_pos: tuple[float, float, float]) -> LiberoMountedPanda:
    return LiberoMountedPanda(
        mount=RethinkMount(
            default_root_pos=root_pos,
            material=RigidMaterialOptions(gravity_compensation=1.0),
        )
    )


def LiberoTabletopSceneOptions(
    mount: LiberoMountedPanda | None = None,
    **entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Return the mounted-Panda tabletop preset."""
    return SceneOptions(
        **{
            LiberoTabletopWorkspace.workspace_entity_name: LiberoTabletopWorkspace(),
            LIBERO_MOUNT_ENTITY_NAME: _mounted_panda((-0.66, 0.0, 0.922)) if mount is None else mount,
        },
        **entities,
    )


def LiberoFloorSceneOptions(
    robot: LiberoPanda | None = None,
    **entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Return the fixed-base Panda floor preset."""
    return SceneOptions(
        **{
            LiberoFloorWorkspace.workspace_entity_name: LiberoFloorWorkspace(),
            LIBERO_ROBOT_ENTITY_NAME: (
                LiberoPanda(default_root_pos=(-0.6, 0.0, 0.0), is_fixed_base=True) if robot is None else robot
            ),
        },
        **entities,
    )


def LiberoKitchenTabletopSceneOptions(
    mount: LiberoMountedPanda | None = None,
    **entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Return the mounted-Panda kitchen-tabletop preset."""
    return SceneOptions(
        **{
            LiberoKitchenTabletopWorkspace.workspace_entity_name: LiberoKitchenTabletopWorkspace(),
            LIBERO_MOUNT_ENTITY_NAME: _mounted_panda((-0.66, 0.0, 0.922)) if mount is None else mount,
        },
        **entities,
    )


def LiberoLivingRoomTabletopSceneOptions(
    robot: LiberoPanda | None = None,
    **entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Return the fixed-base Panda living-room tabletop preset."""
    return SceneOptions(
        **{
            LiberoLivingRoomTabletopWorkspace.workspace_entity_name: LiberoLivingRoomTabletopWorkspace(),
            LIBERO_ROBOT_ENTITY_NAME: (
                LiberoPanda(default_root_pos=(-0.51, 0.0, 0.42), is_fixed_base=True) if robot is None else robot
            ),
        },
        **entities,
    )


def LiberoStudyTabletopSceneOptions(
    mount: LiberoMountedPanda | None = None,
    **entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Return the mounted-Panda study-tabletop preset."""
    return SceneOptions(
        **{
            LiberoStudyTabletopWorkspace.workspace_entity_name: LiberoStudyTabletopWorkspace(),
            LIBERO_MOUNT_ENTITY_NAME: _mounted_panda((-0.75, 0.0, 0.922)) if mount is None else mount,
        },
        **entities,
    )


def LiberoCoffeeTableSceneOptions(
    robot: LiberoPanda | None = None,
    **entities: BaseEntityOptions | CompositeEntityOptions,
) -> SceneOptions:
    """Return the fixed-base Panda coffee-table preset."""
    return SceneOptions(
        **{
            LiberoCoffeeTableWorkspace.workspace_entity_name: LiberoCoffeeTableWorkspace(),
            LIBERO_ROBOT_ENTITY_NAME: (
                LiberoPanda(default_root_pos=(-0.51, 0.0, 0.41), is_fixed_base=True) if robot is None else robot
            ),
        },
        **entities,
    )


__all__ = [
    "LiberoCoffeeTableSceneOptions",
    "LiberoFloorSceneOptions",
    "LiberoKitchenTabletopSceneOptions",
    "LiberoLivingRoomTabletopSceneOptions",
    "LiberoStudyTabletopSceneOptions",
    "LiberoTabletopSceneOptions",
]
