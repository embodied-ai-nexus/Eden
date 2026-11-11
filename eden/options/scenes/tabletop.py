"""Tabletop and kitchen-countertop scene options.

These are **factory functions**, not ``SceneOptions`` subclasses: each returns a plain
``SceneOptions`` with the table/countertop and ground plane added as named children, so a
config round-trips through *validated* load (the saved/reconstructed type is a base
``SceneOptions``).
"""

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions


def _default_work_table() -> EntityOptions:
    return EntityOptions(
        file="objects/tables/work_table/work_table.obj",
        is_fixed_base=True,
        default_root_pos=(0.5, 0.0, 0.0),
    )


def _default_kitchen_counter() -> EntityOptions:
    return EntityOptions(
        file="objects/tables/kitchen_counter.glb",
        is_fixed_base=True,
        up=(0, 0, 1),
        front=(0, -1, 0),
        default_root_pos=(0.5, 0.0, 0.0),
    )


def TabletopSceneOptions(
    robot: EntityOptions,
    *,
    plane: PlaneOptions | None = None,
    table: EntityOptions | None = None,
    **entities: EntityOptions,
) -> SceneOptions:
    """Tabletop scene with a robot, a ground plane, and a work table.

    Parameters
    ----------
    robot : EntityOptions
        The entity configuration to be used for the robot.
    plane : PlaneOptions | None
        The plane entity configuration (defaults to a bare :class:`PlaneOptions`).
    table : EntityOptions | None
        The table entity configuration (defaults to ``work_table.obj``).
    **entities : EntityOptions
        Additional named entities to add to the scene.
    """
    return SceneOptions(
        robot=robot,
        plane=plane if plane is not None else PlaneOptions(),
        table=table if table is not None else _default_work_table(),
        **entities,
    )


def KitchenCountertopSceneOptions(
    robot: EntityOptions,
    *,
    plane: PlaneOptions | None = None,
    countertop: EntityOptions | None = None,
    **entities: EntityOptions,
) -> SceneOptions:
    """Kitchen countertop scene with a robot, a ground plane, and a countertop.

    Parameters
    ----------
    robot : EntityOptions
        The entity configuration to be used for the robot.
    plane : PlaneOptions | None
        The plane entity configuration (defaults to a bare :class:`PlaneOptions`).
    countertop : EntityOptions | None
        The countertop entity configuration (defaults to ``kitchen_counter.glb``).
    **entities : EntityOptions
        Additional named entities to add to the scene.
    """
    return SceneOptions(
        robot=robot,
        plane=plane if plane is not None else PlaneOptions(),
        countertop=countertop if countertop is not None else _default_kitchen_counter(),
        **entities,
    )
