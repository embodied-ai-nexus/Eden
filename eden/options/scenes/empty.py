"""Empty scene presets (plane-only, single and bimanual).

These are **factory functions**, not ``SceneOptions`` subclasses: each returns a plain
``SceneOptions`` with the ground plane and robot(s) added as named children. A factory (rather
than a builder ``__init__`` on a subclass) keeps the saved/reconstructed type a base
``SceneOptions``, so a config round-trips through *validated* load — a builder ``__init__`` that
takes non-persisted args (``is_support_enabled``/``surface``) and injects entities cannot be
reconstructed from the serialized entity list.
"""

from __future__ import annotations

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions
from eden.options.surfaces import SurfaceLike


def _ground_plane(is_support_enabled: bool, surface: SurfaceLike | None) -> PlaneOptions:
    return PlaneOptions(surface=surface, is_support_enabled=is_support_enabled)


def _ground(entities: dict, is_support_enabled: bool, surface: SurfaceLike | None):
    """Pop a caller-supplied ``plane`` out of ``entities`` and return the ground to use.

    ``plane`` is the stable name for whatever the robot stands on, so passing one (e.g.
    ``TerrainOptions``) replaces the flat default and everything addressing the ground by name
    keeps working. ``is_support_enabled`` / ``surface`` configure the default plane only.
    """
    override = entities.pop("plane", None)
    if override is None:
        return _ground_plane(is_support_enabled, surface)
    if is_support_enabled or surface is not None:
        raise ValueError(
            "plane=... replaces the preset's ground, so is_support_enabled / surface have "
            "nothing to configure. Set them on the entity you pass as `plane` instead."
        )
    return override


def EmptySceneOptions(
    robot: EntityOptions,
    *,
    is_support_enabled: bool = False,
    surface: SurfaceLike | None = None,
    **entities: EntityOptions,
) -> SceneOptions:
    """Basic scene with a robot and a ground plane.

    Parameters
    ----------
    robot : EntityOptions
        The robot entity configuration.
    is_support_enabled : bool
        Whether to enable support-surface tracking on the plane.
    surface : SurfaceLike | None
        Surface for the plane, e.g. ``PlasticSurfaceOptions(color=(1.0, 1.0, 1.0))``.
    **entities : EntityOptions
        Additional named entities to add to the scene. Passing ``plane=`` replaces the flat
        ground with that entity (e.g. ``TerrainOptions``) under the same name.
    """
    ground = _ground(entities, is_support_enabled, surface)  # pops a `plane` override out of entities
    return SceneOptions(robot=robot, plane=ground, **entities)


def BimanualEmptySceneOptions(
    right_robot: EntityOptions,
    left_robot: EntityOptions,
    *,
    is_support_enabled: bool = False,
    surface: SurfaceLike | None = None,
    **entities: EntityOptions,
) -> SceneOptions:
    """Basic bimanual scene with two robots and a ground plane.

    Parameters
    ----------
    right_robot, left_robot : EntityOptions
        The right/left robot entity configurations.
    is_support_enabled : bool
        Whether to enable support-surface tracking on the plane.
    surface : SurfaceLike | None
        Surface for the plane.
    **entities : EntityOptions
        Additional named entities to add to the scene. Passing ``plane=`` replaces the flat
        ground with that entity (e.g. ``TerrainOptions``) under the same name.
    """
    ground = _ground(entities, is_support_enabled, surface)  # pops a `plane` override out of entities
    return SceneOptions(
        right_robot=right_robot,
        left_robot=left_robot,
        plane=ground,
        **entities,
    )
