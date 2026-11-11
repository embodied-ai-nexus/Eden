"""CALVIN play-table scene options."""

from __future__ import annotations

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions
from eden.options.objects.calvin import CalvinBlueBlock, CalvinPinkBlock, CalvinPlayTable, CalvinRedBlock


def CalvinPlayTableSceneOptions(robot: EntityOptions | None = None, **kwargs: EntityOptions) -> SceneOptions:
    """CALVIN play-table scene built from the original table-D and block URDFs.

    Returns a plain :class:`SceneOptions` with the play table and the three
    colored blocks added as named children so the config round-trips through
    validated load.
    """
    scene_kwargs: dict[str, EntityOptions] = {
        "plane": PlaneOptions(),
        "table": CalvinPlayTable(default_root_pos=(0.0, 0.0, 0.0)),
        "block_red": CalvinRedBlock(default_root_pos=(-0.12, -0.12, 0.49)),
        "block_blue": CalvinBlueBlock(default_root_pos=(0.0, -0.12, 0.49)),
        "block_pink": CalvinPinkBlock(default_root_pos=(0.12, -0.12, 0.49)),
    }
    if robot is not None:
        scene_kwargs["robot"] = robot
    return SceneOptions(**scene_kwargs, **kwargs)


__all__ = ["CalvinPlayTableSceneOptions"]
