"""RoboCasa kitchen scene options and layout presets."""

from __future__ import annotations
from typing import Literal

from eden.options.entities import SceneOptions
from eden.options.objects.robocasa import ROBOCASA_KITCHEN_REGISTRY, ROBOCASA_ACCESSORIES_REGISTRY


ROBOCASA_STYLES = Literal[
    "industrial",
    "scandinavian",
    "coastal",
    "modern_1",
    "modern_2",
    "traditional_1",
    "traditional_2",
    "farmhouse",
    "rustic",
    "mediterranean",
    "transitional_1",
    "transitional_2",
]


def _build_scene_kwargs(
    layout: str,
    style: str,
    accessory_names: list[str] | None = None,
    is_support_enabled: bool = True,
) -> dict:
    """Build keyword arguments for a RoboCasa scene from layout, style, and accessory list."""
    if accessory_names is None:
        accessory_names = []
    kitchen_key = f"robocasa_{layout}_{style}"
    kitchen_cls = ROBOCASA_KITCHEN_REGISTRY.get(kitchen_key)
    kwargs = {
        "kitchen": kitchen_cls(is_support_enabled=is_support_enabled),
    }
    for name in accessory_names:
        acc_key = f"robocasa_{name}"
        acc_cls = ROBOCASA_ACCESSORIES_REGISTRY.get(acc_key)
        kwargs[name] = acc_cls()
    return kwargs


# ──────────────────────────────────────────────────────────────────────────────
# 10 layouts (empty — kitchen only, no accessories)
# ──────────────────────────────────────────────────────────────────────────────


def RobocasaOneWallSmallEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """ONE_WALL_SMALL layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="one_wall_small",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaOneWallLargeEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """ONE_WALL_LARGE layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="one_wall_large",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaLShapedSmallEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """L_SHAPED_SMALL layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="l_shaped_small",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaLShapedLargeEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """L_SHAPED_LARGE layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="l_shaped_large",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaGalleyEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """GALLEY layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="galley",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaUShapedSmallEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """U_SHAPED_SMALL layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="u_shaped_small",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaUShapedLargeEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """U_SHAPED_LARGE layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="u_shaped_large",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaGShapedSmallEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """G_SHAPED_SMALL layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="g_shaped_small",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaGShapedLargeEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """G_SHAPED_LARGE layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="g_shaped_large",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaWraparoundEmpty(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """WRAPAROUND layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="wraparound",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 10 layouts (with accessories)
# ──────────────────────────────────────────────────────────────────────────────


def RobocasaOneWallSmall(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """ONE_WALL_SMALL layout — 4 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="one_wall_small",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "toaster_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaOneWallLarge(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """ONE_WALL_LARGE layout — 4 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="one_wall_large",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "toaster_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaLShapedSmall(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """L_SHAPED_SMALL layout — 5 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="l_shaped_small",
            style=style,
            accessory_names=[
                "coffee_machine_right_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "plant_main_group",
                "toaster_right_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaLShapedLarge(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """L_SHAPED_LARGE layout — 6 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="l_shaped_large",
            style=style,
            accessory_names=[
                "coffee_machine_left_group",
                "knife_block_left_group",
                "paper_towel_left_group",
                "plant_left_group",
                "toaster_main_group",
                "utensil_holder_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaGalley(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """GALLEY layout — 5 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="galley",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "knife_block_main_group",
                "paper_towel_left_group",
                "toaster_main_group",
                "utensil_holder_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaUShapedSmall(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """U_SHAPED_SMALL layout — 6 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="u_shaped_small",
            style=style,
            accessory_names=[
                "coffee_machine_left_group",
                "knife_block_right_group",
                "paper_towel_left_group",
                "plant_main_group",
                "toaster_main_group",
                "utensil_holder_right_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaUShapedLarge(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """U_SHAPED_LARGE layout — 6 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="u_shaped_large",
            style=style,
            accessory_names=[
                "coffee_machine_right_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "plant_island_group",
                "toaster_right_group",
                "utensil_holder_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaGShapedSmall(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """G_SHAPED_SMALL layout — 6 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="g_shaped_small",
            style=style,
            accessory_names=[
                "coffee_machine_left_group",
                "knife_block_main_group",
                "paper_towel_right_group",
                "plant_front_group",
                "toaster_main_group",
                "utensil_holder_left_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaGShapedLarge(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """G_SHAPED_LARGE layout — 5 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="g_shaped_large",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "paper_towel_left_group",
                "plant_left_group",
                "toaster_main_group",
                "utensil_holder_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def RobocasaWraparound(
    style: ROBOCASA_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """WRAPAROUND layout — 5 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="wraparound",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "paper_towel_right_group",
                "plant_main_group",
                "toaster_main_group",
                "utensil_holder_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )
