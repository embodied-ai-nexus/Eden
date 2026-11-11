"""RoboCasa365 (eden_robocasa365) kitchen scene options and layout presets."""

from __future__ import annotations
from typing import Literal

from eden.options.entities import SceneOptions
from eden.options.objects.robocasa365 import (
    ROBOCASA365_KITCHEN_REGISTRY,
    ROBOCASA365_ACCESSORIES_REGISTRY,
)


ROBOCASA365_STYLES = Literal[
    "style001",
    "style002",
    "style003",
    "style004",
    "style005",
    "style006",
    "style007",
    "style008",
    "style009",
    "style010",
    "style011",
    "style012",
]


def _build_scene_kwargs(
    layout: str,
    style: str,
    accessory_names: list[str] | None = None,
    is_support_enabled: bool = True,
) -> dict:
    """Build kwargs for a RoboCasa365 scene from layout, style, accessory list."""
    if accessory_names is None:
        accessory_names = []
    kitchen_key = f"robocasa365_{layout}_{style}"
    kitchen_cls = ROBOCASA365_KITCHEN_REGISTRY.get(kitchen_key)
    kwargs = {
        "kitchen": kitchen_cls(is_support_enabled=is_support_enabled),
    }
    for name in accessory_names:
        acc_key = f"robocasa365_{name}"
        acc_cls = ROBOCASA365_ACCESSORIES_REGISTRY.get(acc_key)
        kwargs[name] = acc_cls()
    return kwargs


# ──────────────────────────────────────────────────────────────────────────────
# 10 layouts (empty — kitchen only, no accessories)
# ──────────────────────────────────────────────────────────────────────────────


def Robocasa365Layout001Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT001 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout001",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout002Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT002 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout002",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout003Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT003 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout003",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout004Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT004 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout004",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout005Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT005 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout005",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout006Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT006 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout006",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout007Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT007 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout007",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout008Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT008 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout008",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout009Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT009 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout009",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout010Empty(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT010 layout — kitchen only, no accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout010",
            style=style,
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 10 layouts (with accessories)
# ──────────────────────────────────────────────────────────────────────────────


def Robocasa365Layout001(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT001 layout — 4 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout001",
            style=style,
            accessory_names=[
                "toaster_main_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "coffee_machine_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout002(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT002 layout — 5 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout002",
            style=style,
            accessory_names=[
                "toaster_main_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "coffee_machine_main_group",
                "plant_island_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout003(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT003 layout — 6 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout003",
            style=style,
            accessory_names=[
                "toaster_oven_main_group",
                "plant_main_group",
                "knife_block_main_group",
                "paper_towel_main_group",
                "toaster_right_group",
                "coffee_machine_right_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout004(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT004 layout — 7 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout004",
            style=style,
            accessory_names=[
                "paper_towel_main_group",
                "knife_block_main_group",
                "toaster_main_group",
                "utensil_set_main_group",
                "toaster_oven_main_group",
                "coffee_machine_left_group",
                "plant_island_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout005(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT005 layout — 5 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout005",
            style=style,
            accessory_names=[
                "paper_towel_left_group",
                "coffee_machine_main_group",
                "utensil_set_main_group",
                "toaster_main_group",
                "toaster_oven_main_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout006(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT006 layout — 7 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout006",
            style=style,
            accessory_names=[
                "toaster_oven_main_group",
                "knife_block_main_group",
                "utensil_set_main_group",
                "paper_towel_main_group",
                "toaster_right_group",
                "coffee_machine_left_group",
                "plant_left_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout007(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT007 layout — 7 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout007",
            style=style,
            accessory_names=[
                "knife_block_main_group",
                "paper_towel_main_group",
                "utensil_set_main_group",
                "plant_island_group",
                "toaster_oven_left_group",
                "coffee_machine_right_group",
                "toaster_right_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout008(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT008 layout — 6 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout008",
            style=style,
            accessory_names=[
                "toaster_main_group",
                "toaster_oven_main_group",
                "knife_block_main_group",
                "utensil_set_main_group",
                "paper_towel_main_group",
                "coffee_machine_left_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout009(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT009 layout — 7 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout009",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "paper_towel_main_group",
                "toaster_main_group",
                "toaster_oven_main_group",
                "utensil_set_main_group",
                "knife_block_main_group",
                "plant_left_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )


def Robocasa365Layout010(
    style: ROBOCASA365_STYLES,
    is_support_enabled: bool = False,
    **data,
) -> SceneOptions:
    """LAYOUT010 layout — 7 accessories."""
    return SceneOptions(
        **_build_scene_kwargs(
            layout="layout010",
            style=style,
            accessory_names=[
                "coffee_machine_main_group",
                "knife_block_main_group",
                "plant_main_group",
                "toaster_main_group",
                "toaster_oven_main_group",
                "utensil_set_front_group",
                "paper_towel_dining_area_group",
            ],
            is_support_enabled=is_support_enabled,
        ),
        **data,
    )
