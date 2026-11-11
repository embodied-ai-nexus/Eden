"""SIMPLER benchmark object options."""

from __future__ import annotations

from typing import ClassVar

import genesis as gs
from genesis.options.surfaces import Surface
from genesis.typing import Vec3FType
from pydantic import Field

from eden.options.entities import BoxOptions, EntityOptions, MetadataOptions


class SimplerObjectOptions(EntityOptions):
    """Simpler object options."""

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    dataset: str = "eden_simpler"
    assets_subdir: ClassVar[str] = "simpler"


class SimplerCarrot(SimplerObjectOptions):
    file: str = "models/bridge_carrot_generated_modified/object.urdf"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=700)


class SimplerSpoon(SimplerObjectOptions):
    file: str = "models/bridge_spoon_generated_modified/object.urdf"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=1200)


class SimplerEggplant(SimplerObjectOptions):
    file: str = "models/eggplant/object.urdf"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=400,
    )


class SimplerTableCloth(SimplerObjectOptions):
    file: str = "models/table_cloth_generated/object.urdf"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=330,
    )


class SimplerGreenCube3cm(BoxOptions):
    size: Vec3FType = (0.03, 0.03, 0.03)
    # NOTE: passing diffuse_texture directly (not the `color=` shortcut). Genesis re-runs
    # Surface._resolve_shortcuts during parent-field validation; if `color` is still set on
    # the second pass it conflicts with the auto-resolved diffuse_texture and raises.
    surface: Surface = Field(
        default_factory=lambda: gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ColorTexture(color=(62 / 255.0, 89 / 255.0, 73 / 255.0)),
        )
    )
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=1000,
    )


class SimplerYellowCube3cm(BoxOptions):
    size: Vec3FType = (0.03, 0.03, 0.03)
    surface: Surface = Field(
        default_factory=lambda: gs.surfaces.Plastic(
            diffuse_texture=gs.textures.ColorTexture(color=(221 / 255.0, 175 / 255.0, 91 / 255.0)),
        )
    )
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=1000,
    )


class SimplerPlateObjaverse(SimplerObjectOptions):
    file: str = "models/bridge_plate_objaverse/object.urdf"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=1000,
    )


class SimplerPlateObjaverseLarger(SimplerObjectOptions):
    file: str = "models/bridge_plate_objaverse_larger/object.urdf"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=1000,
    )


class SimplerSink(SimplerObjectOptions):
    file: str = "models/sink/object.urdf"
    front: tuple[int, int, int] = (-1, 0, 0)
    is_fixed_base: bool = True
    scale: float = 1.15
    vis_mode: str = "collision"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        density=1000,
    )
