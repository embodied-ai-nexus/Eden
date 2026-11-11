"""OMOMO dataset object options (rigid interaction objects)."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions
from eden.utils.registry import Registry

# NOTE: a registry used for datasets/omomo to load the corresponding object in reference motion
OMOMO_OBJECTS_REGISTRY = Registry("OMOMO_OBJECTS")


class OMOMOObjectOptions(EntityOptions):
    """OMOMO object options. These are rigid (non-articulated) mesh objects."""

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_articulated: bool = False

    registry: str = "Kashu7100"
    dataset: str = "eden_omomo_holosoma"
    assets_subdir: ClassVar[str] = "omomo"


@OMOMO_OBJECTS_REGISTRY.register(name="clothesstand")
class OMOMOClothesstand(OMOMOObjectOptions):
    file: str = "captured_objects/clothesstand_cleaned_simplified.obj"
    scale: float = 0.2746


@OMOMO_OBJECTS_REGISTRY.register(name="floorlamp")
class OMOMOFloorlamp(OMOMOObjectOptions):
    file: str = "captured_objects/floorlamp_cleaned_simplified.obj"
    scale: float = 0.3793


@OMOMO_OBJECTS_REGISTRY.register(name="largebox")
class OMOMOLargebox(OMOMOObjectOptions):
    file: str = "captured_objects/largebox_cleaned_simplified.obj"
    scale: float = 0.3486


@OMOMO_OBJECTS_REGISTRY.register(name="largetable")
class OMOMOLargetable(OMOMOObjectOptions):
    file: str = "captured_objects/largetable_cleaned_simplified.obj"
    scale: float = 0.0258


@OMOMO_OBJECTS_REGISTRY.register(name="monitor")
class OMOMOMonitor(OMOMOObjectOptions):
    file: str = "captured_objects/monitor_cleaned_simplified.obj"
    scale: float = 0.1580


@OMOMO_OBJECTS_REGISTRY.register(name="mop")
class OMOMOMop(OMOMOObjectOptions):
    file: str = "captured_objects/mop_cleaned_simplified.obj"
    scale: float = 0.1676


@OMOMO_OBJECTS_REGISTRY.register(name="plasticbox")
class OMOMOPlasticbox(OMOMOObjectOptions):
    file: str = "captured_objects/plasticbox_cleaned_simplified.obj"
    scale: float = 0.0252


@OMOMO_OBJECTS_REGISTRY.register(name="smallbox")
class OMOMOSmallbox(OMOMOObjectOptions):
    file: str = "captured_objects/smallbox_cleaned_simplified.obj"
    scale: float = 0.3142


@OMOMO_OBJECTS_REGISTRY.register(name="smalltable")
class OMOMOSmalltable(OMOMOObjectOptions):
    file: str = "captured_objects/smalltable_cleaned_simplified.obj"
    scale: float = 0.0163


@OMOMO_OBJECTS_REGISTRY.register(name="suitcase")
class OMOMOSuitcase(OMOMOObjectOptions):
    file: str = "captured_objects/suitcase_cleaned_simplified.obj"
    scale: float = 0.3676


@OMOMO_OBJECTS_REGISTRY.register(name="trashcan")
class OMOMOTrashcan(OMOMOObjectOptions):
    file: str = "captured_objects/trashcan_cleaned_simplified.obj"
    scale: float = 0.2319


@OMOMO_OBJECTS_REGISTRY.register(name="tripod")
class OMOMOTripod(OMOMOObjectOptions):
    file: str = "captured_objects/tripod_cleaned_simplified.obj"
    scale: float = 0.2240


@OMOMO_OBJECTS_REGISTRY.register(name="vacuum")
class OMOMOVacuum(OMOMOObjectOptions):
    file: str = "captured_objects/vacuum_cleaned_simplified.obj"
    scale: float = 0.1915


@OMOMO_OBJECTS_REGISTRY.register(name="whitechair")
class OMOMOWhitechair(OMOMOObjectOptions):
    file: str = "captured_objects/whitechair_cleaned_simplified.obj"
    scale: float = 0.3131


@OMOMO_OBJECTS_REGISTRY.register(name="woodchair")
class OMOMOWoodchair(OMOMOObjectOptions):
    file: str = "captured_objects/woodchair_cleaned_simplified.obj"
    scale: float = 0.0249
