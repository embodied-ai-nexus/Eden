"""RoboLab basic object options (auto-generated from object_catalog.json).

Assets converted from the RoboLab USD library to URDF; hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabObjectOptions


@ROBOLAB_REGISTRY.register(name="basic/blue_block")
class RoboLabBasicBlueBlock(RoboLabObjectOptions):
    """This is a blue colored block."""

    file: str = "objects/basic/blue_block/blue_block.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="block")


@ROBOLAB_REGISTRY.register(name="basic/green_block")
class RoboLabBasicGreenBlock(RoboLabObjectOptions):
    """This is a green colored block."""

    file: str = "objects/basic/green_block/green_block.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="block")


@ROBOLAB_REGISTRY.register(name="basic/red_block")
class RoboLabBasicRedBlock(RoboLabObjectOptions):
    """This is a red colored block."""

    file: str = "objects/basic/red_block/red_block.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="block")


@ROBOLAB_REGISTRY.register(name="basic/yellow_block")
class RoboLabBasicYellowBlock(RoboLabObjectOptions):
    """This is a yellow colored block."""

    file: str = "objects/basic/yellow_block/yellow_block.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="block")
