"""RoboLab fruits veggies object options (auto-generated from object_catalog.json).

Assets converted from the RoboLab USD library to URDF; hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabObjectOptions


@ROBOLAB_REGISTRY.register(name="fruits_veggies/avocado01")
class RoboLabFruitsVeggiesAvocado01(RoboLabObjectOptions):
    """This avocado has a dark green, bumpy skin with a pear-like shape. Its interior reveals a creamy, light green flesh surrounding a large, smooth brown seed."""

    file: str = "objects/fruits_veggies/avocado01/avocado01.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/lemon_01")
class RoboLabFruitsVeggiesLemon01(RoboLabObjectOptions):
    """This lemon is oval-shaped with a bright yellow, slightly textured rind. It has a fresh, citrus scent and a small protruding stem at one end."""

    file: str = "objects/fruits_veggies/lemon_01/lemon_01.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/lemon_02")
class RoboLabFruitsVeggiesLemon02(RoboLabObjectOptions):
    """This lemon is oval-shaped with a textured, bright yellow rind. It has a slightly pointed end and a fresh, citrus aroma, characteristic of its juicy and tangy interior."""

    file: str = "objects/fruits_veggies/lemon_02/lemon_02.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/lime01")
class RoboLabFruitsVeggiesLime01(RoboLabObjectOptions):
    """This lime is small and round with a vibrant green, slightly bumpy skin. It has an opaque appearance and a fresh, citrusy aroma, typical of organic limes."""

    file: str = "objects/fruits_veggies/lime01/lime01.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/lychee01")
class RoboLabFruitsVeggiesLychee01(RoboLabObjectOptions):
    """This lychee has a rough, textured pinkish-red skin with small, bumpy protrusions. Inside, it reveals a translucent white, juicy flesh surrounding a glossy brown seed."""

    file: str = "objects/fruits_veggies/lychee01/lychee01.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/orange_01")
class RoboLabFruitsVeggiesOrange01(RoboLabObjectOptions):
    """This orange is spherical with a textured, bright orange skin and a small, green stem. Its surface is dimpled, typical of citrus fruits, and it exudes a fresh, citrus scent."""

    file: str = "objects/fruits_veggies/orange_01/orange_01.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/orange_02")
class RoboLabFruitsVeggiesOrange02(RoboLabObjectOptions):
    """This orange has a vibrant, textured orange skin and a small, green stem. Its round shape and dimpled surface are characteristic of fresh citrus fruits."""

    file: str = "objects/fruits_veggies/orange_02/orange_02.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/pomegranate01")
class RoboLabFruitsVeggiesPomegranate01(RoboLabObjectOptions):
    """This pomegranate is round and slightly uneven, with a deep red, leathery skin. It features a distinctive crown-like calyx at the top and contains numerous juicy, ruby-red seeds inside."""

    file: str = "objects/fruits_veggies/pomegranate01/pomegranate01.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="pomegranate")


@ROBOLAB_REGISTRY.register(name="fruits_veggies/red_onion")
class RoboLabFruitsVeggiesRedOnion(RoboLabObjectOptions):
    """The red onion has a smooth, opaque purple-red skin with a spherical shape. Its layers are crisp and white with a hint of purple, offering a sharp, pungent flavor commonly used in cooking."""

    file: str = "objects/fruits_veggies/red_onion/red_onion.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="vegetable")
