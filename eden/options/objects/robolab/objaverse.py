"""RoboLab objaverse object options (auto-generated from object_catalog.json).

Assets converted from the RoboLab USD library to URDF; hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.
"""

from __future__ import annotations

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabObjectOptions


@ROBOLAB_REGISTRY.register(name="objaverse/apple")
class RoboLabObjaverseApple(RoboLabObjectOptions):
    file: str = "objects/objaverse/apple/apple.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/apple_01")
class RoboLabObjaverseApple01(RoboLabObjectOptions):
    file: str = "objects/objaverse/apple_01/apple_01.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/bagel")
class RoboLabObjaverseBagel(RoboLabObjectOptions):
    file: str = "objects/objaverse/bagel/bagel.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/bagel_00")
class RoboLabObjaverseBagel00(RoboLabObjectOptions):
    file: str = "objects/objaverse/bagel_00/bagel_00.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/baguette_02")
class RoboLabObjaverseBaguette02(RoboLabObjectOptions):
    file: str = "objects/objaverse/baguette_02/baguette_02.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/gregorys_coffee_cup")
class RoboLabObjaverseGregorysCoffeeCup(RoboLabObjectOptions):
    file: str = "objects/objaverse/gregorys_coffee_cup/gregorys_coffee_cup.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/lunchbag")
class RoboLabObjaverseLunchbag(RoboLabObjectOptions):
    file: str = "objects/objaverse/lunchbag/lunchbag.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/red_bell_pepper")
class RoboLabObjaverseRedBellPepper(RoboLabObjectOptions):
    file: str = "objects/objaverse/red_bell_pepper/red_bell_pepper.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/sesame_bagel")
class RoboLabObjaverseSesameBagel(RoboLabObjectOptions):
    file: str = "objects/objaverse/sesame_bagel/sesame_bagel.urdf"


@ROBOLAB_REGISTRY.register(name="objaverse/snickers_bar")
class RoboLabObjaverseSnickersBar(RoboLabObjectOptions):
    file: str = "objects/objaverse/snickers_bar/snickers_bar.urdf"
