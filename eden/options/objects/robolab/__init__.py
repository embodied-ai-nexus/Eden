"""RoboLab asset options — 309 tabletop manipulation objects + 12 fixtures.

Converted from the RoboLab USD library (YCB, HOPE, HOT3D, HANDAL, VOMP, Objaverse,
fruits/veggies, basic blocks, kitchen/table fixtures) to URDF and hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``. See ``base.py`` for
conversion details.

Usage::

    from eden.options.objects.robolab import ROBOLAB_REGISTRY
    from eden.options.objects.robolab.ycb import RoboLabYcbBanana

    banana = ROBOLAB_REGISTRY.get("ycb/banana")(default_root_pos=(0.3, 0.0, 0.8))
    kitchen = ROBOLAB_REGISTRY.get("fixtures/kitchen_island")()
"""

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabFixtureOptions, RoboLabObjectOptions
from eden.options.objects.robolab import (  # noqa: F401  (imported for their registry side effects)
    basic,
    fixtures,
    fruits_veggies,
    handal,
    hope,
    hot3d,
    objaverse,
    vomp,
    ycb,
)

__all__ = [
    "ROBOLAB_REGISTRY",
    "RoboLabObjectOptions",
    "RoboLabFixtureOptions",
]
