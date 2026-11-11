"""Base options for the RoboLab asset library.

RoboLab ships a curated USD library of 309 tabletop manipulation objects (YCB,
HOPE, HOT3D, HANDAL, VOMP, Objaverse, fruits/veggies, basic blocks) plus 12
kitchen/table fixtures. The USDs were converted to single-link
URDFs (world-space Z-up meshes, catalog masses, box inertia) and are hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.

Look up assets through the registry::

    from eden.options.objects.robolab import ROBOLAB_REGISTRY

    opts_cls = ROBOLAB_REGISTRY.get("ycb/banana")
    cfg = opts_cls(default_root_pos=(0.3, 0.0, 0.8))

Semantic metadata (``category`` from the RoboLab object catalog) lives on each
class's ``metadata``; the catalog description is the class docstring.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions
from eden.utils.registry import Registry

ROBOLAB_REGISTRY = Registry("ROBOLAB")


class RoboLabObjectOptions(EntityOptions):
    """Base options for RoboLab rigid objects.

    Meshes are baked in stage world space (Z-up, meters), so the URDF base_link
    frame matches the RoboLab catalog pose with identity ``up``/``front``.
    """

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    registry: str = "Kashu7100"
    dataset: str = "eden_robolab"
    assets_subdir: ClassVar[str] = "robolab"


class RoboLabFixtureOptions(RoboLabObjectOptions):
    """Base options for RoboLab fixtures (tables, kitchen counters, stands, bins).

    Fixtures load fixed-base by default; pass ``is_fixed_base=False`` for movable
    props like the grey bin. Kitchen counters approximate their MDL materials
    with flat colors (textures are not extractable from .mdl files).
    """

    is_fixed_base: bool = True
