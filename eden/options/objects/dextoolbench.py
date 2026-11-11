"""DexToolBench hand-tool options — the twelve benchmark tools the SimToolReal policy manipulates.

Two variants per tool category: screwdriver, hammer, marker, brush, spatula, eraser.

Assets: `Kashu7100/eden_dextoolbench <https://huggingface.co/datasets/Kashu7100/eden_dextoolbench>`_,
mirrored from DexToolBench via `tylerlum/simtoolreal <https://github.com/tylerlum/simtoolreal>`_.

These point at the ``_decomposed`` URDF deliberately: the plain ``<tool>.urdf`` declares
inertia via the PyBullet ``<inertial><density>`` extension, which Genesis cannot parse.

``metadata.grasp_bbox`` is the tool's grasp bounding box in metres;
:func:`~eden.extensions.policies.simtoolreal.constants.object_scales_for` turns it into the
policy's ``object_scales`` observation (that x25 is a policy convention, not a mesh property).

Usage::

    tool = MalletHammer(default_root_pos=(0.0, 0.0, 0.58))
    cls = DEXTOOLBENCH_REGISTRY.get("mallet_hammer")     # or `eden view mallet_hammer`
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions, MetadataOptions
from eden.utils.registry import Registry

DEXTOOLBENCH_REGISTRY = Registry("dextoolbench")


class DexToolBenchOptions(EntityOptions):
    """A DexToolBench tool: a free rigid body loaded from its decomposed URDF."""

    registry: str = "Kashu7100"
    dataset: str = "eden_dextoolbench"
    assets_subdir: ClassVar[str] = "dextoolbench"

    is_fixed_base: bool = False


@DEXTOOLBENCH_REGISTRY.register(name="long_screwdriver")
class LongScrewdriver(DexToolBenchOptions):
    file: str = "long_screwdriver/long_screwdriver_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.10, 0.030, 0.030))


@DEXTOOLBENCH_REGISTRY.register(name="short_screwdriver")
class ShortScrewdriver(DexToolBenchOptions):
    file: str = "short_screwdriver/short_screwdriver_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.07, 0.035, 0.035))


@DEXTOOLBENCH_REGISTRY.register(name="mallet_hammer")
class MalletHammer(DexToolBenchOptions):
    file: str = "mallet_hammer/mallet_hammer_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.24, 0.030, 0.020))


@DEXTOOLBENCH_REGISTRY.register(name="claw_hammer")
class ClawHammer(DexToolBenchOptions):
    file: str = "claw_hammer/claw_hammer_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.10, 0.0225, 0.015))


@DEXTOOLBENCH_REGISTRY.register(name="sharpie_marker")
class SharpieMarker(DexToolBenchOptions):
    file: str = "sharpie_marker/sharpie_marker_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.085, 0.022, 0.022))


@DEXTOOLBENCH_REGISTRY.register(name="staples_marker")
class StaplesMarker(DexToolBenchOptions):
    file: str = "staples_marker/staples_marker_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.12, 0.018, 0.018))


@DEXTOOLBENCH_REGISTRY.register(name="red_brush")
class RedBrush(DexToolBenchOptions):
    # 64-part convex decomposition — by far the heaviest collision geometry of the twelve.
    file: str = "red_brush/red_brush_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.10, 0.020, 0.015))


@DEXTOOLBENCH_REGISTRY.register(name="blue_brush")
class BlueBrush(DexToolBenchOptions):
    file: str = "blue_brush/blue_brush_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.12, 0.035, 0.020))


@DEXTOOLBENCH_REGISTRY.register(name="spoon_spatula")
class SpoonSpatula(DexToolBenchOptions):
    file: str = "spoon_spatula/spoon_spatula_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.12, 0.020, 0.020))


@DEXTOOLBENCH_REGISTRY.register(name="flat_spatula")
class FlatSpatula(DexToolBenchOptions):
    file: str = "flat_spatula/flat_spatula_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.20, 0.015, 0.0075))


@DEXTOOLBENCH_REGISTRY.register(name="flat_eraser")
class FlatEraser(DexToolBenchOptions):
    file: str = "flat_eraser/flat_eraser_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.10, 0.028, 0.050))


@DEXTOOLBENCH_REGISTRY.register(name="handle_eraser")
class HandleEraser(DexToolBenchOptions):
    # 54-part convex decomposition — second only to the red_brush.
    file: str = "handle_eraser/handle_eraser_decomposed.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(grasp_bbox=(0.09, 0.032, 0.010))
