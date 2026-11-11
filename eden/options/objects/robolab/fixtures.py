"""RoboLab fixture object options (auto-generated from object_catalog.json).

Assets converted from the RoboLab USD library to URDF; hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabFixtureOptions


@ROBOLAB_REGISTRY.register(name="fixtures/franka_table")
class RoboLabFixtureFrankaTable(RoboLabFixtureOptions):
    """A work table sized for a tabletop Franka arm."""

    file: str = "fixtures/franka_table/franka_table.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="table")


@ROBOLAB_REGISTRY.register(name="fixtures/grey_bin")
class RoboLabFixtureGreyBin(RoboLabFixtureOptions):
    """A shallow grey plastic tote bin."""

    file: str = "fixtures/grey_bin/grey_bin.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="container")


@ROBOLAB_REGISTRY.register(name="fixtures/kitchen_island")
class RoboLabFixtureKitchenIsland(RoboLabFixtureOptions):
    """A kitchen counter set with a free-standing island, range, and vent hood."""

    file: str = "fixtures/kitchen_island/kitchen_island.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchen")


@ROBOLAB_REGISTRY.register(name="fixtures/kitchen_l_shaped")
class RoboLabFixtureKitchenLShaped(RoboLabFixtureOptions):
    """An L-shaped kitchen counter with range and vent hood."""

    file: str = "fixtures/kitchen_l_shaped/kitchen_l_shaped.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchen")


@ROBOLAB_REGISTRY.register(name="fixtures/kitchen_peninsula")
class RoboLabFixtureKitchenPeninsula(RoboLabFixtureOptions):
    """A kitchen counter with a peninsula extension, range, and vent hood."""

    file: str = "fixtures/kitchen_peninsula/kitchen_peninsula.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchen")


@ROBOLAB_REGISTRY.register(name="fixtures/kitchen_single_wall")
class RoboLabFixtureKitchenSingleWall(RoboLabFixtureOptions):
    """A single-wall kitchen counter with range and vent hood."""

    file: str = "fixtures/kitchen_single_wall/kitchen_single_wall.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchen")


@ROBOLAB_REGISTRY.register(name="fixtures/kitchen_u_shaped")
class RoboLabFixtureKitchenUShaped(RoboLabFixtureOptions):
    """A U-shaped kitchen counter with range and vent hood."""

    file: str = "fixtures/kitchen_u_shaped/kitchen_u_shaped.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchen")


@ROBOLAB_REGISTRY.register(name="fixtures/stand")
class RoboLabFixtureStand(RoboLabFixtureOptions):
    """A pedestal stand for mounting a robot arm."""

    file: str = "fixtures/stand/stand.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="stand")


@ROBOLAB_REGISTRY.register(name="fixtures/table_bamboo")
class RoboLabFixtureTableBamboo(RoboLabFixtureOptions):
    """A bamboo dining table."""

    file: str = "fixtures/table_bamboo/table_bamboo.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="table")


@ROBOLAB_REGISTRY.register(name="fixtures/table_black")
class RoboLabFixtureTableBlack(RoboLabFixtureOptions):
    """A black dining table."""

    file: str = "fixtures/table_black/table_black.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="table")


@ROBOLAB_REGISTRY.register(name="fixtures/table_maple")
class RoboLabFixtureTableMaple(RoboLabFixtureOptions):
    """A maple dining table."""

    file: str = "fixtures/table_maple/table_maple.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="table")


@ROBOLAB_REGISTRY.register(name="fixtures/table_oak")
class RoboLabFixtureTableOak(RoboLabFixtureOptions):
    """An oak dining table."""

    file: str = "fixtures/table_oak/table_oak.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="table")
