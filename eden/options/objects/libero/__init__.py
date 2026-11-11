"""LIBERO benchmark object and workspace options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions, MetadataOptions
from eden.options.objects.libero.mjcf_transforms import (
    _prepare_object_xml,
    _prepare_scene_xml,
    _prepare_workspace_table_xml,
)
from eden.utils.registry import Registry

_X90_QUAT = (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)
_Z90_QUAT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)


class LiberoObjectOptions(EntityOptions):
    """Base options for an object from the published LIBERO asset bundle."""

    registry: str = "Kashu7100"
    dataset: str = "eden_libero"
    assets_subdir: ClassVar[str] = "libero"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(benchmark="libero")
    # The bundle's object files carry no free joint; a free object gets one added at load.
    mjcf_follows_is_fixed_base: ClassVar[bool] = True

    def resolve_runtime_file(self) -> str:
        """Derive one Genesis-ready MJCF without changing the serialized source file."""
        return _prepare_object_xml(
            self.file,
            is_fixed_base=self.is_fixed_base,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )


class LiberoSceneAssetOptions(LiberoObjectOptions):
    """Base options for fixed workspace geometry from the LIBERO bundle."""

    is_fixed_base: bool = True
    # The workspace MJCF is derived without reading the flag, so the file alone decides.
    mjcf_follows_is_fixed_base: ClassVar[bool] = False

    def resolve_runtime_file(self) -> str:
        """Derive one Genesis-ready workspace MJCF."""
        return _prepare_scene_xml(
            self.file,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )


class _LiberoProceduralTableWorkspaceOptions(LiberoSceneAssetOptions):
    """Canonical table geometry materialized by LIBERO's table arenas."""

    table_body_name: ClassVar[str] = "table"
    table_top_pos_local: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.90)
    table_full_size: ClassVar[tuple[float, float, float]] = (1.0, 1.2, 0.05)
    table_friction: ClassVar[tuple[float, float, float]]

    def resolve_runtime_file(self) -> str:
        """Derive one Genesis-ready procedural table MJCF."""
        return _prepare_workspace_table_xml(
            self.file,
            table_body_name=self.table_body_name,
            table_top_pos_local=self.table_top_pos_local,
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )


LIBERO_OBJECT_REGISTRY = Registry("LIBERO_OBJECT")


@LIBERO_OBJECT_REGISTRY.register(name="akita_black_bowl")
class LiberoAkitaBlackBowl(LiberoObjectOptions):
    file: str = "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"


@LIBERO_OBJECT_REGISTRY.register(name="alphabet_soup")
class LiberoAlphabetSoup(LiberoObjectOptions):
    file: str = "stable_hope_objects/alphabet_soup/alphabet_soup.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="basket")
class LiberoBasket(LiberoObjectOptions):
    file: str = "stable_scanned_objects/basket/basket.xml"
    default_root_quat: tuple[float, float, float, float] = _Z90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="bbq_sauce")
class LiberoBbqSauce(LiberoObjectOptions):
    file: str = "stable_hope_objects/bbq_sauce/bbq_sauce.xml"
    default_root_quat: tuple[float, float, float, float] = _Z90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="black_book")
class LiberoBlackBook(LiberoObjectOptions):
    file: str = "turbosquid_objects/black_book/black_book.xml"
    default_root_quat: tuple[float, float, float, float] = (0.19509032, 0.98078528, 0.0, 0.0)


@LIBERO_OBJECT_REGISTRY.register(name="butter")
class LiberoButter(LiberoObjectOptions):
    file: str = "stable_hope_objects/butter/butter.xml"


@LIBERO_OBJECT_REGISTRY.register(name="chefmate_8_frypan")
class LiberoChefmate8Frypan(LiberoObjectOptions):
    file: str = "stable_scanned_objects/chefmate_8_frypan/chefmate_8_frypan.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="chocolate_pudding")
class LiberoChocolatePudding(LiberoObjectOptions):
    file: str = "stable_hope_objects/chocolate_pudding/chocolate_pudding.xml"


@LIBERO_OBJECT_REGISTRY.register(name="cookies")
class LiberoCookies(LiberoObjectOptions):
    file: str = "stable_hope_objects/cookies/cookies.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="cream_cheese")
class LiberoCreamCheese(LiberoObjectOptions):
    file: str = "stable_hope_objects/cream_cheese/cream_cheese.xml"


@LIBERO_OBJECT_REGISTRY.register(name="desk_caddy")
class LiberoDeskCaddy(LiberoObjectOptions):
    file: str = "turbosquid_objects/desk_caddy/desk_caddy.xml"
    is_fixed_base: bool = True


@LIBERO_OBJECT_REGISTRY.register(name="flat_stove")
class LiberoFlatStove(LiberoObjectOptions):
    file: str = "articulated_objects/flat_stove.xml"
    is_fixed_base: bool = True
    category_name: ClassVar[str] = "flat_stove"
    button_joint_name: ClassVar[str] = "button"
    burner_link_name: ClassVar[str] = "burner"
    burner_local_pos: ClassVar[tuple[float, float, float]] = (0.0, 0.0, 0.022)
    on_threshold: ClassVar[float] = 0.5


@LIBERO_OBJECT_REGISTRY.register(name="glazed_rim_porcelain_ramekin")
class LiberoGlazedRimPorcelainRamekin(LiberoObjectOptions):
    file: str = "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="ketchup")
class LiberoKetchup(LiberoObjectOptions):
    file: str = "stable_hope_objects/ketchup/ketchup.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="microwave")
class LiberoMicrowave(LiberoObjectOptions):
    file: str = "articulated_objects/microwave.xml"
    is_fixed_base: bool = True


@LIBERO_OBJECT_REGISTRY.register(name="milk")
class LiberoMilk(LiberoObjectOptions):
    file: str = "stable_hope_objects/milk/milk.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="moka_pot")
class LiberoMokaPot(LiberoObjectOptions):
    file: str = "turbosquid_objects/moka_pot/moka_pot.xml"


@LIBERO_OBJECT_REGISTRY.register(name="new_salad_dressing")
class LiberoNewSaladDressing(LiberoObjectOptions):
    file: str = "stable_hope_objects/new_salad_dressing/new_salad_dressing.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="orange_juice")
class LiberoOrangeJuice(LiberoObjectOptions):
    file: str = "stable_hope_objects/orange_juice/orange_juice.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="plate")
class LiberoPlate(LiberoObjectOptions):
    file: str = "stable_scanned_objects/plate/plate.xml"


@LIBERO_OBJECT_REGISTRY.register(name="porcelain_mug")
class LiberoPorcelainMug(LiberoObjectOptions):
    file: str = "turbosquid_objects/porcelain_mug/porcelain_mug.xml"


@LIBERO_OBJECT_REGISTRY.register(name="red_coffee_mug")
class LiberoRedCoffeeMug(LiberoObjectOptions):
    file: str = "turbosquid_objects/red_coffee_mug/red_coffee_mug.xml"


@LIBERO_OBJECT_REGISTRY.register(name="salad_dressing")
class LiberoSaladDressing(LiberoObjectOptions):
    file: str = "stable_hope_objects/salad_dressing/salad_dressing.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="tomato_sauce")
class LiberoTomatoSauce(LiberoObjectOptions):
    file: str = "stable_hope_objects/tomato_sauce/tomato_sauce.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="white_bowl")
class LiberoWhiteBowl(LiberoObjectOptions):
    file: str = "stable_scanned_objects/white_bowl/white_bowl.xml"
    default_root_quat: tuple[float, float, float, float] = _X90_QUAT


@LIBERO_OBJECT_REGISTRY.register(name="white_cabinet")
class LiberoWhiteCabinet(LiberoObjectOptions):
    file: str = "articulated_objects/white_cabinet.xml"
    is_fixed_base: bool = True


@LIBERO_OBJECT_REGISTRY.register(name="white_yellow_mug")
class LiberoWhiteYellowMug(LiberoObjectOptions):
    file: str = "turbosquid_objects/white_yellow_mug/white_yellow_mug.xml"


@LIBERO_OBJECT_REGISTRY.register(name="wine_bottle")
class LiberoWineBottle(LiberoObjectOptions):
    file: str = "turbosquid_objects/wine_bottle/wine_bottle.xml"


@LIBERO_OBJECT_REGISTRY.register(name="wine_rack")
class LiberoWineRack(LiberoObjectOptions):
    file: str = "turbosquid_objects/wine_rack/wine_rack.xml"
    is_fixed_base: bool = True


@LIBERO_OBJECT_REGISTRY.register(name="wooden_cabinet")
class LiberoWoodenCabinet(LiberoObjectOptions):
    file: str = "articulated_objects/wooden_cabinet.xml"
    is_fixed_base: bool = True


@LIBERO_OBJECT_REGISTRY.register(name="wooden_tray")
class LiberoWoodenTray(LiberoObjectOptions):
    file: str = "turbosquid_objects/wooden_tray/wooden_tray.xml"


@LIBERO_OBJECT_REGISTRY.register(name="wooden_two_layer_shelf")
class LiberoWoodenTwoLayerShelf(LiberoObjectOptions):
    file: str = "turbosquid_objects/wooden_two_layer_shelf/wooden_two_layer_shelf.xml"
    is_fixed_base: bool = True


@LIBERO_OBJECT_REGISTRY.register(name="yellow_book")
class LiberoYellowBook(LiberoObjectOptions):
    file: str = "turbosquid_objects/yellow_book/yellow_book.xml"
    default_root_quat: tuple[float, float, float, float] = (0.8314696123025452, -0.5555702330196022, 0.0, 0.0)


class LiberoTabletopWorkspace(_LiberoProceduralTableWorkspaceOptions):
    workspace_entity_name: ClassVar[str] = "main_table"
    file: str = "scenes/libero_tabletop_base_style.xml"
    table_friction: ClassVar[tuple[float, float, float]] = (0.6, 0.005, 0.0001)


class LiberoFloorWorkspace(LiberoSceneAssetOptions):
    workspace_entity_name: ClassVar[str] = "floor"
    file: str = "scenes/libero_floor_base_style.xml"


class LiberoKitchenTabletopWorkspace(_LiberoProceduralTableWorkspaceOptions):
    workspace_entity_name: ClassVar[str] = "kitchen_table"
    file: str = "scenes/libero_kitchen_tabletop_base_style.xml"
    table_friction: ClassVar[tuple[float, float, float]] = (1.0, 0.005, 0.0001)


class LiberoLivingRoomTabletopWorkspace(LiberoSceneAssetOptions):
    workspace_entity_name: ClassVar[str] = "living_room_table"
    file: str = "scenes/libero_living_room_tabletop_base_style.xml"


class LiberoStudyTabletopWorkspace(LiberoSceneAssetOptions):
    workspace_entity_name: ClassVar[str] = "study_table"
    file: str = "scenes/libero_study_base_style.xml"


class LiberoCoffeeTableWorkspace(LiberoSceneAssetOptions):
    workspace_entity_name: ClassVar[str] = "coffee_table"
    file: str = "scenes/libero_coffee_table_base_style.xml"


__all__ = [
    "LIBERO_OBJECT_REGISTRY",
    "LiberoAkitaBlackBowl",
    "LiberoAlphabetSoup",
    "LiberoBasket",
    "LiberoBbqSauce",
    "LiberoBlackBook",
    "LiberoButter",
    "LiberoChefmate8Frypan",
    "LiberoChocolatePudding",
    "LiberoCoffeeTableWorkspace",
    "LiberoCookies",
    "LiberoCreamCheese",
    "LiberoDeskCaddy",
    "LiberoFlatStove",
    "LiberoFloorWorkspace",
    "LiberoGlazedRimPorcelainRamekin",
    "LiberoKetchup",
    "LiberoKitchenTabletopWorkspace",
    "LiberoLivingRoomTabletopWorkspace",
    "LiberoMicrowave",
    "LiberoMilk",
    "LiberoMokaPot",
    "LiberoNewSaladDressing",
    "LiberoObjectOptions",
    "LiberoOrangeJuice",
    "LiberoPlate",
    "LiberoPorcelainMug",
    "LiberoRedCoffeeMug",
    "LiberoSaladDressing",
    "LiberoSceneAssetOptions",
    "LiberoStudyTabletopWorkspace",
    "LiberoTabletopWorkspace",
    "LiberoTomatoSauce",
    "LiberoWhiteBowl",
    "LiberoWhiteCabinet",
    "LiberoWhiteYellowMug",
    "LiberoWineBottle",
    "LiberoWineRack",
    "LiberoWoodenCabinet",
    "LiberoWoodenTray",
    "LiberoWoodenTwoLayerShelf",
    "LiberoYellowBook",
]
