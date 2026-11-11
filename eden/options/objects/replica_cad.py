"""ReplicaCAD stage and object options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.catalog import MANIFEST
from eden.options.entities import EntityOptions, MetadataOptions
from eden.utils.registry import Registry

REPLICA_CAD_OBJECT_REGISTRY = Registry("replica_cad_object")
REPLICA_CAD_STAGE_REGISTRY = Registry("replica_cad_stage")


class ReplicaCADObjectOptions(EntityOptions):
    """ReplicaCAD object options."""

    registry: str = "haosulab"
    dataset: str = "ReplicaCAD"
    assets_subdir: ClassVar[str] = "replica_cad"


# ------------------------------------------------------------
# Stages
# ------------------------------------------------------------


@REPLICA_CAD_STAGE_REGISTRY.register(name="frl_apartment_stage")
class FRLApartmentStageOptions(ReplicaCADObjectOptions):
    file: str = "stages/frl_apartment_stage.glb"
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_fixed_base: bool = True


@REPLICA_CAD_STAGE_REGISTRY.register(name="Stage_v3_sc0_staging")
class StageV3SC0Options(ReplicaCADObjectOptions):
    file: str = "stages/Stage_v3_sc0_staging.glb"
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_fixed_base: bool = True


@REPLICA_CAD_STAGE_REGISTRY.register(name="Stage_v3_sc1_staging")
class StageV3SC1Options(ReplicaCADObjectOptions):
    file: str = "stages/Stage_v3_sc1_staging.glb"
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_fixed_base: bool = True


@REPLICA_CAD_STAGE_REGISTRY.register(name="Stage_v3_sc2_staging")
class StageV3SC2Options(ReplicaCADObjectOptions):
    file: str = "stages/Stage_v3_sc2_staging.glb"
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_fixed_base: bool = True


@REPLICA_CAD_STAGE_REGISTRY.register(name="Stage_v3_sc3_staging")
class StageV3SC3Options(ReplicaCADObjectOptions):
    file: str = "stages/Stage_v3_sc3_staging.glb"
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_fixed_base: bool = True


# ------------------------------------------------------------
# Objects
#
# Curation (render-verified): every placeable object carries a
# @MANIFEST.curate decorator. Deliberately uncurated: cloth_01..04 (hanging-garment
# meshes that need a rack parent, not floor/surface placeable), monitor_stand
# (wall-mount rail hardware for the monitor), tv_object (ambiguous small media box).
# ------------------------------------------------------------


@MANIFEST.curate(
    "basket", "container/basket", container=True, tags=("basket", "storage"), description="a storage basket"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_basket")
class FRLApartmentBasket(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_basket.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.5,
        semantic_id=3,
    )


@MANIFEST.curate("beanbag", "furniture/seating", tags=("seating", "beanbag"), description="a round beanbag chair")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_beanbag")
class FRLApartmentBeanbag(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_beanbag.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=11,
        semantic_id=6,
    )


@MANIFEST.curate("bike", "transport/bicycle", tags=("bike", "bicycle"), description="a bicycle (dark blue)")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bike_01")
class FRLApartmentBike01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bike_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=9,
        semantic_id=9,
    )


@MANIFEST.curate("red_bike", "transport/bicycle", tags=("bike", "bicycle"), description="a bicycle (red)")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bike_02")
class FRLApartmentBike02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bike_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=9,
        semantic_id=9,
    )


@MANIFEST.curate("small_bin", "container/bin", container=True, tags=("bin", "trash"), description="a small waste bin")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bin_01")
class FRLApartmentBin01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bin_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.36,
        semantic_id=10,
    )


@MANIFEST.curate(
    "pedal_bin", "container/bin", container=True, tags=("bin", "trash"), description="a tall dark pedal bin"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bin_02")
class FRLApartmentBin02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bin_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.6,
        semantic_id=10,
    )


@MANIFEST.curate(
    "lidded_bin", "container/bin", container=True, tags=("bin", "trash"), description="a waste bin with lid"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bin_03")
class FRLApartmentBin03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bin_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.81437,
        semantic_id=10,
    )


@MANIFEST.curate("book", "media/book", placement="surface", tags=("book",), description="a book")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_book_01")
class FRLApartmentBook01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_book_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.85,
        semantic_id=13,
    )


@MANIFEST.curate("notebook", "media/book", placement="surface", tags=("book", "notebook"), description="a notebook")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_book_02")
class FRLApartmentBook02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_book_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=2.0,
        semantic_id=13,
    )


@MANIFEST.curate("book_3", "media/book", placement="surface", tags=("book",), description="a thin paperback book")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_book_03")
class FRLApartmentBook03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_book_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.35,
        semantic_id=13,
    )


@MANIFEST.curate("book_4", "media/book", placement="surface", tags=("book",), description="a thin book")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_book_04")
class FRLApartmentBook04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_book_04.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.3,
        semantic_id=13,
    )


@MANIFEST.curate("book_5", "media/book", placement="surface", tags=("book",), description="a hardcover book")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_book_05")
class FRLApartmentBook05(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_book_05.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.8,
        semantic_id=13,
    )


@MANIFEST.curate(
    "binder", "media/book", placement="surface", tags=("book", "binder"), description="a thick blue binder"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_book_06")
class FRLApartmentBook06(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_book_06.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=4.8,
        semantic_id=13,
    )


@MANIFEST.curate(
    "bowl_1",
    "tableware/bowl",
    placement="surface",
    container=True,
    tags=("bowl",),
    description="a white ceramic bowl",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bowl_01")
class FRLApartmentBowl01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bowl_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=4.5,
        semantic_id=16,
    )


@MANIFEST.curate(
    "bowl_2",
    "tableware/bowl",
    placement="surface",
    container=True,
    tags=("bowl",),
    description="a bowl with a red rim",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bowl_02")
class FRLApartmentBowl02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bowl_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3.6,
        semantic_id=16,
    )


@MANIFEST.curate(
    "bowl_3",
    "tableware/bowl",
    placement="surface",
    container=True,
    tags=("bowl",),
    description="a blue-and-white bowl",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bowl_03")
class FRLApartmentBowl03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bowl_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.95,
        semantic_id=16,
    )


@MANIFEST.curate(
    "bowl_6",
    "tableware/bowl",
    placement="surface",
    container=True,
    tags=("bowl",),
    description="a small white bowl",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bowl_06")
class FRLApartmentBowl06(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bowl_06.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.3,
        semantic_id=16,
    )


@MANIFEST.curate(
    "serving_bowl",
    "tableware/bowl",
    placement="surface",
    container=True,
    tags=("bowl", "serving"),
    description="a large shallow serving bowl",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_bowl_07")
class FRLApartmentBowl07(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_bowl_07.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=5.2,
        semantic_id=16,
    )


@MANIFEST.curate("crate", "container/box", container=True, tags=("box", "crate"), description="a small wooden crate")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_box")
class FRLApartmentBox(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_box.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3.1,
        semantic_id=15,
    )


@MANIFEST.curate(
    "cabinet",
    "furniture/storage",
    support_surface=True,
    container=True,
    tags=("storage", "shelf", "cabinet"),
    description="a low storage cabinet",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cabinet")
class FRLApartmentCabinet(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cabinet.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=31.8,
        semantic_id=18,
    )


@MANIFEST.curate("camera", "electronics/device", placement="surface", tags=("camera",), description="a camera")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_camera_02")
class FRLApartmentCamera02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_camera_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.0,
        semantic_id=17,
    )


@MANIFEST.curate("chair", "furniture/seating", tags=("seating", "chair"), description="a chair")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_chair_01")
class FRLApartmentChair01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_chair_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=22,
        semantic_id=20,
    )


@MANIFEST.curate("armchair", "furniture/seating", tags=("seating", "armchair"), description="an armchair")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_chair_04")
class FRLApartmentChair04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_chair_04.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=4,
        semantic_id=20,
    )


@MANIFEST.curate("side_chair", "furniture/seating", tags=("seating", "chair"), description="a tan sled-base side chair")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_chair_05")
class FRLApartmentChair05(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_chair_05.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=15.4,
        semantic_id=20,
    )


@MANIFEST.curate(
    "chopping_board",
    "kitchen/cookware",
    placement="surface",
    tags=("board", "cookware", "kitchen"),
    description="a wooden chopping board",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_choppingboard_02")
class FRLApartmentChoppingboard02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_choppingboard_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.9,
        semantic_id=21,
    )


@MANIFEST.curate("clock", "decor/clock", placement="surface", tags=("clock",), description="a table clock")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_clock")
class FRLApartmentClock(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_clock.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.8,
        semantic_id=22,
    )


@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cloth_01")
class FRLApartmentCloth01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cloth_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=2,
        semantic_id=23,
    )


@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cloth_02")
class FRLApartmentCloth02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cloth_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=23,
    )


@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cloth_03")
class FRLApartmentCloth03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cloth_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.3,
        semantic_id=23,
    )


@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cloth_04")
class FRLApartmentCloth04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cloth_04.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=2,
        semantic_id=23,
    )


@MANIFEST.curate(
    "drying_rack",
    "household/laundry",
    tags=("rack", "laundry"),
    description="a folding clothes-drying rack",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_clothes_hanger_01")
class FRLApartmentClothesHanger01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_clothes_hanger_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.9,
        semantic_id=21,
    )


@MANIFEST.curate(
    "valet_stand",
    "household/laundry",
    tags=("rack", "laundry"),
    description="a small clothes rack / valet stand",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_clothes_hanger_02")
class FRLApartmentClothesHanger02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_clothes_hanger_02.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.9,
        semantic_id=21,
    )


@MANIFEST.curate(
    "coffee_cup", "tableware/cup", placement="surface", container=True, tags=("cup", "mug"), description="a coffee cup"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cup_01")
class FRLApartmentCup01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cup_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.5,
        semantic_id=28,
    )


@MANIFEST.curate(
    "floral_mug",
    "tableware/cup",
    placement="surface",
    container=True,
    tags=("cup", "mug"),
    description="a mug with a blue floral print",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cup_02")
class FRLApartmentCup02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cup_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=28,
    )


@MANIFEST.curate(
    "teacup",
    "tableware/cup",
    placement="surface",
    container=True,
    tags=("cup", "tea"),
    description="a wide teacup with handle",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cup_03")
class FRLApartmentCup03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cup_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=28,
    )


@MANIFEST.curate(
    "dark_mug",
    "tableware/cup",
    placement="surface",
    container=True,
    tags=("cup", "mug"),
    description="a dark mug",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cup_05")
class FRLApartmentCup05(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cup_05.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.5,
        semantic_id=28,
    )


@MANIFEST.curate(
    "knit_cushion",
    "decor/cushion",
    placement="surface",
    tags=("cushion", "pillow"),
    description="a knitted cushion (light blue)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cushion_01")
class FRLApartmentCushion01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cushion_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.3,
        semantic_id=29,
    )


@MANIFEST.curate(
    "cushion",
    "decor/cushion",
    placement="surface",
    tags=("cushion", "pillow"),
    description="a white cushion",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_cushion_03")
class FRLApartmentCushion03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_cushion_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.3,
        semantic_id=29,
    )


@MANIFEST.curate("handbag", "clothing/bag", container=True, tags=("bag",), description="a leather handbag")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_handbag")
class FRLApartmentHandbag(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_handbag.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3,
        semantic_id=41,
    )


@MANIFEST.curate("plant", "decor/plant", tags=("plant",), description="a potted floor plant")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_indoor_plant_01")
class FRLApartmentIndoorPlant01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_indoor_plant_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=25,
        semantic_id=44,
    )


@MANIFEST.curate("tall_plant", "decor/plant", tags=("plant",), description="a tall potted plant")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_indoor_plant_02")
class FRLApartmentIndoorPlant02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_indoor_plant_02.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=25,
        semantic_id=44,
    )


@MANIFEST.curate(
    "napkin_holder",
    "kitchen/utensil",
    placement="surface",
    tags=("kitchen", "utensil"),
    description="a white ceramic kitchen holder",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_01")
class FRLApartmentKitchenUtensil01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3,
        semantic_id=46,
    )


@MANIFEST.curate(
    "tumbler",
    "kitchen/utensil",
    placement="surface",
    container=True,
    tags=("kitchen", "cup", "tumbler"),
    description="a green tumbler",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_02")
class FRLApartmentKitchenUtensil02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=46,
    )


@MANIFEST.curate(
    "spice_jar",
    "kitchen/utensil",
    placement="surface",
    container=True,
    tags=("kitchen", "jar"),
    description="a small spice jar",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_03")
class FRLApartmentKitchenUtensil03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.1,
        semantic_id=46,
    )


@MANIFEST.curate(
    "kitchen_tray",
    "kitchen/utensil",
    placement="surface",
    tags=("kitchen", "tray"),
    description="a stacked kitchen tray",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_04")
class FRLApartmentKitchenUtensil04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_04.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.5,
        semantic_id=46,
    )


@MANIFEST.curate(
    "serving_tray",
    "kitchen/utensil",
    placement="surface",
    tags=("kitchen", "tray", "serving"),
    description="an oval serving tray",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_05")
class FRLApartmentKitchenUtensil05(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_05.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.9,
        semantic_id=46,
    )


@MANIFEST.curate(
    "small_canister",
    "kitchen/utensil",
    placement="surface",
    container=True,
    tags=("kitchen", "canister"),
    description="a small kitchen canister",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_06")
class FRLApartmentKitchenUtensil06(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_06.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.2,
        semantic_id=46,
    )


@MANIFEST.curate(
    "canister",
    "kitchen/utensil",
    placement="surface",
    container=True,
    tags=("kitchen", "canister"),
    description="a kitchen canister",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_08")
class FRLApartmentKitchenUtensil08(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_08.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.5,
        semantic_id=46,
    )


@MANIFEST.curate(
    "bottle",
    "kitchen/utensil",
    placement="surface",
    container=True,
    tags=("kitchen", "bottle"),
    description="a tall blue bottle",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_kitchen_utensil_09")
class FRLApartmentKitchenUtensil09(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_kitchen_utensil_09.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=46,
    )


@MANIFEST.curate(
    "knife_block",
    "kitchen/cookware",
    placement="surface",
    tags=("knife", "cookware", "kitchen"),
    description="a knife block with knives",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_knifeblock")
class FRLApartmentKnifeblock(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_knifeblock.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=4.5,
        semantic_id=45,
    )


@MANIFEST.curate(
    "table_lamp",
    "decor/lamp",
    placement="surface",
    tags=("lamp", "light"),
    description="a table lamp with shade",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_lamp_01")
class FRLApartmentLamp01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_lamp_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=6.5,
        semantic_id=47,
    )


@MANIFEST.curate(
    "small_lamp",
    "decor/lamp",
    placement="surface",
    tags=("lamp", "light"),
    description="a small table lamp",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_lamp_02")
class FRLApartmentLamp02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_lamp_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=4,
        semantic_id=47,
    )


@MANIFEST.curate("door_mat", "decor/rug", tags=("mat", "rug"), description="a small door mat (cat motif)")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_mat")
class FRLApartmentMat(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_mat.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1,
        semantic_id=50,
    )


@MANIFEST.curate(
    "monitor",
    "electronics/display",
    placement="wall",
    tags=("screen", "monitor"),
    description="a wall-mounted monitor",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_monitor")
class FRLApartmentMonitor(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_monitor.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=22,
        semantic_id=52,
    )


@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_monitor_stand")
class FRLApartmentMonitorStand(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_monitor_stand.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=None,
        semantic_id=52,
    )


@MANIFEST.curate(
    "pan",
    "kitchen/cookware",
    placement="surface",
    container=True,
    tags=("pan", "cookware", "kitchen"),
    description="a saucepan with handle",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_pan_01")
class FRLApartmentPan01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_pan_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1,
        semantic_id=55,
    )


@MANIFEST.curate(
    "picture_butterflies",
    "decor/picture",
    placement="wall",
    tags=("picture", "frame", "art"),
    description="a framed picture (butterflies)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_picture_01")
class FRLApartmentPicture01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_picture_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.8,
        semantic_id=59,
    )


@MANIFEST.curate(
    "picture_tree",
    "decor/picture",
    placement="wall",
    tags=("picture", "frame", "art"),
    description="a framed picture (autumn tree)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_picture_02")
class FRLApartmentPicture02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_picture_02.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.8,
        semantic_id=59,
    )


@MANIFEST.curate(
    "picture_cat",
    "decor/picture",
    placement="wall",
    tags=("picture", "frame", "photo"),
    description="a framed photo (cat)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_picture_03")
class FRLApartmentPicture03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_picture_03.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.4,
        semantic_id=59,
    )


@MANIFEST.curate(
    "picture_abstract",
    "decor/picture",
    placement="wall",
    tags=("picture", "frame", "art"),
    description="a framed picture (abstract)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_picture_04")
class FRLApartmentPicture04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_picture_04.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.4,
        semantic_id=59,
    )


@MANIFEST.curate("plate", "tableware/plate", placement="surface", tags=("plate",), description="a gray dinner plate")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_plate_01")
class FRLApartmentPlate01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_plate_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=64,
    )


@MANIFEST.curate(
    "serving_plate",
    "tableware/plate",
    placement="surface",
    tags=("plate", "serving"),
    description="a large plate on a wooden base",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_plate_02")
class FRLApartmentPlate02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_plate_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=2,
        semantic_id=64,
    )


@MANIFEST.curate(
    "rack",
    "furniture/storage",
    support_surface=True,
    tags=("rack", "shelf", "storage"),
    description="a slatted storage rack",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_rack_01")
class FRLApartmentRack01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_rack_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3.6,
        semantic_id=66,
    )


@MANIFEST.curate(
    "refrigerator",
    "appliance/refrigerator",
    tags=("kitchen", "fridge", "static"),
    description="a refrigerator (static mesh; see 'fridge' for an openable one)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_refrigerator")
class FRLApartmentRefrigerator(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_refrigerator.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=90,
        semantic_id=67,
    )


@MANIFEST.curate(
    "remote",
    "electronics/device",
    placement="surface",
    tags=("remote", "tv", "device"),
    description="a TV remote control",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_remote-control_01")
class FRLApartmentRemoteControl01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_remote-control_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.1,
        semantic_id=68,
    )


@MANIFEST.curate("rug", "decor/rug", tags=("rug",), description="a floor rug")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_rug_01")
class FRLApartmentRug01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_rug_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=5.4,
        semantic_id=98,
    )


@MANIFEST.curate("large_rug", "decor/rug", tags=("rug",), description="a large gray area rug")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_rug_02")
class FRLApartmentRug02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_rug_02.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=12,
        semantic_id=98,
    )


@MANIFEST.curate(
    "settop_box",
    "electronics/device",
    placement="surface",
    tags=("settop_box", "media", "device"),
    description="a slim set-top media box",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_setupbox")
class FRLApartmentSetupbox(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_setupbox.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.5,
        semantic_id=82,
    )


@MANIFEST.curate("black_shoes", "clothing/shoe", tags=("shoes",), description="a pair of black dress shoes")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_shoe_01")
class FRLApartmentShoe01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_shoe_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1,
        semantic_id=72,
    )


@MANIFEST.curate("brown_shoes", "clothing/shoe", tags=("shoes",), description="a pair of brown dress shoes")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_shoe_02")
class FRLApartmentShoe02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_shoe_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1,
        semantic_id=72,
    )


@MANIFEST.curate("red_flats", "clothing/shoe", tags=("shoes",), description="a pair of red flats")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_shoe_03")
class FRLApartmentShoe03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_shoe_03.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.7,
        semantic_id=72,
    )


@MANIFEST.curate("sneakers", "clothing/shoe", tags=("shoes",), description="a pair of blue sneakers")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_shoe_04")
class FRLApartmentShoe04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_shoe_04.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.9,
        semantic_id=72,
    )


@MANIFEST.curate(
    "shoebox", "container/box", container=True, tags=("box", "storage"), description="a lidded storage box"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_shoebox_01")
class FRLApartmentShoebox01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_shoebox_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.3,
        semantic_id=15,
    )


@MANIFEST.curate(
    "stand_mixer",
    "appliance/small",
    placement="surface",
    tags=("kitchen", "appliance", "mixer"),
    description="a stand mixer",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_small_appliance_01")
class FRLApartmentSmallAppliance01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_small_appliance_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=11,
        semantic_id=75,
    )


@MANIFEST.curate(
    "kettle",
    "appliance/small",
    placement="surface",
    tags=("kitchen", "appliance", "kettle"),
    description="an electric kettle",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_small_appliance_02")
class FRLApartmentSmallAppliance02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_small_appliance_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3.8,
        semantic_id=75,
    )


@MANIFEST.curate("sofa", "furniture/seating", tags=("seating", "couch"), description="a sofa / couch")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_sofa")
class FRLApartmentSofa(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_sofa.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=68,
        semantic_id=76,
    )


@MANIFEST.curate(
    "sponge_dish",
    "kitchen/utensil",
    placement="surface",
    tags=("sponge", "kitchen"),
    description="a dish holding a sponge",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_sponge_dish")
class FRLApartmentSpongeDish(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_sponge_dish.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=46,
    )


@MANIFEST.curate(
    "stool",
    "furniture/seating",
    support_surface=True,
    tags=("seating", "stool"),
    description="a low wooden stool",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_stool_02")
class FRLApartmentStool02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_stool_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=5.8,
        semantic_id=78,
    )


@MANIFEST.curate("side_table", "furniture/table", support_surface=True, tags=("table",), description="a side table")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_table_01")
class FRLApartmentTable01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_table_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=18,
        semantic_id=80,
    )


@MANIFEST.curate("desk", "furniture/table", support_surface=True, tags=("table", "desk"), description="a work desk")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_table_02")
class FRLApartmentTable02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_table_02.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=24,
        semantic_id=80,
    )


@MANIFEST.curate(
    "coffee_table",
    "furniture/table",
    support_surface=True,
    tags=("table", "coffee_table"),
    description="a low rectangular coffee table",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_table_03")
class FRLApartmentTable03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_table_03.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=22,
        semantic_id=80,
    )


@MANIFEST.curate(
    "end_table",
    "furniture/table",
    support_surface=True,
    tags=("table", "round"),
    description="a small round end table",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_table_04")
class FRLApartmentTable04(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_table_04.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=3.1,
        semantic_id=80,
    )


@MANIFEST.curate(
    "towel", "household/textile", placement="surface", tags=("towel", "cloth"), description="a crumpled towel"
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_towel")
class FRLApartmentTowel(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_towel.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=86,
    )


@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_tv_object")
class FRLApartmentTvObject(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_tv_object.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=1.5,
        semantic_id=82,
    )


@MANIFEST.curate(
    "tv",
    "electronics/display",
    placement="wall",
    tags=("tv", "screen", "display"),
    description="a large flat-screen TV",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_tv_screen")
class FRLApartmentTvScreen(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_tv_screen.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=22,
        semantic_id=87,
    )


@MANIFEST.curate(
    "tv_stand",
    "furniture/storage",
    support_surface=True,
    container=True,
    tags=("media", "storage"),
    description="a TV / media stand",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_tvstand")
class FRLApartmentTvstand(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_tvstand.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=29,
        semantic_id=88,
    )


@MANIFEST.curate("umbrella", "household/umbrella", tags=("umbrella",), description="a closed standing umbrella")
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_umbrella")
class FRLApartmentUmbrella(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_umbrella.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.3,
        semantic_id=89,
    )


@MANIFEST.curate(
    "small_vase",
    "decor/vase",
    placement="surface",
    container=True,
    tags=("vase",),
    description="a small ceramic vase",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_vase_01")
class FRLApartmentVase01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_vase_01.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.6,
        semantic_id=91,
    )


@MANIFEST.curate(
    "vase",
    "decor/vase",
    placement="surface",
    container=True,
    tags=("vase",),
    description="a white ceramic vase",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_vase_02")
class FRLApartmentVase02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_vase_02.glb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=0.8,
        semantic_id=91,
    )


@MANIFEST.curate(
    "tall_cabinet",
    "furniture/storage",
    container=True,
    tags=("cabinet", "storage", "tall"),
    description="a tall narrow storage cabinet",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_wall_cabinet_01")
class FRLApartmentWallCabinet01(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_wall_cabinet_01.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=11,
        semantic_id=94,
    )


@MANIFEST.curate(
    "shelving_cabinet",
    "furniture/storage",
    container=True,
    tags=("cabinet", "shelf", "storage", "tall"),
    description="a tall cabinet with open shelving",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_wall_cabinet_02")
class FRLApartmentWallCabinet02(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_wall_cabinet_02.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=19,
        semantic_id=94,
    )


@MANIFEST.curate(
    "two_door_cabinet",
    "furniture/storage",
    container=True,
    tags=("cabinet", "storage"),
    description="a two-door storage cabinet",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="frl_apartment_wall_cabinet_03")
class FRLApartmentWallCabinet03(ReplicaCADObjectOptions):
    file: str = "objects/frl_apartment_wall_cabinet_03.glb"
    is_fixed_base: bool = True
    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        mass=40,
        semantic_id=94,
    )


# ------------------------------------------------------------
# Articulated Objects
# ------------------------------------------------------------


@MANIFEST.curate(
    "sliding_cabinet",
    "furniture/storage",
    support_surface=True,
    container=True,
    tags=("openable", "articulated", "cabinet", "sliding"),
    description="a low sideboard cabinet with two sliding doors",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="cabinet")
class Cabinet(ReplicaCADObjectOptions):
    file: str = "urdf/cabinet/cabinet.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = ["left_slide", "right_slide"]


@MANIFEST.curate(
    "chest_of_drawers",
    "furniture/storage",
    support_surface=True,
    container=True,
    tags=("openable", "articulated", "drawers", "storage"),
    description="a chest of drawers (six openable drawers)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="chestOfDrawers_01")
class ChestOfDrawers01(ReplicaCADObjectOptions):
    file: str = "urdf/chestOfDrawers_01/chestOfDrawers_01.urdf"
    up: tuple[int, int, int] = (0, 0, 1)  # z-up authored, unlike the other ReplicaCAD urdfs (render-verified)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = [
        "drawer_topL",
        "drawer_topR",
        "drawer_midL",
        "drawer_midR",
        "drawer_botL",
        "drawer_botR",
    ]


@MANIFEST.curate(
    "door_1",
    "architecture/door",
    placement="wall",
    tags=("openable", "articulated", "door"),
    description="a hinged interior door (white panel)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="door1")
class Door1(ReplicaCADObjectOptions):
    file: str = "urdf/door1/door1.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = ["root_rotation"]


@MANIFEST.curate(
    "door_2",
    "architecture/door",
    placement="wall",
    tags=("openable", "articulated", "door"),
    description="a tall hinged interior door (wood)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="door2")
class Door2(ReplicaCADObjectOptions):
    file: str = "urdf/door2/door2.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = ["root_rotation"]


@MANIFEST.curate(
    "door_3",
    "architecture/door",
    placement="wall",
    tags=("openable", "articulated", "door"),
    description="a hinged interior door (wood)",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="door3")
class Door3(ReplicaCADObjectOptions):
    file: str = "urdf/door3/door3.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = ["root_rotation"]


@MANIFEST.curate(
    "double_door",
    "architecture/door",
    placement="wall",
    tags=("openable", "articulated", "door", "double"),
    description="a double hinged door",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="doubleDoor")
class DoubleDoor(ReplicaCADObjectOptions):
    file: str = "urdf/doubledoor/doubledoor.urdf"  # lowercase in the upstream HF dataset
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = ["left_hinge", "right_hinge"]


@MANIFEST.curate(
    "fridge",
    "appliance/refrigerator",
    container=True,
    tags=("openable", "articulated", "kitchen", "fridge"),
    description="a refrigerator with openable top and bottom doors",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="fridge")
class Fridge(ReplicaCADObjectOptions):
    file: str = "urdf/fridge/fridge.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = ["top_door_hinge", "bottom_door_hinge"]


@MANIFEST.curate(
    "kitchen_counter",
    "furniture/counter",
    support_surface=True,
    container=True,
    tags=("openable", "articulated", "kitchen", "counter", "sink"),
    description="a kitchen counter with a sink and sliding drawers",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="kitchen_counter")
class KitchenCounter(ReplicaCADObjectOptions):
    file: str = "urdf/kitchen_counter/kitchen_counter.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = [
        "left_slide_top",
        "left_slide_bottom",
        "middle_slide_top",
        "middle_slide_middle",
        "middle_slide_bottom",
        "right_slide_top",
        "right_slide_bottom",
    ]


@MANIFEST.curate(
    "kitchen_cupboard",
    "furniture/storage",
    placement="wall",
    container=True,
    tags=("openable", "articulated", "kitchen", "cupboard"),
    description="a wall-mounted kitchen cupboard with hinged doors",
)
@REPLICA_CAD_OBJECT_REGISTRY.register(name="kitchenCupboard_01")
class KitchenCupboard01(ReplicaCADObjectOptions):
    file: str = "urdf/kitchenCupboard_01/kitchenCupboard_01.urdf"
    up: tuple[int, int, int] = (0, 1, 0)
    is_fixed_base: bool = True
    is_articulated: bool = True
    dofs_name: list[str] = [
        "doorWhole_1L_hinge",
        "doorWhole_1R_hinge",
        "doorWhole_2L_hinge",
        "doorWhole_2R_hinge",
        "doorWindow_1L_hinge",
        "doorWindow_1R_hinge",
        "doorWindow_2L_hinge",
        "doorWindow_2R_hinge",
    ]
