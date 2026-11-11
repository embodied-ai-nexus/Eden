"""YCB object-set options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions, MetadataOptions


class YCBOptions(EntityOptions):
    """YCB options for Eden, wrapping entity options."""

    # The YCB Object/Model Set's `textured.obj` files have the cylinder/long
    # axis along mesh-Z and the label-facing direction along mesh-Y. Aligning
    # the option defaults with that native frame makes pose annotations from
    # YCB-based datasets (e.g. DexYCB) apply directly without per-call frame
    # corrections.
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (0, 1, 0)

    registry: str = "Kashu7100"
    dataset: str = "eden_ycb"
    assets_subdir: ClassVar[str] = "ycb"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=1000)


class YCB002MasterChefCan(YCBOptions):
    file: str = "models/002_master_chef_can/textured.obj"
    scale: float = 0.7


class YCB003CrackerBox(YCBOptions):
    file: str = "models/003_cracker_box/textured.obj"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=50)


class YCB004SugarBox(YCBOptions):
    file: str = "models/004_sugar_box/textured.obj"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=50)


class YCB005TomatoSoupCan(YCBOptions):
    file: str = "models/005_tomato_soup_can/textured.obj"
    scale: float = 1.0


class YCB006MustardBottle(YCBOptions):
    file: str = "models/006_mustard_bottle/textured.obj"
    scale: float = 1.0


class YCB007TunaFishCan(YCBOptions):
    file: str = "models/007_tuna_fish_can/textured.obj"
    scale: float = 0.8


class YCB008PuddingBox(YCBOptions):
    file: str = "models/008_pudding_box/textured.obj"
    scale: float = 0.8
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=50)


class YCB009GelatinBox(YCBOptions):
    file: str = "models/009_gelatin_box/textured.obj"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=50)


class YCB010PottedMeatCan(YCBOptions):
    file: str = "models/010_potted_meat_can/textured.obj"
    scale: float = 1.0


class YCB011Banana(YCBOptions):
    file: str = "models/011_banana/textured.obj"
    scale: float = 1.0


class YCB012Strawberry(YCBOptions):
    file: str = "models/012_strawberry/textured.obj"
    scale: float = 1.0


class YCB013Apple(YCBOptions):
    file: str = "models/013_apple/textured.obj"
    scale: float = 1.0


class YCB014Lemon(YCBOptions):
    file: str = "models/014_lemon/textured.obj"
    scale: float = 1.0


class YCB015Peach(YCBOptions):
    file: str = "models/015_peach/textured.obj"
    scale: float = 1.0


class YCB016Pear(YCBOptions):
    file: str = "models/016_pear/textured.obj"
    scale: float = 1.0


class YCB017Orange(YCBOptions):
    file: str = "models/017_orange/textured.obj"
    scale: float = 1.0


class YCB018Plum(YCBOptions):
    file: str = "models/018_plum/textured.obj"
    scale: float = 1.0


class YCB019PitcherBase(YCBOptions):
    file: str = "models/019_pitcher_base/textured.obj"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=100)


class YCB021BleachCleanser(YCBOptions):
    file: str = "models/021_bleach_cleanser/textured.obj"
    scale: float = 1.0


class YCB022WindexBottle(YCBOptions):
    file: str = "models/022_windex_bottle/textured.obj"
    scale: float = 1.0


class YCB024Bowl(YCBOptions):
    file: str = "models/024_bowl/textured.obj"
    scale: float = 1.0


class YCB025Mug(YCBOptions):
    file: str = "models/025_mug/textured.obj"
    scale: float = 1.0


class YCB026Sponge(YCBOptions):
    file: str = "models/026_sponge/textured.obj"
    scale: float = 1.0


class YCB028SkilletLid(YCBOptions):
    file: str = "models/028_skillet_lid/textured.obj"
    scale: float = 1.0


class YCB029Plate(YCBOptions):
    file: str = "models/029_plate/textured.obj"
    scale: float = 1.0


class YCB030Fork(YCBOptions):
    file: str = "models/030_fork/textured.obj"
    scale: float = 1.0


class YCB031Spoon(YCBOptions):
    file: str = "models/031_spoon/textured.obj"
    scale: float = 1.0


class YCB032Knife(YCBOptions):
    file: str = "models/032_knife/textured.obj"
    scale: float = 1.0


class YCB033Spatula(YCBOptions):
    file: str = "models/033_spatula/textured.obj"
    scale: float = 1.0


class YCB035PowerDrill(YCBOptions):
    file: str = "models/035_power_drill/textured.obj"
    scale: float = 1.0


class YCB036WoodBlock(YCBOptions):
    file: str = "models/036_wood_block/textured.obj"
    scale: float = 0.8


class YCB037Scissors(YCBOptions):
    file: str = "models/037_scissors/textured.obj"
    scale: float = 1.0


class YCB038Padlock(YCBOptions):
    file: str = "models/038_padlock/textured.obj"
    scale: float = 1.0


class YCB040LargeMarker(YCBOptions):
    file: str = "models/040_large_marker/textured.obj"
    scale: float = 1.0


class YCB042AdjustableWrench(YCBOptions):
    file: str = "models/042_adjustable_wrench/textured.obj"
    scale: float = 1.0


class YCB043PhillipsScrewdriver(YCBOptions):
    file: str = "models/043_phillips_screwdriver/textured.obj"
    scale: float = 1.0


class YCB044FlatScrewdriver(YCBOptions):
    file: str = "models/044_flat_screwdriver/textured.obj"
    scale: float = 1.0


class YCB048Hammer(YCBOptions):
    file: str = "models/048_hammer/textured.obj"
    scale: float = 1.0


class YCB050MediumClamp(YCBOptions):
    file: str = "models/050_medium_clamp/textured.obj"
    scale: float = 1.0


class YCB051LargeClamp(YCBOptions):
    file: str = "models/051_large_clamp/textured.obj"
    scale: float = 1.0


class YCB052ExtraLargeClamp(YCBOptions):
    file: str = "models/052_extra_large_clamp/textured.obj"
    scale: float = 1.0


class YCB053MiniSoccerBall(YCBOptions):
    file: str = "models/053_mini_soccer_ball/textured.obj"
    scale: float = 0.5


class YCB054Softball(YCBOptions):
    file: str = "models/054_softball/textured.obj"
    scale: float = 0.7


class YCB055Baseball(YCBOptions):
    file: str = "models/055_baseball/textured.obj"
    scale: float = 1.0


class YCB056TennisBall(YCBOptions):
    file: str = "models/056_tennis_ball/textured.obj"
    scale: float = 1.0


class YCB057Racquetball(YCBOptions):
    file: str = "models/057_racquetball/textured.obj"
    scale: float = 1.0


class YCB058GolfBall(YCBOptions):
    file: str = "models/058_golf_ball/textured.obj"
    scale: float = 1.0


class YCB059Chain(YCBOptions):
    file: str = "models/059_chain/textured.obj"
    scale: float = 1.0


class YCB061FoamBrick(YCBOptions):
    file: str = "models/061_foam_brick/textured.obj"
    scale: float = 1.0


class YCB062Dice(YCBOptions):
    file: str = "models/062_dice/textured.obj"
    scale: float = 1.0


class YCB063AMarbles(YCBOptions):
    file: str = "models/063-a_marbles/textured.obj"
    scale: float = 1.0


class YCB063BMarbles(YCBOptions):
    file: str = "models/063-b_marbles/textured.obj"
    scale: float = 1.0


class YCB065ACups(YCBOptions):
    file: str = "models/065-a_cups/textured.obj"
    scale: float = 1.0


class YCB065BCups(YCBOptions):
    file: str = "models/065-b_cups/textured.obj"
    scale: float = 1.0


class YCB065CCups(YCBOptions):
    file: str = "models/065-c_cups/textured.obj"
    scale: float = 1.0


class YCB065DCups(YCBOptions):
    file: str = "models/065-d_cups/textured.obj"
    scale: float = 1.0


class YCB065ECups(YCBOptions):
    file: str = "models/065-e_cups/textured.obj"
    scale: float = 1.0


class YCB065FCups(YCBOptions):
    file: str = "models/065-f_cups/textured.obj"
    scale: float = 1.0


class YCB065GCups(YCBOptions):
    file: str = "models/065-g_cups/textured.obj"
    scale: float = 1.0


class YCB065HCups(YCBOptions):
    file: str = "models/065-h_cups/textured.obj"
    scale: float = 1.0


class YCB065ICups(YCBOptions):
    file: str = "models/065-i_cups/textured.obj"
    scale: float = 1.0


class YCB065JCups(YCBOptions):
    file: str = "models/065-j_cups/textured.obj"
    scale: float = 1.0


class YCB070AColoredWoodBlocks(YCBOptions):
    file: str = "models/070-a_colored_wood_blocks/textured.obj"
    scale: float = 0.5


class YCB070BColoredWoodBlocks(YCBOptions):
    file: str = "models/070-b_colored_wood_blocks/textured.obj"
    scale: float = 1.0


class YCB071NineHolePegTest(YCBOptions):
    file: str = "models/071_nine_hole_peg_test/textured.obj"
    scale: float = 0.5


class YCB072AToyAirplane(YCBOptions):
    file: str = "models/072-a_toy_airplane/textured.obj"
    scale: float = 1.0
    metadata: ClassVar[MetadataOptions] = MetadataOptions(density=200)


class YCB072BToyAirplane(YCBOptions):
    file: str = "models/072-b_toy_airplane/textured.obj"
    scale: float = 1.0


class YCB072CToyAirplane(YCBOptions):
    file: str = "models/072-c_toy_airplane/textured.obj"
    scale: float = 1.0


class YCB072DToyAirplane(YCBOptions):
    file: str = "models/072-d_toy_airplane/textured.obj"
    scale: float = 1.0


class YCB072EToyAirplane(YCBOptions):
    file: str = "models/072-e_toy_airplane/textured.obj"
    scale: float = 1.0


class YCB073ALegoDuplo(YCBOptions):
    file: str = "models/073-a_lego_duplo/textured.obj"
    scale: float = 1.0


class YCB073BLegoDuplo(YCBOptions):
    file: str = "models/073-b_lego_duplo/textured.obj"
    scale: float = 1.0


class YCB073CLegoDuplo(YCBOptions):
    file: str = "models/073-c_lego_duplo/textured.obj"
    scale: float = 1.0


class YCB073DLegoDuplo(YCBOptions):
    file: str = "models/073-d_lego_duplo/textured.obj"
    scale: float = 1.0


class YCB073ELegoDuplo(YCBOptions):
    file: str = "models/073-e_lego_duplo/textured.obj"
    scale: float = 1.0


class YCB073FLegoDuplo(YCBOptions):
    file: str = "models/073-f_lego_duplo/textured.obj"
    scale: float = 1.0


class YCB073GLegoDuplo(YCBOptions):
    file: str = "models/073-g_lego_duplo/textured.obj"
    scale: float = 0.5


class YCB077RubiksCube(YCBOptions):
    file: str = "models/077_rubiks_cube/textured.obj"
    scale: float = 1.0
