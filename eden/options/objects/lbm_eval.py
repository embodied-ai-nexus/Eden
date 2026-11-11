"""LBM-eval benchmark object options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions
from eden.utils.registry import Registry

LBM_EVAL_REGISTRY = Registry("LBM_EVAL")


class LBMEvalObjectOptions(EntityOptions):
    """Base options for LBM-eval graspable objects (groceries, kitchen accessories).

    Assets are ported from Drake-flavored SDFs and live at
    ``https://huggingface.co/datasets/Kashu7100/eden_lbm_eval``. The URDF link
    frame is z-up (collision primitives + inertia tensors are authored against
    that convention, and visual glTFs are rotated to match), with the object's
    natural "label-facing" front along +x.
    """

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    registry: str = "Kashu7100"
    dataset: str = "eden_lbm_eval"
    assets_subdir: ClassVar[str] = "lbm_eval"


# ─── Groceries / apples ──────────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMFakeBlueSweetCrispinApple(LBMEvalObjectOptions):
    file: str = "groceries/apples/fake_blue_sweet_crispin_apple.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeGreenSweetCrispinApple(LBMEvalObjectOptions):
    file: str = "groceries/apples/fake_green_sweet_crispin_apple.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeRedSweetCrispinApple(LBMEvalObjectOptions):
    file: str = "groceries/apples/fake_red_sweet_crispin_apple.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMGalaApple(LBMEvalObjectOptions):
    file: str = "groceries/apples/gala_apple.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMGoldenDeliciousApple(LBMEvalObjectOptions):
    file: str = "groceries/apples/golden_delicious_apple.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMGrannySmithApple(LBMEvalObjectOptions):
    file: str = "groceries/apples/granny_smith_apple.urdf"


# ─── Groceries / avocados ────────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMFuerteAvocado(LBMEvalObjectOptions):
    file: str = "groceries/avocados/fuerte_avocado.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMHassAvocado(LBMEvalObjectOptions):
    file: str = "groceries/avocados/hass_avocado.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMLulaAvocado(LBMEvalObjectOptions):
    file: str = "groceries/avocados/lula_avocado.urdf"


# ─── Groceries / bananas ─────────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMFakeOverripeBanana(LBMEvalObjectOptions):
    file: str = "groceries/bananas/fake_overripe_banana.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeRipeBanana(LBMEvalObjectOptions):
    file: str = "groceries/bananas/fake_ripe_banana.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeUnripeBanana(LBMEvalObjectOptions):
    file: str = "groceries/bananas/fake_unripe_banana.urdf"


# ─── Groceries / bell peppers ────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMRedBellPepperNoStem(LBMEvalObjectOptions):
    file: str = "groceries/bell_peppers/red_bell_pepper_no_stem.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMRedBellPepperNoStemScale95(LBMEvalObjectOptions):
    file: str = "groceries/bell_peppers/red_bell_pepper_no_stem_scale_95.urdf"


# ─── Groceries / berries (kiwis) ─────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMFakeRedKiwi(LBMEvalObjectOptions):
    file: str = "groceries/berries/fake_red_kiwi.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeRipeKiwi(LBMEvalObjectOptions):
    file: str = "groceries/berries/fake_ripe_kiwi.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeUnripeKiwi(LBMEvalObjectOptions):
    file: str = "groceries/berries/fake_unripe_kiwi.urdf"


# ─── Groceries / cereals ─────────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMDrakeFlakesCerealBox(LBMEvalObjectOptions):
    file: str = "groceries/cereals/drake_flakes_cereal_box.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMDrakeFlakesCerealBoxColorVariant1(LBMEvalObjectOptions):
    file: str = "groceries/cereals/drake_flakes_cereal_box_color_variant_1.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMLbmCerealBox(LBMEvalObjectOptions):
    file: str = "groceries/cereals/lbm_cereal_box.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMLbmCerealBoxColorVariant1(LBMEvalObjectOptions):
    file: str = "groceries/cereals/lbm_cereal_box_color_variant_1.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMPunyosCerealBox(LBMEvalObjectOptions):
    file: str = "groceries/cereals/punyos_cereal_box.urdf"


# ─── Groceries / oranges ─────────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMFakeRipeYcbOrange(LBMEvalObjectOptions):
    file: str = "groceries/oranges/fake_ripe_ycb_orange.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeSemiRipeYcbOrange(LBMEvalObjectOptions):
    file: str = "groceries/oranges/fake_semi_ripe_ycb_orange.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMFakeUnripeYcbOrange(LBMEvalObjectOptions):
    file: str = "groceries/oranges/fake_unripe_ycb_orange.urdf"


# ─── Groceries / pears ───────────────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMBosePear(LBMEvalObjectOptions):
    file: str = "groceries/pears/bose_pear.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMGreenAnjouPear(LBMEvalObjectOptions):
    file: str = "groceries/pears/green_anjou_pear.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMStarkrimsonPear(LBMEvalObjectOptions):
    file: str = "groceries/pears/starkrimson_pear.urdf"


# ─── Kitchen accessories / bins ──────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodyLargeBlack(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_large_black.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodyLargeBlue(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_large_blue.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodyLargeGreen(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_large_green.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodyLargeRed(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_large_red.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodyLargeWhite(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_large_white.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodyLargeYellow(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_large_yellow.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodySmallBlack(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_small_black.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodySmallBlue(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_small_blue.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodySmallGreen(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_small_green.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodySmallRed(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_small_red.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodySmallWhite(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_small_white.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseBodySmallYellow(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_body_small_yellow.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMContainerAmuseWhiteIssue7265BodyLow(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/bins/container_amuse_white_issue7265_body_low.urdf"


# ─── Kitchen accessories / crocks ────────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMSlnihamsBambooUtensilCrock(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/crocks/slnihams_bamboo_utensil_crock.urdf"


# ─── Kitchen accessories / cutting boards ────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMOxoPlasticCuttingBoard(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/cutting_boards/oxo_plastic_cutting_board.urdf"


# ─── Kitchen accessories / drying racks ──────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMSweetHomeDishDryingRackBase(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/drying_racks/sweet_home_dish_drying_rack_base.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMSweetHomeDishDryingRackUtensilHolder(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/drying_racks/sweet_home_dish_drying_rack_utensil_holder.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMSweetHomeDishDryingRackWireframe(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/drying_racks/sweet_home_dish_drying_rack_wireframe.urdf"


@LBM_EVAL_REGISTRY.register()
class LBMSweetHomeDishDryingRackWireframeInitializationBody(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/drying_racks/sweet_home_dish_drying_rack_wireframe_initialization_body.urdf"


# ─── Kitchen accessories / mug holders ───────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMDorhorsWoodMugHolder(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/mug_holders/dorhors_wood_mug_holder.urdf"


# ─── Kitchen accessories / shelves ───────────────────────────────────────────


@LBM_EVAL_REGISTRY.register()
class LBMVivoCustomSingleShelf(LBMEvalObjectOptions):
    file: str = "kitchen_accessories/shelves/vivo_custom_single_shelf.urdf"


# ─── Riverway franka station ─────────────────────────────────────────────────
#
# Station kinematics (all values in base_link frame):
#   base_link  -> station_table_cart  : xyz=(0, 0, -0.099295)
#   table_cart -> arm_cart_left       : xyz=(0.6896, 0, 0), yaw=+90°
#                -> gantry_left       : xyz=(0, -0.019, 0)
#                -> plate_left        : xyz=(-0.1675, 0, 0.041)
#   table_cart -> arm_cart_right      : xyz=(0, -0.6896, 0)
#                -> gantry_right      : xyz=(0, -0.012, 0)
#                -> plate_right       : xyz=(0.185, 0, 0.041)
#
# Plate top surface (+0.0127 above the plate link origin) lands at:
#   left  plate top: ( 0.7086, -0.1675, -0.04560)  ← +x site
#   right plate top: ( 0.1850, -0.7016, -0.04560)  ← -y site
#
# Wheel bottoms are at z=-0.63185 in base_link, so the station's default root
# pose lifts everything by 0.63185 along +z to rest the cart on z=0.

#: Vertical lift applied to the station so cart wheels rest on z=0.
LBM_STATION_BASE_HEIGHT: float = 0.63185

#: Plate-top z in base_link frame; world z = station_base_z + this.
LBM_PLATE_TOP_Z_LOCAL: float = -0.04560

ROBOT_OFFSET_XY = 0.04

#: Default world-frame mount poses for the left/right Franka, derived from the
#: station gantry base plates and the station's nominal lift.
LBM_LEFT_ROBOT_POS: tuple[float, float, float] = (
    0.7086 - ROBOT_OFFSET_XY,
    -0.1675,
    LBM_STATION_BASE_HEIGHT + LBM_PLATE_TOP_Z_LOCAL,
)
LBM_RIGHT_ROBOT_POS: tuple[float, float, float] = (
    0.1850,
    -0.7016 + ROBOT_OFFSET_XY,
    LBM_STATION_BASE_HEIGHT + LBM_PLATE_TOP_Z_LOCAL,
)

#: Default world-frame mount orientations (wxyz). Left=180° about z, right=90° about z.
LBM_LEFT_ROBOT_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
LBM_RIGHT_ROBOT_QUAT: tuple[float, float, float, float] = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)


@LBM_EVAL_REGISTRY.register()
class LBMRiverwayStation(LBMEvalObjectOptions):
    """Riverway dual-Franka mobile station.

    Like the graspable LBM-eval objects, the station URDF is authored z-up /
    x-forward (the standard ROS frame). Fixed-base by default.
    """

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    file: str = "franka_station/riverway_station.urdf"
    is_fixed_base: bool = True

    default_root_pos: tuple[float, float, float] = (0.0, 0.0, LBM_STATION_BASE_HEIGHT)

    # Support-surface configuration: only the table top should accept placed
    # objects. The arm-cart tops, gantry plates, and lower cart frame are also
    # horizontal but reserved for the franka mount / are otherwise not part of
    # the workspace. links_to_keep is required because Genesis collapses
    # fixed-joint chains into base_link by default, which would erase the
    # station_table_top link before _select_support_links could match it.
    is_support_enabled: bool = True
    support_links_name: list[str] = ["station_table_top"]
    links_to_keep: list[str] = ["station_table_top"]
