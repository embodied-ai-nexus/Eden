"""RoboCasa object and kitchen-layout options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions
from eden.utils.registry import Registry

ROBOCASA_ACCESSORIES_REGISTRY = Registry("ROBOCASA_ACCESSORIES")
ROBOCASA_KITCHEN_REGISTRY = Registry("ROBOCASA_KITCHEN")


class _RobocasaOptions(EntityOptions):
    """Base options for RoboCasa (https://robocasa.ai/) kitchen scenes and accessories: fixed-base, z-up, front -y."""

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (0, -1, 0)
    is_fixed_base: bool = True

    registry: str = "Kashu7100"
    dataset: str = "eden_robocasa"
    assets_subdir: ClassVar[str] = "robocasa"


# ──────────────────────────────────────────────────────────────────────────────
# 10 layouts × 12 styles = 120 kitchen scenes
# ──────────────────────────────────────────────────────────────────────────────


# --- ONE_WALL_SMALL (1/10) ---
class _RobocasaOneWallSmallOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f12_sink_main_group_handle_joint",
        "f12_sink_main_group_handle_temp_joint",
        "f12_sink_main_group_spout_joint",
        "f14_stove_main_group_knob_front_left_joint",
        "f14_stove_main_group_knob_front_right_joint",
        "f14_stove_main_group_knob_rear_center_joint",
        "f14_stove_main_group_knob_rear_left_joint",
        "f14_stove_main_group_knob_rear_right_joint",
        "f18_cab_1_main_group_doorhinge",
        "f19_cab_2_main_group_leftdoorhinge",
        "f19_cab_2_main_group_rightdoorhinge",
        "f20_cab_main_main_group_leftdoorhinge",
        "f20_cab_main_main_group_rightdoorhinge",
        "f21_microwave_main_group_microjoint",
        "f22_cab_micro_main_group_leftdoorhinge",
        "f22_cab_micro_main_group_rightdoorhinge",
        "f24_cab_4_main_group_leftdoorhinge",
        "f24_cab_4_main_group_rightdoorhinge",
        "f26_stack_1_main_group_1_slidejoint",
        "f27_stack_1_main_group_2_slidejoint",
        "f28_stack_1_main_group_3_slidejoint",
        "f29_stack_1_main_group_4_slidejoint",
        "f31_stack_2_main_group_1_leftdoorhinge",
        "f31_stack_2_main_group_1_rightdoorhinge",
        "f35_stack_3_main_group_1_doorhinge",
        "f37_stack_4_main_group_1_doorhinge",
        "f38_stack_4_main_group_2_slidejoint",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallSmallIndustrial(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallSmallScandinavian(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallSmallCoastal(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_small_modern_1")
class RobocasaOneWallSmallModern1(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_small_modern_2")
class RobocasaOneWallSmallModern2(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_small_traditional_1")
class RobocasaOneWallSmallTraditional1(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_small_traditional_2")
class RobocasaOneWallSmallTraditional2(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallSmallFarmhouse(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallSmallRustic(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallSmallMediterranean(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_small_transitional_1")
class RobocasaOneWallSmallTransitional1(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_small_transitional_2")
class RobocasaOneWallSmallTransitional2(_RobocasaOneWallSmallOptions):
    file: str = "one_wall_small__transitional_2/kitchen_scene.xml"


# --- ONE_WALL_LARGE (2/10) ---
class _RobocasaOneWallLargeOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f14_stovetop_main_group_knob_front_center_joint",
        "f14_stovetop_main_group_knob_front_left_joint",
        "f14_stovetop_main_group_knob_front_right_joint",
        "f14_stovetop_main_group_knob_rear_center_joint",
        "f14_stovetop_main_group_knob_rear_left_joint",
        "f14_stovetop_main_group_knob_rear_right_joint",
        "f18_bottom_main_group_1_slidejoint",
        "f19_bottom_main_group_2_slidejoint",
        "f22_microwave_main_group_microjoint",
        "f24_top_main_group_leftdoorhinge",
        "f24_top_main_group_rightdoorhinge",
        "f27_fridge_top_main_group_leftdoorhinge",
        "f27_fridge_top_main_group_rightdoorhinge",
        "f28_cab_main_main_group_leftdoorhinge",
        "f28_cab_main_main_group_rightdoorhinge",
        "f31_cab_2_main_group_leftdoorhinge",
        "f31_cab_2_main_group_rightdoorhinge",
        "f33_stack_3_main_group_1_leftdoorhinge",
        "f33_stack_3_main_group_1_rightdoorhinge",
        "f34_stack_3_main_group_2_slidejoint",
        "f35_stack_3_main_group_3_slidejoint",
        "f37_stack_4_main_group_1_leftdoorhinge",
        "f37_stack_4_main_group_1_rightdoorhinge",
        "f38_stack_4_main_group_2_slidejoint",
        "f39_stack_4_main_group_3_slidejoint",
        "f41_stack_5_main_group_1_leftdoorhinge",
        "f41_stack_5_main_group_1_rightdoorhinge",
        "f42_stack_5_main_group_2_slidejoint",
        "f43_stack_5_main_group_3_slidejoint",
        "f45_stack_6_main_group_1_slidejoint",
        "f46_stack_6_main_group_2_slidejoint",
        "f47_stack_6_main_group_3_slidejoint",
        "f48_stack_6_main_group_4_slidejoint",
        "f53_sink_island_group_handle_joint",
        "f53_sink_island_group_handle_temp_joint",
        "f53_sink_island_group_spout_joint",
        "f56_island_stack_1_island_group_1_leftdoorhinge",
        "f56_island_stack_1_island_group_1_rightdoorhinge",
        "f62_island_cab_left_island_group_1_slidejoint",
        "f63_island_cab_left_island_group_2_slidejoint",
        "f64_island_cab_left_island_group_3_slidejoint",
        "f65_island_cab_left_island_group_4_slidejoint",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallLargeIndustrial(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallLargeScandinavian(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallLargeCoastal(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_large_modern_1")
class RobocasaOneWallLargeModern1(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_large_modern_2")
class RobocasaOneWallLargeModern2(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_large_traditional_1")
class RobocasaOneWallLargeTraditional1(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_large_traditional_2")
class RobocasaOneWallLargeTraditional2(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallLargeFarmhouse(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallLargeRustic(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaOneWallLargeMediterranean(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_large_transitional_1")
class RobocasaOneWallLargeTransitional1(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_one_wall_large_transitional_2")
class RobocasaOneWallLargeTransitional2(_RobocasaOneWallLargeOptions):
    file: str = "one_wall_large__transitional_2/kitchen_scene.xml"


# --- L_SHAPED_SMALL (3/10) ---
class _RobocasaLShapedSmallOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f11_cab_1_main_group_leftdoorhinge",
        "f11_cab_1_main_group_rightdoorhinge",
        "f12_cab_2_main_group_leftdoorhinge",
        "f12_cab_2_main_group_rightdoorhinge",
        "f13_cab_main_main_group_leftdoorhinge",
        "f13_cab_main_main_group_rightdoorhinge",
        "f16_sink_main_group_handle_joint",
        "f16_sink_main_group_handle_temp_joint",
        "f16_sink_main_group_spout_joint",
        "f20_stack_1_main_group_1_slidejoint",
        "f21_stack_1_main_group_2_slidejoint",
        "f22_stack_1_main_group_3_slidejoint",
        "f23_stack_1_main_group_4_slidejoint",
        "f25_stack_2_main_group_1_leftdoorhinge",
        "f25_stack_2_main_group_1_rightdoorhinge",
        "f28_stack_3_main_group_1_doorhinge",
        "f34_stove_right_group_knob_front_left_joint",
        "f34_stove_right_group_knob_front_right_joint",
        "f34_stove_right_group_knob_rear_center_joint",
        "f34_stove_right_group_knob_rear_left_joint",
        "f34_stove_right_group_knob_rear_right_joint",
        "f39_stack_1_right_group_1_doorhinge",
        "f40_stack_1_right_group_2_slidejoint",
        "f42_stack_2_right_group_1_leftdoorhinge",
        "f42_stack_2_right_group_1_rightdoorhinge",
        "f43_stack_2_right_group_2_slidejoint",
        "f44_stack_2_right_group_3_slidejoint",
        "f46_stack_3_right_group_1_slidejoint",
        "f47_stack_3_right_group_2_slidejoint",
        "f48_stack_3_right_group_3_slidejoint",
        "f49_stack_3_right_group_4_slidejoint",
        "f50_cab_1_right_group_doorhinge",
        "f51_microwave_right_group_microjoint",
        "f52_cab_micro_right_group_leftdoorhinge",
        "f52_cab_micro_right_group_rightdoorhinge",
        "f53_cab_2_right_group_doorhinge",
        "f55_cab_3_right_group_leftdoorhinge",
        "f55_cab_3_right_group_rightdoorhinge",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedSmallIndustrial(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedSmallScandinavian(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedSmallCoastal(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_small_modern_1")
class RobocasaLShapedSmallModern1(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_small_modern_2")
class RobocasaLShapedSmallModern2(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_small_traditional_1")
class RobocasaLShapedSmallTraditional1(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_small_traditional_2")
class RobocasaLShapedSmallTraditional2(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedSmallFarmhouse(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedSmallRustic(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedSmallMediterranean(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_small_transitional_1")
class RobocasaLShapedSmallTransitional1(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_small_transitional_2")
class RobocasaLShapedSmallTransitional2(_RobocasaLShapedSmallOptions):
    file: str = "l_shaped_small__transitional_2/kitchen_scene.xml"


# --- L_SHAPED_LARGE (4/10) ---
class _RobocasaLShapedLargeOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f12_stovetop_main_group_knob_front_center_joint",
        "f12_stovetop_main_group_knob_front_left_joint",
        "f12_stovetop_main_group_knob_front_right_joint",
        "f12_stovetop_main_group_knob_rear_center_joint",
        "f12_stovetop_main_group_knob_rear_left_joint",
        "f12_stovetop_main_group_knob_rear_right_joint",
        "f17_stack_1_main_group_1_slidejoint",
        "f18_stack_1_main_group_2_slidejoint",
        "f19_stack_1_main_group_3_slidejoint",
        "f20_stack_1_main_group_4_slidejoint",
        "f22_stack_2_main_group_1_leftdoorhinge",
        "f22_stack_2_main_group_1_rightdoorhinge",
        "f25_stack_3_main_group_1_leftdoorhinge",
        "f25_stack_3_main_group_1_rightdoorhinge",
        "f26_stack_3_main_group_2_slidejoint",
        "f27_stack_3_main_group_3_slidejoint",
        "f29_cab_main_main_group_leftdoorhinge",
        "f29_cab_main_main_group_rightdoorhinge",
        "f31_fridge_cab_main_group_leftdoorhinge",
        "f31_fridge_cab_main_group_rightdoorhinge",
        "f34_cab_1_left_group_doorhinge",
        "f36_cab_2_left_group_doorhinge",
        "f39_sink_left_group_handle_joint",
        "f39_sink_left_group_handle_temp_joint",
        "f39_sink_left_group_spout_joint",
        "f44_bottom_left_group_1_slidejoint",
        "f45_bottom_left_group_2_slidejoint",
        "f48_microwave_left_group_microjoint",
        "f50_top_left_group_leftdoorhinge",
        "f50_top_left_group_rightdoorhinge",
        "f51_stack_1_left_group_doorhinge",
        "f54_stack_2_left_group_1_leftdoorhinge",
        "f54_stack_2_left_group_1_rightdoorhinge",
        "f57_stack_3_left_group_1_leftdoorhinge",
        "f57_stack_3_left_group_1_rightdoorhinge",
        "f58_stack_3_left_group_2_slidejoint",
        "f59_stack_3_left_group_3_slidejoint",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedLargeIndustrial(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedLargeScandinavian(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedLargeCoastal(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_large_modern_1")
class RobocasaLShapedLargeModern1(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_large_modern_2")
class RobocasaLShapedLargeModern2(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_large_traditional_1")
class RobocasaLShapedLargeTraditional1(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_large_traditional_2")
class RobocasaLShapedLargeTraditional2(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedLargeFarmhouse(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedLargeRustic(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaLShapedLargeMediterranean(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_large_transitional_1")
class RobocasaLShapedLargeTransitional1(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_l_shaped_large_transitional_2")
class RobocasaLShapedLargeTransitional2(_RobocasaLShapedLargeOptions):
    file: str = "l_shaped_large__transitional_2/kitchen_scene.xml"


# --- GALLEY (5/10) ---
class _RobocasaGalleyOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f15_sink_left_group_handle_joint",
        "f15_sink_left_group_handle_temp_joint",
        "f15_sink_left_group_spout_joint",
        "f17_cab_1_left_group_leftdoorhinge",
        "f17_cab_1_left_group_rightdoorhinge",
        "f21_stack_1_left_group_1_slidejoint",
        "f22_stack_1_left_group_2_slidejoint",
        "f23_stack_1_left_group_3_slidejoint",
        "f24_stack_1_left_group_4_slidejoint",
        "f26_stack_2_left_group_1_leftdoorhinge",
        "f26_stack_2_left_group_1_rightdoorhinge",
        "f27_stack_2_left_group_2_slidejoint",
        "f28_stack_2_left_group_3_slidejoint",
        "f31_stack_3_left_group_1_leftdoorhinge",
        "f31_stack_3_left_group_1_rightdoorhinge",
        "f34_stack_4_left_group_1_doorhinge",
        "f35_stack_4_left_group_2_slidejoint",
        "f38_stove_main_group_knob_front_left_joint",
        "f38_stove_main_group_knob_front_right_joint",
        "f38_stove_main_group_knob_rear_center_joint",
        "f38_stove_main_group_knob_rear_left_joint",
        "f38_stove_main_group_knob_rear_right_joint",
        "f40_cab_main_main_group_leftdoorhinge",
        "f40_cab_main_main_group_rightdoorhinge",
        "f41_cab_1_main_group_leftdoorhinge",
        "f41_cab_1_main_group_rightdoorhinge",
        "f42_microwave_main_group_microjoint",
        "f43_cab_micro_main_group_leftdoorhinge",
        "f43_cab_micro_main_group_rightdoorhinge",
        "f44_cab_2_main_group_leftdoorhinge",
        "f44_cab_2_main_group_rightdoorhinge",
        "f45_cab_3_main_group_leftdoorhinge",
        "f45_cab_3_main_group_rightdoorhinge",
        "f47_stack_1_main_group_1_leftdoorhinge",
        "f47_stack_1_main_group_1_rightdoorhinge",
        "f48_stack_1_main_group_2_slidejoint",
        "f49_stack_1_main_group_3_slidejoint",
        "f51_stack_2_main_group_1_slidejoint",
        "f52_stack_2_main_group_2_slidejoint",
        "f53_stack_2_main_group_3_slidejoint",
        "f54_stack_2_main_group_4_slidejoint",
        "f56_stack_3_main_group_1_leftdoorhinge",
        "f56_stack_3_main_group_1_rightdoorhinge",
        "f57_stack_3_main_group_2_slidejoint",
        "f58_stack_3_main_group_3_slidejoint",
        "f60_stack_4_main_group_1_leftdoorhinge",
        "f60_stack_4_main_group_1_rightdoorhinge",
        "f61_stack_4_main_group_2_slidejoint",
        "f62_stack_4_main_group_3_slidejoint",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGalleyIndustrial(_RobocasaGalleyOptions):
    file: str = "galley__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGalleyScandinavian(_RobocasaGalleyOptions):
    file: str = "galley__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGalleyCoastal(_RobocasaGalleyOptions):
    file: str = "galley__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_galley_modern_1")
class RobocasaGalleyModern1(_RobocasaGalleyOptions):
    file: str = "galley__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_galley_modern_2")
class RobocasaGalleyModern2(_RobocasaGalleyOptions):
    file: str = "galley__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_galley_traditional_1")
class RobocasaGalleyTraditional1(_RobocasaGalleyOptions):
    file: str = "galley__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_galley_traditional_2")
class RobocasaGalleyTraditional2(_RobocasaGalleyOptions):
    file: str = "galley__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGalleyFarmhouse(_RobocasaGalleyOptions):
    file: str = "galley__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGalleyRustic(_RobocasaGalleyOptions):
    file: str = "galley__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGalleyMediterranean(_RobocasaGalleyOptions):
    file: str = "galley__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_galley_transitional_1")
class RobocasaGalleyTransitional1(_RobocasaGalleyOptions):
    file: str = "galley__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_galley_transitional_2")
class RobocasaGalleyTransitional2(_RobocasaGalleyOptions):
    file: str = "galley__transitional_2/kitchen_scene.xml"


# --- U_SHAPED_SMALL (6/10) ---
class _RobocasaUShapedSmallOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f15_cab_main_main_group_leftdoorhinge",
        "f15_cab_main_main_group_rightdoorhinge",
        "f19_sink_main_group_handle_joint",
        "f19_sink_main_group_handle_temp_joint",
        "f19_sink_main_group_spout_joint",
        "f23_stack_1_main_group_1_slidejoint",
        "f24_stack_1_main_group_2_slidejoint",
        "f25_stack_1_main_group_3_slidejoint",
        "f26_stack_1_main_group_4_slidejoint",
        "f28_stack_2_main_group_1_leftdoorhinge",
        "f28_stack_2_main_group_1_rightdoorhinge",
        "f32_stack_3_main_group_1_leftdoorhinge",
        "f32_stack_3_main_group_1_rightdoorhinge",
        "f33_stack_3_main_group_2_slidejoint",
        "f34_stack_3_main_group_3_slidejoint",
        "f38_stove_right_group_knob_front_left_joint",
        "f38_stove_right_group_knob_front_right_joint",
        "f38_stove_right_group_knob_rear_center_joint",
        "f38_stove_right_group_knob_rear_left_joint",
        "f38_stove_right_group_knob_rear_right_joint",
        "f44_stack_1_right_group_1_doorhinge",
        "f45_stack_1_right_group_2_slidejoint",
        "f47_stack_2_right_group_1_doorhinge",
        "f48_stack_2_right_group_2_slidejoint",
        "f49_cab_1_right_group_doorhinge",
        "f50_microwave_right_group_microjoint",
        "f51_cab_micro_right_group_leftdoorhinge",
        "f51_cab_micro_right_group_rightdoorhinge",
        "f52_cab_2_right_group_doorhinge",
        "f53_cab_3_right_group_leftdoorhinge",
        "f53_cab_3_right_group_rightdoorhinge",
        "f59_stack_1_left_group_1_slidejoint",
        "f60_stack_1_left_group_2_slidejoint",
        "f61_stack_1_left_group_3_slidejoint",
        "f62_stack_1_left_group_4_slidejoint",
        "f64_stack_2_left_group_1_leftdoorhinge",
        "f64_stack_2_left_group_1_rightdoorhinge",
        "f65_stack_2_left_group_2_slidejoint",
        "f66_stack_2_left_group_3_slidejoint",
        "f68_stack_3_left_group_1_leftdoorhinge",
        "f68_stack_3_left_group_1_rightdoorhinge",
        "f69_stack_3_left_group_2_slidejoint",
        "f70_cab_1_left_group_leftdoorhinge",
        "f70_cab_1_left_group_rightdoorhinge",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedSmallIndustrial(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedSmallScandinavian(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedSmallCoastal(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_small_modern_1")
class RobocasaUShapedSmallModern1(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_small_modern_2")
class RobocasaUShapedSmallModern2(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_small_traditional_1")
class RobocasaUShapedSmallTraditional1(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_small_traditional_2")
class RobocasaUShapedSmallTraditional2(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedSmallFarmhouse(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedSmallRustic(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedSmallMediterranean(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_small_transitional_1")
class RobocasaUShapedSmallTransitional1(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_small_transitional_2")
class RobocasaUShapedSmallTransitional2(_RobocasaUShapedSmallOptions):
    file: str = "u_shaped_small__transitional_2/kitchen_scene.xml"


# --- U_SHAPED_LARGE (7/10) ---
class _RobocasaUShapedLargeOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f16_stovetop_main_group_knob_front_center_joint",
        "f16_stovetop_main_group_knob_front_left_joint",
        "f16_stovetop_main_group_knob_front_right_joint",
        "f16_stovetop_main_group_knob_rear_center_joint",
        "f16_stovetop_main_group_knob_rear_left_joint",
        "f16_stovetop_main_group_knob_rear_right_joint",
        "f21_cab_1_main_group_doorhinge",
        "f25_cab_main_main_group_doorhinge",
        "f29_stack_1_main_group_1_leftdoorhinge",
        "f29_stack_1_main_group_1_rightdoorhinge",
        "f30_stack_1_main_group_2_slidejoint",
        "f31_stack_1_main_group_3_slidejoint",
        "f33_stack_2_main_group_1_leftdoorhinge",
        "f33_stack_2_main_group_1_rightdoorhinge",
        "f34_stack_2_main_group_2_slidejoint",
        "f35_stack_2_main_group_3_slidejoint",
        "f37_stack_3_main_group_1_slidejoint",
        "f38_stack_3_main_group_2_slidejoint",
        "f39_stack_3_main_group_3_slidejoint",
        "f40_stack_3_main_group_4_slidejoint",
        "f42_stack_4_main_group_1_leftdoorhinge",
        "f42_stack_4_main_group_1_rightdoorhinge",
        "f44_stack_5_main_group_1_leftdoorhinge",
        "f44_stack_5_main_group_1_rightdoorhinge",
        "f45_stack_5_main_group_2_slidejoint",
        "f46_stack_5_main_group_3_slidejoint",
        "f48_stack_6_main_group_1_doorhinge",
        "f49_stack_6_main_group_2_slidejoint",
        "f53_sink_island_group_handle_joint",
        "f53_sink_island_group_handle_temp_joint",
        "f53_sink_island_group_spout_joint",
        "f56_island_stack_1_island_group_1_leftdoorhinge",
        "f56_island_stack_1_island_group_1_rightdoorhinge",
        "f62_island_cab_right_island_group_1_slidejoint",
        "f63_island_cab_right_island_group_2_slidejoint",
        "f64_island_cab_right_island_group_3_slidejoint",
        "f65_island_cab_right_island_group_4_slidejoint",
        "f76_bottom_left_group_1_slidejoint",
        "f77_bottom_left_group_2_slidejoint",
        "f80_microwave_left_group_microjoint",
        "f82_top_left_group_leftdoorhinge",
        "f82_top_left_group_rightdoorhinge",
        "f84_stack_1_left_group_1_doorhinge",
        "f85_stack_1_left_group_2_slidejoint",
        "f87_stack_2_left_group_1_leftdoorhinge",
        "f87_stack_2_left_group_1_rightdoorhinge",
        "f88_stack_2_left_group_2_slidejoint",
        "f89_stack_2_left_group_3_slidejoint",
        "f91_stack_3_left_group_1_slidejoint",
        "f92_stack_3_left_group_2_slidejoint",
        "f93_stack_3_left_group_3_slidejoint",
        "f94_stack_3_left_group_4_slidejoint",
        "f96_cab_2_left_group_doorhinge",
        "f102_stack_1_right_group_1_leftdoorhinge",
        "f102_stack_1_right_group_1_rightdoorhinge",
        "f103_stack_1_right_group_2_slidejoint",
        "f104_stack_1_right_group_3_slidejoint",
        "f106_stack_2_right_group_1_doorhinge",
        "f107_stack_2_right_group_2_slidejoint",
        "f109_stack_3_right_group_1_slidejoint",
        "f110_stack_3_right_group_2_slidejoint",
        "f111_stack_3_right_group_3_slidejoint",
        "f112_stack_3_right_group_4_slidejoint",
        "f113_cab_1_right_group_leftdoorhinge",
        "f113_cab_1_right_group_rightdoorhinge",
        "f114_cab_2_right_group_leftdoorhinge",
        "f114_cab_2_right_group_rightdoorhinge",
        "f115_fridge_cab_right_group_leftdoorhinge",
        "f115_fridge_cab_right_group_rightdoorhinge",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedLargeIndustrial(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedLargeScandinavian(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedLargeCoastal(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_large_modern_1")
class RobocasaUShapedLargeModern1(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_large_modern_2")
class RobocasaUShapedLargeModern2(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_large_traditional_1")
class RobocasaUShapedLargeTraditional1(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_large_traditional_2")
class RobocasaUShapedLargeTraditional2(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedLargeFarmhouse(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedLargeRustic(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaUShapedLargeMediterranean(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_large_transitional_1")
class RobocasaUShapedLargeTransitional1(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_u_shaped_large_transitional_2")
class RobocasaUShapedLargeTransitional2(_RobocasaUShapedLargeOptions):
    file: str = "u_shaped_large__transitional_2/kitchen_scene.xml"


# --- G_SHAPED_SMALL (8/10) ---
class _RobocasaGShapedSmallOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f18_cab_main_main_group_leftdoorhinge",
        "f18_cab_main_main_group_rightdoorhinge",
        "f20_cab_1_main_group_doorhinge",
        "f24_stack_1_main_group_1_leftdoorhinge",
        "f24_stack_1_main_group_1_rightdoorhinge",
        "f25_stack_1_main_group_2_slidejoint",
        "f26_stack_1_main_group_3_slidejoint",
        "f28_stack_2_main_group_1_leftdoorhinge",
        "f28_stack_2_main_group_1_rightdoorhinge",
        "f29_stack_2_main_group_2_slidejoint",
        "f30_stack_2_main_group_3_slidejoint",
        "f32_stack_3_main_group_1_slidejoint",
        "f33_stack_3_main_group_2_slidejoint",
        "f34_stack_3_main_group_3_slidejoint",
        "f35_stack_3_main_group_4_slidejoint",
        "f44_stove_left_group_knob_front_left_joint",
        "f44_stove_left_group_knob_front_right_joint",
        "f44_stove_left_group_knob_rear_center_joint",
        "f44_stove_left_group_knob_rear_left_joint",
        "f44_stove_left_group_knob_rear_right_joint",
        "f50_stack_1_left_group_1_doorhinge",
        "f51_stack_1_left_group_2_slidejoint",
        "f53_stack_2_left_group_1_doorhinge",
        "f54_stack_2_left_group_2_slidejoint",
        "f55_cab_1_left_group_doorhinge",
        "f56_microwave_left_group_microjoint",
        "f57_cab_micro_left_group_leftdoorhinge",
        "f57_cab_micro_left_group_rightdoorhinge",
        "f58_cab_2_left_group_doorhinge",
        "f59_cab_3_left_group_leftdoorhinge",
        "f59_cab_3_left_group_rightdoorhinge",
        "f62_sink_right_group_handle_joint",
        "f62_sink_right_group_handle_temp_joint",
        "f62_sink_right_group_spout_joint",
        "f67_stack_1_right_group_1_leftdoorhinge",
        "f67_stack_1_right_group_1_rightdoorhinge",
        "f71_stack_2_right_group_1_slidejoint",
        "f72_stack_2_right_group_2_slidejoint",
        "f73_stack_2_right_group_3_slidejoint",
        "f74_stack_2_right_group_4_slidejoint",
        "f75_cab_1_right_group_leftdoorhinge",
        "f75_cab_1_right_group_rightdoorhinge",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedSmallIndustrial(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedSmallScandinavian(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedSmallCoastal(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_small_modern_1")
class RobocasaGShapedSmallModern1(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_small_modern_2")
class RobocasaGShapedSmallModern2(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_small_traditional_1")
class RobocasaGShapedSmallTraditional1(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_small_traditional_2")
class RobocasaGShapedSmallTraditional2(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedSmallFarmhouse(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedSmallRustic(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedSmallMediterranean(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_small_transitional_1")
class RobocasaGShapedSmallTransitional1(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_small_transitional_2")
class RobocasaGShapedSmallTransitional2(_RobocasaGShapedSmallOptions):
    file: str = "g_shaped_small__transitional_2/kitchen_scene.xml"


# --- G_SHAPED_LARGE (9/10) ---
class _RobocasaGShapedLargeOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f16_stovetop_main_group_knob_front_center_joint",
        "f16_stovetop_main_group_knob_front_left_joint",
        "f16_stovetop_main_group_knob_front_right_joint",
        "f16_stovetop_main_group_knob_rear_center_joint",
        "f16_stovetop_main_group_knob_rear_left_joint",
        "f16_stovetop_main_group_knob_rear_right_joint",
        "f21_cab_1_main_group_leftdoorhinge",
        "f21_cab_1_main_group_rightdoorhinge",
        "f22_cab_main_main_group_leftdoorhinge",
        "f22_cab_main_main_group_rightdoorhinge",
        "f23_cab_2_main_group_doorhinge",
        "f28_stack_1_main_group_1_slidejoint",
        "f29_stack_1_main_group_2_slidejoint",
        "f30_stack_1_main_group_3_slidejoint",
        "f31_stack_1_main_group_4_slidejoint",
        "f33_stack_2_main_group_1_leftdoorhinge",
        "f33_stack_2_main_group_1_rightdoorhinge",
        "f34_stack_2_main_group_2_slidejoint",
        "f35_stack_2_main_group_3_slidejoint",
        "f37_stack_3_main_group_1_leftdoorhinge",
        "f37_stack_3_main_group_1_rightdoorhinge",
        "f38_stack_3_main_group_2_slidejoint",
        "f39_stack_3_main_group_3_slidejoint",
        "f41_stack_4_main_group_1_leftdoorhinge",
        "f41_stack_4_main_group_1_rightdoorhinge",
        "f43_stack_5_main_group_1_doorhinge",
        "f44_stack_5_main_group_2_slidejoint",
        "f54_sink_left_group_handle_joint",
        "f54_sink_left_group_handle_temp_joint",
        "f54_sink_left_group_spout_joint",
        "f58_stack_1_left_group_1_leftdoorhinge",
        "f58_stack_1_left_group_1_rightdoorhinge",
        "f59_stack_1_left_group_2_slidejoint",
        "f60_stack_1_left_group_3_slidejoint",
        "f62_stack_2_left_group_1_leftdoorhinge",
        "f62_stack_2_left_group_1_rightdoorhinge",
        "f65_stack_3_left_group_1_doorhinge",
        "f67_cab_1_left_group_leftdoorhinge",
        "f67_cab_1_left_group_rightdoorhinge",
        "f68_cab_2_left_group_doorhinge",
        "f72_stack_left_right_group_1_doorhinge",
        "f73_stack_left_right_group_2_doorhinge",
        "f76_fridge_top_right_group_leftdoorhinge",
        "f76_fridge_top_right_group_rightdoorhinge",
        "f78_stack_right_right_group_1_doorhinge",
        "f79_stack_right_right_group_2_doorhinge",
        "f83_bottom_right_group_1_slidejoint",
        "f84_bottom_right_group_2_slidejoint",
        "f87_microwave_right_group_microjoint",
        "f89_top_right_group_leftdoorhinge",
        "f89_top_right_group_rightdoorhinge",
        "f91_side_bottom_right_group_1_slidejoint",
        "f92_side_bottom_right_group_2_slidejoint",
        "f93_side_bottom_right_group_3_leftdoorhinge",
        "f93_side_bottom_right_group_3_rightdoorhinge",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedLargeIndustrial(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedLargeScandinavian(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedLargeCoastal(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_large_modern_1")
class RobocasaGShapedLargeModern1(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_large_modern_2")
class RobocasaGShapedLargeModern2(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_large_traditional_1")
class RobocasaGShapedLargeTraditional1(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_large_traditional_2")
class RobocasaGShapedLargeTraditional2(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedLargeFarmhouse(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedLargeRustic(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaGShapedLargeMediterranean(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_large_transitional_1")
class RobocasaGShapedLargeTransitional1(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_g_shaped_large_transitional_2")
class RobocasaGShapedLargeTransitional2(_RobocasaGShapedLargeOptions):
    file: str = "g_shaped_large__transitional_2/kitchen_scene.xml"


# --- WRAPAROUND (10/10) ---
class _RobocasaWraparoundOptions(_RobocasaOptions):
    dofs_name: list[str] = [
        "f21_stove_main_group_knob_front_left_joint",
        "f21_stove_main_group_knob_front_right_joint",
        "f21_stove_main_group_knob_rear_center_joint",
        "f21_stove_main_group_knob_rear_left_joint",
        "f21_stove_main_group_knob_rear_right_joint",
        "f25_cab_main_main_group_leftdoorhinge",
        "f25_cab_main_main_group_rightdoorhinge",
        "f26_cab_1_main_group_leftdoorhinge",
        "f26_cab_1_main_group_rightdoorhinge",
        "f32_stack_1_main_group_1_leftdoorhinge",
        "f32_stack_1_main_group_1_rightdoorhinge",
        "f33_stack_1_main_group_2_slidejoint",
        "f34_stack_1_main_group_3_slidejoint",
        "f36_stack_2_main_group_1_leftdoorhinge",
        "f36_stack_2_main_group_1_rightdoorhinge",
        "f37_stack_2_main_group_2_slidejoint",
        "f38_stack_2_main_group_3_slidejoint",
        "f41_stack_3_main_group_1_leftdoorhinge",
        "f41_stack_3_main_group_1_rightdoorhinge",
        "f42_stack_3_main_group_2_slidejoint",
        "f43_stack_3_main_group_3_slidejoint",
        "f45_stack_4_main_group_1_slidejoint",
        "f46_stack_4_main_group_2_slidejoint",
        "f47_stack_4_main_group_3_slidejoint",
        "f48_stack_4_main_group_4_slidejoint",
        "f55_fridge_top_left_group_leftdoorhinge",
        "f55_fridge_top_left_group_rightdoorhinge",
        "f57_stack_left_left_group_1_doorhinge",
        "f58_stack_left_left_group_2_doorhinge",
        "f60_stack_right_left_group_1_doorhinge",
        "f61_stack_right_left_group_2_doorhinge",
        "f66_sink_right_group_handle_joint",
        "f66_sink_right_group_handle_temp_joint",
        "f66_sink_right_group_spout_joint",
        "f70_stack_1_right_group_1_leftdoorhinge",
        "f70_stack_1_right_group_1_rightdoorhinge",
        "f74_stack_2_right_group_1_doorhinge",
        "f75_cab_1_right_group_leftdoorhinge",
        "f75_cab_1_right_group_rightdoorhinge",
        "f76_cab_2_right_group_doorhinge",
        "f81_stack_1_front_group_1_leftdoorhinge",
        "f81_stack_1_front_group_1_rightdoorhinge",
        "f82_stack_1_front_group_2_slidejoint",
        "f85_bottom_front_group_1_slidejoint",
        "f86_bottom_front_group_2_slidejoint",
        "f87_cab_middle_front_group_leftdoorhinge",
        "f87_cab_middle_front_group_rightdoorhinge",
        "f88_microwave_front_group_microjoint",
        "f90_top_front_group_leftdoorhinge",
        "f90_top_front_group_rightdoorhinge",
    ]


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaWraparoundIndustrial(_RobocasaWraparoundOptions):
    file: str = "wraparound__industrial/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaWraparoundScandinavian(_RobocasaWraparoundOptions):
    file: str = "wraparound__scandanavian/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaWraparoundCoastal(_RobocasaWraparoundOptions):
    file: str = "wraparound__coastal/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_wraparound_modern_1")
class RobocasaWraparoundModern1(_RobocasaWraparoundOptions):
    file: str = "wraparound__modern_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_wraparound_modern_2")
class RobocasaWraparoundModern2(_RobocasaWraparoundOptions):
    file: str = "wraparound__modern_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_wraparound_traditional_1")
class RobocasaWraparoundTraditional1(_RobocasaWraparoundOptions):
    file: str = "wraparound__traditional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_wraparound_traditional_2")
class RobocasaWraparoundTraditional2(_RobocasaWraparoundOptions):
    file: str = "wraparound__traditional_2/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaWraparoundFarmhouse(_RobocasaWraparoundOptions):
    file: str = "wraparound__farmhouse/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaWraparoundRustic(_RobocasaWraparoundOptions):
    file: str = "wraparound__rustic/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register()
class RobocasaWraparoundMediterranean(_RobocasaWraparoundOptions):
    file: str = "wraparound__mediterranean/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_wraparound_transitional_1")
class RobocasaWraparoundTransitional1(_RobocasaWraparoundOptions):
    file: str = "wraparound__transitional_1/kitchen_scene.xml"


@ROBOCASA_KITCHEN_REGISTRY.register(name="robocasa_wraparound_transitional_2")
class RobocasaWraparoundTransitional2(_RobocasaWraparoundOptions):
    file: str = "wraparound__transitional_2/kitchen_scene.xml"


# ──────────────────────────────────────────────────────────────────────────────
# Accessories
# ──────────────────────────────────────────────────────────────────────────────


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaCoffeeMachineLeftGroup(_RobocasaOptions):
    file: str = "u_shaped_small__coastal/urdf/coffee_machine_left_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaCoffeeMachineMainGroup(_RobocasaOptions):
    file: str = "galley__coastal/urdf/coffee_machine_main_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaCoffeeMachineRightGroup(_RobocasaOptions):
    file: str = "l_shaped_small__coastal/urdf/coffee_machine_right_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaKnifeBlockLeftGroup(_RobocasaOptions):
    file: str = "l_shaped_large__coastal/urdf/knife_block_left_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaKnifeBlockMainGroup(_RobocasaOptions):
    file: str = "galley__coastal/urdf/knife_block_main_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaKnifeBlockRightGroup(_RobocasaOptions):
    file: str = "u_shaped_small__coastal/urdf/knife_block_right_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPaperTowelLeftGroup(_RobocasaOptions):
    file: str = "galley__coastal/urdf/paper_towel_left_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPaperTowelMainGroup(_RobocasaOptions):
    file: str = "one_wall_small__coastal/urdf/paper_towel_main_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPaperTowelRightGroup(_RobocasaOptions):
    file: str = "g_shaped_small__coastal/urdf/paper_towel_right_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPlantFrontGroup(_RobocasaOptions):
    file: str = "g_shaped_small__coastal/urdf/plant_front_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPlantIslandGroup(_RobocasaOptions):
    file: str = "u_shaped_large__coastal/urdf/plant_island_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPlantLeftGroup(_RobocasaOptions):
    file: str = "g_shaped_large__coastal/urdf/plant_left_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaPlantMainGroup(_RobocasaOptions):
    file: str = "l_shaped_small__coastal/urdf/plant_main_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaToasterMainGroup(_RobocasaOptions):
    file: str = "galley__coastal/urdf/toaster_main_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaToasterRightGroup(_RobocasaOptions):
    file: str = "l_shaped_small__coastal/urdf/toaster_right_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaUtensilHolderLeftGroup(_RobocasaOptions):
    file: str = "g_shaped_small__coastal/urdf/utensil_holder_left_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaUtensilHolderMainGroup(_RobocasaOptions):
    file: str = "galley__coastal/urdf/utensil_holder_main_group.urdf"


@ROBOCASA_ACCESSORIES_REGISTRY.register()
class RobocasaUtensilHolderRightGroup(_RobocasaOptions):
    file: str = "u_shaped_small__coastal/urdf/utensil_holder_right_group.urdf"
