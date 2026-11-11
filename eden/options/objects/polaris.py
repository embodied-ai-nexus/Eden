"""Polaris benchmark object options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions, SolParamsType
from eden.options.materials import RigidMaterialOptions

# Stiffen each object's own geoms so the 2F85 pad<->object pinch is near-rigid (Genesis
# averages a contact pair's sol params, so a soft object dilutes the tuned pad). Per-object,
# not a global constraint_timeconst -- the global override ejects mesh objects on contact.
_OBJECT_SOL_PARAMS: SolParamsType = (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)


class PolarisObjectOptions(EntityOptions):
    """Polaris object options.

    The source URDFs are authored at unit scale with ``<mass>1.0</mass>``; loaded at the
    per-object ``scale`` (~0.06–0.28), that mass would collapse by ``scale**3`` to a
    fraction of a gram and get clamped to 10 g, leaving objects near-massless and unstable
    to grasp (the pinch limit-cycles). ``recompute_inertia=True`` recomputes mass/inertia
    from the collision geometry * ``material.rho`` instead, giving a scale-correct mass.
    ``default_geoms_sol_params`` stiffens the object's own contacts for the 2F85 grip
    (see ``_OBJECT_SOL_PARAMS``).
    """

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    dataset: str = "eden_polaris"
    assets_subdir: ClassVar[str] = "polaris"
    recompute_inertia: bool = True
    material: RigidMaterialOptions = RigidMaterialOptions(rho=300.0)
    default_geoms_sol_params: dict[str, SolParamsType] = {".*": _OBJECT_SOL_PARAMS}


class PolarisGreenCube(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/green_cube/mesh.urdf"
    scale: float = 0.06


class PolarisGreenTray(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/green_tray/mesh.urdf"
    scale: float = 0.28


class PolarisPinkBowl(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/pink_bowl/mesh.urdf"
    scale: float = 0.13


class PolarisPurplePlate(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/purple_plate/mesh.urdf"
    scale: float = 0.13


class PolarisTomato(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/tomato/mesh.urdf"
    scale: float = 0.06


class PolarisUWCorn(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/uw_corn/mesh.urdf"
    scale: float = 0.08


class PolarisWoodCube(PolarisObjectOptions):
    file: str = "block_stack_kitchen/assets/wood_cube/mesh.urdf"
    scale: float = 0.06


class PolarisBattery(PolarisObjectOptions):
    file: str = "food_bussing/assets/battery/mesh.urdf"
    scale: float = 0.07


class PolarisBlueCup(PolarisObjectOptions):
    file: str = "food_bussing/assets/blue_cup/mesh.urdf"
    scale: float = 0.12


class PolarisGrapes(PolarisObjectOptions):
    file: str = "food_bussing/assets/grapes/mesh.urdf"
    scale: float = 0.08


class PolarisIceCream(PolarisObjectOptions):
    file: str = "food_bussing/assets/ice_cream/mesh.urdf"
    scale: float = 0.12


class PolarisYellowBowl(PolarisObjectOptions):
    file: str = "food_bussing/assets/yellow_bowl/mesh.urdf"
    scale: float = 0.18


class PolarisCleaner(PolarisObjectOptions):
    file: str = "move_latte_cup/assets/cleaner_eval/mesh.urdf"
    scale: float = 1.0


class PolarisCokeEval(PolarisObjectOptions):
    file: str = "move_latte_cup/assets/coke_eval/mesh.urdf"
    scale: float = 1.0


class PolarisCuttingBoard(PolarisObjectOptions):
    file: str = "move_latte_cup/assets/cuttingboard_eval/mesh.urdf"
    scale: float = 1.0


class PolarisLatteArtCup(PolarisObjectOptions):
    file: str = "move_latte_cup/assets/latteartcup_eval/mesh.urdf"
    scale: float = 1.0


class PolarisContainer00(PolarisObjectOptions):
    file: str = "organize_tools/assets/container_00/mesh.urdf"
    scale: float = 1.0


class PolarisContainer01(PolarisObjectOptions):
    file: str = "organize_tools/assets/container_01/mesh.urdf"
    scale: float = 1.0


class PolarisScissor(PolarisObjectOptions):
    file: str = "organize_tools/assets/scissor/mesh.urdf"
    scale: float = 1.2


class PolarisTape(PolarisObjectOptions):
    file: str = "organize_tools/assets/tape/mesh.urdf"
    scale: float = 1.0


class PolarisBellPepper(PolarisObjectOptions):
    file: str = "pan_clean/assets/bell_pepper/mesh.urdf"
    scale: float = 0.08


class PolarisBlueMug(PolarisObjectOptions):
    file: str = "pan_clean/assets/blue_mug/mesh.urdf"
    scale: float = 0.09


class PolarisCoke(PolarisObjectOptions):
    file: str = "pan_clean/assets/coke/mesh.urdf"
    scale: float = 0.09


class PolarisFryingPan(PolarisObjectOptions):
    file: str = "pan_clean/assets/frying_pan/mesh.urdf"
    scale: float = 0.35


class PolarisKetchup(PolarisObjectOptions):
    file: str = "pan_clean/assets/ketchup/mesh.urdf"
    scale: float = 0.13


class PolarisMustard(PolarisObjectOptions):
    file: str = "pan_clean/assets/mustard/mesh.urdf"
    scale: float = 0.13


class PolarisSponge(PolarisObjectOptions):
    # The mesh itself, not its URDF: PolaRiS scales the sponge non-uniformly (9 x 7 x 2.5 cm, a kitchen
    # sponge), which a URDF morph refuses. Uniform 0.09 made it 9 x 9 cm, too wide for the Robotiq 2F85's
    # 8.5 cm stroke across either side.
    file: str = "pan_clean/assets/sponge/mesh.obj"
    scale: tuple[float, float, float] = (0.09, 0.07, 0.09)


class PolarisSushi(PolarisObjectOptions):
    file: str = "pan_clean/assets/sushi/mesh.urdf"
    scale: float = 0.08


class PolarisContainer02(PolarisObjectOptions):
    file: str = "tape_into_container/assets/container_02/mesh.urdf"
    scale: float = 1.0


class PolarisTape00(PolarisObjectOptions):
    file: str = "tape_into_container/assets/tape_00/mesh.urdf"
    scale: float = 1.0


class PolarisWrench(PolarisObjectOptions):
    file: str = "tape_into_container/assets/wrench/mesh.urdf"
    scale: float = 1.0
