"""Polaris Block-Stack-Kitchen task configuration."""

from __future__ import annotations

from pydantic import Field

from eden.options.managers.metrics import MetricManagerOptions
from eden.options.splat import SplatOptions
from eden.tasks.polaris.base_config import (
    DroidBaseConfig,
    DroidBatchBaseConfig,
    DroidSceneOptions,
    droid_field,
    make_polaris_splat,
    polaris_batch_splat,
)
from eden.options.objects.polaris import (
    PolarisGreenCube,
    PolarisGreenTray,
    PolarisPinkBowl,
    PolarisPurplePlate,
    PolarisTomato,
    PolarisUWCorn,
    PolarisWoodCube,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.polaris.rubric import put_in, rubric, within


@TASK_REGISTRY.register()
class PolarisBlockStackKitchenConfig(DroidBaseConfig):
    splat_options: SplatOptions = make_polaris_splat(
        path="block_stack_kitchen/assets/g60_kitchen_table_zed/splat.ply",
        pos=(-0.225, -0.24, 0.0),
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    scene_options: DroidSceneOptions = DroidSceneOptions(
        green_cube=PolarisGreenCube(
            default_root_pos=(
                0.3208774626255044,
                -0.11194607066881795,
                0.0804820482926792,
            )
        ),
        green_tray=PolarisGreenTray(
            default_root_pos=(
                0.3517783956667713,
                0.09136301147498842,
                0.05371089447294813,
            )
        ),
        pink_bowl=PolarisPinkBowl(
            default_root_pos=(
                0.5336140386925854,
                0.11516009020424095,
                0.06645985800557358,
            )
        ),
        purple_plate=PolarisPurplePlate(
            default_root_pos=(
                0.47936457938369215,
                -0.0038916438062898406,
                0.1738220553802526,
            )
        ),
        tomato=PolarisTomato(
            default_root_pos=(
                0.3722157695699017,
                -0.015744171924207098,
                0.1668820572717904,
            )
        ),
        uw_corn=PolarisUWCorn(
            default_root_pos=(
                0.3024292290210789,
                0.09045923170297229,
                0.1857603951949213,
            )
        ),
        wood_cube=PolarisWoodCube(
            default_root_pos=(
                0.41919292386568763,
                -0.09747371822595585,
                0.07723409933417806,
            )
        ),
    )

    language_instructions: list[str] = ["Place and stack the blocks on top of the green tray"]

    # PolaRiS's DROID-BlockStackKitchen rubric: both cubes into the tray, the green one on the wood one.
    metric_options: MetricManagerOptions = rubric(
        [
            *put_in("green_cube", "green_tray", lift_height=0.03),
            *put_in("wood_cube", "green_tray", lift_height=0.03),
            within("green_cube", "wood_cube", 0.5, after=["green_cube_in_green_tray", "wood_cube_in_green_tray"]),
        ]
    )


@TASK_REGISTRY.register()
class PolarisBlockStackKitchenBatchedConfig(DroidBatchBaseConfig):
    """Batched (Madrona + gsplat) variant of :class:`PolarisBlockStackKitchenConfig`.

    Renders every environment's photorealistic kitchen splat in a single gsplat call and
    composites it under the batched Madrona sim pass. Reuses :class:`PolarisBlockStackKitchenConfig`'s objects
    and splat verbatim; the robot/cameras/actions/renderer come from :class:`DroidBatchBaseConfig`.
    """

    scene_options: DroidSceneOptions = Field(
        default_factory=lambda: droid_field(PolarisBlockStackKitchenConfig, "scene_options")
    )
    splat_options: SplatOptions = Field(default_factory=lambda: polaris_batch_splat(PolarisBlockStackKitchenConfig))
    language_instructions: list[str] = Field(
        default_factory=lambda: droid_field(PolarisBlockStackKitchenConfig, "language_instructions")
    )
    metric_options: MetricManagerOptions = Field(
        default_factory=lambda: droid_field(PolarisBlockStackKitchenConfig, "metric_options")
    )
