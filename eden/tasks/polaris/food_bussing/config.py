"""Polaris Food-Bussing task configuration."""

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
    PolarisBattery,
    PolarisBlueCup,
    PolarisGrapes,
    PolarisIceCream,
    PolarisYellowBowl,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.polaris.rubric import put_in, rubric


@TASK_REGISTRY.register()
class PolarisFoodBussingConfig(DroidBaseConfig):
    splat_options: SplatOptions = make_polaris_splat(
        path="food_bussing/assets/g60_corner_charuco_static/splat.ply",
        pos=(0.52, 0.6, 0.42),
        quat=(0.5, -0.5, -0.5, -0.5),
    )
    scene_options: DroidSceneOptions = DroidSceneOptions(
        battery=PolarisBattery(
            default_root_pos=(
                0.385111838579175,
                0.05145630825998462,
                0.009920946206443461,
            )
        ),
        blue_cup=PolarisBlueCup(
            default_root_pos=(
                0.5236179232597317,
                0.05274643234823769,
                0.01804890737800555,
            )
        ),
        grapes=PolarisGrapes(
            default_root_pos=(
                0.4995833039283832,
                -0.050881497631258196,
                0.04123384405184905,
            )
        ),
        ice_cream=PolarisIceCream(
            default_root_pos=(
                0.43022724986076344,
                0.15205697715282424,
                0.02001154885077262,
            )
        ),
        yellow_bowl=PolarisYellowBowl(
            default_root_pos=(
                0.42552608251572016,
                -0.21786836577456603,
                0.014558143884765229,
            )
        ),
    )

    language_instructions: list[str] = ["Put all the foods in the bowl"]

    # PolaRiS's DROID-FoodBussing rubric scores the ice cream and the grapes, not the battery or the cup.
    metric_options: MetricManagerOptions = rubric(
        [
            *put_in("ice_cream", "yellow_bowl", lift_height=0.06),
            *put_in("grapes", "yellow_bowl", lift_height=0.06),
        ]
    )


@TASK_REGISTRY.register()
class PolarisFoodBussingBatchedConfig(DroidBatchBaseConfig):
    """Batched (Madrona + gsplat) variant of :class:`PolarisFoodBussingConfig`."""

    scene_options: DroidSceneOptions = Field(
        default_factory=lambda: droid_field(PolarisFoodBussingConfig, "scene_options")
    )
    splat_options: SplatOptions = Field(default_factory=lambda: polaris_batch_splat(PolarisFoodBussingConfig))
    language_instructions: list[str] = Field(
        default_factory=lambda: droid_field(PolarisFoodBussingConfig, "language_instructions")
    )
    metric_options: MetricManagerOptions = Field(
        default_factory=lambda: droid_field(PolarisFoodBussingConfig, "metric_options")
    )
