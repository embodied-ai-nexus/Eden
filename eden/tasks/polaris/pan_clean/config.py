"""Polaris Pan-Clean task configuration."""

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
    PolarisBellPepper,
    PolarisBlueMug,
    PolarisCoke,
    PolarisFryingPan,
    PolarisKetchup,
    PolarisMustard,
    PolarisSponge,
    PolarisSushi,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.polaris.rubric import put_in, rubric


@TASK_REGISTRY.register()
class PolarisPanCleanConfig(DroidBaseConfig):
    splat_options: SplatOptions = make_polaris_splat(
        path="pan_clean/assets/g60_stovetop_zed/splat.ply",
        pos=(-0.45, 0.0, -0.275),
        quat=(0.7071067811865476, 0.0, 0.0, -0.7071067811865476),
    )
    scene_options: DroidSceneOptions = DroidSceneOptions(
        bell_pepper=PolarisBellPepper(
            default_root_pos=(
                0.6868054606124152,
                -0.09376765412594976,
                0.07364526317467109,
            )
        ),
        blue_mug=PolarisBlueMug(
            default_root_pos=(
                0.5183660226463038,
                0.4163992332634181,
                -0.045273515484170024,
            )
        ),
        coke=PolarisCoke(
            default_root_pos=(
                0.6152954439272693,
                -0.09741169061982272,
                0.08228446415535723,
            )
        ),
        frying_pan=PolarisFryingPan(
            default_root_pos=(
                0.5631938515485317,
                0.21159406034871397,
                0.05608738411494982,
            )
        ),
        ketchup=PolarisKetchup(
            default_root_pos=(
                0.807515098100678,
                -0.09009033510583345,
                0.07616884049541071,
            )
        ),
        mustard=PolarisMustard(
            default_root_pos=(
                0.8041243640229532,
                0.014150611333146025,
                0.07775862496797399,
            )
        ),
        sponge=PolarisSponge(
            default_root_pos=(
                0.5294183970681007,
                -0.022660877717386577,
                0.06433805102708207,
            )
        ),
        sushi=PolarisSushi(
            default_root_pos=(
                0.6696102360067572,
                0.010349063118418574,
                0.0753967120161282,
            )
        ),
    )

    language_instructions: list[str] = ["Use the yellow sponge to scrub the blue handle frying pan"]

    # PolaRiS's DROID-PanClean rubric: the sponge let go over the pan. Its lift is "sponge origin above
    # 9 cm", about 2 cm off the table in its scene; 3 cm off the table here.
    metric_options: MetricManagerOptions = rubric(put_in("sponge", "frying_pan", lift_height=0.03))


@TASK_REGISTRY.register()
class PolarisPanCleanBatchedConfig(DroidBatchBaseConfig):
    """Batched (Madrona + gsplat) variant of :class:`PolarisPanCleanConfig`."""

    scene_options: DroidSceneOptions = Field(
        default_factory=lambda: droid_field(PolarisPanCleanConfig, "scene_options")
    )
    splat_options: SplatOptions = Field(default_factory=lambda: polaris_batch_splat(PolarisPanCleanConfig))
    language_instructions: list[str] = Field(
        default_factory=lambda: droid_field(PolarisPanCleanConfig, "language_instructions")
    )
    metric_options: MetricManagerOptions = Field(
        default_factory=lambda: droid_field(PolarisPanCleanConfig, "metric_options")
    )
