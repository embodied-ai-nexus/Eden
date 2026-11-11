"""Polaris Move-Latte-Cup task configuration."""

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
    PolarisCleaner,
    PolarisCokeEval,
    PolarisCuttingBoard,
    PolarisLatteArtCup,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.polaris.rubric import put_in, rubric


@TASK_REGISTRY.register()
class PolarisMoveLatteCupConfig(DroidBaseConfig):
    splat_options: SplatOptions = make_polaris_splat(
        path="move_latte_cup/assets/princeton_env_1/splat.ply",
        scale=0.98,
        pos=(0.087, 0.175, -0.01),
        quat=(1.0, 0.0, 0.0, 0.0),
    )
    scene_options: DroidSceneOptions = DroidSceneOptions(
        cleaner=PolarisCleaner(
            default_root_pos=(
                0.6732833898725151,
                -0.029656703945114327,
                0.1115361781066837,
            )
        ),
        coke=PolarisCokeEval(
            default_root_pos=(
                0.6801107133255605,
                -0.10206733249862565,
                0.12591785111461729,
            )
        ),
        cutting_board=PolarisCuttingBoard(
            default_root_pos=(
                0.45471049958577026,
                -0.180505348408468,
                0.14329105966329259,
            )
        ),
        latte_art_cup=PolarisLatteArtCup(
            default_root_pos=(
                0.6542510275715706,
                -0.30076297869788315,
                0.12330198101848916,
            )
        ),
    )

    language_instructions: list[str] = ["put the latte art cup on top of the cutting board"]

    metric_options: MetricManagerOptions = rubric(put_in("latte_art_cup", "cutting_board", lift_height=0.04))


@TASK_REGISTRY.register()
class PolarisMoveLatteCupBatchedConfig(DroidBatchBaseConfig):
    """Batched (Madrona + gsplat) variant of :class:`PolarisMoveLatteCupConfig`."""

    scene_options: DroidSceneOptions = Field(
        default_factory=lambda: droid_field(PolarisMoveLatteCupConfig, "scene_options")
    )
    splat_options: SplatOptions = Field(default_factory=lambda: polaris_batch_splat(PolarisMoveLatteCupConfig))
    language_instructions: list[str] = Field(
        default_factory=lambda: droid_field(PolarisMoveLatteCupConfig, "language_instructions")
    )
    metric_options: MetricManagerOptions = Field(
        default_factory=lambda: droid_field(PolarisMoveLatteCupConfig, "metric_options")
    )
