"""Polaris Tape-Into-Container task configuration."""

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
    PolarisContainer02,
    PolarisTape00,
    PolarisWrench,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.polaris.rubric import put_in, rubric


@TASK_REGISTRY.register()
class PolarisTapeIntoContainerConfig(DroidBaseConfig):
    splat_options: SplatOptions = make_polaris_splat(
        path="tape_into_container/assets/lab_bench/splat.ply",
        pos=(0.0, 0.0, 0.0),  # TODO: check this
        quat=(1.0, 0.0, 0.0, 0.0),  # TODO: check this
    )
    scene_options: DroidSceneOptions = DroidSceneOptions(
        container_02=PolarisContainer02(
            default_root_pos=(
                0.44215725465121536,
                0.10966270076795323,
                0.09355609544023435,
            )
        ),
        tape_00=PolarisTape00(
            default_root_pos=(
                0.3822483446264099,
                0.304611901718318,
                0.12136276658690792,
            )
        ),
        wrench=PolarisWrench(
            default_root_pos=(
                0.5042372952961356,
                -0.11081426529984865,
                0.12350797493131171,
            )
        ),
    )

    language_instructions: list[str] = ["put the tape into the container"]

    metric_options: MetricManagerOptions = rubric(put_in("tape_00", "container_02", lift_height=0.04))


@TASK_REGISTRY.register()
class PolarisTapeIntoContainerBatchedConfig(DroidBatchBaseConfig):
    """Batched (Madrona + gsplat) variant of :class:`PolarisTapeIntoContainerConfig`."""

    scene_options: DroidSceneOptions = Field(
        default_factory=lambda: droid_field(PolarisTapeIntoContainerConfig, "scene_options")
    )
    splat_options: SplatOptions = Field(default_factory=lambda: polaris_batch_splat(PolarisTapeIntoContainerConfig))
    language_instructions: list[str] = Field(
        default_factory=lambda: droid_field(PolarisTapeIntoContainerConfig, "language_instructions")
    )
    metric_options: MetricManagerOptions = Field(
        default_factory=lambda: droid_field(PolarisTapeIntoContainerConfig, "metric_options")
    )
