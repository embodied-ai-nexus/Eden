"""Polaris Organize-Tools task configuration."""

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
    PolarisContainer00,
    PolarisContainer01,
    PolarisScissor,
    PolarisTape,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.polaris.rubric import put_in, rubric


@TASK_REGISTRY.register()
class PolarisOrganizeToolsConfig(DroidBaseConfig):
    splat_options: SplatOptions = make_polaris_splat(
        path="organize_tools/assets/princeton_env_2/splat.ply",
        pos=(0.0, 0.0, 0.01),
        quat=(1.0, 0.0, 0.0, 0.0),
    )
    scene_options: DroidSceneOptions = DroidSceneOptions(
        container_00=PolarisContainer00(
            default_root_pos=(
                0.869080252583182,
                -0.20250619640083167,
                0.14476258666169595,
            )
        ),
        container_01=PolarisContainer01(
            default_root_pos=(
                0.605066204255155,
                -0.2104109425515579,
                0.0830626850093904,
            )
        ),
        scissor=PolarisScissor(
            default_root_pos=(
                0.47367935020492635,
                0.008480527112721958,
                0.20958048058047396,
            )
        ),
        tape=PolarisTape(
            default_root_pos=(
                0.840408118120672,
                -0.1766861334390354,
                0.22018023809242637,
            )
        ),
    )

    language_instructions: list[str] = ["put the scissor into the large container"]

    # PolaRiS's DROID-OrganizeTools rubric names container_01, but here that is the 14 cm bucket, which a 23 cm
    # scissor cannot lie 80 % inside; the instruction's "large container" is container_00 (40 x 26 cm).
    metric_options: MetricManagerOptions = rubric(put_in("scissor", "container_00", lift_height=0.04))


@TASK_REGISTRY.register()
class PolarisOrganizeToolsBatchedConfig(DroidBatchBaseConfig):
    """Batched (Madrona + gsplat) variant of :class:`PolarisOrganizeToolsConfig`."""

    scene_options: DroidSceneOptions = Field(
        default_factory=lambda: droid_field(PolarisOrganizeToolsConfig, "scene_options")
    )
    splat_options: SplatOptions = Field(default_factory=lambda: polaris_batch_splat(PolarisOrganizeToolsConfig))
    language_instructions: list[str] = Field(
        default_factory=lambda: droid_field(PolarisOrganizeToolsConfig, "language_instructions")
    )
    metric_options: MetricManagerOptions = Field(
        default_factory=lambda: droid_field(PolarisOrganizeToolsConfig, "metric_options")
    )
