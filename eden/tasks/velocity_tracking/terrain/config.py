"""Rough-terrain velocity-tracking task configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import eden as en
from eden.utils.configs import (
    EnvOptions,
    SceneOptions,
    ObservationManagerOptions,
    TerminationManagerOptions,
)
from eden.options import ObservationGroupOptions
from eden.options.entities import TerrainOptions
from eden.terms import DoneTerm
from eden.managers.terms.observations.locomotion import GridScanDots
from eden.managers.modifiers.observations import UniformNoise
from eden.managers import TERMINATION_TERM_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.velocity_tracking.base_config import (
    VelocityTrackingBaseConfig,
    VelocityTrackingObsMod,
    VelocityTrackingRobotMod,
    _CRITIC_OBSERVATIONS,
    _POLICY_OBSERVATIONS,
    _TERMINATIONS,
)
from eden.utils.terrains import (
    FlatTerrain,
    PyramidStairsTerrain,
    PyramidSlopedTerrain,
    RandomUniformTerrain,
    WaveTerrain,
)

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity
    from eden.entities.terrain import Terrain


# ---------------------------------------------------------------------------
# Custom termination: out of terrain bounds
# ---------------------------------------------------------------------------


@TERMINATION_TERM_REGISTRY.register()
def out_of_terrain_bounds(
    env: EnvBase,
    *,
    entity_name: str,
    terrain_entity_name: str = "terrain",
    margin: float = 3.0,
) -> torch.Tensor:
    """Terminate when the robot leaves the terrain boundary (with margin).

    Checks that the robot XY position is within the terrain bounds minus a
    safety margin. Matches mjlab's ``out_of_terrain_bounds`` behaviour.
    """
    robot: Entity = env.entities[entity_name]
    terrain: Terrain = env.entities[terrain_entity_name]

    cache_key = f"_terrain_bounds_cache_{terrain_entity_name}"
    if not hasattr(env, cache_key):
        x_min, x_max, y_min, y_max = terrain.bounds
        setattr(
            env,
            cache_key,
            {
                "lo": torch.tensor([x_min + margin, y_min + margin], device=env.device),
                "hi": torch.tensor([x_max - margin, y_max - margin], device=env.device),
            },
        )
    cache = getattr(env, cache_key)

    pos_xy = robot.get_pos()[:, :2]  # (num_envs, 2)
    out_lo = (pos_xy < cache["lo"]).any(dim=1)
    out_hi = (pos_xy > cache["hi"]).any(dim=1)
    return out_lo | out_hi


# ---------------------------------------------------------------------------
# Terrain grid
# ---------------------------------------------------------------------------

_NUM_TERRAIN_LEVELS = 10


def _make_terrain_grid() -> list[list]:
    """Build a 2D terrain generator grid matching mjlab's ROUGH_TERRAINS_CFG.

    Produces ``_NUM_TERRAIN_LEVELS`` rows of increasing difficulty and 10
    columns whose proportions approximate the mjlab composition:
      flat 20% | pyramid_stairs 20% | pyramid_stairs_inv 20% |
      pyramid_slope 10% | pyramid_slope_inv 10% | random_rough 10% | wave 10%
    """
    grid: list[list] = []
    for i in range(_NUM_TERRAIN_LEVELS):
        t = i / (_NUM_TERRAIN_LEVELS - 1) if _NUM_TERRAIN_LEVELS > 1 else 0.0
        sh = round(t * 0.1, 4)
        sl = round(t * 1.0, 4)
        rh = round(0.02 + t * 0.08, 4)
        wa = round(t * 0.2, 4)
        grid.append(
            [
                FlatTerrain.configure(),
                FlatTerrain.configure(),
                PyramidStairsTerrain.configure(step_width=0.3, step_height=sh),
                PyramidStairsTerrain.configure(step_width=0.3, step_height=sh),
                PyramidStairsTerrain.configure(step_width=0.3, step_height=-sh),
                PyramidStairsTerrain.configure(step_width=0.3, step_height=-sh),
                PyramidSlopedTerrain.configure(slope=sl),
                PyramidSlopedTerrain.configure(slope=-sl),
                RandomUniformTerrain.configure(min_height=-rh, max_height=rh, step=0.02, downsampled_scale=0.5),
                WaveTerrain.configure(num_waves=4.0, amplitude=wa),
            ]
        )
    return grid


# ---------------------------------------------------------------------------
# Terrain-specific observations (base + height_scan)
# ---------------------------------------------------------------------------

_HEIGHT_SCAN = {
    "height_scan": GridScanDots.configure(
        entity_name="robot",
        terrain_entity_name="terrain",
        noise=UniformNoise.configure(n_min=-0.1, n_max=0.1),
        scale=1.0 / 5.0,
    ),
}

_TERRAIN_POLICY_OBSERVATIONS = {**_POLICY_OBSERVATIONS, **_HEIGHT_SCAN}
_TERRAIN_CRITIC_OBSERVATIONS = {**_CRITIC_OBSERVATIONS, **_HEIGHT_SCAN}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_NUM_TERRAIN_COLS = 10
_SUBTERRAIN_SIZE = (8.0, 8.0)
_TERRAIN_SCENE = SceneOptions(
    terrain=TerrainOptions(
        subterrain_size=_SUBTERRAIN_SIZE,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        terrain_generators=_make_terrain_grid(),
        default_root_pos=(
            -_NUM_TERRAIN_LEVELS * _SUBTERRAIN_SIZE[0] / 2.0,
            -_NUM_TERRAIN_COLS * _SUBTERRAIN_SIZE[1] / 2.0,
            0.0,
        ),
    ),
    robot=en.robots.G1Dofs29(),
)


@TASK_REGISTRY.register(modifiers=(VelocityTrackingRobotMod, VelocityTrackingObsMod))
class VelocityTrackingTerrainConfig(VelocityTrackingBaseConfig):
    """Velocity tracking for a bipedal humanoid on rough terrain.

    Robot is selected by :class:`VelocityTrackingRobotMod` (``--robot g1``
    default, ``--robot oli``); the policy observation recipe by
    :class:`VelocityTrackingObsMod` (``--obs_preset proprioceptive`` default,
    ``--obs_preset privileged``).

    Extends the base velocity tracking config with:

    - Procedurally generated rough terrain (stairs, slopes, random bumps,
      waves) matching mjlab's ``ROUGH_TERRAINS_CFG`` composition.
    - ``height_scan`` observation via ``GridScanDots`` (terrain
      ``get_height()`` with 17x11 grid, matching mjlab's 1.6x1.0 m RayCast
      at 0.1 m resolution).
    - ``out_of_terrain_bounds`` termination to end episodes when the robot
      wanders beyond the terrain boundary.

    Not yet implemented (relative to mjlab rough):

    - Terrain curriculum (difficulty level progression based on walking
      distance).
    - Critic-only obs (``foot_height``, ``foot_air_time``,
      ``foot_contact``, ``foot_contact_forces``) and contact-based rewards
      (``feet_air_time``, ``foot_swing_height``, ``soft_landing``).
    - ``angular_momentum`` reward.
    """

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,
        max_collision_pairs=128,
        use_gjk_collision=False,
        batch_dofs_info=False,
        # Inherited from the base config's event_options, but not its env_options: this
        # variant replaces that whole block, so the flag RandomizeComShift needs has to be
        # restated here or the term raises at build.
        batch_links_info=True,
        enable_multi_contact=False,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # 5 frames of policy observations, as in the base config -- here that stacks the
        # 187-dot height scan too ((96 + 187) * 5 = 1415 on G1), which is the price of one
        # history mechanism rather than two.
        policy=ObservationGroupOptions(**_TERRAIN_POLICY_OBSERVATIONS, enable_corruption=True, history_length=5),
        critic=ObservationGroupOptions(**_TERRAIN_CRITIC_OBSERVATIONS, enable_corruption=False),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        **_TERMINATIONS,
        out_of_terrain_bounds=DoneTerm.configure(
            func=out_of_terrain_bounds,
            time_out=True,
            params={
                "entity_name": "robot",
                "terrain_entity_name": "terrain",
                "margin": 3.0,
            },
        ),
    )

    scene_options: SceneOptions = _TERRAIN_SCENE
    # Same runner as the base, only the experiment (log dir) name differs; the robot mod
    # prefixes the robot key -> "g1_velocity_tracking_terrain". Shallow copy is fine:
    # pydantic deep-copies field defaults per instance at construction.
    runner_options: RslRlOnPolicyRunnerOptions = VelocityTrackingBaseConfig.model_fields[
        "runner_options"
    ].default.model_copy(update={"experiment_name": "velocity_tracking_terrain"})
