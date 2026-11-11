"""Procedurally generate and preview Eden's built-in terrain types.

---
tier: recipe
capabilities: [terrain.procedural]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/terrain/terrain_gen.py -v
---
"""

import time
from argparse import ArgumentParser

import genesis as gs
import numpy as np
import torch

import eden as en
from eden.envs.base import RLEnvBase
from eden.managers.terms.observations.locomotion import GridScanDots
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import BoxOptions, TerrainOptions
from eden.options.managers.observations import (
    ObservationGroupOptions,
    ObservationManagerOptions,
)
from eden.utils.terrains import (
    CraterTerrain,
    DiscreteObstaclesTerrain,
    FractalTerrain,
    GapTerrain,
    NarrowBeamsTerrain,
    NestedRingsTerrain,
    PerlinNoiseTerrain,
    PitTerrain,
    PyramidSlopedTerrain,
    PyramidStairsTerrain,
    RandomBoxesTerrain,
    RandomGridTerrain,
    RandomUniformTerrain,
    RidgeTerrain,
    SlopedTerrain,
    StairsTerrain,
    SteppingStonesTerrain,
    TiltedGridTerrain,
    WaveTerrain,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")

    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            enable_default_keybinds=False,
        ),
        scene_options=SceneOptions(
            terrain=TerrainOptions(
                default_root_pos=(-3.0, -3.0, 0.0),
                # Small tiles on purpose: the probe circle below (center (15, 15), r=12 m) has
                # to sweep several of them to preview more than one terrain type. Pinned rather
                # than inherited because the default is sized for parkour courses (~29 m).
                subterrain_size=(12.0, 12.0),
                terrain_generators=[
                    [
                        SteppingStonesTerrain.configure(),
                        FractalTerrain.configure(),
                        RandomUniformTerrain.configure(),
                        PerlinNoiseTerrain.configure(),
                        NarrowBeamsTerrain.configure(),
                    ],
                    [
                        SlopedTerrain.configure(),
                        PyramidSlopedTerrain.configure(),
                        DiscreteObstaclesTerrain.configure(),
                        GapTerrain.configure(),
                        RandomGridTerrain.configure(),
                    ],
                    [
                        WaveTerrain.configure(),
                        StairsTerrain.configure(),
                        PyramidStairsTerrain.configure(),
                        PitTerrain.configure(),
                        RandomBoxesTerrain.configure(),
                    ],
                    [
                        RidgeTerrain.configure(),
                        CraterTerrain.configure(),
                        PerlinNoiseTerrain.configure(octaves=4, amplitude=0.5),
                        GapTerrain.configure(gap_width=0.5, num_gaps=6),
                        TiltedGridTerrain.configure(),
                    ],
                    [
                        NestedRingsTerrain.configure(),
                        NarrowBeamsTerrain.configure(num_beams=8, beam_width=0.5),
                        RandomBoxesTerrain.configure(num_boxes=20, box_height_range=(0.1, 0.8)),
                        TiltedGridTerrain.configure(tilt_range=25.0),
                        RandomGridTerrain.configure(grid_width=0.3, grid_height_range=(0.0, 0.5)),
                    ],
                ],
            ),
            box=BoxOptions(
                # Not fixed-base: the loop repositions this marker each step with
                # set_pos() to probe terrain height, which a fixed-base entity rejects.
                size=(0.04, 0.04, 0.04),
                fixed=False,
                default_root_pos=(0.0, 0.0, 0.0),
            ),
        ),
        observation_options=ObservationManagerOptions(
            obs_prop=ObservationGroupOptions(
                scan_dots=GridScanDots.configure(
                    entity_name="box",
                    terrain_entity_name="terrain",
                    offset_from_base=False,
                )
            )
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(1.0, 0.5, 3.5),
                cam_lookat=(0.0, 0.0, 0.8),
                cam_resolution=(640, 480),
            )
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    box = env.entities["box"]
    scan_dots = env.observation_manager.get_term("scan_dots")

    center = np.array([15.0, 15.0, 1.0])
    r = 12.0
    resolution = 360

    i = 0
    while True:
        env.clear_debug_objects()
        target_pos = center + np.array([np.cos(i / resolution * np.pi), np.sin(i / resolution * np.pi), 0]) * r
        box.set_pos(target_pos)
        obs, _, _, _, _ = env.step()

        env.draw_debug_spheres(
            torch.cat(
                [
                    scan_dots.grid.view(env.num_envs, -1, 2),
                    obs["obs_prop"].unsqueeze(-1),
                ],
                dim=-1,
            ).view(-1, 3),
            radius=0.005,
            color=(0, 0, 1, 0.8),
        )
        i += 1
        if en.is_smoke() and i >= 5:
            break
        time.sleep(0.01)


if __name__ == "__main__":
    main()
