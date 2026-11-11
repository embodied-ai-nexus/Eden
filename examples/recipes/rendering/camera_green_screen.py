"""Green-screen (chroma-key) batch camera rendering over randomized object placement.

---
tier: recipe
capabilities: [render.green_screen, sensor.camera]
hardware: gpu
optional_deps: [madrona]
run: uv run python examples/recipes/rendering/camera_green_screen.py -v
---
"""

from argparse import ArgumentParser

import genesis as gs
from genesis.options.renderers import BatchRenderer

import eden as en
from eden.entities.camera_overlays import BackgroundImageComposite
from eden.envs.base import EnvBase
from eden.options import (
    CameraOptions,
    CamerasOptions,
    EntityOptions,
    EnvOptions,
    EventManagerOptions,
    PlaneOptions,
    SceneOptions,
)
from eden.terms import EventTerm
from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.constants import EventMode
from eden.tasks.simpler.base_config import simpler_asset_path


def main():
    parser = ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    # The gs-madrona BatchRenderer this recipe demonstrates is CUDA-only, so there is no CPU path.
    en.init(backend=gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=4,
            show_FPS=True,
        ),
        scene_options=SceneOptions(
            floor=PlaneOptions(
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, 0.5),
            ),
            table=EntityOptions(
                file="objects/tables/work_table/work_table.obj", is_fixed_base=True, is_support_enabled=True
            ),
            plate=EntityOptions(
                file="objects/plates/86d4d4f8-ab98-4c48-9630-dc5c7710733b.glb",
            ),
        ),
        event_options=EventManagerOptions(
            place_table=EventTerm.configure(
                mode=EventMode.STARTUP,
                func=place_constant_xy,
                params={
                    "entity_name": "table",
                    "support_entity_name": "floor",
                    "pos_xy": (0.0, 0.0),
                },
            ),
            place_plate=EventTerm.configure(
                mode=EventMode.INTERVAL,
                interval_range_s=(1.0, 1.0),
                func=place_uniform,
                params={
                    "entity_name": "plate",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
            ),
        ),
        renderer_options=BatchRenderer(),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(1.0, 0.5, 1.5),
                cam_lookat=(0.0, 0.0, 0.8),
                cam_resolution=(640, 480),
                overlays=[
                    BackgroundImageComposite.configure(
                        image_path=simpler_asset_path("real_inpainting/bridge_real_eval_1.png"),
                        entities=["floor"],
                    )
                ],
            )
        ),
        show_viewer=args.vis,
    )

    env.scene.add_light(
        pos=(0.0, 0.0, 1.5),
        dir=(1.0, 1.0, -2.0),
        directional=True,
        castshadow=True,
        cutoff=45.0,
        intensity=0.5,
    )
    env.scene.add_light(
        pos=(4.0, -4.0, 4.0),
        dir=(-1.0, 1.0, -1.0),
        directional=False,
        castshadow=True,
        cutoff=45.0,
        intensity=0.5,
    )

    env.build()

    env.reset()

    print(env.segmentation_idx_dict)
    print(env.event_manager.available_modes)
    print(env.event_manager.summary())

    print(env.entities["plate"].idx)
    print(env.entities["floor"].idx)
    print(env.entities["table"].idx)

    env.cameras["camera"].start_recording()

    for i in range(500):
        env.step()
        env.cameras["camera"].render_rgb()

    env.cameras["camera"].stop_recording()


if __name__ == "__main__":
    main()
