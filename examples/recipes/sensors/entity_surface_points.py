"""Sample and track an entity's surface points, visualized through a camera.

---
tier: recipe
capabilities: [sensor.camera]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/sensors/entity_surface_points.py -v
---
"""

from argparse import ArgumentParser
import genesis as gs

import eden as en
from eden.utils.pc import (
    get_entity_surface_points,
    update_entity_surface_points,
)
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.camera import CameraOptions, CamerasOptions


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            apple=EntityOptions(
                file="objects/apples/013_apple/textured.obj",
                is_fixed_base=False,
                default_root_pos=(0.0, 0.0, 0.5),
            ),
            floor=PlaneOptions(
                is_support_enabled=False,
                default_root_pos=(0.0, 0.0, 0.0),
            ),
        ),
        cameras_options=CamerasOptions(
            rec=CameraOptions(
                cam_pos=(1.0, 0.0, 0.8),
                cam_lookat=(0.0, 0.0, 0.0),
            )
        ),
        show_viewer=args.vis,
    )
    env.build()
    env.reset()

    apple = env.entities["apple"]
    camera = env.cameras["rec"]

    points_list = get_entity_surface_points(apple, n_points=1000, radius=0.001)

    if args.record:
        camera.start_recording()
    for _ in range(5 if en.is_smoke() else 250):
        env.scene.clear_debug_objects()
        points = update_entity_surface_points(apple, points_list)
        env.scene.draw_debug_spheres(
            poss=points.view(-1, 3),
            radius=0.005,
            color=(0, 0, 1, 0.8),
        )
        env.step()
        if args.record:
            camera.render_rgb()

    if args.record:
        camera.stop_recording()


if __name__ == "__main__":
    main()
