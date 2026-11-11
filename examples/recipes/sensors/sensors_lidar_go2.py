"""LiDAR / DepthCamera sensor on a Go2 robot surrounded by obstacle rings.

---
tier: recipe
capabilities: [sensor.lidar, sensor.depth_camera, robot.go2]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/sensors/sensors_lidar_go2.py -v
---

Demonstrates how to set up a LiDAR (or DepthCamera) sensor attached to a Go2
robot surrounded by obstacle rings, using the Eden pipeline.

Adapted from Genesis example:
    https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/sensors/lidar_teleop.py

By default the sensor runs in *distance-only* mode (Genesis ``return_points=False``,
genesis-world#2908): it measures only per-ray distances and drops the point cloud, cutting
the sensor's memory footprint and per-step cost to about a quarter — the common case for a
depth camera. Pass ``--keep-points`` to also get the xyz hit points (``read().points``).

Usage:
    python lidar_go2.py -v                     # visualize with default spherical pattern
    python lidar_go2.py -v --pattern grid      # grid lidar pattern
    python lidar_go2.py -v --pattern depth     # depth camera pattern
    python lidar_go2.py -v --keep-points       # also return the xyz hit points
    python lidar_go2.py -c -s 50              # CPU mode, 50 steps
"""

from argparse import ArgumentParser

import numpy as np

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import BoxOptions, CylinderOptions, EnvOptions, PlaneOptions, PrimitiveOptions, SceneOptions
from eden.options.sensors import SensorOptions, SensorsOptions

# Obstacle layout
NUM_CYLINDERS = 8
NUM_BOXES = 6
CYLINDER_RING_RADIUS = 3.0
BOX_RING_RADIUS = 5.0


def build_obstacle_entities(vis: bool = False) -> dict[str, PrimitiveOptions]:
    """Create cylinder and box obstacles arranged in rings around the origin."""
    obstacles: dict[str, PrimitiveOptions] = {}

    for i in range(NUM_CYLINDERS):
        angle = 2 * np.pi * i / NUM_CYLINDERS
        x = CYLINDER_RING_RADIUS * np.cos(angle)
        y = CYLINDER_RING_RADIUS * np.sin(angle)
        obstacles[f"cylinder_{i}"] = CylinderOptions(
            height=1.5,
            radius=0.3,
            fixed=True,
            default_root_pos=(x, y, 0.75),
        )

    for i in range(NUM_BOXES):
        angle = 2 * np.pi * i / NUM_BOXES + np.pi / 6
        x = BOX_RING_RADIUS * np.cos(angle)
        y = BOX_RING_RADIUS * np.sin(angle)
        obstacles[f"box_{i}"] = BoxOptions(
            size=(0.5, 0.5, 2.0 * (i + 1) / NUM_BOXES),
            fixed=False,
            default_root_pos=(x, y, 1.0),
        )

    return obstacles


def build_sensor(pattern: str, vis: bool = False, distance_only: bool = True) -> gs.sensors.SensorOptions:
    """Build a Genesis sensor config for the chosen pattern type.

    ``distance_only`` (default True) maps to Genesis ``return_points=False``: the sensor measures
    only per-ray distances and ``read().points`` is None, cutting its memory/step cost ~4x
    (genesis-world#2908). Set it False (``--keep-points``) when the xyz hit points are needed.
    """
    common = dict(
        pos_offset=(0.3, 0.0, 0.1),
        euler_offset=(0.0, 0.0, 0.0),
        return_world_frame=True,
        return_points=not distance_only,
        draw_debug=vis,
    )

    if pattern == "depth":
        return gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), **common)
    elif pattern == "grid":
        return gs.sensors.Lidar(pattern=gs.sensors.GridPattern(), **common)
    else:
        return gs.sensors.Lidar(pattern=gs.sensors.SphericalPattern(), **common)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument(
        "--pattern",
        type=str,
        default="spherical",
        choices=("spherical", "grid", "depth"),
        help="Sensor pattern type",
    )
    parser.add_argument(
        "--keep-points",
        action="store_true",
        help="Also return xyz hit points (disable distance-only mode)",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    obstacles = build_obstacle_entities(vis=args.vis)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=4,
            env_spacing=(12.0, 12.0),
            enable_default_keybinds=False,
            show_FPS=False,
        ),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            robot=en.robots.Go2(is_fixed_base=True),
            **obstacles,
        ),
        sensors_options=SensorsOptions(
            lidar=SensorOptions(
                sensor=build_sensor(args.pattern, vis=args.vis, distance_only=not args.keep_points),
                attach_entity_name="robot",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    step = 0
    while step < args.steps or args.vis:
        env.step()
        step += 1
        if en.is_smoke() and step >= 5:
            break

        # Raycaster sensors return a (points, distances) NamedTuple; points is None in distance-only mode.
        data = env.sensors["lidar"].read()
        points_shape = None if data.points is None else tuple(data.points.shape)
        print(f"step {step}: distances shape={tuple(data.distances.shape)}, points={points_shape}")


if __name__ == "__main__":
    main()
