"""Export a Go1 trajectory as an interactive, standalone HTML replay.

---
tier: recipe
capabilities: [render.viser, robot.go1]
hardware: cpu
optional_deps: [viser]
run: uv run python examples/recipes/rendering/viser_recording.py -c
---
"""

import argparse
import math
from pathlib import Path

import eden as en  # isort: skip
import genesis as gs
from eden.envs.base import EnvBase
from eden.extensions.visualization.viser import ViserViewer
from eden.options import EnvOptions, PlaneOptions, SceneOptions
from eden.utils.configs import EdenConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--cpu", action="store_true", help="Run simulation on CPU.")
    parser.add_argument("-s", "--steps", type=int, default=200, help="Number of simulation steps to record.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("go1.html"), help="Replay path ending in .html or .viser."
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=0.005, decimation=4),
        scene_options=SceneOptions(plane=PlaneOptions(), robot=en.robots.Go1()),
    )
    env = EnvBase.from_config(config, show_viewer=False)
    env.build()
    env.reset()
    robot = env.entities["robot"]
    viewer = ViserViewer(env, enable_gui=False)
    viewer.server.initial_camera.position = (1.2, -1.2, 0.8)
    viewer.server.initial_camera.look_at = (0.0, 0.0, 0.3)
    try:
        viewer.start_recording(args.output)
        for step in range(5 if en.is_smoke() else args.steps):
            robot.control_dofs_pos(robot.default_dofs_pos * (1 + 0.15 * math.sin(2 * math.pi * step * env.dt)))
            env.step()
            viewer.update()
        print(f"Saved interactive replay to {viewer.stop_recording()}")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
