"""Visualize the LAFAN1-retargeted G1 dataset by replaying a clip on a G1.

---
tier: recipe
capabilities: [dataset.lafan1, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/lafan1.py
---
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.datasets.lafan1_dataset import LAFAN1RetargetedG1Dataset


def show_demo(dataset, show_viewer=False) -> None:
    """Show the demo of the LAFAN1 retargeting dataset."""
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.02, decimation=1, show_world_frame=True),
        scene_options=SceneOptions(
            robot=en.robots.G1Dofs29(),
            plane=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(2.0, 2.0, 1.0),
                cam_lookat=(0.0, 0.0, 0.5),
                cam_fov=40,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()

    robot = env.entities["robot"]

    robot_seq = dataset.motion_sequences["g1"]

    robot.set_pos(robot_seq.base_pos_w[0, 0])
    robot.set_quat(robot_seq.base_quat_w[0, 0])
    robot.set_dofs_pos(robot_seq.dofs_pos[0, 0])

    cam = env.cameras["camera"]
    cam.start_recording()

    for i in range(robot_seq.buffer_length[0]):
        robot.set_pos(robot_seq.base_pos_w[0, i])
        robot.set_quat(robot_seq.base_quat_w[0, i])
        robot.set_dofs_pos(robot_seq.dofs_pos[0, i])

        env.draw_debug_spheres(robot_seq.links_pos_w[0, i], radius=0.05, color=(1.0, 0.0, 0.0, 0.5))

        env.scene.visualizer.update()

        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_lafan1_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="eden/assets/lafan1_retargeting/g1/dance1_subject1.csv",
        help="Path to the dataset on disk",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    # Path to the processed LAFAN1 data file
    data_path = args.data_path

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Initialize the dataset
    dataset = LAFAN1RetargetedG1Dataset(
        options=LAFAN1RetargetedG1Dataset.configure(
            raw_data_path=data_path,
            target_framerate=50.0,  # LAFAN1 default is usually 30
        ),
        device=gs.device,
    )

    # Run the demo
    print("Starting LAFAN1 dataset demo...")
    show_demo(dataset)


if __name__ == "__main__":
    run_lafan1_demo()
