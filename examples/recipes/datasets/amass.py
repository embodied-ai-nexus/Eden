"""Visualize the AMASS-retargeted G1 dataset by replaying clips on a G1.

---
tier: recipe
capabilities: [dataset.amass, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/amass.py
---
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.datasets.amass_dataset import AMASSRetargetedG1Dataset


def show_demo(dataset, show_viewer=False) -> None:
    """Show the demo of the AMASS retargeted G1 dataset."""
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=len(dataset),
            sim_dt=0.02,
            decimation=1,
            show_world_frame=True,
            env_spacing=(1.5, 1.5),
        ),
        scene_options=SceneOptions(
            robot=en.robots.G1Dofs29(),
            plane=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(10.0, 10.0, 3.0),
                cam_lookat=(0.0, 0.0, 0.5),
                cam_fov=60,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()

    robot = env.entities["robot"]

    robot_seq = dataset.motion_sequences["g1"]

    robot.set_pos(robot_seq.base_pos_w[:, 0])
    robot.set_quat(robot_seq.base_quat_w[:, 0])
    robot.set_dofs_pos(robot_seq.dofs_pos[:, 0])

    cam = env.cameras["camera"]
    cam.start_recording()

    for i in range(robot_seq.buffer_length.max()):
        base_pos, valid_mask = robot_seq.base_pos_w.with_mask[:, i]
        robot.set_pos(base_pos[valid_mask], envs_idx=valid_mask)
        robot.set_quat(robot_seq.base_quat_w[:, i][valid_mask], envs_idx=valid_mask)
        robot.set_dofs_pos(robot_seq.dofs_pos[:, i][valid_mask], envs_idx=valid_mask)

        env.draw_debug_spheres(robot_seq.links_pos_w[0, i], radius=0.05, color=(1.0, 0.0, 0.0, 0.5))

        env.scene.visualizer.update()

        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_amass_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="eden/assets/amass/g1/CMU/01/", help="Path to the dataset on disk"
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Initialize the dataset
    dataset = AMASSRetargetedG1Dataset(
        options=AMASSRetargetedG1Dataset.configure(
            raw_data_path=data_path,
            target_framerate=50.0,
        ),
        device=gs.device,
    )

    print(f"Loaded {len(dataset)} motion sequences")
    print(f"Framerate: {dataset.framerate}")
    print(f"Buffer lengths: {dataset.buffer_length}")

    # Run the demo
    print("Starting AMASS dataset demo...")
    show_demo(dataset)


if __name__ == "__main__":
    run_amass_demo()
