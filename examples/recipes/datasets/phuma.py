"""Visualize the PHUMA-retargeted G1 dataset with the offline Translate modifier.

---
tier: recipe
capabilities: [dataset.phuma, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/phuma.py
---
"""

import argparse

from eden.options import GHOST_DEFAULTS
import genesis as gs

import eden as en
from eden.datasets.modifiers import Translate
from eden.datasets.phuma_dataset import PHUMADataset


def show_demo(dataset, show_viewer=False) -> None:
    """Show the demo of the PHUMA dataset."""
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
            robot=en.robots.G1Dofs29(**GHOST_DEFAULTS),
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

        env.scene.visualizer.update()

        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_phuma_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to an existing local PHUMA data directory (e.g. "
        ".../data/g1/aist/subset_0000). Must already exist. Omit to download "
        "the dataset from HuggingFace to the local cache.",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Lift every loaded trajectory by this many metres to demo the Translate modifier.",
    )
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    data_path = args.data_path

    # Initialize the dataset with an offline Translate modifier that lifts the
    # motion off the floor. Modifiers run once after load/prepare and are not
    # baked into the processed cache.
    dataset = PHUMADataset(
        options=PHUMADataset.configure(
            raw_data_path=data_path,
            modifiers=[
                Translate.configure(offset=(0.0, 0.0, args.z_offset)),
            ],
        ),
        device=gs.device,
    )

    print(dataset.metadata)

    dataset.select_trajectories([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Run the demo
    print("Starting PHUMA dataset demo...")
    show_demo(dataset)


if __name__ == "__main__":
    run_phuma_demo()
