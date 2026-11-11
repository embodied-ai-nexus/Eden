"""Visualize the ARCTIC bimanual hand-object interaction dataset.

---
tier: recipe
capabilities: [dataset.arctic]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/arctic.py
---
"""

import argparse
import os

import torch
import genesis as gs

import eden as en
from eden.datasets.arctic_dataset import ArcticDataset


def show_demo(dataset, show_viewer=False) -> None:
    """Show a demo of the Arctic dataset."""
    from eden.options.objects.arctic import ARCTIC_OBJECTS_REGISTRY
    from eden.envs.base import EnvBase
    from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.02, decimation=1, show_world_frame=True),
        scene_options=SceneOptions(
            object=ARCTIC_OBJECTS_REGISTRY.get(dataset.metadata["objects"][0])(),
            support=BoxOptions(
                size=(0.2, 0.2, 0.1),
                fixed=True,
            ),
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

    obj = env.entities["object"]
    support = env.entities["support"]

    obj_seq = dataset.motion_sequences["object"]
    left_hand_seq = dataset.motion_sequences["left_hand"]
    right_hand_seq = dataset.motion_sequences["right_hand"]

    obj.set_pos(obj_seq.base_pos_w[0, 0])
    obj.set_quat(obj_seq.base_quat_w[0, 0])
    obj.set_dofs_pos(obj_seq.dofs_pos[0, 0])
    obj_aabb = obj.get_AABB()[0]
    obj_xy_center = (obj_aabb[0, :2] + obj_aabb[1, :2]) / 2
    obj_z_lower = obj_aabb[0, 2:] - 0.05  # NOTE: subtract support halfheight
    support.set_pos(torch.cat([obj_xy_center, obj_z_lower], dim=-1))

    cam = env.cameras["camera"]
    cam.start_recording()

    for i in range(obj_seq.buffer_length[0]):
        obj.set_pos(obj_seq.base_pos_w[0, i])
        obj.set_quat(obj_seq.base_quat_w[0, i])
        obj.set_dofs_pos(obj_seq.dofs_pos[0, i])

        env.draw_debug_sphere(left_hand_seq.base_pos_w[0, i], color=(0.0, 1.0, 0.0, 0.5))
        env.draw_debug_sphere(right_hand_seq.base_pos_w[0, i], color=(0.0, 1.0, 0.0, 0.5))
        env.draw_debug_spheres(left_hand_seq.links_pos_w[0, i], color=(1.0, 0.0, 0.0, 0.5))
        env.draw_debug_spheres(right_hand_seq.links_pos_w[0, i], color=(0.0, 0.0, 1.0, 0.5))

        env.scene.visualizer.update()

        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_arctic_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="eden/assets/arctic/processed/s01/box_use_01.npy",
        help="Path to the dataset on disk",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    # Path to the processed Arctic data file
    data_path = args.data_path

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Initialize the dataset
    dataset = ArcticDataset(
        options=ArcticDataset.configure(
            raw_data_path=data_path,
            target_framerate=50.0,  # Arctic default is usually 30
        ),
        device=gs.device,
    )
    print("batch length of dataset: ", len(dataset))
    print("metadata of dataset: ", dataset.metadata)
    print("all motion files: ", dataset.all_motion_files)

    dataset.select_trajectories([0])
    print("batch length of dataset: ", len(dataset))
    print("metadata of dataset: ", dataset.metadata)
    print("all motion files: ", dataset.all_motion_files)

    # NOTE: select subset of links for left hand
    dataset.select_links({"left_hand": slice(16, 21)})

    # Run the demo
    print("Starting Arctic dataset demo...")
    show_demo(dataset)


if __name__ == "__main__":
    run_arctic_demo()
