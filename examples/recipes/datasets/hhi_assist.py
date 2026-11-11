"""Visualize the HHI-Assist human-human interaction dataset skeletons.

---
tier: recipe
capabilities: [dataset.hhi_assist]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/hhi_assist.py
---
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.datasets.hhi_assist_dataset import HHIAssistDataset, HHI_JOINT_NAMES


# Skeleton connectivity for drawing limb lines
SKELETON_EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Neck"),
    ("Neck", "Head"),
    ("Spine1", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToeBase"),
]

EDGE_INDICES = [(HHI_JOINT_NAMES.index(a), HHI_JOINT_NAMES.index(b)) for a, b in SKELETON_EDGES]


def show_demo(dataset, show_viewer=False) -> None:
    """
    Visualize the HHI-Assist dataset by drawing skeleton keypoints
    for the caregiver (red) and the care recipient (blue).
    """
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.02, decimation=1, show_world_frame=True),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(3.0, 3.0, 2.0),
                cam_lookat=(0.0, 0.0, 0.8),
                cam_fov=50,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()

    cg_seq = dataset.motion_sequences["caregiver"]
    cr_seq = dataset.motion_sequences["recipient"]

    cam = env.cameras["camera"]
    cam.start_recording()

    cg_color = (1.0, 0.2, 0.2, 0.8)  # red for caregiver
    cr_color = (0.2, 0.4, 1.0, 0.8)  # blue for recipient
    cg_line_color = (1.0, 0.4, 0.4, 0.5)
    cr_line_color = (0.4, 0.6, 1.0, 0.5)

    for i in range(cg_seq.buffer_length[0]):
        cg_links = cg_seq.links_pos_w[0, i]  # (21, 3)
        cr_links = cr_seq.links_pos_w[0, i]  # (21, 3)

        # Draw joint spheres
        env.draw_debug_spheres(cg_links, radius=0.02, color=cg_color)
        env.draw_debug_spheres(cr_links, radius=0.02, color=cr_color)

        # Draw skeleton bones
        for a, b in EDGE_INDICES:
            env.draw_debug_line(cg_links[a], cg_links[b], color=cg_line_color)
            env.draw_debug_line(cr_links[a], cr_links[b], color=cr_line_color)

        env.scene.visualizer.update()

        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_hhi_assist_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="eden/assets/hhi_assist/", help="Path to the dataset on disk")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found at {data_path}")
        return

    dataset = HHIAssistDataset(
        options=HHIAssistDataset.configure(
            raw_data_path=data_path,
            target_framerate=50.0,
        ),
        device=gs.device,
    )

    print(f"Loaded {len(dataset)} interaction pairs")
    print(f"Framerate: {dataset.framerate}")
    print(f"Buffer lengths: {dataset.buffer_length}")

    dataset.select_trajectories([0])

    print("Starting HHI-Assist dataset demo...")
    show_demo(dataset)


if __name__ == "__main__":
    run_hhi_assist_demo()
