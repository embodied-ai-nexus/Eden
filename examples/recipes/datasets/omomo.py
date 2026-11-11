"""Visualize the OMOMO human-object manipulation dataset (SMPL-H skeleton + object).

---
tier: recipe
capabilities: [dataset.omomo]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/omomo.py --data-path /path/to/omomo
---
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.datasets.omomo_dataset import OMOMOHolosomaDataset
from eden.datasets.constants import SMPLH_KEYPOINTS_NAME

# SMPL-H skeleton connectivity for visualization (52 joints)
SKELETON_EDGES = [
    # Spine chain
    ("Pelvis", "Torso"),
    ("Torso", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    # Left leg
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),
    # Right leg
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),
    # Left arm
    ("Chest", "L_Thorax"),
    ("L_Thorax", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    # Right arm
    ("Chest", "R_Thorax"),
    ("R_Thorax", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
    # Left hand fingers
    ("L_Wrist", "L_Index1"),
    ("L_Index1", "L_Index2"),
    ("L_Index2", "L_Index3"),
    ("L_Wrist", "L_Middle1"),
    ("L_Middle1", "L_Middle2"),
    ("L_Middle2", "L_Middle3"),
    ("L_Wrist", "L_Pinky1"),
    ("L_Pinky1", "L_Pinky2"),
    ("L_Pinky2", "L_Pinky3"),
    ("L_Wrist", "L_Ring1"),
    ("L_Ring1", "L_Ring2"),
    ("L_Ring2", "L_Ring3"),
    ("L_Wrist", "L_Thumb1"),
    ("L_Thumb1", "L_Thumb2"),
    ("L_Thumb2", "L_Thumb3"),
    # Right hand fingers
    ("R_Wrist", "R_Index1"),
    ("R_Index1", "R_Index2"),
    ("R_Index2", "R_Index3"),
    ("R_Wrist", "R_Middle1"),
    ("R_Middle1", "R_Middle2"),
    ("R_Middle2", "R_Middle3"),
    ("R_Wrist", "R_Pinky1"),
    ("R_Pinky1", "R_Pinky2"),
    ("R_Pinky2", "R_Pinky3"),
    ("R_Wrist", "R_Ring1"),
    ("R_Ring1", "R_Ring2"),
    ("R_Ring2", "R_Ring3"),
    ("R_Wrist", "R_Thumb1"),
    ("R_Thumb1", "R_Thumb2"),
    ("R_Thumb2", "R_Thumb3"),
]

JOINT_NAMES = list(SMPLH_KEYPOINTS_NAME)
EDGE_INDICES = [(JOINT_NAMES.index(a), JOINT_NAMES.index(b)) for a, b in SKELETON_EDGES]


def show_demo(dataset, traj_idx=0, show_viewer=False) -> None:
    """
    Visualize the OMOMO Holosoma dataset by loading the object mesh and
    drawing skeleton keypoints for the human.
    """
    from eden.options.objects.omomo import OMOMO_OBJECTS_REGISTRY
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.02, decimation=1, show_world_frame=True),
        scene_options=SceneOptions(
            object=OMOMO_OBJECTS_REGISTRY.get(dataset.metadata["objects"][traj_idx])(),
            plane=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(3.0, 3.0, 2.0),
                cam_lookat=(0.0, 0.0, 0.8),
                cam_fov=50,
                debug=True,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()

    obj = env.entities["object"]

    human_seq = dataset.motion_sequences["human"]
    obj_seq = dataset.motion_sequences["object"]

    cam = env.cameras["camera"]
    cam.start_recording()

    human_color = (1.0, 0.2, 0.2, 0.8)
    human_line_color = (1.0, 0.4, 0.4, 0.5)

    for i in range(human_seq.buffer_length[0]):
        human_links = human_seq.links_pos_w[0, i]  # (52, 3)

        # Update object pose
        obj.set_pos(obj_seq.base_pos_w[0, i])
        obj.set_quat(obj_seq.base_quat_w[0, i])

        # Draw human joint spheres
        env.draw_debug_spheres(human_links, radius=0.02, color=human_color)

        # Draw skeleton bones
        for a, b in EDGE_INDICES:
            env.draw_debug_line(human_links[a], human_links[b], color=human_line_color)

        env.scene.visualizer.update()

        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_omomo_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to OMOMO directory containing .pt files",
    )
    parser.add_argument("--traj-idx", type=int, default=0, help="Index of the trajectory to replay from the dataset")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found at {data_path}")
        return

    dataset = OMOMOHolosomaDataset(
        options=OMOMOHolosomaDataset.configure(
            raw_data_path=data_path,
            target_framerate=50.0,
        ),
        device=gs.device,
    )

    print(f"Loaded {len(dataset)} sequences")
    print(f"Framerate: {dataset.framerate}")
    print(f"Buffer lengths (first 5): {dataset.buffer_length[:5]}")
    print(f"Object for seq 0: {dataset.metadata['objects'][0]}")

    dataset.select_trajectories([args.traj_idx])

    print("Starting OMOMO dataset demo...")
    show_demo(dataset, traj_idx=args.traj_idx)


if __name__ == "__main__":
    run_omomo_demo()
