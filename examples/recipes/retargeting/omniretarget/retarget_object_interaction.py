"""Object-interaction retargeting to a G1 with Eden's OmniRetarget.

---
tier: recipe
capabilities: [retarget.human, robot.g1, dataset.omomo]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/omniretarget/retarget_object_interaction.py --data-path /path/to/OMOMO_new --task-name sub3_largebox_003
---

Uses the OMOMO Holosoma dataset (OMOMOHolosomaDataset) to load human motion
and object poses, then retargets to a G1 robot via OmniRetargeter.

Usage:
    python examples/recipes/retargeting/omniretarget/retarget_object_interaction.py \
        --data-path /path/to/OMOMO_new \
        --task-name sub3_largebox_003 \
        --num-frames 10
"""

from __future__ import annotations

import argparse
import os

import genesis as gs
import numpy as np

import eden as en
from eden.datasets.omomo_dataset import OMOMOHolosomaDataset
from eden.envs.base import EnvBase
from eden.extensions.retargeting.constants import (
    JOINTS_MAP_UNITREE_G1,
    TOE_NAMES_BY_FORMAT,
)
from eden.extensions.retargeting.omniretarget import OmniRetargeter
from eden.extensions.retargeting.omniretarget.g1_presets import (
    FOOT_STICKING_LINKS,
    FOOT_STICKING_MAP,
    LINK_OFFSETS,
    MANUAL_COST,
    MANUAL_LB,
    MANUAL_UB,
    NOMINAL_TRACKING_DOFS,
    extract_foot_sticking_velocity,
)
from eden.options import EnvOptions, SceneOptions, CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions
from eden.datasets.constants import SMPLH_KEYPOINTS_NAME
from eden.options.objects.omomo import OMOMO_OBJECTS_REGISTRY

# ===========================================================================
# Constants
# ===========================================================================

# Eden G1 link names (from constants.py, differ from MuJoCo XML names)
SMPLH_JOINTS_MAPPING_G1 = JOINTS_MAP_UNITREE_G1["smplh"]
TOE_NAMES = TOE_NAMES_BY_FORMAT["smplh"]


# ===========================================================================
# Data loading / preprocessing helpers
# ===========================================================================


def calculate_scale_factor(task_name: str, height_dict: dict[str, float], robot_height: float) -> float:
    sub_name = task_name.split("_")[0]
    return robot_height / height_dict[sub_name]


def preprocess_motion_data(
    human_joints: np.ndarray,
    demo_joints: list[str],
    toe_names: list[str],
    scale: float,
    obj_pos: np.ndarray,
    mat_height: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Preprocess: height-adjust and scale human/object data.

    Parameters
    ----------
    human_joints : np.ndarray
        (T, J, 3) human joint positions; height-adjusted and scaled in place.
    demo_joints : list[str]
        Joint names indexing the J axis of ``human_joints``.
    toe_names : list[str]
        The two toe joint names used to estimate ground contact height.
    scale : float
        Human-to-robot size ratio applied to positions.
    obj_pos : np.ndarray
        (T, 3) object positions; height-adjusted and scaled in place.
    mat_height : float
        Height of the support mat subtracted when the toes never reach the floor.
    """
    toe_indices = [demo_joints.index(toe_names[0]), demo_joints.index(toe_names[1])]
    z_min = human_joints[:, toe_indices, 2].min()
    if z_min >= mat_height:
        z_min -= mat_height
    human_joints[:, :, 2] -= z_min
    human_joints = human_joints * scale

    obj_pos[:, 2] -= z_min
    obj_pos[:, :2] *= scale
    z0 = obj_pos[0, 2]
    dz_scale = (obj_pos[:, 2] - z0) * scale
    obj_pos[:, 2] = z0 + dz_scale

    vel = np.linalg.norm(np.diff(obj_pos, axis=0), axis=1)
    moving_mask = vel > 0.0025
    if np.any(moving_mask):
        moving_idx = int(np.argmax(moving_mask))
    else:
        # No motion exceeds the threshold; use -1 to indicate "no start frame".
        moving_idx = -1
    return human_joints, obj_pos, moving_idx


# ===========================================================================
# Main pipeline
# ===========================================================================


def main(
    data_path: str,
    task_name: str = "sub3_largebox_003",
    save_dir: str = "results/omniretarget",
    num_frames: int | None = None,
    rec: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Initialize Eden / Genesis
    # ------------------------------------------------------------------
    en.init(backend=gs.cpu)

    # ------------------------------------------------------------------
    # Step 2: Load dataset and select sequence
    # ------------------------------------------------------------------
    dataset = OMOMOHolosomaDataset(
        options=OMOMOHolosomaDataset.configure(
            raw_data_path=data_path,
            target_framerate=30.0,
        ),
        device=gs.device,
    )
    en.logger.info(f"Loaded {len(dataset)} sequences from {data_path}")

    # Find the sequence matching task_name. all_motion_files stores paths
    # relative to raw_data_path (with dir components), so match on the basename
    # stem rather than exact equality.
    all_files = dataset.metadata["all_motion_files"]
    seq_idx = next(
        (i for i, f in enumerate(all_files) if os.path.splitext(os.path.basename(f))[0] == task_name),
        None,
    )
    if seq_idx is None:
        raise ValueError(f"Task '{task_name}' not found in dataset. Available: {all_files[:5]}...")
    dataset.select_trajectories([seq_idx])

    obj_name = dataset.metadata["objects"][seq_idx]
    en.logger.info(f"Selected sequence '{task_name}', object='{obj_name}'")

    # Extract human joints and object poses from dataset
    human_seq = dataset.motion_sequences["human"]
    obj_seq = dataset.motion_sequences["object"]

    human_joints = human_seq.links_pos_w[0].cpu().numpy()  # (T, 52, 3)
    obj_pos = obj_seq.base_pos_w[0].cpu().numpy()  # (T, 3)
    obj_quat = obj_seq.base_quat_w[0].cpu().numpy()  # (T, 4) wxyz

    en.logger.info(f"Loaded {human_joints.shape[0]} frames, {human_joints.shape[1]} joints")

    # ------------------------------------------------------------------
    # Step 3: Create minimal environment with G1 robot
    # ------------------------------------------------------------------
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=en.robots.G1Dofs29(),
            floor=PlaneOptions(),
            **{obj_name: OMOMO_OBJECTS_REGISTRY.get(obj_name)()},
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(2.0, 0.0, 1.5),
                cam_lookat=(0.0, 0.0, 0.75),
                cam_fov=60.0,
                follow_entity_name="robot",
            )
        )
        if rec
        else None,
        show_viewer=False,
    )
    env.build()
    env.reset()

    robot = env.entities["robot"]
    en.logger.info(f"Robot entity: {robot.name}, n_dofs={robot.n_dofs}")

    # ------------------------------------------------------------------
    # Step 4: Create OmniRetargeter
    # ------------------------------------------------------------------
    retargeter = OmniRetargeter(
        env=env,
        options=OmniRetargeter.configure(
            entity_name="robot",
            foot_links=FOOT_STICKING_LINKS,
            demo_joints=list(SMPLH_KEYPOINTS_NAME),
            laplacian_match_links=dict(SMPLH_JOINTS_MAPPING_G1),
            link_offsets=LINK_OFFSETS,
            manual_lb=MANUAL_LB,
            manual_ub=MANUAL_UB,
            manual_cost=MANUAL_COST,
            nominal_tracking_dofs=NOMINAL_TRACKING_DOFS,
            foot_sticking_map=FOOT_STICKING_MAP,
            activate_joint_limits=True,
            activate_obj_non_penetration=True,
            activate_foot_sticking=True,
            activate_foot_ground_constraint=True,
            foot_ground_z=0.0,
            foot_ground_margin=0.005,
            activate_foot_z_sticking=True,
            foot_z_sticking_tolerance=5e-3,
            penetration_tolerance=0.001,
            foot_sticking_tolerance=1e-3,
            step_size=0.2,
            w_nominal_tracking_init=5.0,
        ),
    )
    en.logger.info(f"OmniRetargeter created (nq_opt={retargeter.nq_opt})")

    # ------------------------------------------------------------------
    # Step 5: Preprocess motion data
    # ------------------------------------------------------------------
    if num_frames is not None and num_frames < human_joints.shape[0]:
        human_joints = human_joints[:num_frames]
        obj_pos = obj_pos[:num_frames]
        obj_quat = obj_quat[:num_frames]
        en.logger.info(f"Truncated to {num_frames} frames")

    smpl_scale = calculate_scale_factor(
        task_name, dataset.metadata["human_height"], en.robots.G1Dofs29.metadata["robot_height"]
    )
    en.logger.info(f"Scale factor: {smpl_scale:.4f}")

    human_joints, obj_pos, moving_idx = preprocess_motion_data(
        human_joints, list(SMPLH_KEYPOINTS_NAME), TOE_NAMES, smpl_scale, obj_pos
    )
    en.logger.info(f"Preprocessed: object starts moving at frame {moving_idx}")

    # ------------------------------------------------------------------
    # Step 6: Initialize and convert formats
    # ------------------------------------------------------------------
    # Combine into [pos(3), quat_wxyz(4)] for the retargeter
    object_poses = np.concatenate([obj_pos, obj_quat], axis=1)  # (T, 7)

    foot_sticking_sequences = extract_foot_sticking_velocity(human_joints, retargeter.demo_joints, TOE_NAMES)
    foot_sticking_sequences[0][TOE_NAMES[0]] = False
    foot_sticking_sequences[0][TOE_NAMES[1]] = False

    # ------------------------------------------------------------------
    # Step 7: Run retargeting
    # ------------------------------------------------------------------
    en.logger.info(f"Starting retargeting ({human_joints.shape[0]} frames) ...")

    retargeted = retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_name=obj_name,
        object_poses=object_poses,
        scale_factor=smpl_scale,
        foot_sticking_sequences=foot_sticking_sequences,
    )

    # ------------------------------------------------------------------
    # Step 8: Visualize results
    # ------------------------------------------------------------------
    if rec:
        camera = env.cameras["camera"]
        camera.start_recording()

        for qpos, obs_pose in zip(retargeted, object_poses):
            robot.set_qpos(qpos)
            env.entities[obj_name].set_pos(obs_pose[:3])
            env.entities[obj_name].set_quat(obs_pose[3:])
            env.scene.visualizer.update()
            camera.render_rgb()

        camera.stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniRetarget example: object interaction retargeting with Eden")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to OMOMO directory containing .pt files",
    )
    parser.add_argument("--task-name", default="sub3_largebox_003", help="Name of the clip to retarget")
    parser.add_argument("-o", "--output-dir", default="results/omniretarget", help="Directory to write results to")
    parser.add_argument(
        "--num-frames", type=int, default=None, help="Stop after this many frames (default: the whole clip)"
    )
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    args = parser.parse_args()

    main(args.data_path, args.task_name, args.output_dir, args.num_frames, args.record)
