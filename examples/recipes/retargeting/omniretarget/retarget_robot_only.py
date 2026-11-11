"""Robot-only retargeting to a G1 with Eden's OmniRetarget (no interaction object).

---
tier: recipe
capabilities: [retarget.human, robot.g1, dataset.omomo]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/omniretarget/retarget_robot_only.py --data-path /path/to/OMOMO_new --task-name sub3_largebox_003
---

Uses the OMOMO Holosoma dataset (OMOMOHolosomaDataset) to load human motion
and retargets to a G1 robot via OmniRetargeter, ignoring the object.

Usage:
    python examples/recipes/retargeting/omniretarget/retarget_robot_only.py \
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
    preprocess_human_motion,
)
from eden.options import EnvOptions, SceneOptions, CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions
from eden.datasets.constants import SMPLH_KEYPOINTS_NAME

# ===========================================================================
# Constants
# ===========================================================================

SMPLH_JOINTS_MAPPING_G1 = JOINTS_MAP_UNITREE_G1["smplh"]
TOE_NAMES = TOE_NAMES_BY_FORMAT["smplh"]


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

    # all_motion_files stores paths relative to raw_data_path (with dir
    # components), so match on the basename stem rather than exact equality.
    all_files = dataset.metadata["all_motion_files"]
    seq_idx = next(
        (i for i, f in enumerate(all_files) if os.path.splitext(os.path.basename(f))[0] == task_name),
        None,
    )
    if seq_idx is None:
        raise ValueError(f"Task '{task_name}' not found in dataset. Available (first 5): {all_files[:5]}...")
    dataset.select_trajectories([seq_idx])
    en.logger.info(f"Selected sequence '{task_name}'")

    # Extract human joints only (ignore object)
    human_seq = dataset.motion_sequences["human"]
    human_joints = human_seq.links_pos_w[0].cpu().numpy()  # (T, 52, 3)

    en.logger.info(f"Loaded {human_joints.shape[0]} frames, {human_joints.shape[1]} joints")

    # ------------------------------------------------------------------
    # Step 3: Create environment with G1 robot (no object)
    # ------------------------------------------------------------------
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=en.robots.G1Dofs29(),
            floor=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(2.0, 0.0, 1.5),
                cam_lookat=(0.0, 0.0, 1.0),
                cam_fov=60.0,
                follow_entity_name="robot",
                debug=True,
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
            activate_obj_non_penetration=False,
            activate_foot_sticking=True,
            activate_foot_ground_constraint=True,
            foot_ground_z=0.0,
            foot_ground_margin=0.005,
            activate_foot_z_sticking=True,
            foot_z_sticking_tolerance=5e-3,
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
        en.logger.info(f"Truncated to {num_frames} frames")

    sub_name = task_name.split("_")[0]
    human_height = dataset.metadata["human_height"][sub_name]
    robot_height = en.robots.G1Dofs29.metadata["robot_height"]
    smpl_scale = robot_height / human_height
    en.logger.info(f"Scale factor: {smpl_scale:.4f}")

    human_joints = preprocess_human_motion(human_joints, list(SMPLH_KEYPOINTS_NAME), TOE_NAMES, smpl_scale)

    foot_sticking_sequences = extract_foot_sticking_velocity(human_joints, retargeter.demo_joints, TOE_NAMES)
    foot_sticking_sequences[0][TOE_NAMES[0]] = False
    foot_sticking_sequences[0][TOE_NAMES[1]] = False

    # ------------------------------------------------------------------
    # Step 6: Run retargeting (no object)
    # ------------------------------------------------------------------
    en.logger.info(f"Starting retargeting ({human_joints.shape[0]} frames) ...")

    retargeted = retargeter.retarget_motion(
        human_joint_motions=human_joints,
        foot_sticking_sequences=foot_sticking_sequences,
    )

    if save_dir is not None:
        np.save(os.path.join(save_dir, f"{task_name}.npy"), retargeted)

    if rec:
        camera = env.cameras["camera"]
        camera.start_recording()
        for qpos in retargeted:
            robot.set_qpos(qpos)
            env.scene.visualizer.update()
            camera.render_rgb()
        camera.stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniRetarget example: robot-only retargeting with Eden")
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
