"""Visualize the DexYCB hand-object grasping dataset (MANO keypoints + YCB objects).

---
tier: recipe
capabilities: [dataset.dex_ycb]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/dex_ycb.py --data-path /path/to/dexycb
---
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.datasets.constants import MANO_KEYPOINTS_NAME
from eden.datasets.dex_ycb_dataset import DexYCBDataset


# Skeleton edges over ``MANO_KEYPOINTS_NAME`` (which the dataset uses for
# ``hand.links_pos_w``): five chains of wrist → 3 knuckles → 1 tip.
_KP = {n: i for i, n in enumerate(MANO_KEYPOINTS_NAME)}
_FINGER_CHAINS = (
    ("thumb1", "thumb2", "thumb3", "thumb_tip"),
    ("index1", "index2", "index3", "index_tip"),
    ("middle1", "middle2", "middle3", "middle_tip"),
    ("ring1", "ring2", "ring3", "ring_tip"),
    ("pinky1", "pinky2", "pinky3", "pinky_tip"),
)
_HAND_BONES: tuple[tuple[int, int], ...] = tuple(
    (_KP["wrist" if i == 0 else chain[i - 1]], _KP[chain[i]]) for chain in _FINGER_CHAINS for i in range(4)
)


def _ycb_class_for_folder(folder_name: str):
    """Resolve a DexYCB ``ycb_id`` folder name to its ``YCBOptions``
    subclass by matching on the declared ``file`` field default.
    """
    import inspect

    from eden.options.objects import ycb as ycb_module
    from eden.options.objects.ycb import YCBOptions

    needle = f"models/{folder_name}/"
    for _, cls in inspect.getmembers(ycb_module, inspect.isclass):
        if cls is YCBOptions or not issubclass(cls, YCBOptions):
            continue
        # Pydantic class-level access returns a FieldInfo; read the
        # declared default explicitly.
        field = cls.model_fields.get("file")
        if field is not None and needle in (field.default or ""):
            return cls
    raise KeyError(
        f"No YCBOptions subclass found for DexYCB object '{folder_name}'. "
        "Check `eden/options/objects/ycb.py` — every DexYCB id should resolve."
    )


def show_demo(dataset: DexYCBDataset, traj_idx: int = 0, show_viewer: bool = False) -> None:
    """Replay a single DexYCB clip with the YCB mesh and MANO keypoints."""
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    object_name = dataset.metadata["objects"][traj_idx]
    side = dataset.metadata["sides"][traj_idx]
    object_cls = _ycb_class_for_folder(object_name)

    # Camera framed for the action cluster in AprilTag world coordinates
    # (~0.5–0.7 m, 0.3–0.6 m, 0.05–0.3 m above the table tag).
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 30.0, decimation=1, show_world_frame=True),
        scene_options=SceneOptions(object=object_cls(), plane=PlaneOptions()),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(0.9, -0.05, 0.45),
                cam_lookat=(0.55, 0.35, 0.15),
                cam_fov=45,
                debug=True,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()

    obj = env.entities["object"]
    hand_seq = dataset.motion_sequences["hand"]
    obj_seq = dataset.motion_sequences["object"]

    cam = env.cameras["camera"]
    cam.start_recording()

    hand_color = (0.0, 0.0, 1.0, 0.8) if side == "right" else (0.0, 1.0, 0.0, 0.8)
    bone_color = (1.0, 0.4, 0.4, 0.8)

    n_frames = int(hand_seq.buffer_length[0].item())
    for i in range(n_frames):
        obj.set_pos(obj_seq.base_pos_w[0, i])
        obj.set_quat(obj_seq.base_quat_w[0, i])

        kp_i = hand_seq.links_pos_w[0, i]  # (21, 3) in MANO_KEYPOINTS_NAME order
        env.draw_debug_spheres(kp_i, radius=0.006, color=hand_color)
        for a, b in _HAND_BONES:
            env.draw_debug_line(kp_i[a], kp_i[b], color=bone_color)

        env.scene.visualizer.update()
        cam.render_rgb()
        env.clear_debug_objects()

    cam.stop_recording()


def run_dex_ycb_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the extracted DexYCB root (the directory holding "
        "`<YYYYMMDD-subject-NN>/<YYYYMMDD_HHMMSS>/pose.npz`).",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Path to DexYCB's `calibration/` directory. Defaults to "
        "`<data-path>/calibration` when present; otherwise the dataset "
        "raises and you'll need to pass this flag explicitly (e.g. when "
        "`--data-path` points at a subject or single-sequence directory).",
    )
    parser.add_argument("--traj-idx", type=int, default=0, help="Index of the trajectory to replay from the dataset")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    if not os.path.exists(args.data_path):
        print(f"Error: Data directory not found at {args.data_path}")
        return

    calibration_dir = args.calibration_dir
    if calibration_dir is None:
        default_calib = os.path.join(args.data_path, "calibration")
        calibration_dir = default_calib if os.path.isdir(default_calib) else None

    dataset = DexYCBDataset(
        options=DexYCBDataset.configure(
            raw_data_path=args.data_path,
            calibration_dir=calibration_dir,
            target_framerate=30.0,
        ),
        device=gs.device,
    )

    print(f"Loaded {len(dataset)} sequences")
    print(f"Framerate: {dataset.framerate}")
    print(f"Buffer lengths (first 5): {dataset.buffer_length[:5]}")
    print(f"Object for seq {args.traj_idx}: {dataset.metadata['objects'][args.traj_idx]}")
    print(f"Side for seq {args.traj_idx}: {dataset.metadata['sides'][args.traj_idx]}")

    dataset.select_trajectories([args.traj_idx])

    print("Starting DexYCB dataset demo...")
    show_demo(dataset, traj_idx=0)


if __name__ == "__main__":
    run_dex_ycb_demo()
