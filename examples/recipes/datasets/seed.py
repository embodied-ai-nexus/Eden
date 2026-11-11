"""Visualize the SEED-retargeted G1 dataset by replaying clips on a G1.

---
tier: recipe
capabilities: [dataset.seed, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/seed.py
---
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.datasets.seed_dataset import SEEDRetargetedG1Dataset


def show_demo(dataset, show_viewer=False) -> None:
    """Show the demo of the SEED retargeted G1 dataset."""
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


class _SubsetSEEDDataset(SEEDRetargetedG1Dataset):
    """Demo wrapper that loads only the first ``num_motions`` clips.

    SEED ships ~142k motions; reading them all into a single
    BatchMotionSequence here would blow out memory. Slicing the file list
    before parallel-read keeps the example cheap to run.
    """

    def __init__(self, options, device, num_motions: int = 10):
        self._num_motions = num_motions
        super().__init__(options, device)

    def _get_all_motion_files(self, data_path: str, extension: str = ".csv") -> list[str]:
        files = sorted(super()._get_all_motion_files(data_path, extension))
        return files[: self._num_motions]


def run_seed_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="eden/assets/seed/",
        help=(
            "Path to a SEED G1 CSV directory, a single .csv file, or the "
            "directory holding g1.tar.gz (auto-extracted on first use)."
        ),
    )
    parser.add_argument(
        "--num-motions",
        type=int,
        default=10,
        help="Load only the first N motions (full SEED is ~142k clips).",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(debug=True, backend=gs.cpu if args.cpu else gs.gpu)

    if not os.path.exists(args.data_path):
        print(f"Error: Data path not found at {args.data_path}")
        return

    dataset = _SubsetSEEDDataset(
        options=_SubsetSEEDDataset.configure(
            raw_data_path=args.data_path,
            target_framerate=50.0,  # SEED native is 120 fps
        ),
        device=gs.device,
        num_motions=args.num_motions,
    )

    print(f"Starting SEED dataset demo on {len(dataset)} clip(s)...")
    show_demo(dataset)


if __name__ == "__main__":
    run_seed_demo()
