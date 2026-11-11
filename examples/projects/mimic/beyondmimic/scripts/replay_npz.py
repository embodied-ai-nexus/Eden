"""Replay a converted .npz motion clip on a G1 in an interactive scene (BeyondMimic data prep).

---
tier: project
capabilities: [task.mimic, robot.g1]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/mimic/beyondmimic/scripts/replay_npz.py --file path_to_motion.npz
---

.. code-block:: bash

    # Usage
    python replay_npz.py --file path_to_motion.npz
"""

import argparse
import torch
from tqdm import tqdm

from genesis.utils.geom import inv_transform_by_quat

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions
from eden.managers.terms.commands.mimic import MotionLoader

from eden.options.camera import CameraOptions, CamerasOptions


def main():
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Replay converted motions.")
    parser.add_argument("--file", type=str, required=True, help="Converted .npz motion to replay")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the viewer.")
    parser.add_argument("-r", "--record", action="store_true", help="Record video.")
    args = parser.parse_args()

    en.init(debug=True)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.02,
        ),
        scene_options=en.scenes.Empty(
            robot=en.robots.G1Dofs29(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(1.5, 0.0, 1),
                cam_lookat=(0.0, 0.0, 0.5),
            ),
        )
        if args.record
        else CamerasOptions(),
        show_viewer=args.vis,
    )

    env.build()

    robot = env.entities["robot"]

    _, robot_base_idx = robot.find_named_dofs_idx_local("floating_base_joint")

    # Only the root/base body is needed for the replay (set_qpos base pose). Resolve it by name so
    # the npz body column is selected correctly regardless of Genesis's internal link ordering.
    base_link_name = robot.links[0].name
    motion = MotionLoader(
        args.file,
        torch.tensor([0], dtype=torch.long, device=env.device),
        env.device,
        links_name=[base_link_name],
    )
    time_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # NOTE: the conversion done in isaac/mjlab has bug in not preserving the specified order of the joints, try following map if the visualization is not correct
    # isaac_map = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

    if args.record:
        cam = env.cameras["camera"]
        cam.start_recording(args.file.replace(".npz", ".mp4"))

    for _ in tqdm(range(motion.time_step_total - 1)):
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        # robot.set_dofs_pos(motion.dofs_pos[time_steps][:, isaac_map])
        # robot.set_dofs_vel(motion.dofs_vel[time_steps][:, isaac_map])
        robot.set_qpos(
            torch.cat(
                [
                    motion.links_pos_w[time_steps][:, 0],
                    motion.links_quat_w[time_steps][:, 0],
                    motion.dofs_pos[time_steps],
                ],
                dim=1,
            ),
        )
        robot.set_dofs_vel(motion.dofs_vel[time_steps])
        # Genesis stores a free joint's rotational DOF velocity in the base BODY frame
        # (MuJoCo qvel convention); links_ang_vel_w is world-frame. Rotate world->body with
        # the base link quat written to qpos above. Linear DOFs stay world-frame.
        base_ang_vel_b = inv_transform_by_quat(
            motion.links_ang_vel_w[time_steps][:, 0],
            motion.links_quat_w[time_steps][:, 0],
        )
        robot.set_dofs_vel(
            torch.cat(
                [
                    motion.links_lin_vel_w[time_steps][:, 0],
                    base_ang_vel_b,
                ],
                dim=1,
            ),
            dofs_idx_local=robot_base_idx,
        )

        env.scene.visualizer.update()
        if args.record:
            cam.render_rgb()

    if args.record:
        cam.stop_recording()


if __name__ == "__main__":
    # run the main function
    main()
