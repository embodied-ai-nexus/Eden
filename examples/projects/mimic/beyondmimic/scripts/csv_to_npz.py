"""Replay a motion from a CSV file and export it to an Eden .npz motion clip (BeyondMimic data prep).

---
tier: project
capabilities: [task.mimic, robot.g1]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/mimic/beyondmimic/scripts/csv_to_npz.py --input-file path_to_input.csv --input-fps 60
---

.. code-block:: bash

    # Usage
    python csv_to_npz.py --input-file path_to_input.csv --input-fps 60
"""

import argparse

import numpy as np
import torch

from genesis.utils.geom import slerp

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions
from genesis.utils.geom import inv_quat, inv_transform_by_quat

from eden.utils.geom import axis_angle_from_quat, quat_mul, xyzw_to_wxyz


class MotionLoader:
    """
    Loader for motion commands.

    Parameters
    ----------
    motion_file: str
        Path to the motion file.
    input_fps: int
        The fps of the input motion.
    output_fps: int
        The fps of the output motion.
    device: torch.device
        Device to load the data on.
    frame_range: tuple[int, int] | None
        The frame range to load. If None, all frames will be loaded.
    """

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = xyzw_to_wxyz(motion[:, 3:7])
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.

        Returns
        -------
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, inv_quat(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def main():
    parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="The path to the input motion csv file.",
    )
    parser.add_argument("--input-fps", type=int, default=60, help="The fps of the input motion.")
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
            " loaded."
        ),
    )
    parser.add_argument("-o", "--output", type=str, help="The name of the motion npz file.")
    parser.add_argument("--output-fps", type=int, default=50, help="The fps of the output motion.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the viewer.")
    args = parser.parse_args()

    if not args.output:
        # generate at the same location as input file
        args.output = (
            "/".join(args.input_file.split("/")[:-1]) + "/" + args.input_file.split("/")[-1].replace(".csv", ".npz")
        )

    en.init(debug=True)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / args.output_fps,
        ),
        scene_options=en.scenes.Empty(
            robot=en.robots.G1Dofs29(),
        ),
        show_viewer=args.vis,
    )

    env.build()

    robot = env.entities["robot"]
    robot_dofs_name, robot_dofs_idx = robot.find_named_dofs_idx_local(
        dofs_name=env.scene_options.robot.dofs_name,
        preserve_order=True,
    )
    _, robot_base_idx = robot.find_named_dofs_idx_local("floating_base_joint")

    # Column labels for the logged arrays. ``joint_*`` are stored in ``robot_dofs_name`` order
    # (the order get_dofs_pos/get_dofs_vel return below) and ``body_*`` in the robot's internal
    # link order (the order get_links_* return below). Persisting these names makes the npz
    # self-describing so consumers resolve columns by name and stay correct across genesis
    # versions that reorder the internal kinematic tree (BFS->DFS in genesis 1.2.0).
    robot_body_names = [link.name for link in robot.links]

    en.logger.info(f"robot_dofs_idx: {robot_dofs_idx}")
    en.logger.info(f"robot_base_idx: {robot_base_idx}")

    motion = MotionLoader(
        motion_file=args.input_file,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        device=env.device,
        frame_range=args.frame_range,
    )

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args.output_fps],
        "dof_names": robot_dofs_name,
        "body_names": robot_body_names,
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while True:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        robot.set_qpos(
            torch.cat([motion_base_pos, motion_base_rot, motion_dof_pos], dim=1),
        )
        robot.set_dofs_vel(motion_dof_vel, dofs_idx_local=robot_dofs_idx)
        if (motion_base_lin_vel.abs() > 1.5).any():
            en.logger.warning(
                f"High linear velocity is detected at {motion.current_idx} frame, value: {motion_base_lin_vel}"
            )
        if (motion_base_ang_vel.abs() > 10.0).any():
            en.logger.warning(
                f"High angular velocity is detected at {motion.current_idx} frame, value: {motion_base_ang_vel}"
            )
        # Genesis stores a free joint's rotational DOF velocity in the base BODY frame
        # (MuJoCo qvel convention); motion_base_ang_vel is world-frame. Rotate world->body
        # with the same quat written to qpos above. Linear DOFs stay world-frame.
        motion_base_ang_vel_b = inv_transform_by_quat(motion_base_ang_vel, motion_base_rot)
        robot.set_dofs_vel(
            torch.cat([motion_base_lin_vel, motion_base_ang_vel_b], dim=1),
            dofs_idx_local=robot_base_idx,
        )

        env.scene.visualizer.update()

        if not file_saved:
            log["joint_pos"].append(robot.get_dofs_pos()[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.get_dofs_vel()[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.get_links_pos()[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.get_links_quat()[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.get_links_vel()[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.get_links_ang()[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez(args.output, **log)
            print("[INFO]: Motion npz file saved to", args.output)
            break


if __name__ == "__main__":
    # run the main function
    main()
