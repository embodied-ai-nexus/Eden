"""Velocity tracking training and interactive play for bipedal humanoids (G1 29DOF, LimX Oli).

---
tier: project
capabilities: [task.locomotion, robot.g1, robot.oli, terrain.procedural, teleop.keyboard]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/locomotion/velocity_tracking/main.py train
---

Usage:
    # Training (flat terrain, default)
    python examples/projects/locomotion/velocity_tracking/main.py train
    python examples/projects/locomotion/velocity_tracking/main.py train --task terrain
    python examples/projects/locomotion/velocity_tracking/main.py train --task terrain --num-envs 2048
    python examples/projects/locomotion/velocity_tracking/main.py train --logger wandb
    python examples/projects/locomotion/velocity_tracking/main.py train --robot oli
    python examples/projects/locomotion/velocity_tracking/main.py train --obs-preset privileged

    # Interactive play with keyboard control
    python examples/projects/locomotion/velocity_tracking/main.py play --model logs/velocity_tracking/g1_velocity_tracking_flat/.../model_10000.pt
    python examples/projects/locomotion/velocity_tracking/main.py play --task terrain --model logs/velocity_tracking/g1_velocity_tracking_terrain/.../model_10000.pt
"""

import argparse

import numpy as np
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.options.camera import CameraOptions
from eden.utils.geom import quat_apply_yaw
from eden.managers.terms.recorders import ActionRecorder, StateRecorder, InitialStateRecorder
from eden.options import RecorderManagerOptions


_TASK_MAP = {
    "flat": "velocity_tracking_flat",
    "terrain": "velocity_tracking_terrain",
}


def _make_config(task: str, robot: str, obs_preset: str = "proprioceptive"):
    """Build the task config through the registry so the task's modifiers run.

    ``TASK_REGISTRY.get(name)()`` would skip them and leave the robot-bound slots
    (torso/feet link names, free-joint name, action scale) at their empty sentinels,
    which then fails at ``env.build()``.

    ``obs_preset`` selects the policy observation recipe: ``proprioceptive`` (sensor-only,
    5-frame history) or ``privileged`` (base_lin_vel in the policy, single frame). ``play``
    has to pass the same one the checkpoint trained with, or the observation width will not
    match the network.
    """
    from eden.tasks import TASK_REGISTRY

    if task not in _TASK_MAP:
        raise ValueError(f"Unknown task: {task!r}. Choose from {list(_TASK_MAP)}")
    return TASK_REGISTRY.build(_TASK_MAP[task], robot=robot, obs_preset=obs_preset)


def train(args):
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/velocity_tracking",
        performance_mode=True,
    )

    config = _make_config(args.task, args.robot, args.obs_preset)
    config.env_options.num_envs = args.num_envs
    config.runner_options.max_iterations = args.max_iterations
    config.runner_options.logger = args.logger
    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config)

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.add_git_repo_to_log(__file__)

    if args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=config.runner_options.max_iterations, init_at_random_ep_len=True)
    policy = runner.get_inference_policy(device=gs.device)
    del env, config

    # Evaluation
    eval_config = _make_config(args.task, args.robot, args.obs_preset)
    eval_config.env_options.num_envs = 1
    eval_config.env_options.episode_length_s = 100.0
    eval_config.cameras_options.camera = CameraOptions(
        cam_pos=(2.0, 1.0, 1.5),
        cam_lookat=(0.0, 0.0, 0.8),
        cam_resolution=(1280, 720),
        follow_entity_name="robot",
    )
    eval_env = RslRlVecEnvWrapper.from_config(eval_config, eval_mode=True)

    obs, _ = eval_env.reset()
    eval_env.env.cameras["camera"].start_recording()
    with torch.no_grad():
        for _ in range(int(eval_config.env_options.episode_length_s / eval_env.dt)):
            action = policy(obs)
            obs, rews, dones, infos = eval_env.step(action)
            eval_env.env.cameras["camera"].render_rgb()
    eval_env.env.cameras["camera"].stop_recording()


def play(args):
    from eden.utils.devices.keyboard import SE2Keyboard

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/velocity_tracking",
        performance_mode=False,
    )

    config = _make_config(args.task, args.robot, args.obs_preset)
    config.env_options.num_envs = 1
    config.env_options.episode_length_s = 1_000_000.0
    config.env_options.enable_default_keybinds = False
    config.event_options.disable_term("reset_base")
    config.event_options.disable_term("reset_dofs")
    config.event_options.disable_term("push_robot")
    config.event_options.disable_term("randomize_friction")
    config.event_options.disable_term("randomize_com")
    config.event_options.disable_term("encoder_bias")
    if hasattr(config.termination_options, "out_of_terrain_bounds"):
        config.termination_options.disable_term("out_of_terrain_bounds")
    config.runner_options.logger = "tensorboard"
    if args.record:
        config.recorder_options = RecorderManagerOptions(
            dataset_export_dir_path="rec",
            action_recorder=ActionRecorder.configure(),
            initial_state_recorder=InitialStateRecorder.configure(),
            state_recorder=StateRecorder.configure(),
        )

    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True, show_viewer=True)
    inner_env = env.unwrapped

    # Load trained policy
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.load(args.model)
    policy = runner.get_inference_policy(device=gs.device)

    # Set up keyboard controller
    keyboard = SE2Keyboard(
        key_bindings=SE2Keyboard.wasd_key_bindings(),
        vel_max=args.vel_max,
        ang_max=args.ang_max,
    )

    def _on_record(on_press):
        if not on_press:
            return
        if args.record:
            if not inner_env.recorder_manager.is_recording:
                inner_env.recorder_manager.start_recording(reset_scene=True)
                en.logger.info("Recording started.")
            else:
                inner_env.recorder_manager.end_recording()
                en.logger.info("Saving episode.")
        else:
            en.logger.warning("Recording is not enabled. Use --record to enable recording.")

    keyboard.add_callback("r", _on_record)

    # Get the velocity command term so we can override it with keyboard input
    cmd_term = inner_env.command_manager.get_term("velocity_command")
    robot = inner_env.entities["robot"]

    # Arrow drawing height offset above robot base
    arrow_z_offset = 1.0
    arrow_cmd = None
    arrow_actual = None

    print("\n" + "=" * 60)
    print(f"  Velocity Tracking ({args.robot}, {args.task}) - Interactive Play")
    print("=" * 60)
    print("  W/S : forward / backward")
    print("  A/D : strafe left / right")
    print("  Q/E : rotate left / right")
    print("  R   : save episode (if recording is enabled), reset & start new recording")
    print("=" * 60 + "\n")

    obs = env.get_observations()

    with torch.no_grad():
        while True:
            # Read keyboard command and override the velocity command
            kb_cmd = keyboard.get_command()  # (3,) [vx, vy, wz]
            cmd_term.vel_command_b[0, :] = kb_cmd.to(gs.device)

            # Run policy
            action = policy(obs)
            obs, rews, dones, infos = env.step(action)

            # -- Debug arrow visualization --
            # Clear previous arrows
            if arrow_cmd is not None:
                inner_env.clear_debug_object(arrow_cmd)
            if arrow_actual is not None:
                inner_env.clear_debug_object(arrow_actual)

            base_pos = robot.get_pos().squeeze(0)  # (3,)
            base_quat = robot.get_quat()  # (1, 4)
            arrow_origin = base_pos.cpu().numpy().copy()
            arrow_origin[2] += arrow_z_offset

            # Commanded velocity (body frame -> world frame via yaw rotation)
            cmd_body = cmd_term.vel_command_b[0:1, :]  # (1, 3)
            cmd_world = quat_apply_yaw(base_quat, cmd_body).squeeze(0)  # (3,)
            cmd_vec = cmd_world.cpu().numpy()
            # Scale for visibility; arrow shows xy velocity direction + magnitude
            cmd_arrow_vec = np.array([cmd_vec[0], cmd_vec[1], 0.0])

            # Actual velocity (already in world frame)
            actual_vel = robot.get_vel(frame="world").squeeze(0)  # (3,)
            actual_vec = actual_vel.cpu().numpy()
            actual_arrow_vec = np.array([actual_vec[0], actual_vec[1], 0.0])

            # Draw: blue = commanded, green = actual
            if np.linalg.norm(cmd_arrow_vec) > 0.01:
                arrow_cmd = inner_env.draw_debug_arrow(
                    pos=arrow_origin,
                    vec=cmd_arrow_vec,
                    radius=0.015,
                    color=(0.2, 0.4, 1.0, 0.8),
                )
            else:
                arrow_cmd = None

            if np.linalg.norm(actual_arrow_vec) > 0.01:
                arrow_actual = inner_env.draw_debug_arrow(
                    pos=arrow_origin,
                    vec=actual_arrow_vec,
                    radius=0.01,
                    color=(0.2, 1.0, 0.4, 0.8),
                )
            else:
                arrow_actual = None


def main():
    parser = argparse.ArgumentParser(description="Humanoid velocity tracking: train or play.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # -- train --
    train_parser = subparsers.add_parser("train", help="Train the velocity tracking policy")
    train_parser.add_argument("--task", type=str, default="flat", choices=["flat", "terrain"], help="Task variant")
    train_parser.add_argument("--robot", type=str, default="g1", choices=["g1", "oli"], help="Humanoid to train / play")
    train_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    train_parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    train_parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "neptune", "wandb"], help="Logger to use"
    )
    train_parser.add_argument("--max-iterations", type=int, default=10001, help="Maximum training iterations")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train_parser.add_argument(
        "--obs-preset",
        type=str,
        default="proprioceptive",
        choices=["proprioceptive", "privileged"],
        help="Policy observation recipe: sensor-only with 5-frame history (default), or "
        "base_lin_vel in the policy at a single frame. `play` must match the checkpoint.",
    )

    # -- play --
    play_parser = subparsers.add_parser("play", help="Interactive play with keyboard control")
    play_parser.add_argument("--task", type=str, default="flat", choices=["flat", "terrain"], help="Task variant")
    play_parser.add_argument("--robot", type=str, default="g1", choices=["g1", "oli"], help="Humanoid to train / play")
    play_parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    play_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    play_parser.add_argument("--vel-max", type=float, default=2.0, help="Maximum linear velocity from keyboard")
    play_parser.add_argument("--ang-max", type=float, default=1.0, help="Maximum angular velocity from keyboard")
    play_parser.add_argument("-r", "--record", action="store_true", help="Record the episode")
    play_parser.add_argument(
        "--obs-preset",
        type=str,
        default="proprioceptive",
        choices=["proprioceptive", "privileged"],
        help="Policy observation recipe: sensor-only with 5-frame history (default), or "
        "base_lin_vel in the policy at a single frame. `play` must match the checkpoint.",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
