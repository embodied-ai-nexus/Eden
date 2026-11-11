"""
BeyondMimic — motion imitation task.

---
tier: project
capabilities: [task.mimic, robot.g1, deploy.real_robot]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/mimic/beyondmimic/main.py train
---

Single-GPU training::

    python main.py train --num-envs 4096

Multi-GPU DDP training (e.g. 4 GPUs)::

    torchrun --standalone --nproc_per_node=4 main.py train --num-envs 4096

"""

import argparse
import os

import eden as en  # must be before genesis to isolate GPUs for DDP
import genesis as gs
import torch
from tqdm import tqdm
from rsl_rl.runners import OnPolicyRunner
from eden.utils import distributed as eden_dist
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, DDPOnPolicyRunner
from eden.options.camera import CameraOptions
from eden.options import GHOST_DEFAULTS, EventManagerOptions
from eden.utils.assets import find_latest_checkpoint


def train(args: argparse.Namespace):
    en.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        log_root_path="logs/beyondmimic",
        performance_mode=True,
    )

    from config import BeyondMimicConfig

    config = BeyondMimicConfig()
    config.env_options.num_envs = args.num_envs
    config.command_options.motion.motion_file = args.motion_file
    config.command_options.motion.debug_vis = False
    if args.deployable:
        config.observation_options.policy.disable_term("motion_anchor_pos_b")
        config.observation_options.policy.disable_term("base_lin_vel")
    agent_config = config.runner_options
    agent_config.logger = args.logger

    if args.cpu:
        agent_config.device = "cpu"

    # In DDP mode, each rank sees one GPU as cuda:0 (CUDA_VISIBLE_DEVICES
    # is set per-process by eden.init).
    if eden_dist.is_distributed():
        agent_config.device = "cuda:0"

    # Only save config on main process to avoid race conditions.
    if eden_dist.is_main_process():
        config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config)

    # Use the DDP-compatible runner when in distributed mode, otherwise
    # fall back to the standard runner for full backward-compatibility.
    runner_cls = DDPOnPolicyRunner if eden_dist.is_distributed() else OnPolicyRunner
    runner = runner_cls(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.add_git_repo_to_log(__file__)

    if args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=agent_config.max_iterations, init_at_random_ep_len=True)

    # Cleanup distributed resources.
    eden_dist.cleanup()


def inference(args: argparse.Namespace):
    if not gs._initialized:
        en.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            log_root_path="logs/beyondmimic",
        )

    from config import BeyondMimicConfig

    config = BeyondMimicConfig()
    config.command_options.motion.motion_file = args.motion_file
    agent_cfg = config.runner_options

    config.env_options.episode_length_s = 1e9
    config.env_options.num_envs = 1
    config.scene_options.ghost_g1 = en.robots.G1Dofs29(**GHOST_DEFAULTS)
    config.command_options.motion.ghost_entity_name = "ghost_g1"
    config.command_options.motion.sampling_mode = "start"
    config.command_options.motion.debug_vis = True
    # NOTE: disable randomization for inference
    config.command_options.motion.pose_range = {}
    config.command_options.motion.velocity_range = {}
    config.command_options.motion.dofs_position_range = (0.0, 0.0)
    if args.deployable:
        config.observation_options.policy.disable_term("motion_anchor_pos_b")
        config.observation_options.policy.disable_term("base_lin_vel")
    config.event_options = EventManagerOptions()

    if args.record:
        config.cameras_options.rec = CameraOptions()

    if args.cpu:
        agent_cfg.device = "cpu"

    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.vis, eval_mode=True)

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.add_git_repo_to_log(__file__)

    cmd = env.unwrapped.command_manager.get_term("motion")
    runner.load(args.model)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    if args.record:
        rec = env.unwrapped.cameras["rec"]
        clip = os.path.splitext(os.path.basename(args.motion_file))[0]
        rec.start_recording(os.path.join(en.log_dir, f"{clip}.mp4"))

    for i in tqdm(range(int(cmd.motion.time_step_total) - 1)):
        with torch.no_grad():
            raw_action = policy(obs)
        obs, _, _, _ = env.step(raw_action)
        if args.record:
            rec.render_rgb()

    if args.record:
        rec.stop_recording()


def deploy(args: argparse.Namespace):
    if not gs._initialized:
        en.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            log_root_path="logs/beyondmimic",
        )

    from config import BeyondMimicConfig

    config = BeyondMimicConfig()
    config.command_options.motion.motion_file = args.motion_file
    config.scene_options.ghost_g1 = en.robots.G1Dofs29(**GHOST_DEFAULTS)
    config.command_options.motion.ghost_entity_name = "ghost_g1"
    config.command_options.motion.sampling_mode = "start"
    config.command_options.motion.debug_vis = True
    config.env_options.episode_length_s = 1e9
    config.env_options.num_envs = 1
    if True:
        config.observation_options.policy.disable_term("motion_anchor_pos_b")
        config.observation_options.policy.disable_term("base_lin_vel")
    agent_config = config.runner_options

    if args.cpu:
        agent_config.device = "cpu"

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.vis, eval_mode=True)

    # NOTE: keep a reference to the sim robot *before* the deployment swaps it
    sim_robot = env.unwrapped.entities["robot"]

    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    dep = UnitreeG1Deployment(
        env.unwrapped,
        UnitreeG1Deployment.configure(
            entity_name="robot",
            sync=True,
            connect_timeout_s=10.0,
            deploy_kp_kd_scale=1.0,
            **publish_rate_configure_kwargs(args.publish_rate),
        ),
    )
    dep.connect()

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.add_git_repo_to_log(__file__)

    cmd = env.unwrapped.command_manager.get_term("motion")
    runner.load(args.model)
    policy = runner.get_inference_policy(device=env.device)

    obs = dep.reset()

    for i in tqdm(range(int(cmd.motion.time_step_total) - 1)):
        with torch.no_grad():
            raw_action = policy(obs)
        obs = dep.step(raw_action)
        sim_robot.set_quat(dep.state.base_quat)
        sim_robot.set_dofs_pos(dep.state.dofs_pos)
        sim_robot.set_dofs_vel(dep.state.dofs_vel)
        env.unwrapped.scene.visualizer.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeyondMimic motion imitation task.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train the motion imitation policy.")
    train_parser.add_argument(
        "--motion-file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sample/G1_gangnam_style_V01_bvh_60hz.npz"),
        help="Path to the reference motion to track",
    )
    train_parser.add_argument("--deployable", action="store_true", help="Make the policy deployable")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint to resume from")
    train_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    train_parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of environments")
    train_parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "neptune", "wandb"], help="Logger to use"
    )
    # --- inference subcommand ---
    inference_parser = subparsers.add_parser("inference", help="Run inference on trained model.")
    inference_parser.add_argument("--model", type=str, help="Path to the model checkpoint .pt file")
    inference_parser.add_argument(
        "--motion-file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sample/G1_gangnam_style_V01_bvh_60hz.npz"),
        help="Path to the reference motion to track",
    )
    inference_parser.add_argument("--deployable", action="store_true", help="Use deployable mode")
    inference_parser.add_argument("-r", "--record", action="store_true", help="Record video")
    inference_parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    inference_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")

    # --- deploy subcommand ---
    deploy_parser = subparsers.add_parser("deploy", help="Deploy the motion imitation policy.")
    deploy_parser.add_argument("--model", type=str, help="Path to the model checkpoint .pt file")
    deploy_parser.add_argument(
        "--motion-file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sample/G1_gangnam_style_V01_bvh_60hz.npz"),
        help="Path to the reference motion to track",
    )
    deploy_parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    deploy_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    deploy_parser.add_argument(
        "--publish-rate",
        type=float,
        default=None,
        help="Hz at which the held wire command is re-sent between control ticks (default: the "
        "backend's 500 Hz, Unitree's own low-level rate). The G1's parallel ankle/waist joints "
        "close their PD once per received packet. Pass 0 for one packet per control tick -- the "
        "A/B against that default.",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
        # Only run inference on the main process after DDP training.
        if eden_dist.is_main_process():
            # Chain into inference through the real parser: a hand-built Namespace has to
            # restate argparse's dest names, and drifts from them silently.
            argv = ["inference", "--record", "--motion-file", args.motion_file]
            argv += ["--model", find_latest_checkpoint(en.log_dir, return_path=True)]
            if args.deployable:
                argv.append("--deployable")
            if args.cpu:
                argv.append("--cpu")
            inference(parser.parse_args(argv))
    elif args.command == "inference":
        inference(args)
    elif args.command == "deploy":
        deploy(args)
