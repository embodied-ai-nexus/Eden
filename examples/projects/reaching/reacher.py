"""Train / infer the Reacher benchmark task on a YAM arm with RSL-RL.

---
tier: project
capabilities: [task.reaching, robot.yam]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/reaching/reacher.py train
---

Usage:
    python examples/projects/reaching/reacher.py train
    python examples/projects/reaching/reacher.py train --cpu
    python examples/projects/reaching/reacher.py inference --model <checkpoint.pt>
"""

import argparse
import genesis as gs
import torch
from tqdm import tqdm
from rsl_rl.runners import OnPolicyRunner

import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.options.camera import CameraOptions
from eden.utils.assets import find_latest_checkpoint


def main(args: argparse.Namespace):
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/reacher",
        performance_mode=True,
    )

    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("reacher")
    config.env_options.num_envs = args.num_envs

    agent_config = config.runner_options
    if args.cpu:
        agent_config.device = "cpu"

    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config)

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])

    runner.learn(num_learning_iterations=agent_config.max_iterations, init_at_random_ep_len=True)


def inference(args: argparse.Namespace):
    if not gs._initialized:
        en.init(
            backend=gs.gpu if not args.cpu else gs.cpu,
            log_root_path="logs/reacher",
            performance_mode=True,
        )

    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("reacher")
    config.env_options.num_envs = 1

    if args.record:
        config.cameras_options.rec = CameraOptions()

    agent_config = config.runner_options
    if args.cpu:
        agent_config.device = "cpu"

    # disable wandb for inference
    agent_config.logger = "tensorboard"

    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True)

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.load(args.model)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    if args.record:
        rec = env.unwrapped.cameras["rec"]
        rec.start_recording()

    command = env.unwrapped.command_manager.get_command("reaching_target")

    for i in tqdm(range(env.unwrapped.max_episode_length - 1)):
        with torch.no_grad():
            raw_action = policy(obs)
        obs, _, _, _ = env.step(raw_action)
        env.unwrapped.draw_debug_sphere(command[..., :3], radius=0.03)
        if args.record:
            rec.render_rgb()
        env.unwrapped.clear_debug_objects()

    if args.record:
        rec.stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reacher task.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    main_parser = subparsers.add_parser("train", help="Train Reacher task.")
    main_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    main_parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of environments")

    inference_parser = subparsers.add_parser("inference", help="Run inference on Reacher task.")
    inference_parser.add_argument("--model", type=str, help="Path to the model checkpoint .pt file")
    inference_parser.add_argument("-r", "--record", action="store_true", help="Record video")
    inference_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    args = parser.parse_args()

    if args.command == "train":
        main(args)
        # Chain into inference through the real parser: a hand-built Namespace has to
        # restate argparse's dest names, and drifts from them silently.
        argv = ["inference", "--record", "--model", find_latest_checkpoint(en.log_dir, return_path=True)]
        if args.cpu:
            argv.append("--cpu")
        inference(parser.parse_args(argv))
    elif args.command == "inference":
        inference(args)
