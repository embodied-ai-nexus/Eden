"""Train and evaluate a Go1 getup (fall recovery) policy using RSL-RL.

---
tier: project
capabilities: [task.locomotion, robot.go1]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/locomotion/mujoco_playground/getup/main.py train
---

Ported from: mujoco_playground/_src/locomotion/go1/getup.py

Usage:
    python examples/projects/locomotion/mujoco_playground/getup/main.py train
    python examples/projects/locomotion/mujoco_playground/getup/main.py train --cpu
    python examples/projects/locomotion/mujoco_playground/getup/main.py train --logger wandb
    python examples/projects/locomotion/mujoco_playground/getup/main.py inference --model <checkpoint.pt>
"""

import argparse

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.options.camera import CameraOptions
from eden.tasks import TASK_REGISTRY
from eden.utils.assets import find_latest_checkpoint


class GetupEnvWrapper(RslRlVecEnvWrapper):
    """Wrapper for the getup task: NaN sanitization and reward clipping.

    The current-position action offset (``motor_targets = current_qpos + action *
    action_scale``) and the raw-delta clamp now live in the config: the controller
    uses ``ReferenceSource.DELTA`` with ``scale=0.5`` and the runner's
    ``clip_actions=1.0`` clamps deltas to [-1, 1]. So the action manager already
    holds the raw policy delta that the obs/reward terms read, and this wrapper only
    needs the reference's output sanitization.

    ``from_config`` is inherited: :class:`RslRlVecEnvWrapper` already dispatches through
    ``env_class_for_config``, builds, logs the summary and wraps. Overriding it here only
    to hard-code ``RLEnvBase`` re-stated the base's body one keyword short of its signature.
    """

    def step(self, actions):
        obs_dict, rew, dones, extras = super().step(actions)

        # Sanitize: replace NaN/inf in observations and rewards to prevent
        # training collapse. The violent drops can cause the physics engine
        # to produce NaN states.
        for key in obs_dict.keys():
            obs_dict[key] = torch.nan_to_num(obs_dict[key], nan=0.0, posinf=100.0, neginf=-100.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        # Match reference: clip total reward to [0, 10000]
        rew.clamp_(0.0, 10000.0)

        return obs_dict, rew, dones, extras


def train(args: argparse.Namespace):
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/go1_getup",
        performance_mode=True,
    )

    config = TASK_REGISTRY.build("mujoco_playground_getup")
    agent_cfg = config.runner_options
    agent_cfg.logger = args.logger
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    config.save_as_json()

    env = GetupEnvWrapper.from_config(config)

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.add_git_repo_to_log(__file__)

    if args.resume:
        runner.load(args.resume)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


def inference(args: argparse.Namespace):
    if not gs._initialized:
        en.init(
            backend=gs.gpu if not args.cpu else gs.cpu,
            log_root_path="logs/go1_getup",
        )

    config = TASK_REGISTRY.build("mujoco_playground_getup")
    agent_cfg = config.runner_options
    agent_cfg.logger = "tensorboard"

    # Always drop from height during inference to demonstrate recovery
    config.event_options.drop_from_height.drop_prob = 1.0

    config.cameras_options.camera = CameraOptions(
        cam_pos=(1.5, 0.5, 0.8),
        cam_lookat=(0.0, 0.0, 0.25),
        cam_resolution=(1280, 720),
        follow_entity_name="robot",
    )
    # Render a single env at eval — eval_mode no longer reduces the count, so otherwise this
    # would inherit the training num_envs (4096) and rendering crawls. Set before save_as_json:
    # that call locks the options, so narrowing the width after it raises.
    config.env_options.num_envs = 1
    config.save_as_json()
    env = GetupEnvWrapper.from_config(config, eval_mode=True)

    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.load(args.model)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    env.env.cameras["camera"].start_recording()
    with torch.no_grad():
        for _ in range(int(config.env_options.episode_length_s / env.dt)):
            act = policy(obs)
            obs, rews, dones, infos = env.step(act)
            env.env.cameras["camera"].render_rgb()
    env.env.cameras["camera"].stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go1 getup (fall recovery) task.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train the getup policy.")
    train_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train_parser.add_argument("--max-iterations", type=int, default=None, help="Override max training iterations")
    train_parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "neptune", "wandb"], help="Logger to use"
    )

    # --- inference subcommand ---
    inference_parser = subparsers.add_parser("inference", help="Run inference on trained model.")
    inference_parser.add_argument("--model", type=str, help="Path to model checkpoint .pt file")
    inference_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
        # Chain into inference through the real parser: a hand-built Namespace has to
        # restate argparse's dest names, and drifts from them silently.
        argv = ["inference", "--model", find_latest_checkpoint(en.log_dir, return_path=True)]
        if args.cpu:
            argv.append("--cpu")
        inference(parser.parse_args(argv))
    elif args.command == "inference":
        inference(args)
    else:
        parser.print_help()
