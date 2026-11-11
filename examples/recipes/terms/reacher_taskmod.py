"""Train/eval Reacher (or any registered task) via the TaskMod registry.

---
tier: recipe
capabilities: [term.taskmod]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/terms/reacher_taskmod.py --task reacher
---

This is the canonical training-script template. The argparse layer comes
from ``get_task_argparser``, which:

  1. Pre-parses ``--task`` and adds *only that task's* TaskMod flags to the
     full parser, so ``--help`` shows the relevant knobs (``--difficulty``
     for reacher, ``--robot`` for an arm task, etc.).
  2. Provides standard plumbing: ``--config`` (a config file: a saved one
     replaces, a partial one layers), ``--override`` (inline YAML overrides,
     repeatable), ``--checkpoint`` (eval mode), ``--cpu`` (backend), and
     ``--run_name`` (sanitized into ``runner_options.run_name``).

``get_task_config_from_args`` then routes through ``TASK_REGISTRY.build``,
which applies the registered TaskMods — so ``--difficulty hard`` actually
runs ``ReacherDifficultyMod.apply`` on the config before training starts.

Usage
-----
::

    # Train then eval the latest checkpoint with recording (same as the old
    # reacher.py auto-flow):
    python reacher_taskmod.py --task reacher --difficulty hard --num-envs 8192

    # Eval only — pass --checkpoint:
    python reacher_taskmod.py --task reacher --checkpoint path/to/model.pt --record

    # Other registered tasks slot into the same script unchanged:
    python reacher_taskmod.py --task velocity_tracking_flat
"""

import argparse

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.options.camera import CameraOptions
from eden.tasks.parser import get_task_argparser, get_task_config_from_args
from eden.utils.assets import find_latest_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = get_task_argparser(
        description="Train or evaluate a registered task with TaskMod-driven CLI.",
        default_task_name="reacher",
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of environments (training).")
    parser.add_argument("-r", "--record", action="store_true", help="Record video during evaluation.")
    return parser


def train(args: argparse.Namespace) -> None:
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{args.task}",
        performance_mode=True,
    )

    config = get_task_config_from_args(args, run_name=args.run_name)
    config.env_options.num_envs = args.num_envs
    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config)
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.learn(num_learning_iterations=config.runner_options.max_iterations, init_at_random_ep_len=True)


def evaluate(args: argparse.Namespace, checkpoint: str) -> None:
    if not gs._initialized:
        en.init(
            backend=gs.gpu if not args.cpu else gs.cpu,
            log_root_path=f"logs/{args.task}",
            performance_mode=True,
        )

    config = get_task_config_from_args(args, run_name=args.run_name)
    config.env_options.num_envs = 1
    if args.record:
        config.cameras_options.rec = CameraOptions()
    # Disable wandb for evaluation runs.
    config.runner_options.logger = "tensorboard"
    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True)
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.reset()
    if args.record:
        rec = env.unwrapped.cameras["rec"]
        rec.start_recording()

    command = env.unwrapped.command_manager.get_command("reaching_target")
    for _ in tqdm(range(env.unwrapped.max_episode_length - 1)):
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
    args = build_parser().parse_args()

    if args.checkpoint:
        # Eval-only: --checkpoint was provided.
        evaluate(args, args.checkpoint)
    else:
        # Train, then eval the latest checkpoint with recording (matches the
        # legacy reacher.py auto-flow).
        train(args)
        eval_args = argparse.Namespace(**vars(args))
        eval_args.record = True
        evaluate(eval_args, find_latest_checkpoint(en.log_dir, return_path=True))
