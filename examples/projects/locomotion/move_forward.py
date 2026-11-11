"""Train/eval the move_forward task with robot + controller + runner picked via CLI.

---
tier: project
capabilities: [task.locomotion, robot.go1, robot.go2]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/locomotion/move_forward.py --task move_forward
---

Three axes are exposed as TaskMod flags:

  * ``--robot {go1, go2}`` — quadruped variant. Swaps the scene robot, the
    controller's joint set + per-joint action scale, and the base-height
    reward target.

  * ``--controller {implicit, explicit}`` — joint PD controller flavor.
    Implicit (default) uses Genesis's built-in position control with
    sim_dt=0.02, substeps=2, decimation=1. Explicit computes torques in
    user space and pulls the sim params to dt=0.01, substeps=1, decimation=2
    for stability.

  * ``--runner {rsl_rl, rl_games}`` — RL training framework. rsl_rl is
    the default; rl_games swaps in its own runner schema and drops the
    unused critic observation group.

The two frameworks have different runner classes / env wrappers / inference
policy shapes, so the script dispatches into ``run_rsl_rl`` or ``run_rl_games``
end-to-end based on ``--runner``. Everything upstream of that — config
construction via the registry, mods applied, save_as_json, camera setup —
is shared.

Usage
-----
::

    # Defaults: go2 + implicit + rsl_rl
    python move_forward.py --task move_forward

    # Go1 instead of Go2
    python move_forward.py --task move_forward --robot go1

    # Explicit PD
    python move_forward.py --task move_forward --controller explicit

    # rl_games runner
    python move_forward.py --task move_forward --runner rl_games

    # Compose: Go1 + explicit PD trained with rl_games
    python move_forward.py --task move_forward --robot go1 --controller explicit --runner rl_games

    # Eval-only (rsl_rl only — rl_games checkpoint-load isn't wired here):
    python move_forward.py --task move_forward --checkpoint path/to/model.pt
"""

import argparse

import genesis as gs
import torch

import eden as en
from eden.options.camera import CameraOptions
from eden.tasks.parser import get_task_argparser, get_task_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = get_task_argparser(
        description="Train/eval the move_forward locomotion task with TaskMod-driven CLI.",
        default_task_name="move_forward",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "neptune", "wandb"],
        help="rsl_rl logger backend (ignored when --runner rl_games).",
    )
    return parser


def _init_genesis(args: argparse.Namespace) -> None:
    if not gs._initialized:
        en.init(
            backend=gs.gpu if not args.cpu else gs.cpu,
            log_root_path=f"logs/{args.task}",
            performance_mode=True,
        )


def _attach_eval_camera(config) -> None:
    """Common eval rig: single env, world frame on, follow camera at 1280x720."""
    # Render a single env at eval. eval_mode no longer reduces the env count (num_eval_envs was
    # removed), so without this the eval env inherits the training num_envs (4096) and rendering crawls.
    config.env_options.num_envs = 1
    config.env_options.show_world_frame = True
    config.cameras_options.camera = CameraOptions(
        cam_pos=(1.0, 0.5, 1.0),
        cam_lookat=(0.0, 0.0, 0.3),
        cam_resolution=(1280, 720),
        follow_entity_name="robot",
    )


# ---------------------------------------------------------------------------
# rsl_rl path
# ---------------------------------------------------------------------------


def run_rsl_rl(args: argparse.Namespace) -> None:
    from rsl_rl.runners import OnPolicyRunner

    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

    _init_genesis(args)

    config = get_task_config_from_args(args, run_name=args.run_name)
    config.runner_options.logger = args.logger

    if args.checkpoint:
        # Eval-only: skip training, build the eval env directly.
        _attach_eval_camera(config)
        config.save_as_json()
        env = RslRlVecEnvWrapper.from_config(config, eval_mode=True)
        runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=gs.device)
        _eval_loop_rsl_rl(env, config, policy)
        return

    # Train.
    config.save_as_json()
    env = RslRlVecEnvWrapper.from_config(config)
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.add_git_repo_to_log(__file__)
    runner.learn(num_learning_iterations=config.runner_options.max_iterations, init_at_random_ep_len=True)
    policy = runner.get_inference_policy(device=gs.device)
    del env

    # Eval the trained policy with recording.
    eval_config = get_task_config_from_args(args, run_name=args.run_name)
    _attach_eval_camera(eval_config)
    eval_env = RslRlVecEnvWrapper.from_config(eval_config, eval_mode=True)
    _eval_loop_rsl_rl(eval_env, eval_config, policy)


def _eval_loop_rsl_rl(env, config, policy) -> None:
    obs, _ = env.reset()
    env.env.cameras["camera"].start_recording()
    with torch.no_grad():
        for _ in range(int(config.env_options.episode_length_s / env.dt)):
            obs, _, _, _ = env.step(policy(obs))
            env.env.cameras["camera"].render_rgb()
    env.env.cameras["camera"].stop_recording()


# ---------------------------------------------------------------------------
# rl_games path
# ---------------------------------------------------------------------------


def run_rl_games(args: argparse.Namespace) -> None:
    from rl_games.torch_runner import Runner

    from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper

    if args.checkpoint:
        raise NotImplementedError(
            "rl_games checkpoint-load isn't wired in this example. "
            "Drop --checkpoint to run train+eval, or load via the rl_games Runner directly."
        )

    _init_genesis(args)

    config = get_task_config_from_args(args, run_name=args.run_name)
    config.save_as_json()

    # Train.
    env = RlGamesVecEnvWrapper.from_config(config)
    runner = Runner()
    runner.load(env.runner_dict)
    env.configure_runner(runner)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    agent.train()

    # Extract a deterministic inference policy from the trained model.
    model = agent.model
    model.eval()
    del env

    # Eval with recording.
    eval_config = get_task_config_from_args(args, run_name=args.run_name)
    _attach_eval_camera(eval_config)
    eval_env = RlGamesVecEnvWrapper.from_config(eval_config, eval_mode=True)

    obs = eval_env.reset()
    eval_env.env.cameras["camera"].start_recording()
    with torch.no_grad():
        for _ in range(int(eval_config.env_options.episode_length_s / eval_env.dt)):
            res = model({"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None})
            obs, _, _, _ = eval_env.step(res["mus"])  # deterministic
            eval_env.env.cameras["camera"].render_rgb()
    eval_env.env.cameras["camera"].stop_recording()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.runner == "rl_games":
        run_rl_games(args)
    else:
        run_rsl_rl(args)
