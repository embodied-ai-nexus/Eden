"""HUSKY humanoid skateboarding: play a trained policy (Unitree G1 23-DOF + skateboard).

---
tier: project
capabilities: [task.locomotion, robot.g1, terrain.procedural]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/locomotion/husky_skate/main.py play --steps 50
---

Training port of `TeleHuman/humanoid_skateboarding
<https://github.com/TeleHuman/humanoid_skateboarding>`_ (HUSKY, RSS 2026): the
G1 pushes the board with its left leg, rides with both feet on the deck, and
steers by tilting it. Train with ``eden train --task husky``: the task selects
the source's AMP-augmented PPO (:mod:`eden.learning.wrappers.rsl_rl.amp_ppo` —
adversarial motion prior over the push-phase joint trajectories). ``play``
rolls out a trained checkpoint (the actor is a stock rsl_rl policy) with the
play-mode overrides the source repo uses (long episode, time-out-only
termination, no pushes) and can record an mp4 — on flat ground or on the
zero-shot terrain runway from the frozen-policy example.

Usage::

    eden train --task husky --logger wandb
    python examples/projects/locomotion/husky_skate/main.py play --model logs/husky/<run>/model_<it>.pt --record
    python examples/projects/locomotion/husky_skate/main.py play --model ... --record --terrain rough
"""

from __future__ import annotations

import argparse
import os

import genesis as gs
import torch
from tqdm import tqdm

import eden as en
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, load_runner_checkpoint, make_rsl_rl_runner
from eden.extensions.policies.husky.constants import ROBOT_DEFAULT_ROOT_POS, SKATEBOARD_DEFAULT_ROOT_POS
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import SceneOptions, TerrainOptions
from eden.tasks import TASK_REGISTRY
from eden.utils.terrains import FlatTerrain, SlopedTerrain, WaveTerrain

# Zero-shot terrain runway (same layout as examples/policies/husky/main.py):
# 4 tiles of 8 m along +x — flat -> waves -> flat -> up-ramp — with the run
# starting 2 m into the first flat tile.
_TILE_M = 8.0
_RUNWAY_TILES = [
    [FlatTerrain.configure()],
    [WaveTerrain.configure(num_waves=3.0, amplitude=0.05)],
    [FlatTerrain.configure()],
    [SlopedTerrain.configure(slope=0.05)],
]
_RUNWAY_START_X = -len(_RUNWAY_TILES) * _TILE_M / 2.0 + 2.0


def _play_config(args):
    """The training config with the source repo's play-mode overrides applied."""
    config = TASK_REGISTRY.build("husky")
    config.env_options.num_envs = 1
    config.env_options.episode_length_s = 1e9
    for name in ("fell_over", "feet_off_board", "illegal_contact"):
        config.termination_options.disable_term(name)
    config.event_options.disable_term("push_robot")
    config.runner_options.logger = "tensorboard"
    # Seed through the config, not a bare set_random_seed(): `from_config` applies
    # runner_options.seed right before it builds the env, so a direct call in play()
    # would just be overwritten by the task's default seed.
    config.runner_options.seed = args.seed
    # Pin the command: fixed board speed, heading held at the board's current
    # heading plus the (constant) relative offset, re-anchored once per cycle.
    config.command_options.skate.lin_vel_x_range = (args.vx, args.vx)
    config.command_options.skate.heading_range = (args.heading, args.heading)

    start_x = 0.0
    if args.terrain == "rough":
        start_x = _RUNWAY_START_X
        robot = config.scene_options.robot.model_copy(
            update={"default_root_pos": (start_x + ROBOT_DEFAULT_ROOT_POS[0], *ROBOT_DEFAULT_ROOT_POS[1:])}
        )
        skateboard = config.scene_options.skateboard.model_copy(
            update={"default_root_pos": (start_x + SKATEBOARD_DEFAULT_ROOT_POS[0], *SKATEBOARD_DEFAULT_ROOT_POS[1:])}
        )
        config.scene_options = SceneOptions(
            terrain=TerrainOptions(
                subterrain_size=(_TILE_M, _TILE_M),
                horizontal_scale=0.1,
                vertical_scale=0.005,
                terrain_generators=_RUNWAY_TILES,
                use_checker_surface=True,
                default_root_pos=(-len(_RUNWAY_TILES) * _TILE_M / 2.0, -_TILE_M / 2.0, 0.0),
            ),
            robot=robot,
            skateboard=skateboard,
        )
        config.command_options.skate.ground_name = "terrain"

    if args.record:
        config.cameras_options = CamerasOptions(
            rec=CameraOptions(
                cam_pos=(start_x + 1.8, -2.6, 1.4),
                cam_lookat=(start_x, 0.0, 0.6),
                cam_fov=45.0,
                cam_resolution=(1280, 720),
                follow_entity_name="robot",
            )
        )
    return config


def play(args):
    """Roll out a trained checkpoint; optionally record an mp4."""
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path="logs/husky")

    config = _play_config(args)
    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True, show_viewer=args.vis)
    runner = make_rsl_rl_runner(env, for_inference=True)
    if args.model is not None:
        load_runner_checkpoint(runner, args.model)
    else:
        en.logger.warning("No --model given: rolling out an UNTRAINED policy (plumbing smoke only).")
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.get_observations()
    if args.record:
        out_path = args.output or os.path.join(os.path.dirname(__file__), f"husky_skate_{args.terrain}.mp4")
        env.unwrapped.cameras["rec"].start_recording(out_path)

    board = env.unwrapped.entities["skateboard"]
    steps = 5 if en.is_smoke() else args.steps
    speeds = []
    with torch.no_grad():
        for _ in tqdm(range(steps)):
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            speeds.append(float(board.get_vel()[0, 0].item()))
            if args.record:
                env.unwrapped.cameras["rec"].render_rgb()

    base = env.unwrapped.entities["robot"].get_pos(relative=True)[0]
    tail = speeds[len(speeds) // 2 :]
    print(
        f"\nFinal base pos: x={base[0]:.3f} y={base[1]:.3f} z={base[2]:.3f}m | "
        f"board forward speed (last half): mean={sum(tail) / len(tail):.2f} m/s (command {args.vx:.2f})"
    )
    if args.record:
        env.unwrapped.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")


def main():
    """Parse the play subcommand and dispatch (training is `eden train --task husky`)."""
    parser = argparse.ArgumentParser(description="HUSKY humanoid skateboarding: play a trained policy.")
    subparsers = parser.add_subparsers(dest="command")

    play_parser = subparsers.add_parser("play", help="Roll out a trained checkpoint")
    play_parser.add_argument("--model", default=None, help="Path to a trained model checkpoint (omit: untrained smoke)")
    play_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    play_parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    play_parser.add_argument("-r", "--record", action="store_true", help="Record an mp4")
    play_parser.add_argument("-o", "--output", default=None, help="Output mp4 path (with --record)")
    play_parser.add_argument("-s", "--steps", type=int, default=1500, help="Number of 50 Hz steps")
    play_parser.add_argument("--vx", type=float, default=1.0, help="Board speed command [m/s] (trained on 0-1.5)")
    play_parser.add_argument("--heading", type=float, default=0.0, help="Relative heading command [rad]")
    play_parser.add_argument("--seed", type=int, default=0, help="RNG seed (reset noise varies the rollout)")
    play_parser.add_argument(
        "--terrain",
        choices=("flat", "rough"),
        default="flat",
        help="Ground: flat plane, or the zero-shot runway (flat -> waves -> slope)",
    )
    args = parser.parse_args()

    if args.command == "play":
        play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
