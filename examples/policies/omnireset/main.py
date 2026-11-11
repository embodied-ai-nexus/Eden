"""Run UWLab's pretrained OmniReset peg-insertion expert in Eden and measure its success rate.
---
tier: policy
capabilities: [task.manipulation, robot.ur5e, robot.robotiq_2f85]
hardware: gpu
optional_deps: []
run: uv run python examples/policies/omnireset/main.py -b 128
---
Builds the ``omnireset`` task as UWLab evaluates it (its scene, observations, operational-space controller and 16 s
episode, every episode starting from a recorded ``ObjectAnywhereEEAnywhere`` state), loads the released state expert
(``peg_state_rl_expert_seed42.pt`` from ``UW-Lab/uwlab-assets``) and runs whole episodes. An episode succeeds as
UWLab grades it: when it ends, the peg sits within 2.5 mm and 0.025 rad (roll + pitch) of its assembled pose in the
hole.

Usage::

    uv run python examples/policies/omnireset/main.py -b 128                   # success rate over 128 episodes
    uv run python examples/policies/omnireset/main.py -b 1 --episodes 4 -r     # video of four episodes

The first run downloads the checkpoint and UWLab's reset states and converts the states once (cached).
"""

from __future__ import annotations

import argparse

import eden as en  # noqa: F401  (import before genesis)
import genesis as gs
import torch

from eden.envs.base import env_class_for_config
from eden.tasks.omnireset.constants import EVAL_FAMILY, INSERTIVE, PAIRS, RECEPTIVE
from eden.tasks.omnireset.custom_terms import OmniResetStates, assembly_errors
from eden.extensions.policies.omnireset.policy import OmniResetPolicy
from eden.options.camera import CameraOptions
from eden.tasks.registry import TASK_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-b", "--num-envs", type=int, default=16, help="Parallel environments (one episode each per round)"
    )
    parser.add_argument("--episodes", type=int, default=1, help="Rounds of episodes to run in every environment")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the reset-state draws")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record an mp4 of environment 0")
    parser.add_argument("-o", "--output", default="omnireset_peg.mp4", help="Video path for --record")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    torch.manual_seed(args.seed)
    cfg = TASK_REGISTRY.build("omnireset")
    # UWLab's evaluation (-Play-v0): every episode from one family, so only that family is converted.
    cfg.event_options.reset_states = OmniResetStates.configure(
        families=[EVAL_FAMILY], category_weights={EVAL_FAMILY: 1.0}
    )
    cfg.env_options.num_envs = args.num_envs
    if args.record:
        if args.num_envs > 1:
            # Parallel envs share one world frame and would render on top of one another.
            parser.error("--record renders every env in one place; pass -b 1")
        # Inside the table frame's posts (the nearest stands at x = 0.91, y = 0.65), facing the robot.
        cfg.cameras_options.camera = CameraOptions(
            cam_pos=(0.82, 0.48, 0.38), cam_lookat=(0.3, 0.0, 0.04), cam_resolution=(1280, 720), cam_fov=58
        )
    env = env_class_for_config(cfg).from_config(cfg, show_viewer=args.vis, eval_mode=True)
    env.build()
    policy = OmniResetPolicy(env=env, options=OmniResetPolicy.configure())
    pair = PAIRS[policy.pair]
    camera = env.cameras["camera"] if args.record else None
    if camera is not None:
        camera.start_recording(video_path=args.output, fps=round(1.0 / env.dt))

    successes, abnormal = [], 0
    with torch.no_grad():
        for episode in range(args.episodes):
            obs, _ = env.reset()
            policy.reset()
            ran_away = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for _ in range(env.max_episode_length):
                pos_error, tilt_error = assembly_errors(env, pair, INSERTIVE, RECEPTIVE)
                obs, _, terminated, timeouts, _ = env.step(policy.step(obs))
                if camera is not None:
                    camera.render_rgb()
                ran_away |= terminated  # the env auto-resets and keeps going; its episode still failed
                done = terminated | timeouts
                if done.all():
                    break
            # Graded on the last state before the episode's reset (UWLab grades at reset time).
            success = (pos_error < pair.success_position) & (tilt_error < pair.success_orientation) & ~ran_away
            abnormal += int(ran_away.sum())
            successes.append(success.cpu())
            print(
                f"episode round {episode}: success {success.float().mean():.3f} "
                f"(median position error {pos_error.median() * 1e3:.1f} mm, tilt {tilt_error.median():.3f} rad)"
            )
    rate = torch.cat(successes).float().mean().item()
    print(f"success rate over {args.num_envs * args.episodes} episodes: {rate:.3f}; abnormal-robot ends: {abnormal}")
    if camera is not None:
        camera.stop_recording()
        print(f"video: {args.output}")


if __name__ == "__main__":
    main()
