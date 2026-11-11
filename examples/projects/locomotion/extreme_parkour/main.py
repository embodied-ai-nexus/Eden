"""Extreme-parkour playback (teacher or distilled depth student) for Unitree Go2, with the viewer.

---
tier: project
capabilities: [task.locomotion, robot.go2, terrain.procedural, sensor.camera]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/locomotion/extreme_parkour/main.py play
---

Both phases of "Extreme Parkour with Legged Robots" (Cheng et al. 2023) train through
``eden train`` — the task configs select the RMA stack in ``eden.learning.wrappers.rsl_rl.rma``:

    # Teacher: RMA actor + estimator, PPO with priv-latent regularization, DAgger, mirror loss
    eden train --task extreme_parkour_teacher --logger wandb
    # Student: DAgger depth distillation from a teacher checkpoint (a student checkpoint resumes)
    eden train --task extreme_parkour_student --checkpoint logs/extreme_parkour_teacher/<run>/model_<it>.pt

``eden inference --task <task> --checkpoint ...`` replays either. This script adds the play-mode
setup the source uses — the physics randomization and pushes off, the viewer on:

    uv run python examples/projects/locomotion/extreme_parkour/main.py play --model logs/extreme_parkour_teacher/<run>/model_<it>.pt
    uv run python examples/projects/locomotion/extreme_parkour/main.py play --student --model logs/extreme_parkour_student/<run>/model_<it>.pt
"""

import argparse

import eden as en  # must be imported before genesis/torch to isolate GPUs for DDP
import genesis as gs
import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.tasks import TASK_REGISTRY

TEACHER_TASK = "extreme_parkour_teacher"
STUDENT_TASK = "extreme_parkour_student"

# Event terms that randomize physics or shove the robot — disabled for play. The student's
# camera-mount randomization stays on (the reference keeps it at eval time too).
_PLAY_DISABLED_EVENTS = ("push_robot", "randomize_friction", "randomize_base_mass", "randomize_com")


def play(args):
    task = STUDENT_TASK if args.student else TEACHER_TASK
    if args.student and args.cpu:
        # The student renders depth via gs-madrona (CUDA-only); fail here, not minutes into the build.
        raise SystemExit("--cpu is unsupported for the student: its depth camera renders via gs-madrona (CUDA).")
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path=f"logs/{task}")

    config = TASK_REGISTRY.build(task)
    config.env_options.num_envs = args.num_envs
    config.runner_options.logger = "tensorboard"
    for term_name in _PLAY_DISABLED_EVENTS:
        config.event_options.disable_term(term_name)
    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True, show_viewer=not en.is_smoke())

    if args.model is None:
        if args.student:
            raise SystemExit("--student needs --model <a student checkpoint>: the depth student cannot run untrained.")
        en.logger.warning("No --model given: rolling out an UNTRAINED teacher (plumbing smoke only).")
    runner = make_rsl_rl_runner(env, checkpoint=args.model, for_inference=True)
    policy = runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    steps = 5 if en.is_smoke() else args.steps
    with torch.no_grad():
        for _ in range(steps):
            obs, _rews, _dones, _infos = env.step(policy(obs))


def main():
    parser = argparse.ArgumentParser(description="Extreme parkour: play a trained teacher or depth student.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    play_parser = subparsers.add_parser("play", help="Run a trained teacher or student with the viewer")
    play_parser.add_argument("--model", type=str, default=None, help="Checkpoint (omit for an untrained teacher smoke)")
    play_parser.add_argument("--student", action="store_true", help="Play a distilled depth student")
    play_parser.add_argument("--cpu", action="store_true", help="Use CPU backend instead of GPU")
    play_parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    play_parser.add_argument("--steps", type=int, default=10_000, help="Steps to run")

    args = parser.parse_args()
    if args.command == "play":
        play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
