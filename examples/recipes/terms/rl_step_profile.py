"""Profile the per-term cost of the RL step loop for any registered task.

---
tier: recipe
capabilities: [term.profiling]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/terms/rl_step_profile.py --task reacher
---

Measures how much wall time
every observation / reward / termination / command / event / curriculum /
metric / action *term* takes inside ``env.step()``, and how much of the step is
Eden framework overhead versus the raw Genesis physics tick.

The env is driven with random actions (no policy / no training), so the numbers
reflect the simulation + manager dispatch cost in isolation.

Usage
-----
::

    # Default: profile the reacher task with 4096 envs.
    uv run python examples/recipes/terms/rl_step_profile.py --task reacher

    # Locomotion task, more envs, dump machine-readable results.
    uv run python examples/recipes/terms/rl_step_profile.py \
        --task move_forward --num-envs 8192 --steps 400 --json reacher_prof.json

    # CPU-bound dispatch view (no CUDA sync): isolates per-term Python overhead.
    uv run python examples/recipes/terms/rl_step_profile.py --task reacher --no-sync

    # Show only the 15 slowest terms.
    uv run python examples/recipes/terms/rl_step_profile.py --task reacher --top 15
"""

import argparse

import genesis as gs
import torch
from tqdm import tqdm

import eden as en
from eden.envs.base import RLEnvBase, env_class_for_config
from eden.tasks import TASK_REGISTRY
from eden.utils.profiling import RLStepProfiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-term RL step-loop profiler.")
    parser.add_argument("--task", type=str, default="reacher", help="Registered task name (see eden.tasks).")
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments.")
    parser.add_argument("-s", "--steps", type=int, default=300, help="Measured steps (after warmup).")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps discarded before measuring.")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use the CPU backend.")
    parser.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        help="Skip CUDA sync per term (measures Python dispatch cost instead of true wall time).",
    )
    parser.add_argument("--top", type=int, default=None, help="Only show the N slowest terms.")
    parser.add_argument("--sort-by", choices=["total", "mean"], default="total", help="Term sort key.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to dump JSON results.")
    parser.add_argument("--list", action="store_true", help="List available tasks and exit.")
    return parser


def main(args: argparse.Namespace) -> None:
    if args.list:
        print("Available tasks:")
        for name in sorted(TASK_REGISTRY.list_tasks()):
            print(f"  {name}")
        return

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/benchmark/{args.task}",
        performance_mode=True,
    )

    cfg = TASK_REGISTRY.build(args.task)
    cfg.env_options.num_envs = args.num_envs
    if args.cpu:
        cfg.runner_options.device = "cpu"

    # Dispatched, not hard-coded: `--task` names any registered task, and a motion-imitation one
    # is an `EdenMimicConfig` that `RLEnvBase` would build without its motion reference.
    env: RLEnvBase = env_class_for_config(cfg).from_config(cfg)
    env.build()  # builds the scene and loads managers (from_config does not build)

    n_dup = len(set(env.observation_manager._dup_sig.values()))
    print(f"[obs-cache] {n_dup} duplicated observation term(s) detected across groups (cached once per step).")

    action_dim = env.action_manager.total_action_dim

    def sample_action() -> torch.Tensor:
        # Random actions in roughly the policy output range; the exact values
        # don't matter for timing, only that every term runs each step.
        return torch.randn(env.num_envs, action_dim, device=env.device)

    env.reset()

    profiler = RLStepProfiler(env, sync=args.sync)
    profiler.attach()

    # Warmup: JIT/autotune kernels, allocate buffers, settle caches.
    for _ in tqdm(range(args.warmup), desc="warmup", leave=False):
        env.step(sample_action())

    profiler.reset()  # discard warmup, keep instrumentation

    for _ in tqdm(range(args.steps), desc="profiling", leave=False):
        with profiler.step():
            env.step(sample_action())

    profiler.detach()

    print(profiler.report(top=args.top, sort_by=args.sort_by))

    if args.json:
        profiler.save_json(args.json)
        print(f"Wrote JSON results to {args.json}")


if __name__ == "__main__":
    main(build_parser().parse_args())
