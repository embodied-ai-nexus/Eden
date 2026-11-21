r"""
Benchmark a SuperMimic checkpoint: play every clip once from its first frame and count the falls.

---
tier: project
capabilities: [task.mimic, robot.g1, dataset.lafan1]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/mimic/supermimic/evaluate.py --checkpoint logs/supermimic/<run>/model_20000.pt
---

Each clip gets one env, which starts on the clip's first frame without reset noise and runs until the clip ends
(a success) or the robot falls (a termination). The script reports the failures and two tracking errors averaged
over each clip's steps: the mean per-body position error of the tracked bodies relative to the root (MPJPE, mm) and
the mean joint angle error (rad). Randomization of friction, centre of mass and encoder bias stays as in training.

Usage::

    python examples/projects/mimic/supermimic/evaluate.py --checkpoint logs/supermimic/<run>/model_20000.pt
    python examples/projects/mimic/supermimic/evaluate.py --checkpoint model.pt --dataset bones_seed \\
        --raw-data-path /data/bones_seed/g1/csv --processed-data-path /data/bones_seed_g1.pt \\
        --clip-list bones_seed_2000_clips.json -b 2000 -o bones_seed_benchmark.json
"""

from __future__ import annotations

import argparse
import json

import eden as en  # must be imported before genesis to isolate GPUs for DDP
import genesis as gs
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n---")[0].strip())
    parser.add_argument("--checkpoint", type=str, required=True, help="A SuperMimic checkpoint (model_<it>.pt).")
    parser.add_argument("--dataset", type=str, default="lafan1", help="Motion library: lafan1 or bones_seed.")
    parser.add_argument("--raw-data-path", type=str, default=None, help="The library's CSV clips.")
    parser.add_argument("--processed-data-path", type=str, default=None, help="Cache of the prepared library.")
    parser.add_argument("--clip-list", type=str, default=None, help="Clips to play (.json/.yaml list).")
    parser.add_argument("-b", "--num-envs", type=int, default=64, help="Envs, at least one per clip.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Per-clip results (.json).")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU.")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, performance_mode=True)
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
    from eden.tasks import TASK_REGISTRY
    from eden.tasks.supermimic.config import TRACKED_LINKS

    cfg = TASK_REGISTRY.build(
        "supermimic",
        dataset=args.dataset,
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
        clip_list=args.clip_list,
    )
    rsi = cfg.event_options.reset_from_motion
    rsi.pose_range, rsi.velocity_range, rsi.dofs_position_range = {}, {}, (0.0, 0.0)
    cfg.motion_options.default.reset_sampling = "clip_start"
    cfg.env_options.episode_length_s = 1e6
    cfg.runner_options.return_sampler.enabled = False
    cfg.runner_options.logger = "tensorboard"

    cfg.env_options.num_envs = args.num_envs
    env = RslRlVecEnvWrapper.from_config(cfg, eval_mode=True)
    motion = env.unwrapped.motion_manager.get_term("default")
    num_clips = motion.library.starts.numel()
    if num_clips > env.num_envs:
        raise SystemExit(f"{num_clips} clips need as many envs: pass -b {num_clips}, or split --clip-list.")
    runner = make_rsl_rl_runner(env, checkpoint=args.checkpoint, for_inference=True)
    results = _play(env, runner.get_inference_policy(), motion, TRACKED_LINKS, num_clips)

    failed = [r for r in results if r["failed"]]
    summary = {
        "clips": len(results),
        "failed": len(failed),
        "success_rate": 1.0 - len(failed) / len(results),
        "mpjpe_mm": sum(r["mpjpe_mm"] for r in results) / len(results),
        "joint_error_rad": sum(r["joint_error_rad"] for r in results) / len(results),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "clips": results}, f, indent=1)


@torch.inference_mode()
def _play(env, policy, motion, links_name: list[str], num_clips: int) -> list[dict]:
    """Run envs ``0 .. num_clips - 1`` until each ends its clip or falls once."""
    base = env.unwrapped
    robot = base.entities["robot"]
    _, links = robot.find_named_links_idx_local(links_name)
    lengths = (motion.library.ends - motion.library.starts)[:num_clips]
    obs, _ = env.reset()
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active[:num_clips] = True
    failed = torch.zeros_like(active)
    pos_error = torch.zeros(env.num_envs, device=env.device)
    joint_error = torch.zeros_like(pos_error)
    steps = torch.zeros_like(pos_error)
    for _ in range(int(lengths.max())):
        obs, _, dones, extras = env.step(policy(obs))
        ref = base.motion_manager.data["robot"]
        done = dones.bool() & active
        graded = active & ~done
        robot_pos = robot.get_links_pos(ls_idx_local=links)
        ref_pos = ref.links_pos_w[:, 0]
        error = ((robot_pos - robot_pos[:, :1]) - (ref_pos - ref_pos[:, :1])).norm(dim=-1).mean(-1)
        pos_error += torch.where(graded, error, 0.0)
        joint_error += torch.where(graded, (robot.get_dofs_pos() - ref.dofs_pos[:, 0]).abs().mean(-1), 0.0)
        steps += graded.float()
        failed |= done & ~extras["time_outs"].bool()
        active &= ~done
        if not active.any():
            break
    names = base.motion_manager.get_term("default").data_source.all_motion_files
    steps = steps.clamp(min=1)
    return [
        {
            "clip": names[i] if i < len(names) else str(i),
            "failed": bool(failed[i]),
            "mpjpe_mm": float(1000 * pos_error[i] / steps[i]),
            "joint_error_rad": float(joint_error[i] / steps[i]),
        }
        for i in range(num_clips)
    ]


if __name__ == "__main__":
    main()
