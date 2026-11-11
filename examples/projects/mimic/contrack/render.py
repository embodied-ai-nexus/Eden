"""ConTrack — render a trained checkpoint to an inference video.

---
tier: project
capabilities: [task.mimic, robot.xarm7_xhand]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/mimic/contrack/render.py --data examples/projects/mimic/contrack/sample/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5 --auto-start
---

``eden inference --task contrack --checkpoint ...`` replays a checkpoint through
the task's own runner (``ContrackOnPolicyRunner``, the fork's runner). This
renderer adds what a ConTrack replay needs on top: start-phase selection, the
object-break termination relaxed for a full-clip view, and a single-env mp4.

    python examples/projects/mimic/contrack/render.py \
      --data examples/projects/mimic/contrack/sample/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5 \
      --model logs/contrack/<run>/model_<it>.pt --auto-start

Start-phase control (the policy may track a mid-clip phase cleanly yet fail from
frame 0, where a kinematic reset can't establish the grasp):

* ``--auto-start`` (recommended): probe several start frames and render from the
  one with the longest clean (termination-free) run.
* ``--start-frame N``: force a specific start frame.
* default: pin frame 0.

``--model`` defaults to the newest checkpoint under ``logs/contrack``.
"""

from __future__ import annotations

import argparse
import os

import eden as en  # must precede genesis/torch so DDP GPU isolation runs first
import genesis as gs
import torch
from tqdm import tqdm

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.options.camera import CameraOptions, CamerasOptions
from eden.tasks import TASK_REGISTRY
from eden.utils.assets import find_latest_checkpoint


def _set_start(mt, start_frame: int) -> None:
    """Force the motion term to start every reset at ``start_frame`` (0 == clip start)."""
    mt.adaptive_sampling = False
    mt.motion_bin_length_s = None
    mt._bin_sampler = None  # force the frame-0 / middle-range path (not bin sampling)
    if start_frame <= 0:
        mt.start_at_timestep_zero_prob = 1.0
        mt.motion_start_from_middle_range = (0.0, 0.0)
    else:
        clip = int(mt.data_source.buffer_length[0].item())
        frac = start_frame / clip
        mt.start_at_timestep_zero_prob = 0.0
        mt.motion_start_from_middle_range = (frac, frac)


def _rollout_survival(env, tm, policy, max_steps: int) -> tuple[int, bool]:
    """Reset and roll out; return (steps survived before a done, reached_clip_end)."""
    obs, _ = env.reset()
    for step in range(max_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, dones, _ = env.step(action)
        if bool(dones.reshape(-1)[0].item()):
            return step, bool(tm.get_term_dones("motion_ends")[0].item())
    return max_steps, True


def render(args: argparse.Namespace) -> None:
    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path="logs/contrack")

    config = TASK_REGISTRY.build("contrack", data=args.data)
    config.env_options.num_envs = 1

    control_dt = config.env_options.sim_dt * config.env_options.decimation
    clip_steps = int(round((config.env_options.episode_length_s - 0.1) / control_dt))
    mopt = config.motion_options.default

    # --auto-start / --early-termination sweep needs object_pose_break ACTIVE to measure
    # a clean run; the static modes disable it so a partial policy still shows the full clip.
    disable_term = not args.early_termination and not args.auto_start
    if disable_term:
        ob = getattr(config.termination_options, "object_pose_break", None)
        if ob is not None:
            ob.pos_threshold = 1e9
            ob.rot_threshold = 1e9
        config.curriculum_options = type(config.curriculum_options)()

    if not args.auto_start:
        # Static start selection, applied to the config before build.
        if args.start_frame is not None:
            mopt.adaptive_sampling = False
            mopt.motion_bin_length_s = None
            mopt.start_at_timestep_zero_prob = 0.0
            frac = args.start_frame / max(1, clip_steps)
            mopt.motion_start_from_middle_range = (frac, frac)
        elif args.start_frame_zero:
            mopt.start_at_timestep_zero_prob = 1.0

    # No episode-length truncation; the roll-out ends at the clip's own end (motion_ends)
    # or a break, and stop-on-done cuts before recording the post-reset frame-0 snap.
    config.env_options.episode_length_s = 1e9
    config.runner_options.logger = "tensorboard"

    if config.cameras_options is None:
        config.cameras_options = CamerasOptions()
    config.cameras_options.rec = CameraOptions(
        cam_pos=tuple(args.cam_pos), cam_lookat=tuple(args.cam_lookat), cam_resolution=tuple(args.resolution)
    )

    env = RslRlVecEnvWrapper.from_config(config, show_viewer=args.vis, eval_mode=True)

    model = args.model or find_latest_checkpoint(en.log_dir, return_path=True)
    if model is None:
        raise SystemExit("No checkpoint found under logs/contrack; pass --model <path>.")
    en.logger.info(f"Rendering checkpoint: {model}")

    runner = make_rsl_rl_runner(env, checkpoint=model, for_inference=True)
    policy = runner.get_inference_policy(device=env.device)

    mt = env.unwrapped.motion_manager._terms["default"]
    tm = env.unwrapped.termination_manager
    clip = int(mt.data_source.buffer_length[0].item())

    start_frame = 0 if args.start_frame is None else args.start_frame
    if args.auto_start:
        # Probe evenly-spaced start frames; keep the one with the longest clean run.
        n = max(2, args.auto_start_candidates)
        cands = sorted({int(round(i * (clip - 1) / (n - 1))) for i in range(n)})
        best = None  # (survived, -start), start
        for s in cands:
            _set_start(mt, s)
            surv, reached = _rollout_survival(env, tm, policy, clip - s + 2)
            en.logger.info(f"auto-start probe: start={s:3d} survived={surv:3d} reached_end={reached}")
            key = (surv, -s)
            if best is None or key > best[0]:
                best = (key, s)
        start_frame = best[1]
        en.logger.info(f"auto-start chose frame {start_frame} (survives {best[0][0]} steps)")
        _set_start(mt, start_frame)

    out_path = args.output or os.path.join(en.log_dir, "contrack_inference.mp4")
    steps = args.steps if args.steps is not None else max(1, clip - start_frame - 1)
    fps = args.fps if args.fps is not None else int(round(1.0 / control_dt))
    en.logger.info(f"Recording <={steps} steps @ {fps} fps from frame {start_frame} (clip={clip})")

    obs, _ = env.reset()
    rec = env.unwrapped.cameras["rec"]
    rec.start_recording(out_path, fps=fps)
    for _ in tqdm(range(steps)):
        rec.render_rgb()  # record the current (pre-step) state
        with torch.no_grad():
            action = policy(obs)
        obs, _, dones, _ = env.step(action)
        # env auto-resets inside step(); stop before recording the post-reset frame-0 snap.
        if bool(dones.reshape(-1)[0].item()):
            break
    rec.stop_recording()

    # A single-env batched camera writes "<stem>_0.mp4".
    stem, ext = os.path.splitext(os.path.abspath(out_path))
    final = f"{stem}_0{ext}" if os.path.exists(f"{stem}_0{ext}") else os.path.abspath(out_path)
    print(f"VIDEO_PATH {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained ConTrack checkpoint to an inference video.")
    parser.add_argument("--data", required=True, help="Path to the ConTrack reference clip (.h5) used for training.")
    parser.add_argument("--model", default=None, help="Checkpoint .pt (default: newest under logs/contrack).")
    parser.add_argument("--output", default=None, help="Output mp4 path (default: <log_dir>/contrack_inference.mp4).")
    parser.add_argument("--steps", type=int, default=None, help="Control steps to record (default: to clip end).")
    parser.add_argument("--fps", type=int, default=None, help="Video framerate (default: real-time control rate).")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Probe start frames and render from the one with the longest clean run (recommended).",
    )
    parser.add_argument("--auto-start-candidates", type=int, default=12, help="Number of start frames to probe.")
    parser.add_argument(
        "--no-start-frame-zero",
        dest="start_frame_zero",
        action="store_false",
        help="Do not pin the reference to frame 0 (use the task's adaptive start-bin sampling).",
    )
    parser.add_argument(
        "--start-frame", type=int, default=None, help="Start every reset at this reference frame (static)."
    )
    parser.add_argument(
        "--early-termination",
        action="store_true",
        help="Keep object_pose_break active (roll-out ends at the first break). Default: disabled for a full pass.",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 720), help="Frame width height.")
    parser.add_argument("--cam-pos", type=float, nargs=3, default=(1.0, 0.9, 0.8), help="Camera position.")
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=(0.3, 0.0, 0.35), help="Camera look-at point.")
    parser.add_argument("--vis", action="store_true", help="Show the viewer.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU backend.")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
