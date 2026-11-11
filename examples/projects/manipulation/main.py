"""Train / inference entry point for the manipulation tasks.

---
tier: project
capabilities: [task.manipulation, robot.franka]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/manipulation/main.py train --task manipulation_reach_droid
---

Mirrors ``examples/projects/reaching/reacher.py`` but takes a ``--task`` so it drives any
registered manipulation task (default ``manipulation_reach_droid``; also ``manipulation_lift_box`` and the dexterous
``manipulation_dex_lift`` — Franka + XHand1_L lifting a cube to a 3D target).

    # train (then auto-render an inference video of the freshly trained policy)
    uv run python examples/projects/manipulation/main.py train --task manipulation_reach_droid
    uv run python examples/projects/manipulation/main.py train --task manipulation_lift_box
    uv run python examples/projects/manipulation/main.py train --task manipulation_dex_lift

    # render a video from an existing checkpoint
    uv run python examples/projects/manipulation/main.py inference --task manipulation_reach_droid \
        --model logs/manipulation/<run>/model_<it>.pt --record
"""

import argparse
import os

import eden as en  # must be imported before genesis
import genesis as gs
import numpy as np
import torch
from genesis.options.renderers import BatchRenderer
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.options.camera import CameraOptions
from eden.tasks import TASK_REGISTRY
from eden.utils.assets import find_latest_checkpoint

# A camera framing the Franka + table + cube for the recorded inference video.
_REC_CAMERA = CameraOptions(
    cam_pos=(1.2, 1.0, 0.95),
    cam_lookat=(0.3, 0.0, 0.45),
    cam_resolution=(1280, 960),
)


def _log_root(task: str) -> str:
    return f"logs/{task.lower().replace('-', '_')}"


def _require_gpu_for_vision_task(config, args) -> None:
    """A vision task (e.g. ``manipulation_lift_box_rgb``) renders its camera observations through the gs-madrona
    BatchRenderer, which is CUDA-only — reject --cpu up front rather than crashing minutes
    later inside scene.build() on a CPU backend.
    """
    if args.cpu and isinstance(config.renderer_options, BatchRenderer):
        raise SystemExit(
            f"--cpu is unsupported for task '{args.task}': its camera observations render via "
            "the gs-madrona BatchRenderer (CUDA)."
        )


def main(args: argparse.Namespace):
    # Compute nodes have no network, so online wandb init times out and kills
    # training. Default to offline (syncable later); override via WANDB_MODE.
    os.environ.setdefault("WANDB_MODE", "offline")

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=_log_root(args.task),
        performance_mode=True,
    )

    config = TASK_REGISTRY.build(args.task)
    _require_gpu_for_vision_task(config, args)
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
            log_root_path=_log_root(args.task),
            performance_mode=True,
        )

    config = TASK_REGISTRY.build(args.task)
    _require_gpu_for_vision_task(config, args)
    config.env_options.num_envs = 1

    # Curricula reset to their easy initial level at inference: the goal-range
    # curriculum's level 0 collapses the lift target to a single point straight
    # above the cube, so the rollout would exercise a degenerate task the final
    # policy never trains on. Force full difficulty for an honest eval.
    goal_range = getattr(config.curriculum_options, "goal_range", None)
    if goal_range is not None:
        goal_range.level_init = 1.0

    # NOTE: observation corruption (the training-time noise) is disabled
    # automatically by `from_config(eval_mode=True)` below, so inference rolls out
    # the deterministic policy on clean observations (see RLEnvBase eval_mode).

    if args.record:
        # A vision task (e.g. `manipulation_lift_box_rgb`) feeds its wrist cam through the gs-madrona
        # BatchRenderer, which requires every *batched* camera to share one
        # resolution — so a separate high-res `rec` camera can't just join it
        # (crashes scene.build with "too many values to unpack"). Switching the whole
        # scene to the Rasterizer isn't an option either: it breaks the wrist-cam obs
        # term (needs the BatchRenderer's torch output, not the Rasterizer's numpy).
        # Instead mark the rec camera `debug=True`: Genesis renders debug cameras with
        # the plain Rasterizer regardless of the batch renderer, and excludes them
        # from the batch uniform-resolution check (batch_renderer.py: "non-debug
        # cameras"). So we get a full-res third-person video while the wrist cam keeps
        # driving the policy from the BatchRenderer. No-op for non-vision tasks.
        rec_cam = _REC_CAMERA
        if isinstance(config.renderer_options, BatchRenderer):
            rec_cam = rec_cam.model_copy(update={"debug": True})
        config.cameras_options.rec = rec_cam

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

    # Reach exposes its target as the ``reaching_target`` command and has no goal
    # marker entity, so draw a debug sphere at the target. Lift moves a real
    # ``goal_marker`` sphere via the command's debug_vis, so nothing to draw. The
    # debug sphere is only visible in a recorded frame, so only draw it when
    # recording (skip the per-step draw/clear entirely otherwise).
    reach_cmd = None
    if args.record:
        try:
            reach_cmd = env.unwrapped.command_manager.get_command("reaching_target")
        except (KeyError, AttributeError):
            reach_cmd = None

    # For a vision task (a "wrist" camera drives the policy), also record that
    # camera's raw input alongside the third-person video — one inference run
    # yields two videos: third-person + wrist POV. No-op for non-vision tasks.
    cams = env.unwrapped.cameras
    wrist_cam = cams["wrist"] if (args.record and "wrist" in cams) else None
    wrist_frames = []
    _WRIST_UP = 6  # nearest-neighbor upscale so the small wrist image is legible

    def _grab_wrist():
        img = wrist_cam.render_rgb()[0, ..., :3].detach().to("cpu").numpy()  # (H, W, 3)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return np.repeat(np.repeat(img, _WRIST_UP, axis=0), _WRIST_UP, axis=1)

    obs, _ = env.reset()
    if args.record:
        rec = env.unwrapped.cameras["rec"]
        fps = int(round(1.0 / float(env.unwrapped.dt)))
        video_path = args.output or f"{en.log_dir}/{args.task.lower().replace('-', '_')}_inference.mp4"
        rec.start_recording(video_path=video_path, fps=fps)
        if wrist_cam is not None:
            wrist_frames.append(_grab_wrist())

    for _ in tqdm(range(env.unwrapped.max_episode_length - 1)):
        with torch.no_grad():
            raw_action = policy(obs)
        obs, _, _, _ = env.step(raw_action)
        if reach_cmd is not None:
            env.unwrapped.draw_debug_sphere(reach_cmd[..., :3], radius=0.03, color=(0.0, 1.0, 0.0, 0.5))
        if args.record:
            rec.render_rgb()
            if wrist_cam is not None:
                wrist_frames.append(_grab_wrist())
        if reach_cmd is not None:
            env.unwrapped.clear_debug_objects()

    if args.record:
        rec.stop_recording()
        print(f"\nsaved inference video -> {video_path}")
        if wrist_cam is not None:
            import imageio.v2 as imageio

            wrist_path = video_path.rsplit(".", 1)[0] + "_wrist.mp4"
            imageio.mimwrite(wrist_path, wrist_frames, fps=fps, quality=8)
            print(f"saved wrist-camera view -> {wrist_path}")


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--task", default="manipulation_reach_droid", help="registered manipulation task")
    p.add_argument("-c", "--cpu", action="store_true", help="use the CPU backend")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / run manipulation tasks.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a manipulation task.")
    _add_common(train_parser)
    train_parser.add_argument("-b", "--num-envs", type=int, default=4096, help="number of environments")

    inf_parser = subparsers.add_parser("inference", help="Run inference / render a video.")
    _add_common(inf_parser)
    inf_parser.add_argument("--model", type=str, help="path to the model checkpoint .pt file")
    inf_parser.add_argument("-r", "--record", action="store_true", help="record an mp4 video")
    inf_parser.add_argument("-o", "--output", type=str, default=None, help="output mp4 path")

    args = parser.parse_args()

    if args.command == "train":
        main(args)
        # Chain into inference through the real parser. A hand-built Namespace has to
        # restate argparse's dest names, and silently drifts from them — passing
        # the flag spellings (rec=, out=) where inference() reads args.record /
        # args.output kills the promised video on AttributeError after the whole
        # training run has been paid for.
        argv = ["inference", "--task", args.task, "--record"]
        argv += ["--model", find_latest_checkpoint(en.log_dir, return_path=True)]
        if args.cpu:
            argv.append("--cpu")
        inference(parser.parse_args(argv))
    elif args.command == "inference":
        inference(args)
    else:
        parser.print_help()
