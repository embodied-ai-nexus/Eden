"""Render the wrist-camera view (the policy's actual RGB input) for a vision manipulation task.

---
tier: recipe
capabilities: [sensor.camera, robot.franka]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/rendering/render_wrist.py --task manipulation_lift_box_rgb --model logs/manipulation_lift_box_rgb/model.pt
---

Unlike ``main.py inference --record`` (a third-person debug camera), this dumps the
exact 128x72 wrist image the ``rgb_image`` obs term feeds the actor each step,
upscaled with nearest-neighbor so individual pixels are legible.

    uv run python examples/recipes/rendering/render_wrist.py --task manipulation_lift_box_rgb \
        --model logs/manipulation_lift_box_rgb/26Jul02T160310/model_2999.pt \
        --output logs/manipulation_lift_box_rgb/26Jul02T160310/wrist_view.mp4
"""

import argparse
import os

import eden as en  # must be imported before genesis
import genesis as gs
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.tasks import TASK_REGISTRY


def main(args):
    os.environ.setdefault("WANDB_MODE", "offline")
    en.init(backend=gs.gpu, log_root_path=f"logs/{args.task.lower()}", performance_mode=True)

    config = TASK_REGISTRY.build(args.task)
    config.env_options.num_envs = 1
    agent_config = config.runner_options
    agent_config.logger = "tensorboard"
    config.save_as_json()

    env = RslRlVecEnvWrapper.from_config(config, eval_mode=True)
    runner = OnPolicyRunner(env, env.runner_dict, log_dir=en.log_dir, device=env.runner_dict["device"])
    runner.load(args.model)
    policy = runner.get_inference_policy(device=env.device)

    frames = []
    up = args.upscale

    def grab():
        rgb = env.unwrapped.cameras[args.camera].render_rgb()  # (1, H, W, C) tensor, RGBA uint8
        img = rgb[0, ..., :3]
        img = img.detach().to("cpu").numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        # nearest-neighbor upscale so the tiny 128x72 pixels are visible
        img = np.repeat(np.repeat(img, up, axis=0), up, axis=1)
        return img

    obs, _ = env.reset()
    frames.append(grab())
    for _ in tqdm(range(env.unwrapped.max_episode_length - 1)):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env.step(action)
        frames.append(grab())

    import imageio.v2 as imageio

    fps = int(round(1.0 / float(env.unwrapped.dt)))
    out = args.output or f"{en.log_dir}/{args.camera}_view.mp4"
    imageio.mimwrite(out, frames, fps=fps, quality=8)
    print(
        f"\nsaved {args.camera}-camera view ({frames[0].shape[1]}x{frames[0].shape[0]}, {len(frames)} frames) -> {out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="manipulation_lift_box_rgb", help="Task to run")
    parser.add_argument("--model", required=True, help="Path to the policy checkpoint to load")
    parser.add_argument("--camera", default="wrist", help="camera name in cameras_options")
    parser.add_argument("--upscale", type=int, default=6, help="nearest-neighbor upscale factor")
    parser.add_argument("-o", "--output", default=None, help="Path to write the result to")
    main(parser.parse_args())
