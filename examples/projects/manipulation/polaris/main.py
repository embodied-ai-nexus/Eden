"""Keyboard-teleop a Droid arm across the Polaris Gaussian-splat manipulation scenes.

---
tier: project
capabilities: [task.manipulation, dataset.droid, render.splat, sensor.camera, teleop.keyboard]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/manipulation/polaris/main.py --task polaris_block_stack_kitchen
---

Usage:
    python examples/projects/manipulation/polaris/main.py --task polaris_block_stack_kitchen --vis
    python examples/projects/manipulation/polaris/main.py --task polaris_food_bussing --cpu
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.envs.base import env_class_for_config
from eden.tasks import TASK_REGISTRY
from eden.utils.devices.keyboard import SE3Keyboard


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="polaris_block_stack_kitchen",
        choices=[
            "polaris_block_stack_kitchen",
            "polaris_food_bussing",
            "polaris_pan_clean",
            "polaris_move_latte_cup",
            "polaris_organize_tools",
            "polaris_tape_into_container",
        ],
        help="Task to run",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    # The --task choices are the registry names, so the registry is the lookup: it imports only
    # the chosen task's module, the same laziness the old per-case imports bought by hand.
    cfg = TASK_REGISTRY.build(args.task)
    cfg.env_options.enable_default_keybinds = False

    env = env_class_for_config(cfg).from_config(cfg, show_viewer=args.vis, eval_mode=True)
    env.build()
    env.reset()

    keyboard = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())
    keyboard.add_callback("x", lambda on_press: env.reset() if on_press else None)

    # env.draw_debug_points(env.splat_debug_points())

    external_camera = env.cameras["external_camera"]
    wrist_camera = env.cameras["wrist_camera"]

    if args.record:
        external_camera.start_recording(f"out/{args.task}_external_camera.mp4")
        wrist_camera.start_recording(f"out/{args.task}_wrist_camera.mp4")

    # `max_episode_length` is an RLEnvBase attribute; SplatEnvBase is not an RL env.
    for _ in range(int(cfg.env_options.episode_length_s / env.dt)):
        if args.vis and env.scene._visualizer.has_display:
            # Postpone import of OpenCV at runtime to reduce hard system dependencies
            import cv2

            rgb = external_camera.render_rgb()
            if rgb.ndim == 4:
                rgb = rgb[0]
            cv2.imshow("Eden - External Camera", rgb[..., ::-1])
            cv2.waitKey(1)

            rgb = wrist_camera.render_rgb()
            if rgb.ndim == 4:
                rgb = rgb[0]
            cv2.imshow("Eden - Wrist Camera", rgb[..., ::-1])
            cv2.waitKey(1)

        action = keyboard.get_command()
        action = {
            "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            "gripper_controller": action[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
        }
        env.step(action)

    if args.record:
        external_camera.stop_recording()
        wrist_camera.stop_recording()


if __name__ == "__main__":
    main()
