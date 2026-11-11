"""Keyboard-teleop a WidowX-250 across the SIMPLER Bridge manipulation tasks.

---
tier: project
capabilities: [task.manipulation, dataset.bridge, robot.wx250s, sensor.camera, teleop.keyboard]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/manipulation/simpler/main.py --vis
---

Usage:
    python examples/projects/manipulation/simpler/main.py --task simpler_put_carrot_on_plate --vis
    python examples/projects/manipulation/simpler/main.py --task simpler_put_eggplant_in_basket --cpu
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from eden.utils.devices.keyboard import SE3Keyboard

SIMPLER_TASKS = [
    "simpler_put_carrot_on_plate",
    "simpler_put_spoon_on_table_cloth",
    "simpler_stack_green_cube_on_yellow_cube",
    "simpler_put_eggplant_in_basket",
]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="simpler_stack_green_cube_on_yellow_cube", choices=SIMPLER_TASKS, help="Task to run"
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")

    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    cfg = TASK_REGISTRY.build(args.task)
    cfg.env_options.enable_default_keybinds = False

    env = RLEnvBase.from_config(cfg, show_viewer=args.vis, eval_mode=True)
    env.build()
    env.reset()

    keyboard = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())
    keyboard.add_callback("x", lambda on_press: env.reset() if on_press else None)

    render_camera = env.cameras["camera"]
    render_camera.snapshot()

    while True:
        if args.vis and env.scene._visualizer.has_display:
            # Postpone import of OpenCV at runtime to reduce hard system dependencies
            import cv2

            rgb = render_camera.render_rgb()
            if rgb.ndim == 4:
                rgb = rgb[0]
            cv2.imshow("Genesis - Camera", rgb[..., ::-1])
            cv2.waitKey(1)

        action = keyboard.get_command()
        action = {
            "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            "gripper_controller": action[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
        }
        env.step(action)

        if env.metric_manager.success_buf:
            print("Success!")
            env.reset()


if __name__ == "__main__":
    main()
