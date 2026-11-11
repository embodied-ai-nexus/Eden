"""Run the GMT general motion-tracking policy on a Unitree G1 in Genesis.

---
tier: policy
capabilities: [task.mimic, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/gmt/main.py --vis
---

Loads a kinematic reference pkl (default ``motions/crouchwalk_stand.pkl``) and
rolls out the frozen GMT tracker; ``--debug-vis`` overlays a ghost reference robot.
"""

import os
from argparse import ArgumentParser
from tqdm import tqdm

import genesis as gs

import eden as en
from eden.envs.base import RLEnvBase
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions
from eden.extensions.policies.gmt.policy import GMTPolicy
from eden.options import GHOST_DEFAULTS


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--motion-file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "motions/crouchwalk_stand.pkl"),
        help="Path to the reference motion to track",
    )
    parser.add_argument("-d", "--debug-vis", action="store_true", help="Draw the debug visualization")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")

    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    if args.debug_vis:
        scene_options = en.scenes.Empty(
            robot=GMTPolicy.robot_options(),
            ghost_robot=en.robots.G1Dofs29NoWristDof(**GHOST_DEFAULTS),
        )
    else:
        scene_options = en.scenes.Empty(
            robot=GMTPolicy.robot_options(),
        )

    env = RLEnvBase(
        env_options=EnvOptions(num_envs=1, **GMTPolicy.env_options()),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **GMTPolicy.command_options(motion_file=args.motion_file, debug_vis=args.debug_vis)
        ),
        action_options=ActionManagerOptions(**GMTPolicy.action_options()),
        observation_options=ObservationManagerOptions(**GMTPolicy.observation_options()),
        cameras_options=CamerasOptions(
            rec=CameraOptions(cam_fov=50.0, follow_entity_name="robot"),
        )
        if args.record
        else None,
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()

    env.config.save_as_json()

    policy = GMTPolicy(env=env, options=GMTPolicy.configure())

    cmd = env.command_manager.get_term("motion")

    obs, _ = env.reset()

    if args.record:
        rec = env.cameras["rec"]
        rec.start_recording(args.motion_file.replace(".pkl", ".mp4"))

    for _ in tqdm(range(int(cmd._motion_lib.get_total_length() / env.dt))):
        raw_action = policy.step(obs)
        obs, _, _, _, _ = env.step(raw_action)
        if args.record:
            rec.render_rgb()

    if args.record:
        rec.stop_recording()


if __name__ == "__main__":
    main()
