"""Drive the HUSKY skateboarding policy on a Unitree G1 + skateboard in Genesis.

---
tier: policy
capabilities: [task.locomotion, robot.g1]
hardware: gpu
optional_deps: [onnx]
run: uv run python examples/policies/husky/main.py
---

Builds a G1-on-skateboard environment and rolls out the frozen HUSKY ONNX
policy (`TeleHuman/humanoid_skateboarding <https://github.com/TeleHuman/humanoid_skateboarding>`_,
RSS 2026). The robot pushes with its right leg, rides with both feet on the
deck, and steers by tilting the board — one push/steer cycle every 6 s.

``--terrain rough`` swaps the flat plane for a 24 m runway (flat, then gentle
waves, then a slope) to reproduce the repo's zero-shot terrain evaluation.

``HuskyPolicy`` defaults its checkpoint to the hub path
``husky/skater_flat_g1.onnx`` (Kashu7100/eden_policies); ``--model`` points at
a local ONNX instead (e.g. the source repo's ``ckpts/test.onnx``).

Usage::

    python examples/policies/husky/main.py --vis --vx 1.0
    python examples/policies/husky/main.py --record --terrain rough
    python examples/policies/husky/main.py --steps 50      # plumbing smoke test
"""

from __future__ import annotations

import argparse
import os

import genesis as gs
import torch
from tqdm import tqdm

import eden as en
from eden.envs.base import RLEnvBase
from eden.extensions.policies.husky.constants import ROBOT_DEFAULT_ROOT_POS, SKATEBOARD_DEFAULT_ROOT_POS
from eden.extensions.policies.husky.policy import HuskyPolicy
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import SceneOptions, TerrainOptions
from eden.utils.terrains import FlatTerrain, SlopedTerrain, WaveTerrain

# Zero-shot terrain runway: 4 tiles of 8 m along +x (flat -> waves -> flat ->
# up-ramp), centered on the origin. Tile heights are edge-continuous (integer
# wave count, slope starting at 0) and the ramp comes last so the runway has no
# cliffs. The run starts 2 m into the first flat tile and ends climbing the
# ramp, like the source repo's terrain demo.
_TILE_M = 8.0
_RUNWAY_TILES = [
    [FlatTerrain.configure()],
    [WaveTerrain.configure(num_waves=3.0, amplitude=0.05)],
    [FlatTerrain.configure()],
    [SlopedTerrain.configure(slope=0.05)],
]
_RUNWAY_START_X = -len(_RUNWAY_TILES) * _TILE_M / 2.0 + 2.0


def _scene_options(terrain: str) -> SceneOptions:
    """Flat plane or the zero-shot terrain runway, with robot + board at the start."""
    if terrain == "flat":
        return en.scenes.Empty(
            robot=HuskyPolicy.robot_options(),
            skateboard=HuskyPolicy.skateboard_options(),
        )
    board_x, board_y, board_z = SKATEBOARD_DEFAULT_ROOT_POS
    robot_x, robot_y, robot_z = ROBOT_DEFAULT_ROOT_POS
    return SceneOptions(
        terrain=TerrainOptions(
            subterrain_size=(_TILE_M, _TILE_M),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            terrain_generators=_RUNWAY_TILES,
            use_checker_surface=True,
            default_root_pos=(-len(_RUNWAY_TILES) * _TILE_M / 2.0, -_TILE_M / 2.0, 0.0),
        ),
        robot=HuskyPolicy.robot_options().model_copy(
            update={"default_root_pos": (_RUNWAY_START_X + robot_x, robot_y, robot_z)}
        ),
        skateboard=HuskyPolicy.skateboard_options().model_copy(
            update={"default_root_pos": (_RUNWAY_START_X + board_x, board_y, board_z)}
        ),
    )


def main() -> None:
    """Parse args, build the env + policy, and roll out the skateboarding policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of 50 Hz steps to run")
    parser.add_argument("--vx", type=float, default=1.0, help="Board speed command [m/s] (trained on 0-1.5)")
    parser.add_argument("--heading", type=float, default=0.0, help="Relative heading command [rad] (trained on +-pi/4)")
    parser.add_argument(
        "--terrain",
        choices=("flat", "rough"),
        default="flat",
        help="Ground: flat plane, or the zero-shot terrain runway (flat -> waves -> slope)",
    )
    parser.add_argument("--model", default=None, help="Local ONNX checkpoint to use instead of the hub download")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    if args.model is not None:
        HuskyPolicy.use_local_checkpoint(args.model)

    start_x = 0.0 if args.terrain == "flat" else _RUNWAY_START_X
    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HuskyPolicy.env_options(),
        ),
        scene_options=_scene_options(args.terrain),
        action_options=ActionManagerOptions(**HuskyPolicy.action_options()),
        command_options=CommandManagerOptions(**HuskyPolicy.command_options()),
        observation_options=ObservationManagerOptions(**HuskyPolicy.observation_options()),
        cameras_options=(
            # The follow camera keeps its initial offset to the robot, so both
            # pos and lookat must be anchored at the robot's start (the rough
            # runway starts at x = _RUNWAY_START_X, not the origin).
            CamerasOptions(
                rec=CameraOptions(
                    cam_pos=(start_x + 1.8, -2.6, 1.4),
                    cam_lookat=(start_x, 0.0, 0.6),
                    cam_fov=45.0,
                    cam_resolution=(1280, 720),
                    follow_entity_name="robot",
                )
            )
            if args.record
            else None
        ),
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()

    policy = HuskyPolicy(env=env, options=HuskyPolicy.configure())
    policy.set_command(torch.tensor([args.vx, args.heading]))

    obs, _ = env.reset()
    policy.reset()

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), f"husky_run_{args.terrain}.mp4")
        env.cameras["rec"].start_recording(out_path)

    steps = 5 if en.is_smoke() else args.steps
    board = env.entities["skateboard"]
    speeds = []
    for _ in tqdm(range(steps)):
        action = policy.step(obs)
        obs, _, _, _, _ = env.step(action)
        speeds.append(float(board.get_vel()[0, 0].item()))
        if args.record:
            env.cameras["rec"].render_rgb()

    base = env.entities["robot"].get_pos(relative=True)[0]
    tail = speeds[len(speeds) // 2 :]
    print(
        f"\nFinal base pos: x={base[0]:.3f} y={base[1]:.3f} z={base[2]:.3f}m | "
        f"board forward speed (last half): mean={sum(tail) / len(tail):.2f} m/s (command {args.vx:.2f})"
    )

    if args.record:
        env.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")


if __name__ == "__main__":
    main()
