"""Run any of the 60 BeyondMimic G1-Moves motion-tracking policies.

---
tier: policy
capabilities: [task.mimic, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/beyondmimic_moves/main.py --clip B_DadDance --vis
---

Source dataset: https://huggingface.co/datasets/exptech/g1-moves

The policy and its motion file are downloaded lazily on first use from the
``exptech/g1-moves`` HuggingFace dataset (see
``BeyondMimicMovesPolicy.resolve_local_dir()`` for the local cache location).

Usage::

    python main.py --clip B_DadDance --vis
    python main.py --clip M_Move5 --record
    python main.py --list-clips
"""

import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

import eden as en
import genesis as gs
from genesis.utils.geom import inv_transform_by_quat
from eden.envs.base import RLEnvBase
from eden.extensions.policies.beyondmimic_moves.policy import (
    BeyondMimicMovesPolicy,
)
from eden.options import (
    GHOST_DEFAULTS,
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions


def replay_motion(motion_file: str, scene_options, args) -> None:
    """Step through the motion's frames via set_qpos / set_dofs_vel — no policy, no physics step.

    This is the ground-truth visualization of the remapped npz. If this looks correct,
    any tracking failure is in the policy / physics, not in our data interpretation.
    """
    import numpy as np
    import torch
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions

    data = np.load(motion_file)
    print(f"Replay: {data['joint_pos'].shape[0]} frames at {float(data['fps'][0])} FPS")

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / float(data["fps"][0])),
        scene_options=scene_options,
        cameras_options=(
            CamerasOptions(rec=CameraOptions(cam_fov=50.0, follow_entity_name="robot")) if args.record else None
        ),
        show_viewer=args.vis,
    )
    env.build()
    robot = env.entities["robot"]
    _, base_idx = robot.find_named_dofs_idx_local("floating_base_joint")

    n_frames = data["joint_pos"].shape[0]
    body_pos = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=env.device)
    body_quat = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=env.device)
    body_lin = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=env.device)
    body_ang = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=env.device)
    j_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=env.device)
    j_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=env.device)

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), f"{args.clip}_replay.mp4")
        env.cameras["rec"].start_recording(out_path)

    for t in tqdm(range(n_frames - 1)):
        # body axis index 0 == pelvis (the root link is index 0 regardless of Genesis's internal
        # link ordering, which changed BFS->DFS in genesis 1.2.0), so this replay needs no remap.
        robot.set_qpos(torch.cat([body_pos[t : t + 1, 0], body_quat[t : t + 1, 0], j_pos[t : t + 1]], dim=1))
        # Genesis stores a free joint's rotational DOF velocity in the base BODY frame
        # (MuJoCo qvel convention); body_ang_vel_w is world-frame. Rotate world->body with
        # the base link quat written to qpos above. Linear DOFs stay world-frame.
        base_ang_vel_b = inv_transform_by_quat(body_ang[t : t + 1, 0], body_quat[t : t + 1, 0])
        robot.set_dofs_vel(
            torch.cat([body_lin[t : t + 1, 0], base_ang_vel_b, j_vel[t : t + 1]], dim=1),
            dofs_idx_local=torch.cat([torch.as_tensor(base_idx, device=env.device), robot.dofs_idx_local]),
        )
        env.scene.visualizer.update()
        if args.record:
            env.cameras["rec"].render_rgb()

    if args.record:
        env.cameras["rec"].stop_recording()


def list_clips() -> None:
    manifest_path = BeyondMimicMovesPolicy.fetch_asset("manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_cat: dict[str, list[str]] = {}
    for name, info in manifest["clips"].items():
        by_cat.setdefault(info["category"], []).append(name)
    for cat in sorted(by_cat):
        print(f"\n[{cat}] ({len(by_cat[cat])} clips)")
        for name in sorted(by_cat[cat]):
            print(f"  {name}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--clip", type=str, default="B_DadDance", help="Clip name (see --list-clips)")
    parser.add_argument("--list-clips", action="store_true", help="Print all 60 available clips and exit")
    parser.add_argument("-d", "--debug-vis", action="store_true", help="Render a ghost robot of the reference motion")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay the motion directly via set_qpos (no policy / no physics) to verify the remapped npz visually.",
    )

    args = parser.parse_args()

    if args.list_clips:
        list_clips()
        return

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    options = BeyondMimicMovesPolicy.configure(clip_name=args.clip)

    if args.debug_vis:
        scene_options = en.scenes.Empty(
            robot=BeyondMimicMovesPolicy.robot_options(), ghost_robot=en.robots.G1Dofs29(**GHOST_DEFAULTS)
        )
        ghost_entity_name = "ghost_robot"
    else:
        scene_options = en.scenes.Empty(robot=BeyondMimicMovesPolicy.robot_options())
        ghost_entity_name = ""

    # Download and remap the motion NPZ into Genesis BFS body order before
    # building the env so MotionCommand reads correctly-ordered body data.
    motion_file = BeyondMimicMovesPolicy.prepare_motion_file(args.clip)

    if args.replay:
        replay_motion(motion_file, scene_options, args)
        return

    env = RLEnvBase(
        env_options=EnvOptions(num_envs=1, episode_length_s=1e9, **BeyondMimicMovesPolicy.env_options()),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **BeyondMimicMovesPolicy.command_options(
                clip_name=args.clip,
                debug_vis=args.debug_vis,
                ghost_entity_name=ghost_entity_name,
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**BeyondMimicMovesPolicy.action_options()),
        observation_options=ObservationManagerOptions(**BeyondMimicMovesPolicy.observation_options()),
        cameras_options=(
            CamerasOptions(rec=CameraOptions(cam_fov=50.0, follow_entity_name="robot")) if args.record else None
        ),
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()

    policy = BeyondMimicMovesPolicy(env=env, options=options)

    cmd = env.command_manager.get_term("motion")

    obs, _ = env.reset()

    if args.record:
        rec = env.cameras["rec"]
        out_path = os.path.join(os.path.dirname(__file__), f"{args.clip}.mp4")
        rec.start_recording(out_path)

    for _ in tqdm(range(int(cmd.motion.time_step_total) - 1)):
        raw_action = policy.step(obs)
        obs, _, _, _, _ = env.step(raw_action)
        if args.record:
            rec.render_rgb()

    if args.record:
        rec.stop_recording()


if __name__ == "__main__":
    main()
