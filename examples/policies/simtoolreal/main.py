"""Sim2sim smoke-test for the SimToolReal policy port.

---
tier: policy
capabilities: [task.manipulation, task.dexterous, robot.iiwa14, robot.sharpa_wave]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/simtoolreal/main.py --object mallet_hammer --render
---

Builds a Genesis scene (KUKA+SharPa robot + box table + a DexToolBench tool),
loads the pretrained SAPG-LSTM actor, and runs the full deploy loop
(obs from sim -> network -> action transform -> PD -> physics) at 60 Hz.

This exercises the real pipeline end-to-end. It is NOT a task-success test
(goal is a fixed lift target); it verifies the plumbing runs without NaNs and
produces in-range targets that move the robot.

Usage::

    uv run python examples/policies/simtoolreal/main.py --object mallet_hammer --render

The checkpoint downloads from the ``Kashu7100/eden_policies`` hub on first use; pass
``--checkpoint /path/to/model.pth --md5 ''`` to run custom weights instead.
"""

from __future__ import annotations

import argparse
import math

import eden  # noqa: F401  (import before genesis)
import genesis as gs
import torch
from genesis.utils.geom import axis_angle_to_quat, transform_quat_by_quat

from eden.envs.base import EnvBase
from eden.options.camera import CameraOptions
from eden.options.envs import EnvOptions
from eden.options.objects.dextoolbench import DEXTOOLBENCH_REGISTRY
from eden.options.scenes.empty import EmptySceneOptions
from eden.utils.configs import EdenConfig
from eden.extensions.policies.simtoolreal import SimToolRealPolicy
from eden.extensions.policies.simtoolreal import constants as C

# URDFs and grasp bboxes come from eden.options.objects.dextoolbench.
TOOLS: list[str] = sorted(name for name, _ in DEXTOOLBENCH_REGISTRY)


def sample_goal_delta(
    cur_pos: torch.Tensor,
    cur_quat: torch.Tensor,
    rng: torch.Generator,
    z_lo: float,
    z_hi: float,
    max_dist: float,
    max_rot: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-walk the goal by a bounded delta from its current pose.

    Matches the policy's training goal distribution (``goal_sampling_type=delta``:
    each goal is a small offset, <=``delta_goal_distance`` / <=``delta_rotation``,
    from the current pose) so every goal stays reachable from where the object
    already is. Absolute uniform goals jump across the whole workspace, which jerks
    the arm and drops the tool, and full-SO(3) orientations are out of distribution
    — both read as the object never quite reaching the ghost.

    Position delta is uniform in a ball of radius ``max_dist``, clamped to the
    target volume in x/y and ``[z_lo, z_hi]`` in z (so the tilted tool clears the
    table). Orientation delta is <=``max_rot`` about a random axis.
    """
    n = cur_pos.shape[0]
    # Explicit device: the generator is CPU (so a seed reproduces across backends) but Eden
    # sets torch's default device to the sim backend, and that mismatch is an error.
    cpu = dict(device="cpu")
    direction = torch.nn.functional.normalize(torch.randn(n, 3, generator=rng, **cpu), dim=-1)
    radius = torch.rand(n, 1, generator=rng, **cpu) ** (1 / 3) * max_dist  # uniform in ball
    lo = torch.tensor([C.TARGET_VOLUME_MINS[0], C.TARGET_VOLUME_MINS[1], z_lo], **cpu)
    hi = torch.tensor([C.TARGET_VOLUME_MAXS[0], C.TARGET_VOLUME_MAXS[1], z_hi], **cpu)
    new_pos = torch.clamp(cur_pos + direction * radius, lo, hi)

    angle = torch.rand(n, generator=rng, **cpu) * max_rot
    axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=rng, **cpu), dim=-1)
    delta = axis_angle_to_quat(angle, axis)
    # transform_quat_by_quat(v, u) applies u on the left: delta ∘ cur, i.e. in the world frame.
    new_quat = torch.nn.functional.normalize(transform_quat_by_quat(cur_quat, delta), dim=-1)
    return new_pos, new_quat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=SimToolRealPolicy.checkpoint_path,
        help="hub-relative path (downloaded from Kashu7100/eden_policies) or a local .pth; "
        "a local file also needs --md5 '' to skip the integrity check",
    )
    parser.add_argument(
        "--md5",
        default=SimToolRealPolicy.md5_hash,
        help="expected checkpoint md5; pass an empty string to skip the check for custom weights",
    )
    parser.add_argument(
        "--object",
        default="long_screwdriver",
        choices=TOOLS,
        help="DexToolBench tool to spawn (selects the URDF + its grasp-bbox object_scales)",
    )
    parser.add_argument("-b", "--num-envs", type=int, default=2, help="Number of parallel environments")
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=1.2,
        help="grid spacing between parallel envs (m); 0 stacks them at the same origin",
    )
    parser.add_argument(
        "-t", "--seconds", type=float, default=15.0, help="rollout length in seconds (at 60 Hz control)"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=0, help="override rollout length in steps (0 = derive from --seconds)"
    )
    parser.add_argument(
        "--goal-period-s",
        type=float,
        default=3.0,
        help="resample a random goal pose every this many seconds (0 = fixed goal)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for goal sampling")
    parser.add_argument("--max-dist", type=float, default=0.12, help="max goal translation step per resample (m)")
    parser.add_argument("--max-rot-deg", type=float, default=45.0, help="max goal rotation step per resample (deg)")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("--render", action="store_true", help="record an mp4 of env 0")
    parser.add_argument("-o", "--output", default="/tmp/simtoolreal_sim2sim.mp4", help="Path to write the result to")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate of the recorded video")
    parser.add_argument(
        "--collision",
        action="store_true",
        help="render collision geometry (vis_mode='collision') instead of visual meshes",
    )
    parser.add_argument(
        "--constraint-timeconst",
        type=float,
        default=0.02,
        help="rigid-solver constraint time constant (softer contacts)",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    tool_cls = DEXTOOLBENCH_REGISTRY.get(args.object)
    object_scales = C.object_scales_for(tool_cls.metadata.grasp_bbox)
    print(f"object: {args.object}  scales={tuple(round(s, 3) for s in object_scales)}")

    vis = "collision" if args.collision else "visual"
    # One composite; the scene expands it into the `robot` (arm) and `hand` entities.
    robot_options = SimToolRealPolicy.robot_options()
    robot_options.arm.vis_mode = vis
    robot_options.hand.vis_mode = vis

    # Table + tool + a translucent kinematic ghost of the tool at the goal pose (whose
    # pose this script then re-sets each step). The policy owns that workspace — `eden
    # inference --policy simtoolreal` builds the same one — so it is not respelled here.
    scene = EmptySceneOptions(
        robot=robot_options,
        **SimToolRealPolicy.scene_entity_options(object_asset=args.object, vis_mode=vis),
    )
    cfg = EdenConfig(
        env_options=EnvOptions(
            num_envs=args.num_envs,
            sim_dt=C.SIM_DT,
            decimation=C.DECIMATION,
            constraint_timeconst=args.constraint_timeconst,
            # Lay the parallel envs out side by side; they overlap at the default (0, 0).
            env_spacing=(args.env_spacing, args.env_spacing),
        ),
        scene_options=scene,
    )
    if args.render:
        # Robot base at (0,0.8,0); table/object near the world origin in front of
        # it. View from +x / front so both the arm and the workspace are visible.
        cfg.cameras_options.camera = CameraOptions(
            cam_pos=(1.5, 0.4, 0.9),
            cam_lookat=(0.0, 0.35, 0.5),
            cam_resolution=(960, 720),
            cam_fov=48,
        )
    env = EnvBase.from_config(cfg, show_viewer=args.vis)
    env.build()

    cam = env.cameras["camera"] if args.render else None
    if cam is not None:
        cam.start_recording(video_path=args.output, fps=args.fps)

    policy = SimToolRealPolicy(
        env=env,
        options=SimToolRealPolicy.configure(checkpoint_path=args.checkpoint, md5_hash=args.md5 or None),
    )
    policy.set_object_scales(torch.tensor(object_scales).expand(args.num_envs, 3).clone())
    policy.reset()

    arm = env.entities["robot"]
    hand = env.entities["hand"]
    obj = env.entities["object"]
    goal_ghost = env.entities["goal_ghost"]
    # Genesis tree-order indices for the canonical joints, per entity.
    arm_dofs_idx, hand_dofs_idx = policy._arm_dofs_idx, policy._hand_dofs_idx

    # Rollout length + random-goal cadence (control runs at 1/(sim_dt*decimation) Hz).
    control_hz = round(1.0 / (C.SIM_DT * C.DECIMATION))
    steps = args.steps if args.steps > 0 else round(args.seconds * control_hz)
    goal_period = round(args.goal_period_s * control_hz) if args.goal_period_s > 0 else 0
    rng = torch.Generator().manual_seed(args.seed)

    # Goal-sampling bounds. The goal orientation random-walks, so it can reach any
    # tilt; set the z floor to table top + the tool's full bounding radius (farthest
    # mesh vertex from the object root) so the tool clears the table at ANY orientation.
    max_rot = math.radians(args.max_rot_deg)
    table_top = C.TABLE_POS[2] + C.TABLE_SIZE[2] / 2
    obj_aabb = obj._entity.get_AABB().reshape(-1, 3)
    tool_radius = (obj_aabb - obj.get_pos()[0]).norm(dim=-1).max().item()
    goal_z_lo = max(C.TARGET_VOLUME_MINS[2], table_top + tool_radius + 0.02)
    goal_z_hi = max(goal_z_lo + 0.05, C.TARGET_VOLUME_MAXS[2])
    print(
        f"tool radius={tool_radius:.3f}m -> goal z in [{goal_z_lo:.3f}, {goal_z_hi:.3f}], "
        f"delta <= {args.max_dist}m / {args.max_rot_deg}deg per {args.goal_period_s}s"
    )

    # Park the goal ghost at the policy goal pose (kinematic; refreshed each step
    # so the moving goal tracks).
    goal_ghost.set_pos(policy._goal_pos)
    goal_ghost.set_quat(policy._goal_quat)

    print(
        f"\n=== running {steps} steps ({steps / control_hz:.1f}s @ {control_hz}Hz), "
        f"num_envs={args.num_envs}, random goal every {args.goal_period_s}s ==="
    )
    obj_start = obj.get_pos().clone()
    # Running goal pose, random-walked by a bounded delta each resample (kept on
    # CPU so the seeded generator is reproducible across backends).
    g_pos = policy._goal_pos.detach().cpu().clone()
    g_quat = policy._goal_quat.detach().cpu().clone()
    for step in range(steps):
        if goal_period and step % goal_period == 0:
            g_pos, g_quat = sample_goal_delta(g_pos, g_quat, rng, goal_z_lo, goal_z_hi, args.max_dist, max_rot)
            policy.set_goal(g_pos, g_quat)
            print(f"  step {step:4d}: new goal pos={[round(x, 3) for x in g_pos[0].tolist()]}")
        out = policy.step({})
        goal_ghost.set_pos(policy._goal_pos)
        goal_ghost.set_quat(policy._goal_quat)
        arm_targets, hand_targets = out["dofs_pos_controller"], out["hand_dofs_pos_controller"]
        assert torch.isfinite(arm_targets).all(), f"non-finite arm targets at step {step}"
        assert torch.isfinite(hand_targets).all(), f"non-finite hand targets at step {step}"
        # Drive Genesis PD with the absolute joint-position targets, then advance
        # `decimation` sim substeps (target persists across substeps).
        arm._entity.control_dofs_position(arm_targets, arm_dofs_idx)
        hand._entity.control_dofs_position(hand_targets, hand_dofs_idx)
        for _ in range(C.DECIMATION):
            env.scene.step()
        if cam is not None:
            cam.render_rgb()

        if step % 25 == 0 or step == steps - 1:
            q, _ = policy._read_q(arm, hand)
            obj_pos = obj.get_pos()
            lifted = (obj_pos[:, 2] - obj_start[:, 2]).max().item()
            print(
                f"step {step:3d} | target[0,:3]={[round(x, 3) for x in arm_targets[0, :3].tolist()]} "
                f"| q finite={torch.isfinite(q).all().item()} "
                f"| obj_z[0]={obj_pos[0, 2].item():.3f} | max lift={lifted:+.3f}"
            )

    if cam is not None:
        cam.stop_recording()
        print(f"\nsaved video -> {args.output}")

    print("\nSIM2SIM PIPELINE OK — ran end-to-end with finite targets/state.")


if __name__ == "__main__":
    main()
