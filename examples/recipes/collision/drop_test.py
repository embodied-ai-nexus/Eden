"""Behavioral test of the collision mask: drop a cube onto the gripper.

---
tier: recipe
capabilities: [collision.mask]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/collision/drop_test.py
---

Default (the gripper's own per-geom collision): the cube is caught by the
gripper body (the inner linkage bears the contact force).
Pad-only mask (``collision_link_patterns=[".*pad"]``): the body links no longer
collide, so they bear **zero** contact force — only the fingertip pads can hold
an object. This is exactly the difference between wedging a grasped object
against the palm (a physics artifact) and a real fingertip pinch.

    uv run python examples/recipes/collision/drop_test.py          # default: body bears the force
    uv run python examples/recipes/collision/drop_test.py --mask    # pad-only: body force -> 0

Prints the per-link contact force (pad vs body) and saves before/after PNGs.
"""

import argparse


import eden as en  # must be imported before genesis
import genesis as gs
from eden.envs.base import EnvBase
from eden.options import EnvOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import BoxOptions
from eden.options.materials import RigidMaterialOptions
from eden.utils.configs import EdenConfig

_CUBE_HALF = 0.025


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", action="store_true", help="restrict gripper collision to the fingertip pads")
    parser.add_argument("-s", "--steps", type=int, default=200, help="physics steps to let the cube fall/settle")
    args = parser.parse_args()

    en.init(backend=gs.gpu, log_root_path="logs/collision")

    gripper = en.robots.Robotiq2f85(
        vis_mode="collision",
        # gravity_compensation keeps the (non-fixed-base MJCF) gripper hovering in
        # place while the cube falls onto it, so the test isolates the mask effect.
        material=RigidMaterialOptions(gravity_compensation=1.0),
        collision_link_patterns=[".*pad"] if args.mask else None,
        collision_friction=2.0 if args.mask else None,
    )
    # Cube dropped from above the gripper body (not between the pads).
    cube = BoxOptions(size=(2 * _CUBE_HALF,) * 3, fixed=False, default_root_pos=(0.0, 0.0, 0.35))
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=en.scenes.Empty(robot=gripper, cube=cube),
        cameras_options=CamerasOptions(
            cam=CameraOptions(cam_pos=(0.45, 0.45, 0.35), cam_lookat=(0.0, 0.0, 0.12), cam_resolution=(960, 720)),
        ),
    )

    env = EnvBase.from_config(config, show_viewer=False)
    env.build()
    cube_ent = env.entities["cube"]
    grip = env.entities["robot"]

    # Drop the cube straight down onto the gripper's BASE (palm) link — a body
    # link the mask disables — rather than a fingertip pad.
    _, base_idx = grip.find_named_links_idx_local(["base"], preserve_order=True)
    base_pos = grip.get_links_pos(ls_idx_local=base_idx).squeeze(1)[0]
    drop = base_pos.clone()
    drop[2] = base_pos[2] + 0.20
    cube_ent.set_pos(drop.unsqueeze(0))

    tag = "padonly" if args.mask else "all"
    env.cameras["cam"].snapshot(f"{en.log_dir}/drop_{tag}_start.png")
    z0 = cube_ent.get_pos()[0, 2].item()

    for _ in range(5 if en.is_smoke() else args.steps):
        env.scene.step()

    zf = cube_ent.get_pos()[0, 2].item()
    env.cameras["cam"].snapshot(f"{en.log_dir}/drop_{tag}_end.png")

    # Net contact force borne by each gripper link (definitive: a contype-disabled
    # link bears zero force — it cannot hold the cube up).
    ncf = grip.get_links_net_contact_force()[0]  # (n_links, 3)
    pad_force = body_force = 0.0
    for link, f in zip(grip.links, ncf):
        mag = float(f.norm())
        if "pad" in link.name:
            pad_force += mag
        else:
            body_force += mag

    mode = "pad-only mask" if args.mask else "default (the gripper's own per-geom collision)"
    print(f"\n=== drop test — {mode} ===")
    print(f"  cube z: start={z0:.3f}  ->  final={zf:.3f}")
    print(f"  contact force on PAD links:  {pad_force:.2f} N")
    print(f"  contact force on BODY links: {body_force:.2f} N   <- mask zeroes this (body can't hold the cube)")
    print(f"  snapshots: {en.log_dir}/drop_{tag}_start.png , drop_{tag}_end.png")


if __name__ == "__main__":
    main()
