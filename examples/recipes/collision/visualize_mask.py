"""Visualize the per-geom collision mask (``collision_link_patterns``).

---
tier: recipe
capabilities: [collision.mask]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/collision/visualize_mask.py
---

``EntityOptions.collision_link_patterns`` restricts an entity's collision to the
geoms on links matching the given regex/glob patterns; every other link's geoms
are disabled (``contype = conaffinity = 0``). This is what lets a parallel-jaw
gripper grasp objects at its *fingertip pads* instead of wedging them against
the inner linkage/palm (see ``eden/tasks/manipulation``).

This script builds a Robotiq 2F85 gripper, prints which links participate in
collision, and renders the collision geometry (``vis_mode="collision"``).

    # default: every link collides
    uv run python examples/recipes/collision/visualize_mask.py

    # pad-only: collision_link_patterns=[".*pad"] -> only the fingertip pads collide
    uv run python examples/recipes/collision/visualize_mask.py --mask

    # open the Genesis viewer instead of saving a PNG (needs a display)
    uv run python examples/recipes/collision/visualize_mask.py --mask --vis
"""

import argparse
from collections import defaultdict

import eden as en  # must be imported before genesis
import genesis as gs
from eden.envs.base import EnvBase
from eden.options import EnvOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.utils.configs import EdenConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", action="store_true", help="restrict collision to the fingertip pads")
    parser.add_argument("-v", "--vis", action="store_true", help="open the Genesis viewer (needs a display)")
    parser.add_argument("-o", "--output", default=None, help="output PNG path (default: under the run log dir)")
    args = parser.parse_args()

    en.init(backend=gs.gpu, log_root_path="logs/collision")

    gripper = en.robots.Robotiq2f85(
        vis_mode="collision",  # render collision geometry instead of visual meshes
        collision_link_patterns=[".*pad"] if args.mask else None,
        collision_friction=2.0 if args.mask else None,
    )
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=en.scenes.Empty(robot=gripper),  # the gripper occupies the scene's robot slot
        cameras_options=CamerasOptions(
            cam=CameraOptions(cam_pos=(0.28, 0.28, 0.18), cam_lookat=(0.0, 0.0, 0.04), cam_resolution=(960, 720)),
        ),
    )

    env = EnvBase.from_config(config, show_viewer=args.vis)
    env.build()

    # Per-link collision flags. A link collides iff any of its geoms has both
    # contype and conaffinity set; the mask zeros these on non-matching links.
    grip = env.entities["robot"]
    flags = defaultdict(lambda: (0, 0, False))
    for g in grip.geoms:
        ct, ca, collide = flags[g.link.name]
        flags[g.link.name] = (g.contype, g.conaffinity, collide or (g.contype != 0 and g.conaffinity != 0))

    mode = (
        "pad-only mask  collision_link_patterns=['.*pad']"
        if args.mask
        else "default  (no mask: the entity's own per-geom collision from its MJCF)"
    )
    print(f"\n=== Robotiq 2F85 collision flags — {mode} ===")
    for link in sorted(flags):
        ct, ca, collide = flags[link]
        print(f"  {link:20s} {'COLLIDES' if collide else 'disabled'}   (contype={ct}, conaffinity={ca})")

    if args.vis:
        print("\nviewer open — Ctrl+C to exit")
        try:
            step = 0
            while True:
                env.scene.step()
                step += 1
                if en.is_smoke() and step >= 5:
                    break
        except KeyboardInterrupt:
            pass
    else:
        out = args.output or f"{en.log_dir}/robotiq_collision_{'padonly' if args.mask else 'all'}.png"
        env.cameras["cam"].snapshot(out)
        print(f"\nsaved collision-geometry render -> {out}")


if __name__ == "__main__":
    main()
