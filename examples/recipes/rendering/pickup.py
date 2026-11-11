"""Render the full collision-mesh setup of a manipulation pick scene.

---
tier: recipe
capabilities: [sensor.camera, robot.franka]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/rendering/pickup.py
---

Loads a registered manipulation task (default ``manipulation_lift_box``), forces every scene entity
that owns collision geometry to ``vis_mode="collision"`` (kinematic entities have none),
and renders the whole scene — Franka arm + Robotiq gripper (pad-only collision) + table
+ cube + ground — so you can see the complete collision-geometry setup, not just the
gripper. Also prints each entity's per-link collision flags (which links actually collide).

    uv run python examples/recipes/rendering/pickup.py                 # manipulation_lift_box
    uv run python examples/recipes/rendering/pickup.py --task manipulation_reach_droid
    uv run python examples/recipes/rendering/pickup.py --vis            # viewer (needs a display)
"""

import argparse

import eden as en  # must be imported before genesis
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import BaseEntityOptions
from eden.options.materials import KinematicMaterialOptions
from eden.tasks import TASK_REGISTRY


def _scene_entity_options(scene_options):
    """All entity-option objects in a SceneOptions (declared 'robot' field +
    the named-entity kwargs in __pydantic_extra__).
    """
    found = {}
    robot = getattr(scene_options, "robot", None)
    if isinstance(robot, BaseEntityOptions):
        found["robot"] = robot
    for key, val in (scene_options.__pydantic_extra__ or {}).items():
        if not key.startswith("_option_") and isinstance(val, BaseEntityOptions):
            found[key] = val
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="manipulation_lift_box", help="registered task whose scene to render")
    parser.add_argument("-v", "--vis", action="store_true", help="open the Genesis viewer (needs a display)")
    parser.add_argument("-o", "--output", default=None, help="output PNG path")
    args = parser.parse_args()

    en.init(backend=gs.gpu, log_root_path="logs/collision")

    config = TASK_REGISTRY.build(args.task)
    config.env_options.num_envs = 1

    # Render collision geometry for every entity that has some. Kinematic entities (the lift-box task's
    # goal marker) own no collision geoms, so asking is a no-op Eden has to warn about; leave
    # them on their visual mesh and say so, since a missing shape here is a real answer.
    entities = _scene_entity_options(config.scene_options)
    for name, ent in entities.items():
        if isinstance(ent.material, KinematicMaterialOptions):
            print(f"[collision vis] '{name}' is kinematic — no collision geometry, rendering its visual mesh")
            continue
        ent.vis_mode = "collision"

    # A camera framing the whole arm + table + cube.
    config.cameras_options = CamerasOptions(
        cam=CameraOptions(cam_pos=(1.2, 1.0, 0.95), cam_lookat=(0.3, 0.0, 0.45), cam_resolution=(1280, 960)),
    )

    env = RLEnvBase.from_config(config, show_viewer=args.vis)
    env.build()

    print(f"\n=== '{args.task}' scene — collision setup ===")
    for name in env.entities:
        ent = env.entities[name]
        if not hasattr(ent, "geoms"):
            continue
        collide, disabled = [], []
        flags = {}
        for g in ent.geoms:
            ln = g.link.name
            flags[ln] = flags.get(ln, False) or (g.contype != 0 and g.conaffinity != 0)
        for ln, c in sorted(flags.items()):
            (collide if c else disabled).append(ln)
        print(f"  {name:9s} collides: {collide or '-'}")
        if disabled:
            print(f"  {'':9s} disabled: {disabled}")

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
        out = args.output or f"{en.log_dir}/{args.task}_collision_scene.png"
        env.cameras["cam"].snapshot(out)
        print(f"\nsaved collision-scene render -> {out}")


if __name__ == "__main__":
    main()
