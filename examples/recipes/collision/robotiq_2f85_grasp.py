"""Teleop-grasp the closed-loop Robotiq 2F85 on a bare plane — the pad-only mask in action.

---
tier: recipe
capabilities: [collision.mask, robot.robotiq_2f85, robot.franka, teleop.keyboard]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/collision/robotiq_2f85_grasp.py --cpu -s 150
---

Companion to ``visualize_mask.py`` (see which links collide) and ``drop_test.py`` (the
mask lets a cube fall through the body): this shows the payoff — with the pad-only mask
the closed-loop 2F85 actually **grasps**. A minimal Plane + robot + object scene, so the
grasp is isolated from any task/scene complexity. The 2F85 loop-closure / pad / spring /
dissipation tuning ships as declarative ``Robotiq2f85`` defaults (applied in
``RigidEntity.post_build``), so there is no runtime patching. The three things that make the grasp hold:

- **Pad-only collision** (``collision_link_patterns=[".*pad"]``, via ``en.robots.Droid``)
  so the gripper body does not wedge/penetrate the object — the point of this recipe.
- **Sim regime** ``sim_dt=0.004, sim_substeps=4`` (substep dt 0.001) so Genesis's
  ``2*substep_dt`` solref clamp does not soften the 2F85's 0.002 timeconst.
- **Per-object contact stiffness** via each object's own ``default_geoms_sol_params``
  (NOT a global ``EnvOptions.constraint_timeconst`` — that over-stiffens a mesh
  object's plane contact and ejects it; see ``docs/guides/entities.md``).

Grasping is driven by teleop (a hands-off regression
lives in ``tests/options/robots/test_robotiq_2f85.py``). Two embodiments compare the closed-loop
2F85 against the loop-free Franka Hand on an identical object.

Usage:
    # keyboard teleop (WASD + gripper key), viewer on:
    python examples/recipes/collision/robotiq_2f85_grasp.py --vis
    python examples/recipes/collision/robotiq_2f85_grasp.py --vis --object tomato
    python examples/recipes/collision/robotiq_2f85_grasp.py --vis --robot panda
    python examples/recipes/collision/robotiq_2f85_grasp.py --vis --spacemouse   # SE3 SpaceMouse
    # headless smoke (builds + steps, no teleop input):
    python examples/recipes/collision/robotiq_2f85_grasp.py --cpu -s 150
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.envs.base import RLEnvBase
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions, SceneOptions, SolParamsType
from eden.options.envs import EnvOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects import polaris as polaris_objects
from eden.options.objects.polaris import PolarisObjectOptions
from eden.utils.configs import EdenRLConfig

_ARM = [f"joint{i}" for i in range(1, 8)]

# Grasp-ready arm home pose (from eden.options.robots.composite.droid): lets the Franka
# reach down toward an object on the ground plane.
_HOME = {
    "joint1": 0.0,
    "joint2": -0.569,
    "joint3": 0.0,
    "joint4": -2.810,
    "joint5": 0.0,
    "joint6": 3.037,
    "joint7": 0.741,
}

# Stiffen the *object's own* collision geoms so the pad<->object pinch is near-rigid
# (Genesis averages a contact pair's sol params, so a soft object dilutes the tuned pad).
# Applied per-object, never globally -- a global constraint_timeconst ejects mesh objects.
_OBJECT_SOL_PARAMS: SolParamsType = (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)


def _polaris_classes() -> dict[str, type[PolarisObjectOptions]]:
    return {
        name: cls
        for name, cls in vars(polaris_objects).items()
        if isinstance(cls, type) and issubclass(cls, PolarisObjectOptions) and cls is not PolarisObjectOptions
    }


def make_object(name: str) -> EntityOptions:
    """Resolve --object: "box" (primitive control case) or a Polaris asset ("tomato")."""
    if name == "box":
        return BoxOptions(
            size=(0.05, 0.05, 0.05),
            fixed=False,
            default_root_pos=(0.5, 0.0, 0.03),
            material=RigidMaterialOptions(friction=2.0),
            default_geoms_sol_params={".*": _OBJECT_SOL_PARAMS},
        )
    classes = _polaris_classes()
    wanted = name.lower().replace("_", "")
    for cls_name, cls in classes.items():
        if cls_name.lower() in (wanted, f"polaris{wanted}"):
            return cls(default_root_pos=(0.5, 0.0, 0.05), default_geoms_sol_params={".*": _OBJECT_SOL_PARAMS})
    raise SystemExit(f"Unknown object '{name}'. Choices: box, {', '.join(sorted(classes))}")


def build_cfg(robot_kind: str, object_name: str) -> EdenRLConfig:
    obj = make_object(object_name)
    if robot_kind == "panda":
        robot = en.robots.FrankaEmikaPanda(
            default_dofs_pos={**_HOME, "finger_joint1": 0.04, "finger_joint2": 0.04},
            material=RigidMaterialOptions(gravity_compensation=1.0),
        )
        scene = SceneOptions(robot=robot, obj=obj, plane=PlaneOptions())
        ee_link = "hand"
        gripper = BinaryJointController.configure(
            entity_name="robot",  # Franka's gripper is part of the same entity
            dofs_name=["finger_joint1", "finger_joint2"],
            open_action={"finger_joint1": 0.04, "finger_joint2": 0.04},
            close_action={"finger_joint1": 0.0, "finger_joint2": 0.0},
        )
    else:  # droid: hand-less Panda + welded Robotiq 2F85 (pad-only, tuned declarative defaults)
        scene = SceneOptions(robot=en.robots.Droid(), obj=obj, plane=PlaneOptions())
        ee_link = "attachment"
        gripper = BinaryJointController.configure(
            entity_name="gripper",  # separate welded entity
            dofs_name=["left_driver_joint", "right_driver_joint"],
            open_action={"left_driver_joint": 0.0, "right_driver_joint": 0.0},
            close_action={"left_driver_joint": 0.8, "right_driver_joint": 0.8},
        )

    return EdenRLConfig(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e6,  # effectively no auto-reset during teleop
            enable_multi_contact=True,
            # Substep dt 0.001 keeps the 2F85's tuned 0.002 solref off Genesis's clamp;
            # control period 0.02 s (50 Hz). No global constraint_timeconst (see docstring).
            sim_dt=0.004,
            sim_substeps=4,
            decimation=5,
        ),
        scene_options=scene,
        action_options=ActionManagerOptions(
            ee_controller=DifferentialIKController.configure(
                entity_name="robot",
                ee_link_name=ee_link,
                dofs_name=_ARM,
                position_scale=0.05,
                orientation_scale=0.05,
            ),
            gripper_controller=gripper,
        ),
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--robot", choices=["droid", "panda"], default="droid", help="Robot to load into the scene")
    parser.add_argument("--object", default="box", help='"box" or a Polaris asset, e.g. "tomato"')
    parser.add_argument("--spacemouse", action="store_true", help="Use an SE3 SpaceMouse instead of the keyboard")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer (required to actually teleop)")
    parser.add_argument("-s", "--steps", type=int, default=100_000, help="Number of control steps to run")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    cfg = build_cfg(args.robot, args.object)
    cfg.env_options.enable_default_keybinds = False
    env = RLEnvBase.from_config(cfg, show_viewer=args.vis, eval_mode=True)
    env.build()
    env.reset()

    if args.spacemouse:
        from eden.utils.devices.spacemouse import SE3Spacemouse

        device = SE3Spacemouse(deadzone=0.2, is_toggle_btn0=True)
        print(f"[{args.robot}] SpaceMouse: move = EE delta, button 0 = toggle grip.")
    else:
        from eden.utils.devices.keyboard import SE3Keyboard

        device = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())
        print(f"[{args.robot}] Keyboard (WASD/QE = EE delta, gripper key toggles). Need --vis to send input.")

    for _ in range(args.steps):
        a = device.get_command()
        env.step(
            {
                "ee_controller": a[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
                "gripper_controller": a[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            }
        )


if __name__ == "__main__":
    main()
