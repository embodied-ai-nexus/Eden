"""Keyboard-teleoperate a Franka to slice an MPM-elastic dragon via two-way MPM-rigid coupling.

---
tier: recipe
capabilities: [solver.mpm, solver.rigid, robot.franka, teleop.keyboard]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/physics/coupler_cut_dragon_teleop.py -v
---

Eden port of Genesis' ``examples/coupling/cut_dragon.py`` with live keyboard teleop. A
``SE3Keyboard`` device drives a Franka Research 3 through a ``DifferentialIKController``;
the end-effector-mounted cutter slices a deformable dragon under two-way MPM-rigid
coupling. Runs on the CPU backend with ``decimation=1`` so teleop input is live.

    uv run python examples/recipes/physics/coupler_cut_dragon_teleop.py -v
"""

import os
from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MPMElasticMaterialOptions, RigidMaterialOptions
from eden.utils.devices.keyboard import SE3Keyboard
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.cpu)

    gs_assets = get_gs_assets_dir()

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=4e-3,
            sim_substeps=10,
            decimation=1,  # teleop drives the env with env.step(action), so decimation is live
            enable_default_keybinds=False,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.05, -0.25, -0.05),
                upper_bound=(0.6, 0.25, 0.5),
                grid_density=64,
                enable_CPIC=True,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(
                material=RigidMaterialOptions(
                    coup_friction=1.0,
                ),
            ),
            arm=en.robots.FrankaResearch3(
                material=RigidMaterialOptions(
                    gravity_compensation=1.0,
                ),
            ),
            cutter=EntityOptions(
                file=os.path.join(gs_assets, "meshes/cross_cutter.obj"),
                scale=0.1,
                is_fixed_base=False,
                convexify=False,
                default_root_quat=(0.7071068, 0.7071068, 0.0, 0.0),
                surface=gs.surfaces.Iron(),
            ),
            dragon=EntityOptions(
                file=os.path.join(gs_assets, "meshes/dragon/dragon.obj"),
                scale=0.0015,
                up=(0, 0, 1),
                front=(0, 1, 0),
                default_root_pos=(0.4, 0.0, 0.15),
                material=MPMElasticMaterialOptions(
                    sampler="pbs-64",
                ),
                surface=gs.surfaces.Rough(
                    color=(0.6, 1.0, 0.8, 1.0),
                    vis_mode="particle",
                ),
                vis_mode="particle",
            ),
            attachments_dict={
                "cutter": ("arm", "attachment"),
            },
        ),
        action_options=ActionManagerOptions(
            ee_controller=DifferentialIKController.configure(
                entity_name="arm",
                ee_link_name="attachment",
                dofs_name=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                ],
                position_scale=0.05,
                orientation_scale=0.05,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    en.logger.info(env.action_manager.summary())

    keyboard = SE3Keyboard(SE3Keyboard.wasd_key_bindings())
    keyboard.add_callback("x", lambda on_press: env.reset() if on_press else None)

    while True:
        action = keyboard.get_command()
        action = {
            "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
        }
        env.step(action)


if __name__ == "__main__":
    main()
