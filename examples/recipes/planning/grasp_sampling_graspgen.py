"""Sample 6-DoF grasps with GraspGen and execute them on a Franka arm.

---
tier: recipe
capabilities: [planning.graspgen, robot.franka]
hardware: cpu
optional_deps: [graspgen]
run: uv run python examples/recipes/planning/grasp_sampling_graspgen.py -v
---
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.extensions.grasp_sampling.graspgen import GraspGenSampler
from eden.options.entities import EntityOptions, PlaneOptions
from eden.envs.base import EnvBase
from eden.options import (
    CamerasOptions,
    CameraOptions,
    EnvOptions,
    SceneOptions,
    EventManagerOptions,
)
from eden.options import ActionManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.terms import EventTerm

from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.constants import EventMode


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--action",
        type=str,
        default="diffik",
        choices=["diffik", "osc"],
        help="End-effector controller to drive the arm with",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, env_spacing=(1.5, 1.5)),
        scene_options=SceneOptions(
            table=EntityOptions(
                file="objects/tables/work_table/work_table.obj", is_fixed_base=True, is_support_enabled=True
            ),
            plate=EntityOptions(
                file="objects/plates/86d4d4f8-ab98-4c48-9630-dc5c7710733b.glb",
                is_fixed_base=False,
            ),
            apple=EntityOptions(
                file="objects/apples/013_apple/textured.obj",
                is_fixed_base=False,
            ),
            floor=PlaneOptions(
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, -0.5),
            ),
            robot=en.robots.FrankaEmikaPanda(
                material=RigidMaterialOptions(
                    gravity_compensation=1.0,
                )
            ),
        ),
        event_options=EventManagerOptions(
            place_table=EventTerm.configure(
                mode=EventMode.STARTUP,
                func=place_constant_xy,
                params={
                    "entity_name": "table",
                    "support_entity_name": "floor",
                    "pos_xy": (0.45, 0.0),
                },
                priority=0,
            ),
            place_plate=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "plate",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
                priority=1,
            ),
            place_apple=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "apple",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
                priority=2,
            ),
        ),
        action_options=ActionManagerOptions(
            ee_controller=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                ],
                scale=1.0,
            ),
            gripper_controller=BinaryJointController.configure(
                entity_name="robot",
                dofs_name=[
                    "finger_joint1",
                    "finger_joint2",
                ],
                open_action=0.04,
                close_action=0.0,
            ),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(1.2, 0.0, 0.6),
                cam_lookat=(0.5, 0.0, 0.0),
                cam_resolution=(640, 480),
                cam_fov=55.0,
            )
        ),
        # show_viewer=args.vis,
    )

    env.build()
    env.reset()

    env.step()

    en.logger.info(env.action_manager.summary())

    # end_effector picks GraspGen's franka_panda model and returns poses of the arm's ``hand`` link, which
    # is GraspGen's grasp frame turned a quarter turn about its approach axis.
    options = GraspGenSampler.configure(
        camera_name="camera",
        end_effector="FrankaEmikaPanda",
        object_cloud="surface",
        filter_collisions=True,
        grasp_threshold=0.80,
        num_grasps=200,
    )
    grasp_sampler = GraspGenSampler(env, options)

    result = grasp_sampler.sample_grasps(entity_name="apple")
    if not result.success:
        en.logger.info("No grasps found")
        return
    en.logger.info(f"Found {len(result)} grasps; confidences (best first): {result.confidences.tolist()}")

    robot = env.entities["robot"]
    camera = env.cameras["camera"]

    for i in range(len(result)):
        qpos, error = robot.inverse_kinematics(
            grasp_sampler.pose_link_name, result.pos[i : i + 1], result.quat[i : i + 1], return_error=True
        )
        position_error = error[0, :3].norm()
        orientation_error = error[0, 3:].norm()
        en.logger.info(f"Position error: {position_error}, Orientation error: {orientation_error}")
        if position_error > 0.01 or orientation_error > 0.01:
            en.logger.warning(f"Position or orientation error is too large, skipping grasp {i}")
            continue

        robot.set_dofs_pos(
            qpos[:, :7],
            dofs_idx_local=robot.find_named_dofs_idx_local(
                ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            )[1],
        )

        env.scene.visualizer.update()
        camera.snapshot(f"grasp_{i}.png")


if __name__ == "__main__":
    main()
