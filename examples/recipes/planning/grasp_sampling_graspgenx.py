"""Sample 6-DoF grasps with GraspGenX for the gripper a robot carries, and keep those the robot can execute.

---
tier: recipe
capabilities: [planning.graspgenx, robot.franka, robot.ur5e]
hardware: gpu
optional_deps: [graspgenx]
run: uv run python examples/recipes/planning/grasp_sampling_graspgenx.py -v
---

GraspGenX is one model for every gripper it has a description of, so the same code samples for the
Franka hand and for a Robotiq 2F-85 on a UR5e: only the Eden end-effector (``RIGS[...].end_effector``)
changes. It picks GraspGenX's description of that gripper, and each pose the sampler returns places
the Eden link ``sampler.pose_link_name`` (the Franka's ``hand``, the 2F-85's root), so driving that
link onto it puts the jaws around the grasp.

A grasp is kept when the arm reaches it (IK) and the posed arm and gripper, jaws open, touch nothing
but the target. That check runs in simulation, with ``detect_collision``, because GraspGenX ships no
gripper collision meshes for its own point-cloud filter. Snapshots of the kept poses go to ``-o``.

Usage:
    uv run python examples/recipes/planning/grasp_sampling_graspgenx.py -v
    uv run python examples/recipes/planning/grasp_sampling_graspgenx.py --rig ur5e_2f85 -o /tmp/graspgenx
"""

import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable

import torch

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.grasp_sampling.graspgenx import GraspGenXSampler
from eden.options import CameraOptions, CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.ycb import YCB006MustardBottle
from eden.options.robots.composite.ur_robotiq import UR5eRobotiq
from eden.utils.geom import combine_frame_transforms, subtract_frame_transforms

TABLE_HEIGHT = 0.4


@dataclass(frozen=True)
class Rig:
    """An Eden robot, and the end-effector on it GraspGenX samples for."""

    #: The Eden end-effector (``GraspGenXSampler.end_effector``): one of ``GraspGenXSampler.END_EFFECTORS``.
    end_effector: str
    #: The robot, added under the scene key ``robot``; the arm is that entity.
    make: Callable[[], EntityOptions]
    #: The open-jaw targets of the gripper entity's DOFs, read off the robot's options.
    open_dofs_pos: Callable[[EntityOptions], dict[str, float]]
    #: The arm link IK drives.
    ik_link: str
    #: The entity that is the end-effector: the robot itself, or the gripper mounted on it.
    gripper_entity: str


RIGS = {
    "franka": Rig(
        end_effector="FrankaEmikaPanda",
        make=lambda: en.robots.FrankaEmikaPanda(
            default_root_pos=(0.0, 0.0, TABLE_HEIGHT), material=RigidMaterialOptions(gravity_compensation=1.0)
        ),
        open_dofs_pos=lambda robot: dict(robot.open_dofs_pos),
        ik_link="hand",
        gripper_entity="robot",
    ),
    "ur5e_2f85": Rig(
        end_effector="Robotiq2f85",
        make=lambda: UR5eRobotiq(base_pos=(0.0, 0.0, TABLE_HEIGHT)),
        open_dofs_pos=lambda robot: robot.gripper_open,
        ik_link="ee_link",
        gripper_entity="gripper",
    ),
}


def link_pose(entity, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """World ``(pos (1, 3), wxyz quat (1, 4))`` of one link."""
    _, idx = entity.find_named_links_idx_local([link_name])
    return entity.get_links_pos(ls_idx_local=idx)[:, 0], entity.get_links_quat(ls_idx_local=idx)[:, 0]


def contact_pairs(env: EnvBase, entity_names: list[str]) -> set[tuple[int, int]]:
    """Geom pairs in contact with any of ``entity_names`` in their current pose, without stepping."""
    pairs = set()
    for name in entity_names:
        pairs |= {tuple(sorted(map(int, pair))) for pair in env.entities[name].detect_collision()}
    return pairs


def geom_owner(env: EnvBase, geom: int) -> str:
    return next(
        name for name, e in env.entities.items() if hasattr(e, "geom_start") and e.geom_start <= geom < e.geom_end
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the viewer and cycle through the kept grasps")
    parser.add_argument(
        "--rig", type=str, default="franka", choices=sorted(RIGS), help="Robot and gripper to sample for"
    )
    parser.add_argument("--num-poses", type=int, default=4, help="Number of kept grasps to show / snapshot")
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None, help="Directory to write snapshots of the kept grasps to"
    )
    args = parser.parse_args()

    # The model runs on CUDA only; the scene can share the GPU.
    en.init(backend=gs.gpu)
    rig = RIGS[args.rig]
    robot_options = rig.make()

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            table=BoxOptions(
                size=(0.6, 0.8, TABLE_HEIGHT), default_root_pos=(0.55, 0.0, TABLE_HEIGHT / 2), is_fixed_base=True
            ),
            mustard=YCB006MustardBottle(default_root_pos=(0.5, 0.05, TABLE_HEIGHT + 0.1)),
            robot=robot_options,
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(1.1, 0.3, 0.9), cam_lookat=(0.5, 0.0, TABLE_HEIGHT), cam_resolution=(640, 480), cam_fov=55.0
            )
        ),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    env.reset()
    for _ in range(50):  # let the bottle settle on the table
        env.step()

    sampler = GraspGenXSampler(env, GraspGenXSampler.configure(end_effector=rig.end_effector))
    result = sampler.sample_grasps("mustard")
    en.logger.info(
        f"GraspGenX proposed {len(result)} grasps for '{rig.end_effector}' ({sampler.gripper_name}), "
        f"placing its link '{sampler.pose_link_name}'"
    )

    arm, gripper = env.entities["robot"], env.entities[rig.gripper_entity]
    rig_entities = sorted({"robot", rig.gripper_entity})
    # Where the IK link sits in the posed link's frame: identity for the Franka, the weld for a mounted gripper.
    ik_in_gripper = subtract_frame_transforms(*link_pose(gripper, sampler.pose_link_name), *link_pose(arm, rig.ik_link))
    open_pos = rig.open_dofs_pos(robot_options)
    open_names, open_idx = gripper.find_named_dofs_idx_local(list(open_pos))
    open_target = torch.tensor([[open_pos[name] for name in open_names]], device=env.device)
    # Contacts of the home pose (a gripper welded onto its flange) are not the grasp's fault.
    baseline = contact_pairs(env, rig_entities)
    target_geoms = range(env.entities["mustard"].geom_start, env.entities["mustard"].geom_end)

    kept = []
    for i in range(min(2, len(result)) if en.is_smoke() else len(result)):
        target = combine_frame_transforms(result.pos[i : i + 1], result.quat[i : i + 1], *ik_in_gripper)
        qpos, error = arm.inverse_kinematics(rig.ik_link, *target, return_error=True)
        if error[0, :3].norm() > 0.005 or error[0, 3:].norm() > 0.02:
            continue
        arm.set_qpos(qpos)
        gripper.set_dofs_pos(open_target, dofs_idx_local=open_idx)
        hits = {
            " <-> ".join(sorted(geom_owner(env, geom) for geom in pair))
            for pair in contact_pairs(env, rig_entities) - baseline
            if not any(geom in target_geoms for geom in pair)
        }
        if hits:
            en.logger.debug(f"grasp {i} collides: {sorted(hits)}")
            continue
        kept.append((i, qpos))
        if len(kept) == args.num_poses:
            break
    en.logger.info(f"Kept {len(kept)} reachable, collision-free grasps (best first): {[i for i, _ in kept]}")

    for i, qpos in kept:
        arm.set_qpos(qpos)
        gripper.set_dofs_pos(open_target, dofs_idx_local=open_idx)
        env.scene.visualizer.update()  # a camera or viewer renders the posed robot only after this
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            env.cameras["camera"].snapshot(os.path.join(args.output_dir, f"{args.rig}_grasp_{i:03d}.png"))
        if args.vis and not en.is_smoke():
            time.sleep(2.0)


if __name__ == "__main__":
    main()
