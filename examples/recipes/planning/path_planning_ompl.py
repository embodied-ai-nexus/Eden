"""Collision-free arm motion planning with the OMPL planner on a Franka.

---
tier: recipe
capabilities: [planning.ompl, robot.franka]
hardware: cpu
optional_deps: [ompl]
run: uv run python examples/recipes/planning/path_planning_ompl.py -v
---
"""

from argparse import ArgumentParser

import numpy as np
import genesis as gs
import eden as en

from eden.options.entities import EntityOptions, PlaneOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions, EventManagerOptions
from eden.options import ActionManagerOptions
from eden.constants import ReferenceSource
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.terms import EventTerm
from eden.extensions.motion_planning.ompl import OMPLPlanner
from eden.options.extensions.motion_planning import PlannerOptions

from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.constants import EventMode


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
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
            floor=PlaneOptions(
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, -0.5),
            ),
            robot=en.robots.FrankaEmikaPanda(),
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
        ),
        action_options=ActionManagerOptions(
            control_type=ImplicitPDController.configure(
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
                reference_source=ReferenceSource.ZERO,
            )
        ),
        show_viewer=args.vis,
    )

    env.build()

    env.reset()

    robot = env.entities["robot"]
    plate = env.entities["plate"]

    planner = OMPLPlanner(
        env,
        PlannerOptions(
            entity_name="robot",
        ),
    )

    for i in range(10):
        target_pos = plate.get_pos()
        target_pos[:, 2] += 0.2
        qpos = robot.inverse_kinematics(
            links_name="hand",
            poss=target_pos,
            quats=np.array([[0, 1, 0, 0]] * env.num_envs),
        )[0]

        waypoints = planner.plan_motion(qpos_goal=qpos, envs_idx=0)

        for waypoint in waypoints:
            action = waypoint[:7].unsqueeze(0).repeat(env.num_envs, 1)
            if env.num_envs > 1:
                action[1:] = 0.0
            env.step(action)

        env.reset()


if __name__ == "__main__":
    main()
