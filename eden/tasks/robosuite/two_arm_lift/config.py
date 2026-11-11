"""robosuite ``TwoArmLift``: two arms lift a two-handled pot 10 cm off the table, keeping it level.

Port of ``robosuite/environments/manipulation/two_arm_lift.py``. Two Pandas on their pedestals
face each other across the 0.8 m table (``opposed``: robot0 at y = -0.56 m facing +y, robot1 at
y = +0.56 m facing -y); ``--arrangement parallel`` puts both 0.56 m behind the table, 0.25 m either
side of centre, facing +x. The pot (``objects/pot_with_handles.xml`` in ``eden_robosuite``, robosuite's
``PotWithHandlesObject`` defaults: a 14 cm open box, 1.2 kg, green ``handle0`` on its +y side, blue
``handle1`` on -y) spawns within +-3 cm of the table centre at a yaw of 180 +- 60 degrees, so
handle0 faces robot0.

Reward, as in robosuite's ``reward(shaping=True)``:

* ``elevation`` — ``10 [upright] clip(pot_bottom - table_top - 0.05, 0, 0.15)`` (up to 1.5);
* ``grasp0`` / ``grasp1`` — 0.25 while both of that arm's fingerpads touch the pot;
* ``reach0`` / ``reach1`` — ``0.5 (1 - tanh(10 d))`` from each grasp point to its handle;
* ``success`` — 3.0 while the pot's bottom is 10 cm above the table and the pot is within 30
  degrees of level (robosuite's ``direction_coef``, which also zeroes its sparse reward).

Deviations: robosuite's ``_check_grasp`` checks each arm against *its* handle's geoms; Genesis reports
contacts per entity, so each arm's grasp counts pads on any part of the pot (documented on
``grasp0``/``grasp1``). The handle positions are the pot's root pose plus robosuite's site offsets
(:mod:`eden.tasks.robosuite.two_arm_lift.custom_terms`). The episode is
:data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` rather than 50 s, and the pot falling off the
table ends the episode.
"""

from __future__ import annotations

import math
from typing import ClassVar

from eden import EventMode, events, observations, rewards, terminations
from eden.options.camera import CamerasOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.robots import ManipulatorOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    fallen_termination,
    object_pose_noise,
    object_pose_observations,
    PANDA_TABLE_XY,
    Placements,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
    table_options,
    TABLE_TOP_Z,
    TwoArmArrangementMod,
    yaw_quat,
)
from eden.options.objects.robosuite import RobosuitePotWithHandles
from eden.tasks.robosuite.two_arm_lift.custom_terms import (
    ObjectElevationUpright,
    ObjectLiftedUpright,
)
from eden.terms import DoneTerm, EventTerm, ObsTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)


#: robosuite ``TwoArmEnv`` placements of ``base_xpos_offset["table"] = (-0.56, 0, 0)`` for two single arms:
#: opposed rotates it by +-90 degrees about z; parallel shifts it by -+0.25 m in y.
_PLACEMENTS: Placements = {
    "opposed": (
        ((0.0, PANDA_TABLE_XY[0], ROBOT_BASE_Z), yaw_quat(math.pi / 2)),  # y = -0.56, facing +y
        ((0.0, -PANDA_TABLE_XY[0], ROBOT_BASE_Z), yaw_quat(-math.pi / 2)),  # y = +0.56, facing -y
    ),
    "parallel": (
        ((PANDA_TABLE_XY[0], -0.25, ROBOT_BASE_Z), yaw_quat(0.0)),
        ((PANDA_TABLE_XY[0], 0.25, ROBOT_BASE_Z), yaw_quat(0.0)),
    ),
}

_RIG0 = panda_rig("robot0", *_PLACEMENTS["opposed"][0])
_RIG1 = panda_rig("robot1", *_PLACEMENTS["opposed"][1])

_POT_HALF_HEIGHT = 0.07
_POT_REST_Z = TABLE_TOP_Z + _POT_HALF_HEIGHT
#: robosuite ``pot_handle0`` / ``pot_handle1`` sites in the pot frame.
_HANDLE0 = (0.0, 0.155, 0.06)
_HANDLE1 = (0.0, -0.155, 0.06)
_LIFT_HEIGHT = TABLE_TOP_Z + 0.10


def _two_arm_lift_scene() -> SceneOptions:
    return SceneOptions(
        robot0=_RIG0,
        robot1=_RIG1,
        **arena_entities(),
        mount0=rethink_mount(*_PLACEMENTS["opposed"][0]),
        mount1=rethink_mount(*_PLACEMENTS["opposed"][1]),
        table=table_options(),
        pot=RobosuitePotWithHandles(default_root_pos=(0.0, 0.0, _POT_REST_Z), default_root_quat=yaw_quat(math.pi)),
    )


def _arm_observations(rig: ManipulatorOptions, index: int, handle: tuple[float, float, float]) -> dict:
    return {
        **proprio_observations(rig, prefix=f"{rig.arm_name}_"),
        f"handle{index}_pos": custom_terms.ObjectPointPos.configure(
            object_name="pot", object_offset=handle, noise=object_pose_noise()
        ),
        f"gripper{index}_to_handle{index}": custom_terms.GripperToObject.configure(
            rig=rig.arm_name, object_name="pot", object_offset=handle
        ),
    }


_OBSERVATIONS = dict(
    **_arm_observations(_RIG0, 0, _HANDLE0),
    **_arm_observations(_RIG1, 1, _HANDLE1),
    **object_pose_observations("pot"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, TwoArmArrangementMod(placements=_PLACEMENTS)))
class RobosuiteTwoArmLiftConfig(RobosuiteTaskConfig):
    robot_slots: ClassVar[tuple[str, ...]] = ("robot0", "robot1")

    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _two_arm_lift_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG0), **arm_actions(_RIG1))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        # Genesis contacts are per entity: each arm's grasp counts pads on any part of the pot, not only its handle.
        grasp0=custom_terms.PadsGrasping.configure(rig=_RIG0.arm_name, object_name="pot", contributes_to_success=False),
        grasp1=custom_terms.PadsGrasping.configure(rig=_RIG1.arm_name, object_name="pot", contributes_to_success=False),
        pot_lifted=ObjectLiftedUpright.configure(
            object_name="pot", bottom_offset=_POT_HALF_HEIGHT, height=_LIFT_HEIGHT, max_tilt_deg=30.0
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        elevation=ObjectElevationUpright.configure(
            object_name="pot",
            bottom_offset=_POT_HALF_HEIGHT,
            table_top_z=TABLE_TOP_Z,
            min_elevation=0.05,
            max_elevation=0.15,
            max_tilt_deg=30.0,
            weight=10.0,
        ),
        grasp0=rewards.MetricReward.configure(metric_name="grasp0", weight=0.25),
        grasp1=rewards.MetricReward.configure(metric_name="grasp1", weight=0.25),
        reach0=custom_terms.ReachTanh.configure(
            rig=_RIG0.arm_name, object_name="pot", object_offset=_HANDLE0, coef=10.0, weight=0.5
        ),
        reach1=custom_terms.ReachTanh.configure(
            rig=_RIG1.arm_name, object_name="pot", object_offset=_HANDLE1, coef=10.0, weight=0.5
        ),
        success=rewards.MetricReward.configure(metric_name="pot_lifted", weight=3.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        pot_fallen=fallen_termination("pot", top_z=TABLE_TOP_Z),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # UniformRandomSampler(x, y in [-0.03, 0.03], rotation in (pi - pi/3, pi + pi/3)) about the table centre.
        place_pot=EventTerm.configure(
            # ``_POT_REST_Z`` is a body-origin number (the pot's centre of mass is 1.3 cm below its
            # origin), which is the frame this term writes. The sampled yaw is *composed* onto
            # ``default_root_quat``, and the pot's default is already the 180 deg robosuite's range is
            # centred on — so the same absolute band is written as an offset about it.
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "pot",
                "pos_x_range": (-0.03, 0.03),
                "pos_y_range": (-0.03, 0.03),
                "yaw_range": (-math.pi / 3, math.pi / 3),
            },
        ),
        reset_robot0=arm_init_noise(_RIG0),
        reset_robot1=arm_init_noise(_RIG1),
    )

    # Pulled back along +x so both pedestals (1.1 m apart in y) are in frame.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(1.9, 0.0, 1.7), cam_lookat=(0.0, 0.0, 0.85))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_two_arm_lift")
