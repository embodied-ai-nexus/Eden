"""robosuite ``TwoArmHandover``: robot0 lifts a hammer off a narrow table and hands it to robot1.

Port of ``robosuite/environments/manipulation/two_arm_handover.py``. A 0.8 x 0.3 m table (top at
0.8 m) sits at ``(0, -0.45)``; the two Pandas face each other across it (``opposed``, robot0 at
``y = -0.81`` facing +y, robot1 at ``y = +0.81`` facing -y) or stand side by side 1.2 m apart facing
+x (``--arrangement parallel``). A hammer (17.5 cm handle, 4 cm head — robosuite's ``HammerObject``
with its randomized dimensions frozen at their midpoints, see ``hammer.xml``) lies flat on the
table, handle along x, within ``x +- 0.1, y +- 0.05`` of the table centre. Success: robot1 alone
holds the hammer while it is more than 10 cm above the table.

robosuite's ``reward(shaping=True)`` is staged (one value per stage, not summed); here the stages are
separate terms whose sum orders the states the same way:

* ``reaching0`` — ``0.25 (1 - tanh(d0))`` from robot0's grasp point to the handle.
* ``grasping0`` — 0.25 while robot0's pads touch the hammer (``arm0_grasp_any``).
* ``lifting`` — 0.5 while the hammer is lifted (``hammer_height > table_top + 0.1``).
* ``reaching1`` — ``0.25 (1 - tanh(d1))`` from robot1's grasp point to the handle, only once lifted.
* ``grasping1`` — 0.5 while robot1's pads touch the hammer.
* ``handover`` — 2.0 for the success predicate (robot1 grasping, robot0 not, lifted).

Deviations from robosuite. Grasps: ``arm1_grasp_handle`` checks contact with the handle geom only;
the hammer is a single link in Genesis, so both grasp checks count any hammer geom. Lift height:
robosuite subtracts the tilt-dependent reach of the head (``hammer_height = handle_z - r - (L/2 + 2h)
sin(angle)``); here the lift test is on the root: ``handle_z - r > table_top + 0.1``, i.e. root
``z > 0.9175``. Initial orientation: robosuite samples a full turn about the world *y* axis on top of
the lying pose (which pitches the hammer up and lets it fall back onto the table, so it ends up along
+-x); here the hammer is placed flat with a uniform yaw jitter instead, so it may also lie along y.
The prehensile variant only (``prehensile=False`` starts the hammer in robot0's gripper). Fixed hammer
dimensions, :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` instead of 50 s, and the hammer
falling off the table ends the episode.
"""

from __future__ import annotations

import math
from typing import ClassVar

from eden import EventMode, events, metrics, observations, rewards, terminations
from eden.managers.terms.metrics.boolean import MetricTermAND, MetricTermNOT
from eden.options.camera import CamerasOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    fallen_termination,
    NARROW_TABLE_PLACEMENTS,
    NARROW_TABLE_SIZE,
    object_pose_observations,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    table_options,
    TABLE_TOP_Z,
    TwoArmArrangementMod,
)
from eden.options.objects.robosuite import RobosuiteHammer
from eden.tasks.robosuite.two_arm_handover.custom_terms import LiftedReachTanh
from eden.terms import DoneTerm, EventTerm, MetricTerm, ObsTerm
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

#: robosuite's ``table_offset = (0, -0.45, 0.8)``: the narrow table runs along x on robot0's side of the gap.
TABLE_XY = (0.0, -0.45)

_RIG0 = panda_rig("robot0", *NARROW_TABLE_PLACEMENTS["opposed"][0])
_RIG1 = panda_rig("robot1", *NARROW_TABLE_PLACEMENTS["opposed"][1])

#: hammer.xml: handle radius r = 0.0175 (``bottom_offset = -r``), so the lying hammer's root is r above the table.
_HAMMER_RADIUS = 0.0175
_HAMMER_REST_Z = TABLE_TOP_Z + _HAMMER_RADIUS
#: robosuite ``init_quat`` (xyzw ``(0.5, -0.5, 0.5, -0.5)``): handle along +x, head pointing -y, lying flat.
_HAMMER_FLAT_QUAT = (0.5, -0.5, 0.5, -0.5)
#: ``hammer_height - table_top > height_threshold (0.1)`` on the root.
_LIFT_HEIGHT = TABLE_TOP_Z + 0.1 + _HAMMER_RADIUS


def _handover_scene() -> SceneOptions:
    return SceneOptions(
        robot0=_RIG0,
        robot1=_RIG1,
        **arena_entities(),
        mount0=rethink_mount(_RIG0.arm.default_root_pos, _RIG0.arm.default_root_quat),
        mount1=rethink_mount(_RIG1.arm.default_root_pos, _RIG1.arm.default_root_quat),
        table=table_options(NARROW_TABLE_SIZE, xy=TABLE_XY),
        hammer=RobosuiteHammer(
            default_root_pos=(TABLE_XY[0], TABLE_XY[1], _HAMMER_REST_Z), default_root_quat=_HAMMER_FLAT_QUAT
        ),
    )


_OBSERVATIONS = dict(
    **proprio_observations(_RIG0, prefix="robot0_"),
    **proprio_observations(_RIG1, prefix="robot1_"),
    **object_pose_observations("hammer"),
    # robosuite reports ``handle_xpos`` (the handle geom's centre) alongside ``hammer_pos``. In
    # hammer.xml the handle geom sits at the body origin, so the two coincide — there as here. Kept
    # so the observation vector matches robosuite's, and read in the same frame as ``hammer_pos``.
    handle_pos=custom_terms.ObjectLinkPos.configure(object_name="hammer", link_name="handle"),
    gripper0_to_hammer=custom_terms.GripperToObject.configure(rig=_RIG0.arm_name, object_name="hammer"),
    gripper1_to_hammer=custom_terms.GripperToObject.configure(rig=_RIG1.arm_name, object_name="hammer"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, TwoArmArrangementMod(placements=NARROW_TABLE_PLACEMENTS)))
class RobosuiteTwoArmHandoverConfig(RobosuiteTaskConfig):
    robot_slots: ClassVar[tuple[str, ...]] = ("robot0", "robot1")

    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _handover_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG0), **arm_actions(_RIG1))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        arm0_grasp=custom_terms.PadsGrasping.configure(
            rig=_RIG0.arm_name, object_name="hammer", contributes_to_success=False
        ),
        arm1_grasp=custom_terms.PadsGrasping.configure(
            rig=_RIG1.arm_name, object_name="hammer", contributes_to_success=False
        ),
        hammer_lifted=MetricTerm.configure(
            func=metrics.object_above_height,
            params={"entity_name": "hammer", "height": _LIFT_HEIGHT, "relative": True},
            contributes_to_success=False,
        ),
        # robosuite _check_success: arm1_grasp_handle and not arm0_grasp_any and lifted.
        handover=MetricTermAND.configure(
            terms=[
                custom_terms.PadsGrasping.configure(rig=_RIG1.arm_name, object_name="hammer"),
                MetricTermNOT.configure(
                    term=custom_terms.PadsGrasping.configure(rig=_RIG0.arm_name, object_name="hammer")
                ),
                MetricTerm.configure(
                    func=metrics.object_above_height,
                    params={"entity_name": "hammer", "height": _LIFT_HEIGHT, "relative": True},
                ),
            ]
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching0=custom_terms.ReachTanh.configure(rig=_RIG0.arm_name, object_name="hammer", coef=1.0, weight=0.25),
        grasping0=rewards.MetricReward.configure(metric_name="arm0_grasp", weight=0.25),
        lifting=rewards.MetricReward.configure(metric_name="hammer_lifted", weight=0.5),
        reaching1=LiftedReachTanh.configure(
            rig=_RIG1.arm_name, object_name="hammer", gate_metric_name="hammer_lifted", coef=1.0, weight=0.25
        ),
        grasping1=rewards.MetricReward.configure(metric_name="arm1_grasp", weight=0.5),
        handover=rewards.MetricReward.configure(metric_name="handover", weight=2.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        hammer_fallen=fallen_termination("hammer", top_z=TABLE_TOP_Z),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # UniformRandomSampler(x_range=[-0.1, 0.1], y_range=[-0.05, 0.05]) about the table centre; the yaw
        # jitter stands in for robosuite's rotation about y (see the module docstring).
        place_hammer=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "hammer",
                "pos_x_range": (-0.1, 0.1),
                "pos_y_range": (-0.05, 0.05),
                "yaw_range": (-math.pi, math.pi),
            },
        ),
        reset_robot0=arm_init_noise(_RIG0),
        reset_robot1=arm_init_noise(_RIG1),
    )

    # robosuite agentview (0.89, 0, 1.74); pulled back to frame both arms and the table.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(2.3, -0.3, 2.0), cam_lookat=(0.0, -0.2, 0.9))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_two_arm_handover")
