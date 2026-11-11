"""robosuite ``Stack``: put the small red cube on top of the larger green one.

Port of ``robosuite/environments/manipulation/stack.py``. Same arena as Lift; ``cubeA`` (4 cm, red)
and ``cubeB`` (5 cm, green) spawn within ±8 cm of the table centre in x, each in its own y half-plane,
at random yaws.

robosuite's ``staged_rewards`` are ``max``-ed into one scalar; here they are separate terms whose
sum orders states the same way:

* ``reaching`` — ``0.25 (1 - tanh(10 d))`` to cubeA, ``grasping`` — 0.25 while both pads touch cubeA.
* ``lifting`` — 1.0 while cubeA is 4 cm above the table, ``aligning`` — ``0.5 (1 - tanh(d_xy))`` on the
  A→B horizontal offset, gated on ``lifting``.
* ``stacked`` — 2.0 for the success predicate: cubeA lifted, touching cubeB, and released.

Deviations: :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` instead of 50 s, either cube falling
off the table ends the episode, and the spawn split into y half-planes stands in for robosuite's
overlap rejection. The cube sizes are fixed, as in robosuite.
"""

from __future__ import annotations

from eden import EventMode, events, metrics, observations, rewards, terminations
from eden.managers.terms.metrics.boolean import MetricTermAND, MetricTermNOT
from eden.options.camera import CamerasOptions
from eden.options.entities import BoxOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.managers.events import EventTermOptions
from eden.options.materials import RigidMaterialOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.stack.custom_terms import LiftedAlignXY
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    fallen_termination,
    object_observations,
    PANDA_TABLE_XY,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
    table_options,
    TABLE_TOP_Z,
)
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

_BASE_POS = (PANDA_TABLE_XY[0], PANDA_TABLE_XY[1], ROBOT_BASE_Z)
_RIG = panda_rig("robot", _BASE_POS)

_CUBE_A_HALF = 0.02
_CUBE_B_HALF = 0.025
_LIFT_HEIGHT = TABLE_TOP_Z + 0.04
_PLACE_RANGE = (-0.08, 0.08)


def _cube(half: float, color: tuple[float, float, float, float], home_y: float) -> BoxOptions:
    return BoxOptions(
        size=(2 * half,) * 3,
        default_root_pos=(0.0, home_y, TABLE_TOP_Z + half),
        material=RigidMaterialOptions(rho=1000.0, friction=1.0),
        color=color,
    )


def _stack_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        table=table_options(),
        # Home poses 12 cm apart so the two independently jittered cubes never spawn overlapping.
        cubeA=_cube(_CUBE_A_HALF, (0.80, 0.10, 0.10, 1.0), home_y=-0.06),
        cubeB=_cube(_CUBE_B_HALF, (0.10, 0.60, 0.15, 1.0), home_y=0.06),
    )


_OBSERVATIONS = dict(
    **proprio_observations(_RIG),
    **object_observations(_RIG, "cubeA"),
    **object_observations(_RIG, "cubeB"),
    cubeA_to_cubeB=custom_terms.ObjectToObject.configure(object_a_name="cubeA", object_b_name="cubeB"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


def _place_event(name: str) -> EventTermOptions:
    return EventTerm.configure(
        func=events.reset_base_state_uniform,
        mode=EventMode.RESET,
        params={
            "entity_name": name,
            "pos_x_range": _PLACE_RANGE,
            "pos_y_range": (-0.02, 0.02),
            "yaw_range": (-3.14159, 3.14159),
        },
    )


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod,))
class RobosuiteStackConfig(RobosuiteTaskConfig):
    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _stack_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        grasping=custom_terms.PadsGrasping.configure(
            rig=_RIG.arm_name, object_name="cubeA", contributes_to_success=False
        ),
        lifted=MetricTerm.configure(
            func=metrics.object_above_height,
            params={"entity_name": "cubeA", "height": _LIFT_HEIGHT, "relative": True},
            contributes_to_success=False,
        ),
        touching=custom_terms.EntitiesInContact.configure(
            entity_a_name="cubeA", entity_b_name="cubeB", contributes_to_success=False
        ),
        # robosuite r_stack: lifted & touching cubeB & not grasped.
        stacked=MetricTermAND.configure(
            terms=[
                MetricTerm.configure(
                    func=metrics.object_above_height,
                    params={"entity_name": "cubeA", "height": _LIFT_HEIGHT, "relative": True},
                ),
                custom_terms.EntitiesInContact.configure(entity_a_name="cubeA", entity_b_name="cubeB"),
                MetricTermNOT.configure(
                    term=custom_terms.PadsGrasping.configure(rig=_RIG.arm_name, object_name="cubeA")
                ),
            ]
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching=custom_terms.ReachTanh.configure(rig=_RIG.arm_name, object_name="cubeA", coef=10.0, weight=0.25),
        grasping=rewards.MetricReward.configure(metric_name="grasping", weight=0.25),
        lifting=rewards.MetricReward.configure(metric_name="lifted", weight=1.0),
        aligning=LiftedAlignXY.configure(
            object_name="cubeA", target_name="cubeB", lift_height=_LIFT_HEIGHT, coef=1.0, weight=0.5
        ),
        stacked=rewards.MetricReward.configure(metric_name="stacked", weight=2.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        cubeA_fallen=fallen_termination("cubeA", top_z=TABLE_TOP_Z),
        cubeB_fallen=fallen_termination("cubeB", top_z=TABLE_TOP_Z),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # robosuite samples both cubes in x, y in [-0.08, 0.08] with rejection of overlaps; Eden has
        # no rejection sampler, so each cube keeps its own y half-plane (home_y +- 2 cm) and the full x range.
        place_cubeA=_place_event("cubeA"),
        place_cubeB=_place_event("cubeB"),
        reset_robot=arm_init_noise(_RIG),
    )

    cameras_options: CamerasOptions = agentview_camera()

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_stack")
