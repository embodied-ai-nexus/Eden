"""robosuite ``Lift``: pick a small cube off the table and raise it 4 cm.

Port of ``robosuite/environments/manipulation/lift.py``. The Panda stands on its pedestal 0.56 m
behind the centre of a 0.8 m table (top at 0.8 m); a 4.2 cm cube spawns within ±3 cm of the table
centre at a random yaw.

Reward, as in robosuite's ``reward(shaping=True)``:

* ``reaching`` — ``1 - tanh(10 d)`` from the grasp point to the cube (weight 1).
* ``grasping`` — 0.25 while both fingerpads touch the cube (``_check_grasp``).
* ``lifting``  — 2.25 while the cube centre is above ``table_top + 0.04`` (the success predicate).

robosuite returns *only* the 2.25 on success and drops the shaping; Eden sums the terms, which
keeps the same ordering of states (a lifted cube always beats an unlifted one) with a dense signal
throughout. Deviations forced by Genesis: the cube half-size is fixed at 0.021 m (robosuite resamples
it in ``[0.020, 0.022]`` per reset), and the episode is :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S`
long rather than 50 s. The cube falling off the table ends the episode early.
"""

from __future__ import annotations

from eden import EventMode, events, metrics, observations, rewards, terminations
from eden.options.camera import CamerasOptions
from eden.options.entities import BoxOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.materials import RigidMaterialOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    fallen_termination,
    object_pose_observations,
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

_CUBE_HALF = 0.021
_CUBE_REST_Z = TABLE_TOP_Z + _CUBE_HALF
_LIFT_HEIGHT = TABLE_TOP_Z + 0.04


def _lift_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        table=table_options(),
        cube=BoxOptions(
            size=(2 * _CUBE_HALF,) * 3,
            default_root_pos=(0.0, 0.0, _CUBE_REST_Z),
            material=RigidMaterialOptions(rho=1000.0, friction=1.0),
            color=(0.80, 0.10, 0.10, 1.0),
        ),
    )


_OBSERVATIONS = dict(
    **proprio_observations(_RIG),
    **object_pose_observations("cube"),
    gripper_to_cube=custom_terms.GripperToObject.configure(rig=_RIG.arm_name, object_name="cube"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod,))
class RobosuiteLiftConfig(RobosuiteTaskConfig):
    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _lift_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        grasping=custom_terms.PadsGrasping.configure(
            rig=_RIG.arm_name, object_name="cube", contributes_to_success=False
        ),
        lifted=MetricTerm.configure(
            func=metrics.object_above_height, params={"entity_name": "cube", "height": _LIFT_HEIGHT, "relative": True}
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching=custom_terms.ReachTanh.configure(rig=_RIG.arm_name, object_name="cube", coef=10.0, weight=1.0),
        grasping=rewards.MetricReward.configure(metric_name="grasping", weight=0.25),
        lifting=rewards.MetricReward.configure(metric_name="lifted", weight=2.25),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        cube_fallen=fallen_termination("cube", top_z=TABLE_TOP_Z),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # UniformRandomSampler(x_range=[-0.03, 0.03], y_range=[-0.03, 0.03], rotation=None) about the table centre.
        place_cube=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "cube",
                "pos_x_range": (-0.03, 0.03),
                "pos_y_range": (-0.03, 0.03),
                "yaw_range": (-3.14159, 3.14159),
            },
        ),
        reset_robot=arm_init_noise(_RIG),
    )

    cameras_options: CamerasOptions = agentview_camera()

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_lift")
