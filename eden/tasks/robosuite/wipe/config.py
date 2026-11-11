"""robosuite ``Wipe``: sweep the end-effector over every dirt marker on a low-friction table.

Port of ``robosuite/environments/manipulation/wipe.py`` with its ``DEFAULT_WIPE_CONFIG``. The Panda
stands at ``(-0.41, 0)`` behind a 0.5 x 0.8 m table whose top is 0.9 m above the floor (friction
0.03); on every reset a random-walk path of dirt markers is laid across the table top. The episode
ends when all of them are wiped, when an arm link hits the table, when an arm joint comes within
0.1 rad of a stop, or on time-out — robosuite's ``early_terminations``, its only early-terminating task.

Reward, robosuite's ``reward(shaping=True)`` term by term (weights are its config values). An
*unsafe* step — an arm link on the table or a joint at its limit — pays ``arm_collision`` alone, as
in robosuite, where it replaces every other component:

* ``unit_wiped`` — 50 per marker wiped for the first time this step.
* ``distance`` — ``5 (1 - tanh(5 d))`` from the grasp point to the centroid of the unwiped markers,
  while any remain.
* ``contact`` — 0.01 while the wrist force exceeds 1 N (``contact_threshold``).
* ``pressure`` — ``-0.05 F`` above 60 N, else ``0.01 + 0.01 F`` above 0.5 N: the excess-force
  penalty and the pressing bonus.
* ``task_complete`` — 25.5 (``50 x (0.01 + 0.5)``) once every marker is wiped.
* ``arm_collision`` — -10 on an unsafe step.

The wrist force is the net contact force on the gripper from everything but the robot
(:class:`~eden.tasks.robosuite.wipe.custom_terms.GripperContactForce`), what robosuite's bias-subtracted
force-torque sensor reads. The observations are robosuite's condensed set plus ``robot0_contact``.

Deviations. robosuite's ``WipingGripper`` is a rigid 12 x 5 cm pad with no fingers; the port keeps
the rig's two-finger gripper, whose action slot the policy still receives (``robot_gripper``) — the
wiping point is the pad midpoint, which does not move with the finger opening. robosuite scatters
100 markers 5 mm apart; the port lays :data:`NUM_MARKERS` (16) 3 cm apart so the path still spans
the table (each marker is one scene entity, so the count is fixed at authoring time — there is no
``--num_markers`` mod). A marker is wiped when the grasp point passes within 2 cm horizontally and
3 cm vertically of it, a geometric stand-in for robosuite's corner-plane test under the pad
(kinematic markers produce no contacts). robosuite's pressing bonus also pays ``+0.1`` while MuJoCo
reports more than 50 contacts scene-wide, a count with no counterpart here; it is dropped, as are the
``ee_accel_penalty`` (0 by default) and the ``1000 / 30510``-style normalisation of the summed reward.
The episode is :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` long rather than 50 s.
"""

from __future__ import annotations

from eden import EventMode, observations, rewards, terminations
from eden.managers.terms.metrics.boolean import MetricTermOR
from eden.options.camera import CamerasOptions
from eden.options.entities import CylinderOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.materials import KinematicMaterialOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
    table_options,
)
from eden.tasks.robosuite.wipe.custom_terms import (
    ArmLinksInContact,
    ArmNearJointLimits,
    GripperContact,
    GripperContactForce,
    GripperInContact,
    GripperToWipeCentroid,
    MarkersWiped,
    NewlyWipedMarkers,
    SampleWipePath,
    WipeCentroid,
    WipeCentroidDistanceTanh,
    WipePressure,
    WipeRadius,
    WipeTaskComplete,
)
from eden.terms import DoneTerm, ObsTerm
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

#: ``DEFAULT_WIPE_CONFIG``: table 0.5 x 0.8 x 0.05 at (0.15, 0), top z 0.9, friction 0.03.
_TABLE_SIZE = (0.5, 0.8, 0.05)
_TABLE_XY = (0.15, 0.0)
_TABLE_TOP_Z = 0.9
_TABLE_FRICTION = 0.03
#: Panda ``base_xpos_offset["table"]`` for the 0.5 m table: ``-0.16 - 0.25``.
_BASE_POS = (-0.41, 0.0, ROBOT_BASE_Z)
_RIG = panda_rig("robot", _BASE_POS)

#: robosuite: 100 markers, ``CylinderObject(size=[0.02, 0.001])``, 5 mm apart. Here 16, 3 cm apart.
NUM_MARKERS = 16
MARKER_NAMES = tuple(f"marker_{i}" for i in range(NUM_MARKERS))
_MARKER_RADIUS = 0.02
_MARKER_HEIGHT = 0.002
_MARKER_COLOR = (0.42, 0.29, 0.16, 1.0)
#: robosuite's path box: ``coverage_factor 0.6`` of the table half-extents minus the marker radius.
_PATH_X_HALF = 0.25 * 0.6 - _MARKER_RADIUS
_PATH_Y_HALF = 0.40 * 0.6 - _MARKER_RADIUS
_PATH_STEP = 0.03
_TABLE_TOP_CENTRE = (_TABLE_XY[0], _TABLE_XY[1], _TABLE_TOP_Z)

#: ``DEFAULT_WIPE_CONFIG`` reward multipliers.
_UNIT_WIPED_REWARD = 50.0
_DISTANCE_MULTIPLIER = 5.0
_DISTANCE_TH_MULTIPLIER = 5.0
_WIPE_CONTACT_REWARD = 0.01
_TASK_COMPLETE_REWARD = _UNIT_WIPED_REWARD * (_WIPE_CONTACT_REWARD + 0.5)
_ARM_COLLISION_PENALTY = -10.0
#: ``DEFAULT_WIPE_CONFIG`` force thresholds [N] and penalty multiplier.
_CONTACT_THRESHOLD = 1.0
_PRESSURE_THRESHOLD = 0.5
_PRESSURE_THRESHOLD_MAX = 60.0
_EXCESS_FORCE_PENALTY_MUL = 0.05
#: ``Robot.check_q_limits``: a joint this close [rad] to a stop is unsafe.
_JOINT_LIMIT_TOLERANCE = 0.1


def _marker(index: int) -> CylinderOptions:
    # Kinematic = not simulated, no contacts, pose settable; the reset event teleports it onto the path.
    # The build-time pose is a line across the table so the markers never start stacked.
    x = _TABLE_XY[0] - _PATH_X_HALF + 2 * _PATH_X_HALF * index / max(NUM_MARKERS - 1, 1)
    return CylinderOptions(
        radius=_MARKER_RADIUS,
        height=_MARKER_HEIGHT,
        material=KinematicMaterialOptions(),
        color=_MARKER_COLOR,
        default_root_pos=(x, _TABLE_XY[1], _TABLE_TOP_Z + _MARKER_HEIGHT / 2),
    )


def _wipe_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        table=table_options(_TABLE_SIZE, top_z=_TABLE_TOP_Z, xy=_TABLE_XY, friction=_TABLE_FRICTION),
        **{name: _marker(i) for i, name in enumerate(MARKER_NAMES)},
    )


_OBSERVATIONS = dict(
    **proprio_observations(_RIG),
    # robosuite's condensed object observables (use_condensed_obj_obs=True).
    proportion_wiped=observations.MetricObservation.configure(metric_name="markers_wiped"),
    wipe_radius=WipeRadius.configure(metric_name="markers_wiped"),
    wipe_centroid=WipeCentroid.configure(metric_name="markers_wiped"),
    gripper_to_wipe_centroid=GripperToWipeCentroid.configure(rig=_RIG.arm_name, metric_name="markers_wiped"),
    robot0_contact=GripperInContact.configure(force_metric="gripper_force", threshold=_CONTACT_THRESHOLD),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod,))
class RobosuiteWipeConfig(RobosuiteTaskConfig):
    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _wipe_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        # Success: every marker wiped (fraction reaches 1.0).
        markers_wiped=MarkersWiped.configure(
            rig=_RIG.arm_name, marker_names=list(MARKER_NAMES), xy_tolerance=0.02, z_tolerance=0.03
        ),
        # robosuite's two unsafe states: check_contact(robot_model) and check_q_limits().
        unsafe=MetricTermOR.configure(
            terms=[
                ArmLinksInContact.configure(rig=_RIG.arm_name, other_name="table"),
                ArmNearJointLimits.configure(rig=_RIG.arm_name, tolerance=_JOINT_LIMIT_TOLERANCE),
            ],
            contributes_to_success=False,
        ),
        gripper_force=GripperContactForce.configure(rig=_RIG.arm_name, contributes_to_success=False),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        unit_wiped=NewlyWipedMarkers.configure(metric_name="markers_wiped", weight=_UNIT_WIPED_REWARD),
        distance=WipeCentroidDistanceTanh.configure(
            rig=_RIG.arm_name, metric_name="markers_wiped", coef=_DISTANCE_TH_MULTIPLIER, weight=_DISTANCE_MULTIPLIER
        ),
        contact=GripperContact.configure(threshold=_CONTACT_THRESHOLD, weight=_WIPE_CONTACT_REWARD),
        pressure=WipePressure.configure(
            min_force=_PRESSURE_THRESHOLD,
            max_force=_PRESSURE_THRESHOLD_MAX,
            excess_penalty=_EXCESS_FORCE_PENALTY_MUL,
            contact_reward=_WIPE_CONTACT_REWARD,
            weight=1.0,
        ),
        task_complete=WipeTaskComplete.configure(metric_name="markers_wiped", weight=_TASK_COMPLETE_REWARD),
        arm_collision=rewards.MetricReward.configure(metric_name="unsafe", weight=_ARM_COLLISION_PENALTY),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        # robosuite early_terminations=True: success, an arm collision and a joint limit all end the episode.
        success=terminations.OnMetricSuccess.configure(metric_name="markers_wiped"),
        unsafe=terminations.MetricAbove.configure(metric_name="unsafe"),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        sample_wipe_path=SampleWipePath.configure(
            mode=EventMode.RESET,
            marker_names=list(MARKER_NAMES),
            origin=_TABLE_TOP_CENTRE,
            x_half=_PATH_X_HALF,
            y_half=_PATH_Y_HALF,
            step=_PATH_STEP,
            lift=_MARKER_HEIGHT / 2,
        ),
        reset_robot=arm_init_noise(_RIG),
    )

    cameras_options: CamerasOptions = agentview_camera(cam_pos=(1.35, 0.0, 1.85), cam_lookat=(0.1, 0.0, 0.9))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_wipe")
