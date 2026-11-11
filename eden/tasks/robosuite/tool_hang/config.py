"""robosuite ``ToolHang``: stand the L-shaped frame in the stand's socket, then hang the wrench on it.

Port of ``robosuite/environments/manipulation/tool_hang.py``. Same arena as Lift (Panda 0.56 m
behind the centre of a 0.8 m table). Three objects, authored as MJCF from robosuite's composite
geometry (``objects/tool_hang_*.xml`` in the ``eden_robosuite`` dataset):

* ``stand`` — a 12 x 14 cm slab with a 15 cm tall, 12 mm square socket. robosuite gives it an
  8 kg base so it never moves; here it is a **fixed fixture** at robosuite's spawn pose.
* ``frame`` — the L: a 17 cm rod with a steel tip, a 9.5 cm hook bar across the top and a grip
  block. It spawns lying flat 24 cm to the robot's right, rod pointing away from the stand.
* ``tool`` — a ratcheting wrench (flat handle, a ring at each end, a grip block), flat on the
  table next to the frame. Its ``hole1`` ring is the one that goes onto the bar.

robosuite's reward is **sparse**: 1.0 for ``frame_assembled and tool_on_frame``, and it has no dense
version. Only ``success`` below is robosuite's; the other five terms are Eden's own shaping, in the
order the task is solved, so a policy can train from scratch:

* ``reach_frame`` — ``0.1 (1 - tanh(10 d))`` from the grasp point to the frame;
* ``frame_lifted`` — 0.5 while the frame is 4 cm off the table;
* ``frame_assembled`` — 1.0 while the frame stands in the socket;
* ``reach_tool`` — ``0.1 (1 - tanh(10 d))`` to the wrench;
* ``tool_on_frame`` — 2.0 while the ring hangs on the bar, released;
* ``success`` — robosuite's 1.0 for both at once.

The success predicates are robosuite's, test for test (:mod:`eden.tasks.robosuite.tool_hang.custom_terms`).
:class:`FrameAssembled` is ``_check_frame_assembled``: the stand upright, the frame's tip within 5 cm
of the stand's base slab, and the rod between both pairs of opposite socket walls. ``tool_on_frame``
is ``_check_tool_on_frame``: the ring touches the bar, no part of the gripper touches the wrench,
and :class:`ToolHoleOnBar` — the hole centre within 6.75 mm of the bar axis, the bar through the
ring, and the hole threaded onto the bar rather than past its end.

Placement follows robosuite's sampler (each object its own +-2 cm box, +-10 degree yaw jitter).
robosuite quotes the frame's spawn as a rotation about *y*, but its sampler multiplies a wxyz
quaternion as xyzw, and the pose MuJoCo actually receives is the frame lying flat with a yaw of
150 +- 10 degrees — that effective pose is what ``default_root_quat`` holds, jittered by
``events.reset_base_state_uniform`` (which composes the yaw onto it and places the MJCF body origin,
robosuite's frame position, not Genesis' centre-of-mass re-origin 5 cm away).
Other deviations: the wrench's contacts are stiffened (``GRASPED_SOL_PARAMS``: on Genesis' default
time constant it turns in a pinch three times as far), the cone tip is 5 stacked boxes instead of
51, the episode is :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` rather than 50 s, and the
frame or wrench falling off the table ends the episode.
"""

from __future__ import annotations

import math

from eden import EventMode, events, metrics, observations, rewards, terminations
from eden.managers.terms.metrics.boolean import MetricTermAND, MetricTermNOT
from eden.options.camera import CamerasOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.managers.metrics import MetricTermOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.common import (
    GRASPED_SOL_PARAMS,
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
    yaw_quat,
)
from eden.options.objects.robosuite import RobosuiteToolHangFrame, RobosuiteToolHangStand, RobosuiteToolHangWrench
from eden.tasks.robosuite.tool_hang.custom_terms import (
    FrameAssembled,
    GripperTouching,
    ToolHoleOnBar,
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

# robosuite sampler: stand at (-0.1 L, 0), frame at (-0.05 L, -0.3 W), tool at (0.05 L, -0.25 W), L = W = 0.8.
_STAND_POS = (-0.08, 0.0, TABLE_TOP_Z + 0.081)  # 16 cm tall, bottom 1 mm above the table
_FRAME_HOME = (-0.04, -0.24, TABLE_TOP_Z + 0.0137)  # robosuite's sampler centre; flat on its 2.54 cm grip block
_FRAME_QUAT = (0.1830127, -0.1830127, -0.6830127, 0.6830127)  # flat, rod heading 150 deg (see module docstring)
_TOOL_HOME = (0.04, -0.20, TABLE_TOP_Z + 0.016)  # flat on its 3 cm grip block
_TOOL_YAW = -math.pi / 2 - math.pi / 9
_JITTER_XY = (-0.02, 0.02)
_JITTER_YAW = math.pi / 18
_FRAME_LIFT_HEIGHT = TABLE_TOP_Z + 0.04


def _tool_hang_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        table=table_options(),
        stand=RobosuiteToolHangStand(default_root_pos=_STAND_POS),
        frame=RobosuiteToolHangFrame(default_root_pos=_FRAME_HOME, default_root_quat=_FRAME_QUAT),
        tool=RobosuiteToolHangWrench(
            default_root_pos=_TOOL_HOME,
            default_root_quat=yaw_quat(_TOOL_YAW),
            default_geoms_sol_params={".*": GRASPED_SOL_PARAMS},
        ),
    )


_OBSERVATIONS = dict(
    **proprio_observations(_RIG),
    **object_pose_observations("stand"),
    **object_pose_observations("frame"),
    **object_pose_observations("tool"),
    gripper_to_frame=custom_terms.GripperToObject.configure(rig=_RIG.arm_name, object_name="frame"),
    gripper_to_tool=custom_terms.GripperToObject.configure(rig=_RIG.arm_name, object_name="tool"),
    frame_is_assembled=observations.MetricObservation.configure(metric_name="frame_assembled"),
    tool_on_frame=observations.MetricObservation.configure(metric_name="tool_on_frame"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


def _tool_on_frame_terms() -> list[MetricTermOptions]:
    """The three tests of robosuite's ``_check_tool_on_frame``.

    Ring on bar, gripper off the wrench, hole threaded on the bar.
    """
    return [
        custom_terms.EntitiesInContact.configure(
            entity_a_name="tool", entity_b_name="frame", links_a=["tool_hole1"], links_b=["frame_horizontal_frame"]
        ),
        # robosuite: none of the gripper's geoms touches the wrench — any contact, not a two-pad grasp.
        MetricTermNOT.configure(term=GripperTouching.configure(rig=_RIG.arm_name, object_name="tool")),
        ToolHoleOnBar.configure(tool_name="tool", frame_name="frame"),
    ]


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod,))
class RobosuiteToolHangConfig(RobosuiteTaskConfig):
    # Three multi-geom objects on one table: the wrench's 16 ring boxes alone can hold 40 contact pairs.
    env_options: EnvOptions = robosuite_env_options(max_collision_pairs=100)

    scene_options: SceneOptions = _tool_hang_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        frame_lifted=MetricTerm.configure(
            func=metrics.object_above_height,
            params={"entity_name": "frame", "height": _FRAME_LIFT_HEIGHT, "relative": True},
            contributes_to_success=False,
        ),
        frame_assembled=FrameAssembled.configure(frame_name="frame", stand_name="stand", contributes_to_success=False),
        tool_on_frame=MetricTermAND.configure(terms=_tool_on_frame_terms(), contributes_to_success=False),
        # robosuite _check_success: both stages at once.
        success=MetricTermAND.configure(
            terms=[FrameAssembled.configure(frame_name="frame", stand_name="stand"), *_tool_on_frame_terms()]
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reach_frame=custom_terms.ReachTanh.configure(rig=_RIG.arm_name, object_name="frame", coef=10.0, weight=0.1),
        frame_lifted=rewards.MetricReward.configure(metric_name="frame_lifted", weight=0.5),
        frame_assembled=rewards.MetricReward.configure(metric_name="frame_assembled", weight=1.0),
        reach_tool=custom_terms.ReachTanh.configure(rig=_RIG.arm_name, object_name="tool", coef=10.0, weight=0.1),
        tool_on_frame=rewards.MetricReward.configure(metric_name="tool_on_frame", weight=2.0),
        success=rewards.MetricReward.configure(metric_name="success", weight=1.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        frame_fallen=fallen_termination("frame", top_z=TABLE_TOP_Z),
        tool_fallen=fallen_termination("tool", top_z=TABLE_TOP_Z),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Both lie on their side: the yaw jitter composes onto default_root_quat.
        place_frame=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "frame",
                "pos_x_range": _JITTER_XY,
                "pos_y_range": _JITTER_XY,
                "yaw_range": (-_JITTER_YAW, _JITTER_YAW),
            },
        ),
        place_tool=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "tool",
                "pos_x_range": _JITTER_XY,
                "pos_y_range": _JITTER_XY,
                "yaw_range": (-_JITTER_YAW, _JITTER_YAW),
            },
        ),
        reset_robot=arm_init_noise(_RIG),
    )

    # robosuite's ToolHang agentview sits front-right of the table; pulled back so the stand, frame and tool fit.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(1.1, 0.55, 1.45), cam_lookat=(-0.05, -0.1, 0.85))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_tool_hang")
