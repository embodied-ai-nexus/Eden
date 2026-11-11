"""robosuite ``Door``: turn the handle and pull a hinged door open past 0.3 rad.

Port of ``robosuite/environments/manipulation/door.py``. The Panda stands on its pedestal at
``(-0.56, 0)``; a 0.8 x 0.3 m table (top at 0.8 m) sits at ``(-0.2, -0.35)`` and the door fixture
(``mj_envs`` ``door_lock.xml``: frame, hinged panel, sprung handle latch) stands over it, its root at
``z = 1.1`` so the panel's lower edge clears the table top by 1 cm; the panel runs along x at
``y ~ -0.34`` with the handle (~1.08 m up, at ``(-0.14, -0.25)``) facing the arm. The latch is a
mechanical lock: its bolt holds the panel shut until the handle is turned past ~1.25 rad (of the
joint's +-1.57), so ``--latch latch`` is the harder task, not just the one with a handle reward.

Reward, as in robosuite's ``reward(shaping=True)``:

* ``reaching`` — ``0.25 (1 - tanh(10 d))`` from the grasp point to the ``handle`` site (12.5 cm along the
  lever, 10 cm off the hub, in the ``latch`` frame — see :mod:`eden.tasks.robosuite.door.custom_terms`).
* ``handle_rotation`` — ``0.25 clip(|handle_qpos| / (pi/2), 0, 1)`` (latch variant only).
* ``opened`` — 1.0 while ``hinge_qpos > 0.3`` (the success predicate).

robosuite returns *only* the 1.0 once the door is open and drops the shaping; Eden sums the terms
(up to 1.5 with the hand still on a turned handle), which keeps the same ordering of states — an
open door always beats a shut one — with a dense signal throughout.

Placement, as robosuite's ``UniformRandomSampler``: the door root is re-posed every reset within
``x in [0.07, 0.09]``, ``y in [-0.01, 0.01]`` and yaw in ``[-pi/2 - 0.25, -pi/2]``, all about
``table_offset``. The door is a fixed-base fixture, and the reset event moves its base per env.

Deviations. The hinge damping / friction loss follow robosuite's override (0.1 / 0.0) via the entity
options, and the episode is :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` long rather than 50 s.

``--latch no_latch`` is robosuite's ``use_latch=False``: ``door.xml`` (a bare handle, no latch joint),
no handle-rotation reward, and the ``door_qpos`` observation shrinks to the hinge alone.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from eden import EventMode, events, observations, rewards, terminations
from eden.options.camera import CamerasOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    object_pose_noise,
    PANDA_TABLE_XY,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
    table_options,
    yaw_quat,
)
from eden.tasks.robosuite.door.custom_terms import DofAbove, DofFractionOfRange
from eden.options.objects.robosuite import RobosuiteDoor, RobosuiteDoorNoLatch
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

_BASE_POS = (PANDA_TABLE_XY[0], PANDA_TABLE_XY[1], ROBOT_BASE_Z)
_RIG = panda_rig("robot", _BASE_POS)

#: robosuite ``TableArena(table_full_size=(0.8, 0.3, 0.05), table_offset=(-0.2, -0.35, 0.8))``.
_TABLE_SIZE = (0.8, 0.3, 0.05)
_TABLE_XY = (-0.2, -0.35)
_TABLE_TOP_Z = 0.8

#: robosuite's door sampler, referenced to ``table_offset``: x in [0.07, 0.09], y in [-0.01, 0.01] about
#: (-0.2, -0.35), yaw in [-pi/2 - 0.25, -pi/2]. The default pose is its centre, the reset jitters about
#: it; root z = table_top - bottom_site_z.
_DOOR_POS = (_TABLE_XY[0] + 0.08, _TABLE_XY[1] + 0.0, _TABLE_TOP_Z + 0.3)
_DOOR_YAW = -math.pi / 2 - 0.125
_DOOR_XY_JITTER = (-0.01, 0.01)
_DOOR_YAW_JITTER = (-0.125, 0.125)
#: The ``handle`` site of ``door_lock.xml`` / ``door.xml``, in the ``latch`` body frame.
_HANDLE_SITE = (0.125, -0.10, 0.0)
_HINGE_SUCCESS = 0.3

_DOOR_OPTIONS: dict[str, type[RobosuiteDoor]] = {"latch": RobosuiteDoor, "no_latch": RobosuiteDoorNoLatch}


def _door_scene() -> SceneOptions:
    return SceneOptions(
        robot=_RIG,
        **arena_entities(),
        mount=rethink_mount(_BASE_POS),
        table=table_options(_TABLE_SIZE, top_z=_TABLE_TOP_Z, xy=_TABLE_XY),
        door=_door_options("latch"),
    )


def _door_options(latch: str) -> RobosuiteDoor:
    door = _DOOR_OPTIONS[latch](
        default_root_pos=_DOOR_POS,
        default_root_quat=yaw_quat(_DOOR_YAW),
        # robosuite DoorObject(friction=0.0, damping=0.1) overrides the XML's hinge damping/frictionloss.
        default_dofs_damping={"hinge": 0.1},
        default_dofs_frictionloss={"hinge": 0.0},
    )
    # The XML declares no armature, so MuJoCo runs both joints at 0; Eden's loader would fill in 0.1.
    door.default_dofs_armature = dict.fromkeys(door.dofs_name, 0.0)
    return door


_OBSERVATIONS = dict(
    **proprio_observations(_RIG),
    door_pos=custom_terms.ObjectLinkPos.configure(object_name="door", link_name="door", noise=object_pose_noise()),
    handle_pos=custom_terms.ObjectPointPos.configure(
        object_name="door", object_link_name="latch", object_offset=_HANDLE_SITE, noise=object_pose_noise()
    ),
    # hinge_qpos (+ handle_qpos with the latch): every DOF of the door, in dofs_name order.
    door_qpos=observations.DofsPos.configure(entity_name="door"),
    door_to_eef=custom_terms.GripperToObject.configure(rig=_RIG.arm_name, object_name="door", object_link_name="door"),
    handle_to_eef=custom_terms.GripperToObject.configure(
        rig=_RIG.arm_name, object_name="door", object_link_name="latch", object_offset=_HANDLE_SITE
    ),
    last_action=ObsTerm.configure(func=observations.last_action),
)


class DoorLatchMod(TaskMod):
    """robosuite's ``use_latch`` switch (``--latch``).

    ``latch`` (default) is ``door_lock.xml``: a sprung handle on its own joint plus the
    handle-rotation reward. ``no_latch`` is ``door.xml``: a fixed handle, hinge only, and the reward
    keeps just the reach and open terms; ``door_qpos`` shrinks from 2 to 1.
    """

    def __init__(
        self,
        latch: Annotated[
            Literal["latch", "no_latch"],
            "Door model: latch (robosuite use_latch=True, sprung handle joint) or no_latch (bare handle).",
        ] = "latch",
    ):
        self.latch = latch

    def apply(self, config: RobosuiteDoorConfig) -> RobosuiteDoorConfig:
        if self.latch == "latch":
            return config  # the config is authored around the latched door
        config.scene_options.door = _door_options(self.latch)
        config.reward_options.__pydantic_extra__.pop("handle_rotation")
        config.runner_options.experiment_name = f"{config.runner_options.experiment_name}_{self.latch}"
        return config


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, DoorLatchMod))
class RobosuiteDoorConfig(RobosuiteTaskConfig):
    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _door_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        door_open=DofAbove.configure(entity_name="door", dof_name="hinge", threshold=_HINGE_SUCCESS),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching=custom_terms.ReachTanh.configure(
            rig=_RIG.arm_name,
            object_name="door",
            object_link_name="latch",
            object_offset=_HANDLE_SITE,
            coef=10.0,
            weight=0.25,
        ),
        handle_rotation=DofFractionOfRange.configure(entity_name="door", dof_name="latch_joint", weight=0.25),
        opened=rewards.MetricReward.configure(metric_name="door_open", weight=1.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        place_door=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "door",
                "pos_x_range": _DOOR_XY_JITTER,
                "pos_y_range": _DOOR_XY_JITTER,
                "yaw_range": _DOOR_YAW_JITTER,
            },
        ),
        # Closed, handle at rest (zero-width offset range).
        reset_door=EventTerm.configure(
            func=events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={"entity_name": "door", "dofs_pos_range": (0.0, 0.0)},
        ),
        reset_robot=arm_init_noise(_RIG),
    )

    # robosuite moves agentview to (0.6, 0, 1.59) looking down -x; pulled towards +y so the handle face of
    # the panel, the table under it and the arm are all in frame.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(0.9, 0.9, 1.8), cam_lookat=(-0.25, -0.3, 1.0))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_door")
