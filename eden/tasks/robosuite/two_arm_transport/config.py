"""robosuite ``TwoArmTransport``: uncover a payload, carry it to the far table's target bin, clear the trash.

Port of ``robosuite/environments/manipulation/two_arm_transport.py``. Two 0.8 x 0.3 m tables (tops at
0.8 m) at ``y = -0.45`` and ``y = +0.45``; the Pandas face each other across them (``opposed``,
robot0 at ``y = -0.81`` facing +y, robot1 at ``y = +0.81`` facing -y) or stand side by side facing
+x (``--arrangement parallel``). On robot0's table a lidded ``start_bin`` (at ``x = 0.2``) holds the
payload — a small hammer (``hammer_payload.xml``: 20 cm handle, 1.5 cm radius). On robot1's table the
``target_bin`` (at ``x = -0.2``) holds a 4 cm red ``trash`` cube and the empty ``trash_bin`` stands at
``x = 0.2``. The team has to lift the lid, take the payload out, hand it across into the target bin,
and move the trash cube from the target bin into the trash bin. Success (the only robosuite reward,
sparse): the payload rests on the target bin's base slab and the trash on the trash bin's base slab.

robosuite implements **no** dense reward for this task (``reward_shaping=True`` is ignored with a
warning). The shaping here is Eden's own and is marked as a deviation; it follows the natural stage
order with small weights so the sparse ``transported`` term (1.0) dominates:

* ``reaching0_lid`` 0.1, ``lid_lifted`` 0.2 (lid root 10 cm above its start), ``reaching0_payload`` 0.1,
  ``payload_lifted`` 0.3 (payload root above the bin walls), ``reaching1_trash`` 0.1, ``trash_lifted`` 0.2,
  ``payload_in_target`` 1.0 and ``trash_in_trash`` 1.0 (the two halves of the success predicate).

Other deviations. The bins are fixtures (robosuite's are free but weigh 26 kg and never move; free in
Genesis they sink 2 cm into the table under the soft contact) — so the ``pos_tol = 0.005`` bin
jitter is dropped, and the lid, which starts on the start bin, is not jittered either. The trash keeps
robosuite's yaw jitter (``+- 0.3 pi``) about its bin's centre; the payload's pitch jitter about y is cut
from ``+- pi/6`` to ``+- pi/18``: at 30 degrees off vertical the 25 cm hammer's head is 11 cm below its
origin, which in robosuite spawns it several centimetres inside the bin base (MuJoCo pushes it out),
whereas Genesis would launch it. The lid, payload and trash — the three objects a hand picks up —
carry a 0.005 s contact time constant (``GRASPED_SOL_PARAMS``): at Genesis' default 0.02 s the
gripper's 20 N pinch sinks the pads 6 mm into the hammer's handle and the hammer creeps out of the
fingers within seconds, so no hand-off holds. Fixed payload dimensions (robosuite resamples the head size in
``[0.015, 0.018]``), :data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` instead of 50 s, and the
payload, trash or lid falling off a table ends the episode.
"""

from __future__ import annotations

import math
from typing import ClassVar

from eden import EventMode, events, metrics, observations, rewards, terminations
from eden.managers.terms.metrics.boolean import MetricTermAND
from eden.options.camera import CamerasOptions
from eden.options.entities import BoxOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.managers.metrics import MetricTermOptions
from eden.options.materials import RigidMaterialOptions
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
    NARROW_TABLE_PLACEMENTS,
    NARROW_TABLE_SIZE,
    object_pose_observations,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    rpy_quat,
    table_options,
    TABLE_TOP_Z,
    TwoArmArrangementMod,
)
from eden.options.objects.robosuite import RobosuiteHammerPayload, RobosuiteLid, RobosuiteTransportBin
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

_RIG0 = panda_rig("robot0", *NARROW_TABLE_PLACEMENTS["opposed"][0])
_RIG1 = panda_rig("robot1", *NARROW_TABLE_PLACEMENTS["opposed"][1])

#: robosuite ``MultiTableArena``: table0 (robot0's side) at y = -0.45, table1 at y = +0.45.
_TABLE0_Y = -0.45
_TABLE1_Y = 0.45

#: bin.xml: ``Bin(bin_size=(0.3, 0.3, 0.15))`` — total half-height 0.075, base slab 1 cm thick at z = -0.07.
_BIN_HALF_HEIGHT = 0.075
_BIN_ROOT_Z = TABLE_TOP_Z + _BIN_HALF_HEIGHT
_BIN_FLOOR_Z = _BIN_ROOT_Z - 0.07 + 0.005  # top surface of the base slab
_BIN_RIM_Z = _BIN_ROOT_Z + _BIN_HALF_HEIGHT
_START_BIN_XY = (0.2, _TABLE0_Y)
_TARGET_BIN_XY = (-0.2, _TABLE1_Y)
_TRASH_BIN_XY = (0.2, _TABLE1_Y)

#: lid.xml: ``Lid(lid_size=(0.3, 0.3, 0.01))`` — total half-height 0.02; it rests on the bin rim.
_LID_REST_Z = _BIN_RIM_Z + 0.02
#: hammer_payload.xml: handle radius 0.015, head half-size 0.0165. robosuite's ``pi/2`` about y lays the handle
#: along x with the head bar vertical, face down: the face end is 5.3 cm below the body origin, so the origin
#: starts 7.5 cm above the bin floor (2 cm drop at zero pitch jitter, touching at the +-10 degree extremes).
_PAYLOAD_START_Z = _BIN_FLOOR_Z + 0.075
_PAYLOAD_PITCHED_QUAT = rpy_quat(0.0, math.pi / 2, 0.0)
_PAYLOAD_PITCH_JITTER = math.pi / 18
#: BoxObject(size=[0.02] * 3): 4 cm cube resting on the target bin floor.
_TRASH_HALF = 0.02
_TRASH_REST_Z = _BIN_FLOOR_Z + _TRASH_HALF

#: Shaping thresholds (Eden's, see the module docstring). Measured against the body origin
#: (``object_above_height`` with ``relative=True``, the frame these heights are reasoned from), which
#: is also the frame the object-pose observables report.
_LID_LIFT_HEIGHT = _BIN_RIM_Z + 0.1
_PAYLOAD_LIFT_HEIGHT = _BIN_RIM_Z
_TRASH_LIFT_HEIGHT = _BIN_RIM_Z


def _bin(xy: tuple[float, float]) -> RobosuiteTransportBin:
    return RobosuiteTransportBin(default_root_pos=(xy[0], xy[1], _BIN_ROOT_Z))


def _transport_scene() -> SceneOptions:
    return SceneOptions(
        robot0=_RIG0,
        robot1=_RIG1,
        **arena_entities(),
        mount0=rethink_mount(_RIG0.arm.default_root_pos, _RIG0.arm.default_root_quat),
        mount1=rethink_mount(_RIG1.arm.default_root_pos, _RIG1.arm.default_root_quat),
        table0=table_options(NARROW_TABLE_SIZE, xy=(0.0, _TABLE0_Y)),
        table1=table_options(NARROW_TABLE_SIZE, xy=(0.0, _TABLE1_Y)),
        start_bin=_bin(_START_BIN_XY),
        target_bin=_bin(_TARGET_BIN_XY),
        trash_bin=_bin(_TRASH_BIN_XY),
        lid=RobosuiteLid(
            default_root_pos=(_START_BIN_XY[0], _START_BIN_XY[1], _LID_REST_Z),
            default_geoms_sol_params={".*": GRASPED_SOL_PARAMS},
        ),
        payload=RobosuiteHammerPayload(
            default_root_pos=(_START_BIN_XY[0], _START_BIN_XY[1], _PAYLOAD_START_Z),
            default_root_quat=_PAYLOAD_PITCHED_QUAT,
            default_geoms_sol_params={".*": GRASPED_SOL_PARAMS},
        ),
        trash=BoxOptions(
            size=(2 * _TRASH_HALF,) * 3,
            default_root_pos=(_TARGET_BIN_XY[0], _TARGET_BIN_XY[1], _TRASH_REST_Z),
            material=RigidMaterialOptions(rho=1000.0, friction=1.0),
            color=(0.80, 0.10, 0.10, 1.0),
            default_geoms_sol_params={".*": GRASPED_SOL_PARAMS},
        ),
    )


_OBSERVATIONS = dict(
    **proprio_observations(_RIG0, prefix="robot0_"),
    **proprio_observations(_RIG1, prefix="robot1_"),
    **object_pose_observations("payload"),
    **object_pose_observations("trash"),
    **object_pose_observations("lid"),
    # robosuite reports the bins' base geom positions; the base slab is its own link here.
    target_bin_pos=custom_terms.ObjectLinkPos.configure(object_name="target_bin", link_name="base"),
    trash_bin_pos=custom_terms.ObjectLinkPos.configure(object_name="trash_bin", link_name="base"),
    payload_in_target=observations.MetricObservation.configure(metric_name="payload_in_target"),
    trash_in_trash=observations.MetricObservation.configure(metric_name="trash_in_trash"),
    gripper0_to_payload=custom_terms.GripperToObject.configure(rig=_RIG0.arm_name, object_name="payload"),
    gripper1_to_payload=custom_terms.GripperToObject.configure(rig=_RIG1.arm_name, object_name="payload"),
    gripper0_to_lid=custom_terms.GripperToObject.configure(rig=_RIG0.arm_name, object_name="lid"),
    gripper1_to_trash=custom_terms.GripperToObject.configure(rig=_RIG1.arm_name, object_name="trash"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


def _in_bin(obj: str, bin_name: str) -> MetricTermOptions:
    """Contact metric: robosuite ``check_contact(<bin>_base geom, <obj> geoms)``."""
    return custom_terms.EntitiesInContact.configure(entity_a_name=obj, entity_b_name=bin_name, links_b=["base"])


def _above(name: str, height: float) -> MetricTermOptions:
    return MetricTerm.configure(
        func=metrics.object_above_height,
        params={"entity_name": name, "height": height, "relative": True},
        contributes_to_success=False,
    )


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, TwoArmArrangementMod(placements=NARROW_TABLE_PLACEMENTS)))
class RobosuiteTwoArmTransportConfig(RobosuiteTaskConfig):
    robot_slots: ClassVar[tuple[str, ...]] = ("robot0", "robot1")

    # 3 bins x 5 geoms + 2 tables + lid + hammer + cube + two arms: more potential contact pairs than the
    # single-arm tasks.
    env_options: EnvOptions = robosuite_env_options(max_collision_pairs=80)

    scene_options: SceneOptions = _transport_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**arm_actions(_RIG0), **arm_actions(_RIG1))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        payload_in_target=custom_terms.EntitiesInContact.configure(
            entity_a_name="payload", entity_b_name="target_bin", links_b=["base"], contributes_to_success=False
        ),
        trash_in_trash=custom_terms.EntitiesInContact.configure(
            entity_a_name="trash", entity_b_name="trash_bin", links_b=["base"], contributes_to_success=False
        ),
        lid_lifted=_above("lid", _LID_LIFT_HEIGHT),
        payload_lifted=_above("payload", _PAYLOAD_LIFT_HEIGHT),
        trash_lifted=_above("trash", _TRASH_LIFT_HEIGHT),
        # robosuite _check_success: payload_in_target_bin and trash_in_trash_bin.
        transported=MetricTermAND.configure(terms=[_in_bin("payload", "target_bin"), _in_bin("trash", "trash_bin")]),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching0_lid=custom_terms.ReachTanh.configure(rig=_RIG0.arm_name, object_name="lid", coef=1.0, weight=0.1),
        lid_lifted=rewards.MetricReward.configure(metric_name="lid_lifted", weight=0.2),
        reaching0_payload=custom_terms.ReachTanh.configure(
            rig=_RIG0.arm_name, object_name="payload", coef=1.0, weight=0.1
        ),
        payload_lifted=rewards.MetricReward.configure(metric_name="payload_lifted", weight=0.3),
        reaching1_trash=custom_terms.ReachTanh.configure(rig=_RIG1.arm_name, object_name="trash", coef=1.0, weight=0.1),
        trash_lifted=rewards.MetricReward.configure(metric_name="trash_lifted", weight=0.2),
        payload_in_target=rewards.MetricReward.configure(metric_name="payload_in_target", weight=1.0),
        trash_in_trash=rewards.MetricReward.configure(metric_name="trash_in_trash", weight=1.0),
        transported=rewards.MetricReward.configure(metric_name="transported", weight=1.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        payload_fallen=fallen_termination("payload"),
        trash_fallen=fallen_termination("trash"),
        lid_fallen=fallen_termination("lid"),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # robosuite: payload at the start bin's centre (pos_tol 0.005), rotation pi/2 +- pi/6 about y (here
        # +- pi/18, see the module docstring); trash at the target bin's centre, yaw +- 0.3 pi.
        place_payload=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "payload",
                "pos_x_range": (-0.005, 0.005),
                "pos_y_range": (-0.005, 0.005),
                "pitch_range": (-_PAYLOAD_PITCH_JITTER, _PAYLOAD_PITCH_JITTER),
                "yaw_range": (0.0, 0.0),
            },
        ),
        place_trash=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "trash",
                "pos_x_range": (-0.005, 0.005),
                "pos_y_range": (-0.005, 0.005),
                "yaw_range": (-0.3 * math.pi, 0.3 * math.pi),
            },
        ),
        # The lid has no jitter in robosuite beyond the bin's 5 mm; it is re-seated on the start bin.
        place_lid=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "lid",
                "pos_x_range": (0.0, 0.0),
                "pos_y_range": (0.0, 0.0),
                "yaw_range": (0.0, 0.0),
            },
        ),
        reset_robot0=arm_init_noise(_RIG0),
        reset_robot1=arm_init_noise(_RIG1),
    )

    # robosuite agentview (0.89, 0, 1.74); pulled back to frame both arms and tables.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(2.4, 0.0, 2.1), cam_lookat=(0.0, 0.0, 0.85))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_two_arm_transport")
