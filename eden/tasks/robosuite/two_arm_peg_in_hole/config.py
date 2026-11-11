"""robosuite ``TwoArmPegInHole``: two arms, one holding a peg and the other a plate with a hole; insert.

Port of ``robosuite/environments/manipulation/two_arm_peg_in_hole.py``. An empty arena (floor only):
two Pandas face each other 1.2 m apart (``opposed``, robot0 at ``y = -0.6`` facing +y, robot1 at
``y = +0.6`` facing -y) or stand side by side 0.5 m apart facing +x (``--arrangement parallel``).
Neither object is free: a 26 cm x 2 cm-radius green peg is welded to robot0's hand along its z
axis, and robosuite's ``plate-with-hole.xml`` (34 x 34 x 2 cm, 10 cm square hole) is welded to
robot1's hand standing edge-on, its hole centred on the hand axis 17 cm out. The task is to bring
the two hands together so the peg passes through the hole.

Reward, as in robosuite's ``reward(shaping=True)`` (the four shaping terms are summed there too):

* ``peg_hole_distance`` — ``1 - tanh(||peg - hole||)`` between the two body origins.
* ``perp_distance`` — ``1 - tanh(d)``, ``d`` the off-axis distance of the hole centre from the peg line.
* ``axial_offset`` — ``1 - tanh(|t|)``, ``t`` the hole centre's offset along the peg axis.
* ``axis_cos`` — ``|hole_normal . peg_axis|``.
* ``inserted`` — 1.0 for the success predicate ``d < 0.06, -0.12 <= t <= 0.14, cos > 0.95``.

``t``, ``d`` and ``cos`` come from :mod:`eden.tasks.robosuite.two_arm_peg_in_hole.custom_terms`.

No gripper actions: robosuite builds these robots with ``gripper_types=None`` and mounts both
objects on the bare flange (its ``right_hand`` body). Eden's rigs keep their grippers (the Franka Hand
welded to the Panda's flange), so the flange frame is full of hand: mounted there, the plate
would slice through the hand body and the peg would emerge from inside it. The objects are
therefore held *by* the gripper instead — robosuite's mounting poses are re-based on the rig's tool
point (:attr:`~eden.options.robots.ManipulatorOptions.tcp_offset`, the pad midpoint) so each object's near
end sits :data:`_GRIP_DEPTH` past it between the pads, and the fingers start closed on the object
(:meth:`~eden.options.robots.ManipulatorOptions.preset_hold`). The peg / hole geometry the reward reads is
unchanged, both objects having moved the same distance out along their own hand axes. The action
space is the two 6-D delta end-effector poses only. Deviations: the peg radius is fixed at 2 cm
(robosuite resamples it in ``[0.015, 0.03]`` per reset) and the episode is
:data:`~eden.tasks.robosuite.common.EPISODE_LENGTH_S` long rather than 50 s.
"""

from __future__ import annotations

import math
from typing import ClassVar

from eden import observations, rewards, terminations
from eden.options.camera import CamerasOptions
from eden.options.entities import AttachmentSpec, CylinderOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import ManipulatorOptions
from eden.tasks.registry import TASK_REGISTRY
from eden.tasks.robosuite.rigs import panda_rig, RobosuiteRobotMod
from eden.tasks.robosuite import custom_terms
from eden.tasks.robosuite.common import (
    agentview_camera,
    arena_entities,
    arm_actions,
    arm_init_noise,
    axis_angle_quat,
    object_pose_observations,
    Placements,
    policy_and_critic,
    ppo_runner_options,
    proprio_observations,
    rethink_mount,
    robosuite_env_options,
    RobosuiteTaskConfig,
    ROBOT_BASE_Z,
    TwoArmArrangementMod,
    yaw_quat,
)
from eden.options.objects.robosuite import RobosuitePlateWithHole
from eden.tasks.robosuite.two_arm_peg_in_hole.custom_terms import (
    EntitiesDistanceTanh,
    OneMinusTanhMetric,
    PegHoleAxialOffset,
    PegHoleAxisCos,
    PegHolePerpDistance,
    PegInHole,
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

_IDENTITY = (1.0, 0.0, 0.0, 0.0)

#: robosuite ``base_xpos_offset["empty"]`` for the Panda is ``(-0.6, 0, 0)``: opposed rotates it by +-90 degrees
#: about z, parallel shifts it by ``(0, -+0.25, 0)`` (``TwoArmEnv._load_model``).
_PLACEMENTS: Placements = {
    "opposed": (
        ((0.0, -0.6, ROBOT_BASE_Z), yaw_quat(math.pi / 2)),
        ((0.0, 0.6, ROBOT_BASE_Z), yaw_quat(-math.pi / 2)),
    ),
    "parallel": (
        ((-0.6, -0.25, ROBOT_BASE_Z), _IDENTITY),
        ((-0.6, 0.25, ROBOT_BASE_Z), _IDENTITY),
    ),
}

_RIG0 = panda_rig("robot0", *_PLACEMENTS["opposed"][0])
_RIG1 = panda_rig("robot1", *_PLACEMENTS["opposed"][1])

_PEG_RADIUS = 0.02
_PEG_LENGTH = 0.26
_PLATE_THICKNESS = 0.02
#: How far each object's near end reaches past the tool point (pad midpoint), so the pads sit on it.
_GRIP_DEPTH = 0.02
#: robosuite's mounting poses in the ``right_hand`` (bare flange) frame: the peg's centre is half its length
#: out along z; the plate is rotated 180 degrees about (0, 1, 1) so it stands edge-on in the hand's xz-plane
#: with its hole centred on the hand axis 17 cm out and its near edge at the flange.
_PEG_MOUNT = (0.0, 0.0, _PEG_LENGTH / 2)
_HOLE_MOUNT = (0.11, 0.0, 0.17)
_HOLE_MOUNT_QUAT = axis_angle_quat(math.pi, (0.0, 1.0, 1.0))


def _held_pose(rig: ManipulatorOptions, flange_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Re-base a robosuite flange-frame mounting position so the object's near end is ``_GRIP_DEPTH`` past the pads."""
    tx, ty, tz = rig.tcp_offset
    return (tx + flange_pos[0], ty + flange_pos[1], tz - _GRIP_DEPTH + flange_pos[2])


def _weld_specs(rig0: ManipulatorOptions, rig1: ManipulatorOptions) -> dict[str, AttachmentSpec]:
    """Weld the peg into robot0's gripper and the plate into robot1's (see the module docstring)."""
    return {
        "peg": (rig0.arm_name, rig0.ee_link, _held_pose(rig0, _PEG_MOUNT), _IDENTITY),
        "hole": (rig1.arm_name, rig1.ee_link, _held_pose(rig1, _HOLE_MOUNT), _HOLE_MOUNT_QUAT),
    }


def _hold_objects(rig0: ManipulatorOptions, rig1: ManipulatorOptions) -> None:
    """Start robot0's fingers closed on the peg and robot1's on the plate."""
    rig0.preset_hold(2 * _PEG_RADIUS)
    rig1.preset_hold(_PLATE_THICKNESS)


def _peg_in_hole_scene() -> SceneOptions:
    _hold_objects(_RIG0, _RIG1)
    return SceneOptions(
        robot0=_RIG0,
        robot1=_RIG1,
        **arena_entities(),
        mount0=rethink_mount(_RIG0.arm.default_root_pos, _RIG0.arm.default_root_quat),
        mount1=rethink_mount(_RIG1.arm.default_root_pos, _RIG1.arm.default_root_quat),
        # CylinderObject(size=(0.02, 0.13)): density 1000, greenwood; welded, so its default pose is unused.
        peg=CylinderOptions(
            radius=_PEG_RADIUS,
            height=_PEG_LENGTH,
            material=RigidMaterialOptions(rho=1000.0, friction=1.0),
            color=(0.10, 0.60, 0.15, 1.0),
        ),
        hole=RobosuitePlateWithHole(),
        attachments_dict=_weld_specs(_RIG0, _RIG1),
    )


def _arm_only_actions(*rigs: ManipulatorOptions) -> dict:
    """The ``<slot>_arm`` IK terms of each rig, without the ``<slot>_gripper`` ones (``gripper_types=None``)."""
    actions = {}
    for rig in rigs:
        actions.update({k: v for k, v in arm_actions(rig).items() if not k.endswith("_gripper")})
    return actions


_OBSERVATIONS = dict(
    **proprio_observations(_RIG0, prefix="robot0_"),
    **proprio_observations(_RIG1, prefix="robot1_"),
    **object_pose_observations("hole"),
    peg_to_hole=custom_terms.ObjectToObject.configure(object_a_name="peg", object_b_name="hole"),
    peg_quat=object_pose_observations("peg")["peg_quat"],
    angle=observations.MetricObservation.configure(metric_name="axis_cos"),
    t=observations.MetricObservation.configure(metric_name="axial_offset"),
    d=observations.MetricObservation.configure(metric_name="perp_distance"),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(modifiers=(RobosuiteRobotMod, TwoArmArrangementMod(placements=_PLACEMENTS)))
class RobosuiteTwoArmPegInHoleConfig(RobosuiteTaskConfig):
    robot_slots: ClassVar[tuple[str, ...]] = ("robot0", "robot1")

    def on_rigs_changed(self, rigs: list[ManipulatorOptions]) -> None:
        """Re-weld the peg / plate into the new rigs' grippers and close those fingers on them."""
        _hold_objects(*rigs)
        self.scene_options.attachments_dict = _weld_specs(*rigs)

    env_options: EnvOptions = robosuite_env_options()

    scene_options: SceneOptions = _peg_in_hole_scene()

    observation_options: ObservationManagerOptions = policy_and_critic(_OBSERVATIONS)

    action_options: ActionManagerOptions = ActionManagerOptions(**_arm_only_actions(_RIG0, _RIG1))

    metric_options: MetricManagerOptions = MetricManagerOptions(
        axial_offset=PegHoleAxialOffset.configure(contributes_to_success=False),
        perp_distance=PegHolePerpDistance.configure(contributes_to_success=False),
        axis_cos=PegHoleAxisCos.configure(contributes_to_success=False),
        inserted=PegInHole.configure(),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        peg_hole_distance=EntitiesDistanceTanh.configure(
            object_a_name="peg", object_b_name="hole", coef=1.0, weight=1.0
        ),
        perp_distance=OneMinusTanhMetric.configure(metric_name="perp_distance", coef=1.0, weight=1.0),
        axial_offset=OneMinusTanhMetric.configure(metric_name="axial_offset", coef=1.0, absolute=True, weight=1.0),
        axis_cos=rewards.MetricReward.configure(metric_name="axis_cos", weight=1.0),
        inserted=rewards.MetricReward.configure(metric_name="inserted", weight=1.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Nothing to place (both objects are welded); robosuite's "default" initialization noise on each arm.
        reset_robot0=arm_init_noise(_RIG0),
        reset_robot1=arm_init_noise(_RIG1),
    )

    # robosuite moves agentview to (1.07, 0, 2.06) for this task; pulled back to frame both arms.
    cameras_options: CamerasOptions = agentview_camera(cam_pos=(2.2, 0.0, 2.1), cam_lookat=(0.0, 0.0, 1.0))

    runner_options: RslRlOnPolicyRunnerOptions = ppo_runner_options("robosuite_two_arm_peg_in_hole")
