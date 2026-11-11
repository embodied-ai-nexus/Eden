"""Dexterous lift (``manipulation_dex_lift``): grasp a box with a dexterous hand and lift it to a 3D target.

Dexterous-hand counterpart of :class:`~eden.tasks.manipulation.lift_box.config.ManipulationLiftBoxConfig`
(``manipulation_lift_box``). It keeps that task's mature mjlab ``lift_env_cfg`` structure — shaped
reach, dense lift, grasped-and-lifted gate, coarse+fine 3D goal tracking, a
goal-range curriculum, staged smoothness curricula and contact stiffening — but
swaps the two-finger Robotiq 2F85 for the 12-DOF ``XHand1_L`` dexterous hand.

Two things change versus the lift-box task:

* **End-effector = fingertip centroid.** The shared manipulation MDP terms take the
  grasp center as the midpoint of the ``grasp_links`` *collision geoms*
  (:func:`~eden.tasks.manipulation.custom_terms.common._grasp_point_w`). Passing the
  five XHand fingertip links (all of which carry collision geoms) makes that the
  centroid of the fingertips, so every lift-box reward/observation reuses unchanged.
* **Grasp shaping is dex-native.** The lift-box task's ``grasp_bonus`` rewards closing a *single*
  gripper driver toward a scalar target — meaningless for a 12-DOF hand. It is
  replaced by :func:`dex_fingertip_to_object_reward` (mean per-fingertip exponential
  approach), which teaches the fingers to wrap the cube.

Arm control is 6-DOF differential IK (:class:`DifferentialIKController`); the hand is
driven in joint space by an :class:`ImplicitPDController` over its 12 DOFs.

The rig is the one a lab builds, :class:`~eden.options.robots.FrankaXHand` (the hand on the
LeFranX right-angle coupling, palm down at the ready pose), and the policy is trained to be deployed on it:

* **The policy observes only what the rig measures.** Franka and XHand joint encoders, its own last
  action, the goal, forward kinematics of those, and the cube pose a camera estimates from AprilTags
  on the cube (as the Wuji in-hand deployments get theirs). Each channel carries its sensor's error
  in training, the cube pose an AprilTag estimate's (a few mm, a few degrees) with a 3-frame history
  to filter it. The cube's orientation is a 6D rotation, not a quaternion, so the sign a pose
  estimator picks does not matter. What only the simulator knows (the XHand's joint velocities,
  which its SDK only finite-differences; the cube's velocity; the fingertip-to-cube offsets, which
  would leak a noise-free cube position) is the critic's alone.
* **The cube pose arrives through the ``cube`` entity.** Every cube observation reads that entity,
  so a deployment writes the camera's estimate into it each tick and the policy reads the estimate.
  The Franka and XHand backends ship with Eden; the tag-tracking observer that does that write is
  the Wuji deployments' and is not in Eden yet. Until it lands, the policy runs on the real arm
  and hand only with that observer added. Sim-to-sim is not there yet either:
  :class:`~eden.extensions.deployment.mujoco_composite.MujocoCompositeDeployment` cannot load this rig,
  because MuJoCo rejects the XHand URDF's rank-1 link inertias; the inertia repair, the cube's
  simulated camera and the stiff hand contact that sim-to-sim needs are not in Eden yet.
* **Training noise is per step; the camera's timing is checked in sim-to-sim.** The cube noise is
  drawn fresh every control step. A real camera delivers frames at its own rate, late, and holds
  each one. That is deliberately not trained: an observation modifier (``ObservationDelay``) also
  runs at deployment, where it would delay a reading that already arrives late. The policies are
  checked instead in MuJoCo against a simulated AprilTag camera (30 Hz, 40 ms latency).
* **The hand is as strong as the XHand.** Its joints keep the URDF effort limits (0.4 / 1.1 N m), not
  :class:`~eden.options.robots.XHand1_L`'s 5 N m teleoperation ceiling, and every hand link meets the
  cube with stiffened contact, so a grasp cannot rest on fingers sunk centimetres into the cube.
* **Friction is randomized, not fixed high.** The hand's friction is drawn per env from U(0.8, 2.0)
  against a 0.5 cube and table (a contact takes the larger of its two coefficients). The range spans
  rubber finger pads on a printed cube; its high end is also what lets PPO find the lift at all: at
  U(0.7, 1.3) no seed lifted the cube in 2800 iterations.
* **Goals are reachable.** Lift targets put the cube 10 to 30 cm above where it rests, in the arm's
  base frame.
* **Actions are smooth under camera noise.** The action-rate penalty is 5x the lift-box task's, so
  the policy does not answer the noisy cube pose with chattering actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import eden as en
from eden import EventMode, events, observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.managers import OBSERVATION_TERM_REGISTRY, REWARD_TERM_REGISTRY
from eden.managers.modifiers.observations.noise import GaussianNoise, UniformNoise
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
from eden.managers.terms.events import RandomizeFrictionRatio
from eden.options import ObservationGroupOptions
from eden.options.entities import BoxOptions, SphereOptions
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.options.robots import FrankaXHand, XHand1_L
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.manipulation import custom_terms
from eden.tasks.manipulation.custom_terms.common import _to_base_frame
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import DoneTerm, EventTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


# ---------------------------------------------------------------------------
# Geometry (table + cube mirror the lift-box task; arm home + DOFs/links for the Franka + XHand1_L)
# ---------------------------------------------------------------------------

# Table-mounted arm; cube on a low table close to the base, goal above the table.
# (See lift_box/config.py for the Genesis-IK reachability rationale behind these numbers.)
_ARM_BASE_Z = 0.40
_TABLE_TOP = 0.535
_TABLE_SIZE = (0.35, 0.50, _TABLE_TOP)
_TABLE_HOME = (0.30, 0.0, _TABLE_TOP / 2)

# 6 cm cube (sized for the dexterous hand) resting on the table.
_CUBE_HALF = 0.03
_CUBE_SIZE = (2 * _CUBE_HALF,) * 3
_REST_Z = _TABLE_TOP + _CUBE_HALF
_CUBE_HOME = (0.28, 0.0, _REST_Z)
_EPISODE_S = 6.0
# Cube and table friction; the hand's is randomized above it (see the ``hand_friction`` event).
_CUBE_FRICTION = 0.5

# Franka + XHand1 (left) on the LeFranX coupling: the arm + home pose, the 7 arm DOFs, the
# 12 hand DOFs, the 5 fingertip links, and the hand->arm weld all come from this single
# rig (see eden/options/robots/composite). Mounted on the table, so the arm base is
# raised along z. Placed into the scene as one entity and queried for its derived DOFs.
# The hand keeps its asset's effort limits (an empty ``default_dofs_force_limits`` is the URDF's
# 0.4 / 1.1 N m): XHand1_L's 5 N m is a teleoperation ceiling 4.5-12x the real actuators, with which a
# policy learns to squeeze the cube harder than the hand it is deployed on can.
_DEX = FrankaXHand(
    base_pos=(0.0, 0.0, _ARM_BASE_Z),
    hand=XHand1_L(
        is_fixed_base=False,
        material=RigidMaterialOptions(gravity_compensation=1.0),
        default_dofs_force_limits={},
    ),
)

# Grasp center = fingertip-geom midpoint (the EE the shared manipulation terms use); the
# shared terms take the grasping entity's name under the ``gripper_name`` param.
_GRASP = {
    "gripper_name": _DEX.hand_name,
    "grasp_links": _DEX.fingertip_links,
    "approach_offset": 0.0,
}
_LIFT_GATE = {
    "object_name": "cube",
    "rest_z": _REST_Z,
    "lift_margin": 0.02,
    "grasp_threshold": 0.06,
}


# ---------------------------------------------------------------------------
# Dex-specific terms (the hand-only reward/observation that replace the lift-box task's
# two-finger grasp shaping).
# ---------------------------------------------------------------------------


# Function names are ``dex_``-prefixed so their auto-derived registry keys stay
# unambiguous and don't collide with any similarly-named term another task might
# register (the term registry raises on duplicate names, and configure(func=...)
# resolves a term by to_snake_case(func.__name__)). They stay function terms, although a class term
# would resolve the fingertip links once in ``build()``: a run's saved ``config.json`` names each term
# by its kind, and turning one into a class stops the checkpoints trained with it from loading.


def _fingertips_pos_w(env: "EnvBase", hand_name: str, fingertip_links: list[str]) -> torch.Tensor:
    """World positions of the fingertips, ``(num_envs, n_tips, 3)``."""
    hand: Entity = env.entities[hand_name]
    _, tip_idx = hand.find_named_links_idx_local(fingertip_links, preserve_order=True)
    return hand.get_links_pos(ls_idx_local=tip_idx)


@REWARD_TERM_REGISTRY.register()
def dex_fingertip_to_object_reward(
    env: "EnvBase",
    *,
    hand_name: str,
    object_name: str,
    fingertip_links: list[str],
    sigma: float = 0.06,
) -> torch.Tensor:
    """Mean exponential reward for all fingertips approaching the object.

    Dex-native replacement for the lift-box task's two-finger ``grasp_bonus``: teaches the
    fingers to wrap the cube (there is no single "close" DOF to reward on a hand).
    """
    tips_pos = _fingertips_pos_w(env, hand_name, fingertip_links)  # (N, 5, 3)
    obj_pos = env.entities[object_name].get_pos().unsqueeze(1)  # (N, 1, 3)
    dist_sq = torch.sum(torch.square(tips_pos - obj_pos), dim=-1)  # (N, 5)
    per_finger = torch.exp(-dist_sq / (sigma**2))  # (N, 5)
    return per_finger.mean(dim=-1)  # (N,)


@OBSERVATION_TERM_REGISTRY.register()
def dex_fingertips_to_object_obs(
    env: "EnvBase",
    *,
    hand_name: str,
    object_name: str,
    fingertip_links: list[str],
) -> torch.Tensor:
    """Per-fingertip displacement vectors to the object. Shape: (num_envs, 3*n_tips)."""
    tips_pos = _fingertips_pos_w(env, hand_name, fingertip_links)  # (N, 5, 3)
    obj_pos = env.entities[object_name].get_pos().unsqueeze(1)  # (N, 1, 3)
    return (obj_pos - tips_pos).flatten(start_dim=1)  # (N, 15)


@OBSERVATION_TERM_REGISTRY.register()
def dex_fingertips_pos_b(
    env: "EnvBase",
    *,
    robot_name: str,
    hand_name: str,
    fingertip_links: list[str],
) -> torch.Tensor:
    """Fingertip positions in the arm's base frame, from forward kinematics. Shape: (num_envs, 3*n_tips)."""
    tips_w = _fingertips_pos_w(env, hand_name, fingertip_links)  # (N, 5, 3)
    return _to_base_frame(env, robot_name, tips_w).flatten(start_dim=1)  # (N, 15)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _dex_lift_scene():
    return en.scenes.Empty(
        is_support_enabled=True,
        # The Franka + XHand composite expands into the ``robot`` (arm) and ``hand``
        # entities welded at the Franka "attachment" frame (added by its primary name).
        robot=_DEX,
        # Low friction on both, so the hand's randomized friction decides every hand-cube contact.
        table=BoxOptions(
            size=_TABLE_SIZE,
            fixed=True,
            default_root_pos=_TABLE_HOME,
            material=RigidMaterialOptions(friction=_CUBE_FRICTION),
        ),
        cube=BoxOptions(
            size=_CUBE_SIZE,
            fixed=False,
            default_root_pos=_CUBE_HOME,
            material=RigidMaterialOptions(friction=_CUBE_FRICTION),
        ),
        # Visual-only goal marker (renders in videos): a kinematic sphere the command
        # teleports to the per-env goal each eval step. Kinematic = not simulated and
        # generates no contacts, so it holds its pose and never perturbs the scene.
        goal_marker=SphereOptions(
            radius=0.025,
            material=KinematicMaterialOptions(),
            color=(0.1, 0.9, 0.1, 0.85),
            default_root_pos=(_CUBE_HOME[0], 0.0, _REST_Z + 0.20),
        ),
    )


_CUBE_B = {"robot_name": _DEX.arm_name, "object_name": "cube"}

# What the deployed rig measures, each with its sensor's error (noise applies to the policy group only).
_Measured = dict(
    # Franka joint encoders and joint velocities (libfranka reports both).
    arm_dofs_pos=observations.DofsPos.configure(
        entity_name=_DEX.arm_name, noise=UniformNoise.configure(n_min=-0.005, n_max=0.005)
    ),
    arm_dofs_vel=observations.DofsVel.configure(
        entity_name=_DEX.arm_name, noise=UniformNoise.configure(n_min=-0.05, n_max=0.05)
    ),
    # XHand joint encoders (backlash and calibration error of a few hundredths of a radian).
    hand_dofs_pos=observations.DofsPos.configure(
        entity_name=_DEX.hand_name, noise=UniformNoise.configure(n_min=-0.02, n_max=0.02)
    ),
    # The cube pose from the AprilTag camera: +/-8 mm per axis and ~3 deg (the error the Wuji reorient
    # policy is trained with for its tag-tracked cube), with 3 frames of history to filter it.
    object_position=ObsTerm.configure(
        func=custom_terms.object_position_b,
        params=_CUBE_B,
        noise=UniformNoise.configure(n_min=-0.008, n_max=0.008),
        history_length=3,
    ),
    object_orientation=ObsTerm.configure(
        func=custom_terms.object_rot6d_b,
        params=_CUBE_B,
        noise=GaussianNoise.configure(std=0.05),
        history_length=3,
    ),
    # The goal is the operator's command: known exactly.
    target_object_position=ObsTerm.configure(
        func=observations.generated_commands,
        params={"command_name": "lift_target"},
    ),
    # Forward kinematics of the joints. Training computes it from the true joints, not the noisy encoder
    # readings above, so it carries no encoder error; on hardware it is computed from the real encoders.
    grasp_point=ObsTerm.configure(
        func=custom_terms.grasp_point_b,
        params={"robot_name": _DEX.arm_name, **_GRASP},
    ),
    fingertips_pos=ObsTerm.configure(
        func=dex_fingertips_pos_b,
        params={"robot_name": _DEX.arm_name, "hand_name": _DEX.hand_name, "fingertip_links": _DEX.fingertip_links},
    ),
    last_action=ObsTerm.configure(func=observations.last_action),
)

# What only the simulator knows: the critic's alone.
_Privileged = dict(
    # The XHand SDK reports no joint velocity; its backend finite-differences positions at the loop rate.
    hand_dofs_vel=observations.DofsVel.configure(entity_name=_DEX.hand_name),
    # Fingertip-to-cube offsets from the true cube pose (in the policy group they would bypass the cube noise).
    fingertips_to_object=ObsTerm.configure(
        func=dex_fingertips_to_object_obs,
        params={"hand_name": _DEX.hand_name, "object_name": "cube", "fingertip_links": _DEX.fingertip_links},
    ),
    object_lin_vel=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "cube"}),
    object_ang_vel=ObsTerm.configure(func=observations.base_ang_vel, params={"entity_name": "cube"}),
)


@TASK_REGISTRY.register()
class ManipulationDexLiftConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=_EPISODE_S,
        sim_dt=0.005,
        sim_substeps=2,
        decimation=4,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = _dex_lift_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Measured, enable_corruption=True),
        critic=ObservationGroupOptions(**_Measured, **_Privileged, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # 6-DOF differential IK on the arm (position + orientation): the wrist can
        # tilt to reach off-center cube poses. Well-damped + per-step joint clamp.
        ee_ik=DifferentialIKController.configure(
            entity_name=_DEX.arm_name,
            ee_link_name=_DEX.ee_link,
            dofs_name=_DEX.arm_dofs,
            position_scale=0.05,
            orientation_scale=0.10,
            damping=0.1,
            max_dq=0.2,
        ),
        # Joint-space PD over the 12 hand DOFs, targeting
        # default + scale*action. Replaces the lift-box task's binary gripper controller.
        hand=ImplicitPDController.configure(
            entity_name=_DEX.hand_name,
            dofs_name=_DEX.hand_dofs,
            scale=0.2,
            reference_source=ReferenceSource.DEFAULT,
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        lift_target=UniformT3Command.configure(
            x_range=(0.18, 0.40),
            y_range=(-0.15, 0.15),
            # The command is in the arm's base frame: 10-30 cm above the table.
            z_range=(_REST_Z - _ARM_BASE_Z + 0.10, _REST_Z - _ARM_BASE_Z + 0.30),
            # > episode length: the goal is fixed per episode (resampled on reset).
            resampling_time_range=(2 * _EPISODE_S, 2 * _EPISODE_S),
            # Move the green goal_marker sphere to the goal during eval/inference
            # (goal is in the robot base frame). No effect during training.
            debug_vis=True,
            marker_frame_entity=_DEX.arm_name,
            marker_entity_name="goal_marker",
        )
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching_object=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=5.0,
            params={**_GRASP, "object_name": "cube", "std": 0.3},
        ),
        # Sharp, unclamped fine-reach: gradient across the last few cm into the grasp.
        reaching_object_fine=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=3.0,
            params={**_GRASP, "object_name": "cube", "std": 0.05, "clamp_max": 1.0},
        ),
        # Dex grasp shaping: reward the fingertips wrapping the cube (replaces the
        # two-finger grasp_bonus, which has no analog for a 12-DOF hand).
        fingertip_to_object=RewardTerm.configure(
            func=dex_fingertip_to_object_reward,
            weight=2.0,
            params={
                "hand_name": _DEX.hand_name,
                "object_name": "cube",
                "fingertip_links": _DEX.fingertip_links,
                "sigma": 0.06,
            },
        ),
        # Dense height reward: any upward cube motion is rewarded.
        lift_height=RewardTerm.configure(
            func=custom_terms.lift_height,
            weight=15.0,
            params={"object_name": "cube", "rest_z": _REST_Z, "cap": 0.3},
        ),
        lifting_object=RewardTerm.configure(
            func=custom_terms.object_is_lifted,
            weight=5.0,
            params={**_GRASP, **_LIFT_GATE},
        ),
        object_goal_tracking=RewardTerm.configure(
            func=custom_terms.object_goal_distance,
            weight=25.0,
            params={
                "robot_name": _DEX.arm_name,
                **_GRASP,
                **_LIFT_GATE,
                "command_name": "lift_target",
                "std": 0.3,
            },
        ),
        # 2 cm deadband so settle_at_goal can still the arm at the goal.
        object_goal_tracking_fine=RewardTerm.configure(
            func=custom_terms.object_goal_distance,
            weight=10.0,
            params={
                "robot_name": _DEX.arm_name,
                **_GRASP,
                **_LIFT_GATE,
                "command_name": "lift_target",
                "std": 0.05,
                "deadband": 0.02,
            },
        ),
        # Smoothness regularizers, ramped in by the curriculum (grasp/lift learned first).
        action_rate=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-0.05,
        ),
        arm_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits,
            weight=-0.01,
            params={"entity_name": _DEX.arm_name, "max_vel": 0.5},
        ),
        # Hold still once the cube is at the goal (goal-tracking is flat there).
        settle_at_goal=RewardTerm.configure(
            func=custom_terms.settle_at_goal,
            weight=-6.0,
            params={
                "robot_name": _DEX.arm_name,
                "object_name": "cube",
                "command_name": "lift_target",
                "std": 0.05,
            },
        ),
        arm_joint_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-1.0,
            params={"entity_name": _DEX.arm_name},
        ),
        hand_joint_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-1.0,
            params={"entity_name": _DEX.hand_name},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        cube_fallen=DoneTerm.configure(
            func=custom_terms.object_fallen,
            time_out=False,
            params={"object_name": "cube", "threshold": _TABLE_TOP - 0.04},
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Jitter the cube's root pose around its home (place_range_xy crashes on
        # partial/staggered resets — see the lift_box/config.py note).
        place_cube=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "cube",
                "pos_x_range": (-0.06, 0.06),
                "pos_y_range": (-0.12, 0.12),
                "pos_z_range": (0.0, 0.0),
                "yaw_range": (-3.14159, 3.14159),
            },
        ),
        reset_robot_dofs=EventTerm.configure(
            func=events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={"entity_name": _DEX.arm_name, "dofs_pos_range": (-0.05, 0.05)},
        ),
        reset_hand_dofs=EventTerm.configure(
            func=events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={"entity_name": _DEX.hand_name, "dofs_pos_range": (-0.02, 0.02)},
        ),
        # Stiffen every hand link's contact with the cube (see lift_box/config.py for the impedance rationale).
        # Not the fingertips alone: a palm-down grasp presses with the phalanges, which on Genesis'
        # default impedance sink 1-2 cm into the cube.
        stiffen_contact=EventTerm.configure(
            func=events.set_contact_sol_params,
            mode=EventMode.STARTUP,
            params={
                "sol_params": (0.005, 1.0, 0.95, 0.999, 0.0005, 0.5, 2.0),
                "entity_links": {"cube": None, _DEX.hand_name: None},
            },
        ),
        # Per-env hand friction; it decides every hand-cube contact (the cube's is lower, and a pair takes the max).
        # The high end matters for learning: at U(0.7, 1.3) PPO never discovers the lift.
        hand_friction=RandomizeFrictionRatio.configure(
            mode=EventMode.STARTUP,
            entity_name=_DEX.hand_name,
            links_name=[".*"],
            friction_range=(0.8, 2.0),
        ),
    )

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        # Goal-range curriculum: start the target as a single point above the cube,
        # expand to the full ranges as the cube learns to reach it (self-gates on grasp).
        goal_range=custom_terms.GoalRangeCurriculum.configure(
            command_name="lift_target",
            object_name="cube",
            robot_name=_DEX.arm_name,
            reach_thresh=0.08,
            advance_thresh=0.7,
            retreat_thresh=0.4,
            step_up=0.05,
            step_down=0.05,
            level_init=0.0,
            easy_center=(_CUBE_HOME[0], 0.0, _REST_Z - _ARM_BASE_Z + 0.10),  # base frame
        ),
        # Ramp the smoothness penalties in after the grasp/lift is solid. Stage steps
        # are common_step_counter (== iteration * num_steps_per_env, here 24).
        # The action-rate weights are 5x LiftBox's: at -0.01 -> -0.1 the policy answered the
        # cube pose's camera noise with its actions, and in MuJoCo under a simulated AprilTag
        # camera two of three seeds chattered and dropped the cube. At these, two of three hold
        # it; the third drops it 4-5 s in with the exact cube pose too, so not for the camera noise.
        action_rate_ramp=StageRewardWeightCurriculum.configure(
            reward_term_name="action_rate",
            weight_stages=[
                (0, -0.05),
                (1200 * 24, -0.2),
                (1800 * 24, -0.5),
            ],
        ),
        arm_joint_vel_ramp=StageRewardWeightCurriculum.configure(
            reward_term_name="arm_joint_vel",
            weight_stages=[
                (0, -0.01),
                (1200 * 24, -0.2),
                (1800 * 24, -0.5),
            ],
        ),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        # Larger nets than the lift-box task's: 18-D action (6 IK + 12 hand) and a richer obs.
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 512, 256, 128],
        ),
        max_iterations=3000,
        save_interval=50,
        experiment_name="dex_lift",
        run_name="v1",
    )
