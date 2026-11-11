"""Lift box (``manipulation_lift_box``): reach a box on a table, grasp it, and lift it to a 3D target.

Port of mjlab ``Lift-{Robot}`` (``mjlab_manipulation/tasks/lift``), trained
from scratch: every episode starts at the arm's ready-to-grasp *home* pose
with the cube spawned in a region on the table, so the policy learns the full
reach -> grasp -> move-to-target pipeline (no warm start). The reward
structure mirrors mjlab's ``lift_env_cfg``:

* ``reaching_object``     — shaped grasp-point→object distance (clamped tanh).
* ``lifting_object``      — object grasped and lifted off the surface.
* ``object_goal_tracking``       — coarse (std 0.3) goal tracking, gated on lift.
* ``object_goal_tracking_fine``  — fine (std 0.05) goal tracking, gated on lift.

Arm control is 6-DOF differential IK (:class:`DifferentialIKController`); the
gripper is a binary open/close controller on the Robotiq drivers, which are
**force-limited** (2.4 N·m) so closing on the cube produces a surface friction
grip rather than crushing through it (see :class:`Robotiq2f85`). The
finger pads use a flat box collision geom and the grasp point / rewards key off
the pad *contact-geom* midpoint, so the reward fires on a real pinch. Like
mjlab, the cube rests on a table with the arm mounted at table height, and the
goal is a sampled point above the table. Object-on-surface is approximated by
height in place of mjlab's contact sensor.

The default manipulator is ``Droid`` (FR3 + Robotiq 2F85); pass
``--manipulator ur5e_2f85|ur10e_2f85`` (:class:`LiftBoxManipulatorMod`) to swap the arm for
another rig from :data:`eden.options.robots.COMPOSITE_ROBOT_REGISTRY`. Every such
rig carries the same Robotiq 2F85, so the grasp point, gripper action, and every
gripper-keyed reward/observation are embodiment-invariant; only the arm entity,
the IK frame + joint set, and the experiment name change (the obs dim follows the
arm's DOF count, so checkpoints do not transfer across embodiments).

No warm start or reverse curriculum is needed: ``grasp_bonus``'s 3 cm proximity gate
is what makes from-scratch grasp discovery work — see the note on that term below.
"""

from __future__ import annotations

from typing import Annotated, Literal

import eden as en
from eden import EventMode, events, rewards, terminations
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
from eden.managers.terms.events.state_buffer import ResetFromStateBuffer
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.options import ObservationGroupOptions
from eden.options.entities import BoxOptions, SphereOptions
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden import observations
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.options.managers.rewards import RewardTermOptions
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

from eden.tasks.manipulation import custom_terms
from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, Droid, ManipulatorOptions

# Geometry chosen from a Genesis-IK reachability sweep: a vertically-pinned
# Robotiq gripper can only fold down to a near-vertical low pose when the target
# is CLOSE to the base (at x~0.45 it's stuck ~0.65; at x~0.28 it reaches a clean
# near-vertical pad midpoint ~0.43). So the arm sits at 0.4 m and the cube is on
# a low table close to the base (cube at x~0.28), inside the
# gripper's vertical-reach zone, with a ready-to-grasp home pose (see composite/droid.py).
_ARM_BASE_Z = 0.40
_TABLE_TOP = 0.535
_TABLE_SIZE = (0.35, 0.50, _TABLE_TOP)
_TABLE_HOME = (0.30, 0.0, _TABLE_TOP / 2)

# 5 cm cube resting on the table; the reset event jitters its xy and yaw.
_CUBE_HALF = 0.025
_CUBE_SIZE = (2 * _CUBE_HALF,) * 3
_REST_Z = _TABLE_TOP + _CUBE_HALF
_CUBE_HOME = (0.28, 0.0, _REST_Z)
_EPISODE_S = 6.0

# Pad midpoint = grasp center (no offset): with pad-only collision the pads
# straddle the cube, so the reward fires on a real pinch.
_APPROACH_OFFSET = 0.0
_DROID = Droid(base_pos=(0.0, 0.0, _ARM_BASE_Z))
_GRASP = {
    "gripper_name": _DROID.gripper_name,
    "grasp_links": _DROID.grasp_links,
    "approach_offset": _APPROACH_OFFSET,
}
_LIFT_GATE = {
    "object_name": "cube",
    "rest_z": _REST_Z,
    "lift_margin": 0.02,
    "grasp_threshold": 0.06,
}


def _droid_lift_scene():
    return en.scenes.Empty(
        is_support_enabled=True,
        robot=_DROID,
        table=BoxOptions(
            size=_TABLE_SIZE,
            fixed=True,
            default_root_pos=_TABLE_HOME,
            material=RigidMaterialOptions(friction=2.0),
        ),
        cube=BoxOptions(
            size=_CUBE_SIZE,
            fixed=False,
            default_root_pos=_CUBE_HOME,
            material=RigidMaterialOptions(friction=2.0),
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


_Observations = dict(
    arm_dofs_pos=observations.DofsPos.configure(entity_name="robot"),
    arm_dofs_vel=observations.DofsVel.configure(entity_name="robot"),
    gripper_dofs_pos=observations.DofsPos.configure(entity_name="gripper"),
    object_position=ObsTerm.configure(
        func=custom_terms.object_position_b,
        params={"robot_name": "robot", "object_name": "cube"},
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    object_orientation=ObsTerm.configure(
        func=custom_terms.object_orientation_b,
        params={"robot_name": "robot", "object_name": "cube"},
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    target_object_position=ObsTerm.configure(
        func=observations.generated_commands,
        params={"command_name": "lift_target"},
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    grasp_point=ObsTerm.configure(
        func=custom_terms.grasp_point_b,
        params={"robot_name": "robot", **_GRASP},
    ),
    last_action=ObsTerm.configure(func=observations.last_action),
)


class LiftBoxManipulatorMod(TaskMod):
    """Swap the arm embodiment; the Robotiq 2F85 gripper and the task stay fixed.

    Every composite in :data:`eden.options.robots.COMPOSITE_ROBOT_REGISTRY` used
    here carries the same pad-only Robotiq 2F85, so the grasp point, gripper
    action, and every reward/observation that keys off the gripper are
    embodiment-independent. Swapping the arm only rewires the scene ``robot``
    (mounted at the same ``_ARM_BASE_Z``, with the gripper re-welded at the new
    arm's ``ee_link``) and the IK controller's frame + joint set. The obs dim
    follows the arm's DOF count, so checkpoints do not transfer across embodiments.

    The flag is ``--manipulator`` (not ``--robot``): ``eden inference`` also defines a
    shared ``--robot`` flag for extension-policy deploys, which a task mod's flag of that
    name replaces.
    """

    def __init__(
        self,
        manipulator: Annotated[
            Literal["droid", "ur5e_2f85", "ur10e_2f85"],
            "Arm embodiment (droid = FR3; all carry the same Robotiq 2F85).",
        ] = "droid",
    ):
        self.manipulator = manipulator

    def apply(self, config: ManipulationLiftBoxConfig) -> ManipulationLiftBoxConfig:
        if self.manipulator == "droid":
            return config  # the config is authored around Droid
        # The registry IS the manipulator lookup: an unknown name raises here
        # (fail-fast) rather than silently building the default arm.
        composite: ManipulatorOptions = COMPOSITE_ROBOT_REGISTRY.get(self.manipulator)(base_pos=(0.0, 0.0, _ARM_BASE_Z))
        config.scene_options.robot = composite
        ik = config.action_options.ee_ik
        ik.ee_link_name = composite.ee_link
        ik.dofs_name = composite.arm_dofs
        config.runner_options.experiment_name = f"lift_box_{self.manipulator}"
        return config


class LiftBoxResetBufferMod(TaskMod):
    """Restore every reset from a reset-state buffer instead of the task's own randomized resets.

    Given a buffer file (``ResetStateBuffer.save``), the cube placement (``place_cube``) and the arm-DOF jitter
    (``reset_robot_dofs``) are replaced by one :class:`~eden.managers.terms.events.state_buffer.ResetFromStateBuffer`
    that restores the entities the buffer covers, typically ``robot``, ``gripper`` and ``cube``: precollected start
    states rather than one home pose. Only the scene state is used. Rows are drawn uniformly, whatever categories
    the buffer labels, and its task columns are not read: the goal command still draws each episode's goal, so a
    row stored next to its own goal (OmniReset's Near-Goal family) does not start near the goal the episode uses.
    A buffer missing the cube restores nothing for it, so the cube stays at its default pose every episode. Capture
    one with ``capture_reset_states(env, ["robot", "gripper", "cube"])``. Without a path, the task is unchanged.
    """

    def __init__(
        self,
        reset_buffer: Annotated[
            str | None,
            "Reset-state buffer file to restore every reset from (replaces the cube placement and arm jitter). "
            "None → the task's own randomized resets.",
        ] = None,
    ):
        self.reset_buffer = reset_buffer

    def apply(self, config: ManipulationLiftBoxConfig) -> ManipulationLiftBoxConfig:
        if self.reset_buffer is None:
            return config
        config.event_options.disable_term("place_cube")
        config.event_options.disable_term("reset_robot_dofs")
        config.event_options.restore_reset_state = ResetFromStateBuffer.configure(path=self.reset_buffer)
        return config


def _lift_env_options(num_envs: int = 4096) -> EnvOptions:
    """The lift-box sim/env settings, with a tunable env count.

    A factory (like :func:`_droid_lift_scene`) so a subclass that only needs a different
    ``num_envs`` — the wrist-camera variants, whose RGB rollout storage does not fit 4096 —
    can still declare ``env_options`` as a *field default* instead of assigning it in
    ``__init__``. Pydantic calls a custom ``__init__`` from ``model_validate`` too, so an
    ``__init__`` assignment silently reverts every ``--config`` / ``load_from_file`` override
    of the field it touches (pinned by ``test_env_and_reward_overrides_survive_reconstruction``).
    """
    return EnvOptions(
        num_envs=num_envs,
        episode_length_s=_EPISODE_S,
        sim_dt=0.005,
        sim_substeps=2,
        decimation=4,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )


def _lift_rewards(
    *, add: dict[str, RewardTermOptions] | None = None, **overrides: RewardTermOptions
) -> RewardManagerOptions:
    """The lift-box reward set, with the named terms replaced by ``overrides``.

    A factory (like :func:`_droid_lift_scene`) so a subclass that retargets a few terms —
    e.g. the heterogeneous-object variant, which sources a per-env resting height — keeps
    ``reward_options`` a *field default* instead of reassigning terms in ``__init__``.
    Pydantic calls a custom ``__init__`` from ``model_validate`` too, so an ``__init__``
    assignment silently reverts every ``--config`` / ``load_from_file`` override of the
    terms it touches (see ``test_env_and_reward_overrides_survive_reconstruction``, and the
    ``test_lift_rewards_*`` tests for what this factory itself guarantees).

    ``overrides`` may only name terms this set already defines: ``RewardManagerOptions``
    allows extras, so a typo'd name would otherwise be accepted as a brand-new reward term
    rather than retargeting an existing one. A variant that genuinely wants an *additional*
    term passes it in ``add``, which is checked the other way round — a name that already
    exists there is a collision, not an addition.
    """
    terms = dict(
        reaching_object=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=5.0,
            params={**_GRASP, "object_name": "cube", "std": 0.3},
        ),
        # Sharp, unclamped fine-reaching: gives gradient across the last few cm
        # into the grasp zone (the clamped coarse term saturates ~9 cm out).
        reaching_object_fine=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=3.0,
            params={**_GRASP, "object_name": "cube", "std": 0.05, "clamp_max": 1.0},
        ),
        # Teach the close action: reward gripper closure while centered on the cube.
        # The 3 cm gate (vs the 6 cm used by the lifted&close gates) is load-bearing
        # for from-scratch training: at 6 cm the policy farms this term by hovering
        # the *closed* gripper ~5 cm from the cube without ever pinching — a stable
        # local optimum (observed as a hard plateau, lift_height ~ 0). The cube
        # half-extent is 2.5 cm and a true straddle puts the pad midpoint within
        # ~1 cm of the center, so at 3 cm closure only pays where a pinch is real.
        # The lifted&close gates keep 6 cm — they also require lift-off, which
        # cannot be farmed on the table.
        grasp_bonus=RewardTerm.configure(
            func=custom_terms.grasp_bonus,
            weight=2.0,
            params={
                **_GRASP,
                "object_name": "cube",
                "gripper_dofs": _DROID.gripper_dofs,
                "close_target": 0.8,
                "grasp_threshold": 0.03,
            },
        ),
        # Dense height reward: any upward cube motion is rewarded, so an
        # incidental grasp gets reinforced before the sparse gates fire.
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
                "robot_name": "robot",
                **_GRASP,
                **_LIFT_GATE,
                "command_name": "lift_target",
                "std": 0.3,
            },
        ),
        # 2 cm deadband: the fine term saturates within 2 cm so there's no sub-cm
        # gradient to chase, letting settle_at_goal still the arm at the goal.
        object_goal_tracking_fine=RewardTerm.configure(
            func=custom_terms.object_goal_distance,
            weight=10.0,
            params={
                "robot_name": "robot",
                **_GRASP,
                **_LIFT_GATE,
                "command_name": "lift_target",
                "std": 0.05,
                "deadband": 0.02,
            },
        ),
        # Smoothness regularizers, ramped in by the curriculum below so the
        # grasp/lift is learned first (the IK's tanh(action) makes action
        # magnitude free, so without a penalty the policy chatters).
        action_rate=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-0.01,
        ),
        # Hinge penalty on arm joint velocity above 0.5 rad/s. Starts nonzero so
        # the reward manager doesn't prune a zero-weight term the curriculum ramps.
        arm_joint_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits,
            weight=-0.01,
            params={"entity_name": "robot", "max_vel": 0.5},
        ),
        # Hold still once the cube is at the goal: the goal-tracking reward is flat
        # there, so nothing else stops the arm drifting. Self-gating (~0 off-goal,
        # so the approach is unhindered). mjlab needs no equivalent — its episode
        # is 1 s and ends at the lift; Eden's 6 s episode has a long hold phase.
        settle_at_goal=RewardTerm.configure(
            func=custom_terms.settle_at_goal,
            weight=-6.0,
            params={
                "robot_name": "robot",
                "object_name": "cube",
                "command_name": "lift_target",
                "std": 0.05,
            },
        ),
        arm_joint_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-1.0,
            params={"entity_name": "robot"},
        ),
    )
    unknown = sorted(set(overrides) - set(terms))
    if unknown:
        raise ValueError(
            f"_lift_rewards() got override(s) for unknown reward term(s) {unknown}; "
            f"the lift-box reward set "
            f"defines {sorted(terms)}. To retarget one, use its exact name; to add a brand-new term, pass it "
            f"in add={{...}} so the intent is explicit."
        )
    collisions = sorted(set(add or ()) & set(terms))
    if collisions:
        raise ValueError(
            f"_lift_rewards(add=...) got term(s) {collisions} that the lift-box reward set "
            f"already defines; "
            f"pass them as keyword overrides to retarget them instead of adding them."
        )
    terms.update(overrides)
    terms.update(add or {})
    return RewardManagerOptions(**terms)


@TASK_REGISTRY.register(modifiers=(LiftBoxManipulatorMod, LiftBoxResetBufferMod))
class ManipulationLiftBoxConfig(EdenRLConfig):
    env_options: EnvOptions = _lift_env_options()

    scene_options: SceneOptions = _droid_lift_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # 6-DOF differential IK: the policy controls EE position AND orientation.
        # A hard vertical pin put the cube outside the reachable workspace; here the
        # arm can tilt as needed to reach. Pad-only collision (see Robotiq2f85) is what
        # forces a real fingertip grasp instead of wedging the cube against the palm.
        ee_ik=DifferentialIKController.configure(
            entity_name="robot",
            ee_link_name=_DROID.ee_link,
            dofs_name=_DROID.arm_dofs,
            # Small position delta + smaller orientation delta to limit wrist
            # jitter from the full 6-DOF action.
            position_scale=0.05,
            orientation_scale=0.10,
            # Well-damped IK + per-step joint clamp (mjlab parity); a near-singular
            # damping (1e-4, no clamp) spikes delta_q. max_dq=0.2 rad/step ~ 10 rad/s at 50 Hz.
            damping=0.1,
            max_dq=0.2,
        ),
        gripper=BinaryJointController.configure(
            entity_name="gripper",
            dofs_name=_DROID.gripper_dofs,
            open_action=dict(_DROID.gripper_open),
            close_action=dict(_DROID.gripper_close),
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        lift_target=UniformT3Command.configure(
            x_range=(0.18, 0.40),
            y_range=(-0.15, 0.15),
            z_range=(_REST_Z + 0.10, _REST_Z + 0.30),
            # > episode length: the goal is fixed per episode (resampled on reset).
            resampling_time_range=(2 * _EPISODE_S, 2 * _EPISODE_S),
            # Move the green goal_marker sphere to the goal during eval/inference
            # (goal is in the robot base frame). No effect during training.
            debug_vis=True,
            marker_frame_entity="robot",
            marker_entity_name="goal_marker",
        )
    )

    reward_options: RewardManagerOptions = _lift_rewards()

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        cube_fallen=DoneTerm.configure(
            func=custom_terms.object_fallen,
            time_out=False,
            params={"object_name": "cube", "threshold": _TABLE_TOP - 0.04},
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Direct root-pose reset (mjlab's reset_root_state_uniform analog).
        # NOTE: place_range_xy / place_on_to crashes on *partial* (staggered)
        # resets, so jitter the cube's root pose around its home instead.
        # The spawn region is wide enough that reaching is a non-trivial learned
        # phase (episodes always start at the home pose), while staying inside the
        # arm's graspable workspace (the 6-DOF IK lets the wrist tilt to reach
        # off-center poses).
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
            params={"entity_name": "robot", "dofs_pos_range": (-0.05, 0.05)},
        ),
        # Stiff gripper<->cube contact: raise the solimp impedance on the cube
        # + finger-pad geoms (dmin .95, dmax .999, width .5 mm; timeconst stays
        # at the substep-clamped default). The mujoco-default impedance
        # (dmax .95 = ~5% permanent give) let the pads sink ~15 mm into the cube
        # under the drivers' squeeze, with the gripper oscillating to resolve
        # the contact; at dmax .999 sustained penetration is ~0.1 mm and hold
        # oscillation drops 3x, at zero solver cost (same kernels, same
        # iterations). Don't exceed dmax~0.999: near-rigid contact turns bouncy
        # and grasping degrades. Genesis mixes a pair's params as the mean of its
        # two geoms', so the cube side also mildly stiffens cube<->table
        # (dmax ~.975 post-mix — measured harmless: cube_fallen rate unchanged).
        stiffen_contact=EventTerm.configure(
            func=events.set_contact_sol_params,
            mode=EventMode.STARTUP,
            params={
                "sol_params": (0.005, 1.0, 0.95, 0.999, 0.0005, 0.5, 2.0),
                "entity_links": {"cube": None, "gripper": _DROID.grasp_links},
            },
        ),
    )

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        # Goal-range curriculum: start the target as a single point straight above
        # the cube (lowest height), expand to the full x/y/z ranges as the cube
        # learns to reach it.
        goal_range=custom_terms.GoalRangeCurriculum.configure(
            command_name="lift_target",
            object_name="cube",
            robot_name="robot",
            reach_thresh=0.08,
            advance_thresh=0.7,
            retreat_thresh=0.4,
            step_up=0.05,
            step_down=0.05,
            level_init=0.0,
            # base/command frame: cube's xy, lowest target height (_REST_Z + 0.10)
            easy_center=(_CUBE_HOME[0], 0.0, _REST_Z + 0.10),
        ),
        # Ramp the smoothness penalties in only after the grasp/lift is solid, so
        # they don't fight task acquisition (grasp discovery from the home pose
        # is slow). Stage steps are
        # common_step_counter (== iteration * num_steps_per_env, here 24).
        action_rate_ramp=StageRewardWeightCurriculum.configure(
            reward_term_name="action_rate",
            weight_stages=[
                (0, -0.01),
                (1200 * 24, -0.04),
                (1800 * 24, -0.1),
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
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
        ),
        max_iterations=3000,
        save_interval=50,
        experiment_name="lift_box",
        run_name="v1",
    )
