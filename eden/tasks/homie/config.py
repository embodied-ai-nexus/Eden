"""HOMIE training task: the OpenHomie G1 velocity + base-height locomotion controller.

A port of OpenHomie's G1 lower-body controller
(``HomieRL/legged_gym/legged_gym/envs/g1/g1_29dof_config.py`` in
https://github.com/InternRobotics/OpenHomie) onto Eden. The policy tracks a 4-D command
``[lin_vel_x, lin_vel_y, ang_vel_z, base_height]`` with the 12 leg joints; the 3 waist +
14 arm joints are the non-policy "upper body", driven to random poses by the upper-body
pose curriculum so the legs stay robust to arbitrary torso + arm motion. It trains with
OpenHomie's HIMLoco learner (:mod:`eden.learning.wrappers.rsl_rl.him`), which is the config's
``runner_options.algorithm``, handed this task's mirror (:mod:`eden.tasks.homie.symmetry`); the stock
commands drive it::

    eden train --task homie
    eden inference --task homie --checkpoint logs/homie/<run>/model_<it>.pt

The frozen OpenHomie release checkpoints (15 actions, 86-d frame) are a separate,
inference-only port: :class:`eden.extensions.policies.homie.HomiePolicy`.

Differences from OpenHomie:

- **Robot: 29-DOF G1.** OpenHomie welds the waist to one DOF (27 total); this keeps the
  physical 29-DOF G1 and never lets the policy drive the waist, so a frame is 80-d, not 76.
- **No restitution randomization** — Genesis' rigid solver has no per-geom restitution.
- **Actions are delayed** by ``U[0, decimation)`` physics steps, drawn per episode
  (:class:`ActionDelay`). OpenHomie computes ``delayed_actions`` with ``domain_rand.delay`` but builds
  its torques from the undelayed ``self.actions``, so its policies train with no delay.
- **The upper-body curriculum ratio ramps on a fixed step schedule**, not on velocity-tracking
  performance (:class:`~eden.tasks.homie.custom_terms.actions.UpperBodyPoseCurriculum`).
- **Encoder bias** (``encoder_bias``, ±0.02 rad) is an Eden-only realism term.
- **Torso and hand payloads apply.** OpenHomie writes ``payload_mass`` and then overwrites every
  body's mass with ``link_mass_range`` x default, so its effective torso randomization is only
  ``x U(0.8, 1.2)``. Here the +U(-5, 10) kg torso and +U(-0.1, 0.3) kg hand payloads take effect.
- **``joint_power`` uses the upper body's actual holding torques.** OpenHomie's ``torques[:, 12:]``
  holds the upper body's *position targets* (rad), so its upper-body ``|v| |tau|`` multiplies speed
  by an angle.
- **Resets randomize the base pose** (yaw ±π, xy ±1 m, z +[0, 0.1] m); OpenHomie respawns at
  its initial pose.
- **Friction is drawn per link, once at startup**; OpenHomie draws one coefficient per env and
  redraws it at every reset.
- **Friction combines as ``max``** in Genesis and as the average in Isaac Gym, so the ground is
  authored at 0.01 and the robot draws OpenHomie's *effective* range: ``(f + 1.0) / 2`` for
  ``f ~ U(0.1, 3.0)`` against its friction-1.0 plane is ``U(0.55, 2.0)``.

OpenHomie is released under CC BY-NC-SA 4.0 (non-commercial); this task inherits it.
"""

from __future__ import annotations

import math

import eden as en
from eden import actions, events, observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.extensions.policies.homie.policy import homie_g1_robot
from eden.learning.wrappers.rsl_rl.him import HIMPPOAlgorithmOptions
from eden.managers.modifiers.actions import ActionDelay, Compose
from eden.managers.modifiers.observations import UniformNoise
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
)
from eden.options.entities import PlaneOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from eden.tasks import TASK_REGISTRY
from eden.tasks.homie.custom_terms import rewards as homie_rewards
from eden.tasks.homie.custom_terms.actions import HomieMotorModel, UpperBodyPoseCurriculum
from eden.tasks.homie.custom_terms.commands import HomieCommand
from eden.terms import DoneTerm, EventTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

# G1Dofs29 declaration order: legs 0-11, waist 12-14, arms 15-28.
LEG_DOFS: list[str] = G1_DOFS29_NAME[0:12]
UPPER_BODY_DOFS: list[str] = G1_DOFS29_NAME[12:29]

FEET_LINK_NAMES: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]

HOMIE_ACTION_SCALE: float = 0.25
HOMIE_DEFAULT_HEIGHT: float = 0.74
TRACKING_SIGMA: float = 0.25  # OpenHomie rewards.tracking_sigma
DECIMATION: int = 4

# Limb links that get the multiplicative link-mass DR. The torso and the wrist-yaw hand mounts
# carry their own additive payload DR; both terms write ``set_mass_shift``, which overwrites per
# link, so a link in both sets would keep only the later draw.
_LINK_MASS_SCALE_LINKS: list[str] = [
    "*_hip_*_link",
    "*_knee_link",
    "*_ankle_*_link",
    "waist_*_link",
    "*_shoulder_*_link",
    "*_elbow_link",
    "*_wrist_roll_link",
    "*_wrist_pitch_link",
]

# One proprioceptive frame: ``[command(4), base_ang_vel(3), projected_gravity(3), dofs_pos(29),
# dofs_vel(29), last_action(12)] = 80``. The order is load-bearing: the HIM symmetry maps
# (:mod:`eden.tasks.homie.symmetry`) index it by position, and the estimator drops the three
# velocity commands at its front to form its SwAV target. Scales are OpenHomie's obs_scales; the
# noise is in raw (pre-scale) units, so scale × noise reproduces legged_gym's
# ``noise_scales × obs_scales``. Commands and previous actions carry no noise.
_OBSERVATIONS = dict(
    command=ObsTerm.configure(
        func=observations.generated_commands,
        params={"command_name": "homie_command"},
        scale=[2.0, 2.0, 0.5, 1.0],  # OpenHomie: lin_vel 2, ang_vel 0.5, height raw
    ),
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        scale=0.5,
        noise=UniformNoise.configure(n_min=-0.5, n_max=0.5),  # ±0.25 after scale
    ),
    projected_gravity=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    dofs_pos=observations.DofsPos.configure(
        entity_name="robot", offset_from_default=True, noise=UniformNoise.configure(n_min=-0.02, n_max=0.02)
    ),
    dofs_vel=observations.DofsVel.configure(
        entity_name="robot",
        scale=0.05,
        noise=UniformNoise.configure(n_min=-2.0, n_max=2.0),  # ±0.1 after scale
    ),
    actions=ObsTerm.configure(func=observations.last_action, params={"action_name": "dofs_pos_controller"}),
)


@TASK_REGISTRY.register(name="homie")
class HomieLocomotionConfig(EdenRLConfig):
    """OpenHomie G1 velocity + base-height tracking controller (12-DOF legs, HIMLoco learner)."""

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=DECIMATION,  # 50 Hz control, as OpenHomie
        max_collision_pairs=64,
        use_gjk_collision=False,
        enable_multi_contact=False,
        # OpenHomie's G1 URDF has collision geometry on the two feet only, so nothing self-collides
        # there. With self-collision on, the upper-body curriculum drives the arms into the torso
        # and torso_contact ends the episode on a contact the leg policy cannot prevent.
        enable_self_collision=False,
        batch_dofs_info=True,  # per-env kp/kd DR
        batch_links_info=True,  # per-env mass / COM DR
        # The HIM estimator's target is the true *next* critic observation, which at an episode
        # boundary is the pre-reset one HIMPPO reads from extras["final_observations"].
        record_final_observations=True,
    )

    # Low-friction ground so the robot's randomized coefficient always wins Genesis' max-combine
    # (see randomize_friction).
    scene_options: SceneOptions = en.scenes.Empty(
        robot=homie_g1_robot(), plane=PlaneOptions(default_geoms_friction={"*": 0.01})
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        homie_command=HomieCommand.configure(
            entity_name="robot",
            resampling_time_range=(4.0, 4.0),
            lin_vel_x_range=(-0.8, 1.2),
            lin_vel_y_range=(-0.5, 0.5),
            ang_vel_z_range=(-0.8, 0.8),
            height_offset_range=(-0.5, 0.0),
            default_height=HOMIE_DEFAULT_HEIGHT,
        ),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # The 6-frame history feeds the HIM estimator; the actor itself sees the newest frame
        # plus the estimator's velocity and latent.
        policy=ObservationGroupOptions(
            **_OBSERVATIONS,
            enable_corruption=True,
            concatenate_terms=True,
            history_length=6,
            flatten_history_dim=True,
        ),
        # One clean frame + the privileged true base linear velocity, last (the estimator's
        # velocity target is read off the critic's final three entries).
        critic=ObservationGroupOptions(
            **_OBSERVATIONS,
            base_lin_vel=ObsTerm.configure(
                func=observations.base_lin_vel,
                params={"entity_name": "robot", "frame": "body"},
                scale=2.0,
            ),
            enable_corruption=False,
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # OpenHomie control_type='M': explicit PD, then the per-episode actuation offset and the
        # per-step joint injection, clipped to the effort limits (HomieMotorModel).
        dofs_pos_controller=actions.ExplicitPDController.configure(
            entity_name="robot",
            dofs_name=LEG_DOFS,
            scale=HOMIE_ACTION_SCALE,
            reference_source=ReferenceSource.DEFAULT,
            modifier=Compose.configure(
                modifiers=[
                    HomieMotorModel.configure(rfi_scale=0.05, offset_scale=0.05),
                    # U[0, decimation) physics steps of delay. OpenHomie configures this but never applies
                    # it (see "Differences" above); kept for sim-to-real.
                    ActionDelay.configure(delay_range_steps=(0, DECIMATION - 1)),
                ]
            ),
        ),
        upper_body=UpperBodyPoseCurriculum.configure(
            entity_name="robot",
            dofs_name=UPPER_BODY_DOFS,
            resample_interval_s=1.0,
            curriculum_length_steps=250_000,
            max_ratio=1.0,
            action_scale=HOMIE_ACTION_SCALE,
        ),
    )

    # Weights are OpenHomie's rewards.scales verbatim — the reward manager multiplies by dt, as
    # legged_gym does.
    reward_options: RewardManagerOptions = RewardManagerOptions(
        # --- Tracking ---
        tracking_x_vel=homie_rewards.TrackingXVel.configure(
            weight=1.5, command_name="homie_command", entity_name="robot", sigma=TRACKING_SIGMA
        ),
        tracking_y_vel=homie_rewards.TrackingYVel.configure(
            weight=1.0, command_name="homie_command", entity_name="robot", sigma=TRACKING_SIGMA
        ),
        tracking_ang_vel=rewards.TrackingAngVel.configure(
            weight=2.0, command_name="homie_command", entity_name="robot", sigma=TRACKING_SIGMA
        ),
        tracking_base_height=homie_rewards.TrackingBaseHeight.configure(
            weight=2.0,
            command_name="homie_command",
            entity_name="robot",
            feet_link_names=FEET_LINK_NAMES,
            ankle_sole_distance=0.02,
        ),
        # --- Velocity / orientation ---
        lin_vel_z=homie_rewards.LinVelZGated.configure(weight=-0.5, command_name="homie_command", entity_name="robot"),
        ang_vel_xy=RewardTerm.configure(func=homie_rewards.ang_vel_xy, weight=-0.025, params={"entity_name": "robot"}),
        orientation=rewards.FlatOrientationL2.configure(weight=-1.5, entity_name="robot"),
        # --- Effort / smoothness (action_rate and smoothness span the legs + the upper body) ---
        action_rate=homie_rewards.HomieActionRate.configure(weight=-0.01),
        smoothness=homie_rewards.HomieActionSmoothness.configure(weight=-0.05),
        dof_acc=homie_rewards.HomieDofsAcc.configure(weight=-2.5e-7, entity_name="robot"),
        dof_pos_limits=homie_rewards.HomieDofsPosLimits.configure(weight=-2.0, entity_name="robot"),
        dof_vel_limits=homie_rewards.HomieDofsVelLimits.configure(weight=-2.0e-3, entity_name="robot"),
        dof_vel=homie_rewards.HomieDofsVel.configure(weight=-1.0e-4, entity_name="robot"),
        torques=homie_rewards.HomieTorques.configure(weight=-2.5e-6, entity_name="robot"),
        torque_limits=homie_rewards.HomieTorqueLimits.configure(weight=-0.1, entity_name="robot"),
        joint_power=homie_rewards.HomieJointPower.configure(
            weight=-2.0e-5, command_name="homie_command", entity_name="robot"
        ),
        joint_tracking_error=homie_rewards.HomieJointTrackingError.configure(weight=-0.1, entity_name="robot"),
        action_vanish=homie_rewards.HomieActionVanish.configure(weight=-1.0, entity_name="robot"),
        # --- Posture (gated on standing) ---
        deviation_hip_joint=homie_rewards.DeviationHipJoint.configure(
            weight=-0.2, command_name="homie_command", entity_name="robot"
        ),
        deviation_ankle_joint=homie_rewards.DeviationAnkleJoint.configure(
            weight=-0.5, command_name="homie_command", entity_name="robot"
        ),
        deviation_knee_joint=homie_rewards.DeviationKneeJoint.configure(
            weight=-0.75, command_name="homie_command", entity_name="robot"
        ),
        # --- Feet / contact ---
        feet_air_time=homie_rewards.HomieFeetAirTime.configure(
            weight=0.05, command_name="homie_command", entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        feet_clearance=homie_rewards.HomieFeetClearance.configure(
            weight=-0.25, command_name="homie_command", entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        feet_slip=homie_rewards.HomieFeetSlip.configure(
            weight=-0.25, entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        # OpenHomie's stumble ratio is 3.0 (|F_xy| > 3 |F_z|), not Eden's default 4.0.
        feet_stumble=rewards.StumblePenalty.configure(
            weight=-1.5, entity_name="robot", feet_link_names=FEET_LINK_NAMES, horizontal_to_vertical_ratio=3.0
        ),
        feet_contact_forces=homie_rewards.HomieFeetContactForces.configure(
            weight=-0.00025, entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        contact_momentum=homie_rewards.HomieContactMomentum.configure(
            weight=2.5e-4, entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        feet_distance_lateral=homie_rewards.FeetDistanceLateral.configure(
            weight=0.5, command_name="homie_command", entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        knee_distance_lateral=homie_rewards.KneeDistanceLateral.configure(
            weight=1.0, command_name="homie_command", entity_name="robot"
        ),
        feet_ground_parallel=homie_rewards.FeetGroundParallel.configure(
            weight=-2.0, entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        feet_parallel=homie_rewards.FeetParallel.configure(
            weight=-3.0, command_name="homie_command", entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        no_fly=homie_rewards.HomieNoFly.configure(
            weight=0.75, command_name="homie_command", entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
        stand_still=homie_rewards.HomieStandStill.configure(
            weight=-0.15, command_name="homie_command", entity_name="robot", feet_link_names=FEET_LINK_NAMES
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        # |proj_grav_xy| > 0.8, i.e. a lean past ~53 deg.
        fell_over=DoneTerm.configure(
            func=terminations.fell_over,
            params={"entity_name": "robot", "limit_angle": math.radians(53.0)},
        ),
        # OpenHomie terminate_after_contacts_on=['torso'], > 10 N.
        torso_contact=terminations.IllegalLinkContact.configure(
            entity_name="robot", links_name=["torso_link"], force_threshold=10.0
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # --- Resets ---
        reset_base=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode="reset",
            params={
                "entity_name": "robot",
                "pos_x_range": (-1.0, 1.0),
                "pos_y_range": (-1.0, 1.0),
                "pos_z_range": (0.0, 0.1),
                "yaw_range": (-3.14, 3.14),
            },
        ),
        # default * U(0.8, 1.2) + U(-0.1, 0.1), OpenHomie's initial joint pose.
        reset_dofs=EventTerm.configure(
            func=events.reset_dofs_by_scale_and_offset,
            mode="reset",
            params={"entity_name": "robot", "dofs_scale_range": (0.8, 1.2), "dofs_offset_range": (-0.1, 0.1)},
        ),
        # OpenHomie sets all six base-velocity axes to U(-0.5, 0.5) on reset. The resets above
        # zero the base velocity, so this additive push is that SET; priority keeps it after the
        # pose reset, so the angular axes are drawn in the episode's own base frame.
        reset_base_velocity=events.PushByVelocity.configure(
            mode="reset",
            priority=10,
            entity_name="robot",
            velocity_range={axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        ),
        # --- Pushes: every 4 s, SET the planar base velocity (OpenHomie max_push_vel_xy=0.5) ---
        push_robot=events.PushSetPlanarVelocity.configure(
            mode="interval", interval_range_s=(4.0, 4.0), entity_name="robot", velocity_xy_range=(-0.5, 0.5)
        ),
        # --- Physics randomization (OpenHomie domain_rand ranges) ---
        # OpenHomie's friction_range (0.1, 3.0) averaged with its 1.0 plane, as Isaac Gym combines it.
        randomize_friction=events.RandomizeFrictionRatio.configure(
            mode="startup", entity_name="robot", links_name=["*"], friction_range=(0.55, 2.0)
        ),
        randomize_torso_payload=events.RandomizeMassShift.configure(
            mode="startup", entity_name="robot", links_name=["torso_link"], mass_shift_range=(-5.0, 10.0)
        ),
        randomize_hand_payload=events.RandomizeMassShift.configure(
            mode="startup", entity_name="robot", links_name=["*_wrist_yaw_link"], mass_shift_range=(-0.1, 0.3)
        ),
        randomize_link_mass=events.RandomizeLinkMassScale.configure(
            mode="startup", entity_name="robot", links_name=_LINK_MASS_SCALE_LINKS, mass_scale_range=(0.8, 1.2)
        ),
        randomize_com=events.RandomizeComShift.configure(
            mode="startup",
            entity_name="robot",
            links_name=["torso_link"],
            com_shift_x_range=(-0.1, 0.1),
            com_shift_y_range=(-0.1, 0.1),
            com_shift_z_range=(-0.1, 0.1),
        ),
        # action_term_name=None writes the entity gains (all 29 DOFs); the explicit leg
        # controller re-reads its kp/kd from the entity on reset.
        randomize_pd_gains=events.RandomizeKpKdGains.configure(
            mode="reset", entity_name="robot", action_term_name=None, kp_range=(0.9, 1.1), kd_range=(0.9, 1.1)
        ),
        # Eden-only: a startup encoder-calibration bias on the leg targets.
        encoder_bias=events.RandomizeDofsPosOffset.configure(
            mode="startup", entity_name="robot", action_term_name="dofs_pos_controller", dofs_pos_range=(-0.02, 0.02)
        ),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        experiment_name="homie",
        max_iterations=30000,
        num_steps_per_env=50,
        save_interval=200,
        # OpenHomie's normalization.clip_actions. Unclipped, one runaway action mean makes the
        # quadratic action penalties (action_rate, smoothness, joint_tracking_error) explode:
        # an unclipped 30k run blew its value loss up to ~1e21 and collapsed at iteration 29000.
        clip_actions=100.0,
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        algorithm=HIMPPOAlgorithmOptions(
            data_augmentation_func="eden.tasks.homie.symmetry:mirror_observations_and_actions",
            clip_param=0.2,
            desired_kl=0.01,
            entropy_coef=0.01,
            gamma=0.99,
            lam=0.95,
            learning_rate=1.0e-3,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
        ),
        # HIMActorCritic reads hidden_dims, activation and init_std from these. OpenHomie uses
        # fixed observation scales, so no empirical normalization.
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 256],
            obs_normalization=False,
            # HIM learns a raw std (OpenHomie's), and refuses any other std_type.
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[512, 256, 256], obs_normalization=False),
    )
