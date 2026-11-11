import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eden as en
from eden.terms import ObsTerm, RewardTerm, DoneTerm, MetricTerm
from eden import actions, events, observations, rewards, terminations
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.managers.terms.commands.mimic import MotionCommand
from eden.extensions.policies.beyondmimic_moves.custom_terms import metrics
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.utils.configs import (
    EdenRLConfig,
    EnvOptions,
    RewardManagerOptions,
    CommandManagerOptions,
    SceneOptions,
    ActionManagerOptions,
    ObservationManagerOptions,
    TerminationManagerOptions,
    EventManagerOptions,
    MetricManagerOptions,
)
from eden.constants import ReferenceSource

from custom_terms.observations import (
    motion_anchor_ori_b,
    motion_anchor_pos_b,
    beyondmimic_robot_body_pos_b,
    beyondmimic_robot_body_ori_b,
)
from custom_terms.rewards import (
    motion_global_anchor_position_error_exp,
    motion_global_anchor_orientation_error_exp,
    BeyondmimicMotionRelativeBodyPositionErrorExp,
    BeyondmimicMotionRelativeBodyOrientationErrorExp,
    BeyondmimicMotionGlobalBodyLinearVelocityErrorExp,
    BeyondmimicMotionGlobalBodyAngularVelocityErrorExp,
)
from custom_terms.terminations import (
    beyondmimic_bad_anchor_pos_z_only,
    beyondmimic_bad_anchor_ori,
    BeyondmimicBadMotionBodyPosZOnly,
)


_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# NOTE: lazy initialization to reuse the same instance for multiple fields
_G1Options = en.robots.G1Dofs29()
_MotionCommandOptions = MotionCommand.configure(
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=False,
    entity_name="robot",
    ghost_entity_name="",
    motion_file="motion.npz",
    anchor_link_name="torso_link",
    links_name=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    },
    velocity_range=_VELOCITY_RANGE,
    dofs_position_range=(-0.1, 0.1),
)
_CommonObservations = dict(
    motion=ObsTerm.configure(func=observations.generated_commands, params={"command_name": "motion"}),
    motion_anchor_pos_b=ObsTerm.configure(
        func=motion_anchor_pos_b,
        params={"command_name": "motion"},
        noise=UniformNoise.configure(n_min=-0.25, n_max=0.25),
    ),
    motion_anchor_ori_b=ObsTerm.configure(
        func=motion_anchor_ori_b,
        params={"command_name": "motion"},
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.5, n_max=0.5),
    ),
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
    ),
    dofs_pos=observations.DofsPos.configure(
        entity_name="robot", offset_from_default=True, noise=UniformNoise.configure(n_min=-0.01, n_max=0.01)
    ),
    dofs_vel=observations.DofsVel.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-0.5, n_max=0.5)),
    last_action=ObsTerm.configure(func=observations.last_action),
)


class BeyondMimicConfig(EdenRLConfig):
    """Configuration for the BeyondMimic tracking policy environment."""

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=10.0,
        iterations=10,
        ls_iterations=20,
        max_collision_pairs=20,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,
        batch_dofs_info=False,
        batch_links_info=True,  # RandomizeComShift writes per-env link COMs
        use_gjk_collision=False,
        enable_self_collision=False,
        enable_multi_contact=False,
    )

    scene_options: SceneOptions = en.scenes.Empty(robot=_G1Options)

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=True,
            **_CommonObservations,
        ),
        critic=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=False,
            **_CommonObservations,
            body_pos_b=ObsTerm.configure(func=beyondmimic_robot_body_pos_b, params={"command_name": "motion"}),
            body_ori_b=ObsTerm.configure(func=beyondmimic_robot_body_ori_b, params={"command_name": "motion"}),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        motion_global_anchor_pos=RewardTerm.configure(
            func=motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        motion_global_anchor_ori=RewardTerm.configure(
            func=motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        motion_body_pos=BeyondmimicMotionRelativeBodyPositionErrorExp.configure(
            weight=1.0,
            command_name="motion",
            entity_name="robot",
            links_name=_MotionCommandOptions.links_name,
            std=0.3,
        ),
        motion_body_ori=BeyondmimicMotionRelativeBodyOrientationErrorExp.configure(
            weight=1.0,
            command_name="motion",
            entity_name="robot",
            links_name=_MotionCommandOptions.links_name,
            std=0.4,
        ),
        motion_body_lin_vel=BeyondmimicMotionGlobalBodyLinearVelocityErrorExp.configure(
            weight=1.0,
            command_name="motion",
            entity_name="robot",
            links_name=_MotionCommandOptions.links_name,
            std=1.0,
        ),
        motion_body_ang_vel=BeyondmimicMotionGlobalBodyAngularVelocityErrorExp.configure(
            weight=1.0,
            command_name="motion",
            entity_name="robot",
            links_name=_MotionCommandOptions.links_name,
            std=3.14,
        ),
        action_rate_l2=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-1e-1,
        ),
        joint_limit=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
        self_collisions=RewardTerm.configure(
            func=rewards.self_collision_cost,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        anchor_pos=DoneTerm.configure(
            func=beyondmimic_bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},
        ),
        anchor_ori=DoneTerm.configure(
            func=beyondmimic_bad_anchor_ori, params={"command_name": "motion", "threshold": 0.8}
        ),
        ee_body_pos=BeyondmimicBadMotionBodyPosZOnly.configure(
            command_name="motion",
            threshold=0.25,
            entity_name="robot",
            links_name=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        push_robot=events.PushByVelocity.configure(
            mode=en.EventMode.INTERVAL,
            interval_range_s=(1.0, 3.0),
            entity_name="robot",
            base_joint_name="floating_base_joint",
            velocity_range=_VELOCITY_RANGE,
        ),
        base_com_shift=events.RandomizeComShift.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["pelvis"],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.05, 0.05),
            com_shift_z_range=(-0.05, 0.05),
        ),
        dofs_default_pos=events.RandomizeDofsPosOffset.configure(
            mode=en.EventMode.STARTUP,
            action_term_name="dofs_pos_controller",
            entity_name="robot",
            dofs_pos_range=(-0.01, 0.01),
        ),
        foot_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["*_ankle_roll_link"],
            friction_range=(0.3, 1.2),
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(motion=_MotionCommandOptions)

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_G1Options.dofs_name,
            scale=en.robots.G1_ACTION_SCALE,
            reference_source=ReferenceSource.DEFAULT,
        )
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        error_anchor_pos=MetricTerm.configure(
            func=metrics.error_anchor_pos,
            params={"command_name": "motion"},
        ),
        error_anchor_rot=MetricTerm.configure(
            func=metrics.error_anchor_rot,
            params={"command_name": "motion"},
        ),
        error_anchor_lin_vel=MetricTerm.configure(
            func=metrics.error_anchor_lin_vel,
            params={"command_name": "motion"},
        ),
        error_anchor_ang_vel=MetricTerm.configure(
            func=metrics.error_anchor_ang_vel,
            params={"command_name": "motion"},
        ),
        error_body_pos=MetricTerm.configure(
            func=metrics.error_body_pos,
            params={"command_name": "motion"},
        ),
        error_body_rot=MetricTerm.configure(
            func=metrics.error_body_rot,
            params={"command_name": "motion"},
        ),
        error_body_lin_vel=MetricTerm.configure(
            func=metrics.error_body_lin_vel,
            params={"command_name": "motion"},
        ),
        error_body_ang_vel=MetricTerm.configure(
            func=metrics.error_body_ang_vel,
            params={"command_name": "motion"},
        ),
        error_dofs_pos=MetricTerm.configure(
            func=metrics.error_dofs_pos,
            params={"command_name": "motion"},
        ),
        error_dofs_vel=MetricTerm.configure(
            func=metrics.error_dofs_vel,
            params={"command_name": "motion"},
        ),
    )

    # NOTE: include the training config here for reproducibility.
    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        num_steps_per_env=24,
        max_iterations=15000,
        save_interval=500,
        experiment_name="beyondmimic",
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
        ),
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=0.001,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
