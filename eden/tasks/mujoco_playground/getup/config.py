"""Configuration for the Go1 getup (fall recovery) task.

Ported from: mujoco_playground/_src/locomotion/go1/getup.py

Observation space (42 dims):
    - Projected gravity (3) — with noise
    - Base angular velocity / gyro (3) — with noise
    - Joint angles offset from default (12) — with noise
    - Joint velocities (12) — with noise
    - Last action (12)

Action space: 12 joint position targets (scaled, offset from default pose).

Reward function:
    - orientation: torso should be upright
    - torso_height: torso at desired height (0.275m)
    - posture: default pose when upright & at height
    - stand_still: zero action when upright & at height
    - similar_to_default: stay near the default pose (from 1 s into the episode)
    - hip_abduction: penalize hip abduction away from default
    - action_rate: minimize 1st + 2nd derivative of actions
    - torques: minimize joint torques
    - dof_pos_limits: penalize soft joint limit violations
    - dof_acc: penalize joint accelerations
    - dof_vel: penalize excessive joint velocities
"""

from __future__ import annotations

import eden as en
from eden.utils.configs import (
    EdenRLConfig,
    EnvOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
    EventManagerOptions,
    ActionManagerOptions,
)
from eden.options import ObservationGroupOptions, SceneOptions
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks import TASK_REGISTRY
from eden.terms import ObsTerm, RewardTerm, DoneTerm
from eden import actions, observations, rewards, terminations
from eden.constants import EventMode, ReferenceSource
from eden.managers.modifiers.observations.noise import UniformNoise

# Import custom terms so they register with the global registries
from eden.tasks.mujoco_playground.getup import custom_terms  # noqa: F401


# Noise levels matching the reference:
#   joint_pos: 0.03, joint_vel: 1.5, gyro: 0.2, gravity: 0.05
# The reference uses uniform noise in [-level*scale, +level*scale] with level=1.0

_PolicyObs = ObservationGroupOptions(
    concatenate_terms=True,
    # Projected gravity (3) — reference: gravity noise scale 0.05
    projected_gravity=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    # Gyroscope / angular velocity in body frame (3) — reference: gyro noise scale 0.2
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
        clip=(-100.0, 100.0),
    ),
    # Joint positions offset from default (12) — reference: joint_pos noise scale 0.03
    dofs_pos=observations.DofsPos.configure(
        entity_name="robot",
        offset_from_default=True,
        noise=UniformNoise.configure(n_min=-0.03, n_max=0.03),
        clip=(-100.0, 100.0),
    ),
    # Joint velocities (12) — reference: joint_vel noise scale 1.5
    dofs_vel=observations.DofsVel.configure(
        entity_name="robot", noise=UniformNoise.configure(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0)
    ),
    # Last action (12) — the raw policy delta the controller consumes
    last_action=ObsTerm.configure(
        func=observations.last_action,
    ),
)

# Critic gets privileged observations (state + clean sensors + torques + height)
_CriticObs = ObservationGroupOptions(
    concatenate_terms=True,
    # Same noisy observations as policy
    projected_gravity_noisy=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    base_ang_vel_noisy=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
        clip=(-100.0, 100.0),
    ),
    dofs_pos_noisy=observations.DofsPos.configure(
        entity_name="robot",
        offset_from_default=True,
        noise=UniformNoise.configure(n_min=-0.03, n_max=0.03),
        clip=(-100.0, 100.0),
    ),
    dofs_vel_noisy=observations.DofsVel.configure(
        entity_name="robot", noise=UniformNoise.configure(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0)
    ),
    last_action=ObsTerm.configure(
        func=observations.last_action,
    ),
    # Privileged: clean angular velocity (gyro)
    base_ang_vel_clean=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        clip=(-100.0, 100.0),
    ),
    # Privileged: linear velocity in body frame
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        params={"entity_name": "robot", "frame": "body"},
        clip=(-100.0, 100.0),
    ),
    # Privileged: angular velocity in world frame
    base_ang_vel_world=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "world"},
        clip=(-100.0, 100.0),
    ),
    # Privileged: clean joint positions (absolute)
    dofs_pos_clean=observations.DofsPos.configure(entity_name="robot", offset_from_default=False, clip=(-100.0, 100.0)),
    # Privileged: clean joint velocities
    dofs_vel_clean=observations.DofsVel.configure(entity_name="robot", clip=(-100.0, 100.0)),
    # Privileged: joint torques (actuator force)
    dofs_force=observations.DofsForce.configure(entity_name="robot", clip=(-100.0, 100.0)),
    # Privileged: base position (includes torso height as z)
    base_pos=ObsTerm.configure(
        func=observations.base_pos,
        params={"entity_name": "robot"},
    ),
)


_GO1 = en.robots.Go1()
_DT = 0.02  # ctrl_dt


@TASK_REGISTRY.register()
class MujocoPlaygroundGetupConfig(EdenRLConfig):
    """Go1 fall recovery (getup) task configuration."""

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        env_spacing=(0.5, 0.5),
        episode_length_s=6.0,  # 300 steps * 0.02s ctrl_dt = 6s
        sim_dt=0.004,
        sim_substeps=1,
        decimation=5,  # ctrl_dt = 0.004 * 5 = 0.02s
        max_collision_pairs=30,
        use_gjk_collision=False,
        batch_dofs_info=False,
        enable_multi_contact=False,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=_PolicyObs,
        critic=_CriticObs,
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        body_contact=custom_terms.GetupBodyContact.configure(entity_name="robot", contact_threshold=0.05, delay_s=1.0),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        experiment_name="go1_getup",
        max_iterations=3000,
        num_steps_per_env=24,
        save_interval=200,
        clip_actions=1.0,  # clamp raw policy deltas to [-1, 1] before they reach the controller
        algorithm=RslRlPpoAlgorithmOptions(
            class_name="PPO",
            clip_param=0.2,
            desired_kl=0.01,
            entropy_coef=0.01,
            gamma=0.99,
            lam=0.95,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            obs_normalization=False,
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            obs_normalization=False,
        ),
    )

    scene_options: SceneOptions = en.scenes.Empty(robot=_GO1)

    # Action: 12 joint position deltas on top of the CURRENT joint position.
    # ReferenceSource.DELTA makes the controller compute
    #   target = current_qpos + action * scale
    # (matching reference: motor_targets = qpos[7:] + action * action_scale),
    # so the action manager holds the raw policy delta directly, which the obs
    # `last_action` and the action-rate / stand-still rewards read. Raw actions
    # are clamped to [-1, 1] via the runner's `clip_actions` (Brax PPO convention).
    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_GO1.dofs_name,
            scale=0.5,
            reference_source=ReferenceSource.DELTA,
        )
    )

    # Rewards: Eden's reward_manager already multiplies by dt, matching
    # the reference `reward = clip(sum(scale_i * r_i) * dt, 0, 10000)`.
    # So weights = reference scales directly.
    # Exception: dof_acc uses sum(dvel^2) instead of sum(qacc^2).
    #   qacc = dvel / dt  =>  qacc^2 = dvel^2 / dt^2
    #   ref contribution per step = scale * sum(qacc^2) * dt = scale * sum(dvel^2) / dt
    #   eden contribution per step = weight * sum(dvel^2) * dt
    #   => weight = scale / dt^2 = -2.5e-7 / 0.02^2 = -6.25e-4
    reward_options: RewardManagerOptions = RewardManagerOptions(
        orientation=custom_terms.GetupOrientation.configure(
            weight=1.0,
            entity_name="robot",
        ),
        torso_height=custom_terms.GetupTorsoHeight.configure(
            weight=1.0,
            entity_name="robot",
            z_des=0.275,
        ),
        posture=custom_terms.GetupPosture.configure(
            weight=1.0,
            entity_name="robot",
            z_des=0.275,
        ),
        stand_still=custom_terms.GetupStandStill.configure(
            weight=1.0,
            entity_name="robot",
            z_des=0.275,
        ),
        similar_to_default=RewardTerm.configure(
            func=rewards.similar_to_default,
            weight=-1.0,
            params={"entity_name": "robot"},
            range_s=[(1.0, 6.0)],
        ),
        hip_abduction=custom_terms.GetupHipAbductionPenalty.configure(
            weight=-0.5,
            entity_name="robot",
        ),
        # Penalizes the 1st + 2nd derivative of the raw policy deltas, which live
        # in the action manager (ReferenceSource.DELTA), so the stock term applies.
        action_rate=rewards.ActionRateL2Smooth.configure(
            weight=-0.001,
        ),
        dof_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-0.1,
            params={"entity_name": "robot"},
        ),
        torques=RewardTerm.configure(
            func=rewards.dofs_torques,
            weight=-1e-5,
            params={"entity_name": "robot"},
        ),
        dof_acc=rewards.DofsAccL2.configure(
            weight=-2.5e-7 / (_DT * _DT),  # compensate for using dvel instead of qacc
            entity_name="robot",
        ),
        dof_vel=RewardTerm.configure(
            func=rewards.dofs_vel_limits,
            weight=-0.1,
            params={"entity_name": "robot", "max_vel": 6.283185},
        ),
    )

    # Events: drop from height with random orientation at reset
    event_options: EventManagerOptions = EventManagerOptions(
        drop_from_height=custom_terms.DropFromHeight.configure(
            entity_name="robot",
            drop_height=0.5,
            drop_prob=0.6,
            root_vel_range=(-0.5, 0.5),
            mode=EventMode.RESET,
        ),
    )
