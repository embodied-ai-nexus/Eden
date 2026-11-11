"""Configuration for the Go1 handstand task.

Ported from: mujoco_playground/_src/locomotion/go1/handstand.py

Observation space (45 dims):
    - Base linear velocity in body frame (3) — with noise
    - Base angular velocity / gyro (3) — with noise
    - Projected gravity (3) — with noise
    - Joint angles offset from default (12) — with noise
    - Joint velocities (12) — with noise
    - Last action delta (12)

Action space: 12 joint position deltas.
    The PD controller uses ReferenceSource.ACCUMULATE: target += delta * 0.3, clamped
    to the soft joint limits, so the action manager holds the raw policy delta directly
    (read by the `last_action` obs). Deltas are clamped to [-1, 1] via the runner's
    clip_actions.

Reward function (active by default):
    - height: torso at desired height (0.55m) — exp(-(z_des - min(h, z_des)))
    - orientation: body x-axis should point down — (0.5*cos+0.5)^2
    - contact: penalize rear feet touching floor
    - dof_pos_limits: penalize soft joint limit violations
    - pose: penalize rear leg joints deviating from default

Not configured (zero-weight in the reference):
    - action_rate, torques, dof_acc, stay_still, energy, termination
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
from eden.tasks.mujoco_playground.handstand import custom_terms  # noqa: F401


# Noise levels from reference:
#   joint_pos: 0.01, joint_vel: 1.5, gyro: 0.2, gravity: 0.05, linvel: 0.1

_PolicyObs = ObservationGroupOptions(
    concatenate_terms=True,
    # Base linear velocity in body frame (3) — reference: linvel noise 0.1
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.1, n_max=0.1),
        clip=(-100.0, 100.0),
    ),
    # Gyroscope / angular velocity in body frame (3) — reference: gyro noise 0.2
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
        clip=(-100.0, 100.0),
    ),
    # Projected gravity (3) — reference: gravity noise 0.05
    projected_gravity=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    # Joint positions offset from default (12) — reference: joint_pos noise 0.01
    dofs_pos=observations.DofsPos.configure(
        entity_name="robot",
        offset_from_default=True,
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
        clip=(-100.0, 100.0),
    ),
    # Joint velocities (12) — reference: joint_vel noise 1.5
    dofs_vel=observations.DofsVel.configure(
        entity_name="robot", noise=UniformNoise.configure(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0)
    ),
    # Last action delta (12) — the raw policy delta, from the action manager
    last_action=ObsTerm.configure(
        func=custom_terms.handstand_last_action,
    ),
)

# Critic gets privileged observations (state + clean sensors + torques + height)
_CriticObs = ObservationGroupOptions(
    concatenate_terms=True,
    # Same noisy observations as policy
    base_lin_vel_noisy=ObsTerm.configure(
        func=observations.base_lin_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.1, n_max=0.1),
        clip=(-100.0, 100.0),
    ),
    base_ang_vel_noisy=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
        clip=(-100.0, 100.0),
    ),
    projected_gravity_noisy=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    dofs_pos_noisy=observations.DofsPos.configure(
        entity_name="robot",
        offset_from_default=True,
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
        clip=(-100.0, 100.0),
    ),
    dofs_vel_noisy=observations.DofsVel.configure(
        entity_name="robot", noise=UniformNoise.configure(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0)
    ),
    last_action=ObsTerm.configure(
        func=custom_terms.handstand_last_action,
    ),
    # Privileged: clean gyro
    base_ang_vel_clean=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        clip=(-100.0, 100.0),
    ),
    # Privileged: clean linear velocity in body frame
    base_lin_vel_clean=ObsTerm.configure(
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


@TASK_REGISTRY.register()
class MujocoPlaygroundHandstandConfig(EdenRLConfig):
    """Go1 handstand task configuration."""

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=10.0,  # 500 steps * 0.02s ctrl_dt = 10s
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
        fall=DoneTerm.configure(
            func=custom_terms.handstand_fall,
            params={"entity_name": "robot", "threshold": -0.25},
        ),
        front_body_contact=custom_terms.HandstandFrontBodyContact.configure(
            entity_name="robot", contact_threshold=0.05
        ),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        experiment_name="go1_handstand",
        max_iterations=3000,
        num_steps_per_env=24,
        save_interval=200,
        clip_actions=1.0,  # clamp raw policy deltas to [-1, 1] before the integrator
        algorithm=RslRlPpoAlgorithmOptions(
            class_name="PPO",
            clip_param=0.2,
            desired_kl=0.01,
            entropy_coef=0.0,
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

    # Action: 12 joint position deltas, integrated onto a running target.
    # ReferenceSource.ACCUMULATE makes the controller compute target += delta * 0.3
    # and clamp to the soft joint limits, so the action manager stores the raw delta
    # (read by the `last_action` obs).
    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_GO1.dofs_name,
            scale=0.3,
            reference_source=ReferenceSource.ACCUMULATE,
        )
    )

    # Rewards: Eden's reward_manager multiplies by dt, matching
    # the reference `reward = clip(sum(scale_i * r_i) * dt, 0, 10000)`.
    reward_options: RewardManagerOptions = RewardManagerOptions(
        height=RewardTerm.configure(
            func=custom_terms.handstand_height,
            weight=1.0,
            params={"entity_name": "robot", "z_des": 0.55},
        ),
        orientation=RewardTerm.configure(
            func=custom_terms.handstand_orientation,
            weight=1.0,
            params={"entity_name": "robot", "desired_forward": (0.0, 0.0, -1.0)},
        ),
        contact=custom_terms.HandstandRearFeetContact.configure(
            weight=-0.1,
            entity_name="robot",
        ),
        dof_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-0.5,
            params={"entity_name": "robot"},
        ),
        pose=RewardTerm.configure(
            func=custom_terms.handstand_pose,
            weight=-0.1,
            params={"entity_name": "robot"},
        ),
    )

    # Events: standing start with random yaw and root velocity
    event_options: EventManagerOptions = EventManagerOptions(
        random_standing_reset=custom_terms.RandomStandingReset.configure(
            entity_name="robot",
            xy_range=(-0.5, 0.5),
            yaw_range=(-3.14, 3.14),
            root_vel_range=(-0.5, 0.5),
            mode=EventMode.RESET,
        ),
    )
