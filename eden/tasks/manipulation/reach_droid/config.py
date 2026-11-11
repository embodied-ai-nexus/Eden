"""Reach (``manipulation_reach_droid``): move the Franka + Robotiq grasp point to a sampled 3D target.

This is the manipulation analog of ``eden/tasks/benchmark/reacher`` (the
canonical Eden reach task): joint-position arm control, a uniform T(3) target
command, an exponential reaching reward, and the same action-rate /
joint-limit / joint-velocity penalties with a velocity-penalty curriculum.
The only differences from the reacher are the robot (Franka + Robotiq instead
of Yam) and that the end-effector is the gripper finger-pad midpoint.

Mirrors mjlab ``Reach-{Robot}`` (``mjlab_manipulation/tasks/reach``), which is
likewise ``lift`` with the object/gripper removed and a reach command added.
"""

from __future__ import annotations

import eden as en
from eden import actions, observations, rewards, terminations
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
from eden.constants import ReferenceSource
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import DoneTerm, ObsTerm, RewardTerm
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
from eden.options.robots import Droid


_DROID = Droid()


def _droid_reach_scene():
    return en.scenes.Empty(is_support_enabled=True, robot=_DROID)


_GRASP = {
    "robot_name": _DROID.arm_name,
    "gripper_name": _DROID.gripper_name,
    "grasp_links": _DROID.grasp_links,
    "approach_offset": 0.065,
}

_Observations = dict(
    dofs_pos=observations.DofsPos.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-0.01, n_max=0.01)),
    dofs_vel=observations.DofsVel.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-1.5, n_max=1.5)),
    target_position=ObsTerm.configure(
        func=observations.generated_commands,
        params={"command_name": "reaching_target"},
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    grasp_point=ObsTerm.configure(
        func=custom_terms.grasp_point_b,
        params=dict(_GRASP),
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    last_action=ObsTerm.configure(
        func=observations.last_action,
        params={"action_name": "arm_joints"},
    ),
)


@TASK_REGISTRY.register()
class ManipulationReachDroidConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=9.0,
        sim_dt=0.005,
        sim_substeps=2,
        decimation=4,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = _droid_reach_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # Relative (incremental) joint-position control (reference_source=DELTA):
        # target = q_now + 0.05*a, not the absolute q_default + 0.25*a. The
        # incremental delta low-pass filters policy/observation noise (the reason
        # mjlab's relative-action reacher is smooth); mjlab uses scale 0.05 too.
        arm_joints=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_DROID.arm_dofs,
            scale=0.05,
            reference_source=ReferenceSource.DELTA,
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        reaching_target=UniformT3Command.configure(
            x_range=(0.3, 0.5),
            y_range=(-0.2, 0.2),
            z_range=(0.2, 0.4),
            resampling_time_range=(2.0, 3.0),
        )
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching_task=RewardTerm.configure(
            func=custom_terms.reaching_target,
            weight=1.0,
            params={**_GRASP, "command_name": "reaching_target", "reaching_std": 0.2},
        ),
        action_rate=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-0.01,
        ),
        joint_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
        joint_vel_hinge=RewardTerm.configure(
            func=rewards.dofs_vel_limits,
            weight=-0.01,
            params={"entity_name": "robot", "max_vel": 0.5},
        ),
    )

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        joint_vel_weight=StageRewardWeightCurriculum.configure(
            reward_term_name="joint_vel_hinge",
            weight_stages=[
                (0, -0.01),
                (100 * 24, -0.1),
                (200 * 24, -1.0),
            ],
        )
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )

    event_options: EventManagerOptions = EventManagerOptions()

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
            hidden_dims=[256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[256, 128],
        ),
        max_iterations=300,
        save_interval=100,
        experiment_name="reach_droid",
        run_name="v1",
    )
