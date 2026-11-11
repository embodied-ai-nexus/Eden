"""Reacher benchmark task configuration (the canonical task-config template)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal
import torch

from eden.utils.configs import (
    EdenRLConfig,
    EnvOptions,
    SceneOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
    EventManagerOptions,
    CommandManagerOptions,
    ActionManagerOptions,
    CurriculumManagerOptions,
)
import eden as en
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.terms import ObsTerm, RewardTerm, DoneTerm
from eden import actions, observations, rewards, terminations
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.observations.manipulation import (
    ObjectToOracleDistance,
)
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
from eden.managers import REWARD_TERM_REGISTRY
from eden.constants import ReferenceSource


if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@REWARD_TERM_REGISTRY.register()
def reaching_reward(
    env: EnvBase,
    *,
    robot_name: str,
    command_name: str,
    reaching_std: float,
) -> torch.Tensor:
    """Reward for reaching the target."""
    robot: Entity = env.entities[robot_name]
    command = env.command_manager.get_term(command_name)

    # Get mean position of grippers for EE position
    _, ee_idx_local = robot.find_named_links_idx_local(
        ["link_left_finger", "link_right_finger"],
        preserve_order=True,
    )
    ee_pos_w = robot.get_links_pos(ls_idx_local=ee_idx_local).mean(dim=1)

    # Reaching reward: distance between EE and target
    reach_dist_sq = torch.sum(torch.square(ee_pos_w - command.command[..., :3]), dim=-1)
    reaching = torch.exp(-reach_dist_sq / reaching_std**2)

    return reaching


_Observations = dict(
    dofs_pos=observations.DofsPos.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-0.01, n_max=0.01)),
    dofs_vel=observations.DofsVel.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-1.5, n_max=1.5)),
    object_to_oracle_distance=ObjectToOracleDistance.configure(
        entity_name="robot",
        entity_link_name="link_left_finger",
        command_name="reaching_target",
        robot_name="robot",
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    last_action=ObsTerm.configure(
        func=observations.last_action,
        params={"action_name": "joint_pos_controller"},
    ),
)


_DIFFICULTY_PRESETS = {
    # name → (target half-extent xy, target half-extent z, reaching_std, episode_length_s)
    "easy": (0.15, 0.075, 0.30, 12.0),
    "medium": (0.10, 0.050, 0.20, 9.0),  # matches ReacherConfig defaults
    "hard": (0.05, 0.025, 0.10, 6.0),
}


class ReacherDifficultyMod(TaskMod):
    """Coordinated preset for target spread, reward sharpness, and episode length.

    A single ``--difficulty`` flag changes:
    - ``command_options.reaching_target.{x,y,z}_range`` — target placement spread
    - ``reward_options.reaching_task.params.reaching_std`` — exponential reward width
    - ``env_options.episode_length_s`` — time the policy gets to reach

    These are correlated: shrinking the target without sharpening the reward or
    shortening the episode would just produce a longer, sparser-reward task.
    """

    def __init__(
        self,
        # Positional-or-keyword params with type annotations become CLI flags.
        # Keyword-only params (after `*`) are non-CLI ctor state — keep them
        # for things you set at registration time, not per build.
        # Wrap the type in `Annotated[T, "..."]` to populate argparse --help.
        difficulty: Annotated[
            Literal["easy", "medium", "hard"],
            "Reaching difficulty: scales target spread, reward sharpness, episode length.",
        ] = "medium",
    ):
        self.difficulty = difficulty

    def apply(self, config: ReacherConfig) -> ReacherConfig:
        half_xy, half_z, reaching_std, episode_len = _DIFFICULTY_PRESETS[self.difficulty]

        # Centered on the default target's centre (x=0.4, y=0.0, z=0.3).
        target = config.command_options.reaching_target
        target.x_range = (0.4 - half_xy, 0.4 + half_xy)
        target.y_range = (-half_xy, half_xy)
        target.z_range = (0.3 - half_z, 0.3 + half_z)

        config.reward_options.reaching_task.params["reaching_std"] = reaching_std
        config.env_options.episode_length_s = episode_len
        return config


@TASK_REGISTRY.register(modifiers=(ReacherDifficultyMod,))
class ReacherConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=9.0,
        iterations=10,
        ls_iterations=20,
        sim_dt=0.01,
        sim_substeps=2,
        decimation=1,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=en.robots.Yam(
            default_root_pos=(0.0, 0.0, 0.0),
            default_root_quat=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Observations, enable_corruption=True),
        critic=ObservationGroupOptions(**_Observations, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # Relative (incremental) joint-position control (reference_source=DELTA):
        # target = q_now + 0.05*a. The small incremental delta low-pass filters
        # observation noise, which an absolute target (q_default + 0.25*a) turns
        # into +/-0.25 rad target swings (a visible shake).
        joint_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name="joint*",
            scale=0.05,
            reference_source=ReferenceSource.DELTA,
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        reaching_target=UniformT3Command.configure(
            # The `medium` preset of ReacherDifficultyMod, spelled out: a bare ReacherConfig()
            # and TASK_REGISTRY.build("reacher") must describe the same task.
            x_range=(0.3, 0.5),
            y_range=(-0.1, 0.1),
            z_range=(0.25, 0.35),
            resampling_time_range=(2.0, 3.0),
        )
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching_task=RewardTerm.configure(
            func=reaching_reward,
            weight=1.0,
            params={
                "robot_name": "robot",
                "command_name": "reaching_target",
                "reaching_std": 0.2,
            },
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
        reaching_weight=StageRewardWeightCurriculum.configure(
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
        max_iterations=100,
        save_interval=100,
        experiment_name="reacher_yam",
        run_name="test_run",
    )
