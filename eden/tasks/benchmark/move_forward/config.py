"""Move-Forward benchmark task configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from eden.utils.configs import (
    EdenRLConfig,
    EnvOptions,
    SceneOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
    CommandManagerOptions,
    ActionManagerOptions,
)
import eden as en
from eden.options import ObservationGroupOptions
from eden.options.learning.rl_games import (
    RlGamesNetworkOptions,
    RlGamesPpoRunnerOptions,
)
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskMod
from eden.terms import ObsTerm, RewardTerm, DoneTerm
from eden import actions, observations, rewards, terminations
from eden.managers.terms.commands.velocity import UniformVelocityCommand
from eden.constants import ReferenceSource


_Observations = ObservationGroupOptions(
    concatenate_terms=True,
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        scale=0.25,
        params={"entity_name": "robot", "frame": "body"},
    ),
    projected_gravity=observations.ProjectedGravity.configure(entity_name="robot", scale=1.0),
    commands=observations.ScaledCommands.configure(
        command_name="velocity_command",
        command_scale=[2.0, 2.0, 0.25],  # lin_vel x/y: 2.0, ang_vel z: 0.25
    ),
    dofs_pos=observations.DofsPos.configure(scale=1.0, entity_name="robot", offset_from_default=True),
    dofs_vel=observations.DofsVel.configure(scale=0.05, entity_name="robot"),
    actions=ObsTerm.configure(func=observations.last_action, scale=1.0),
)


# ---------------------------------------------------------------------------
# TaskMods
# ---------------------------------------------------------------------------


# Sim parameters paired with each PD controller flavor. Explicit needs a
# smaller dt and higher decimation because user-space torques are stiffer
# than Genesis's substep-level implicit damping.
_CONTROLLER_SIM_PARAMS: dict[str, dict[str, float | int]] = {
    "implicit": {"sim_dt": 0.02, "sim_substeps": 2, "decimation": 1},
    "explicit": {"sim_dt": 0.01, "sim_substeps": 1, "decimation": 2},
}


# Quadruped variants the robot mod can swap in. Action-scale dicts are
# robot-specific because the joint torque/velocity profiles differ; the
# mod pulls both from this table together.
_ROBOT_PRESETS: dict[str, tuple[type, "dict[str, float]"]] = {
    "go1": (en.robots.Go1, en.robots.GO1_ACTION_SCALE),
    "go2": (en.robots.Go2, en.robots.GO2_ACTION_SCALE),
}


class MoveForwardRobotMod(TaskMod):
    """Swap the quadruped robot for the move_forward task.

    Replaces three things in lockstep: ``scene_options.robot`` (the entity
    itself), the action controller's ``dofs_name`` and ``scale`` (each
    quadruped has a distinct joint set and action-scale preset), and the
    base-height reward target (which depends on the robot's standing
    height = ``default_root_pos[2] - 0.05``).

    Composes orthogonally with :class:`MoveForwardControllerMod`: whichever
    runs first sets dofs_name/scale, and the other reads-modified-state
    when it builds its own term. ``--robot`` defaults to ``go2`` to match
    :class:`MoveForwardConfig`'s declarative go2 field defaults.
    """

    def __init__(
        self,
        robot: Annotated[
            Literal["go1", "go2"],
            "Quadruped robot variant: go1 or go2 (default).",
        ] = "go2",
    ):
        self.robot = robot

    def apply(self, config: MoveForwardConfig) -> MoveForwardConfig:
        robot_cls, action_scale = _ROBOT_PRESETS[self.robot]
        robot_options = robot_cls()

        # 1. Swap the robot in the scene.
        config.scene_options.robot = robot_options

        # 2. Update the controller's joint set and per-joint action scale.
        controller = config.action_options.dofs_pos_controller
        controller.dofs_name = robot_options.dofs_name
        controller.scale = action_scale

        # 3. Update the base-height reward target (depends on robot stance height).
        config.reward_options.base_height.params["target_height"] = robot_options.default_root_pos[2] - 0.05
        return config


class MoveForwardControllerMod(TaskMod):
    """Swap the joint PD controller and the sim parameters that go with it.

    ``implicit`` (default) uses Genesis's built-in position control with
    implicit substep damping. ``explicit`` computes torques in user space
    via ``control_dofs_force``; that integration is stiffer and requires
    halved ``sim_dt`` plus ``decimation=2`` for stable training.

    The mod preserves the existing controller's ``entity_name`` /
    ``dofs_name`` / ``scale`` / ``reference_source`` — switch only the
    integration scheme, not the action shape — so it composes cleanly
    with whichever robot the config (or :class:`MoveForwardRobotMod`) set.
    """

    def __init__(
        self,
        controller: Annotated[
            Literal["implicit", "explicit"],
            "Joint PD controller flavor: implicit (Genesis built-in, default) "
            "or explicit (user-space torques, sim_dt halved + decimation=2).",
        ] = "implicit",
    ):
        self.controller = controller

    def apply(self, config: MoveForwardConfig) -> MoveForwardConfig:
        existing = config.action_options.dofs_pos_controller
        controller_cls = actions.ExplicitPDController if self.controller == "explicit" else actions.ImplicitPDController
        config.action_options.dofs_pos_controller = controller_cls.configure(
            entity_name=existing.entity_name,
            dofs_name=existing.dofs_name,
            scale=existing.scale,
            reference_source=existing.reference_source,
        )
        for field, value in _CONTROLLER_SIM_PARAMS[self.controller].items():
            setattr(config.env_options, field, value)
        return config


# rl_games PPO defaults for move_forward.
_RL_GAMES_DEFAULTS: dict[str, object] = {
    "max_epochs": 101,
    "save_frequency": 100,
    "horizon_length": 24,
    "mini_epochs": 5,
    # = horizon_length * num_envs / num_mini_batches = 24 * 4096 / 4
    "minibatch_size": 24576,
    "learning_rate": 0.001,
    "lr_schedule": "adaptive",
    "kl_threshold": 0.01,
    "gamma": 0.99,
    "tau": 0.95,
    "e_clip": 0.2,
    "clip_value": True,
    "entropy_coef": 0.01,
    "critic_coef": 1.0,
    "grad_norm": 1.0,
    "truncate_grads": True,
    "normalize_input": False,
    "normalize_value": False,
    "normalize_advantage": True,
    "value_bootstrap": True,
}


class MoveForwardRunnerMod(TaskMod):
    """Swap the RL training framework between rsl_rl (default) and rl_games.

    The default :class:`MoveForwardConfig` ships with an
    ``RslRlOnPolicyRunnerOptions`` and ``policy``+``critic`` observation
    groups, so ``--runner rsl_rl`` is a no-op. ``--runner rl_games``
    replaces ``runner_options`` with the rl_games schema and drops the
    ``critic`` observation group, which rl_games doesn't consume (an
    unused critic group on rsl_rl is harmless; on rl_games it can confuse
    the wrapper's obs-dict construction).

    The experiment-name suffix (``_rl_games``) is appended automatically
    to keep logs partitioned per-runner.
    """

    def __init__(
        self,
        runner: Annotated[
            Literal["rsl_rl", "rl_games"],
            "RL training framework: rsl_rl (default, on-policy PPO with explicit "
            "actor/critic) or rl_games (single network spec, policy-only obs).",
        ] = "rsl_rl",
        *,
        # Non-CLI: log namespace prefix, set at registration time.
        experiment_name: str = "move_forward_go2",
    ):
        self.runner = runner
        self.experiment_name = experiment_name

    def apply(self, config: MoveForwardConfig) -> MoveForwardConfig:
        if self.runner == "rl_games":
            config.runner_options = RlGamesPpoRunnerOptions(
                experiment_name=f"{self.experiment_name}_rl_games",
                network=RlGamesNetworkOptions(separate=True, units=[512, 256, 128]),
                **_RL_GAMES_DEFAULTS,
            )
            # rl_games consumes only the policy obs group.
            if hasattr(config.observation_options, "critic"):
                delattr(config.observation_options, "critic")
        # rsl_rl branch: leave the class-level defaults alone — they're
        # already the rsl_rl shape. (Re-applying after rl_games would
        # require restoring the critic group; not handled here since
        # build() runs each mod once on a fresh config.)
        return config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_GO2 = en.robots.Go2()


@TASK_REGISTRY.register(modifiers=(MoveForwardRobotMod, MoveForwardControllerMod, MoveForwardRunnerMod))
class MoveForwardConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=0.02,
        sim_substeps=2,
        decimation=1,
        max_collision_pairs=30,
        use_gjk_collision=False,
        batch_dofs_info=False,
        enable_multi_contact=False,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=_Observations,
        critic=_Observations,
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        roll_limit=DoneTerm.configure(
            func=terminations.base_orientation_roll_greater_than,
            params={"entity_name": "robot", "threshold": 10.0},
        ),
        pitch_limit=DoneTerm.configure(
            func=terminations.base_orientation_pitch_greater_than,
            params={"entity_name": "robot", "threshold": 10.0},
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        velocity_command=UniformVelocityCommand.configure(
            lin_vel_x_range=(0.5, 0.5),
            lin_vel_y_range=(0.0, 0.0),
            ang_vel_z_range=(0.0, 0.0),
            resampling_time_range=(4.0, 4.0),
            entity_name="robot",
        )
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        experiment_name="move_forward_go2",
        max_iterations=101,
        num_steps_per_env=24,
        save_interval=100,
        algorithm=RslRlPpoAlgorithmOptions(
            class_name="PPO",
            clip_param=0.2,
            desired_kl=0.01,
            entropy_coef=0.01,
            gamma=0.99,
            lam=0.95,
            learning_rate=0.001,
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

    # Base (go2) defaults declared as fields; the go1 variant is applied by MoveForwardRobotMod,
    # which mutates the built config. Declaring them as fields (rather than building in __init__)
    # keeps --config overrides from being clobbered on reconstruction. See reacher for the pattern.
    scene_options: SceneOptions = en.scenes.Empty(robot=_GO2)
    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_GO2.dofs_name,
            scale=en.robots.GO2_ACTION_SCALE,
            reference_source=ReferenceSource.DEFAULT,
        )
    )
    reward_options: RewardManagerOptions = RewardManagerOptions(
        tracking_lin_vel=rewards.TrackingLinVel.configure(
            weight=1.0,
            command_name="velocity_command",
            entity_name="robot",
            sigma=0.25,
        ),
        tracking_ang_vel=rewards.TrackingAngVel.configure(
            weight=0.2,
            command_name="velocity_command",
            entity_name="robot",
            sigma=0.25,
        ),
        lin_vel_z=RewardTerm.configure(func=rewards.lin_vel_z, weight=-1.0, params={"entity_name": "robot"}),
        base_height=RewardTerm.configure(
            func=rewards.constant_base_height,
            weight=-50.0,
            params={
                "entity_name": "robot",
                "target_height": _GO2.default_root_pos[2] - 0.05,
            },
        ),
        action_rate=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-0.005,
        ),
        similar_to_default=RewardTerm.configure(
            func=rewards.similar_to_default,
            weight=-0.1,
            params={"entity_name": "robot"},
        ),
    )
