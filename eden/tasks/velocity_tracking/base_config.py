"""Shared velocity-tracking configuration for bipedal humanoids.

Robot-agnostic except the slots :class:`VelocityTrackingRobotMod` fills at build time (scene entity,
controller joints and action scale, torso and feet links, foot-clearance target). They ship empty,
the mod finds them by term *type* and fills only unset ones, and
:meth:`VelocityTrackingBaseConfig.check_bindings` names any still empty. :class:`VelocityTrackingObsMod`
picks what the policy observes. Provides the posture std maps, the shared observation and termination
dicts, both mods, and the base config the ``flat`` and ``terrain`` variants inherit. Design
notes and provenance: ``docs/guides/velocity-tracking.md`` in the source repository (it is not in the
installed package).

Build with::

    TASK_REGISTRY.build("velocity_tracking_flat")               # G1 29DoF (default)
    TASK_REGISTRY.build("velocity_tracking_flat", robot="oli")  # LimX Oli 31DoF
    TASK_REGISTRY.build("velocity_tracking_flat", obs_preset="privileged")
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from eden.utils.configs import (
    EdenRLConfig,
    EnvOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
    CommandManagerOptions,
    ActionManagerOptions,
    EventManagerOptions,
    CurriculumManagerOptions,
)
import eden as en
from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.managers.terms.actions.joint_actions import is_joint_position_controller
from eden.options import ObservationGroupOptions
from eden.options.managers.base import ManagerOptions
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks.registry import TaskMod
from eden.terms import ObsTerm, RewardTerm, DoneTerm, EventTerm
from eden import actions, observations, rewards, terminations, events
from eden.managers.terms import UniformVelocityCommand, VelocityCommandCurriculum
from eden.managers.modifiers.observations import UniformNoise
from eden.constants import ReferenceSource

# ---------------------------------------------------------------------------
# Posture standard deviation maps (per-joint tolerance), regex over joint names.
# Every DOF must match: ``resolve_per_dof_std`` raises on one the map misses.
# ---------------------------------------------------------------------------

STD_STANDING = {".*": 0.05}

STD_WALKING = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*head.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
}

STD_RUNNING = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*head.*": 0.1,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
}

# ---------------------------------------------------------------------------
# Observation terms: the policy gets only what the robot measures (encoders, IMU).
# ``base_lin_vel`` has no sensor on a humanoid, so it is critic-only.
# ---------------------------------------------------------------------------

_POLICY_OBSERVATIONS = dict(
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
        params={"entity_name": "robot", "frame": "body"},
    ),
    projected_gravity=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    dofs_pos=observations.DofsPos.configure(
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01), entity_name="robot", offset_from_default=True
    ),
    dofs_vel=observations.DofsVel.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-1.5, n_max=1.5),
        # Bounds one noisy encoder; the noise stays in rad/s (raw -> noise -> clip -> scale).
        scale=0.05,
    ),
    actions=ObsTerm.configure(func=observations.last_action),
    commands=ObsTerm.configure(
        func=observations.generated_commands,
        params={"command_name": "velocity_command"},
    ),
)

#: Critic group: the policy's terms plus privileged ``base_lin_vel`` (the critic runs only in sim).
_CRITIC_OBSERVATIONS = dict(
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        noise=UniformNoise.configure(n_min=-0.5, n_max=0.5),
        params={"entity_name": "robot", "frame": "body"},
    ),
    **_POLICY_OBSERVATIONS,
)

# ---------------------------------------------------------------------------
# Shared termination terms (dict spread into TerminationManagerOptions)
# ---------------------------------------------------------------------------

_TERMINATIONS = dict(
    time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
    fell_over=DoneTerm.configure(
        func=terminations.fell_over,
        params={"entity_name": "robot", "limit_angle": math.radians(70.0)},
    ),
)


# ---------------------------------------------------------------------------
# Robot binding -- TaskMod
# ---------------------------------------------------------------------------

#: The scene entity every robot-bound term acts on; the mod places the robot under this key.
_ROBOT = "robot"

#: The robot-specific fields of this task's reward and event terms, by term class, and what fills
#: each: the trunk link (``torso``, or ``torso_links`` for a link list), the robot's feet links, or the
#: preset's swing-foot apex target. Keyed by class, not term name, so a variant can rename, add or drop terms.
_ROBOT_BOUND_FIELDS: dict[type, dict[str, str]] = {
    rewards.Upright: {"link_name": "torso"},
    rewards.BodyAngVelXYPenalty: {"link_name": "torso"},
    rewards.FeetClearance: {"feet_link_names": "feet", "target_height": "foot_clearance_height"},
    rewards.FeetSlip: {"feet_link_names": "feet"},
    events.RandomizeFrictionRatio: {"links_name": "feet"},
    events.RandomizeComShift: {"links_name": "torso_links"},
}


def _is_unset(value: Any) -> bool:
    """Whether a robot-bound field still holds its empty sentinel: ``""``, ``[]``, or ``0.0``."""
    return not value


def _is_unset_scale(term: Any) -> bool:
    """Whether a controller's action scale is unset: ``scale`` at its ``None`` default and no ``scale_ratio``."""
    return term.scale is None and term.scale_ratio is None


def _term_items(options: ManagerOptions | None) -> Iterator[tuple[str, Any]]:
    """``(path, term options)`` for every term in ``options``, descending into term groups; ``None`` yields nothing."""
    if options is None:
        return
    for key in options.term_keys():
        entry = getattr(options, key)
        if isinstance(entry, ManagerOptions):
            yield from ((f"{key}.{sub}", term) for sub, term in _term_items(entry))
        else:
            yield key, entry


def _robot_controllers(config: VelocityTrackingBaseConfig) -> list[tuple[str, Any]]:
    """``(key, term options)`` of every joint position controller on the robot, in declaration order."""
    return [
        (key, term)
        for key, term in _term_items(config.action_options)
        if term.entity_name == _ROBOT and is_joint_position_controller(ACTION_TERM_REGISTRY.get(term.name))
    ]


def _robot_dof_owners(config: VelocityTrackingBaseConfig) -> list[tuple[str, Any]]:
    """``(key, term options)`` of every action term that will command DOFs of the robot.

    A joint position controller counts even with an empty ``dofs_name`` (the mod is about to fill it);
    any other action term counts once it names DOFs of its own.
    """
    return [
        (key, term)
        for key, term in _term_items(config.action_options)
        if term.entity_name == _ROBOT
        and (term.dofs_name or is_joint_position_controller(ACTION_TERM_REGISTRY.get(term.name)))
    ]


def _unset_bound_fields(config: VelocityTrackingBaseConfig) -> Iterator[tuple[str, Any, str, str]]:
    """``(path, term options, field, source)`` for each robot-bound reward/event field still at its sentinel.

    ``source`` is the :data:`_ROBOT_BOUND_FIELDS` value naming what fills the field. The mod fills what this
    yields, and :meth:`VelocityTrackingBaseConfig.check_bindings` reports it.
    """
    for group, registry in (("reward_options", REWARD_TERM_REGISTRY), ("event_options", EVENT_TERM_REGISTRY)):
        for path, term in _term_items(getattr(config, group)):
            if getattr(term, "entity_name", None) != _ROBOT:
                continue
            term_cls = registry.get(term.name)
            if not isinstance(term_cls, type):
                continue  # a function term resolves to a partial over its wrapper, never a bound class
            fields = next((f for cls, f in _ROBOT_BOUND_FIELDS.items() if issubclass(term_cls, cls)), {})
            for field, source in fields.items():
                if _is_unset(getattr(term, field)):
                    yield f"{group}.{path}", term, field, source


class VelocityTrackingRobotMod(TaskMod):
    """Bind a specific bipedal humanoid to the velocity-tracking task.

    Sets the scene entity, every joint position controller's joints and action scale, the torso and
    feet links, the foot-clearance target, and prefixes the experiment name with the robot key. Terms
    are found by **type** on the ``robot`` entity, never by key, and only fields still at their unset
    sentinel are filled, so a variant may rename, add, drop or split terms. A controller with empty
    ``dofs_name`` gets every joint (then it must be the robot's only DOF-owning action term); one with
    no ``scale``/``scale_ratio`` gets the robot's action-scale dict with ``dict_keys_scope="entity"``.
    Raises if the config has no joint position controller on the robot. Adding a robot is one entry
    in :attr:`_PRESETS`.
    """

    _PRESETS: dict[str, dict] = {
        "g1": {
            "robot_cls": en.robots.G1Dofs29,
            "action_scale": en.robots.G1_ACTION_SCALE,
            "torso_link": "torso_link",
            # Swing-foot apex target [m]; mjlab's G1 value.
            "foot_clearance_height": 0.10,
        },
        "oli": {
            "robot_cls": en.robots.Oli,
            "action_scale": en.robots.OLI_ACTION_SCALE,
            # No ``torso_link`` in Oli's URDF: the last waist link is the trunk.
            "torso_link": "waist_pitch_link",
            # G1's 0.10 scaled by standing height (1.65 m / 1.32 m).
            "foot_clearance_height": 0.125,
        },
    }

    def __init__(
        self,
        robot: Annotated[
            Literal["g1", "oli"],
            "Humanoid variant: g1 (Unitree G1 29DoF, default) or oli (LimX Oli 31DoF).",
        ] = "g1",
    ):
        self.robot = robot

    def apply(self, config: VelocityTrackingBaseConfig) -> VelocityTrackingBaseConfig:
        preset = self._PRESETS[self.robot]
        robot_options = preset["robot_cls"]()
        torso = preset["torso_link"]
        # Every mutable value below is copied: assignment on an options model is not validated,
        # so pydantic does not copy, and a shared list would alias terms to each other.

        # 1. Scene + action controllers -- the robot's identity in the sim.
        config.scene_options.robot = robot_options
        self._bind_controllers(config, robot_options.dofs_name, preset["action_scale"])

        # 2. Rewards and events bound to robot-specific links (option fields on the class terms).
        values = {
            "torso": torso,
            "torso_links": [torso],
            "feet": list(robot_options.feet_links_name),
            "foot_clearance_height": preset["foot_clearance_height"],
        }
        for _path, term, field, source in _unset_bound_fields(config):
            setattr(term, field, copy.deepcopy(values[source]))

        # 3. Log namespace per-robot: "<robot>_<variant>".
        config.runner_options.experiment_name = f"{self.robot}_{config.runner_options.experiment_name}"
        return config

    def _bind_controllers(
        self, config: VelocityTrackingBaseConfig, dofs_name: list[str], action_scale: dict[str, float]
    ) -> None:
        """Fill the unset joint set and action scale of every joint position controller on the robot.

        Parameters
        ----------
        config: VelocityTrackingBaseConfig
            The config being bound; its controllers are edited in place.
        dofs_name: list[str]
            The robot's joints, in the robot's order: what a lone controller with no joints gets.
        action_scale: dict[str, float]
            The robot's whole-body action-scale dict: what each controller with no scale gets.
        """
        controllers = _robot_controllers(config)
        if not controllers:
            available = {key: term.name for key, term in _term_items(config.action_options)}
            raise ValueError(
                f"{type(self).__name__} binds the robot's joint set and action scale onto every joint position "
                f"controller (see is_joint_position_controller) with entity_name='{_ROBOT}', and action_options "
                f"has none. Its terms: {available or 'none'}."
            )
        unnamed = [key for key, term in controllers if not term.dofs_name]
        owners = _robot_dof_owners(config)
        if unnamed and len(owners) > 1:
            raise ValueError(
                f"{type(self).__name__}: action terms {unnamed} leave dofs_name empty, so the mod would give each "
                f"of them every joint of the robot, and action_options has {len(owners)} terms that command DOFs "
                f"of '{_ROBOT}' ({[key for key, _ in owners]}). Only a lone DOF-owning term may leave its "
                f"joints to the mod. Two action terms on one DOF is last-writer-wins, which ActionManager "
                f"rejects when the env is built.\n"
                f"Fix: set `dofs_name` explicitly on every listed term, so each names the joints it "
                f"controls and the sets are disjoint. See docs/guides/task-mods.md."
            )
        for _key, term in controllers:
            if not term.dofs_name:
                term.dofs_name = list(dofs_name)
            if _is_unset_scale(term):
                term.scale = dict(action_scale)
                # The dict covers the whole robot; the controller may own only part of it.
                term.dict_keys_scope = "entity"


# ---------------------------------------------------------------------------
# Policy observation recipe -- TaskMod
# ---------------------------------------------------------------------------


class VelocityTrackingObsMod(TaskMod):
    """Choose what the *policy* observation carries; the critic is the same under both presets.

    Presets
    -------
    ``proprioceptive``
        Sensor-only, 5 frames of history (96/480 on G1, 102/510 on Oli). The default and a no-op.
    ``privileged``
        Adds ``base_lin_vel`` (leading), single frame, ``dofs_vel`` unscaled (99 on G1, 105 on Oli).
        Needs ``eden deploy --estimate_lin_vel`` on hardware.

    CLI usage::

        eden train --task velocity_tracking_flat --robot oli
        eden train --task velocity_tracking_flat --robot oli --obs_preset privileged
    """

    def __init__(
        self,
        obs_preset: Annotated[
            Literal["proprioceptive", "privileged"],
            "Policy observation recipe. 'proprioceptive' is sensor-only with 5 frames of "
            "history (default, no-op vs the unmodified config); 'privileged' gives the policy "
            "the simulator's base_lin_vel in a single frame, which then needs a deploy-time "
            "linear-velocity estimator.",
        ] = "proprioceptive",
    ):
        self.obs_preset = obs_preset

    def apply(self, config: VelocityTrackingBaseConfig) -> VelocityTrackingBaseConfig:
        if self.obs_preset == "proprioceptive":
            # No-op: VelocityTrackingBaseConfig already holds this observation.
            return config
        if self.obs_preset == "privileged":
            policy = config.observation_options.policy
            # base_lin_vel leads; a new extra field would land last, so state the order. The terrain
            # variant's height_scan keeps its place at the end.
            order = ["base_lin_vel", *policy.term_keys()]
            # terms_order first: an assigned term is checked against the order on its own line.
            # No clone() — the group deep-copies what it is assigned, so the critic's term is not aliased.
            policy.terms_order = order
            policy.base_lin_vel = _CRITIC_OBSERVATIONS["base_lin_vel"]
            policy.history_length = None
            for group in (policy, config.observation_options.critic):
                group.dofs_vel.scale = 1.0
            return config
        raise ValueError(f"Unknown obs_preset {self.obs_preset!r}; expected one of 'proprioceptive', 'privileged'.")


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------


class VelocityTrackingBaseConfig(EdenRLConfig):
    """Robot-parametric velocity-tracking base configuration.

    Sets up all shared options as class-level defaults. Subclasses must set
    ``scene_options`` and may override other options; the robot binding is
    applied by :class:`VelocityTrackingRobotMod` at build time.
    """

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,  # Control at 50 Hz (0.005 * 4 = 0.02s per step)
        max_collision_pairs=64,
        use_gjk_collision=False,
        batch_dofs_info=False,
        batch_links_info=True,  # RandomizeComShift writes per-env link COMs
        enable_multi_contact=False,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # 5 frames, so the policy can average out a one-step sensor glitch.
        policy=ObservationGroupOptions(**_POLICY_OBSERVATIONS, enable_corruption=True, history_length=5),
        # Sim-only critic: privileged base_lin_vel, no history needed.
        critic=ObservationGroupOptions(**_CRITIC_OBSERVATIONS, enable_corruption=False),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        velocity_command=UniformVelocityCommand.configure(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            heading_command=True,
            heading_control_stiffness=0.5,
            lin_vel_x_range=(-1.0, 1.0),
            lin_vel_y_range=(-1.0, 1.0),
            ang_vel_z_range=(-0.5, 0.5),
            heading_range=(-math.pi, math.pi),
        )
    )

    event_options: EventManagerOptions = EventManagerOptions(
        reset_base=EventTerm.configure(
            func=en.events.reset_base_state_uniform,
            mode="reset",
            params={
                "entity_name": "robot",
                "pos_x_range": (-0.5, 0.5),
                "pos_y_range": (-0.5, 0.5),
                "pos_z_range": (0.01, 0.05),
                "yaw_range": (-3.14, 3.14),
            },
        ),
        reset_dofs=EventTerm.configure(
            func=en.events.reset_dofs_by_offset,
            mode="reset",
            params={
                "entity_name": "robot",
                "dofs_pos_range": (0.0, 0.0),
                "dofs_vel_range": (0.0, 0.0),
            },
        ),
        # base_joint_name is left at its default: the free base joint is found by type,
        # so this term is already robot-agnostic.
        push_robot=events.PushByVelocity.configure(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            entity_name="robot",
            velocity_range={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        ),
        # links_name is rewired per-robot by VelocityTrackingRobotMod.
        randomize_friction=events.RandomizeFrictionRatio.configure(
            mode="startup",
            entity_name="robot",
            links_name=[],
            friction_range=(0.3, 1.2),
        ),
        # links_name is rewired per-robot by VelocityTrackingRobotMod.
        randomize_com=events.RandomizeComShift.configure(
            mode="startup",
            entity_name="robot",
            links_name=[],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.025, 0.025),
            com_shift_z_range=(-0.03, 0.03),
        ),
        # action_term_name is left at None: every joint position controller on the robot, so a
        # variant that splits the controller keeps the bias on all of its joints.
        encoder_bias=events.RandomizeDofsPosOffset.configure(
            mode="startup",
            entity_name="robot",
            dofs_pos_range=(-0.015, 0.015),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        track_linear_velocity=RewardTerm.configure(
            func=rewards.track_lin_vel_with_z_penalty,
            weight=2.0,
            params={
                "command_name": "velocity_command",
                "entity_name": "robot",
                "std": math.sqrt(0.25),
            },
        ),
        track_angular_velocity=RewardTerm.configure(
            func=rewards.track_ang_vel_with_xy_penalty,
            weight=2.0,
            params={
                "command_name": "velocity_command",
                "entity_name": "robot",
                "std": math.sqrt(0.5),
            },
        ),
        # link_name is rewired per-robot by VelocityTrackingRobotMod.
        upright=rewards.Upright.configure(
            weight=1.0,
            entity_name="robot",
            link_name="",
            std=math.sqrt(0.2),
        ),
        pose=rewards.VariablePosture.configure(
            weight=1.0,
            entity_name="robot",
            command_name="velocity_command",
            std_standing=STD_STANDING,
            std_walking=STD_WALKING,
            std_running=STD_RUNNING,
            walking_threshold=0.05,
            running_threshold=1.5,
        ),
        # link_name is rewired per-robot by VelocityTrackingRobotMod.
        body_ang_vel=rewards.BodyAngVelXYPenalty.configure(
            weight=-0.05,
            entity_name="robot",
            link_name="",
        ),
        joint_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-1.0,
            params={"entity_name": "robot"},
        ),
        # Smoothness: 1st and 2nd action differences plus joint acceleration.
        action_rate_l2=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-0.1,
        ),
        action_rate_l2_smooth=rewards.ActionRateL2Smooth.configure(
            weight=-0.05,
        ),
        # A velocity difference, not an acceleration: legged_gym's -2.5e-7 scaled by 1/dt^2.
        dofs_acc_l2=rewards.DofsAccL2.configure(
            weight=-2.5e-7 / 0.02**2,
            entity_name="robot",
        ),
        # feet_link_names / target_height are rewired per-robot by VelocityTrackingRobotMod.
        foot_clearance=rewards.FeetClearance.configure(
            weight=-2.0,
            entity_name="robot",
            feet_link_names=[],
            target_height=0.0,
            command_name="velocity_command",
            command_threshold=0.05,
        ),
        # feet_link_names is rewired per-robot by VelocityTrackingRobotMod.
        foot_slip=rewards.FeetSlip.configure(
            weight=-0.1,
            entity_name="robot",
            feet_link_names=[],
            command_name="velocity_command",
            command_threshold=0.05,
        ),
        self_collisions=RewardTerm.configure(
            func=rewards.self_collision_cost,
            weight=-1.0,
            params={"entity_name": "robot"},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(**_TERMINATIONS)

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        command_vel=VelocityCommandCurriculum.configure(
            command_name="velocity_command",
            velocity_stages=[
                {
                    "step": 0,
                    "lin_vel_x": (-1.0, 1.0),
                    "ang_vel_z": (-0.5, 0.5),
                },
                {
                    "step": 5000 * 24,
                    "lin_vel_x": (-1.5, 2.0),
                    "ang_vel_z": (-0.7, 0.7),
                },
                {
                    "step": 10000 * 24,
                    "lin_vel_x": (-2.0, 3.0),
                },
            ],
        ),
    )

    # dofs_name / scale left unset: VelocityTrackingRobotMod fills them (found by type, not key).
    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            reference_source=ReferenceSource.DEFAULT,
        )
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        # VelocityTrackingRobotMod prefixes the robot key: "g1_velocity_tracking".
        experiment_name="velocity_tracking",
        max_iterations=30000,
        num_steps_per_env=24,
        save_interval=100,
        algorithm=RslRlPpoAlgorithmOptions(
            class_name="PPO",
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
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            obs_normalization=True,
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            obs_normalization=True,
        ),
    )

    def check_bindings(self) -> None:
        """Raise if a field :class:`VelocityTrackingRobotMod` fills is still unset, naming each one and the mod.

        Checks the same terms the mod binds, found the same way (by type, on the robot), for the same
        sentinels: an empty link field or joint set, a zero foot-clearance target, a controller with
        neither ``scale`` nor ``scale_ratio`` set.

        Raises
        ------
        ValueError
            Listing every unset field, if there is one.
        """
        unset = [f"{path}.{field} = {getattr(term, field)!r}" for path, term, field, _ in _unset_bound_fields(self)]
        for key, term in _robot_controllers(self):
            if not term.dofs_name:
                unset.append(f"action_options.{key}.dofs_name = {term.dofs_name!r}")
            if _is_unset_scale(term):
                unset.append(f"action_options.{key}.scale = {term.scale!r}")
        if unset:
            lines = "\n".join(f"  {line}" for line in unset)
            raise ValueError(
                f"{type(self).__name__} has fields that {VelocityTrackingRobotMod.__name__} binds per robot, "
                f"still unset:\n{lines}\n"
                f"TASK_REGISTRY.build(<task>) applies the mod, and a subclass registered with "
                f"@TASK_REGISTRY.register() inherits it unless it passes modifiers=(). A config made any other "
                f"way (TASK_REGISTRY.get(<task>)(), the class itself, a subclass of VelocityTrackingBaseConfig) "
                f"has to set these fields itself, as does a term added after the mod ran."
            )
