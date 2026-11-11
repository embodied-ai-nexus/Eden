"""HUSKY: humanoid skateboarding on a physically simulated board (G1 23-DOF).

Training port of `TeleHuman/humanoid_skateboarding
<https://github.com/TeleHuman/humanoid_skateboarding>`_ (HUSKY, RSS 2026,
CC BY-NC 4.0) — the task behind the frozen policy shipped in
``eden/extensions/policies/husky``. The G1 rides a free-floating
skateboard whose trucks steer when the deck tilts; a 6 s phase clock cycles
push (left leg strokes the ground) → transition → steer (both feet on the
deck) → transition, and the command is ``[board speed, relative heading]``.

Everything scene-side reuses the validated frozen-policy port: the skater G1
(:class:`~eden.extensions.policies.husky.policy.HuskyG1` — training MJCF,
Raibert-heuristic gains equal to the released checkpoint's ONNX metadata) and
the skateboard asset (:class:`~eden.options.objects.husky.HuskySkateboard`).
The MDP is declared here: the phase/contact state machine lives on the
``skate`` command term and rewards come in the source's four phase-gated
groups, with the custom terms under ``eden.tasks.husky.custom_terms``. The
source additionally augments PPO with an adversarial motion prior over the
push-phase joint trajectories; the config selects that trainer —
:class:`~eden.learning.wrappers.rsl_rl.amp_ppo.options.RslRlAmpPpoAlgorithmOptions`
over the retargeted human pushing clips — as its ``algorithm``, so ``eden train
--task husky`` is the full recipe. Swapping in plain PPO
(``algorithm=RslRlPpoAlgorithmOptions(...)``, not ``class_name`` alone) trains the same MDP
without the style reward;
:class:`~eden.tasks.husky.custom_terms.observations.SkatePushFlag` warns a
training run when that is what it is about to do, since the plain MDP is a
harder, sparser problem whose numbers are not comparable to a HUSKY result.

The actor observation reproduces the source layout exactly: per-term 5-step
history blocks, each flattened oldest-first (Eden's per-term ``history_length``
produces the same term-major layout as mjlab's group-level history), 395 dims.

Deliberate deviations from the source (all judged non-load-bearing):

- The wheels' rolling-friction randomization is dropped: Genesis's rolling
  friction is opt-in per scene and the validated frozen-policy port runs with
  it disabled.
- The ground plane gets sliding friction 0.01 so foot/wheel coefficients win
  Genesis's ``max``-combine, reproducing MuJoCo's geom-priority resolution
  (feet/wheels have ``priority=1`` in the source scene).
- Self-collision is penalized as a boolean flag (the source sensor's single
  ``found`` slot), and observation-corruption noise is dropped from the critic
  group (the source declared it but disabled corruption).

Train / play::

    eden train --task husky --logger wandb
    eden inference --task husky --checkpoint logs/husky/<run>/model_<it>.pt --rec
    uv run python examples/projects/locomotion/husky_skate/main.py play --model ... --record --terrain rough
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import model_validator

import eden as en
from eden import events, observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.extensions.policies.husky.constants import (
    ACTION_SCALE,
    ANG_VEL_SCALE,
    COMMAND_SCALE,
    DOFS_VEL_SCALE,
    HEADING_SCALE,
    HISTORY_LENGTH,
    POLICY_JOINT_NAMES,
)
from eden.extensions.policies.husky.policy import HuskyG1
from eden.learning.wrappers.rsl_rl.amp_ppo import RslRlAmpPpoAlgorithmOptions
from eden.managers.modifiers.observations import UniformNoise
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.options import ObservationGroupOptions, RewardGroupOptions
from eden.options.entities import PlaneOptions, SceneOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.objects.husky import HuskySkateboard
from eden.tasks import TASK_REGISTRY
from eden.tasks.husky.custom_terms import observations as husky_obs
from eden.tasks.husky.custom_terms import rewards as husky_rewards
from eden.tasks.husky.custom_terms import terminations as husky_terminations
from eden.tasks.husky.custom_terms.commands import SkateCommand
from eden.terms import DoneTerm, EventTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
)

if TYPE_CHECKING:
    from eden.options.managers.actions import ActionTermOptions

_HISTORY = {"history_length": HISTORY_LENGTH, "flatten_history_dim": True}


def actor_observation_terms(*, entity_name: str = "robot", noise: bool, history: bool) -> dict:
    """The 8 actor observation terms, in the source's order — the frozen HUSKY actor's input contract.

    Shared by this task's ``policy`` (noise and history) and ``critic`` (neither) groups and by
    :meth:`~eden.extensions.policies.husky.policy.HuskyPolicy.observation_options` (history, no
    noise), so the released checkpoint and the policies trained here read one layout. With
    ``history`` each term stacks its own 5-step window, flattened oldest-first, so the group is
    term-major: 8 blocks of ``5 x dim``, 395 dims in all. The command and phase terms read the
    ``skate`` command term (:class:`~eden.tasks.husky.custom_terms.commands.SkateCommand` here,
    :class:`~eden.tasks.husky.custom_terms.commands.ConstantSkateCommand` for the frozen policy).
    """

    def corruption(n: float):
        return UniformNoise.configure(n_min=-n, n_max=n) if noise else None

    window = _HISTORY if history else {}
    entity = {"entity_name": entity_name}
    return {
        "command": observations.ScaledCommands.configure(
            command_name="skate", command_scale=list(COMMAND_SCALE), **window
        ),
        "heading": ObsTerm.configure(func=husky_obs.robot_heading, params=entity, scale=HEADING_SCALE, **window),
        "base_ang_vel": ObsTerm.configure(
            func=observations.base_ang_vel,
            params={**entity, "frame": "body"},
            noise=corruption(0.2),
            scale=ANG_VEL_SCALE,
            **window,
        ),
        "projected_gravity": ObsTerm.configure(
            func=husky_obs.pelvis_projected_gravity, params=entity, noise=corruption(0.05), **window
        ),
        "joint_pos": observations.DofsPos.configure(
            **entity,
            offset_from_default=True,
            noise=corruption(0.01),
            **window,
        ),
        "joint_vel": observations.DofsVel.configure(**entity, noise=corruption(1.5), scale=DOFS_VEL_SCALE, **window),
        "actions": ObsTerm.configure(func=observations.last_action, **window),
        "phase": ObsTerm.configure(func=husky_obs.skate_phase, **window),
    }


def joint_position_action(entity_name: str = "robot") -> ActionTermOptions:
    """The actor's action term: a normalized action, scaled per joint and offset onto the default pose.

    Shared with :meth:`~eden.extensions.policies.husky.policy.HuskyPolicy.action_options`, so the
    frozen actor's raw output reaches the robot, and its ``actions`` observation, exactly as in training.
    """
    return ImplicitPDController.configure(
        entity_name=entity_name,
        dofs_name=list(POLICY_JOINT_NAMES),
        # 0.25 * effort_limit / kp per joint — the source's action scale.
        scale=dict(zip(POLICY_JOINT_NAMES, ACTION_SCALE)),
        reference_source=ReferenceSource.DEFAULT,
    )


def _foot_force(surface: str, foot: int) -> ObsTerm:
    return ObsTerm.configure(func=husky_obs.foot_contact_force, params={"surface": surface, "foot": foot})


#: The retargeted human pushing clips AmpPPO's discriminator trains against.
_AMP_MOTION_FILE = "objects/skateboard/husky_amp_push_motions.npz"
#: What the husky_skate example's ``train`` subcommand wrote as ``algorithm.class_name`` before AmpPPO moved into
#: eden/learning: the example's own module, importable only from that example's directory.
_LEGACY_EXAMPLE_AMP_CLASS_NAME = "amp_training.ppo:AmpPPO"


@TASK_REGISTRY.register()
class HuskyConfig(EdenRLConfig):
    """G1 humanoid skateboarding on flat ground (HUSKY training port)."""

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_example_algorithm(cls, data: Any) -> Any:
        """Read the AMP algorithm of a config.json the husky_skate example saved before AmpPPO moved into eden/learning.

        That example selected AmpPPO by setting only ``class_name`` on the stock PPO options, so its runs' config.json
        holds an ``RslRlPpoAlgorithmOptions`` naming ``amp_training.ppo:AmpPPO`` and no ``amp_*`` field (its AmpPPO
        defaulted every one, the motion file included). That is :class:`RslRlAmpPpoAlgorithmOptions` with the same PPO
        fields and its defaults, so it is read as that, and those runs' checkpoints replay through ``eden inference``.
        A saved plain-PPO algorithm (``class_name="PPO"``) is left alone: it is what that run trained with.
        """
        runner = data.get("runner_options") if isinstance(data, dict) else None
        if isinstance(runner, dict):
            algorithm = runner.get("algorithm")
        else:
            algorithm = getattr(runner, "algorithm", None)
        if isinstance(algorithm, RslRlPpoAlgorithmOptions):
            fields = {name: getattr(algorithm, name) for name in RslRlPpoAlgorithmOptions.model_fields}
        elif isinstance(algorithm, dict):
            fields = {name: value for name, value in algorithm.items() if name in RslRlPpoAlgorithmOptions.model_fields}
        else:
            return data
        if fields.get("class_name") != _LEGACY_EXAMPLE_AMP_CLASS_NAME:
            return data
        fields.pop("class_name")
        migrated = RslRlAmpPpoAlgorithmOptions(**fields, amp_motion_file=_AMP_MOTION_FILE)
        if isinstance(runner, dict):
            return {**data, "runner_options": {**runner, "algorithm": migrated}}
        return {**data, "runner_options": runner.model_copy(update={"algorithm": migrated})}

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=0.005,
        decimation=4,  # 50 Hz control
        iterations=10,
        ls_iterations=20,
        max_collision_pairs=128,
        batch_dofs_info=False,
        batch_links_info=True,  # RandomizeComShift writes per-env link COMs
        enable_self_collision=True,
    )

    scene_options: SceneOptions = SceneOptions(
        robot=HuskyG1(),
        skateboard=HuskySkateboard(),
        # Low-friction ground so the feet/wheel coefficients win Genesis's
        # max-combine (MuJoCo resolves these contacts by geom priority instead).
        plane=PlaneOptions(default_geoms_friction={"*": 0.01}),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**actor_observation_terms(noise=True, history=True), enable_corruption=True),
        critic=ObservationGroupOptions(
            **actor_observation_terms(noise=False, history=False),
            base_lin_vel=ObsTerm.configure(
                func=observations.base_lin_vel, params={"entity_name": "robot", "frame": "body"}
            ),
            heading_error=ObsTerm.configure(func=husky_obs.skate_heading_error, scale=1.0 / math.pi),
            skate_pose_local=ObsTerm.configure(func=husky_obs.skate_pose_local),
            skate_vel_local=ObsTerm.configure(func=husky_obs.skate_vel_local),
            skate_ang_vel_local=ObsTerm.configure(func=husky_obs.skate_ang_vel_local),
            skateboard_roll=ObsTerm.configure(func=husky_obs.skateboard_roll),
            trans_target_pos=ObsTerm.configure(func=husky_obs.transition_target_pos),
            trans_target_quat=ObsTerm.configure(func=husky_obs.transition_target_quat),
            l_foot_forces_ground=_foot_force("ground", 0),
            r_foot_forces_ground=_foot_force("ground", 1),
            l_foot_forces_board=_foot_force("board", 0),
            r_foot_forces_board=_foot_force("board", 1),
            contact_phase=ObsTerm.configure(func=husky_obs.skate_contact_phase),
            enable_corruption=False,
        ),
        # Consumed by AmpPPO (via runner obs passthrough), not by the actor/critic —
        # plain PPO would ignore the group, and `SkatePushFlag.build` warns a training run
        # when that is what it is about to do. Layout is position-sliced:
        # [joint_pos 5-frame window (oldest first), push-phase flag].
        amp=ObservationGroupOptions(
            joint_pos=observations.DofsPos.configure(entity_name="robot", **_HISTORY),
            push=husky_obs.SkatePushFlag.configure(),
            enable_corruption=False,
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        skate=SkateCommand.configure(
            resampling_time_range=(20.0, 20.0),
            lin_vel_x_range=(0.0, 1.5),
            heading_range=(-math.pi / 4, math.pi / 4),
        )
    )

    action_options: ActionManagerOptions = ActionManagerOptions(dofs_pos_controller=joint_position_action())

    event_options: EventManagerOptions = EventManagerOptions(
        push_robot=events.PushByVelocity.configure(
            mode="interval",
            interval_range_s=(5.0, 10.0),
            entity_name="robot",
            velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        ),
        reset_robot_joints=EventTerm.configure(
            func=en.events.reset_dofs_by_offset,
            mode="reset",
            params={"entity_name": "robot", "dofs_pos_range": (-0.01, 0.01), "dofs_vel_range": (0.0, 0.0)},
        ),
        base_com=events.RandomizeComShift.configure(
            mode="startup",
            entity_name="robot",
            links_name=["torso_link"],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.025, 0.025),
            com_shift_z_range=(-0.03, 0.03),
        ),
        skate_com=events.RandomizeComShift.configure(
            mode="startup",
            entity_name="skateboard",
            links_name=["skateboard_deck"],
            com_shift_x_range=(-0.02, 0.02),
            com_shift_y_range=(-0.02, 0.02),
            com_shift_z_range=(-0.01, 0.01),
        ),
        # Authored friction is 1.0 on every randomized geom, so the source's
        # "scale" ranges are absolute coefficients here. Feet re-randomize after
        # the whole-robot pass (priority orders startup events).
        robot_friction=events.RandomizeFrictionRatio.configure(
            mode="startup", entity_name="robot", links_name=["*"], friction_range=(0.3, 1.6), priority=0
        ),
        foot_friction=events.RandomizeFrictionRatio.configure(
            mode="startup",
            entity_name="robot",
            links_name=["left_ankle_roll_link", "right_ankle_roll_link"],
            friction_range=(0.3, 1.8),
            priority=10,
        ),
        board_friction=events.RandomizeFrictionRatio.configure(
            mode="startup", entity_name="skateboard", links_name=["board_tilt_body"], friction_range=(0.8, 2.0)
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        push=RewardGroupOptions(
            push_skateboard_lin_vel=RewardTerm.configure(
                func=husky_rewards.push_skateboard_lin_vel, weight=3.0, params={"std": math.sqrt(0.25)}
            ),
            push_yaw_align=RewardTerm.configure(
                func=husky_rewards.push_yaw_align, weight=1.0, params={"std": math.sqrt(0.25)}
            ),
            push_air_time=RewardTerm.configure(
                func=husky_rewards.push_left_foot_air_time,
                weight=3.0,
                params={"threshold_min": 0.1, "threshold_max": 0.5, "command_threshold": 0.1},
            ),
            push_contact_ground_parallel=RewardTerm.configure(
                func=husky_rewards.push_contact_ground_parallel, weight=0.5
            ),
        ),
        steer=RewardGroupOptions(
            steer_contact_num=RewardTerm.configure(func=husky_rewards.steer_contact_num, weight=3.0),
            steer_joint_pos=RewardTerm.configure(
                func=husky_rewards.steer_joint_pos, weight=1.5, params={"std": math.sqrt(0.2)}
            ),
            steer_feet_dis=RewardTerm.configure(
                func=husky_rewards.steer_feet_dis, weight=1.0, params={"std": math.sqrt(0.1)}
            ),
            steer_track_heading=RewardTerm.configure(
                func=husky_rewards.steer_track_heading, weight=5.0, params={"std": math.sqrt(0.02)}
            ),
            steer_tilt_guide=RewardTerm.configure(
                func=husky_rewards.steer_tilt_guide, weight=4.0, params={"std": math.sqrt(0.02)}
            ),
        ),
        transition=RewardGroupOptions(
            transition_body_pos_tracking=RewardTerm.configure(
                func=husky_rewards.transition_body_pos_tracking, weight=10.0, params={"std": math.sqrt(0.05)}
            ),
            transition_body_rot_tracking=RewardTerm.configure(
                func=husky_rewards.transition_body_rot_tracking, weight=10.0, params={"std": math.sqrt(0.10)}
            ),
            transition_penalty_contact=RewardTerm.configure(func=husky_rewards.transition_penalty_contact, weight=-0.5),
        ),
        regularization=RewardGroupOptions(
            reg_wheel_contact_number=RewardTerm.configure(func=husky_rewards.reg_wheel_contact_number, weight=0.5),
            dof_pos_limits=RewardTerm.configure(
                func=rewards.dofs_pos_limits, weight=-5.0, params={"entity_name": "robot"}
            ),
            # = the source's action_rate_l2 (-0.1) + action_acc_l2 (-0.1): the smooth
            # variant returns first + second difference, so one weight covers both.
            action_rate_l2=rewards.ActionRateL2Smooth.configure(weight=-0.1),
            joint_vel_l2=RewardTerm.configure(func=husky_rewards.dofs_vel_l2, weight=-1e-3),
            # Source: -2.5e-7 on joint acceleration [rad/s^2]; Eden's term is the
            # velocity difference over one control step, so scale by 1/dt^2.
            joint_acc_l2=rewards.DofsAccL2.configure(weight=-2.5e-7 / 0.02**2, entity_name="robot"),
            joint_torques_l2=RewardTerm.configure(func=husky_rewards.dofs_torques_l2, weight=-1e-6),
            self_collisions=RewardTerm.configure(func=husky_rewards.self_collision_flag, weight=-10.0),
            board_flat=RewardTerm.configure(func=husky_rewards.board_flat, weight=3.0, params={"std": math.sqrt(0.05)}),
            stand_still=RewardTerm.configure(
                func=husky_rewards.stand_still, weight=1.0, params={"std": math.sqrt(0.1)}
            ),
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        fell_over=DoneTerm.configure(
            func=husky_terminations.pelvis_fell_over, params={"limit_angle": math.radians(70.0)}
        ),
        feet_off_board=DoneTerm.configure(func=husky_terminations.feet_off_board),
        illegal_contact=DoneTerm.configure(func=husky_terminations.skate_illegal_contact),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        experiment_name="husky",
        max_iterations=50000,
        num_steps_per_env=24,
        save_interval=500,
        wandb_project="husky",
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        # The source's AMP-augmented PPO: style reward from a discriminator over the
        # push-phase joint windows of the `amp` group (see the module docstring).
        algorithm=RslRlAmpPpoAlgorithmOptions(
            amp_motion_file=_AMP_MOTION_FILE,
            entropy_coef=0.005,
            learning_rate=1.0e-3,
            schedule="adaptive",
            desired_kl=0.01,
            gamma=0.99,
            lam=0.95,
            num_learning_epochs=5,
            num_mini_batches=4,
        ),
        actor=RslRlActorOptions(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
        ),
        critic=RslRlCriticOptions(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True),
    )
