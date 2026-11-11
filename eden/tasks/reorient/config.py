"""Reorient: SO(3) in-hand cube reorientation with the Wuji Hand 1.

Port of ``wuji-technology/wuji-mjlab``'s ``reorient`` task (Apache-2.0) from mjlab /
MuJoCo-Warp to Eden / Genesis. A palm-up, fixed-base 20-DOF dexterous hand holds a 54 mm
cube and must rotate it in place to a target orientation given in the wrist-tag frame,
hold it there, then chase the next target -- without dropping it.

    eden train --task reorient --hand right --num_envs 4096 --logger wandb
    eden inference --task reorient --hand right

Design notes worth reading before changing anything:

* **The goal is a state machine, not a timer** (``custom_terms/commands.py``). Reaching a
  goal opens a success *window* that pays an escalating hold reward before the next goal is
  drawn, which is what turns one episode into a stream of reorientations.
* **Drop detection is a shared computation** (``custom_terms/cage.py``). Upstream publishes
  it by writing an attribute on the env; here one metric term owns it and the reward,
  termination and critic observation all pull from it.
* **The goal is drawn, not just scored** (``commands.InHandReorientCommand.draw_vis``). A
  kinematic twin of the cube floats over the palm at the target orientation, wearing the
  same face texture, so a recorded rollout shows what the policy is chasing. It is moved
  only at eval, and it is invisible to the physics.
* **The policy group is hardware-reproducible.** Joint angles, joint tracking error, and
  the cube pose relative to the wrist tag -- all measurable on the real hand from encoders
  plus one camera. Everything privileged is critic-only.
* **The hand's actuator table is the manufacturer's** (see
  ``eden/options/robots/wuji_hand1.py``): kp ~0.2-0.7 Nm/rad against ~0.15-0.65 Nm
  actuators. This is a *soft* hand, and blanket gains break it.

Deliberate deviations from upstream, all for want of a Genesis equivalent:

* Per-env collision-geom scaling and per-env contact solver parameters are dropped. In
  their place the cube/fingertip contact is stiffened once at startup
  (``stiffen_contact``), which fixes the impedance rather than randomizing it.
* Per-env link-inertia randomization is dropped; per-link *mass* randomization is kept.
* Upstream's +/-3 mm cube COM offset is dropped. Genesis refuses inertial writes on a
  *primitive* morph, but the cube is an MJCF asset (:class:`DexCube`, for the face texture),
  so nothing blocks it any more; it stays off because turning it on would change the dynamics
  the shipped policy was trained against. The per-link mass randomization that is kept covers
  much of the same ground.
* ``RandomizedParams`` reports five DR channels to the critic where upstream reports six
  (no per-env geom size or friction to read back).
* Upstream's wrist-pitch randomization becomes a per-env gravity tilt: Genesis refuses a
  per-env root orientation on a fixed base that carries collision geometry, and for a hand
  held rigidly in place the two are the same perturbation.
* rsl-rl has no ``SoftplusGaussianDistribution``, so the actor uses a plain Gaussian at
  ``init_std=0.5`` with ``std_range=(0.2, 1.0)`` -- the hard std floor is the part of
  upstream's distribution that matters, and rsl-rl >= 5's Gaussian supports it directly.

One deviation that is *not* a Genesis limitation: the success state machine is **measured** in
``InHandReorientCommand.post_physics_refresh`` (phase 5) and only the goal *switch* is
published in ``_update_command`` (phase 17), following Eden's hook rule
(``docs/guides/terms.md``, "Which hook: grade the step, or prepare the next one"). Upstream
runs the whole machine below the reward and reaches back across the switch with three
``reward_*`` snapshots; they cover the hold counter but not the gate beside it, so the
highest-paying step of every goal cycle pays zero. Grading the step in the step it happened
removes the shadow buffers and the bug together, and is worth ``goal_switch_delay - 1`` = 19
extra units of hold reward per goal -- so the checkpoint rendered in ``docs/assets/reorient/``
was trained against the un-corrected shaping and reproducing its numbers needs a retrain.
"""

from __future__ import annotations

from typing import Annotated, Literal

import eden as en
from eden import EventMode, events, rewards, terminations
from eden.constants import ReferenceSource
from eden.managers.modifiers.observations.noise import GaussianNoise, UniformNoise
from eden.options import ObservationGroupOptions
from eden.options.entities import SceneOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.options.objects.mujoco_playground import DexCube
from eden.options.robots import WujiHand1Simplified_L, WujiHand1Simplified_R
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.tasks.reorient import custom_terms as ct
from eden.tasks.reorient.constants import (
    PALM_LINK,
    REORIENT_CUBE_SPAWN_POS,
    REORIENT_HAND_ROOT_POS,
    REORIENT_HAND_ROOT_QUAT,
    REORIENT_RESET_DOFS_POS_RANGE,
    cube_in_palm_for_side,
    dofs_pos_for_side,
    goal_in_palm_for_side,
)
from eden.terms import DoneTerm, EventTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
)

# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

_NUM_ENVS = 4096
_MAX_ITERATIONS = 5000
# 40 rather than Eden's default 24, matching upstream: the goal state machine needs ~25
# steps per goal cycle (5 hold + 20 window), so a shorter rollout would rarely contain a
# complete success and the advantage estimate for the hold reward would be badly biased.
_NUM_STEPS_PER_ENV = 40
_TOTAL_STEPS = _MAX_ITERATIONS * _NUM_STEPS_PER_ENV

# Absolute action scale [rad] on top of the default pose, straight from upstream. NOT
# ``scale_ratio``: that would derive the scale from each DOF's range (1.5-2.0 rad here),
# giving a target span 3-4x wider than upstream's.
_ACTION_SCALE = 0.5

# Friction. Genesis resolves a contact pair's coefficient as ``max(a, b)``, so the
# randomized hand side only takes effect against a surface authored *below* it -- hence the
# low cube friction paired with the hand's absolute 0.5-1.0 range. Upstream instead relies
# on MuJoCo geom ``priority`` with hand friction 0.7 and cube friction 0.3.
_CUBE_FRICTION = 0.2
_HAND_FRICTION_RANGE = (0.5, 1.0)


def _distal_link_names(side: Literal["right", "left"]) -> list[str]:
    """Explicit distal-phalanx link names for the given side.

    ``events.set_contact_sol_params`` matches link names *literally*, not by pattern, so
    this is one of the two places the ``--hand`` mod has to rewrite (the other is the action
    term's ``dofs_name``).
    """
    return [f"{side}_finger{finger}_link4" for finger in range(1, 6)]


def _hand(side: Literal["right", "left"]):
    """Fixed-base, gravity-compensated hand mounted palm-up at the nominal pose."""
    cls = WujiHand1Simplified_R if side == "right" else WujiHand1Simplified_L
    return cls(
        is_fixed_base=True,
        default_root_pos=REORIENT_HAND_ROOT_POS,
        default_root_quat=REORIENT_HAND_ROOT_QUAT,
        # The hand is bolted to a rig in the real setup, so its own weight is carried by
        # the mount, not by the finger actuators.
        material=RigidMaterialOptions(gravity_compensation=1.0),
        default_dofs_pos=dofs_pos_for_side(side),
    )


def _cube(**overrides) -> DexCube:
    """The 54 mm cube, shared by the real one and its goal marker.

    Size and mass are :class:`DexCube`'s (i.e. the asset's), so nothing here touches them.
    Only the declared morph spawn is set: ``reset_cube`` re-places the real cube from the
    live palm pose on every reset, per side, and the marker is placed by the command term's
    ``draw_vis``.
    """
    return DexCube(default_root_pos=REORIENT_CUBE_SPAWN_POS, **overrides)


def _reorient_scene(side: Literal["right", "left"] = "right") -> SceneOptions:
    return en.scenes.Empty(
        robot=_hand(side),
        cube=_cube(material=RigidMaterialOptions(friction=_CUBE_FRICTION)),
        # The goal marker: the same asset wearing the same face texture, kinematic and
        # collision-free, so it is invisible to the physics and to every term except the
        # command's ``draw_vis``. Deliberately *not* ``GHOST_DEFAULTS``, whose translucent
        # blue surface would paint over the texture that is the entire point of the marker
        # -- what identifies it as a target is that it floats above the hand, not its
        # shading, and a solid one stays readable at video scale against a dark background.
        # Left free-jointed like the real cube (``DexCube``'s default): Genesis' MJCF morph
        # has no `fixed` flag, and a kinematic entity is never integrated anyway -- it simply
        # stays wherever `draw_vis` last put it.
        cube_goal=_cube(collision=False, material=KinematicMaterialOptions()),
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

# Three-step history on every policy term. The hand is controlled at 20 Hz through an EMA
# filter, so a single frame cannot reveal whether a finger is loading up against the cube
# or sliding off it.
#
# ``flatten_history_dim=True`` on each term (the default, spelled out because this layout depends
# on it) folds the history into the feature axis, so a term contributes ``(num_envs, 3 * dim)``
# and the group stays a flat vector. Both things depend on it: rsl-rl's ActorCritic accepts 1-D
# observations only, and a group can only concatenate terms of equal rank -- so without it
# the history terms could not share a group with the unhistoried privileged ones.
_HISTORY = 3

# Policy: encoder-measurable joint state plus the tag-frame cube pose. The two cube channels
# carry `injection_prob`, which occasionally replaces them with noise so the policy keeps a
# proprioceptive fallback for when a real tag detection drops out.
_POLICY_OBS = dict(
    dofs_pos=ct.DofsPosNormalized.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.06, n_max=0.06),
        history_length=_HISTORY,
        flatten_history_dim=True,
    ),
    dofs_pos_error=ct.DofsPosTargetError.configure(
        entity_name="robot", action_term_name="joint_pos", history_length=_HISTORY, flatten_history_dim=True
    ),
    cube_pos_in_tag=ct.CubePosInTag.configure(
        command_name="reorient",
        object_entity_name="cube",
        injection_prob=0.02,
        noise=UniformNoise.configure(n_min=-0.008, n_max=0.008),
        history_length=_HISTORY,
        flatten_history_dim=True,
    ),
    cube_ori_error=ct.GoalRotErrorInTag.configure(
        command_name="reorient",
        object_entity_name="cube",
        injection_prob=0.02,
        noise=GaussianNoise.configure(std=0.05),
        history_length=_HISTORY,
        flatten_history_dim=True,
    ),
    last_action=ObsTerm.configure(func=en.observations.last_action, history_length=_HISTORY, flatten_history_dim=True),
)

# Critic: the same five channels the policy sees, but clean (no noise, no dropout
# injection), plus privileged state the real hand cannot measure.
_CRITIC_OBS = dict(
    dofs_pos=ct.DofsPosNormalized.configure(entity_name="robot", history_length=_HISTORY, flatten_history_dim=True),
    dofs_pos_error=ct.DofsPosTargetError.configure(
        entity_name="robot", action_term_name="joint_pos", history_length=_HISTORY, flatten_history_dim=True
    ),
    cube_pos_in_tag=ct.CubePosInTag.configure(
        command_name="reorient", object_entity_name="cube", history_length=_HISTORY, flatten_history_dim=True
    ),
    cube_ori_error=ct.GoalRotErrorInTag.configure(
        command_name="reorient", object_entity_name="cube", history_length=_HISTORY, flatten_history_dim=True
    ),
    last_action=ObsTerm.configure(func=en.observations.last_action, history_length=_HISTORY, flatten_history_dim=True),
    true_dofs_pos=en.observations.DofsPos.configure(
        entity_name="robot", offset_from_default=True, history_length=_HISTORY, flatten_history_dim=True
    ),
    true_dofs_vel=en.observations.DofsVel.configure(
        entity_name="robot", history_length=_HISTORY, flatten_history_dim=True
    ),
    fingertip_positions=ct.FingertipPositions.configure(
        entity_name="robot", scale=5.0, history_length=_HISTORY, flatten_history_dim=True
    ),
    # Instantaneous privileged state -- no history, because these are either already
    # rate-like (velocities) or slowly-varying per-episode constants (the DR draw).
    cube_lin_vel=ObsTerm.configure(func=ct.object_lin_vel, params={"entity_name": "cube"}),
    cube_ang_vel=ObsTerm.configure(func=ct.object_ang_vel, params={"entity_name": "cube"}),
    palm_rot_6d=ct.PalmRot6d.configure(entity_name="robot"),
    command_state=ObsTerm.configure(func=ct.command_state_progress, params={"command_name": "reorient"}),
    cage_counter=ObsTerm.configure(func=ct.cage_counter_progress, params={"metric_name": "cage_escape"}),
    perturbation_dir=ct.PerturbationDirection.configure(event_name="object_disturbance"),
    perturbation_vel=ct.PerturbationVelocity.configure(event_name="object_disturbance"),
    dr_params=ct.RandomizedParams.configure(entity_name="robot", object_entity_name="cube"),
)


# ---------------------------------------------------------------------------
# TaskMod
# ---------------------------------------------------------------------------


class HandSideMod(TaskMod):
    """Swap the chirality of the hand via ``--hand``.

    Upstream hardcodes the right hand (its contact-sensor patterns name
    ``right_palm_link`` directly). Every pattern in this port is side-agnostic
    (``.*_palm_link``, ``.*_finger[1-5]_link4``), so the swap only has to replace the robot
    options plus the four things that cannot be written side-agnostically: the action term's
    explicit DOF-name list, the contact-stiffening event's explicit link names, and the
    palm-frame positions of the cube and of its goal marker (both y offsets lean toward the
    thumb, so both flip).

    The grasp keyframe is mirrored inside ``_hand`` -- see
    :func:`~eden.tasks.reorient.constants.dofs_pos_for_side` for why the left hand needs
    four of its twenty joint angles negated.
    """

    def __init__(
        self,
        hand: Annotated[Literal["right", "left"], "Which Wuji Hand 1 to reorient with."] = "right",
    ):
        self.hand = hand

    def apply(self, config: ReorientConfig) -> ReorientConfig:
        robot = _hand(self.hand)
        config.scene_options.robot = robot
        config.action_options.joint_pos.dofs_name = list(robot.dofs_name)
        config.event_options.stiffen_contact.params["entity_links"]["robot"] = _distal_link_names(self.hand)
        config.event_options.reset_cube.pos_in_palm = cube_in_palm_for_side(self.hand)
        config.command_options.reorient.goal_marker_in_palm = goal_in_palm_for_side(self.hand)
        return config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@TASK_REGISTRY.register(modifiers=(HandSideMod(),))
class ReorientConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=_NUM_ENVS,
        # 100 Hz physics, 20 Hz control -- upstream's timestep=0.01 / decimation=5. The
        # sub-stepping keeps the soft fingertip contacts stable at that outer rate.
        sim_dt=0.01,
        sim_substeps=2,
        decimation=5,
        episode_length_s=50.0,
        iterations=20,
        ls_iterations=20,
        enable_multi_contact=True,
        # A cube in a five-fingered hand is nothing but contact.
        max_collision_pairs=128,
        # Finger-on-finger jamming is a real failure mode here and is penalized, so the
        # solver has to actually resolve it.
        enable_self_collision=True,
        # Both required by the per-env DR events below: Genesis stores DOF and link
        # inertial properties once for the whole batch otherwise.
        batch_dofs_info=True,
        batch_links_info=True,
    )

    scene_options: SceneOptions = _reorient_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_POLICY_OBS, enable_corruption=True),
        critic=ObservationGroupOptions(**_CRITIC_OBS, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        joint_pos=ct.JointPositionOffsetEma.configure(
            entity_name="robot",
            dofs_name=list(WujiHand1Simplified_R().dofs_name),
            scale=_ACTION_SCALE,
            reference_source=ReferenceSource.DEFAULT,
            ema_alpha=0.5,
            warmup_time_s=0.4,
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        reorient=ct.InHandReorientCommand.configure(
            hand_entity_name="robot",
            object_entity_name="cube",
            palm_link=PALM_LINK,
            success_threshold=0.2,
            success_hold_steps=5,
            goal_switch_delay=20,
            # Rendering only: ``RLEnvBase.step`` calls ``draw_vis`` when the env is not
            # training, so this costs a training run nothing and makes every recorded
            # rollout self-explanatory.
            debug_vis=True,
            goal_marker_entity_name="cube_goal",
            goal_marker_in_palm=goal_in_palm_for_side("right"),
            # Timed resampling is effectively disabled: the state machine owns goal
            # advancement, and a wall-clock resample would cut a success window short.
            resampling_time_range=(1.0e6, 1.0e6),
        ),
    )

    # Declared before the rewards and terminations that pull from them. The cage and contact
    # terms are shared computations rather than scores, so they sit out of success accounting.
    metric_options: MetricManagerOptions = MetricManagerOptions(
        cage_escape=ct.CageEscape.configure(
            hand_entity_name="robot",
            object_entity_name="cube",
            margin=0.01,
            z_upper_margin=0.03,
            max_outside_steps=10,
            drop_scale=4.0,
        ),
        hand_object_contact=ct.HandObjectContact.configure(hand_entity_name="robot", object_entity_name="cube"),
        self_collision=ct.SelfCollisionCount.configure(hand_entity_name="robot"),
        dofs_vel_rms=ct.DofsVelRms.configure(entity_name="robot"),
        dofs_acc_rms=ct.DofsAccRms.configure(entity_name="robot"),
        torque_saturation=ct.TorqueSaturationRatio.configure(entity_name="robot"),
        cube_height_above_palm=ct.CubeHeightAbovePalm.configure(hand_entity_name="robot", object_entity_name="cube"),
        goal_reach_count=MetricTerm.configure(
            func=ct.goal_reach_count, params={"command_name": "reorient"}, contributes_to_success=False
        ),
        goal_interval=MetricTerm.configure(
            func=ct.goal_interval, params={"command_name": "reorient"}, contributes_to_success=False
        ),
        cube_survival_steps=MetricTerm.configure(func=ct.cube_survival_steps, contributes_to_success=False),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        orientation_alignment=RewardTerm.configure(
            func=ct.orientation_alignment,
            weight=15.0,
            params={"command_name": "reorient", "object_entity_name": "cube"},
        ),
        hold_escalation=RewardTerm.configure(func=ct.hold_escalation, weight=11.4, params={"command_name": "reorient"}),
        palm_detach=RewardTerm.configure(
            func=ct.palm_detach, weight=0.5, params={"metric_name": "hand_object_contact"}
        ),
        hand_pose=RewardTerm.configure(func=ct.dofs_pos_deviation_l2, weight=-0.2, params={"entity_name": "robot"}),
        action_rate=rewards.ActionRateL2Smooth.configure(weight=-1.0),
        torque=ct.NormalizedTorque.configure(weight=-24.0, entity_name="robot"),
        tip_slide=ct.FingertipSlide.configure(
            weight=-0.3,
            entity_name="robot",
            object_entity_name="cube",
            metric_name="hand_object_contact",
        ),
        # Large and negative on purpose: dropping the cube ends the episode, so the penalty
        # has to outweigh everything the policy could still have earned by holding on.
        cage_escape=rewards.MetricReward.configure(metric_name="cage_escape", weight=-500.0),
        finger_collision=rewards.MetricReward.configure(metric_name="self_collision", weight=-1.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        cage_drop=DoneTerm.configure(func=ct.cage_drop, params={"metric_name": "cage_escape"}),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # --- reset ---
        # Stands in for upstream's wrist-pitch randomization -- see the term's docstring for
        # why a fixed-base hand has to express that as a gravity tilt. Pitch only, and
        # asymmetric: pitching the palm down toward the fingertips (the negative direction)
        # is the hard case, so it gets the wider range.
        randomize_gravity=ct.RandomizeGravityTilt.configure(pitch_range=(-0.4, 0.1)),
        reset_hand_dofs=ct.ResetDofsByPatternOffset.configure(
            entity_name="robot",
            dofs_pos_range=REORIENT_RESET_DOFS_POS_RANGE,
        ),
        # After the wrist and the fingers, so the cube is placed against the pose it will
        # actually start in.
        reset_cube=ct.ResetObjectInPalm.configure(
            hand_entity_name="robot",
            object_entity_name="cube",
            pos_in_palm=cube_in_palm_for_side("right"),
            pos_noise=0.01,
            priority=10,
        ),
        # --- interval ---
        object_disturbance=ct.ObjectVelocityDisturbance.configure(
            object_entity_name="cube",
            interval_range_s=(0.6, 1.8),
            min_speed=0.05,
            max_speed=0.15,
            warmup_time_s=3.0,
            total_steps=_TOTAL_STEPS,
            adaptive_curriculum_name="adaptive_episode",
        ),
        # --- startup domain randomization ---
        hand_friction=events.RandomizeFrictionRatio.configure(
            mode=EventMode.STARTUP,
            entity_name="robot",
            links_name=[PALM_LINK, ".*_finger[1-5]_link[1-4]"],
            friction_range=_HAND_FRICTION_RANGE,
        ),
        pd_gains=events.RandomizeKpKdGains.configure(
            mode=EventMode.STARTUP, entity_name="robot", kp_range=(0.75, 1.5), kd_range=(0.5, 2.0)
        ),
        dofs_armature=ct.RandomizeDofsArmature.configure(entity_name="robot", scale_range=(0.75, 1.3)),
        dofs_damping=ct.RandomizeDofsDamping.configure(entity_name="robot"),
        encoder_bias=events.RandomizeDofsPosOffset.configure(
            mode=EventMode.STARTUP,
            entity_name="robot",
            action_term_name="joint_pos",
            dofs_pos_range=(-0.01, 0.01),
        ),
        # The finger links only: the palm is fixed to the world, so the solver never reads its mass.
        hand_link_mass=events.RandomizeLinkMassScale.configure(
            mode=EventMode.STARTUP, entity_name="robot", links_name=[".*_finger.*"], mass_scale_range=(0.4, 1.5)
        ),
        cube_mass=events.RandomizeLinkMassScale.configure(
            mode=EventMode.STARTUP, entity_name="cube", links_name=[".*"], mass_scale_range=(0.4, 1.6)
        ),
        # Stands in for upstream's per-env contact-parameter randomization, which Genesis
        # cannot express: a fixed stiffening of the cube <-> fingertip pair so a 120 g cube
        # does not sink into the compliant default contact instead of being gripped.
        stiffen_contact=EventTerm.configure(
            func=events.set_contact_sol_params,
            mode=EventMode.STARTUP,
            params={
                # timeconst = 2 * substep_dt exactly: Genesis clamps a smaller value up to
                # that bound anyway (and warns), so anything stiffer here would be a
                # silently-ignored setting rather than a stiffer contact.
                "sol_params": (0.01, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0),
                "entity_links": {"cube": None, "robot": _distal_link_names("right")},
            },
        ),
    )

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        success_level=ct.ReorientSuccessCurriculum.configure(
            command_name="reorient", count_threshold=3, delta_per_loop=0.08
        ),
        # Declared after ``success_level`` and read by ``object_disturbance``: the
        # curriculum manager caches each term's value as it computes, and the disturbance
        # event fires later in the step, so it sees this step's value.
        adaptive_episode=ct.AdaptiveEpisodeCurriculum.configure(
            target_steps=800, inc_rate=0.1, dec_rate=0.2, min_scale=0.05
        ),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        # The actor reads only the noisy, encoder-measurable group; the critic reads the
        # clean one with the privileged state appended.
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        # Upstream's PPO, unchanged: a fixed low learning rate with many mini-batches,
        # which is what a long-horizon contact task with a sparse-ish success signal needs.
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=0.5,
            use_clipped_value_loss=False,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=4,
            num_mini_batches=32,
            learning_rate=1.0e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            # ``std_range`` floor = upstream's ``SoftplusGaussianDistribution(min_std=0.2)``.
            # Eden has no softplus distribution, but rsl-rl >= 5's Gaussian takes the same
            # hard clamp, which is the part that matters. Dropped on older rsl-rl (see the
            # field's docstring), so treat it as insurance rather than a dependency.
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=0.5, std_range=(0.2, 1.0)),
        ),
        # Wider than the actor: the critic sees 16 observation terms including the DR draw.
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[512, 512, 256, 128]),
        # NO runner-level ``clip_actions``, matching upstream -- the action term does its own
        # clamping. This is not a detail: the wrapper clips *before* ``env.step``
        # (``eden/envs/wrappers/rsl_rl_env.py``), so a runner clip is also applied to the raw
        # action the ``action_rate`` penalty reads. The penalty then stops growing once the
        # policy's mean leaves [-1, 1], the reward goes action-independent, and the commanded
        # target pins at a corner of the box -- a frozen hand. Measured over 300 iterations:
        # with the clip, orientation error rose 2.20 -> 2.33 rad and joint velocity RMS decayed
        # to 0.37; without it, error fell 2.16 -> 1.35 and velocity RMS held at 1.01.
        num_steps_per_env=_NUM_STEPS_PER_ENV,
        max_iterations=_MAX_ITERATIONS,
        save_interval=250,
        experiment_name="reorient",
        run_name="v1",
        wandb_project="reorient",
    )
