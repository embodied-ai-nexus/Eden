"""Extreme Parkour teacher task configuration (Unitree Go2).

Port of the IsaacLab/legged_gym "Extreme Parkour with Legged Robots"
(Cheng et al. 2023) teacher stage: a Go2 follows terrain goal waypoints across
a 10x20 grid of parkour courses (gaps, hurdles, flat lanes, steps, stepping
stones) with a walked-distance terrain-level curriculum.

Observation layout — ONE ``policy`` group, 753 dims, fixed slot table::

    prop[0:53] | scan[53:185] | priv_explicit[185:194] | priv_latent[194:223] | hist[223:753]

The ``priv_latent`` chunk reuses Eden's privileged DR readback terms
(``AddedMass`` 1 | ``ComOffset`` 3 | ``FootFriction`` 1 | ``GainScale`` 24);
these are shift/ratio-vs-nominal encodings (an affine transform of IsaacLab's
absolute values). ``priv_explicit`` is the slot the RMA estimator overwrites at
act time during distillation/deployment.

The runner options select the RMA teacher stack from
:mod:`eden.learning.wrappers.rsl_rl.rma` — an actor with scan / privileged /
history encoders and a concurrently trained base-velocity estimator, PPO with
privileged-latent regularization, periodic DAgger updates of the history encoder,
and a left-right mirror loss (:class:`~eden.tasks.extreme_parkour.symmetry.ParkourMirror`)
— which reads the ``policy`` group by the slot table above. Stock PPO does not learn
this task (a plain MLP over the raw 753-d obs — no scan encoder, history dumped in
flat — plateaus at negative reward). Train / replay::

    eden train --task extreme_parkour_teacher --logger wandb
    eden inference --task extreme_parkour_teacher --checkpoint logs/extreme_parkour_teacher/<run>/model_<it>.pt
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import eden as en
from eden import events, observations, terminations
from eden.constants import ReferenceSource
from eden.managers.terms.curricula.success_guided import SuccessGuidedSampling
from eden.managers.terms.rewards.common import DofsAccL2, UndesiredContacts
from eden.managers.terms.rewards.locomotion import (
    FeetAirTime,
    FeetContactBalancePenalty,
    StumblePenalty,
    WeightedPosePenalty,
)
from eden.learning.wrappers.rsl_rl.rma import RslRlRmaActorOptions, RslRlRmaPpoAlgorithmOptions
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import RslRlCriticOptions, RslRlOnPolicyRunnerOptions
from eden.options.entities import TerrainOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand
from eden.tasks.extreme_parkour.custom_terms.curricula import TerrainLevelCurriculum
from eden.tasks.extreme_parkour.custom_terms.events import ResetParkourRootState
from eden.tasks.extreme_parkour.custom_terms.metrics import parkour_goal_progress
from eden.tasks.extreme_parkour.custom_terms.observations import (
    ParkourPrivExplicit,
    ParkourProprio,
    ParkourProprioHistory,
    ParkourScanDots,
)
from eden.tasks.extreme_parkour.custom_terms.rewards import (
    DeltaTorquesL2,
    FeetEdgePenalty,
    FlatOrientationOnFlat,
    LinVelZTerrainWeighted,
    action_rate_norm,
    ang_vel_xy_l2,
    torques_l2,
    tracking_goal_vel,
    tracking_yaw,
)
from eden.tasks.extreme_parkour.custom_terms.terminations import all_goals_reached, base_height_below
from eden.tasks.registry import TaskMod
from eden.terms import DoneTerm, EventTerm, MetricTerm, RewardTerm
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
    SceneOptions,
    TerminationManagerOptions,
)
from eden.utils.terrains import (
    ParkourGapTerrain,
    ParkourHurdleTerrain,
    ParkourStepTerrain,
    ParkourTerrain,
)

# ---------------------------------------------------------------------------
# Terrain grid
# ---------------------------------------------------------------------------

_ROWS, _COLS = 10, 20
_SUBTERRAIN_SIZE = (16.0, 4.0)
_HORIZONTAL_SCALE = 0.08
_VERTICAL_SCALE = 0.005

FEET_LINK_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def _course_generators(d: float, *, corridor_y_range: tuple[float, float] = (-0.4, 0.4)) -> dict:
    """The five teacher course generators at difficulty ``d``.

    ``num_gaps`` / ``num_stones`` = 6 so every cell yields exactly 8 goals (6 obstacles
    + start + end); a homogeneous goal count keeps the padded goal grid tight.

    ``corridor_y_range`` is the lateral goal-offset range of the corridor courses (gap,
    hurdle, flat, step); the student phase narrows it to ``(-0.1, 0.1)`` while the
    stepping-stones course keeps its own difficulty-scaled lateral range in both phases.
    """
    common = dict(pad_height=0.0, apply_roughness=True)
    gap = ParkourGapTerrain.configure(
        num_gaps=6,
        gap_size=0.1 + 0.7 * d,
        gap_depth=(0.2, 1.0),
        x_range=(0.8, 1.5),
        y_range=corridor_y_range,
        half_valid_width=(0.6, 1.2),
        difficulty=d,
        **common,
    )
    hurdle = ParkourHurdleTerrain.configure(
        num_stones=6,
        stone_len=0.1 + 0.3 * d,
        hurdle_height_range=(0.1 + 0.1 * d, 0.15 + 0.25 * d),
        x_range=(1.2, 2.2),
        y_range=corridor_y_range,
        half_valid_width=(0.4, 0.8),
        difficulty=d,
        **common,
    )
    flat = ParkourHurdleTerrain.configure(
        flat=True,  # obstacles suppressed; the corridor carving + goals remain
        num_stones=6,
        stone_len=0.1 + 0.3 * d,
        hurdle_height_range=(0.1 + 0.1 * d, 0.15 + 0.15 * d),
        x_range=(1.2, 2.2),
        y_range=corridor_y_range,
        half_valid_width=(0.4, 0.8),
        difficulty=d,
        **common,
    )
    step = ParkourStepTerrain.configure(
        num_stones=6,
        step_height=0.1 + 0.35 * d,
        x_range=(0.3, 1.5),
        y_range=corridor_y_range,
        half_valid_width=(0.5, 1.0),
        difficulty=d,
        **common,
    )
    stones = ParkourTerrain.configure(
        num_stones=6,
        x_range=(-0.1, 0.1 + 0.3 * d),
        y_range=(0.2, 0.3 + 0.1 * d),
        stone_len=(0.9 - 0.3 * d, 1.0 - 0.2 * d),
        incline_height=0.25 * d,
        last_incline_height=0.25 * d + 0.1 - 0.1 * d,
        stone_width=1.0,
        last_stone_len=1.6,
        pit_depth=(0.2, 1.0),
        difficulty=d,
        **common,
    )
    return {"gap": gap, "hurdle": hurdle, "flat": flat, "step": step, "stones": stones}


def _make_parkour_grid() -> list[list]:
    """10 difficulty rows x 20 columns; column pattern [gap, hurdle, flat, step, stones] * 4."""
    grid: list[list] = []
    for i in range(_ROWS):
        c = _course_generators(i / (_ROWS - 1))
        grid.append([c["gap"], c["hurdle"], c["flat"], c["step"], c["stones"]] * (_COLS // 5))
    return grid


# ---------------------------------------------------------------------------
# Robot (Go2, Extreme Parkour gains/pose)
# ---------------------------------------------------------------------------

_GO2 = en.robots.Go2(
    default_root_pos=(0.0, 0.0, 0.4),
    default_dofs_pos={
        "FL_hip_joint": 0.1,
        "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    },
    default_dofs_kp={"*_hip_joint": 40.0, "*_thigh_joint": 40.0, "*_calf_joint": 40.0},
    default_dofs_kd={"*_hip_joint": 1.0, "*_thigh_joint": 1.0, "*_calf_joint": 1.0},
    default_dofs_force_limits={"*_hip_joint": 35.0, "*_thigh_joint": 40.0, "*_calf_joint": 40.0},
)


def _make_terrain_scene(terrain_generators: list[list], horizontal_scale: float) -> SceneOptions:
    """A parkour terrain grid + the Go2, centered so subterrain (0, 0) sits at the world origin corner."""
    rows, cols = len(terrain_generators), len(terrain_generators[0])
    return SceneOptions(
        terrain=TerrainOptions(
            subterrain_size=_SUBTERRAIN_SIZE,
            horizontal_scale=horizontal_scale,
            vertical_scale=_VERTICAL_SCALE,
            terrain_generators=terrain_generators,
            randomize=True,  # REQUIRED: per-cell RNG so repeats of each course differ
            default_root_pos=(
                -rows * _SUBTERRAIN_SIZE[0] / 2.0,
                -cols * _SUBTERRAIN_SIZE[1] / 2.0,
                0.0,
            ),
        ),
        robot=_GO2,
    )


def _make_env_options(num_envs: int) -> EnvOptions:
    """Shared physics/env settings of the teacher and student phases (they differ only in scale)."""
    return EnvOptions(
        num_envs=num_envs,
        episode_length_s=20.0,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,  # control at 50 Hz
        max_collision_pairs=128,
        use_gjk_collision=False,
        batch_dofs_info=False,
        batch_links_info=True,  # RandomizeMassShift / RandomizeComShift write per-env link info
        enable_multi_contact=False,
        enable_self_collision=True,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CLIP = (-100.0, 100.0)


def _policy_observation_group() -> ObservationGroupOptions:
    """The single 753-d ``policy`` group; term order == network slot layout (module docstring).

    A factory rather than a shared default instance: the student config declares the exact
    same group, and sharing one mutable options object across two task classes would alias
    their configs. Corruption off — the source adds no observation noise in either phase.
    """
    return ObservationGroupOptions(
        enable_corruption=False,
        proprio=ParkourProprio.configure(
            entity_name="robot",
            command_name="parkour_command",
            feet_link_names=FEET_LINK_NAMES,
            clip=_CLIP,
        ),
        scan_dots=ParkourScanDots.configure(
            entity_name="robot",
            terrain_entity_name="terrain",
            clip=_CLIP,
        ),
        priv_explicit=ParkourPrivExplicit.configure(entity_name="robot", clip=_CLIP),
        added_mass=observations.AddedMass.configure(entity_name="robot", links_name=["base"], clip=_CLIP),
        com_offset=observations.ComOffset.configure(entity_name="robot", links_name=["base"], clip=_CLIP),
        foot_friction=observations.FootFriction.configure(entity_name="robot", links_name=FEET_LINK_NAMES, clip=_CLIP),
        gain_scale=observations.GainScale.configure(entity_name="robot", clip=_CLIP),
        proprio_history=ParkourProprioHistory.configure(clip=_CLIP),
    )


def rma_actor_options() -> RslRlRmaActorOptions:
    """The RMA actor over :func:`_policy_observation_group`'s slot table (source network sizes).

    Shared with the student config, whose distillation rebuilds this exact actor to load the
    teacher checkpoint into — a factory for the same aliasing reason as the observation group.
    """
    return RslRlRmaActorOptions(
        num_prop=53,
        num_scan=132,
        num_priv_explicit=9,
        num_priv_latent=29,
        num_hist=10,
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        hidden_dims=[512, 256, 128],
        activation="elu",
        estimator_hidden_dims=[128, 64],
    )


def _make_event_terms() -> dict:
    """The teacher's event terms (order matters — see the ``event_options`` comment below).

    A factory for the same aliasing reason as :func:`_policy_observation_group`; the
    student adds camera-mount randomization on top of these.
    """
    return {
        "reset_parkour_root_state": ResetParkourRootState.configure(
            mode="reset",
            entity_name="robot",
            command_name="parkour_command",
        ),
        "reset_dofs": EventTerm.configure(
            func=events.reset_dofs_by_scale,
            mode="reset",
            params={"entity_name": "robot", "dofs_pos_scale_range": (0.95, 1.05)},
        ),
        "randomize_friction": events.RandomizeFrictionRatio.configure(
            mode="startup",
            entity_name="robot",
            links_name=FEET_LINK_NAMES,
            friction_range=(0.6, 2.0),
        ),
        "randomize_base_mass": events.RandomizeMassShift.configure(
            mode="startup",
            entity_name="robot",
            links_name=["base"],
            mass_shift_range=(-1.0, 3.0),
        ),
        "randomize_com": events.RandomizeComShift.configure(
            mode="startup",
            entity_name="robot",
            links_name=["base"],
            com_shift_x_range=(-0.02, 0.02),
            com_shift_y_range=(-0.02, 0.02),
            com_shift_z_range=(-0.02, 0.02),
        ),
        "push_robot": events.PushSetPlanarVelocity.configure(
            mode="interval",
            interval_range_s=(8.0, 8.0),
            entity_name="robot",
            velocity_xy_range=(-0.5, 0.5),
        ),
    }


class ExtremeParkourTerrainSamplerMod(TaskMod):
    """Who decides each episode's terrain cell: ``--terrain_sampler level|uniform|sgs``.

    - ``level`` (default, the shipped task): ``TerrainLevelCurriculum`` moves each env up or down a
      difficulty row by its walked distance; the env keeps its course column.
    - ``uniform``: every cell of the grid, uniformly at every reset.
    - ``sgs``: Success Guided Sampling over every cell (``SuccessGuidedSampling``, the paper's locomotion
      settings), graded by ``all_goals_reached``. Its outcome history is logged under
      ``Curriculum/terrain_sampler/*``. The term's ``state_dict`` carries it, but Eden's rsl_rl and
      rl_games runners do not save curriculum-term state, so under them a resumed run starts
      sampling uniformly again.

    ``uniform`` and ``sgs`` drop the ``terrain_level`` curriculum and repoint
    ``reset_parkour_root_state.sampler``; the configuration set is every cell of the grid either way.
    Compare the modes on ``Commands/parkour_command/terrain_level`` and ``Metrics/Success``, which all three log.
    The reward is not the same across them: the ``feet_edge`` penalty applies only above its
    ``min_terrain_level``, and ``uniform`` / ``sgs`` place envs on those rows from the first episode,
    where ``level`` reaches them only by promotion.
    """

    def __init__(
        self,
        terrain_sampler: Annotated[
            Literal["level", "uniform", "sgs"],
            "Terrain-cell choice at reset: walked-distance level curriculum, uniform, or Success Guided Sampling.",
        ] = "level",
    ):
        self.terrain_sampler = terrain_sampler

    def apply(self, config: ExtremeParkourTeacherConfig) -> ExtremeParkourTeacherConfig:
        if self.terrain_sampler == "level":
            return config
        config.curriculum_options.disable_term("terrain_level")
        spawn = config.event_options.reset_parkour_root_state
        if self.terrain_sampler == "uniform":
            spawn.sampler = None
            return config
        config.curriculum_options.terrain_sampler = SuccessGuidedSampling.configure(success_term="all_goals_reached")
        spawn.sampler = "terrain_sampler"
        return config


@TASK_REGISTRY.register(modifiers=(ExtremeParkourTerrainSamplerMod,))
class ExtremeParkourTeacherConfig(EdenRLConfig):
    """Extreme Parkour teacher (see module docstring for the 753-dim obs slot table)."""

    env_options: EnvOptions = _make_env_options(num_envs=4096)

    scene_options: SceneOptions = _make_terrain_scene(_make_parkour_grid(), _HORIZONTAL_SCALE)

    # ONE group; term order == network slot layout (753 total).
    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=_policy_observation_group(),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        parkour_command=ParkourCommand.configure(
            entity_name="robot",
            terrain_entity_name="terrain",
            resampling_time_range=(6.0, 6.0),
            lin_vel_x_range=(0.3, 0.8),
            heading_range=(-1.6, 1.6),
            heading_control_stiffness=0.8,
            lin_vel_clip=0.2,
            ang_vel_clip=0.4,
            next_goal_threshold=0.2,
            reach_goal_delay=0.1,
            num_future_goal_obs=2,
            spawn_offset_x=7.0,
        ),
    )

    # Terrain-level progression (walked distance vs commanded distance). The spawn event
    # (reset_parkour_root_state) samples its cells at reset time — before the pose write —
    # and logs surface as Curriculum/terrain_level/mean_terrain_level.
    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        terrain_level=TerrainLevelCurriculum.configure(
            command_name="parkour_command",
            max_init_terrain_level=2,
            level_up_frac=0.8,
            level_down_frac=0.4,
        ),
    )

    # The success curve under every --terrain_sampler: episodes that reach every goal are Metrics/Success. The level
    # each finished episode ran on is the command's own Commands/parkour_command/terrain_level.
    metric_options: MetricManagerOptions = MetricManagerOptions(
        parkour_goal_progress=MetricTerm.configure(
            func=parkour_goal_progress, params={"command_name": "parkour_command"}, success_threshold=1.0
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=en.actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_GO2.dofs_name,
            scale=0.25,
            reference_source=ReferenceSource.DEFAULT,
        )
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        tracking_goal_vel=RewardTerm.configure(
            func=tracking_goal_vel,
            weight=1.5,
            params={"command_name": "parkour_command", "entity_name": "robot"},
        ),
        tracking_yaw=RewardTerm.configure(
            func=tracking_yaw,
            weight=0.5,
            params={"command_name": "parkour_command", "entity_name": "robot"},
        ),
        lin_vel_z=LinVelZTerrainWeighted.configure(weight=-1.0, entity_name="robot", command_name="parkour_command"),
        ang_vel_xy=RewardTerm.configure(func=ang_vel_xy_l2, weight=-0.05, params={"entity_name": "robot"}),
        # Uprightness penalty: stronger than the source's -1.0 and not flat-only. The source's
        # flat-only -1.0 leaves a persistent ~17 deg forward lean, driven by the unconstrained
        # obstacle cells (~80% of the grid); a reduced penalty there (nonflat_scale) removes it
        # while still allowing climbing pitch.
        orientation=FlatOrientationOnFlat.configure(
            weight=-2.0, entity_name="robot", command_name="parkour_command", nonflat_scale=0.3
        ),
        # Eden's DofsAccL2 uses velocity differences with no 1/dt^2 factor, so the source's
        # -2.5e-7 (on true accelerations) becomes -2.5e-7 / 0.02^2 = -6.25e-4 here.
        dofs_acc=DofsAccL2.configure(weight=-6.25e-4, entity_name="robot"),
        collision=UndesiredContacts.configure(
            weight=-10.0,
            entity_name="robot",
            links_name=["base", ".*_thigh", ".*_calf"],
            threshold=0.1,
        ),
        action_rate=RewardTerm.configure(func=action_rate_norm, weight=-0.1),
        delta_torques=DeltaTorquesL2.configure(weight=-1.0e-7, entity_name="robot"),
        torques=RewardTerm.configure(func=torques_l2, weight=-1.0e-5, params={"entity_name": "robot"}),
        hip_pos=WeightedPosePenalty.configure(
            weight=-0.5,
            entity_name="robot",
            # Go2 dofs_name order: 4 hips first (FL, RL, FR, RR), then thighs, then calves.
            pose_weights=[1.0] * 4 + [0.0] * 8,
        ),
        dof_error=WeightedPosePenalty.configure(
            weight=-0.04,
            entity_name="robot",
            pose_weights=[1.0] * 12,
        ),
        feet_stumble=StumblePenalty.configure(
            weight=-1.0,
            entity_name="robot",
            feet_link_names=FEET_LINK_NAMES,
            horizontal_to_vertical_ratio=4.0,
        ),
        feet_edge=FeetEdgePenalty.configure(
            weight=-1.0,
            entity_name="robot",
            terrain_entity_name="terrain",
            command_name="parkour_command",
            feet_link_names=FEET_LINK_NAMES,
        ),
        # Gait-quality regularizers against asymmetric/limping gaits. feet_air_time
        # rewards proper swing cycles (kills a permanently planted/airborne leg);
        # contact_balance penalizes uneven per-foot duty (catches front/rear imbalance).
        # L/R symmetry is enforced separately by RmaPPO's mirror loss (see runner_options).
        feet_air_time=FeetAirTime.configure(
            weight=1.0,
            entity_name="robot",
            command_name="parkour_command",
            feet_link_names=FEET_LINK_NAMES,
            target_s=0.35,
        ),
        contact_balance=FeetContactBalancePenalty.configure(
            weight=-0.5,
            entity_name="robot",
            feet_link_names=FEET_LINK_NAMES,
        ),
    )

    # All terminations are declared time_out=True: the source bootstraps every episode
    # end (truncation semantics); the rsl_rl wrapper forwards them as time_outs.
    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        roll_limit=DoneTerm.configure(
            func=terminations.base_orientation_roll_greater_than,
            time_out=True,
            # base_orientation_*_greater_than take DEGREES; the source threshold is 1.5 rad.
            params={"entity_name": "robot", "threshold": math.degrees(1.5)},
        ),
        pitch_limit=DoneTerm.configure(
            func=terminations.base_orientation_pitch_greater_than,
            time_out=True,
            params={"entity_name": "robot", "threshold": math.degrees(1.5)},
        ),
        base_height=DoneTerm.configure(
            func=base_height_below,
            time_out=True,
            params={"entity_name": "robot", "threshold": -0.25},
        ),
        all_goals_reached=DoneTerm.configure(
            func=all_goals_reached,
            time_out=True,
            params={"command_name": "parkour_command"},
        ),
    )

    # Order matters: reset_parkour_root_state MUST run before the command reset (it does —
    # reset events fire before command_manager.reset) and before reset_dofs writes the pose.
    event_options: EventManagerOptions = EventManagerOptions(**_make_event_terms())

    # The RMA teacher stack (see the module docstring); source hyperparameters, Cheng et al. 2023.
    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        seed=1,
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=100,
        clip_actions=4.8,
        experiment_name="extreme_parkour_teacher",
        wandb_project="extreme_parkour_teacher",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=rma_actor_options(),
        critic=RslRlCriticOptions(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False),
        algorithm=RslRlRmaPpoAlgorithmOptions(
            learning_rate=2.0e-4,
            schedule="adaptive",
            desired_kl=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.01,
            clip_param=0.2,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            max_grad_norm=1.0,
            estimator_learning_rate=1.0e-4,
            train_with_estimated_states=True,
            dagger_update_freq=20,
            priv_reg_coef_schedule=[0.0, 0.1, 2000, 3000],
            # Mirror-equivariance weight: enforces an L/R-symmetric policy, removing the
            # asymmetric/limping gaits the RMA teacher otherwise learns.
            symmetry_coef=1.0,
            mirror_class_name="eden.tasks.extreme_parkour.symmetry:ParkourMirror",
        ),
    )
