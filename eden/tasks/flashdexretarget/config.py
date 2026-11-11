"""FlashDexRetarget: dexterous hands track human hand-object demos in physics, objects in the loop.

Port of the RL half of FlashDexRetarget (Lee et al., arXiv 2610.01849; github.com/Holiday-Robot/FlashDexRetarget,
Isaac Sim) onto Genesis: one policy, trained across every clip of a ``motion.pt`` built by
``eden retarget flashdex kinematic``, drives one or two robot hands (6 actuated forearm joints + fingers each) so the
objects follow their recorded trajectories while the hands follow the MANO wrists and fingertips. A PD wrench pulls
each object toward its reference at first and is withdrawn as the policy improves (:class:`FDRAssistCurriculum`).
The clips each env plays all use the object it holds (one object variant per env).

::

    eden train --task flashdexretarget --motion <dataset>/xhand/motion.pt                    # XHand
    eden train --task flashdexretarget --motion <dataset>/sharpa/motion.pt                   # Sharpa Wave (bimanual)
    eden train --task flashdexretarget --motion <dataset>/xhand/motion.pt --learner flashsac

The MDP follows FlashDexRetarget's ``config/envs`` (observations, rewards, terminations, curriculum, 60 Hz control
on a 240 Hz simulation). Differences, all documented on the terms they concern:

* PPO (rsl_rl) by default; ``--learner flashsac`` trains with FlashSAC, one actor-critic on the summed reward where
  FlashDexRetarget has one per hand.
* Genesis contacts, not PhysX: friction 2.0 everywhere (Genesis combines a pair as ``max``, PhysX here averaged two
  2.0s), no contact offset, the object-reset depenetration capped at 1 m/s as PhysX's ``max_depenetration_velocity``.
* The residual action is built on the reference of the step it acts in (Eden advances the motion before the action),
  FlashDexRetarget's on the previous one.
* The fingertip distance-bound reward measures the MANO tips' distance with the object SDF, where FlashDexRetarget
  precomputes it to the nearest mesh vertex.
* Each object variant gets an equal block of envs (FlashDexRetarget sizes them by clip count).
"""

from __future__ import annotations
from typing import Annotated

import eden as en
from eden import observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.extensions.retargeting.flashdexretarget.clips import MANO_JOINT_NAMES
from eden.learning.wrappers.rsl_rl.flashsac import (
    RslRlFlashSacActorOptions,
    RslRlFlashSacAlgorithmOptions,
    RslRlFlashSacCriticOptions,
    RslRlFlashSacRunnerOptions,
)
from eden.managers.terms.events.motion import ObjectReferenceStateInitialization
from eden.managers.terms.terminations.common import cumulative_reward_below_threshold
from eden.managers.terms.terminations.tracking import motion_ends
from eden.options import ObservationGroupOptions
from eden.options.entities import EntityOptions, GroupedEntityOptions, PlaneOptions, RobotOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.managers.motions import MotionGroupOptions, MotionManagerOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.flashdexretarget.assets import motion_summary, robot_spec
from eden.tasks.flashdexretarget.dataset import FDR_FRAMERATE, FlashDexRetargetDataset
from eden.tasks.flashdexretarget.custom_terms import (
    FDRActionPenalty,
    FDRAssistCurriculum,
    FDREpisodeCompleted,
    FDRFingertipDiverged,
    FDRFutureTrajectory,
    FDRHandAction,
    FDRHandObservation,
    FDRHandReset,
    FDRJointBCReward,
    FDRMotionTerm,
    FDRObjectAssist,
    FDRObjectDropped,
    FDRObjectPointCloudReward,
    FDRObjectPosError,
    FDRObjectRotError,
    FDRTargetError,
    FDRTipDistanceBoundReward,
    FDRTipPosError,
    FDRTipTransReward,
    FDRVelocitySanity,
    FDRWristRotReward,
    FDRWristTransReward,
)
from eden.tasks.registry import TaskMod
from eden.terms import DoneTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CurriculumManagerOptions,
    EdenMimicConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)
from eden.utils.distribution import install_hint

#: `--learner`'s help. A name, not a literal in the annotation, so it can call `install_hint`: `cli_fields` resolves the
#: postponed annotation against this module's globals.
_LEARNER_HELP = (
    f"ppo (rsl_rl PPO) or flashsac (FlashDexRetarget's off-policy learner; install it with {install_hint('flashsac')})."
)

#: Contact friction of every body (FlashDexRetarget's PhysX material).
FRICTION = 2.0
#: Next reference frames the future-trajectory observation sees (FlashDexRetarget: 10).
N_FUTURE = 10


def _robot_of(summary, robot: str | None) -> str:
    """The robot ``summary``'s motion was retargeted to; ``robot`` (``--robot``) must match it, and names it for a
    motion file from before ``motion.pt`` carried its robot.
    """
    if summary.robot is None and robot is None:
        raise ValueError(
            f"{summary.path} does not name the robot it was retargeted to (an earlier `eden retarget flashdex "
            "kinematic` wrote it); pass --robot xhand or --robot sharpa, or rebuild it."
        )
    if summary.robot is not None and robot is not None and robot != summary.robot:
        raise ValueError(
            f"--robot {robot} but {summary.path} was retargeted to {summary.robot}: its joint_pos columns are "
            f"{summary.robot}'s joints. Drop --robot, or pass the {robot} motion (<dataset>/{robot}/motion.pt)."
        )
    return robot or summary.robot


def _robot_options(robot: str, sides: tuple[str, ...]) -> RobotOptions:
    """The fixed-base hands, with the MJCF's drive gains, armature, damping, friction loss and force limits."""
    spec = robot_spec(robot, sides)
    keep = sorted(
        {spec.palm_link[s] for s in sides}
        | {n for s in sides for n in spec.tip_links[s]}
        | {n for s in sides for n in spec.contact_links[s]}
        | {r for s in sides for r, _ in spec.body_map[s]}
    )
    return RobotOptions(
        file=spec.file,
        is_fixed_base=True,
        links_to_keep=keep,
        dofs_name=list(spec.joint_names),
        default_dofs_kp=spec.kp,
        default_dofs_kd=spec.kd,
        default_dofs_damping=spec.damping,
        default_dofs_armature=spec.armature,
        default_dofs_frictionloss=spec.frictionloss,
        default_dofs_force_limits=spec.force_limit,
        # FlashDexRetarget clips resets and targets to the MJCF ranges themselves, not Eden's 0.9 soft range.
        soft_dofs_pos_limits=spec.limits,
        default_geoms_friction={"*": FRICTION},
    )


def _object_options(urdfs: list[str]) -> EntityOptions | GroupedEntityOptions:
    """One free object, or one variant per object slot (motion.pt's slot order).

    ``align=False`` keeps the free joint at the authored mesh frame, the frame of the references, the SDF grids and
    the surface keypoints (Genesis would otherwise re-frame the body onto its centre of mass, as PhysX does not).
    """
    variants = [EntityOptions(file=u, align=False, default_geoms_friction={"*": FRICTION}) for u in urdfs]
    if len(variants) == 1:
        variants[0].is_fixed_base = False
        return variants[0]
    return GroupedEntityOptions(grouped_entities=variants, is_fixed_base=False)


class FlashDexRetargetDataMod(TaskMod):
    """Point the task at a ``motion.pt`` and a robot; build everything that depends on them.

    The scene (robot hands, one object entity per held object, a variant per object slot), the motion manager
    and every term take the motion file's hands and objects, so they are all set here::

        eden train --task flashdexretarget --motion <dataset>/xhand/motion.pt

    The robot is the one the motion was retargeted to (``motion.pt``'s ``robot``); ``--robot`` only needs giving for a
    motion file from before it carried one, and must otherwise match it.
    """

    def __init__(
        self,
        motion: Annotated[
            str | None,
            "The <dataset>/<robot>/motion.pt `eden retarget flashdex kinematic` wrote (with the object URDFs).",
        ] = None,
        robot: Annotated[
            str | None, "The robot the motion was retargeted to (xhand or sharpa); by default the motion's own."
        ] = None,
        learner: Annotated[
            str,
            _LEARNER_HELP,
        ] = "ppo",
    ):
        self.motion = motion
        self.robot = robot
        self.learner = learner

    def apply(self, config: FlashDexRetargetConfig) -> FlashDexRetargetConfig:
        if self.learner not in ("ppo", "flashsac"):
            raise ValueError(f"--learner must be 'ppo' or 'flashsac', got {self.learner!r}.")
        if self.motion is None:
            raise ValueError(
                "flashdexretarget needs a motion file: pass --motion <dataset>/<robot>/motion.pt, built by "
                "`eden retarget flashdex import` + `eden retarget flashdex kinematic`."
            )
        summary = motion_summary(self.motion)
        sides, robot = summary.sides, _robot_of(summary, self.robot)
        spec = robot_spec(robot, sides)
        objects = summary.object_entities
        common = dict(hand_model=robot, sides=list(sides), objects=objects, dataset=str(summary.path))

        config.scene_options = en.scenes.Empty(
            robot=_robot_options(robot, sides),
            plane=PlaneOptions(default_geoms_friction={"*": FRICTION}),
            **{name: _object_options(urdfs) for name, urdfs in summary.slot_urdfs.items()},
        )

        object_entities = sorted(set(objects.values()))
        entities_map = {"robot": "robot", **{f"mano_{s}": f"mano_{s}" for s in sides}}
        entities_map.update(
            {name: f"object_{next(s for s in sides if objects[s] == name)}" for name in object_entities}
        )
        seq_keys = ["robot", *(f"mano_{s}" for s in sides), *(f"object_{s}" for s in sides)]
        many_slots = len(summary.slot_labels) > 1
        config.motion_options = MotionManagerOptions(
            num_frames=N_FUTURE + 1,
            data_start_from="current_time",
            frame_interval_s_range=(1.0 / FDR_FRAMERATE, 1.0 / FDR_FRAMERATE),
            default=FDRMotionTerm.configure(
                dataset_options=FlashDexRetargetDataset.configure(raw_data_path=str(summary.path), robot=robot),
                entities_map=entities_map,
                target_dofs_name={k: (list(spec.joint_names) if k == "robot" else []) for k in seq_keys},
                target_links_name={k: (list(MANO_JOINT_NAMES) if k.startswith("mano_") else []) for k in seq_keys},
                # FlashDexRetarget starts a training episode at a uniform frame in [0, 0.99 T); eval, at frame 0.
                motion_start_from_middle_range=(0.0, 0.99),
                env_motion_groups=(
                    MotionGroupOptions(
                        entity_name=object_entities[0], metadata_key="objects", variant_labels=summary.slot_labels
                    )
                    if many_slots
                    else None
                ),
            ),
        )

        wrist_trans = [n for s in sides for n in spec.wrist_trans[s]]
        wrist_rot = [n for s in sides for n in spec.wrist_rot[s]]
        config.action_options = ActionManagerOptions(
            hand=FDRHandAction.configure(
                entity_name="robot",
                dofs_name=list(spec.joint_names),
                reference_source=ReferenceSource.MOTION_REF,
                residual_scale={**dict.fromkeys(wrist_trans, 0.1), **dict.fromkeys(wrist_rot, 1.0)},
                cutoff_hz=dict.fromkeys(wrist_trans + wrist_rot, 5.0),
            ),
            assist=FDRObjectAssist.configure(**common),
        )
        config.observation_options = _observations(common)
        config.reward_options = _rewards(common)
        config.termination_options = _terminations(common)
        config.event_options = _events(common, object_entities)
        config.metric_options = _metrics(common)
        config.curriculum_options = CurriculumManagerOptions(
            assist_decay=FDRAssistCurriculum.configure(
                assist_term="assist",
                reward_terms={
                    "task": ["object_pcd"],
                    "imi": ["wrist_trans", "wrist_rot", "tip_trans"],
                    "bc": ["joint_bc"],
                },
                rew_thresholds={"task": 0.6, "imi": 0.01, "bc": 0.01},
            )
        )
        if self.learner == "flashsac":
            config.runner_options = flashsac_runner_options()
            # FlashSAC bootstraps a timed-out episode from its last observation. PPO never reads it, and recording it
            # computes every observation term a second time each step.
            config.env_options.record_final_observations = True
        # <dataset>/<robot>/motion.pt: the dataset is the directory above the robot's.
        dataset_name = summary.path.parent.parent.name
        config.runner_options.experiment_name = f"flashdexretarget_{dataset_name}_{robot}_{self.learner}"
        return config


def flashsac_runner_options(num_env_steps: int = 500_000_000) -> RslRlFlashSacRunnerOptions:
    """FlashDexRetarget's FlashSAC (``config/runner/flashsac.yaml``): gamma 0.98, 5-step returns, 2 updates of 4096
    per env step, a 128-wide actor and a 1024-wide distributional critic (2 blocks each), a float16 replay buffer of
    50M transitions (each the whole 1544-value observation with one XHand: ~154 GB on the GPU, where
    FlashDexRetarget's compact rows need 88 GB; set ``algorithm.replay_buffer_size`` for the GPU, e.g. 10M on 48 GB),
    learning rate 3e-4 decaying to 1.5e-4 over the run.

    FlashDexRetarget trains one actor and one critic per hand on that hand's reward; this is one of each on the
    summed reward.

    Parameters
    ----------
    num_env_steps : int
        Env steps of the run over every GPU (FlashDexRetarget: 500M): ``num_env_steps / (num_envs x ranks)``
        iterations of one env step each, sized when the env is built, so ``--num_envs`` and ``torchrun`` keep the
        run's length; the learning rate decays over it.
    """
    return RslRlFlashSacRunnerOptions(
        num_steps_per_env=1,
        max_iterations=None,
        num_env_steps=num_env_steps,
        save_interval=5_000,  # ~20M env steps at 4096 envs, FlashDexRetarget's
        log_interval=100,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlFlashSacActorOptions(num_blocks=2, hidden_dim=128),
        critic=RslRlFlashSacCriticOptions(num_blocks=2, hidden_dim=1024, num_bins=101, min_v=-5.0, max_v=5.0),
        algorithm=RslRlFlashSacAlgorithmOptions(
            replay_buffer_size=50_000_000,
            buffer_min_length=100_000,
            buffer_obs_dtype="float16",
            num_learning_epochs=1,
            num_mini_batches=2,
            mini_batch_size=4096,
            learning_rate_init=3e-4,
            learning_rate_peak=3e-4,
            learning_rate_end=1.5e-4,
            learning_rate_warmup_steps=0,  # FlashDexRetarget's 1e-6 of the run: 0 at any env count
            learning_rate_decay_steps=None,  # the run: max_iterations x 2 updates
            gamma=0.98,
            n_steps=5,
            normalize_reward=True,
            normalized_G_max=5.0,
            temp_initial_value=0.01,
            temp_target_sigma=0.15,
        ),
    )


def _hand_obs(common: dict, kind: str, scale: float = 1.0) -> ObsTerm:
    return FDRHandObservation.configure(kind=kind, scale=scale, **common)


def _observations(common: dict) -> ObservationManagerOptions:
    """FlashDexRetarget's observation list, with its scales; the future-trajectory window last.

    One group: FlashDexRetarget's actor and critic observe the same list, so the critic reads ``policy`` (rsl_rl
    falls back to it without a ``critic`` group) rather than computing it twice. The contact and the palm/fingertip
    SDF observations are each one fused term (the same values in the same order as FlashDexRetarget's pairs), so
    each SDF query runs once.
    """
    terms = dict(
        robot_wrist_pose_w=_hand_obs(common, "robot_wrist_pose_w"),
        robot_wrist_lin_vel_w=_hand_obs(common, "robot_wrist_lin_vel_w", 2.0),
        robot_wrist_ang_vel_w=_hand_obs(common, "robot_wrist_ang_vel_w", 0.25),
        robot_joint_pos=observations.DofsPos.configure(entity_name="robot"),
        robot_joint_vel=observations.DofsVel.configure(entity_name="robot", scale=0.1),
        robot_tip_trans_wrist=_hand_obs(common, "robot_tip_trans_wrist"),
        robot_tip_lin_vel_wrist=_hand_obs(common, "robot_tip_lin_vel_wrist", 2.0),
        dof_target_pos=FDRTargetError.configure(action_term="hand"),
        ref_future_mano_body_delta_w=_hand_obs(common, "ref_future_mano_body_delta_w"),
        obj_trans_wrist=_hand_obs(common, "obj_trans_wrist"),
        obj_quat_wrist=_hand_obs(common, "obj_quat_wrist"),
        obj_lin_vel_wrist=_hand_obs(common, "obj_lin_vel_wrist", 2.0),
        obj_ang_vel_wrist=_hand_obs(common, "obj_ang_vel_wrist", 0.25),
        obj_point_cloud_wrist=_hand_obs(common, "obj_point_cloud_wrist"),
        ref_future_obj_trans_delta_wrist=_hand_obs(common, "ref_future_obj_trans_delta_wrist"),
        ref_future_obj_quat_delta_wrist=_hand_obs(common, "ref_future_obj_quat_delta_wrist"),
        ref_future_obj_point_cloud_delta_wrist=_hand_obs(common, "ref_future_obj_point_cloud_delta_wrist"),
        robot_body_contact_wrist=_hand_obs(common, "robot_body_contact_wrist"),
        robot_links_obj_sdf_normal_wrist=_hand_obs(common, "robot_links_obj_sdf_normal_wrist"),
        future_traj=FDRFutureTrajectory.configure(n_future=N_FUTURE, **common),
    )
    return ObservationManagerOptions(
        policy=ObservationGroupOptions(concatenate_terms=True, enable_corruption=False, **terms),
    )


def _rewards(common: dict) -> RewardManagerOptions:
    """FlashDexRetarget's ``rewards/flashdexretarget.yaml``: task, imitation, behaviour cloning, regularisation."""
    return RewardManagerOptions(
        object_pcd=FDRObjectPointCloudReward.configure(weight=1.0, scale=10.0, topk=3, norm_radius=0.058, **common),
        tip_distance_bound=FDRTipDistanceBoundReward.configure(weight=0.3, sigma=0.01, **common),
        wrist_trans=FDRWristTransReward.configure(weight=0.15, scale=10.0, **common),
        wrist_rot=FDRWristRotReward.configure(weight=0.15, scale=3.0, **common),
        tip_trans=FDRTipTransReward.configure(weight=0.3, scale=20.0, **common),
        joint_bc=FDRJointBCReward.configure(weight=0.3, scale=500.0, action_term="hand"),
        action_penalty=FDRActionPenalty.configure(weight=-0.01, action_term="hand"),
        termination_penalty=RewardTerm.configure(func=rewards.is_terminated_penalty, weight=-100.0),
    )


def _terminations(common: dict) -> TerminationManagerOptions:
    return TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        motion_clip_ended=DoneTerm.configure(func=motion_ends, params={"entity_name": "robot"}, time_out=True),
        object_dropped=FDRObjectDropped.configure(drop_margin=0.10, **common),
        task_reward_early_reset=DoneTerm.configure(
            func=cumulative_reward_below_threshold,
            params={"reward_term": "object_pcd", "per_step_threshold": 0.6, "check_interval": 5},
        ),
        fingertip_diverged=FDRFingertipDiverged.configure(threshold=0.2, grace_steps=15, **common),
        velocity_sanity=FDRVelocitySanity.configure(**common),
    )


def _metrics(common: dict) -> MetricManagerOptions:
    """Per episode: whether it completed (``Metrics/Success``) and its mean tracking errors."""
    return MetricManagerOptions(
        completed=FDREpisodeCompleted.configure(entity_name="robot"),
        object_pos_error=FDRObjectPosError.configure(**common),
        object_rot_error=FDRObjectRotError.configure(**common),
        tip_pos_error=FDRTipPosError.configure(**common),
    )


def _events(common: dict, object_entities: list[str]) -> EventManagerOptions:
    """Reset onto the reference: the hands with FlashDexRetarget's small noise, each object exactly (1 m/s cap)."""
    events = dict(
        reset_hands=FDRHandReset.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            hand_model=common["hand_model"],
            sides=common["sides"],
            priority=100,
        )
    )
    for name in object_entities:
        events[f"reset_{name}"] = ObjectReferenceStateInitialization.configure(
            mode=en.EventMode.RESET, entity_name=name, xy_range=0.0, max_depenetration_velocity=1.0, priority=99
        )
    return EventManagerOptions(**events)


@TASK_REGISTRY.register(name="flashdexretarget", modifiers=(FlashDexRetargetDataMod(),))
class FlashDexRetargetConfig(EdenMimicConfig):
    """FlashDexRetarget: robot hands track the human hand-object demos of a ``motion.pt``, objects in the loop.

    :class:`FlashDexRetargetDataMod` (a default modifier) builds the scene and every term from ``--motion`` and
    ``--robot``.
    """

    # Replaced by FlashDexRetargetDataMod; a placeholder robot so the class builds without reading any file.
    scene_options: SceneOptions = en.scenes.Empty(
        robot=RobotOptions(file="hands/flashdexretarget/xhand/urdf/xhand_right_spec.urdf", is_fixed_base=True),
        plane=PlaneOptions(),
    )

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=1.0 / 240.0,
        sim_substeps=1,
        decimation=4,  # 60 Hz control
        enable_self_collision=True,
        batch_dofs_info=False,
    )

    action_options: ActionManagerOptions = ActionManagerOptions()
    observation_options: ObservationManagerOptions = ObservationManagerOptions()
    reward_options: RewardManagerOptions = RewardManagerOptions()
    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )
    event_options: EventManagerOptions = EventManagerOptions()
    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions()

    # PPO by default; --learner flashsac swaps in FlashDexRetarget's FlashSAC (flashsac_runner_options).
    runner_options: RslRlOnPolicyRunnerOptions | RslRlFlashSacRunnerOptions = RslRlOnPolicyRunnerOptions(
        num_steps_per_env=32,
        max_iterations=20000,
        save_interval=500,
        experiment_name="flashdexretarget",
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[1024, 512, 256],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[1024, 512, 256]),
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    def check_bindings(self) -> None:
        """Raise if the runner is FlashSAC but the env does not record final observations, naming ``--learner``.

        :class:`FlashDexRetargetDataMod` turns both on together (``--learner flashsac``). A config that switches
        ``runner_options`` to FlashSAC any other way (``--override``, a PPO run's saved config) keeps the PPO default
        off, and the FlashSAC wrapper's own check would refuse it with advice that does not name the variant flag.

        Raises
        ------
        ValueError
            If ``runner_options`` is :class:`RslRlFlashSacRunnerOptions` and ``record_final_observations`` is off.
        """
        super().check_bindings()
        if (
            isinstance(self.runner_options, RslRlFlashSacRunnerOptions)
            and not self.env_options.record_final_observations
        ):
            raise ValueError(
                f"{type(self).__name__} trains with FlashSAC (runner_options {type(self.runner_options).__name__}), "
                "but env_options.record_final_observations is False, and FlashSAC bootstraps a timed-out episode from "
                "its last observation. Select FlashSAC with `--learner flashsac`, which turns it on together with "
                "FlashDexRetarget's FlashSAC hyperparameters, rather than swapping runner_options with --override; "
                "to keep a hand-written FlashSAC runner_options, also pass "
                "--override 'env_options: {record_final_observations: true}'."
            )
