"""BeyondMimic motion-tracking task configuration."""

from typing import Annotated

import eden as en
from eden.terms import (
    ObsTerm,
    RewardTerm,
    DoneTerm,
    MetricTerm,
)
from eden import actions, events, observations, rewards, terminations
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlOnPolicyRunnerOptions,
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.managers.motions import MotionManagerOptions
from eden.managers.motion_manager import MotionTerm
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskMod
from eden.utils.configs import (
    EdenMimicConfig,
    EnvOptions,
    RewardManagerOptions,
    SceneOptions,
    ActionManagerOptions,
    ObservationManagerOptions,
    TerminationManagerOptions,
    EventManagerOptions,
    MetricManagerOptions,
)

from eden.tasks.beyondmimic.dataset import BvhCsvG1Dataset
from eden.tasks.beyondmimic.custom_terms.metrics import (
    MotionErrorBodyAngVel,
    MotionErrorBodyLinVel,
    MotionErrorBodyPos,
    MotionErrorBodyRot,
    motion_error_anchor_ang_vel,
    motion_error_anchor_lin_vel,
    motion_error_anchor_pos,
    motion_error_anchor_rot,
    motion_error_dofs_pos,
    motion_error_dofs_vel,
)
from eden.managers.terms.events.motion import ReferenceStateInitialization
from eden.managers.terms.observations.tracking import (
    motion_dofs_pos,
    motion_dofs_vel,
    motion_links_ori_b,
    motion_links_pos_b,
    robot_body_ori_b,
    robot_body_pos_b,
)
from eden.managers.terms.rewards.tracking import (
    MotionGlobalBodyAngularVelocityErrorExp,
    MotionGlobalBodyLinearVelocityErrorExp,
    MotionRelativeBodyOrientationErrorExp,
    MotionRelativeBodyPositionErrorExp,
    motion_link_orientation_tracking_exp,
    motion_link_position_tracking_exp,
)
from eden.managers.terms.terminations.tracking import (
    BadMotionLinkPosZOnly,
    bad_anchor_ori,
    bad_anchor_pos_z_only,
)

from eden.options.robots.unitree_g1 import G1Dofs29, G1_ACTION_SCALE
from eden.constants import ReferenceSource

_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

_G1Options = G1Dofs29()

_TRACKED_LINKS = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

_ANCHOR_LINK = "torso_link"


# Neither default is derived from the source-checkout layout: ``examples/`` is not packaged, so on a
# pip-installed Eden a checkout-relative path points inside ``site-packages/``, and the task builds fine
# but dies a minute in, once the dataset loader runs. Both resolve inside ``BvhCsvG1Dataset`` instead —
# see ``BvhCsvG1Dataset._download_from_hf`` and ``_default_processed_path`` in ``dataset.py``.
class BeyondMimicDataMod(TaskMod):
    """Configure dataset paths + framerate for the BeyondMimic motion-imitation task.

    Passes the paths through to ``motion_options`` untouched, so a build with no paths given
    (``TASK_REGISTRY.build("beyondmimic")``) is immediately runnable against the bundled sample clip —
    on a pip install as much as in a checkout, though the first build of either fetches the clip.
    Override via CLI:

    ``eden train --task beyondmimic --raw_data_path /my/bvh_csv_dir --processed_data_path /my/cache.pt``
    """

    def __init__(
        self,
        raw_data_path: Annotated[
            str | None,
            "Directory of BVH CSV files for the G1 robot, or a single one. Omit to fetch the "
            "bundled sample clip from HuggingFace (Kashu7100/eden_example, beyondmimic/) on "
            "first use.",
        ] = None,
        processed_data_path: Annotated[
            str | None,
            "Path to the processed-motion cache .pt file. Omit for a keyed cache in Eden's "
            "cache dir — one file per set of settings that shape the prepared motion, so a "
            "cache built from one clip set, rate or frame range is never loaded for another.",
        ] = None,
        target_framerate: Annotated[float, "Target framerate (Hz) for retiming the motion clip."] = 50.0,
    ):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.target_framerate = target_framerate

    def apply(self, config: "BeyondMimicConfig") -> "BeyondMimicConfig":
        # An unset shell variable (``--raw_data_path "$DATA_DIR"``) expands to the empty string,
        # which must not read as "omitted" — that would silently train on the sample clip.
        # Rejected here, where the flag can be named. Existence of a path the caller *did* type
        # is deliberately unchecked: unlike contrack this mod reads nothing, so a config can be
        # built on a host that has not mounted the data yet.
        for flag, value in (("raw_data_path", self.raw_data_path), ("processed_data_path", self.processed_data_path)):
            if value is not None and not value.strip():
                raise ValueError(
                    f"--{flag} was given an empty path. Omit the flag to use the default "
                    f"(the sample clip fetched from HuggingFace / a keyed cache in Eden's "
                    f"cache dir), or pass a real path."
                )
        config.motion_options = MotionManagerOptions(
            num_frames=1,
            data_start_from="current_time",
            frame_interval_s_range=(0.02, 0.02),
            # At clip end, soft-reset (resample start phase + teleport onto the
            # reference) instead of freezing at the last frame
            resync_on_motion_end=True,
            default=MotionTerm.configure(
                dataset_options=BvhCsvG1Dataset.configure(
                    # Both left as given, None included, because both defaults resolve inside
                    # the dataset: ``_download_from_hf`` fetches the clip from within
                    # ``_initialize``'s rank-0 gate, and ``_default_processed_path`` keys the
                    # cache once the options are final. Resolving either here would bake a
                    # ``$HOME`` path into the saved config (unreplayable on another machine),
                    # and the cache key would miss every ``--override``, which a task mod runs
                    # before.
                    raw_data_path=self.raw_data_path,
                    processed_data_path=self.processed_data_path,
                    target_framerate=self.target_framerate,
                ),
                entities_map={"robot": "g1"},
                target_dofs_name={"g1": _G1Options.dofs_name},
                target_links_name={"g1": BvhCsvG1Dataset.motion_links_name["g1"]},
                motion_bin_length_s=1.0,
                adaptive_sampling=False,
                motion_start_from_middle_range=(0.0, 1.0),
            ),
        )
        return config


_CommonObservations = dict(
    motion_dofs_pos=ObsTerm.configure(
        func=motion_dofs_pos,
        params={"entity_name": "robot"},
    ),
    motion_dofs_vel=ObsTerm.configure(
        func=motion_dofs_vel,
        params={"entity_name": "robot"},
    ),
    motion_anchor_pos_b=ObsTerm.configure(
        func=motion_links_pos_b,
        params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "anchor_link_name": _ANCHOR_LINK},
        noise=UniformNoise.configure(n_min=-0.25, n_max=0.25),
    ),
    motion_anchor_ori_b=ObsTerm.configure(
        func=motion_links_ori_b,
        params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "anchor_link_name": _ANCHOR_LINK},
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.5, n_max=0.5),
    ),
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        params={"entity_name": "robot", "frame": "body"},
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
    ),
    dofs_pos=observations.DofsPos.configure(
        entity_name="robot", offset_from_default=True, noise=UniformNoise.configure(n_min=-0.01, n_max=0.01)
    ),
    dofs_vel=observations.DofsVel.configure(entity_name="robot", noise=UniformNoise.configure(n_min=-0.5, n_max=0.5)),
    last_action=ObsTerm.configure(func=observations.last_action),
)


@TASK_REGISTRY.register(name="beyondmimic", modifiers=(BeyondMimicDataMod(),))
class BeyondMimicConfig(EdenMimicConfig):
    """Configuration for the MimicRLEnvBase-based BeyondMimic tracking policy.

    Dataset paths and framerate are configured by :class:`BeyondMimicDataMod`,
    which is attached as a default modifier. Construct via
    ``TASK_REGISTRY.build("beyondmimic", raw_data_path=..., processed_data_path=...)``
    or apply the mod directly. A bare ``BeyondMimicConfig()`` populates
    everything except ``motion_options`` — call the mod (or
    :func:`eden.tasks.parser.get_task_config_from_args`) to finish setup.
    """

    # A field default, not a model_post_init assignment: model_post_init re-runs on every
    # reconstruction (load / --config merge) and would clobber user overrides back to this default.
    scene_options: SceneOptions = en.scenes.Empty(robot=_G1Options)

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=10.0,
        iterations=10,
        ls_iterations=20,
        max_collision_pairs=20,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,
        batch_dofs_info=False,
        batch_links_info=True,  # RandomizeComShift writes per-env link COMs
        use_gjk_collision=False,
        enable_self_collision=False,
        enable_multi_contact=False,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=True,
            **_CommonObservations,
        ),
        critic=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=False,
            **_CommonObservations,
            body_pos_b=ObsTerm.configure(
                func=robot_body_pos_b,
                params={
                    "entity_name": "robot",
                    "anchor_link_name": _ANCHOR_LINK,
                    "tracked_link_names": _TRACKED_LINKS,
                },
            ),
            body_ori_b=ObsTerm.configure(
                func=robot_body_ori_b,
                params={
                    "entity_name": "robot",
                    "anchor_link_name": _ANCHOR_LINK,
                    "tracked_link_names": _TRACKED_LINKS,
                },
            ),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        motion_global_anchor_pos=RewardTerm.configure(
            func=motion_link_position_tracking_exp,
            weight=0.5,
            params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "std": 0.3},
        ),
        motion_global_anchor_ori=RewardTerm.configure(
            func=motion_link_orientation_tracking_exp,
            weight=0.5,
            params={"entity_name": "robot", "links_name": [_ANCHOR_LINK], "std": 0.4},
        ),
        motion_body_pos=MotionRelativeBodyPositionErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
            std=0.3,
        ),
        motion_body_ori=MotionRelativeBodyOrientationErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
            std=0.4,
        ),
        motion_body_lin_vel=MotionGlobalBodyLinearVelocityErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            links_name=_TRACKED_LINKS,
            std=1.0,
        ),
        motion_body_ang_vel=MotionGlobalBodyAngularVelocityErrorExp.configure(
            weight=1.0,
            entity_name="robot",
            links_name=_TRACKED_LINKS,
            std=3.14,
        ),
        action_rate_l2=RewardTerm.configure(
            func=rewards.action_rate_l2,
            weight=-1e-1,
        ),
        joint_limit=RewardTerm.configure(
            func=rewards.dofs_pos_limits,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
        self_collisions=RewardTerm.configure(
            func=rewards.self_collision_cost,
            weight=-10.0,
            params={"entity_name": "robot"},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        anchor_pos=DoneTerm.configure(
            func=bad_anchor_pos_z_only,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK, "threshold": 0.25},
        ),
        # Compared on the unit gravity direction, as in BeyondMimic: 0.8 is ~78 deg off an upright reference.
        anchor_ori=DoneTerm.configure(
            func=bad_anchor_ori,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK, "threshold": 0.8},
        ),
        ee_body_pos=BadMotionLinkPosZOnly.configure(
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            threshold=0.25,
            links_name=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        reset_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=_VELOCITY_RANGE,
            dofs_position_range=(-0.1, 0.1),
            priority=100,
        ),
        push_robot=events.PushByVelocity.configure(
            mode=en.EventMode.INTERVAL,
            interval_range_s=(1.0, 3.0),
            entity_name="robot",
            velocity_range=_VELOCITY_RANGE,
        ),
        base_com_shift=events.RandomizeComShift.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["pelvis"],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.05, 0.05),
            com_shift_z_range=(-0.05, 0.05),
        ),
        dofs_default_pos=events.RandomizeDofsPosOffset.configure(
            mode=en.EventMode.STARTUP,
            action_term_name="dofs_pos_controller",
            entity_name="robot",
            dofs_pos_range=(-0.01, 0.01),
        ),
        foot_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["*_ankle_roll_link"],
            friction_range=(0.3, 1.2),
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=actions.ExplicitPDController.configure(
            entity_name="robot",
            dofs_name=_G1Options.dofs_name,
            scale=G1_ACTION_SCALE,
            reference_source=ReferenceSource.DEFAULT,
        )
    )

    # Tracking-error diagnostics (not part of the reward); they read ``env.motion_manager``.
    metric_options: MetricManagerOptions = MetricManagerOptions(
        error_anchor_pos=MetricTerm.configure(
            func=motion_error_anchor_pos,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK},
        ),
        error_anchor_rot=MetricTerm.configure(
            func=motion_error_anchor_rot,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK},
        ),
        error_anchor_lin_vel=MetricTerm.configure(
            func=motion_error_anchor_lin_vel,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK},
        ),
        error_anchor_ang_vel=MetricTerm.configure(
            func=motion_error_anchor_ang_vel,
            params={"entity_name": "robot", "anchor_link_name": _ANCHOR_LINK},
        ),
        error_body_pos=MotionErrorBodyPos.configure(
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
        ),
        error_body_rot=MotionErrorBodyRot.configure(
            entity_name="robot",
            anchor_link_name=_ANCHOR_LINK,
            links_name=_TRACKED_LINKS,
        ),
        error_body_lin_vel=MotionErrorBodyLinVel.configure(
            entity_name="robot",
            links_name=_TRACKED_LINKS,
        ),
        error_body_ang_vel=MotionErrorBodyAngVel.configure(
            entity_name="robot",
            links_name=_TRACKED_LINKS,
        ),
        error_dofs_pos=MetricTerm.configure(
            func=motion_error_dofs_pos,
            params={"entity_name": "robot"},
        ),
        error_dofs_vel=MetricTerm.configure(
            func=motion_error_dofs_vel,
            params={"entity_name": "robot"},
        ),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        num_steps_per_env=24,
        max_iterations=15000,
        save_interval=500,
        experiment_name="mimic_beyondmimic",
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
        ),
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=0.001,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
