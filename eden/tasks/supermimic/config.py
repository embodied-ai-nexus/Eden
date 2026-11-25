"""SuperMimic: whole-body motion tracking on the Unitree G1, trained through a differentiable world model.

The environment is the BeyondMimic tracking task as SuperMimic configures it: the pelvis as the anchor, no pushes,
no self-collision penalty, a joint-limit weight of -2, foot friction, torso centre-of-mass and encoder-bias
randomization fixed per env, and episodes truncated at the end of each reference clip. The learner is
:class:`~eden.learning.wrappers.rsl_rl.supermimic.SuperMimicRunnerOptions`.
"""

from __future__ import annotations

from typing import Annotated, Literal

import eden as en
from eden import actions, events, observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.datasets import LAFAN1RootJointsG1Dataset, SEEDRootJointsG1Dataset
from eden.learning.wrappers.rsl_rl.supermimic import SuperMimicRunnerOptions
from eden.learning.wrappers.rsl_rl.supermimic.motion import SuperMimicMotion
from eden.learning.wrappers.rsl_rl.supermimic.terms import DofsPosBiased, ReferenceClipEnd, RootJointState
from eden.managers.terms.events.motion import ReferenceStateInitialization
from eden.managers.terms.rewards.tracking import (
    MotionGlobalBodyAngularVelocityErrorExp,
    MotionGlobalBodyLinearVelocityErrorExp,
    MotionRelativeBodyOrientationErrorExp,
    MotionRelativeBodyPositionErrorExp,
    motion_link_orientation_tracking_exp,
    motion_link_position_tracking_exp,
)
from eden.managers.terms.terminations.tracking import bad_anchor_ori, bad_anchor_pos_z_only
from eden.options import CudaGraphOptions, ObservationGroupOptions
from eden.options.entities import PlaneOptions
from eden.options.managers.motions import MotionManagerOptions
from eden.options.robots.unitree_g1 import G1_ACTION_SCALE, G1Dofs29
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskMod
from eden.terms import DoneTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EdenMimicConfig,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

#: mjlab's ``KNEES_BENT_KEYFRAME``: the default pose, which is the action offset.
KNEES_BENT_DOFS_POS: dict[str, float] = {
    "*_hip_pitch_joint": -0.312,
    "*_hip_roll_joint": 0.0,
    "*_hip_yaw_joint": 0.0,
    "*_knee_joint": 0.669,
    "*_ankle_pitch_joint": -0.363,
    "*_ankle_roll_joint": 0.0,
    "waist_*_joint": 0.0,
    "*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "*_shoulder_yaw_joint": 0.0,
    "*_elbow_joint": 0.6,
    "*_wrist_*_joint": 0.0,
}

TRACKED_LINKS: list[str] = [
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
ANCHOR_LINK = "pelvis"

_G1 = G1Dofs29(default_dofs_pos=KNEES_BENT_DOFS_POS, default_root_pos=(0.0, 0.0, 0.76))

_DATASETS = {"lafan1": LAFAN1RootJointsG1Dataset, "bones_seed": SEEDRootJointsG1Dataset}


class SuperMimicDataMod(TaskMod):
    """Select the motion library: LAFAN1 or BONES-SEED, each retargeted to the G1.

    ``eden train --task supermimic --dataset bones_seed --raw_data_path /data/bones_seed/g1/csv
    --processed_data_path /data/bones_seed_g1.pt --clip_list bones_seed_clips.txt``
    """

    def __init__(
        self,
        dataset: Annotated[
            Literal["lafan1", "bones_seed"],
            "Motion library. LAFAN1 downloads on first use; BONES-SEED is gated on HuggingFace.",
        ] = "lafan1",
        raw_data_path: Annotated[str | None, "Directory of the library's CSV clips; omit to download it."] = None,
        processed_data_path: Annotated[
            str | None, "Cache of the prepared library (.pt); written on first use, read afterwards."
        ] = None,
        clip_list: Annotated[
            str | None,
            "A .json/.yaml list of clip paths relative to raw_data_path (or a {motions: [...]} mapping); omit for all.",
        ] = None,
    ):
        self.dataset = dataset
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.clip_list = clip_list

    def apply(self, config: SuperMimicConfig) -> SuperMimicConfig:
        for flag in ("raw_data_path", "processed_data_path", "clip_list"):
            value = getattr(self, flag)
            if value is not None and not value.strip():
                raise ValueError(f"--{flag} was given an empty path. Omit the flag or pass a real path.")
        dataset_options = _DATASETS[self.dataset].configure(
            raw_data_path=self.raw_data_path,
            processed_data_path=self.processed_data_path,
            include_list_file=self.clip_list,
            target_framerate=1.0 / (config.env_options.sim_dt * config.env_options.decimation),
        )
        config.motion_options = MotionManagerOptions(
            num_frames=2,
            data_start_from="current_time",
            frame_interval_s_range=(0.02, 0.02),
            default=SuperMimicMotion.configure(
                dataset_options=dataset_options,
                entities_map={"robot": "g1"},
                target_dofs_name={"g1": _G1.dofs_name},
                target_links_name={"g1": TRACKED_LINKS},
            ),
        )
        return config


def _tracked(name: str) -> dict:
    return {"entity_name": "robot", "links_name": [name]}


@TASK_REGISTRY.register(name="supermimic", modifiers=(SuperMimicDataMod(),), tags=("motion_tracking",))
class SuperMimicConfig(EdenMimicConfig):
    """G1 whole-body motion tracking trained with SuperMimic.

    ``motion_options`` is set by :class:`SuperMimicDataMod`; build the task with ``TASK_REGISTRY.build("supermimic")``.
    """

    scene_options: SceneOptions = en.scenes.Empty(robot=_G1, plane=PlaneOptions(default_geoms_friction={"*": 0.01}))

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=10.0,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,
        iterations=10,
        ls_iterations=20,
        max_collision_pairs=35,
        # MuJoCo's default contact time constant (solref 0.02 s), which the G1 MJCF keeps. Genesis' own 0.01 s
        # stiffens contact enough to launch kneeling and lying reference poses out of the floor at reset.
        constraint_timeconst=0.02,
        use_gjk_collision=False,
        batch_links_info=True,
        batch_dofs_info=False,
        enable_self_collision=True,
        enable_multi_contact=True,
        # The terminations and rewards replay from a CUDA graph; a CPU host computes them eagerly.
        cuda_graph_options=CudaGraphOptions(require_cuda=False),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        # The deployable proprioception. The runner adds its noise, so the group has none.
        policy=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=False,
            projected_gravity=observations.ProjectedGravity.configure(entity_name="robot"),
            base_ang_vel=ObsTerm.configure(
                func=observations.base_ang_vel, params={"entity_name": "robot", "frame": "body"}
            ),
            dofs_pos=DofsPosBiased.configure(entity_name="robot", action_term_name="joint_pos"),
            dofs_vel=observations.DofsVel.configure(entity_name="robot"),
            last_action=ObsTerm.configure(func=observations.last_action),
        ),
        full_state=ObservationGroupOptions(
            concatenate_terms=True,
            enable_corruption=False,
            root_joint_state=RootJointState.configure(entity_name="robot"),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        motion_global_root_pos=RewardTerm.configure(
            func=motion_link_position_tracking_exp, weight=0.5, params={**_tracked(ANCHOR_LINK), "std": 0.3}
        ),
        motion_global_root_ori=RewardTerm.configure(
            func=motion_link_orientation_tracking_exp, weight=0.5, params={**_tracked(ANCHOR_LINK), "std": 0.4}
        ),
        motion_body_pos=MotionRelativeBodyPositionErrorExp.configure(
            weight=1.0, entity_name="robot", anchor_link_name=ANCHOR_LINK, links_name=TRACKED_LINKS, std=0.3
        ),
        motion_body_ori=MotionRelativeBodyOrientationErrorExp.configure(
            weight=1.0, entity_name="robot", anchor_link_name=ANCHOR_LINK, links_name=TRACKED_LINKS, std=0.4
        ),
        motion_body_lin_vel=MotionGlobalBodyLinearVelocityErrorExp.configure(
            weight=1.0, entity_name="robot", links_name=TRACKED_LINKS, std=1.0
        ),
        motion_body_ang_vel=MotionGlobalBodyAngularVelocityErrorExp.configure(
            weight=1.0, entity_name="robot", links_name=TRACKED_LINKS, std=3.14
        ),
        action_rate_l2=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.1),
        joint_limit=RewardTerm.configure(func=rewards.dofs_pos_limits, weight=-2.0, params={"entity_name": "robot"}),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        clip_end=ReferenceClipEnd.configure(entity_name="robot", time_out=True),
        anchor_pos=DoneTerm.configure(
            func=bad_anchor_pos_z_only,
            params={"entity_name": "robot", "anchor_link_name": ANCHOR_LINK, "threshold": 0.25},
        ),
        anchor_ori=DoneTerm.configure(
            func=bad_anchor_ori,
            params={"entity_name": "robot", "anchor_link_name": ANCHOR_LINK, "threshold": 0.8},
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
            velocity_range={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
            dofs_position_range=(-0.1, 0.1),
            priority=100,
        ),
        foot_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["left_ankle_roll_link", "right_ankle_roll_link"],
            friction_range=(0.3, 1.2),
        ),
        base_com=events.RandomizeComShift.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["torso_link"],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.05, 0.05),
            com_shift_z_range=(-0.05, 0.05),
        ),
        encoder_bias=events.RandomizeDofsPosOffset.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            action_term_name="joint_pos",
            dofs_pos_range=(-0.01, 0.01),
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        joint_pos=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_G1.dofs_name,
            scale=G1_ACTION_SCALE,
            reference_source=ReferenceSource.DEFAULT,
        )
    )

    runner_options: SuperMimicRunnerOptions = SuperMimicRunnerOptions(experiment_name="supermimic")
