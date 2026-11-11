"""ConTrack: constrained hand-object motion tracking with an xArm7 + XHand.

Port of ConTrack (https://github.com/EmptyBlueBox/ConTrack, arXiv:2606.03177):
a fixed-base xArm7+XHand tracks a reference clip of robot joints, manipulated
object poses, and per-finger-link contact annotations. The rig is the
:class:`~eden.options.robots.composite.xarm_xhand.XArm7XHand1_R` composite, which welds
Eden's own xArm7 and XHand1 unit robots into one kinematic tree — it expands into the
``robot`` (arm, 7 DOFs) and ``hand`` (12 DOFs) entities, so every DOF-addressed term below
comes in an arm/hand pair. The policy outputs
residual joint targets around the reference qpos and is trained with the
external ``rsl_rl_contrack`` CMDP-PPO backend (installed separately; selected by
:class:`~eden.learning.wrappers.rsl_rl_contrack.options.RslRlContrackRunnerOptions`,
whose runner raises with the install command when it is missing) — rewards are declared in named reward groups
(task ``rg``, style ``rs``, penalty ``rp``; see ``RewardGroupOptions``), and a
dual controller balances task vs style advantages online.

Deliberate differences from the source implementation:

- Reset sampling uses Eden's RSI + adaptive bin sampler (kinematic reference
  states + failure-biased start bins) instead of ConTrack's cached-simulated
  -state reset library.
- Control runs at 50 Hz (Eden's default) instead of 60 Hz; the reference is
  retimed accordingly at dataset-prepare time.

Train (single clip per run, like the source):

``eden train --task contrack --data <path/to/clip.h5> --num_envs 8000``
"""

from __future__ import annotations
from typing import Annotated

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import torch

import eden as en
from eden import observations, terminations
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.motion_manager import MotionTerm
from eden.managers.terms.curricula.motion import MotionBasePoseThresholdCurriculum
from eden.managers.terms.events.motion import (
    ObjectReferenceStateInitialization,
    ReferenceStateInitialization,
)
from eden.managers.terms.observations.contact import GroupedContactFlags, motion_contact_flags
from eden.managers.terms.observations.tracking import (
    motion_future_base_pos_w,
    motion_future_base_quat_w,
    motion_future_dofs_pos,
    motion_phase,
)
from eden.managers.terms.rewards.contact import MotionContactDistanceReward, MotionContactMatchReward
from eden.managers.terms.rewards.tracking import (
    motion_base_orientation_tracking_exp,
    motion_base_position_tracking_exp,
    motion_dofs_pos_tracking_exp,
)
from eden.managers.terms.terminations.tracking import MotionBasePoseTrackingBreak, motion_ends
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.constants import ReferenceSource
from eden.tasks.contrack.dataset import (
    XARM_JOINT_NAMES,
    DexterHandDataset,
    contact_link_names,
    hand_joint_names,
)
from eden.learning.wrappers.rsl_rl_contrack import RslRlContrackRunnerOptions
from eden.options import ObservationGroupOptions, RewardGroupOptions
from eden.options.entities import EntityOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.managers.motions import MotionManagerOptions
from eden.options.robots.composite import XArm7XHand1_L, XArm7XHand1_R
from eden.options.scenes.empty import EmptySceneOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskMod
from eden.terms import DoneTerm, ObsTerm
from eden.terms import RewardTerm as RewardTermOptions
from eden.utils.assets import sample_data_path
from eden.utils.configs import (
    ActionManagerOptions,
    CurriculumManagerOptions,
    EdenMimicConfig,
    EnvOptions,
    EventManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
)
from eden.utils.motion_reference.lookup import get_robot_dofs_idx

# The clip in ``contrack/`` of the shared example-data repo that ``--data`` resolves when
# omitted. Fetched rather than read from ``examples/``, which is not packaged: on a pip-installed
# Eden that path would name a file inside ``site-packages/`` that cannot exist.
_DEFAULT_CLIP = "dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5"


# Reference frames shown to the policy: the current one and the next 5
# (ConTrack NUM_OBS_REFERENCE_POSE = 5).
_REF_FRAME_OFFSETS = (0, 1, 2, 3, 4, 5)

# ConTrack reward stds. Object position uses mean-over-3-dims squared error
# with sigma_p = 0.05 m; Eden's motion_base_position_tracking_exp uses the
# summed squared error, so the equivalent std is sigma_p * sqrt(3).
_OBJ_POS_STD = 0.05 * float(np.sqrt(3.0))
_OBJ_ROT_STD = 0.5
_DOFS_STD = 0.5
_CONTACT_DISTANCE_STD = 0.03

# ConTrack residual action scales (radians): arm 0.01, fingers 0.25.
_ACTION_SCALE_ARM = 0.01
_ACTION_SCALE_FINGER = 0.25


# ---------------------------------------------------------------------------
# ConTrack penalty terms (rp group): 1 - exp(-mean(x^2) / sigma^2) kernels
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def contrack_dofs_vel_penalty(env, *, entity_name: str, dofs_name: list[str], sigma: float = 1.0) -> torch.Tensor:
    """ConTrack joint-velocity penalty: ``1 - exp(-mean(v²) / sigma²)`` over the named DOFs."""
    idx = get_robot_dofs_idx(env, entity_name, dofs_name)
    v = env.entities[entity_name].get_dofs_vel(dofs_idx_local=idx)
    return 1.0 - torch.exp(-torch.mean(v * v, dim=-1) / sigma**2)


@REWARD_TERM_REGISTRY.register()
def contrack_object_vel_penalty(env, *, entity_name: str, angular: bool = False, sigma: float = 1.0) -> torch.Tensor:
    """ConTrack object-velocity penalty on the base linear (or angular) velocity."""
    entity = env.entities[entity_name]
    v = entity.get_ang() if angular else entity.get_vel()
    return 1.0 - torch.exp(-torch.mean(v * v, dim=-1) / sigma**2)


@REWARD_TERM_REGISTRY.register()
class ContrackDofsAccPenalty(RewardTerm):
    """ConTrack joint-acceleration penalty: ``1 - exp(-mean(a²) / sigma²)``.

    Accelerations are finite-differenced from the previous control step's
    joint velocities (a buffer this term owns). The first step after a reset
    reports zero acceleration — the previous velocity belongs to the old
    episode (ConTrack instead seeds it from the reset state; the one-step
    difference is negligible).
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []
    sigma: float = 50.0

    def build(self) -> None:
        self._dofs_idx = get_robot_dofs_idx(self._env, self.entity_name, self.dofs_name)
        self._prev_vel = torch.zeros(self._env.num_envs, self._dofs_idx.numel(), device=self.device)
        self._fresh = torch.ones(self._env.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        v = self._env.entities[self.entity_name].get_dofs_vel(dofs_idx_local=self._dofs_idx)
        a = (v - self._prev_vel) / float(self._env.dt)
        a = torch.where(self._fresh.unsqueeze(-1), torch.zeros_like(a), a)
        self._prev_vel.copy_(v)
        self._fresh.fill_(False)
        return 1.0 - torch.exp(-torch.mean(a * a, dim=-1) / self.sigma**2)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self._fresh[envs_idx if envs_idx is not None else slice(None)] = True


@REWARD_TERM_REGISTRY.register()
class ContrackObjectAccPenalty(RewardTerm):
    """ConTrack object-acceleration penalty on base linear (or angular) velocity FD."""

    entity_name: str = "object"
    angular: bool = False
    sigma: float = 50.0

    def build(self) -> None:
        self._prev_vel = torch.zeros(self._env.num_envs, 3, device=self.device)
        self._fresh = torch.ones(self._env.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        entity = self._env.entities[self.entity_name]
        v = entity.get_ang() if self.angular else entity.get_vel()
        a = (v - self._prev_vel) / float(self._env.dt)
        a = torch.where(self._fresh.unsqueeze(-1), torch.zeros_like(a), a)
        self._prev_vel.copy_(v)
        self._fresh.fill_(False)
        return 1.0 - torch.exp(-torch.mean(a * a, dim=-1) / self.sigma**2)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self._fresh[envs_idx if envs_idx is not None else slice(None)] = True


# ---------------------------------------------------------------------------
# Data-dependent configuration
# ---------------------------------------------------------------------------


def _export_object_mesh(data_path: Path, vertices: np.ndarray, faces: np.ndarray) -> str:
    """Write the clip's object mesh to a cached ``.obj`` next to the data file.

    The cache key includes a digest of the geometry, so editing a clip in place yields a new
    path instead of silently reusing the previous mesh. The write goes to a per-process temp
    file swapped in with :func:`os.replace`: every DDP rank runs the task mod, and a plain
    ``export(obj_path)`` lets one rank load a half-written file another rank is still writing.
    """
    import trimesh

    cache_dir = data_path.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    geometry = np.ascontiguousarray(vertices).tobytes() + np.ascontiguousarray(faces).tobytes()
    obj_path = cache_dir / f"{data_path.stem}_object0_{hashlib.sha1(geometry).hexdigest()[:12]}.obj"
    if not obj_path.exists():
        tmp_path = cache_dir / f".{obj_path.stem}.{os.getpid()}.tmp.obj"
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(tmp_path)
        os.replace(tmp_path, obj_path)  # atomic within the cache dir
    return str(obj_path)


class ContrackDataMod(TaskMod):
    """Wire a ConTrack ``.h5`` clip into the task config.

    Reads the clip once to determine handedness, export the object mesh, and
    size the episode; then builds the scene, motion, event, and episode-length
    options that depend on the data. One clip per training run (matching the
    source implementation — ConTrack trains per-clip specialists).

    Notes
    -----
    Unlike ``BeyondMimicDataMod``, an omitted ``--data`` is resolved *here* rather than
    deferred to the dataset, so building the config fetches the clip on a cold cache. That is
    forced by the reads above — handedness picks the L/R robot, so it has to be known before
    the entity options exist, which is long before any dataset is constructed. Two
    consequences worth knowing: building a ConTrack config needs the network the first time
    even for a run that would only have read the processed cache, and every DDP rank does the
    fetch (safe, but unbatched — ``_snapshot_download_locked`` serializes them on the
    huggingface_hub file lock rather than gating on rank 0).

    Making this lazy is the intended end state: it needs the handedness/mesh/frame-count reads
    moved to build time so the clip is only opened once an env is actually being constructed.
    Until then, pass ``--data`` explicitly on an air-gapped host.
    """

    def __init__(
        self,
        data: Annotated[
            str | None,
            "Path to a single ConTrack reference clip (.h5). Omit to fetch the default sample "
            "clip from HuggingFace (Kashu7100/eden_example, contrack/) on first use.",
        ] = None,
        target_framerate: Annotated[float, "Reference retiming rate (Hz); must match the control rate."] = 50.0,
    ):
        self.data = data
        self.target_framerate = target_framerate

    def apply(self, config: "ContrackConfig") -> "ContrackConfig":
        # ``is None``, not ``or``: an empty --data is a caller mistake to surface (the
        # FileNotFoundError below names it), not a silent request for the default clip.
        clip = sample_data_path("contrack", _DEFAULT_CLIP) if self.data is None else self.data
        data_path = Path(clip).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(
                f"--data must point at a single ConTrack clip (.h5); got {str(data_path)!r}. "
                f"Omit --data to fetch the default sample clip, or pass one of the four that ship "
                f"in a source checkout under examples/projects/mimic/contrack/sample/."
            )
        control_dt = config.env_options.sim_dt * config.env_options.decimation
        if abs(1.0 / self.target_framerate - control_dt) > 1e-9:
            raise ValueError(
                f"--target_framerate={self.target_framerate} Hz does not match the control rate "
                f"{1.0 / control_dt:.6g} Hz (sim_dt={config.env_options.sim_dt} x "
                f"decimation={config.env_options.decimation}). The reference is retimed to "
                f"target_framerate and the lookahead frames are spaced at 1/target_framerate, so a "
                f"mismatch aliases the reference against the control step."
            )
        with h5py.File(data_path, "r") as f:
            is_rhand = bool(np.asarray(f["is_rhand"], np.uint8)[0])
            base_translation = np.asarray(f["base_translation"], np.float32)[0]
            num_frames = int(f["qpos"].shape[0])
            fps = float(np.max(np.asarray(f["fps"], np.float32)))
            obj = f["object_tracks"]["0"]
            vertices = np.asarray(obj["vertices"], np.float32)
            faces = np.asarray(obj["faces"], np.uint32)
            first_pos = np.asarray(obj["translations"], np.float32)[0]
            first_quat_xyzw = np.asarray(obj["orientations_xyzw"], np.float32)[0]

        side = "right" if is_rhand else "left"
        robot_cls = XArm7XHand1_R if is_rhand else XArm7XHand1_L
        obj_path = _export_object_mesh(data_path, vertices, faces)

        # The composite expands into the ``robot`` (arm) + ``hand`` entities and welds them;
        # ``base_pos`` places the arm base at the clip's recorded mount position. The friction
        # is set here rather than on the rig because 2.0 is ConTrack's *task* physics (it must
        # match the object's), not a property of the robot — the composite stays reusable.
        robot = robot_cls(base_pos=tuple(float(x) for x in base_translation))
        robot.arm.material = RigidMaterialOptions(friction=2.0)
        robot.hand.material = RigidMaterialOptions(friction=2.0)
        config.scene_options = EmptySceneOptions(
            robot=robot,
            object=EntityOptions(
                file=obj_path,
                default_root_pos=tuple(float(x) for x in first_pos),
                default_root_quat=(
                    float(first_quat_xyzw[3]),
                    float(first_quat_xyzw[0]),
                    float(first_quat_xyzw[1]),
                    float(first_quat_xyzw[2]),
                ),
                # ConTrack object physics: density 200 kg/m^3, friction 2.0.
                material=RigidMaterialOptions(rho=200.0, friction=2.0),
            ),
        )

        # The clip duration bounds the episode; motion_ends truncates earlier
        # for mid-clip RSI starts.
        config.env_options.episode_length_s = num_frames / fps + 0.1

        arm_joints = list(XARM_JOINT_NAMES)
        hand_joints = hand_joint_names(side)
        contact_links = contact_link_names(side)
        config.motion_options = MotionManagerOptions(
            num_frames=len(_REF_FRAME_OFFSETS),
            data_start_from="current_time",
            frame_interval_s_range=(1.0 / self.target_framerate, 1.0 / self.target_framerate),
            default=MotionTerm.configure(
                dataset_options=DexterHandDataset.configure(
                    raw_data_path=str(data_path),
                    # The side is baked into the cache name: the loader validates
                    # is_rhand only on the prepare path, so a side-agnostic cache
                    # reused across configs could silently serve the wrong hand.
                    # ``armhand`` marks the sequence layout (arm/hand split across the
                    # composite's two entities), so a cache of another layout cannot
                    # collide with it (a collision would fail only at load time).
                    processed_data_path=str(
                        data_path.parent / ".cache" / f"{data_path.stem}_{side}_armhand_processed.pt"
                    ),
                    target_framerate=self.target_framerate,
                    side=side,
                ),
                entities_map={"robot": "robot", "hand": "hand", "object": "object"},
                target_dofs_name={"robot": arm_joints, "hand": hand_joints, "object": []},
                target_links_name={"robot": [], "hand": contact_links, "object": []},
                # Failure-biased start-bin sampling stands in for ConTrack's
                # reset-library difficulty sampling (bin weights control the
                # start distribution; no motion_start_from_middle_range needed).
                motion_bin_length_s=0.5,
                adaptive_sampling=True,
            ),
        )

        # Data-dependent term parameters (handedness changes all joint/link names).
        # Term order fixes the observation layout, and each DOF-addressed signal is an
        # ``arm_``/``hand_`` pair ordered arm-then-hand — the clip's qpos column order
        # (ConTrack's flat 19-vector), though the two halves live on separate entities.
        common_obs = dict(
            arm_dofs_pos=observations.DofsPos.configure(entity_name="robot"),
            hand_dofs_pos=observations.DofsPos.configure(entity_name="hand"),
            object_pos=ObsTerm.configure(func=observations.base_pos, params={"entity_name": "object"}),
            object_quat=ObsTerm.configure(func=observations.base_quat, params={"entity_name": "object"}),
            arm_dofs_vel=observations.DofsVel.configure(entity_name="robot"),
            hand_dofs_vel=observations.DofsVel.configure(entity_name="hand"),
            object_lin_vel=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "object"}),
            object_ang_vel=ObsTerm.configure(func=observations.base_ang_vel, params={"entity_name": "object"}),
            hand_contact=GroupedContactFlags.configure(
                entity_name="hand",
                object_entity_name="object",
                links_name=contact_links,
            ),
            ref_arm_dofs_pos=ObsTerm.configure(
                func=motion_future_dofs_pos,
                params={"entity_name": "robot", "frame_offsets": list(_REF_FRAME_OFFSETS)},
            ),
            ref_hand_dofs_pos=ObsTerm.configure(
                func=motion_future_dofs_pos,
                params={"entity_name": "hand", "frame_offsets": list(_REF_FRAME_OFFSETS)},
            ),
            ref_object_pos=ObsTerm.configure(
                func=motion_future_base_pos_w,
                params={"entity_name": "object", "frame_offsets": list(_REF_FRAME_OFFSETS)},
            ),
            ref_object_quat=ObsTerm.configure(
                func=motion_future_base_quat_w,
                params={"entity_name": "object", "frame_offsets": list(_REF_FRAME_OFFSETS)},
            ),
            ref_hand_contact=ObsTerm.configure(
                func=motion_contact_flags,
                params={"entity_name": "hand", "links_name": contact_links},
            ),
            phase=ObsTerm.configure(func=motion_phase, params={"entity_name": "robot"}),
        )
        config.observation_options = ObservationManagerOptions(
            policy=ObservationGroupOptions(concatenate_terms=True, enable_corruption=True, **common_obs),
            critic=ObservationGroupOptions(concatenate_terms=True, enable_corruption=False, **common_obs),
        )

        # ConTrack's CMDP reward channels as first-class reward groups: the
        # wrapper reads reward_manager.get_group_reward("rg"/"rs"/"rp") per step.
        config.reward_options = RewardManagerOptions(
            # rg: object tracking (task)
            rg=RewardGroupOptions(
                tracking_obj_pos=RewardTermOptions.configure(
                    func=motion_base_position_tracking_exp,
                    weight=500.0,
                    params={"entity_name": "object", "std": _OBJ_POS_STD},
                ),
                tracking_obj_rot=RewardTermOptions.configure(
                    func=motion_base_orientation_tracking_exp,
                    weight=500.0,
                    params={"entity_name": "object", "std": _OBJ_ROT_STD},
                ),
            ),
            # rs: joint + contact imitation (style)
            rs=RewardGroupOptions(
                tracking_arm=RewardTermOptions.configure(
                    func=motion_dofs_pos_tracking_exp,
                    weight=10.0,
                    params={"entity_name": "robot", "std": _DOFS_STD, "dofs_name": arm_joints, "reduction": "mean"},
                ),
                tracking_finger=RewardTermOptions.configure(
                    func=motion_dofs_pos_tracking_exp,
                    weight=10.0,
                    params={"entity_name": "hand", "std": _DOFS_STD, "dofs_name": hand_joints, "reduction": "mean"},
                ),
                contact_match=MotionContactMatchReward.configure(
                    weight=100.0,
                    hand_entity_name="hand",
                    object_entity_name="object",
                    hand_links_name=contact_links,
                ),
                contact_distance=MotionContactDistanceReward.configure(
                    weight=100.0,
                    hand_entity_name="hand",
                    object_entity_name="object",
                    hand_links_name=contact_links,
                    std=_CONTACT_DISTANCE_STD,
                ),
            ),
            # rp: smoothness penalties
            rp=RewardGroupOptions(
                qvel_penalty_arm=RewardTermOptions.configure(
                    func=contrack_dofs_vel_penalty,
                    weight=-1.0,
                    params={"entity_name": "robot", "dofs_name": arm_joints},
                ),
                qvel_penalty_finger=RewardTermOptions.configure(
                    func=contrack_dofs_vel_penalty,
                    weight=-1.0,
                    params={"entity_name": "hand", "dofs_name": hand_joints},
                ),
                qvel_penalty_obj_pos=RewardTermOptions.configure(
                    func=contrack_object_vel_penalty,
                    weight=-1.0,
                    params={"entity_name": "object", "angular": False},
                ),
                qvel_penalty_obj_rot=RewardTermOptions.configure(
                    func=contrack_object_vel_penalty,
                    weight=-0.1,
                    params={"entity_name": "object", "angular": True},
                ),
                qacc_penalty_arm=ContrackDofsAccPenalty.configure(
                    weight=-10.0,
                    entity_name="robot",
                    dofs_name=arm_joints,
                ),
                qacc_penalty_finger=ContrackDofsAccPenalty.configure(
                    weight=-10.0,
                    entity_name="hand",
                    dofs_name=hand_joints,
                ),
                qacc_penalty_obj_pos=ContrackObjectAccPenalty.configure(
                    weight=-100.0,
                    entity_name="object",
                    angular=False,
                ),
                qacc_penalty_obj_rot=ContrackObjectAccPenalty.configure(
                    weight=-10.0,
                    entity_name="object",
                    angular=True,
                ),
            ),
        )

        # Two PD terms (one per entity) instead of ConTrack's single 19-DOF actuator group.
        # Term order sets the action layout: 7 arm residuals then 12 finger residuals — the
        # same ordering as the source's flat action vector.
        config.action_options = ActionManagerOptions(
            residual_arm_dofs_pos=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=arm_joints,
                scale={name: _ACTION_SCALE_ARM for name in arm_joints},
                reference_source=ReferenceSource.MOTION_REF,
            ),
            residual_hand_dofs_pos=ImplicitPDController.configure(
                entity_name="hand",
                dofs_name=hand_joints,
                scale={name: _ACTION_SCALE_FINGER for name in hand_joints},
                reference_source=ReferenceSource.MOTION_REF,
            ),
        )

        return config


@TASK_REGISTRY.register(modifiers=(ContrackDataMod(),))
class ContrackConfig(EdenMimicConfig):
    """ConTrack hand-object tracking (xArm7 + XHand, CMDP-PPO).

    Data-dependent options (scene, motion, observations, rewards, actions) are
    built by :class:`ContrackDataMod` — construct via
    ``TASK_REGISTRY.build("contrack", data=...)``.

    Its ``runner_options`` select ConTrack's CMDP-PPO through the external
    ``rsl_rl_contrack`` backend (:mod:`eden.learning.wrappers.rsl_rl_contrack`), so
    ``eden train --task contrack`` trains it once the backend is installed;
    ``eden smoke --task contrack`` builds and steps the env without it.
    """

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=5.0,  # overwritten per clip by ContrackDataMod
        sim_dt=0.005,
        sim_substeps=2,
        decimation=4,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
        # Load-bearing for in-hand manipulation: without spin resistance at the contacts the
        # object slowly rotates inside the grasp, and the policy cannot hold the reference
        # orientation. Genesis made torsional friction an opt-in flag in 1.2.2 (default off);
        # before that its friction model resisted spin implicitly. Measured on Cylinder_00 at
        # 450 iterations, 8000 envs, off -> on: mean reward 865 -> 1297, episode length 46 -> 67.
        enable_torsional_friction=True,
        batch_dofs_info=False,
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        motion_ends=DoneTerm.configure(
            func=motion_ends,
            params={"entity_name": "robot"},
            time_out=True,
        ),
        # Starts at ConTrack's lenient tolerance; the curriculum below tightens it to
        # (0.05 m, 0.5 rad) as the success EMA rises.
        object_pose_break=MotionBasePoseTrackingBreak.configure(
            entity_name="object",
            pos_threshold=0.1,
            rot_threshold=1.0,
        ),
    )

    # ConTrack's threshold curriculum. It owns the success EMA (not the termination term)
    # so the schedule is logged under ``Curriculum/object_pose_threshold/*`` and frozen
    # during eval rollouts.
    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        object_pose_threshold=MotionBasePoseThresholdCurriculum.configure(
            termination_term_name="object_pose_break",
            entity_name="object",
            pos_threshold_max=0.1,
            rot_threshold_max=1.0,
            pos_threshold_min=0.05,
            rot_threshold_min=0.5,
            success_ema_alpha=0.05,
            success_rate_low=0.9,
            success_rate_high=0.999,
        ),
    )

    # One RSI per robot entity: the arm's joints and the welded hand's fingers are separate
    # entities, and both are base-immobile (the hand's root joint is removed by the weld), so
    # each term writes only its own DOFs.
    event_options: EventManagerOptions = EventManagerOptions(
        reset_arm_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            dofs_position_range=(0.0, 0.0),
            priority=100,
        ),
        reset_hand_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="hand",
            dofs_position_range=(0.0, 0.0),
            priority=100,
        ),
        reset_object_from_motion=ObjectReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="object",
            xy_range=0.0,
            priority=99,
        ),
    )

    # ConTrack's CMDP-PPO (the release hyperparameters). The runner its class_name names
    # needs the external `rsl_rl_contrack` backend and says how to install it.
    runner_options: RslRlContrackRunnerOptions = RslRlContrackRunnerOptions()
