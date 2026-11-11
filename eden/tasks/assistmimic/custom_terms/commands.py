"""Paired reference motion for AssistMimic's caregiver and recipient (Inter-X help-up or HHI-Assist bed).

One Eden env holds both humanoids (AssistMimic pairs two Isaac Gym envs instead), so a single command term
owns the clip assignment, the reference clock and the reset of both — including AssistMimic's PSI, which
restarts episodes from simulated states of good past rollouts.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.managers.command_manager import COMMAND_TERM_REGISTRY, CommandTerm
from eden.options.robots.smpl_humanoid import SMPL_BODY_NAMES, SMPL_DOF_NAMES
from eden.tasks.assistmimic.custom_terms.psi import PsiTrajectoryBuffer
from eden.utils.assets import sample_data_path
from eden.utils.geom import (
    axis_angle_from_quat,
    intrinsic_xyz_from_quat,
    intrinsic_xyz_nearest,
    intrinsic_xyz_rates,
    inv_quat,
    quat_mul,
    quat_slerp,
)
from eden.utils.misc import sanitize_envs_idx

ROLES = ("caregiver", "recipient")
PARTNER = {"caregiver": "recipient", "recipient": "caregiver"}
NUM_BODIES = len(SMPL_BODY_NAMES)
NUM_DOFS = len(SMPL_DOF_NAMES)
#: Generalized coordinates of one humanoid: free-joint position + wxyz quaternion, then the hinges.
NQ = 7 + NUM_DOFS
#: Its velocity coordinates: free-joint linear + angular velocity, then the hinges.
NV = 6 + NUM_DOFS
#: ``resampling_time_range`` that never fires: the clip, not a timer, decides when the command changes.
NEVER_RESAMPLE = (1.0e9, 1.0e9)


@dataclass
class BodyRef:
    """Reference body state of one role at one time per env: ``(N, 24, *)`` in world frame, quats wxyz."""

    pos: torch.Tensor
    rot: torch.Tensor
    vel: torch.Tensor
    ang_vel: torch.Tensor

    @classmethod
    def zeros(cls, num_envs: int, device) -> BodyRef:
        """An all-zero ``BodyRef`` of ``num_envs`` rows, to fill in place."""
        return cls(*(torch.zeros(num_envs, NUM_BODIES, k, device=device) for k in (3, 4, 3, 3)))


@COMMAND_TERM_REGISTRY.register()
class AssistMimicPairMotion(CommandTerm):
    """Clip assignment, reference clock and reference-state reset for a caregiver/recipient pair.

    Loads the ``.npz`` written by ``scripts/convert_pair_motion.py``. Each reset draws a clip uniformly for the
    env and puts both humanoids on its first frame (AssistMimic's ``stateInit: Start``). The reference is read
    at ``t = reference_frame * dt`` for grading (after ``episode_length`` has advanced past the step) and at
    ``t + dt`` for the observation, as PHC does. ``reference_frame`` is the episode length plus the frame a
    :meth:`restore_state` restarted the env at (0 after a plain reset), so a restored episode reads its clip
    from that frame on while the env's own episode length still counts the steps it has run.

    **PSI** (``psi_ratio > 0``, training envs only): the pair state is recorded after every control step; a
    reset stores the ended episode's trajectory when ``max`` of the two roles' returns — the ``caregiver`` and
    ``recipient`` reward groups' episode sums, per control step — reaches ``psi_return_threshold``, and restarts
    each reset env with probability ``psi_ratio`` from a stored state of its new clip instead of the clip start
    (:class:`~eden.tasks.assistmimic.custom_terms.psi.PsiTrajectoryBuffer`). It all happens inside the env's
    reset, so the observations a step returns already show the restarted pair, whatever the runner. An
    ``eval_mode`` env always starts at the clip start.

    Every reference array is indexed by the ``.npz``'s ``body_names`` while the terms read the simulated bodies
    by link index, so ``build()`` requires both humanoids' links to be exactly those bodies in that order.

    Parameters
    ----------
    motion_file : str
        The converted ``.npz``. Empty: the task's published ``sample_clip``.
    sample_clip : str
        File under ``assistmimic/`` of the shared sample-data repo, fetched on first use when ``motion_file``
        is empty.
    caregiver_entity_name : str
        Entity playing the caregiver.
    recipient_entity_name : str
        Entity playing the recipient.
    recipient_offset, caregiver_offset : tuple[float, float, float]
        World offset added to that role's whole reference track, for the diagnostics that need the pair out
        of each other's reach: ``main.py adapt`` moves the recipient aside so each humanoid learns plain
        imitation, and ``main.py play --no-caregiver`` moves the *caregiver* aside — leaving the recipient
        and its bed exactly where the clip puts them — to ask whether the recipient can get up unaided.
    fixed_clip : int
        Play only this clip index (``-1``: draw a clip uniformly at every reset).
    bed_entity_name : str
        Furniture the recipient starts on (HHI-Assist). When set, ``build()`` checks every clip's reset pose
        against the *built* box, so a bed in the wrong place is a construction error rather than a humanoid
        that explodes out of the mattress on step 1. Empty for Inter-X, which has no furniture.
    bed_penetration_tol : float
        How far [m] a reference body may sit below the bed's top face at ``t = 0`` before ``build()`` raises.
    lock_band : float
        ``|cos b|`` below which the reset pose's hinge angles and velocities are damped near gimbal lock
        (:func:`~eden.utils.geom.intrinsic_xyz_nearest`, :func:`~eden.utils.geom.intrinsic_xyz_rates`): a knee
        starting near 90 deg and moving would otherwise spin its x and z hinges at tens of rad/s. 0: exact.
    psi_ratio : float
        Probability that a reset restarts from a stored trajectory of its clip (AssistMimic: 0.5); 0 turns PSI
        off and allocates nothing. The buffer's live recording takes ``num_envs x (longest clip in control
        steps + 2) x 302`` float32 — about 1 GB per 100 frames at 8192 envs.
    psi_return_threshold : float
        Minimum ``max`` of the two roles' episode returns (summed per-step rewards) to store a trajectory.
    psi_buffer_size : int
        Trajectories kept per clip, FIFO (AssistMimic ``trajectory_buffer_size``).
    psi_tail_margin : int
        Frames dropped from the end of an episode that ended by early termination.
    psi_max_store_per_step : int
        Most trajectories stored in one reset.
    resampling_time_range : tuple[float, float]
        Must stay :data:`NEVER_RESAMPLE`: a clip is resampled only at reset, and ``build()`` refuses a timer.
    """

    motion_file: str = ""
    sample_clip: str = ""
    caregiver_entity_name: str = "caregiver"
    recipient_entity_name: str = "recipient"
    recipient_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    caregiver_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bed_entity_name: str = ""
    bed_penetration_tol: float = 0.05
    fixed_clip: int = -1
    lock_band: float = 0.3
    psi_ratio: float = 0.0
    psi_return_threshold: float = 300.0
    psi_buffer_size: int = 10
    psi_tail_margin: int = 10
    psi_max_store_per_step: int = 128
    resampling_time_range: tuple[float, float] = NEVER_RESAMPLE

    def build(self) -> None:
        if min(self.resampling_time_range) < NEVER_RESAMPLE[0]:
            raise ValueError(
                f"AssistMimicPairMotion: resampling_time_range={self.resampling_time_range}, but the pair's clip "
                "may change only at a reset. A timed resample would teleport both humanoids mid-episode and move "
                "the reference clock (_frame_offset) while the episode length keeps counting, so every step after "
                f"it would be graded against the wrong frame. Leave it at {NEVER_RESAMPLE}; a clip's end is "
                "handled by the ReferenceMotionEnd termination."
            )
        self._motion_path = motion_file = self._resolve_motion_file()
        self._entity_names = {"caregiver": self.caregiver_entity_name, "recipient": self.recipient_entity_name}
        for name in self._entity_names.values():
            entity = self.resolve_entity(name)
            if entity.n_dofs != NV:
                raise ValueError(
                    f"AssistMimicPairMotion: entity '{name}' has {entity.n_dofs} DOFs; the SMPL reference needs a free "
                    f"base plus {NUM_DOFS} hinge DOFs (eden.options.robots.smpl_humanoid.PhcSmplHumanoid)."
                )
        data = np.load(motion_file)
        self.clip_names = [str(c) for c in data["clips"]]
        self.body_names = [str(b) for b in data["body_names"]]
        for name in self._entity_names.values():
            links = [link.name for link in self.resolve_entity(name).links]
            if links != self.body_names:
                i = next(
                    (i for i, (a, b) in enumerate(zip(links, self.body_names)) if a != b),
                    min(len(links), len(self.body_names)),
                )
                link_i = repr(links[i]) if i < len(links) else "nothing"
                body_i = repr(self.body_names[i]) if i < len(self.body_names) else "nothing"
                raise ValueError(
                    f"AssistMimicPairMotion: entity '{name}' has link {link_i} at index {i} where '{motion_file}' has "
                    f"body {body_i} (links: {links}; body_names: {self.body_names}). Every reference array is "
                    "indexed by body_names and every term reads the simulated bodies by link index, so the two must be "
                    "the same bodies in the same order. Use PhcSmplHumanoid's MJCF and a clip converted against it "
                    "(scripts/convert_pair_motion.py); reordering one side would grade every body against another's "
                    "reference."
                )
        self._head_idx = self.body_names.index("Head")
        if self.fixed_clip >= len(self.clip_names):
            raise ValueError(
                f"AssistMimicPairMotion: fixed_clip={self.fixed_clip} but '{motion_file}' has {len(self.clip_names)} "
                f"clips: {self.clip_names}. Pass an index in [0, {len(self.clip_names) - 1}] or -1 for all."
            )
        if not 0.0 <= self.lock_band < 1.0:
            raise ValueError(
                f"AssistMimicPairMotion: lock_band={self.lock_band} bounds |cos b|, so it must be in [0, 1); 0 keeps "
                "the exact hinge angles and velocities, the default 0.3 damps within 17 deg of gimbal lock."
            )
        dev = self.device
        short = [(c, int(n)) for c, n in zip(self.clip_names, data["num_frames"]) if n < 2]
        if short:
            raise ValueError(
                f"AssistMimicPairMotion: clips {short} of '{motion_file}' have fewer than 2 frames (clip, frames); a "
                "clip needs a duration to be played and a next frame to take velocities from. Drop them when "
                "converting (scripts/convert_pair_motion.py)."
            )
        num_frames = torch.as_tensor(data["num_frames"], device=dev)
        fps = torch.as_tensor(data["fps"], device=dev)
        self.num_clips = len(self.clip_names)
        self._clip_num_frames = num_frames
        self._clip_dt = 1.0 / fps
        self.clip_lengths = (num_frames - 1).float() / fps  # PHC: (num_frames - 1) / fps, per clip
        self._clip_start = torch.cumsum(num_frames, 0) - num_frames
        self._ref = {}
        for role in ROLES:
            arrays = {k: torch.as_tensor(data[f"{role}_{k}"], device=dev) for k in ("gts", "grs", "lrs", "gvs", "gavs")}
            # Genesis drives the three hinges of a body as intrinsic X-Y-Z Euler angles (see robots.py). Each
            # frame takes the decomposition nearest the zero pose (a knee bent 2.5 rad is (0, 2.5, 0), not
            # (pi, 0.64, pi), which sits on the hinges' +-180 deg limits). The hinge velocities are each body's
            # turn to the next frame mapped through the angle rates, not differenced angles, which swing far for
            # a small turn near 90 deg about y; both are damped there (lock_band).
            lrs = arrays["lrs"][:, 1:]  # (frames, 23, 4)
            euler = intrinsic_xyz_from_quat(lrs)
            euler = intrinsic_xyz_nearest(euler, torch.zeros_like(euler), self.lock_band)
            turn = axis_angle_from_quat(quat_mul(inv_quat(lrs[:-1]), lrs[1:]))  # body frame, frame f -> f + 1
            euler_vel = torch.zeros_like(euler)
            for c in range(self.num_clips):
                s, n = int(self._clip_start[c]), int(num_frames[c])
                e = s + n - 1
                euler_vel[s:e] = intrinsic_xyz_rates(euler[s:e], turn[s:e] * fps[c], self.lock_band)
                euler_vel[e] = euler_vel[e - 1]
            euler, euler_vel = euler.flatten(1), euler_vel.flatten(1)
            offset = self.recipient_offset if role == "recipient" else self.caregiver_offset
            arrays["gts"] = arrays["gts"] + torch.tensor(offset, device=dev)
            arrays["euler"] = euler
            arrays["euler_vel"] = euler_vel
            self._ref[role] = arrays

        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        # Reference frame the env's episode started at: 0 after a reset, the restored frame after restore_state.
        self._frame_offset = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.grade_ref = {role: BodyRef.zeros(self.num_envs, dev) for role in ROLES}
        self.obs_ref = {role: BodyRef.zeros(self.num_envs, dev) for role in ROLES}
        self._qs_idx = torch.arange(NQ, device=dev)
        self._dofs_idx = torch.arange(NV, device=dev)
        if self.bed_entity_name:
            self._check_bed_support()
        self.psi = self._build_psi()

    def _build_psi(self) -> PsiTrajectoryBuffer | None:
        """The PSI buffer, or ``None`` when ``psi_ratio`` is 0 or the env is in eval mode; raise on bad settings."""
        if not 0.0 <= self.psi_ratio <= 1.0:
            raise ValueError(
                f"AssistMimicPairMotion: psi_ratio={self.psi_ratio} is the probability that a reset restarts from a "
                "stored trajectory, so it must be in [0, 1]; 0 turns PSI off, AssistMimic trains with 0.5."
            )
        if self.psi_buffer_size < 1 or self.psi_max_store_per_step < 1 or self.psi_tail_margin < 0:
            raise ValueError(
                f"AssistMimicPairMotion: psi_buffer_size={self.psi_buffer_size} and psi_max_store_per_step="
                f"{self.psi_max_store_per_step} must be >= 1 and psi_tail_margin={self.psi_tail_margin} >= 0. To "
                "turn PSI off, set psi_ratio=0 instead."
            )
        if self.psi_ratio == 0.0 or not self._env.training:
            return None
        # Frames are clip steps: a restarted episode starts mid-clip and may run on to its clip's end, past
        # max_episode_length, so the longest clip bounds them (+1 for the end frame, +1 for rounding).
        max_len = math.ceil(float(self.clip_lengths.max()) / self._env.dt) + 2
        return PsiTrajectoryBuffer(
            self.num_envs,
            self.num_clips,
            self.state_dim,
            max_len,
            self.device,
            buffer_size=self.psi_buffer_size,
            return_threshold=self.psi_return_threshold,
            usage_ratio=self.psi_ratio,
            tail_margin=self.psi_tail_margin,
            max_store_per_step=self.psi_max_store_per_step,
        )

    def _resolve_motion_file(self) -> str:
        """``motion_file``, or the task's published ``sample_clip`` when it is empty; raise if neither exists."""
        convert = (
            "Convert an AssistMimic pkl with `python examples/projects/mimic/assistmimic/scripts/convert_pair_motion.py "
            "--dataset <inter-x|hhi-assist> --pkl <pkl> --smpl-dir <data/smpl> -o <name>.npz` and pass it as "
            "motion_file (`main.py --motion-file <name>.npz`); see examples/projects/mimic/assistmimic/README.md."
        )
        if self.motion_file:
            if not os.path.isfile(self.motion_file):
                raise FileNotFoundError(
                    f"AssistMimicPairMotion: motion_file '{self.motion_file}' does not exist. {convert}"
                )
            return self.motion_file
        if not self.sample_clip:
            raise ValueError(f"AssistMimicPairMotion: set motion_file (or sample_clip). {convert}")
        try:
            return sample_data_path("assistmimic", self.sample_clip)
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"AssistMimicPairMotion: motion_file is empty and the sample clip 'assistmimic/{self.sample_clip}' is not "
                f"in the shared sample-data repo. {convert}"
            ) from err

    def _check_bed_support(self) -> None:
        """Raise unless every clip's recipient reset pose rests on top of the bed.

        Its root (pelvis) must start over the bed's footprint — limbs may hang over the edge — and no body over
        the footprint may start inside it. Read off the *built* box (``get_AABB``) rather than the config, so
        moving the bed in the scene or converting the motion at the wrong height is caught here either way. The
        trap this exists for: the two datasets are played at different root translations — HHI-Assist at
        ``trans``, Inter-X at ``root_trans_offset`` — and picking the wrong one drops the recipient ~0.35 m, the
        height of the bed.
        """
        aabb = self.resolve_entity(self.bed_entity_name).get_AABB()[0]  # (2, 3) world-frame corners
        lo, hi = aabb[0], aabb[1]
        footprint = f"footprint x=[{float(lo[0]):.3f}, {float(hi[0]):.3f}] y=[{float(lo[1]):.3f}, {float(hi[1]):.3f}]"
        pos = self._ref["recipient"]["gts"][self._clip_start]  # (num_clips, 24, 3), reset pose per clip
        over_bed = ((pos[..., :2] >= lo[:2]) & (pos[..., :2] <= hi[:2])).all(dim=-1)  # (num_clips, 24)
        off = (~over_bed[:, 0]).nonzero()[:, 0]  # body 0 is the root: build() matched body_names to the links
        if off.numel():
            lines = "\n".join(
                f"  {self.clip_names[c]}: {self.body_names[0]} at x={float(pos[c, 0, 0]):.3f} y={float(pos[c, 0, 1]):.3f}"
                for c in off.tolist()[:5]
            )
            raise ValueError(
                f"AssistMimicPairMotion: the recipient's reset pose is not over the bed '{self.bed_entity_name}' "
                f"({footprint}) in {off.numel()} of {self.num_clips} clips of "
                f"'{self._motion_path}':\n{lines}\n"
                "The recipient would start beside the bed and fall to the floor. Move the bed in the scene config "
                "together with any recipient_offset (assistmimic.config.separate_pair moves both), or, for a clip "
                "that has no bed, use the `inter-x` task (bed_entity_name=''). Do not clear bed_entity_name on "
                "the bed task to silence this: the clip would still start the recipient in mid-air."
            )
        depth = torch.where(over_bed, hi[2] - pos[..., 2], torch.full_like(pos[..., 2], -torch.inf))
        worst, body = depth.max(dim=1)  # deepest body of each clip over the footprint, (num_clips,)
        bad = (worst > self.bed_penetration_tol).nonzero()[:, 0]
        if bad.numel():
            lines = "\n".join(
                f"  {self.clip_names[c]}: {self.body_names[int(body[c])]} is {float(worst[c]):.3f} m below the top face"
                for c in bad.tolist()[:5]
            )
            raise ValueError(
                f"AssistMimicPairMotion: the recipient's reset pose starts inside the bed '{self.bed_entity_name}' "
                f"(top face z={float(hi[2]):.4f}, {footprint}) in {bad.numel()} of {self.num_clips} clips of "
                f"'{self._motion_path}':\n{lines}\n"
                "Convert the motion with `--dataset hhi-assist` (the AssistMimic task that reads a bed dataset "
                "plays it at `trans`, and `--dataset inter-x` plays it at `root_trans_offset`, ~0.35 m lower), "
                "or move the bed in the scene config. Do not raise bed_penetration_tol to silence this: the "
                "recipient would be ejected from the mattress on the first step and every clip would fail."
            )

    @property
    def command(self) -> torch.Tensor:
        """The clip id of each env, as a float column."""
        return self.clip_id[:, None].float()

    @property
    def clip_length(self) -> torch.Tensor:
        """Duration of each env's clip in seconds, ``(N,)``."""
        return self.clip_lengths[self.clip_id]

    def start_head_height(self, role: str) -> torch.Tensor:
        """Height of ``role``'s head on the first frame of every clip, ``(num_clips,)``."""
        return self._ref[role]["gts"][self._clip_start, self._head_idx, 2]

    def peak_head_height(self, role: str) -> torch.Tensor:
        """Highest ``role``'s head gets anywhere in each clip, ``(num_clips,)``.

        What "helped up" means is the clip's own answer, not a constant: Inter-X help-up ends with the
        recipient standing (head ~1.6 m), HHI-Assist ends with them sitting on the edge of the bed, and four
        of its ten clips never pass 1.2 m at all — so a fixed threshold borrowed from one dataset scores the
        other as a total failure however well it tracks.
        """
        heights = self._ref[role]["gts"][:, self._head_idx, 2]
        return torch.stack(
            [heights[s : s + n].max() for s, n in zip(self._clip_start.tolist(), self._clip_num_frames.tolist())]
        )

    @property
    def reference_frame(self) -> torch.Tensor:
        """Control step of each env's clip the pair is at, ``(N,)`` long: episode length + restored frame."""
        return self._env.episode_length_buf + self._frame_offset

    def motion_time(self, lookahead_steps: int = 0) -> torch.Tensor:
        """Reference time of each env, ``(N,)``: ``(reference_frame + lookahead) * dt`` (start time is 0)."""
        return (self.reference_frame.float() + lookahead_steps) * self._env.dt

    def body_ref(self, role: str, time: torch.Tensor) -> BodyRef:
        """Interpolated reference body state of ``role`` at ``time`` (PHC ``MotionLibBase.get_motion_state``)."""
        f0, f1, blend = self._frame_blend(time)
        arr = self._ref[role]
        b = blend[:, None, None]
        return BodyRef(
            pos=(1 - b) * arr["gts"][f0] + b * arr["gts"][f1],
            rot=quat_slerp(arr["grs"][f0], arr["grs"][f1], b),
            vel=(1 - b) * arr["gvs"][f0] + b * arr["gvs"][f1],
            ang_vel=(1 - b) * arr["gavs"][f0] + b * arr["gavs"][f1],
        )

    def _frame_blend(self, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = self.clip_lengths[self.clip_id]
        n_frames = self._clip_num_frames[self.clip_id]
        dt = self._clip_dt[self.clip_id]
        phase = (time / length).clamp(0.0, 1.0)
        f0 = (phase * (n_frames - 1)).long()
        f1 = torch.minimum(f0 + 1, n_frames - 1)
        blend = ((time.clamp(min=0.0) - f0 * dt) / dt).clamp(0.0, 1.0)
        start = self._clip_start[self.clip_id]
        return f0 + start, f1 + start, blend

    def _refresh(self, cache: dict[str, BodyRef], time: torch.Tensor) -> None:
        for role in ROLES:
            ref = self.body_ref(role, time)
            for field in ("pos", "rot", "vel", "ang_vel"):
                getattr(cache[role], field).copy_(getattr(ref, field))

    def post_physics_refresh(self) -> None:
        self._refresh(self.grade_ref, self.motion_time())
        if self.psi is not None:
            self.psi.record(self.capture_state(), self.reference_frame)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.psi is not None:
            # The command manager resets before the reward and termination managers, and the episode clock is
            # zeroed after every manager: the ended episodes' returns, termination flags and last frame are
            # still readable here.
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            done[sanitize_envs_idx(envs_idx, self.num_envs)] = True
            rm = self._env.reward_manager
            returns = torch.stack([rm.get_group_episode_sum(role, raw=False) for role in ROLES], dim=1) / self._env.dt
            terminated = self._env.termination_manager.terminated
            self.psi.end_episodes(returns, done, terminated, self.reference_frame, self.clip_id)
        return super().reset(envs_idx)

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        self._refresh(self.obs_ref, self.motion_time(1))

    def _update_command(self) -> None:
        self._refresh(self.obs_ref, self.motion_time(1))

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        if self.fixed_clip >= 0:
            new_clip = torch.full((self.num_envs,), self.fixed_clip, device=self.device, dtype=torch.long)
        else:
            new_clip = torch.randint(0, self.num_clips, (self.num_envs,), device=self.device)
        self.clip_id.copy_(torch.where(envs_idx, new_clip, self.clip_id))
        # Every env's clip-start state, written through the mask: an index list would cost a host sync per reset.
        state = self._clip_start_state()
        frame = torch.zeros_like(self._frame_offset)
        if self.psi is not None:
            restart, stored, stored_frame = self.psi.sample_restarts(envs_idx, self.clip_id)
            state = torch.where(restart[:, None], stored, state)
            frame = torch.where(restart, stored_frame, frame)
            self.psi.start(envs_idx, state, frame)
        self._write_state(envs_idx, state)
        self._frame_offset.copy_(torch.where(envs_idx, frame, self._frame_offset))

    def _clip_start_state(self) -> torch.Tensor:
        """Every env's pair state on the first frame of its clip, laid out as :meth:`capture_state`."""
        frame = self._clip_start[self.clip_id]
        parts = []
        for role in ROLES:
            arr = self._ref[role]
            root_rot = arr["grs"][frame, 0]
            # Genesis keeps a free joint's angular velocity in the base (body) frame.
            root_ang_vel_b = inv_transform_by_quat(arr["gavs"][frame, 0], root_rot)
            parts += [
                torch.cat([arr["gts"][frame, 0], root_rot, arr["euler"][frame]], dim=1),
                torch.cat([arr["gvs"][frame, 0], root_ang_vel_b, arr["euler_vel"][frame]], dim=1),
            ]
        return torch.cat(parts, dim=1)

    def _write_state(self, envs_mask: torch.Tensor, state: torch.Tensor) -> None:
        """Write ``state`` (:meth:`capture_state` layout, a row per env) into the pairs of the ``envs_mask`` envs."""
        for i, role in enumerate(ROLES):
            entity = self.resolve_entity(self._entity_names[role])
            s = state[:, i * (NQ + NV) : (i + 1) * (NQ + NV)]
            entity.set_dofs_vel(s[:, NQ:], dofs_idx_local=self._dofs_idx, envs_idx=envs_mask, skip_forward=True)
            entity.set_qpos(s[:, :NQ], qs_idx_local=self._qs_idx, envs_idx=envs_mask, zero_velocity=False)

    # --- pair state capture and restore (PSI) --------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        """Size of one pair state from :meth:`capture_state`: ``qpos`` (76) and ``qvel`` (75) per role."""
        return len(ROLES) * (NQ + NV)

    def capture_state(self) -> torch.Tensor:
        """The simulated state of both humanoids in generalized coordinates, ``(N, state_dim)``."""
        parts = []
        for role in ROLES:
            entity = self.resolve_entity(self._entity_names[role])
            parts += [entity.get_qpos(self._qs_idx), entity.get_dofs_vel(self._dofs_idx)]
        return torch.cat(parts, dim=-1)

    def restore_state(self, envs_mask: torch.Tensor, state: torch.Tensor, frame: torch.Tensor) -> None:
        """Put the pair of each just-reset env in ``envs_mask`` into a captured ``state`` at clip step ``frame``.

        ``envs_mask`` is an ``(N,)`` bool mask, and ``state`` ``(N, state_dim)`` and ``frame`` ``(N,)`` hold a row
        for every env, read on the mask only: Genesis writes a mask with a full batch without reading the GPU
        back, where an index list costs it a sync per write.

        The clip is then read from ``frame * dt`` on — the intent of AssistMimic's ``_sample_from_trajectory_buffer``
        (whose Isaac Gym code passes the frame index where a time in seconds is expected) — while the env's
        episode length keeps counting from 0, as AssistMimic's ``progress_buf`` does: the restored episode gets
        the start-of-episode grace steps (``JointPower``, ``PairTrackingFailure``), the full ``time_out`` budget,
        and per-episode logs normalized by the steps it actually ran. Republishes the observation reference.
        """
        n = self.num_envs
        if envs_mask.dtype != torch.bool or envs_mask.shape != (n,):
            raise ValueError(
                f"AssistMimicPairMotion.restore_state: envs_mask must be an ({n},) bool mask, got a {envs_mask.dtype} "
                f"tensor of shape {tuple(envs_mask.shape)}. For env ids, build the mask: "
                f"torch.zeros({n}, dtype=torch.bool, device=ids.device).index_fill_(0, ids, True)."
            )
        if state.shape != (n, self.state_dim) or frame.shape != (n,):
            raise ValueError(
                f"AssistMimicPairMotion.restore_state: state must be ({n}, {self.state_dim}) and frame ({n},), a "
                f"row for every env, got {tuple(state.shape)} and {tuple(frame.shape)}."
            )
        self._write_state(envs_mask, state)
        if self.psi is not None:
            self.psi.start(envs_mask, state, frame)
        self._frame_offset.copy_(torch.where(envs_mask, frame, self._frame_offset))
        self._refresh(self.obs_ref, self.motion_time(1))
