"""Reference motion for SuperMimic: a root-and-joints motion library with link states by forward kinematics.

A tracker trained on all of BONES-SEED reads ~47.6M frames. Stored as root pose and twist plus joint positions and
velocities they take ~13.5 GB; with every tracked link's pose and twist precomputed they would not fit on one GPU.
The datasets here keep root and joints only, and :class:`SuperMimicMotion` fills the reference buffer's link fields
by forward kinematics of the frames the envs read, so the reward, termination and observation terms work unchanged.

The motion term also owns the reset distribution: resets draw a frame of the concatenated library from per-bin
weights that a training loop sets (:meth:`SuperMimicMotion.set_reset_distribution`), and the term reports each env's
frame on that timeline (:attr:`SuperMimicMotion.frame_index`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from eden.datasets.base import DATASET_REGISTRY
from eden.datasets.lafan1_dataset import LAFAN1RetargetedG1Dataset
from eden.datasets.seed_dataset import SEEDRetargetedG1Dataset
from eden.managers.motion_manager import MOTION_TERM_REGISTRY, MotionTerm
from eden.tasks.supermimic.kinematics import ForwardKinematics
from eden.utils.assets import get_asset_path
from eden.utils.buffers.cbt import ConcatBatchTensor
from eden.utils.misc import is_env_mask
from eden.utils.motion_reference.utils import derive_body_frame_link_fields

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.motions import MotionTermOptions
    from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionReferenceData


_ROOT_JOINT_FIELDS = ("dofs_pos", "dofs_vel", "base_pos_w", "base_lin_vel_w", "base_quat_w", "base_ang_vel_w")
_LINK_FIELDS = (
    "links_pos_w",
    "links_quat_w",
    "links_lin_vel_w",
    "links_ang_vel_w",
    "links_pos_b",
    "links_quat_b",
    "links_lin_vel_b",
    "links_ang_vel_b",
)
_VELOCITY_FIELDS = ("dofs_vel", "base_lin_vel_w", "base_ang_vel_w")


class _RootJointsMixin:
    """Prepare the clips without link fields: no forward kinematics at load, nothing per link on the GPU."""

    motion_links_name: ClassVar[dict[str, list[str]]] = {"g1": []}

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        return self._prepare_single_key(self._read_single_sequence, desc=f"{type(self).__name__}: reading clips")


@DATASET_REGISTRY.register()
class LAFAN1RootJointsG1Dataset(_RootJointsMixin, LAFAN1RetargetedG1Dataset):
    """LAFAN1 retargeted to the G1 (40 clips, 2.45 h), root and joints only."""


@DATASET_REGISTRY.register()
class SEEDRootJointsG1Dataset(_RootJointsMixin, SEEDRetargetedG1Dataset):
    """BONES-SEED retargeted to the G1, root and joints only."""


@dataclass(frozen=True)
class MotionLibrary:
    """The motion library as one timeline: per-frame tensors of every clip, concatenated.

    Positions are free of env origins. Frame ``f`` belongs to the clip ``c`` with ``starts[c] <= f < ends[c]``.
    """

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    fps: float

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    def last_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """The last frame of the clip each of ``frames`` belongs to."""
        clip = torch.searchsorted(self.ends, frames, right=True).clamp(max=self.ends.numel() - 1)
        return self.ends[clip] - 1


@MOTION_TERM_REGISTRY.register()
class SuperMimicMotion(MotionTerm):
    """Motion term over a root-and-joints dataset that computes the tracked links by forward kinematics.

    The reference buffer's world-frame link fields (and the body-frame ones derived from them) hold ``links_name``
    at the frames the envs read. Frames are the dataset's own: the dataset must run at the control rate.

    Resets draw a frame of the concatenated library: the timeline is cut into ``reset_bin_length_s`` bins, a bin is
    drawn from the reset distribution (uniform until :meth:`set_reset_distribution` is called, or always with
    ``reset_sampling="uniform"``), and a frame uniformly within it. A clip's last frame is never a start. With
    ``reset_sampling="clip_start"``, env ``i`` starts clip ``i mod num_clips`` at its first frame, which plays every
    clip once with one env per clip.

    Parameters
    ----------
    entity_name : str
        The robot whose MJCF gives the kinematic tree.
    links_name : list[str]
        Links the reference buffer holds, in buffer order.
    reset_bin_length_s : float
        Length of a reset bin, in seconds.
    reset_sampling : {"external", "uniform", "clip_start"}
        Reset bins from the distribution set by :meth:`set_reset_distribution`, uniform bins, or each env's clip
        from its first frame.
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    reset_bin_length_s: float = 0.2
    reset_sampling: Literal["external", "uniform", "clip_start"] = "external"

    def __init__(self, env: EnvBase, options: MotionTermOptions) -> None:
        super().__init__(env=env, options=options)
        if not self.links_name:
            raise ValueError(f"{type(self).__name__}: links_name is empty; name the links the reference tracks.")
        if len(self.entities_map) != 1:
            raise ValueError(f"{type(self).__name__} tracks one entity, got entities_map={self.entities_map}.")
        (self._key,) = self.entities_map.values()
        data = self.data_source.motion_sequences[self._key]
        rates = data.framerate.unique()
        if rates.numel() != 1 or abs(float(rates[0]) * env.dt - 1.0) > 1e-6:
            raise ValueError(
                f"{type(self).__name__} reads one dataset frame per control step, so the dataset must run at "
                f"{1.0 / env.dt:g} Hz (1 / env dt), got {rates.tolist()} Hz. Set the dataset's target_framerate."
            )
        self.fps = float(rates[0])

        robot_options = getattr(env.scene_options, self.entity_name, None)
        if robot_options is None:
            raise ValueError(
                f"{type(self).__name__}: no entity {self.entity_name!r} in the scene options "
                f"({list(env.scene_options.keys())})."
            )
        self.fk = ForwardKinematics(
            get_asset_path(robot_options.file), self.target_dofs_name[self._key], self.links_name
        ).to(self.device)
        self.target_links_name[self._key] = list(self.links_name)

        # The manager sizes the reference buffer from the dataset's link fields: give it the link width with
        # zero-stride placeholders (no storage). This term fills the buffer itself and never reads them.
        joint_pos = data.dofs_pos.concatenated_tensor
        for field, width in (("links_pos_w", 3), ("links_quat_w", 4), ("links_lin_vel_w", 3), ("links_ang_vel_w", 3)):
            placeholder = joint_pos.new_zeros(1, 1, 1).expand(joint_pos.shape[0], len(self.links_name), width)
            setattr(data, field, ConcatBatchTensor.from_concatenated(placeholder, data.buffer_length))
        for field in ("links_pos_b", "links_quat_b", "links_lin_vel_b", "links_ang_vel_b"):
            setattr(data, field, None)

        lengths = data.buffer_length.to(torch.long)
        ends = lengths.cumsum(0)
        self.library = MotionLibrary(
            root_pos_w=data.base_pos_w.concatenated_tensor,
            root_quat_w=data.base_quat_w.concatenated_tensor,
            root_lin_vel_w=data.base_lin_vel_w.concatenated_tensor,
            root_ang_vel_w=data.base_ang_vel_w.concatenated_tensor,
            joint_pos=data.dofs_pos.concatenated_tensor,
            joint_vel=data.dofs_vel.concatenated_tensor,
            starts=ends - lengths,
            ends=ends,
            fps=self.fps,
        )
        self.bin_count = int(self.library.num_frames // round(self.reset_bin_length_s * self.fps)) + 1
        self._reset_probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=self.device)
        self.frame_index = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)

    @property
    def reset_distribution(self) -> torch.Tensor:
        """The per-bin reset distribution, ``(bin_count,)`` weights (not normalized)."""
        return self._reset_probs

    def set_reset_distribution(self, probs: torch.Tensor) -> None:
        """Set the per-bin reset distribution, ``(bin_count,)`` non-negative weights."""
        if probs.shape != (self.bin_count,):
            raise ValueError(f"reset distribution needs shape ({self.bin_count},), got {tuple(probs.shape)}.")
        self._reset_probs = probs.detach().to(self.device, torch.float).clamp_min(0.0)

    def _sample_frames(self, n: int) -> torch.Tensor:
        """``n`` start frames on the library timeline, none on a clip's last frame."""
        total = self.library.num_frames - 2
        if self.reset_sampling == "uniform":
            frames = torch.randint(0, total + 1, (n,), device=self.device)
        else:
            bins = torch.multinomial(self._reset_probs, n, replacement=True).clamp_(max=self.bin_count - 1)
            lo = (bins * total) // self.bin_count
            span = (((bins + 1) * total) // self.bin_count - lo).clamp_(min=1)
            frames = (lo + (torch.rand(n, device=self.device) * span).long()).clamp_(max=total)
        on_last = frames == self.library.last_frame(frames)
        return torch.where(on_last, frames - 1, frames)

    def _sample_assigned_env_starting_stub(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        assigned_ids = self.envs_idx_to_assigned_ids(envs_idx)
        mask = is_env_mask(assigned_ids)
        if self.reset_sampling == "clip_start":
            envs = torch.arange(self.num_assigned_envs, device=self.device)
            frames = self.library.starts[envs % self.library.starts.numel()]
            frames = frames if mask else frames[assigned_ids]
        else:
            frames = self._sample_frames(self.num_assigned_envs if mask else len(assigned_ids))
        clips = torch.searchsorted(self.library.ends, frames, right=True)
        start_s = (frames - self.library.starts[clips]).float() / self.fps
        if mask:
            selection, start_time = self._assigned_env_motion_selection, self._motion_buffer_start_time_s
            selection.copy_(torch.where(assigned_ids, clips, selection))
            start_time.copy_(torch.where(assigned_ids, start_s, start_time))
        else:
            self._assigned_env_motion_selection[assigned_ids] = clips
            self._motion_buffer_start_time_s[assigned_ids] = start_s

    def _frame_offset(self, assigned_ids: torch.Tensor, sample_timestamp: torch.Tensor) -> torch.Tensor:
        """The clip-local frame at ``sample_timestamp``, rounded: start times and the clock are whole frames."""
        start = self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1)
        return torch.round((start + sample_timestamp) * self.fps)

    def _frame_index_and_validity(self, assigned_ids, sample_timestamp, data):
        frame = self._frame_offset(assigned_ids, sample_timestamp)
        length = data.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
        return frame, frame < length

    def _past_clip_end(self, assigned_ids, sample_timestamp, data):
        return ~self._frame_index_and_validity(assigned_ids, sample_timestamp, data)[1]

    def _copy_frame_data(
        self,
        data: BatchMotionSequence,
        data_buffer: MotionReferenceData,
        envs_idx: slice | torch.Tensor,
        motion_ids: torch.Tensor,
        frame_idx: torch.Tensor,
        validity: torch.Tensor,
        origin_offset: torch.Tensor | None,
    ) -> None:
        rows = (self.library.starts[motion_ids].unsqueeze(-1) + frame_idx).reshape(-1)
        self.frame_index[envs_idx] = rows.view(frame_idx.shape)[:, 0]
        past_end = (~validity).reshape(-1, 1)
        state = {}
        for field in _ROOT_JOINT_FIELDS:
            value = getattr(data, field).concatenated_tensor.index_select(0, rows)
            if field in _VELOCITY_FIELDS:
                value = value.masked_fill(past_end, 0.0)
            state[field] = value
        if origin_offset is not None:
            num = frame_idx.shape[1]
            state["base_pos_w"] = state["base_pos_w"] + origin_offset.repeat_interleave(num, dim=0)
        links = dict(
            zip(
                ("links_pos_w", "links_quat_w", "links_lin_vel_w", "links_ang_vel_w"),
                self.fk(
                    state["base_pos_w"],
                    state["base_quat_w"],
                    state["dofs_pos"],
                    state["base_lin_vel_w"],
                    state["base_ang_vel_w"],
                    state["dofs_vel"],
                ),
                strict=True,
            )
        )
        links.update(
            zip(
                ("links_pos_b", "links_quat_b", "links_lin_vel_b", "links_ang_vel_b"),
                derive_body_frame_link_fields(
                    state["base_pos_w"],
                    state["base_quat_w"],
                    state["base_lin_vel_w"],
                    state["base_ang_vel_w"],
                    links["links_pos_w"],
                    links["links_quat_w"],
                    links["links_lin_vel_w"],
                    links["links_ang_vel_w"],
                ),
                strict=True,
            )
        )
        for field, value in {**state, **links}.items():
            dst = getattr(data_buffer, field, None)
            if dst is not None:
                dst[envs_idx] = value.reshape(dst[envs_idx].shape)
