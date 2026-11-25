"""A motion sequence read as one timeline of frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import BatchMotionSequence


@dataclass(frozen=True)
class MotionLibrary:
    """The root and joint state of every clip of a motion sequence, concatenated into one timeline.

    The tensors are the sequence's own storage, not copies. Positions are free of env origins. Frame ``f`` belongs to
    the clip ``c`` with ``starts[c] <= f < ends[c]``.
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

    @classmethod
    def from_sequence(cls, sequence: BatchMotionSequence) -> MotionLibrary:
        """The timeline of ``sequence``, whose clips must share one frame rate."""
        needed = ("base_pos_w", "base_quat_w", "base_lin_vel_w", "base_ang_vel_w", "dofs_pos", "dofs_vel")
        missing = [field for field in needed if getattr(sequence, field) is None]
        if missing:
            raise ValueError(
                f"A motion library holds the root and joint state of every frame, but the sequence stores no "
                f"{missing}. It needs a dataset of a floating-base, articulated robot."
            )
        rates = sequence.framerate.unique()
        if rates.numel() != 1:
            raise ValueError(
                f"A motion library is one timeline at one frame rate, but its clips run at {rates.tolist()} Hz. "
                "Set the dataset's target_framerate."
            )
        lengths = sequence.buffer_length.to(torch.long)
        ends = lengths.cumsum(0)
        return cls(
            root_pos_w=sequence.base_pos_w.concatenated_tensor,
            root_quat_w=sequence.base_quat_w.concatenated_tensor,
            root_lin_vel_w=sequence.base_lin_vel_w.concatenated_tensor,
            root_ang_vel_w=sequence.base_ang_vel_w.concatenated_tensor,
            joint_pos=sequence.dofs_pos.concatenated_tensor,
            joint_vel=sequence.dofs_vel.concatenated_tensor,
            starts=ends - lengths,
            ends=ends,
            fps=float(rates[0]),
        )

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    def clip(self, frames: torch.Tensor) -> torch.Tensor:
        """The clip each of ``frames`` belongs to."""
        return torch.searchsorted(self.ends, frames, right=True).clamp(max=self.ends.numel() - 1)

    def last_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """The last frame of the clip each of ``frames`` belongs to."""
        return self.ends[self.clip(frames)] - 1
