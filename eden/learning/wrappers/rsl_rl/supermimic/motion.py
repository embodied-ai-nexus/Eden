"""The motion term SuperMimic trains on: forward-kinematics links and a reset distribution the runner sets.

:class:`SuperMimicMotion` is a :class:`~eden.managers.terms.motions.KinematicLinksMotion` (root-and-joints dataset,
link fields by forward kinematics) whose resets draw a frame of the concatenated library from per-bin weights that
the runner sets (:meth:`SuperMimicMotion.set_reset_distribution`). Each env's frame on that timeline is
:meth:`MotionManager.get_library_frame <eden.managers.motion_manager.MotionManager.get_library_frame>`.

A task trained by the SuperMimic runner configures this term; the runner reads its library and kinematics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import bin_frame_range
from eden.managers.motion_manager import MOTION_TERM_REGISTRY
from eden.managers.terms.motions import KinematicLinksMotion
from eden.utils.misc import is_env_mask
from eden.utils.motion_reference.library import MotionLibrary

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.motions import MotionTermOptions


#: ``torch.multinomial`` takes at most this many categories.
_MAX_RESET_BINS = 2**24


@MOTION_TERM_REGISTRY.register()
class SuperMimicMotion(KinematicLinksMotion):
    """A :class:`~eden.managers.terms.motions.KinematicLinksMotion` with a reset distribution over the library.

    Frames are the dataset's own: the dataset must run at the control rate.

    Resets draw a frame of the concatenated library: the timeline is cut into ``reset_bin_length_s`` bins, a bin is
    drawn from the reset distribution (uniform until :meth:`set_reset_distribution` is called, or always with
    ``reset_sampling="uniform"``), and a frame uniformly within it. A clip's last frame is never a start. With
    ``reset_sampling="clip_start"``, env ``i`` starts clip ``i mod num_clips`` at its first frame, which plays every
    clip once with one env per clip.

    The bins lie on the library timeline, not within each clip as ``motion_bin_length_s`` bins do, because the
    training loop that sets the distribution values a bin by the return earned from it and follows that return
    across a clip's end. This sampling replaces the base term's, so the options of the base sampling
    (:meth:`MotionTermOptions.start_sampling_defaults
    <eden.options.managers.motions.MotionTermOptions.start_sampling_defaults>`) must keep their defaults.

    Parameters
    ----------
    reset_bin_length_s : float
        Length of a reset bin, in seconds.
    reset_sampling : {"external", "uniform", "clip_start"}
        Reset bins from the distribution set by :meth:`set_reset_distribution`, uniform bins, or each env's clip
        from its first frame.
    """

    reset_bin_length_s: float = 0.2
    reset_sampling: Literal["external", "uniform", "clip_start"] = "external"

    def __init__(self, env: EnvBase, options: MotionTermOptions) -> None:
        super().__init__(env=env, options=options)
        name = type(self).__name__
        self.require_default_start_options(
            f"draws every start from its own reset distribution (reset_sampling={self.reset_sampling!r}, "
            f"reset_bin_length_s={self.reset_bin_length_s})"
        )
        rates = self.data_source.motion_sequences[self.sequence_key].framerate.unique()
        if rates.numel() != 1 or abs(float(rates[0]) * env.dt - 1.0) > 1e-6:
            raise ValueError(
                f"{name} reads one dataset frame per control step, so the dataset must run at "
                f"{1.0 / env.dt:g} Hz (1 / env dt), got {rates.tolist()} Hz. Set the dataset's target_framerate."
            )
        #: The dataset rate (Hz), which is the control rate.
        self.fps = float(rates[0])
        if round(self.reset_bin_length_s * self.fps) < 1:
            raise ValueError(
                f"{name}: reset_bin_length_s={self.reset_bin_length_s} is shorter than half a frame "
                f"({0.5 / self.fps:g} s at {self.fps:g} Hz), so a reset bin would hold no frame. Set it to at least "
                f"one frame period, {1.0 / self.fps:g} s."
            )
        #: The enabled clips as one timeline, and the number of reset bins laid over it. Set by
        #: ``enable_trajectories``, which the manager calls once the term is assigned its envs.
        self.library: MotionLibrary | None = None
        self.bin_count = 0

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        """Enable the selected clips, and lay the reset bins over their timeline with uniform weights."""
        super().enable_trajectories(traj_ids)
        self.library = MotionLibrary.from_sequence(self.data_source.motion_sequences[self.sequence_key])
        short = ((self.library.ends - self.library.starts) < 2).nonzero().flatten().tolist()
        if short:
            raise ValueError(
                f"{type(self).__name__}: clips {short} of the enabled library hold fewer than 2 frames at "
                f"{self.fps:g} Hz. A reset starts before a clip's last frame, so every clip needs at least two. "
                "Remove them from the dataset's clip list."
            )
        # ceil(frames / frames per bin): every bin holds at least one frame.
        self.bin_count = -(-self.library.num_frames // round(self.reset_bin_length_s * self.fps))
        if self.bin_count > _MAX_RESET_BINS:
            shortest = self.library.num_frames / _MAX_RESET_BINS / self.fps
            raise ValueError(
                f"{type(self).__name__}: reset_bin_length_s={self.reset_bin_length_s} cuts the library's "
                f"{self.library.num_frames} frames into {self.bin_count} reset bins, and a reset draws its bin with "
                f"torch.multinomial, which takes at most {_MAX_RESET_BINS} categories. Set reset_bin_length_s to "
                f"at least {shortest:.3g} s."
            )
        self._reset_probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=self.device)

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
        num_frames = self.library.num_frames
        if self.reset_sampling == "uniform":
            frames = torch.randint(0, num_frames - 1, (n,), device=self.device)
        else:
            bins = torch.multinomial(self._reset_probs, n, replacement=True).clamp_(max=self.bin_count - 1)
            first, end = bin_frame_range(bins, self.bin_count, num_frames)
            span = (end - first).clamp_(min=1)
            frames = (first + (torch.rand(n, device=self.device) * span).long()).clamp_(max=num_frames - 1)
        on_last = frames == self.library.last_frame(frames)
        return torch.where(on_last, frames - 1, frames)

    def sample_starts(self, assigned_ids: torch.Tensor) -> None:
        mask = is_env_mask(assigned_ids)
        if self.reset_sampling == "clip_start":
            envs = torch.arange(self.num_assigned_envs, device=self.device)
            frames = self.library.starts[envs % self.library.starts.numel()]
            frames = frames if mask else frames[assigned_ids]
        else:
            frames = self._sample_frames(self.num_assigned_envs if mask else len(assigned_ids))
        clips = self.library.clip(frames)
        self.set_start_frames(assigned_ids, clips, frames - self.library.starts[clips])
