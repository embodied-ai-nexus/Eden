"""Return-based reset sampling over the bins of the motion library.

:class:`ReturnSampler` measures ``V[b]``, the discounted return the policy earns from bin ``b`` on its own rollouts,
and turns it into the reset distribution: a uniform explore share, a softmax focus on the bins with low return, and
a keep share on the bins the focus has concentrated on, so a hard motion that is learned keeps some practice. An
optional failure share goes to the bins shortly before the frames where episodes recently ended in a fall, after
BlinkMimic's ``--fail_reset`` (https://github.com/fuyichen0725/blinkmimic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from eden.learning.wrappers.rsl_rl.supermimic.options import ReturnSamplerOptions
    from eden.utils.motion_reference.library import MotionLibrary

#: Every bin's fall rate is multiplied by this on an env step in which an episode ended.
_FALL_RATE_FADE = 0.999
#: The weight of a bin's fall rate against the mean of its new samples, on an env step that gives it samples.
_FALL_RATE_MEMORY = 0.9


def frame_bins(frames: torch.Tensor, bin_count: int, frame_count: int) -> torch.Tensor:
    """The reset bin of each frame of a ``frame_count``-frame timeline cut into ``bin_count`` bins."""
    return frames * bin_count // frame_count


def bin_frame_range(bins: torch.Tensor, bin_count: int, frame_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(first, end)`` of the frames :func:`frame_bins` maps to each of ``bins``; ``first == end`` for an empty bin.

    The motion term draws a reset frame from this range and the return sampler credits the frame's return with
    :func:`frame_bins`, so a bin's value is measured on the frames its resets start from.
    """
    # Ceiling division: the first frame f with f * bin_count // frame_count == bin.
    first = -((-bins * frame_count) // bin_count)
    end = -((-(bins + 1) * frame_count) // bin_count)
    return first, end


class ReturnSampler:
    """Bin values from the simulator's rewards, and the reset distribution they give.

    Parameters
    ----------
    options : ReturnSamplerOptions
        Discount, update horizon, memory and the distribution's shares.
    bin_count : int
        Bins of the library timeline.
    frame_count : int
        Frames of the library timeline.
    num_envs : int
        Simulator envs.
    max_step_reward : float
        The largest reward of one step; a bin reads ``v_max = max_step_reward / (1 - gamma)`` until measured.
    library : MotionLibrary | None
        The motion library, whose clip starts bound the failure share's bins. Needed with ``options.failure``.
    bin_frames : int
        Frames of one bin. Needed with ``options.failure``.
    """

    def __init__(
        self,
        options: ReturnSamplerOptions,
        bin_count: int,
        frame_count: int,
        num_envs: int,
        max_step_reward: float,
        device: str | torch.device,
        library: MotionLibrary | None = None,
        bin_frames: int = 0,
    ):
        self.options = options
        self.bin_count = bin_count
        self.frame_count = frame_count
        #: Each bin's rate of falls among the episodes that ended within ``failure_lead_s`` after it, fading.
        self.fall_rate = torch.zeros(bin_count, device=device)
        self._library = library
        if options.failure > 0.0:
            if library is None or bin_frames < 1:
                raise ValueError(
                    f"return_sampler.failure={options.failure} places resets before falls within the fall's clip, "
                    f"which needs the motion library and the bin length in frames (got library={library!r}, "
                    f"bin_frames={bin_frames}). Pass both to ReturnSampler, or set failure to 0."
                )
            lead_bins = max(1, round(options.failure_lead_s * library.fps / bin_frames))
            #: Frames between a fall and each of the bins before it that the fall is a sample of, nearest first.
            self._lead_frames = (torch.arange(lead_bins, device=device) + 1) * bin_frames
        self.v_max = max_step_reward / (1.0 - options.gamma)
        self.value = torch.full((bin_count,), self.v_max, device=device)
        self.count = torch.zeros(bin_count, device=device)
        # Summed over the whole run: float64 keeps the small per-bin terms.
        self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
        shape = (num_envs, options.horizon)
        self._frames = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._step = 0

    def restart_horizon(self) -> None:
        """Discard the steps recorded since the last update. For a caller that resets the envs, after which the
        buffered steps no longer continue into the next ones.
        """
        self._step = 0

    def bins(self, frames: torch.Tensor) -> torch.Tensor:
        return frame_bins(frames, self.bin_count, self.frame_count)

    def record(
        self,
        frames: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        clip_end: torch.Tensor,
        timed_out: torch.Tensor,
    ) -> torch.Tensor | None:
        """Buffer one env step: each env's frame before the step, its reward and how its episode ended, if it did.

        A fall terminates; reaching the clip's last frame and a time-out truncate. Every ``horizon`` steps the bin
        values update and the new reset distribution is returned; otherwise ``None``.
        """
        if self.options.failure > 0.0:
            self._record_falls(frames, terminated, terminated | clip_end | timed_out)
        col = self._step
        self._frames[:, col] = frames
        self._rewards[:, col] = rewards
        self._terminated[:, col] = terminated
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._step += 1
        if self._step < self.options.horizon:
            return None
        self._step = 0
        self._update()
        return self._distribution()

    def _record_falls(self, frames: torch.Tensor, fell: torch.Tensor, ended: torch.Tensor) -> None:
        """Fold one env step's episode ends into the fall rate of the bins before them.

        An episode that ended at frame ``f`` is one sample, 1 for a fall and 0 otherwise, of every bin holding a
        frame ``f - k * bin_frames`` of its clip, ``k = 1 .. lead bins``, or of the clip's first bin if there is
        none. A bin moves towards the mean of its samples of the step.
        """
        start = self._library.starts[self._library.clip(frames)].unsqueeze(1)
        before = frames.unsqueeze(1) - self._lead_frames
        nearest = self._lead_frames == self._lead_frames[0]
        sampled = (ended.unsqueeze(1) & ((before >= start) | nearest)).float()
        bins = self.bins(torch.maximum(before, start)).reshape(-1)
        count = torch.zeros_like(self.fall_rate).index_add_(0, bins, sampled.reshape(-1))
        falls = torch.zeros_like(self.fall_rate).index_add_(0, bins, (sampled * fell.unsqueeze(1)).reshape(-1))
        faded = torch.where(ended.any(), self.fall_rate * _FALL_RATE_FADE, self.fall_rate)
        moved = _FALL_RATE_MEMORY * faded + (1.0 - _FALL_RATE_MEMORY) * falls / count.clamp(min=1.0)
        self.fall_rate = torch.where(count > 0, moved, faded)

    def _dones(self) -> torch.Tensor:
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self) -> torch.Tensor:
        """Discounted return of every buffered step. Where an episode ends, and at the buffer's last step, the return
        continues from the value after the step: none after a fall, the mean bin value after a clip's last frame, and
        the next frame's bin value after a time-out or at the buffer's end.
        """
        gamma = self.options.gamma
        next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
        next_value = torch.where(
            self._terminated,
            0.0,
            torch.where(self._clip_end, self.value.mean(), self.value[self.bins(next_frames)]),
        )
        dones = self._dones()
        returns = torch.empty_like(self._rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.options.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = self._rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self) -> None:
        """Fold the buffer into the bin values. One env's visits to one bin within one episode are one sample, their
        mean return; a bin's earlier samples count at most ``memory`` times.
        """
        returns = self._returns()
        dones = self._dones()
        num_envs, horizon = dones.shape
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
        key = (envs * horizon + episode) * self.bin_count + self.bins(self._frames)
        keys, inverse, visits = torch.unique(key.reshape(-1), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=returns.device).scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.bin_count
        new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        self.count = self.count.clamp(max=self.options.memory) + new
        self.value = torch.where(new > 0, self.value + (total - new * self.value) / self.count, self.value)

    def _focus(self) -> torch.Tensor:
        return torch.softmax(-self.options.steepness * self.value / self.v_max, dim=0)

    def _distribution(self) -> torch.Tensor:
        """The reset distribution after an update of the bin values, which adds to the bins' accumulated effort."""
        self.effort += self._focus().double().square()
        return self.distribution()

    def distribution(self) -> torch.Tensor:
        """The reset distribution of the current bin values and accumulated effort (after at least one update)."""
        o = self.options
        keep = (self.effort / self.effort.sum()).float()
        if o.failure == 0.0:
            return o.explore / self.bin_count + (1.0 - o.explore - o.keep) * self._focus() + o.keep * keep
        focus = 1.0 - o.explore - o.keep - o.failure
        others = (o.explore / self.bin_count + focus * self._focus() + o.keep * keep) / (1.0 - o.failure)
        total = self.fall_rate.sum()
        falls = torch.where(total > 0, self.fall_rate / total.clamp(min=1e-12), others)
        return (1.0 - o.failure) * others + o.failure * falls

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value, "count": self.count, "effort": self.effort, "fall_rate": self.fall_rate}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        shapes = {k: tuple(state[k].shape) for k in ("value", "count", "effort")}
        if any(shape != (self.bin_count,) for shape in shapes.values()):
            raise ValueError(
                f"The checkpoint's return sampler holds {shapes['value'][0] if shapes['value'] else 0} reset bins, "
                f"but this run's motion library has {self.bin_count} ({self.frame_count} frames): the checkpoint "
                "was trained on another library, clip list or reset_bin_length_s. Resume on the library it was "
                "trained on, or start a new run."
            )
        self.value, self.count, self.effort = (state[k].to(self.value.device) for k in ("value", "count", "effort"))
        # A checkpoint written without the failure share holds no fall rate.
        fall_rate = state.get("fall_rate")
        self.fall_rate = torch.zeros_like(self.value) if fall_rate is None else fall_rate.to(self.value.device)
