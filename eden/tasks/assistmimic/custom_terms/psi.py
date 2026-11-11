"""AssistMimic's PSI trajectory buffer (upstream ``RSI.py``): restart episodes from states of good past rollouts.

Every env's pair state is recorded along its episode. When an episode ends with
``max(caregiver return, recipient return) >= return_threshold`` its trajectory is stored, FIFO, in a per-clip
buffer. At reset, each env picks with probability ``usage_ratio`` a stored trajectory of its new clip and a
frame of it, and both humanoids restart from that simulated state (not the reference), with the reference
clock at that frame. Episodes that ended by early termination keep only the frames up to ``tail_margin``
steps before the failure.

:class:`~eden.tasks.assistmimic.custom_terms.commands.AssistMimicPairMotion` drives it from inside the env —
:meth:`record` after each control step, :meth:`end_episodes` and :meth:`sample_restarts` in its reset, then
:meth:`start` with the state each reset env was put in — so any runner trains with it.

Frames are clip steps (``AssistMimicPairMotion.reference_frame``), not episode steps, so ``max_len`` is the
longest clip in control steps (+2). Memory: the live recording is ``num_envs x max_len x state_dim`` float32 —
at the default ``train`` settings (8192 envs, 302-dim pair state) about 1 GB per 100 frames.

Nothing here reads the GPU back: storing takes a fixed-size batch of the qualifying episodes (``topk``, not
``nonzero()``), and restarts come back as an ``(N,)`` mask with every env's draw. The counters are tensors.
"""

from __future__ import annotations

import torch


class PsiTrajectoryBuffer:
    """Per-clip FIFO of pair trajectories, all on the GPU.

    Parameters
    ----------
    num_envs, num_clips, state_dim, max_len : int
        Batch size, clip count, size of one captured pair state, longest episode in clip steps.
    buffer_size : int
        Trajectories kept per clip (AssistMimic ``trajectory_buffer_size``: 10).
    return_threshold : float
        Minimum ``max`` of the two roles' episode returns to store a trajectory.
    usage_ratio : float
        Probability that a reset uses the buffer when its clip has trajectories (AssistMimic: 0.5).
    tail_margin : int
        Frames dropped before an early termination.
    max_store_per_step : int
        Most trajectories stored in one reset; past it a random subset of the qualifying ones is kept. Each reset
        copies this many recordings (``max_store_per_step x max_len x state_dim``), whether or not they qualify.
    """

    def __init__(
        self,
        num_envs: int,
        num_clips: int,
        state_dim: int,
        max_len: int,
        device,
        buffer_size: int = 10,
        return_threshold: float = 300.0,
        usage_ratio: float = 0.5,
        tail_margin: int = 10,
        max_store_per_step: int = 128,
    ):
        self.device, self.max_len, self.buffer_size = device, max_len, buffer_size
        self.return_threshold, self.usage_ratio, self.tail_margin = return_threshold, usage_ratio, tail_margin
        self.num_clips, self.store_batch = num_clips, min(max_store_per_step, num_envs)
        self.live = torch.zeros(num_envs, max_len, state_dim, device=device)
        self.live_start = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Whether every state recorded since ``live_start`` is finite, kept as it is recorded: checking the
        # recording itself at an episode end would scan all of ``live`` (and stale frames of past episodes).
        self.live_finite = torch.ones(num_envs, dtype=torch.bool, device=device)
        # One row per clip plus a scratch row (index ``num_clips``) that takes the batch's non-qualifying rows.
        self.store = torch.zeros(num_clips + 1, buffer_size, max_len, state_dim, device=device)
        self.store_lo = torch.zeros(num_clips + 1, buffer_size, dtype=torch.long, device=device)
        self.store_hi = torch.zeros(num_clips + 1, buffer_size, dtype=torch.long, device=device)  # exclusive; 0: empty
        self.write_idx = torch.zeros(num_clips + 1, dtype=torch.long, device=device)
        self.stored_total = torch.zeros((), dtype=torch.long, device=device)
        self.restored_total = torch.zeros((), dtype=torch.long, device=device)
        self._rows = torch.arange(num_envs, device=device)

    def record(self, state: torch.Tensor, frame: torch.Tensor) -> None:
        """Record the pair state each env is in at clip step ``frame``."""
        self.live[self._rows, frame.clamp(max=self.max_len - 1)] = state
        self.live_finite &= torch.isfinite(state).all(dim=1)

    def start(self, envs_mask: torch.Tensor, state: torch.Tensor, frame: torch.Tensor) -> None:
        """Begin the recording of each env in ``envs_mask`` with the ``state`` it was reset to, at clip step ``frame``.

        ``state`` ``(N, state_dim)`` and ``frame`` ``(N,)`` hold a row for every env, read on the mask only.
        """
        frame = frame.clamp(max=self.max_len - 1)
        self.live[self._rows, frame] = torch.where(envs_mask[:, None], state, self.live[self._rows, frame])
        self.live_start = torch.where(envs_mask, frame, self.live_start)
        self.live_finite = torch.where(envs_mask, torch.isfinite(state).all(dim=1), self.live_finite)

    def end_episodes(self, returns2, done, term, end_frame, clip) -> None:
        """Store the ``done`` episodes that qualify, given their ``(N, 2)`` role returns.

        ``end_frame`` is the clip step each episode ended at — recorded, but left out of the stored trajectory,
        since a restart there would end at once — ``term`` whether it ended by early termination, ``clip`` its clip.
        """
        good = done & (returns2.max(dim=1).values >= self.return_threshold) & self.live_finite
        # ``record`` piles every frame from ``max_len - 1`` on into the last slot, so only the frames before it
        # are a trajectory — an episode the config lets run past its clip's end keeps those.
        hi = (end_frame - self.tail_margin * term.long()).clamp(max=self.max_len - 1)
        good &= (hi - self.live_start) >= 2
        # A fixed-size batch, the qualifying envs first (a random subset of them if there are more): picking
        # exactly the qualifying ones would need their count on the host.
        idx = torch.topk(good.float() + 0.5 * torch.rand(good.shape, device=self.device), self.store_batch).indices
        ok = good[idx]
        c = torch.where(ok, clip[idx], self.num_clips)
        # Vectorized FIFO write: envs storing into the same clip this step take consecutive slots after
        # that clip's write index (their rank within the clip), then the index advances by the count.
        # E.g. sorted clips [0, 0, 1] -> first [T, F, T] -> ranks [0, 1, 0].
        order = torch.argsort(c, stable=True)
        idx, c, ok = idx[order], c[order], ok[order]
        n = c.numel()
        first = torch.ones(n, dtype=torch.bool, device=self.device)
        first[1:] = c[1:] != c[:-1]
        pos = torch.arange(n, device=self.device)
        rank = pos - torch.cummax(torch.where(first, pos, torch.zeros_like(pos)), 0).values
        # Every qualifying episode is written, even when a clip gets more in one call than it has slots: ranks past
        # ``buffer_size`` wrap onto slots this call already writes, and on the GPU duplicate writes land element by
        # element in no fixed order, so such a slot holds a blend of those episodes' states — all of the same
        # clip, recorded at the same frames — with one of their [lo, hi) windows. At 8192 envs that is most
        # stores. Keeping only ``buffer_size`` episodes per call instead trained to 0.50-0.68 help-up success (four
        # runs) where writing them all trained to 0.68-1.00 (five runs, mean 0.87), so they are all written.
        slot = (self.write_idx[c] + rank) % self.buffer_size
        self.store[c, slot] = self.live[idx]
        self.store_lo[c, slot] = self.live_start[idx]
        self.store_hi[c, slot] = hi[idx]
        self.write_idx = (self.write_idx.index_add(0, c, ok.long())) % self.buffer_size
        self.stored_total += ok.sum()

    def sample_restarts(self, reset: torch.Tensor, clip: torch.Tensor):
        """For freshly ``reset`` envs ``(N,)``, pick restarts: ``(mask, states, frames)``, each over all ``N`` envs.

        ``mask`` is the envs to restart — their clip has trajectories, and a ``usage_ratio`` draw — and
        ``states`` / ``frames`` are meaningful on it only (every env draws; the mask selects).
        """
        n = reset.shape[0]
        filled = self.store_hi[clip] > 0  # (N, buffer_size)
        mask = reset & filled.any(dim=1) & (torch.rand(n, device=self.device) < self.usage_ratio)
        slot = torch.argmax(torch.rand(filled.shape, device=self.device) * filled, dim=1)  # uniform over the filled
        lo, hi = self.store_lo[clip, slot], self.store_hi[clip, slot]
        frame = lo + (torch.rand(n, device=self.device) * (hi - lo).float()).long()
        states = self.store[clip, slot, frame]
        self.restored_total += mask.sum()
        return mask, states, frame
