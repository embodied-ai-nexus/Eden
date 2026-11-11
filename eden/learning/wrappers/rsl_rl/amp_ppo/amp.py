"""Adversarial-motion-prior components used by :class:`~eden.learning.wrappers.rsl_rl.amp_ppo.ppo.AmpPPO`.

Ports the adversarial-motion-prior machinery of ``TeleHuman/humanoid_skateboarding``
(its vendored ``rsl_rl`` fork: ``DiscriminatorMulti``, ``G1_AMPLoader``,
``ReplayBufferMulti``, ``Normalizer``):

- :class:`AmpMotionSampler` draws windows of ``num_frames`` consecutive frames
  (linear interpolation between recorded frames) from expert motion clips. A
  frame is one joint-position vector, in the order the env's AMP observation
  group emits it (HUSKY: the G1's 23 no-wrist joints in MJCF order).
- :class:`AmpDiscriminator` is the LSGAN discriminator (spectral-norm MLP,
  expert target +1, policy target -1); its raw output ``d`` maps to the style
  reward ``coef * clamp(1 - 0.25 (d - 1)^2, min=0)``.
- :class:`AmpReplayBuffer` is a flat ring buffer of policy windows.
- :class:`FrameNormalizer` is a running mean/std over single frames, applied
  per-frame to whole windows before the discriminator.

Deviation from the source: the source updated its normalizer with
already-normalized batches (a bug that decays the statistics toward identity);
this one updates from raw frames.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

from eden.utils.assets import get_asset_path


class AmpMotionSampler:
    """Samples expert windows of ``num_frames`` consecutive frames from motion clips.

    Parameters
    ----------
    motion_file : str
        Asset path of the ``.npz`` bundle (``motion_<i>`` arrays of shape (T, frame_dim),
        plus ``fps`` and ``num_motions``).
    num_frames : int
        Frames per window. The window spans ``[t - (num_frames - 2) dt, t + dt]`` around the
        sampled time ``t``, matching the source loader's offsets.
    device : torch.device | str
        Device the sampled windows live on.
    """

    def __init__(self, motion_file: str, num_frames: int, device: torch.device | str) -> None:
        data = np.load(get_asset_path(motion_file))
        self.device = device
        self.num_frames = num_frames
        self.dt = 1.0 / float(data["fps"])
        self.motions = [
            torch.as_tensor(data[f"motion_{i}"], dtype=torch.float32, device=device)
            for i in range(int(data["num_motions"]))
        ]
        self.frame_dim = int(self.motions[0].shape[1])
        for i, motion in enumerate(self.motions):
            if motion.shape[0] < num_frames + 2:
                raise ValueError(f"motion_{i} in {motion_file} is too short ({motion.shape[0]} frames)")

    def sample(self, batch_size: int) -> torch.Tensor:
        """Return expert windows, shape (batch_size, num_frames, frame_dim), oldest frame first."""
        counts = torch.multinomial(
            torch.ones(len(self.motions), device=self.device), batch_size, replacement=True
        ).bincount(minlength=len(self.motions))
        windows = []
        offsets = torch.arange(-(self.num_frames - 2), 2, device=self.device, dtype=torch.float32)
        for motion, count in zip(self.motions, counts.tolist()):
            if count == 0:
                continue
            num_rows = motion.shape[0]
            # Continuous frame index such that every offset stays in [0, num_rows - 1].
            base = torch.empty(count, device=self.device).uniform_(self.num_frames - 2, num_rows - 2)
            idx = base.unsqueeze(1) + offsets.unsqueeze(0)  # (count, num_frames)
            low = idx.floor().long()
            frac = (idx - low.float()).unsqueeze(-1)
            frames = motion[low] * (1.0 - frac) + motion[low + 1] * frac
            windows.append(frames)
        return torch.cat(windows, dim=0)


class AmpReplayBuffer:
    """Flat ring buffer of policy windows (one row = one flattened window)."""

    def __init__(self, window_dim: int, capacity: int, device: torch.device | str) -> None:
        self._buffer = torch.zeros(capacity, window_dim, device=device)
        self._capacity = capacity
        self._next = 0
        self._size = 0

    def insert(self, windows: torch.Tensor) -> None:
        """Insert a batch of flattened windows, overwriting the oldest entries."""
        n = windows.shape[0]
        idx = (self._next + torch.arange(n, device=windows.device)) % self._capacity
        self._buffer[idx] = windows.detach().clone()
        self._next = int((self._next + n) % self._capacity)
        self._size = min(self._size + n, self._capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self._size == 0:
            raise ValueError("AmpReplayBuffer is empty — insert policy windows before sampling.")
        idx = torch.randint(self._size, (batch_size,), device=self._buffer.device)
        return self._buffer[idx]

    def __len__(self) -> int:
        return self._size


class FrameNormalizer(nn.Module):
    """Running mean/std over single frames, applied per-frame to windows."""

    def __init__(self, frame_dim: int, clip: float = 10.0, eps: float = 1e-2) -> None:
        super().__init__()
        self.clip = clip
        self.eps = eps
        self.register_buffer("mean", torch.zeros(frame_dim))
        self.register_buffer("var", torch.ones(frame_dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, frames: torch.Tensor) -> None:
        """Fold a batch of raw frames (..., frame_dim) into the running statistics."""
        flat = frames.reshape(-1, frames.shape[-1])
        batch_count = flat.shape[0]
        if batch_count == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var.copy_((m_a + m_b + delta.square() * self.count * batch_count / total) / total)
        self.count.copy_(total)

    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize frames or windows of frames (..., frame_dim)."""
        std = (self.var + self.eps**2).sqrt()
        return ((frames - self.mean) / std).clamp(-self.clip, self.clip)


class AmpDiscriminator(nn.Module):
    """LSGAN discriminator over flattened windows (spectral-norm MLP + linear head)."""

    def __init__(self, window_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = window_dim
        for hidden in hidden_dims:
            layers += [spectral_norm(nn.Linear(in_dim, hidden)), nn.ReLU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Linear(in_dim, 1))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(windows))

    def style_reward(self, windows: torch.Tensor, coef: float) -> torch.Tensor:
        """Per-window style reward ``coef * clamp(1 - 0.25 (d - 1)^2, min=0)``, shape (N,)."""
        with torch.no_grad():
            d = self.forward(windows).squeeze(-1)
        return coef * torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)

    def gradient_penalty(self, expert_windows: torch.Tensor, scale: float) -> torch.Tensor:
        """Penalize the discriminator's gradient norm at the expert samples."""
        expert = expert_windows.detach().clone().requires_grad_(True)
        d = self.head(self.trunk(expert))
        grad = torch.autograd.grad(
            outputs=d, inputs=expert, grad_outputs=torch.ones_like(d), create_graph=True, retain_graph=True
        )[0]
        return scale * grad.norm(2, dim=1).square().mean()
