"""Depth encoder for the RMA depth student (extreme parkour): CNN backbone + proprio-fused GRU.

Implemented for the distillation phase of "Extreme Parkour with Legged Robots"
(Cheng et al. 2023) from the architecture described in the paper; the upstream
implementations were used as behavioral references only (attribution in
``examples/projects/locomotion/extreme_parkour/README.md``).

The encoder consumes the processed 58x87 depth frame (the env's ``depth_camera``
observation group) plus the 53-d proprioception chunk (delta-yaw slots zeroed) and emits
``latent_dim + 2`` Tanh-bounded values: a depth latent that substitutes for the teacher's
scandots latent, and a 2-d prediction of the (current, next) goal delta-yaw.
"""

from __future__ import annotations

import torch
from torch import nn

from eden.learning.wrappers.rsl_rl.rma.networks import _mlp


class ParkourDepthCNN(nn.Module):
    """58x87 depth frame -> ``latent_dim`` features (the teacher's scan-latent width)."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        activation = nn.ELU()
        # 58x87 -> conv k5 -> 54x83 -> maxpool2 -> 27x41 -> conv k3 -> 25x39.
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(32, 64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, latent_dim),
            activation,
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.net(depth.unsqueeze(1))


class RecurrentDepthEncoder(nn.Module):
    """CNN latent fused with proprioception, integrated by a GRU across camera frames.

    Faithful-port notes: the hidden state is lazily sized on the first forward, carried
    across environment resets (the source never clears it per env), detached — not
    zeroed — between optimizer updates (:meth:`detach_hidden`), and the output head is
    Tanh-bounded (the scandots latent it imitates is Tanh-bounded too).
    """

    def __init__(self, num_prop: int = 53, latent_dim: int = 32, gru_hidden_dim: int = 512):
        super().__init__()
        activation = nn.ELU()  # source-fixed for the whole encoder (not config-resolved)
        self.gru_hidden_dim = gru_hidden_dim
        self.cnn = ParkourDepthCNN(latent_dim)
        self.fuse = _mlp(latent_dim + num_prop, [128, latent_dim], activation, None)
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=gru_hidden_dim, batch_first=True)
        self.head = _mlp(gru_hidden_dim, [latent_dim + 2], activation, nn.Tanh())
        self.hidden: torch.Tensor | None = None

    def forward(self, depth: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        if self.hidden is None or self.hidden.shape[1] != depth.shape[0]:
            self.hidden = torch.zeros(1, depth.shape[0], self.gru_hidden_dim, device=depth.device)
        fused = self.fuse(torch.cat([self.cnn(depth), prop], dim=-1))
        out, self.hidden = self.gru(fused.unsqueeze(1), self.hidden)
        return self.head(out.squeeze(1))

    def detach_hidden(self) -> None:
        """Cut the BPTT graph at the current hidden state (call after each optimizer update)."""
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

    def reset_hidden(self) -> None:
        """Drop the hidden state entirely (fresh rollout, e.g. at play/eval start)."""
        self.hidden = None
