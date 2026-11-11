"""Gaussian-splat parameter containers (single scene and per-env batch).

Vendored from https://github.com/discoverse-dev/GaussianRenderer.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GaussianData:
    """A single set of Gaussians (shared across all environments)."""

    def __init__(self, xyz, rot, scale, opacity, sh):
        self.xyz = xyz  # (N, 3)
        self.rot = rot  # (N, 4) wxyz
        self.scale = scale  # (N, 3)
        self.opacity = opacity  # (N,)
        self.sh = sh  # (N, K, 3) or (N, 3*K)

    def __len__(self):
        return len(self.xyz)

    @property
    def device(self):
        return self.xyz.device

    def to_cuda(self):
        if not torch.is_tensor(self.xyz) or not self.xyz.is_cuda:
            self.xyz = torch.tensor(self.xyz).float().cuda().requires_grad_(False)
            self.rot = torch.tensor(self.rot).float().cuda().requires_grad_(False)
            self.scale = torch.tensor(self.scale).float().cuda().requires_grad_(False)
            self.opacity = torch.tensor(self.opacity).float().cuda().requires_grad_(False)
            self.sh = torch.tensor(self.sh).float().cuda().requires_grad_(False)
        return self


@dataclass
class GaussianBatchData:
    """Per-environment Gaussians produced by :func:`batch_update_gaussians`."""

    xyz: Tensor  # (B, N, 3)
    rot: Tensor  # (B, N, 4) wxyz
    scale: Tensor  # (B, N, 3)
    opacity: Tensor  # (B, N)
    sh: Tensor  # (B, N, K, 3) or (B, N, 3*K)

    def __len__(self):
        return self.xyz.shape[1]

    @property
    def batch_size(self):
        return self.xyz.shape[0]

    @property
    def device(self):
        return self.xyz.device
