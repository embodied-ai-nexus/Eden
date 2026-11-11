"""SimBaV2 backbone and per-observation-group encoders, ported from Weave.

A faithful port of ``g1_hoi_learning/algorithms/networks.py`` (github.com/xiaohu-art/Weave), the network behind
Weave's G1-Inspire-HOI policy. Pure ``torch`` — no rsl_rl import — so the rsl_rl model wrapper in
:mod:`eden.learning.wrappers.rsl_rl.simba.model` is the only version-coupled piece.

The architecture (SimBaV2, Lee et al. 2025, arXiv:2502.15280) keeps every hidden activation on the unit
hypersphere:

* :class:`HyperEmbedder` appends a constant shift, l2-normalises, projects with a weight-normalised bias-free
  dense layer, scales per dimension and l2-normalises again;
* each :class:`HyperLERPBlock` is ``x <- l2norm(x + alpha * (HyperMLP(x) - x))`` with a learnable per-dimension
  ``alpha``;
* the head is ``HyperDense -> Scaler -> Linear``, the last layer orthogonally initialised with gain 0.01 so the
  initial policy mean is near zero.

Weave feeds it through a :class:`GroupEncoder`: every observation group gets its own small MLP to a shared latent
width, and the latents are concatenated. The groups are split off the (already normalised) concatenated input by
their widths, so the input layout is exactly rsl_rl's ``torch.cat([obs[g] for g in obs_groups], -1)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Project ``x`` onto the unit hypersphere along ``dim``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int
        Dimension to normalise over.
    eps : float
        Floor on the norm, so an all-zero vector maps to zero instead of NaN.

    Returns
    -------
    torch.Tensor
        ``x / max(||x||, eps)``.
    """
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)


class Scaler(nn.Module):
    """Per-dimension learnable scale, ``scaler * (init / scale) * x``.

    The parameter is stored at magnitude ``scale`` and rescaled to ``init`` in the forward pass — SimBaV2's trick
    for giving the scale an effective learning rate independent of its initial value.

    Parameters
    ----------
    dim : int
        Width of the scaled vector.
    init : float
        Effective initial scale.
    scale : float
        Stored magnitude of the parameter.
    """

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scale = float(init) / float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler * self.forward_scale * x


class HyperDense(nn.Module):
    """Bias-free linear layer whose weight is l2-normalised per output unit.

    Parameters
    ----------
    in_features : int
        Input width.
    out_features : int
        Output width.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class HyperEmbedder(nn.Module):
    """Input projection onto the hypersphere: shift -> l2norm -> dense -> scaler -> l2norm.

    Parameters
    ----------
    in_dim : int
        Input width.
    hidden_dim : int
        Output (hidden) width.
    scaler_init : float
        Effective initial value of the scaler.
    scaler_scale : float
        Stored magnitude of the scaler.
    c_shift : float
        Constant appended to the input before normalisation, so the input's norm survives the projection.
    """

    def __init__(self, in_dim: int, hidden_dim: int, scaler_init: float, scaler_scale: float, c_shift: float) -> None:
        super().__init__()
        self.c_shift = float(c_shift)
        self.dense = HyperDense(in_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.full_like(x[..., :1], self.c_shift)
        x = l2normalize(torch.cat([x, shift], dim=-1))
        return l2normalize(self.scaler(self.dense(x)))


class HyperMLP(nn.Module):
    """``HyperDense(up) -> Scaler -> ReLU -> HyperDense(down) -> l2norm``.

    Parameters
    ----------
    hidden_dim : int
        Input width.
    out_dim : int
        Output width.
    expansion : int
        Width multiplier of the inner layer.
    scaler_init : float
        Effective initial value of the inner scaler.
    scaler_scale : float
        Stored magnitude of the inner scaler.
    """

    def __init__(self, hidden_dim: int, out_dim: int, expansion: int, scaler_init: float, scaler_scale: float) -> None:
        super().__init__()
        mid = hidden_dim * expansion
        self.up = HyperDense(hidden_dim, mid)
        self.scaler = Scaler(mid, scaler_init, scaler_scale)
        self.down = HyperDense(mid, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2normalize(self.down(F.relu(self.scaler(self.up(x)))))


class HyperLERPBlock(nn.Module):
    """SimBaV2 residual block with a learnable spherical interpolation, ``l2norm(x + alpha * (MLP(x) - x))``.

    Parameters
    ----------
    hidden_dim : int
        Block width.
    expansion : int
        Width multiplier of the inner :class:`HyperMLP` layer.
    scaler_init : float
        Effective initial value of the inner scaler, before the ``1 / sqrt(expansion)`` correction.
    scaler_scale : float
        Stored magnitude of the inner scaler, before the same correction.
    alpha_init : float
        Effective initial interpolation weight.
    alpha_scale : float
        Stored magnitude of the interpolation weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
    ) -> None:
        super().__init__()
        self.mlp = HyperMLP(
            hidden_dim, hidden_dim, expansion, scaler_init / math.sqrt(expansion), scaler_scale / math.sqrt(expansion)
        )
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2normalize(x + self.alpha(self.mlp(x) - x))


class SimBa(nn.Module):
    """SimBaV2 backbone: :class:`HyperEmbedder` -> ``num_blocks`` x :class:`HyperLERPBlock` -> head.

    Parameters
    ----------
    input_dim : int
        Input width.
    output_dim : int
        Output width (action dimension for the actor, 1 for the critic).
    hidden_dim : int
        Width of the hypersphere the blocks live on.
    num_blocks : int
        Number of residual blocks.
    expansion : int
        Inner-layer width multiplier of each block.
    c_shift : float
        Input shift of the :class:`HyperEmbedder`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion: int = 4,
        c_shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        scaler_init = math.sqrt(2.0 / hidden_dim)
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (num_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)

        self.embedder = HyperEmbedder(input_dim, hidden_dim, scaler_init, scaler_scale, c_shift)
        self.blocks = nn.Sequential(
            *[
                HyperLERPBlock(hidden_dim, expansion, scaler_init, scaler_scale, alpha_init, alpha_scale)
                for _ in range(num_blocks)
            ]
        )
        # Output head: HyperDense -> Scaler -> Linear (the only layer that leaves the sphere).
        self.head_dense = HyperDense(hidden_dim, hidden_dim)
        self.head_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.embedder(x))
        return self.output_proj(self.head_scaler(self.head_dense(x)))


class MLPEncoder(nn.Sequential):
    """One observation group to ``latent_dim``: ``Linear -> act -> ... -> Linear(latent_dim)``.

    Parameters
    ----------
    in_dim : int
        Width of the observation group.
    latent_dim : int
        Output width.
    hidden_dims : list[int]
        Hidden widths; empty makes the encoder a single linear layer.
    activation : nn.Module
        Activation instance; a fresh instance of its class goes between layers (Weave: ``nn.SiLU()``).
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: list[int], activation: nn.Module) -> None:
        layers: list[nn.Module] = []
        for a, b in zip([in_dim, *hidden_dims], hidden_dims):
            layers += [nn.Linear(a, b), type(activation)()]
        layers.append(nn.Linear(hidden_dims[-1] if hidden_dims else in_dim, latent_dim))
        super().__init__(*layers)


class GroupEncoder(nn.Module):
    """Bank of per-observation-group :class:`MLPEncoder` s whose latents are concatenated.

    Parameters
    ----------
    group_dims : list[int]
        Width of each group, in the order the groups are concatenated in the input.
    hidden_dims : list[list[int]]
        Hidden widths of each group's encoder, aligned with ``group_dims``.
    latent_dim : int
        Output width of every encoder; the bank outputs ``latent_dim * len(group_dims)``.
    activation : nn.Module
        Activation used inside every encoder.
    """

    def __init__(
        self, group_dims: list[int], hidden_dims: list[list[int]], latent_dim: int, activation: nn.Module
    ) -> None:
        super().__init__()
        if len(group_dims) != len(hidden_dims):
            raise ValueError(
                f"GroupEncoder got {len(group_dims)} group widths but {len(hidden_dims)} encoder specs; "
                "pass one hidden-dims list per observation group."
            )
        self.group_dims: list[int] = list(group_dims)
        self.in_features = sum(group_dims)
        self.out_features = latent_dim * len(group_dims)
        self.encoders = nn.ModuleList(
            [MLPEncoder(dim, latent_dim, list(dims), activation) for dim, dims in zip(group_dims, hidden_dims)]
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.group_dims, dim=-1)
        latents: list[torch.Tensor] = []
        for i, encoder in enumerate(self.encoders):
            latents.append(encoder(chunks[i]))
        return torch.cat(latents, dim=-1)
