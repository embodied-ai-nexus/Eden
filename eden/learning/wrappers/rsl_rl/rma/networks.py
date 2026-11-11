"""RMA actor networks: scan / privileged / history encoders, the actor backbone, the velocity estimator.

Implemented from the architecture described in "Extreme Parkour with Legged Robots" (Cheng, Shi,
Agarwal, Pathak, 2023, https://extreme-parkour.github.io). The upstream implementations
(chengxuxin/extreme-parkour, CC BY-NC 4.0; CAI23sbP/Isaaclab_Parkour, GPL-3.0) were used as
behavioral references only — no code is copied from either.

The actor sees ``[prop | scan latent | priv_explicit | adaptation latent]`` where the
adaptation latent comes from the privileged encoder during PPO and from the proprioception
history encoder during DAgger iterations and at deployment. Plain torch: the rsl_rl model
interface lives in :mod:`eden.learning.wrappers.rsl_rl.rma.model`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from eden.learning.wrappers.rsl_rl.rma.dims import RmaObsDims


def _mlp(
    input_dim: int, layer_dims: Sequence[int], activation: nn.Module, final_activation: nn.Module | None
) -> nn.Sequential:
    """Linear stack with `activation` between layers and `final_activation` (or nothing) after the last."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for i, out_dim in enumerate(layer_dims):
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(layer_dims) - 1:
            layers.append(activation)
        elif final_activation is not None:
            layers.append(final_activation)
        in_dim = out_dim
    return nn.Sequential(*layers)


class StateHistoryEncoder(nn.Module):
    """Encodes the (num_hist, num_prop) proprioception history into the adaptation latent.

    Per-frame linear projection followed by a temporal Conv1d stack. The output matches the
    privileged encoder's latent width so DAgger can regress one onto the other.
    """

    def __init__(self, num_prop: int, num_hist: int, output_dim: int, activation: nn.Module, channel_size: int = 10):
        super().__init__()
        if num_hist != 10:
            raise ValueError(f"StateHistoryEncoder supports a history of exactly 10 frames, got {num_hist}")
        self.num_hist = num_hist
        self.frame_encoder = nn.Sequential(nn.Linear(num_prop, 3 * channel_size), activation)
        self.conv = nn.Sequential(
            nn.Conv1d(3 * channel_size, 2 * channel_size, kernel_size=4, stride=2),
            activation,
            nn.Conv1d(2 * channel_size, channel_size, kernel_size=2, stride=1),
            activation,
            nn.Flatten(),
        )
        # 10 frames -> conv lengths 4 -> 3, flattened to channel_size * 3 features.
        self.output = nn.Sequential(nn.Linear(3 * channel_size, output_dim), activation)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        batch, tsteps, num_prop = hist.shape
        frames = self.frame_encoder(hist.reshape(batch * tsteps, num_prop))
        features = self.conv(frames.reshape(batch, tsteps, -1).permute(0, 2, 1))
        return self.output(features)


class RmaActor(nn.Module):
    """Actor backbone over [prop, scan latent, priv_explicit, adaptation latent]."""

    def __init__(
        self,
        dims: RmaObsDims,
        num_actions: int,
        scan_encoder_dims: Sequence[int],
        priv_encoder_dims: Sequence[int],
        actor_hidden_dims: Sequence[int],
        activation: nn.Module,
        tanh_backbone_output: bool = False,
    ):
        super().__init__()
        self.dims = dims
        # Chunk bounds as plain ints (not the dims' slices) so the module stays TorchScript-able for export.
        self._prop_end = dims.num_prop
        self._scan = (dims.scan_slice.start, dims.scan_slice.stop)
        self._priv_explicit = (dims.priv_explicit_slice.start, dims.priv_explicit_slice.stop)
        self._priv_latent = (dims.priv_latent_slice.start, dims.priv_latent_slice.stop)
        self._hist = (dims.hist_slice.start, dims.hist_slice.stop)
        self._num_hist = dims.num_hist
        # Scan encoder ends in Tanh: the scandots obs is clipped to [-1, 1] and the latent is
        # kept in the same range so a depth latent can substitute for it during distillation.
        self.scan_encoder = _mlp(dims.num_scan, scan_encoder_dims, activation, nn.Tanh())
        self.scan_latent_dim = scan_encoder_dims[-1]
        self.priv_encoder = _mlp(dims.num_priv_latent, priv_encoder_dims, activation, activation)
        latent_dim = priv_encoder_dims[-1]
        self.history_encoder = StateHistoryEncoder(dims.num_prop, dims.num_hist, latent_dim, activation)
        backbone_in = dims.num_prop + scan_encoder_dims[-1] + dims.num_priv_explicit + latent_dim
        self.backbone = _mlp(
            backbone_in,
            [*actor_hidden_dims, num_actions],
            activation,
            nn.Tanh() if tanh_backbone_output else None,
        )

    def forward(
        self, obs: torch.Tensor, hist_encoding: bool, scandots_latent: torch.Tensor | None = None
    ) -> torch.Tensor:
        scan_latent = self.infer_scandots_latent(obs) if scandots_latent is None else scandots_latent
        latent = self.infer_hist_latent(obs) if hist_encoding else self.infer_priv_latent(obs)
        prop = obs[:, : self._prop_end]
        priv_explicit = obs[:, self._priv_explicit[0] : self._priv_explicit[1]]
        return self.backbone(torch.cat([prop, scan_latent, priv_explicit, latent], dim=-1))

    def infer_priv_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.priv_encoder(obs[:, self._priv_latent[0] : self._priv_latent[1]])

    def infer_hist_latent(self, obs: torch.Tensor) -> torch.Tensor:
        hist = obs[:, self._hist[0] : self._hist[1]]
        return self.history_encoder(hist.reshape(-1, self._num_hist, self._prop_end))

    def infer_scandots_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.scan_encoder(obs[:, self._scan[0] : self._scan[1]])


class Estimator(nn.Module):
    """Predicts the privileged-explicit chunk (scaled base linear velocity) from proprioception."""

    def __init__(self, num_prop: int, num_priv_explicit: int, hidden_dims: Sequence[int], activation: nn.Module):
        super().__init__()
        self.mlp = _mlp(num_prop, [*hidden_dims, num_priv_explicit], activation, None)

    def forward(self, prop: torch.Tensor) -> torch.Tensor:
        return self.mlp(prop)
