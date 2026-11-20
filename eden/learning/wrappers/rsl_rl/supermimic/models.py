"""SuperMimic networks: the dynamics model, the multi-encoder actor and the shared MLP and normalizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch import nn

if TYPE_CHECKING:
    from eden.learning.wrappers.rsl_rl.supermimic.goals import GoalSpec
    from eden.learning.wrappers.rsl_rl.supermimic.representation import GenCoordRepresentation


class RunningNormalizer(nn.Module):
    """Normalizes by a running mean and standard deviation over every sample seen: ``(x - mean) / (std + eps)``.

    The statistics update in place, so a captured CUDA graph that reads them stays valid, and only in train mode.
    """

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.training:
            return
        x = x.reshape(-1, x.shape[-1])
        count = x.shape[0]
        batch_mean, batch_var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + count
        rate = count / total
        delta = batch_mean - self.mean
        self.mean.add_(rate * delta)
        self.var.mul_(1 - rate).add_(rate * batch_var + rate * (1 - rate) * delta.square())
        self.std.copy_(self.var.sqrt())
        self.count.copy_(total)


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """``input -> (Linear -> LayerNorm -> GELU) per hidden width -> Linear(output)``."""
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU()]
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    """The differentiable world model ``(model state, action, aux state, sim context) -> (next aux, next model state)``.

    ``action`` is the raw policy output, the input the policy also sees as its last action; the network reads the
    joint position target ``action * scale + offset`` it commands.
    """

    def __init__(
        self,
        representation: GenCoordRepresentation,
        num_actions: int,
        sim_ctx_dim: int,
        action_scale: torch.Tensor,
        action_offset: torch.Tensor,
        hidden_dims: tuple[int, ...],
    ):
        super().__init__()
        self.representation = representation
        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_offset", action_offset.clone())
        self.state_normalizer = RunningNormalizer(representation.rep_state_dim)
        self.action_normalizer = RunningNormalizer(num_actions)
        self.mlp = build_mlp(
            representation.rep_state_dim + num_actions + sim_ctx_dim, hidden_dims, representation.prediction_dim
        )

    def target(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale + self.action_offset

    def forward(
        self, rep_state: torch.Tensor, action: torch.Tensor, aux_state: torch.Tensor, sim_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self.state_normalizer(rep_state), self.action_normalizer(self.target(action)), sim_ctx], -1)
        with torch.autocast("cuda", torch.bfloat16, enabled=x.is_cuda):
            prediction = self.mlp(x).float()
        return self.representation.apply_prediction(aux_state, rep_state, prediction)

    def normalizers(self) -> tuple[RunningNormalizer, RunningNormalizer]:
        return self.state_normalizer, self.action_normalizer


class MultiEncoderActor(nn.Module):
    """One goal encoder per task and a shared decoder over a tanh-bounded latent and the observation history.

    The batch is split into contiguous slots, one per task (``slot_mix``), and each slot's goal goes through its
    task's encoder. In training, every other task's latent is pulled towards the first task's latent of the same
    state (``latent_bc_weight``), which updates only that task's encoder.

    The input ``TensorDict`` holds ``history`` ``(B, H, obs_dim)``, the newest frame last, and ``goal_<task>`` per task.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_specs: list[GoalSpec],
        slot_mix: dict[str, float],
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        latent_bc_weight: float,
        history_length: int,
    ):
        super().__init__()
        if not goal_specs:
            raise ValueError("MultiEncoderActor needs at least one goal spec.")
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_bc_weight = latent_bc_weight
        self.goal_specs = goal_specs
        self._names = [spec.name for spec in goal_specs]
        self._aux_loss = torch.zeros(0)
        self.obs_normalizer = RunningNormalizer(obs_dim)
        self.goal_normalizers = nn.ModuleDict({s.name: RunningNormalizer(s.goal_dim) for s in goal_specs})
        self.encoders = nn.ModuleDict({s.name: build_mlp(s.goal_dim, hidden_dims, latent_dim) for s in goal_specs})
        self.decoder = build_mlp(latent_dim + history_length * obs_dim, hidden_dims, action_dim)
        self.set_slot_mix(slot_mix)

    def set_slot_mix(self, slot_mix: dict[str, float]) -> None:
        """Set each task's share of the batch; the shares must sum to 1, and a task left out gets none."""
        unknown = sorted(set(slot_mix) - set(self._names))
        if unknown:
            raise ValueError(f"slot_mix names unknown task(s) {unknown}; the tasks are {self._names}.")
        fractions = [slot_mix.get(name, 0.0) for name in self._names]
        if any(f < 0.0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError(f"slot_mix shares must be non-negative and sum to 1, got {slot_mix}.")
        self._fractions = fractions

    def slot_ranges(self, batch_size: int) -> list[tuple[int, int]]:
        """Each task's ``(lo, hi)`` rows of a batch; the last task takes the remainder."""
        ranges, lo = [], 0
        for i, fraction in enumerate(self._fractions):
            hi = batch_size if i == len(self._fractions) - 1 else lo + int(round(batch_size * fraction))
            ranges.append((lo, hi))
            lo = hi
        return ranges

    def forward(self, obs: TensorDict) -> torch.Tensor:
        slots = self.slot_ranges(obs.shape[0])
        latents = [
            torch.tanh(self.encoders[name](self.goal_normalizers[name](obs[f"goal_{name}"][lo:hi])))
            for name, (lo, hi) in zip(self._names, slots, strict=True)
            if hi > lo
        ]
        self._aux_loss = self._latent_bc_loss(obs, slots)
        history = self.obs_normalizer(obs["history"]).flatten(start_dim=1)
        return self.decoder(torch.cat([torch.cat(latents, dim=0), history], dim=-1))

    def _latent_bc_loss(self, obs: TensorDict, slots: list[tuple[int, int]]) -> torch.Tensor:
        """Per-row loss pulling each non-anchor task's latent towards the anchor task's (zero in eval mode)."""
        loss = torch.zeros(obs.shape[0], device=obs.device)
        if not self.training or self.latent_bc_weight == 0.0:
            return loss
        anchor = self._names[0]
        anchor_goal = self.goal_normalizers[anchor](obs[f"goal_{anchor}"]).detach()
        for name, (lo, hi) in zip(self._names[1:], slots[1:], strict=True):
            if hi == lo:
                continue
            latent = torch.tanh(self.encoders[name](self.goal_normalizers[name](obs[f"goal_{name}"][lo:hi]).detach()))
            with torch.no_grad():
                target = torch.tanh(self.encoders[anchor](anchor_goal[lo:hi]))
            loss[lo:hi] = self.latent_bc_weight * ((latent - target) ** 2).mean(-1)
        return loss

    def auxiliary_loss(self) -> torch.Tensor:
        return self._aux_loss

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        self.obs_normalizer.update(obs["history"][:, -1])
        for name, (lo, hi) in zip(self._names, self.slot_ranges(obs.shape[0]), strict=True):
            if hi > lo:
                self.goal_normalizers[name].update(obs[f"goal_{name}"][lo:hi])
