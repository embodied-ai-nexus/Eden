"""HIM (History Implicit Model) estimator — the self-supervised learner from HIMLoco.

OpenHomie's ``HIMEstimator`` (``HomieRL/rsl_rl/rsl_rl/modules/him_estimator.py``). It encodes the
proprioceptive history into (a) an estimate of the base linear velocity and (b) an
L2-normalized dynamics latent, trained with a velocity MSE loss against the privileged
next base velocity plus a SwAV swapped-prediction loss against a slow target encoder of the
next observation. It is trained by its **own** Adam optimizer (not the PPO optimizer).

Layout it assumes: the critic observation is one actor frame followed by the true base linear
velocity (``VELOCITY_DIM``). The SwAV target encoder sees the next critic observation minus its first
``target_skip_dim`` entries — OpenHomie drops the three velocity commands a frame starts with.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# The estimator's explicit output: the base linear velocity, the critic observation's last entries.
VELOCITY_DIM = 3


def _get_activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }[name]


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp normalization producing the SwAV assignment targets."""
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()
    for _ in range(iters):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_height_points: int = 0,
        enc_hidden_dims=(256, 256),
        tar_hidden_dims=(256, 256),
        latent_dim: int = 32,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 64,
        temperature: float = 3.0,
        target_skip_dim: int = 3,
    ):
        super().__init__()
        act = _get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_height_points = num_height_points
        self.num_latent = latent_dim
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.target_skip_dim = target_skip_dim

        # Encoder: history -> velocity(3) + latent
        enc_input_dim = self.temporal_steps * self.num_one_step_obs + self.num_height_points
        enc_layers: list[nn.Module] = []
        for hidden in enc_hidden_dims:
            enc_layers += [nn.Linear(enc_input_dim, hidden), act]
            enc_input_dim = hidden
        enc_layers += [nn.Linear(enc_input_dim, VELOCITY_DIM + latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target encoder: one-step next obs -> latent
        tar_input_dim = self.num_one_step_obs + VELOCITY_DIM - target_skip_dim
        tar_layers: list[nn.Module] = []
        for hidden in tar_hidden_dims:
            tar_layers += [nn.Linear(tar_input_dim, hidden), act]
            tar_input_dim = hidden
        tar_layers += [nn.Linear(tar_input_dim, latent_dim)]
        self.target = nn.Sequential(*tar_layers)

        # SwAV prototypes
        self.proto = nn.Embedding(num_prototype, latent_dim)

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :VELOCITY_DIM], parts[..., VELOCITY_DIM:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def update(self, obs_history: torch.Tensor, next_critic_obs: torch.Tensor, lr: float | None = None):
        """One estimator gradient step; returns ``(estimation_loss, swap_loss)`` as floats.

        ``next_critic_obs`` is ``[frame(num_one_step_obs), base_lin_vel(VELOCITY_DIM)]``: the velocity is the
        regression target, and everything after the first ``target_skip_dim`` entries is the SwAV target.
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + VELOCITY_DIM].detach()
        next_obs = next_critic_obs.detach()[:, self.target_skip_dim : self.num_one_step_obs + VELOCITY_DIM]

        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :VELOCITY_DIM], z_s[..., VELOCITY_DIM:]

        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        losses = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()
