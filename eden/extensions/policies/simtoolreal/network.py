"""Pure-torch reimplementation of the SimToolReal rl_games LSTM actor.

The released checkpoint is an rl_games ``A2CBuilder`` network
(``SimToolRealLSTMAsymmetricPPO``) with input normalization, an SAPG
``extra_param`` conditioning vector, an LSTM-before-MLP trunk, and a continuous
``mu`` head. We rebuild only the deterministic actor path (``mu``) in plain
torch so the policy runs in-process inside Eden with no rl_games / gym / IsaacGym
dependency.

Forward (deterministic):

    x = clamp((obs - mean) / sqrt(var + 1e-5), -5, 5)          # 140, normalize_input
    x = cat([x, extra_params[0]])                              # 172, SAPG coef_cond row 0
    h = LSTM(x, (h, c))                                        # 1024, before_mlp
    h = LayerNorm(h)
    h = MLP[1024,1024,512,512] with ELU                        # 512
    action = mu(h)                                             # 29, mu_activation = None

``extra_params[0]`` is the row the deploy code selects: it appends a constant
``50.0`` sentinel that matches no ``coef_id``, so rl_games' ``argmax`` falls back
to index 0. We bake that in directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eden.extensions.policies.simtoolreal.constants import (
    MLP_UNITS,
    NORM_CLAMP,
    NORM_EPS,
    NUM_JOINTS,
    NUM_OBS,
    RNN_HIDDEN,
    RNN_INPUT_DIM,
    SAPG_EXTRA_DIM,
)


class SimToolRealActor(nn.Module):
    """Deterministic actor (mu) of the SimToolReal SAPG-LSTM policy."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

        # Input normalization stats (registered as buffers, not learned here).
        self.register_buffer("obs_mean", torch.zeros(NUM_OBS))
        self.register_buffer("obs_var", torch.ones(NUM_OBS))
        self.register_buffer("coef", torch.zeros(SAPG_EXTRA_DIM))  # extra_params[0]

        self.rnn = nn.LSTM(RNN_INPUT_DIM, RNN_HIDDEN, num_layers=1)
        self.layer_norm = nn.LayerNorm(RNN_HIDDEN)

        mlp_layers: list[nn.Module] = []
        in_dim = RNN_HIDDEN
        for out_dim in MLP_UNITS:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ELU())
            in_dim = out_dim
        self.actor_mlp = nn.Sequential(*mlp_layers)
        self.mu = nn.Linear(in_dim, NUM_JOINTS)

        self._load_checkpoint(checkpoint_path)
        self.to(self.device)
        self.eval()

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # SAPG checkpoint is keyed by agent index; agent 0 is the deployed actor.
        agent = ckpt[0] if isinstance(ckpt, dict) and 0 in ckpt else ckpt
        sd = agent["model"]

        self.obs_mean.copy_(sd["running_mean_std.running_mean"].float())
        self.obs_var.copy_(sd["running_mean_std.running_var"].float())
        self.coef.copy_(sd["a2c_network.extra_params"][0].float())

        # rl_games stores LSTM weights under a2c_network.rnn.rnn.* with the
        # standard nn.LSTM parameter names.
        self.rnn.weight_ih_l0.data.copy_(sd["a2c_network.rnn.rnn.weight_ih_l0"])
        self.rnn.weight_hh_l0.data.copy_(sd["a2c_network.rnn.rnn.weight_hh_l0"])
        self.rnn.bias_ih_l0.data.copy_(sd["a2c_network.rnn.rnn.bias_ih_l0"])
        self.rnn.bias_hh_l0.data.copy_(sd["a2c_network.rnn.rnn.bias_hh_l0"])

        self.layer_norm.weight.data.copy_(sd["a2c_network.layer_norm.weight"])
        self.layer_norm.bias.data.copy_(sd["a2c_network.layer_norm.bias"])

        for i, layer_idx in enumerate(range(0, 2 * len(MLP_UNITS), 2)):
            lin = self.actor_mlp[layer_idx]
            lin.weight.data.copy_(sd[f"a2c_network.actor_mlp.{layer_idx}.weight"])
            lin.bias.data.copy_(sd[f"a2c_network.actor_mlp.{layer_idx}.bias"])

        self.mu.weight.data.copy_(sd["a2c_network.mu.weight"])
        self.mu.bias.data.copy_(sd["a2c_network.mu.bias"])

    def init_state(self, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero LSTM (h, c) state for ``num_envs`` parallel rollouts."""
        h = torch.zeros(1, num_envs, RNN_HIDDEN, device=self.device)
        c = torch.zeros(1, num_envs, RNN_HIDDEN, device=self.device)
        return h, c

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """obs: (N, 140) -> action (N, 29), carrying LSTM state."""
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + NORM_EPS)
        x = torch.clamp(x, -NORM_CLAMP, NORM_CLAMP)
        x = torch.cat([x, self.coef.expand(x.shape[0], -1)], dim=-1)  # (N, 172)

        out, state = self.rnn(x.unsqueeze(0), state)  # (1, N, 1024)
        h = self.layer_norm(out.squeeze(0))
        h = self.actor_mlp(h)
        action = self.mu(h)
        # rl_games player: clip_actions=True, action_space=Box(-1, 1). rescale to
        # [-1, 1] is identity, so the deployed action is just the clamped mu.
        action = torch.clamp(action, -1.0, 1.0)
        return action, state
