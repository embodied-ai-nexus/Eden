"""UWLab's OmniReset state expert: rsl_rl's MLP actor with its empirical observation normalizer."""

from __future__ import annotations

import torch
from torch import nn

from eden.extensions.policies.omnireset.constants import POLICY_ACTION_DIM, POLICY_OBSERVATION_LAYOUT

#: rsl_rl ``EmpiricalNormalization`` divides by ``std + eps`` with this default eps.
NORMALIZER_EPS = 1e-2


class OmniResetActor(nn.Module):
    """``ELU`` MLP 43 -> 512 -> 256 -> 128 -> 64 -> 14 over normalized observations; acts with the mean.

    The head is a heteroscedastic Gaussian: its 14 outputs are the 7 action means, then 7 log standard
    deviations. Inference uses the mean, as rsl_rl's ``act_inference`` does.
    """

    def __init__(self, checkpoint_path: str, device: str | torch.device = "cpu") -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        layout = [tuple(term) for term in checkpoint.get("uwlab_observation_layout", {}).get("target_terms", [])]
        if layout != list(POLICY_OBSERVATION_LAYOUT):
            raise ValueError(
                f"{checkpoint_path} expects observation terms {layout or '(no uwlab_observation_layout)'}, but the "
                f"port builds {list(POLICY_OBSERVATION_LAYOUT)}. Use an Isaac Lab 3 expert from the isaaclab3 "
                "branch of UW-Lab/uwlab-assets (its observations are in declaration order)."
            )
        state = checkpoint["actor_state_dict"]
        sizes = [state["mlp.0.weight"].shape[1]] + [state[f"mlp.{i}.weight"].shape[0] for i in range(0, 9, 2)]
        if sizes[-1] != 2 * POLICY_ACTION_DIM:
            raise ValueError(
                f"{checkpoint_path}: the actor's head has {sizes[-1]} outputs, but the port reads "
                f"{2 * POLICY_ACTION_DIM}: {POLICY_ACTION_DIM} action means, then {POLICY_ACTION_DIM} log standard "
                "deviations (a state-dependent std). A head of action means alone (std as a separate parameter) is "
                "another network; slicing it here would run wrong actions. Use a released OmniReset expert."
            )
        layers: list[nn.Module] = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(fan_in, fan_out))
            if i < len(sizes) - 2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.mlp.load_state_dict(
            {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}
        )
        self.register_buffer("obs_mean", state["obs_normalizer._mean"])
        self.register_buffer("obs_std", state["obs_normalizer._std"])
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Action means ``(N, 7)``: 6 end-effector deltas, then the gripper (closes when negative)."""
        normalized = (obs - self.obs_mean) / (self.obs_std + NORMALIZER_EPS)
        return self.mlp(normalized)[:, :POLICY_ACTION_DIM]
