"""AssistMimic's rl_games network and agent (caregiver + recipient as two agents of one Eden env)."""

import torch

from eden.tasks.assistmimic.custom_terms.commands import ROLES


def interleave(obs: dict) -> torch.Tensor:
    """``{"caregiver": (N, D), "recipient": (N, D)}`` -> ``(2N, D)``, row ``2e + r`` role ``r`` of env ``e``.

    The layout of rl_games' two-agent wrapper (``RlGamesPpoRunnerOptions.agents``).
    """
    return torch.stack([obs[r] for r in ROLES], dim=1).flatten(0, 1)


def deinterleave(actions: torch.Tensor) -> torch.Tensor:
    """``(2N, A)`` interleaved agent actions -> ``(N, 2A)`` env action (caregiver term first)."""
    return actions.reshape(-1, 2 * actions.shape[1])
