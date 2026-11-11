"""CUDA-resident stand-in for the sync-budget test of ``GoalRangeCurriculum`` (the suite's Genesis runs on CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class CudaGoalEnv:
    """Just what ``GoalRangeCurriculum`` reads, on CUDA: a goal command, a cube, a robot base and episode lengths.

    For counting syncs only; what the term computes is pinned on a real env in ``test_curricula.py``.
    """

    def __init__(self, num_envs: int):
        from eden.managers.terms.commands.oracle import UniformT3Command

        device = torch.device("cuda")
        self.num_envs, self.device = num_envs, device
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        cmd = UniformT3Command.__new__(UniformT3Command)  # unbuilt: only its ranges and command buffer are read
        for name, value in (("x_range", (0.2, 0.5)), ("y_range", (-0.3, 0.3)), ("z_range", (0.6, 0.9))):
            object.__setattr__(cmd, name, value)
        object.__setattr__(cmd, "oracle_pos_b", torch.zeros(num_envs, 3, device=device))
        self.command_manager = SimpleNamespace(get_term=lambda name: cmd)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
        zeros = torch.zeros(num_envs, 3, device=device)
        self.entities = {
            "cube": SimpleNamespace(get_pos=lambda: zeros),
            "robot": SimpleNamespace(get_pos=lambda: zeros, get_quat=lambda: identity),
        }


@pytest.fixture
def cuda_goal_env():
    return CudaGoalEnv(num_envs=8)
