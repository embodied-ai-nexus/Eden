"""The homie task's sagittal mirror, through rsl-rl's data-augmentation contract."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from eden.tasks.homie.symmetry import ACTION_DIM, CRITIC_DIM, FRAME_DIM, POLICY_DIM, mirror_observations_and_actions


def _partner(name: str) -> str:
    return name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")


def _mirrored(group: str, x: torch.Tensor) -> torch.Tensor:
    aug, _ = mirror_observations_and_actions(obs=TensorDict({group: x}, batch_size=[x.shape[0]]))
    torch.testing.assert_close(aug[group][: x.shape[0]], x)  # originals first
    return aug[group][x.shape[0] :]


def _critic(frame: torch.Tensor) -> torch.Tensor:
    return torch.cat([frame, torch.zeros(3)]).unsqueeze(0)


def test_mirroring_twice_is_the_identity():
    obs = TensorDict({"policy": torch.randn(5, POLICY_DIM), "critic": torch.randn(5, CRITIC_DIM)}, batch_size=[5])
    once, actions = mirror_observations_and_actions(obs=obs, actions=torch.randn(5, ACTION_DIM))
    twice, actions_twice = mirror_observations_and_actions(obs=once[5:], actions=actions[5:])
    torch.testing.assert_close(twice[5:].to_dict(), obs.to_dict())
    torch.testing.assert_close(actions_twice[5:], actions[:5])


def test_dofs_swap_left_and_right_and_negate_roll_and_yaw():
    """Oracle from the joint names alone, independent of the hand-written pair table."""
    names = list(G1_DOFS29_NAME)
    frame = torch.zeros(FRAME_DIM)
    frame[10:39] = torch.arange(1.0, 30.0)  # dofs_pos block: value = 1 + declaration index
    out = _mirrored("critic", _critic(frame))[0, 10:39]
    for i, name in enumerate(names):
        sign = -1.0 if ("roll" in name or "yaw" in name) else 1.0
        assert out[i].item() == sign * (1 + names.index(_partner(name))), name


def test_lateral_and_yaw_commands_negate():
    frame = torch.zeros(FRAME_DIM)
    frame[:4] = torch.tensor([0.5, 0.2, 0.3, 0.6])  # vx, vy, wz, height
    torch.testing.assert_close(_mirrored("critic", _critic(frame))[0, :4], torch.tensor([0.5, -0.2, -0.3, 0.6]))


def test_a_layout_the_maps_do_not_cover_is_refused():
    with pytest.raises(ValueError, match="400-wide tensor has no mirror map"):
        _mirrored("policy", torch.zeros(2, 400))
    with pytest.raises(KeyError, match="'depth'"):
        mirror_observations_and_actions(obs=TensorDict({"depth": torch.zeros(2, 4)}, batch_size=[2]))
