"""``OmniResetActor``: the released checkpoint's contract, on tiny checkpoints written in its layout."""

from __future__ import annotations

import pytest
import torch

from eden.extensions.policies.omnireset.constants import POLICY_OBSERVATION_LAYOUT
from eden.extensions.policies.omnireset.network import NORMALIZER_EPS, OmniResetActor


def _checkpoint(path, layout=POLICY_OBSERVATION_LAYOUT):
    """A tiny actor in the released checkpoint's layout: 43 -> 8 -> 8 -> 8 -> 8 -> 14."""
    torch.manual_seed(0)
    sizes = [43, 8, 8, 8, 8, 14]
    state = {}
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        state[f"mlp.{2 * i}.weight"] = torch.randn(fan_out, fan_in)
        state[f"mlp.{2 * i}.bias"] = torch.randn(fan_out)
    state["obs_normalizer._mean"] = torch.randn(1, 43)
    state["obs_normalizer._std"] = torch.rand(1, 43) + 0.5
    torch.save({"actor_state_dict": state, "uwlab_observation_layout": {"target_terms": list(layout)}}, path)
    return state


def test_the_actor_normalizes_then_acts_with_the_mean_half_of_its_head(tmp_path):
    state = _checkpoint(tmp_path / "actor.pt")
    actor = OmniResetActor(str(tmp_path / "actor.pt"))
    obs = torch.randn(3, 43)
    x = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + NORMALIZER_EPS)
    for i in range(5):
        x = x @ state[f"mlp.{2 * i}.weight"].T + state[f"mlp.{2 * i}.bias"]
        x = torch.nn.functional.elu(x) if i < 4 else x
    torch.testing.assert_close(actor(obs), x[:, :7])


def test_a_checkpoint_in_another_observation_layout_is_refused(tmp_path):
    reordered = POLICY_OBSERVATION_LAYOUT[-1:] + POLICY_OBSERVATION_LAYOUT[:-1]  # UWLab's pre-migration order
    _checkpoint(tmp_path / "old.pt", layout=reordered)
    with pytest.raises(
        ValueError, match=r"expects observation terms .*insertive_asset_in_receptive_asset_frame.*isaaclab3"
    ):
        OmniResetActor(str(tmp_path / "old.pt"))


def test_a_head_of_action_means_alone_is_refused(tmp_path):
    state = _checkpoint(tmp_path / "actor.pt")
    state["mlp.8.weight"], state["mlp.8.bias"] = state["mlp.8.weight"][:7], state["mlp.8.bias"][:7]
    torch.save(
        {"actor_state_dict": state, "uwlab_observation_layout": {"target_terms": list(POLICY_OBSERVATION_LAYOUT)}},
        tmp_path / "actor.pt",
    )
    with pytest.raises(ValueError, match=r"head has 7 outputs, but the port reads 14"):
        OmniResetActor(str(tmp_path / "actor.pt"))
