"""SimBaModel on rsl-rl-lib >= 5: shapes, one encoder per group, normalisation, JIT export, the compiled trunk."""

from __future__ import annotations

import copy

import pytest
import torch

from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version
from tests.support.simba import GROUP_DIMS, NUM_ACTIONS, OBS_GROUPS, group_obs, runner_dict, simba_models

pytestmark = pytest.mark.skipif(detect_rsl_rl_major_version() < 5, reason="SimBaModel needs rsl-rl-lib >= 5")


def test_the_option_class_name_resolves_to_the_model():
    from tests.support.rsl_rl import resolve_class

    from eden.learning.wrappers.rsl_rl.simba.model import SIMBA_MODEL_CLASS_NAME, SimBaModel

    runner = runner_dict()
    assert runner["actor"]["class_name"] == runner["critic"]["class_name"] == SIMBA_MODEL_CLASS_NAME
    assert resolve_class(runner["actor"])[0] is SimBaModel


def test_actor_and_critic_output_shapes_and_one_encoder_per_group_in_obs_set_order():
    obs = group_obs()
    actor, critic = simba_models(obs)
    assert actor(obs).shape == (8, NUM_ACTIONS)
    assert actor(obs, stochastic_output=True).shape == (8, NUM_ACTIONS)
    assert critic(obs).shape == (8, 1)
    for model, obs_set in ((actor, "actor"), (critic, "critic")):
        encoder = model.mlp[0]
        assert encoder.group_dims == [GROUP_DIMS[g] for g in OBS_GROUPS[obs_set]]
        assert encoder.out_features == 128 * len(OBS_GROUPS[obs_set])
    assert len(actor.mlp[1].blocks) == 1 and len(critic.mlp[1].blocks) == 2
    # A per-action scalar std at 1.0.
    torch.testing.assert_close(actor.output_std, torch.ones(8, NUM_ACTIONS))


def test_the_group_split_follows_the_concatenation_so_one_group_moves_only_its_own_latent():
    obs = group_obs()
    actor, _ = simba_models(obs)
    actor.eval()
    encoder = actor.mlp[0]
    base = encoder(actor.get_latent(obs))
    moved = obs.clone()
    moved["object"] = moved["object"] + 1.0
    delta = (encoder(actor.get_latent(moved)) - base).abs().reshape(8, 4, 128).sum(dim=(0, 2))
    assert delta[2] > 0 and torch.all(delta[[0, 1, 3]] == 0)


def test_unknown_encoder_group_names_raise_with_the_available_groups():
    obs = group_obs()
    with pytest.raises(ValueError, match="\\['propio'\\].*available groups"):
        simba_models(obs, encoder_hidden_dims={"propio": [256]})


def test_update_normalization_moves_the_running_mean():
    obs = group_obs()
    actor, _ = simba_models(obs)
    before = actor.obs_normalizer.mean.clone()
    actor.update_normalization(group_obs(seed=1))
    assert not torch.allclose(actor.obs_normalizer.mean, before)


def test_jit_export_matches_the_eager_actor():
    obs = group_obs()
    actor, _ = simba_models(obs)
    actor.eval()
    scripted = torch.jit.script(actor.as_jit())
    flat = torch.cat([obs[g] for g in OBS_GROUPS["actor"]], dim=-1)
    torch.testing.assert_close(scripted(flat), actor(obs))


@pytest.mark.slow
def test_the_compiled_trunk_matches_eager_and_survives_deepcopy():
    obs = group_obs()
    eager, _ = simba_models(obs)
    compiled, _ = simba_models(obs, compile=True)
    assert (eager._compile_trunk, compiled._compile_trunk) == (False, True)
    compiled.load_state_dict(eager.state_dict())
    eager.eval()
    compiled.eval()
    torch.testing.assert_close(compiled(obs), eager(obs), rtol=1e-5, atol=1e-6)
    clone = copy.deepcopy(compiled)
    torch.testing.assert_close(clone(obs), eager(obs), rtol=1e-5, atol=1e-6)
    assert set(compiled.state_dict()) == set(eager.state_dict())
