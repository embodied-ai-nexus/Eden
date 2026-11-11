"""MuonPPO on rsl-rl-lib >= 5: the optimizer split, a real PPO update, the adaptive schedule, checkpoint state."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version
from tests.support.simba import NUM_ACTIONS, group_obs, runner_dict, simba_models

pytestmark = pytest.mark.skipif(detect_rsl_rl_major_version() < 5, reason="MuonPPO needs rsl-rl-lib >= 5")


def _muon_ppo(obs: TensorDict, num_steps: int = 4, **alg_overrides):
    from rsl_rl.storage import RolloutStorage
    from tests.support.rsl_rl import resolve_class

    actor, critic = simba_models(obs)
    alg_cls, alg_kwargs = resolve_class({**runner_dict()["algorithm"], **alg_overrides})
    alg_kwargs.pop("share_cnn_encoders", None)
    storage = RolloutStorage("rl", obs.batch_size[0], num_steps, obs, [NUM_ACTIONS], "cpu")
    return alg_cls(actor, critic, storage, device="cpu", **alg_kwargs)


def _collect(alg, obs: TensorDict, num_steps: int = 4) -> None:
    g = torch.Generator().manual_seed(3)
    for step in range(num_steps):
        alg.act(obs)
        rewards = torch.randn(obs.batch_size[0], generator=g)
        dones = torch.zeros(obs.batch_size[0], dtype=torch.long)
        alg.process_env_step(group_obs(seed=10 + step), rewards, dones, {})
    alg.compute_returns(obs)


def test_the_option_class_name_resolves_to_the_algorithm():
    from tests.support.rsl_rl import resolve_class

    from eden.learning.wrappers.rsl_rl.muon_ppo.ppo import MUON_PPO_CLASS_NAME, MuonPPO

    algorithm = runner_dict()["algorithm"]
    assert algorithm["class_name"] == MUON_PPO_CLASS_NAME
    assert resolve_class(algorithm)[0] is MuonPPO


def test_muon_takes_every_2d_weight_and_adamw_everything_else():
    from eden.learning.wrappers.rsl_rl.muon_ppo.ppo import MuonAdamW

    obs = group_obs()
    actor, critic = simba_models(obs)
    opt = MuonAdamW([actor, critic], lr=1e-4)
    muon, adamw = opt.optimizers
    assert isinstance(muon, torch.optim.Muon) and isinstance(adamw, torch.optim.AdamW)
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in adamw.param_groups for p in g["params"]}
    params = [*actor.parameters(), *critic.parameters()]
    assert muon_ids == {id(p) for p in params if p.dim() == 2}
    assert adamw_ids == {id(p) for p in params if p.dim() != 2}
    assert id(actor.distribution.std_param) in adamw_ids
    assert adamw.param_groups[0]["weight_decay"] == 0.01 and muon.param_groups[0]["weight_decay"] == 0.1
    # One lr write through the combined param_groups (what the adaptive schedule does) reaches both.
    for group in opt.param_groups:
        group["lr"] = 3e-5
    assert {g["lr"] for o in opt.optimizers for g in o.param_groups} == {3e-5}


def test_a_ppo_update_moves_both_optimizers_parameters_and_logs_explained_variance():
    obs = group_obs(n=16)
    alg = _muon_ppo(obs, num_mini_batches=2, num_learning_epochs=1)
    before = {n: p.detach().clone() for n, p in [*alg.actor.named_parameters(), *alg.critic.named_parameters()]}
    _collect(alg, obs)
    losses = alg.update()
    assert {"value", "surrogate", "entropy", "explained_variance"} <= set(losses)
    after = dict([*alg.actor.named_parameters(), *alg.critic.named_parameters()])
    moved = {n for n, p in after.items() if not torch.equal(p.detach(), before[n])}
    assert "mlp.1.output_proj.weight" in moved  # a Muon parameter
    assert "distribution.std_param" in moved  # an AdamW parameter


def test_the_adaptive_schedule_writes_one_learning_rate_into_both_optimizers():
    obs = group_obs(n=16)
    # desired_kl so small that any policy change overshoots: the schedule must cut the lr.
    alg = _muon_ppo(
        obs, num_mini_batches=2, num_learning_epochs=2, schedule="adaptive", desired_kl=1e-12, learning_rate=1e-3
    )
    _collect(alg, obs)
    alg.update()
    assert alg.learning_rate < 1e-3
    assert {g["lr"] for o in alg.optimizer.optimizers for g in o.param_groups} == {alg.learning_rate}


def test_optimizer_state_round_trips_and_a_plain_adam_checkpoint_is_refused_by_name():
    obs = group_obs(n=16)
    alg = _muon_ppo(obs, num_mini_batches=2, num_learning_epochs=1)
    _collect(alg, obs)
    alg.update()
    saved = alg.save()
    fresh = _muon_ppo(obs)
    fresh.load(saved, load_cfg=None, strict=True)
    assert fresh.learning_rate == alg.learning_rate
    # The combined param_groups still reach the loaded optimizers, so the schedule keeps steering both.
    fresh.optimizer.param_groups[-1]["lr"] = 7e-6
    assert fresh.optimizer.optimizers[-1].param_groups[-1]["lr"] == 7e-6
    assert len(fresh.optimizer.optimizers[0].state) == len(alg.optimizer.optimizers[0].state) > 0
    adam_state = torch.optim.Adam(fresh.actor.parameters()).state_dict()
    with pytest.raises(ValueError, match="stock PPO's Adam.*load_cfg=\\{'optimizer': False\\}"):
        fresh.optimizer.load_state_dict(adam_state)
