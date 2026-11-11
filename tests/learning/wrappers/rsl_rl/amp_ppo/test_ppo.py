"""AmpPPO on rsl-rl-lib >= 5, built by rsl_rl from the husky task's own runner options over synthetic observations."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from eden.extensions.policies.husky.constants import NUM_JOINTS
from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version, translate_runner_dict
from eden.tasks import TASK_REGISTRY
from tests.support.ddp import ddp_spawn

pytestmark = pytest.mark.skipif(detect_rsl_rl_version() < MIN_RSL_RL_VERSION, reason="AmpPPO needs rsl-rl-lib >= 5.5.1")

NUM_ENVS = 6
NUM_STEPS = 4
NUM_ACTIONS = 3
WINDOW = 5 * NUM_JOINTS


def _obs(gate: torch.Tensor, seed: int = 0) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    amp = torch.cat([0.3 * torch.randn(NUM_ENVS, WINDOW, generator=g), gate.float().unsqueeze(-1)], dim=-1)
    return TensorDict(
        {"policy": torch.randn(NUM_ENVS, 7, generator=g), "critic": torch.randn(NUM_ENVS, 9, generator=g), "amp": amp},
        batch_size=[NUM_ENVS],
    )


def _husky_amp_ppo(obs: TensorDict, multi_gpu: dict | None = None, **alg_overrides):
    """rsl_rl's own ``construct_algorithm`` over the husky runner options (its ``algorithm`` is the AMP options)."""
    from rsl_rl.utils import resolve_callable

    options = TASK_REGISTRY.build("husky").runner_options
    cfg = translate_runner_dict(options.dict())
    cfg["algorithm"].update(alg_overrides)
    cfg["num_steps_per_env"] = NUM_STEPS
    cfg["multi_gpu"] = multi_gpu
    env = types.SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS)
    return resolve_callable(cfg["algorithm"]["class_name"]).construct_algorithm(obs, env, cfg, "cpu")


def test_the_option_class_name_resolves_to_the_algorithm():
    from rsl_rl.utils import resolve_callable

    from eden.learning.wrappers.rsl_rl.amp_ppo import RslRlAmpPpoAlgorithmOptions
    from eden.learning.wrappers.rsl_rl.amp_ppo.ppo import AMP_PPO_CLASS_NAME, AmpPPO

    assert RslRlAmpPpoAlgorithmOptions.model_fields["class_name"].default == AMP_PPO_CLASS_NAME
    assert resolve_callable(AMP_PPO_CLASS_NAME) is AmpPPO


def test_the_style_reward_lands_only_where_the_gate_is_set():
    gate = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.bool)
    obs = _obs(gate)
    alg = _husky_amp_ppo(obs)
    alg.act(obs)
    rewards = torch.zeros(NUM_ENVS)
    alg.process_env_step(obs, rewards, torch.zeros(NUM_ENVS, dtype=torch.long), {})

    stored = alg.storage.rewards[0].squeeze(-1)
    assert (stored[~gate] == 0.0).all()
    assert (stored[gate] > 0.0).all()  # LSGAN reward is non-negative and a fresh discriminator scores near 0
    assert len(alg.amp_replay) == NUM_ENVS  # every window is banked, gated or not


def test_an_update_trains_the_discriminator_and_a_checkpoint_round_trips_it():
    gate = torch.ones(NUM_ENVS, dtype=torch.bool)
    obs = _obs(gate)
    alg = _husky_amp_ppo(obs, num_mini_batches=2, num_learning_epochs=1)
    before = {k: v.clone() for k, v in alg.discriminator.state_dict().items()}
    for step in range(NUM_STEPS):
        alg.act(obs)
        alg.process_env_step(
            _obs(gate, seed=step + 1), torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.long), {}
        )
    alg.compute_returns(obs)
    losses = alg.update()

    for key in ("amp", "amp_grad_pen", "amp_expert_d", "amp_policy_d", "amp_style_reward"):
        assert torch.isfinite(torch.tensor(losses[key])), key
    assert any(not torch.equal(before[k], v) for k, v in alg.discriminator.state_dict().items())

    fresh = _husky_amp_ppo(obs)
    fresh.load(alg.save(), load_cfg=None, strict=True)
    for key, value in alg.discriminator.state_dict().items():
        assert torch.equal(fresh.discriminator.state_dict()[key], value), key
    assert torch.equal(fresh.amp_normalizer.mean, alg.amp_normalizer.mean)


def test_a_mis_sized_amp_group_names_the_expected_layout():
    obs = _obs(torch.ones(NUM_ENVS, dtype=torch.bool))
    alg = _husky_amp_ppo(obs)
    short = obs.clone()
    short["amp"] = short["amp"][:, 1:]
    alg.act(short)
    with pytest.raises(ValueError, match=r"5 frames x 23 joints.*history_length"):
        alg.process_env_step(short, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.long), {})


def _rollout_and_update(alg, gate: torch.Tensor, seed: int = 0) -> dict[str, float]:
    with torch.inference_mode():
        for step in range(NUM_STEPS):
            alg.act(_obs(gate, seed=seed + step))
            alg.process_env_step(
                _obs(gate, seed=seed + step + 1), torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.long), {}
            )
        alg.compute_returns(_obs(gate, seed=seed + NUM_STEPS))
    return alg.update()


def test_construction_checks_the_amp_group_against_the_first_observation():
    obs = _obs(torch.ones(NUM_ENVS, dtype=torch.bool))
    without = obs.exclude("amp")
    with pytest.raises(ValueError, match=r"amp_obs_group='amp'.*groups: \['critic', 'policy'\]"):
        _husky_amp_ppo(without)
    short = obs.clone()
    short["amp"] = short["amp"][:, 1:]
    with pytest.raises(ValueError, match=r"5 frames x 23 joints"):
        _husky_amp_ppo(short)


def test_a_plain_ppo_checkpoint_is_refused_unless_amp_warm_start_says_so():
    obs = _obs(torch.ones(NUM_ENVS, dtype=torch.bool))
    plain = {key: value for key, value in _husky_amp_ppo(obs).save().items() if not key.startswith("amp_")}
    plain["iter"] = 7

    with pytest.raises(ValueError, match=r"no AMP state .*trained as plain PPO.*amp_warm_start=True"):
        _husky_amp_ppo(obs).load(plain, load_cfg=None, strict=True)

    warm = _husky_amp_ppo(obs, amp_warm_start=True)
    fresh_discriminator = {k: v.clone() for k, v in warm.discriminator.state_dict().items()}
    assert warm.load(plain, load_cfg=None, strict=True)
    for key, value in warm.discriminator.state_dict().items():
        assert torch.equal(value, fresh_discriminator[key]), key  # the discriminator starts fresh


def test_replay_options_are_stock_ppo_with_the_same_ppo_fields():
    from eden.learning.wrappers.rsl_rl.amp_ppo import RslRlAmpPpoAlgorithmOptions
    from eden.options.learning.rsl_rl import RslRlPpoAlgorithmOptions

    amp = TASK_REGISTRY.build("husky").runner_options.algorithm
    assert isinstance(amp, RslRlAmpPpoAlgorithmOptions)
    replay = amp.for_inference()
    assert type(replay) is RslRlPpoAlgorithmOptions and replay.class_name == "PPO"
    assert replay.dict() == {**{k: v for k, v in amp.dict().items() if not k.startswith("amp_")}, "class_name": "PPO"}


def test_preflight_loads_the_expert_clips_and_names_a_bad_file(tmp_path):
    from eden.learning.wrappers.rsl_rl.amp_ppo import RslRlAmpPpoAlgorithmOptions

    TASK_REGISTRY.build("husky").runner_options.algorithm.preflight()  # the shipped clips load
    short = tmp_path / "short.npz"
    np.savez(short, motion_0=np.zeros((3, NUM_JOINTS)), fps=50.0, num_motions=1)
    with pytest.raises(ValueError, match=r"motion_0 in .*short\.npz is too short \(3 frames\)"):
        RslRlAmpPpoAlgorithmOptions(amp_motion_file=str(short)).preflight()


def _ddp_amp_worker(rank: int, world_size: int, out_dir: str) -> None:
    torch.manual_seed(100 + rank)  # each rank starts from its own init, as rank-offset seeds give it
    gate = torch.ones(NUM_ENVS, dtype=torch.bool)
    alg = _husky_amp_ppo(_obs(gate), multi_gpu={"global_rank": rank, "world_size": world_size})
    initial = {k: v.clone() for k, v in alg.discriminator.state_dict().items()}
    alg.broadcast_parameters()
    _rollout_and_update(alg, gate, seed=10 * rank)  # each rank rolls out its own data
    torch.save(
        {
            "initial": initial,
            "discriminator": alg.discriminator.state_dict(),
            "normalizer": alg.amp_normalizer.state_dict(),
        },
        f"{out_dir}/rank_{rank}.pt",
    )


def test_under_ddp_every_rank_keeps_the_same_discriminator_and_normalizer():
    with ddp_spawn(_ddp_amp_worker, world_size=2) as out_dir:
        ranks = [torch.load(out_dir / f"rank_{rank}.pt", weights_only=False) for rank in range(2)]
    assert any(not torch.equal(ranks[0]["initial"][k], ranks[1]["initial"][k]) for k in ranks[0]["initial"])
    for part in ("discriminator", "normalizer"):
        for key, value in ranks[0][part].items():
            assert torch.allclose(value, ranks[1][part][key]), f"{part}.{key}"
    assert any(not torch.equal(ranks[0]["initial"][k], v) for k, v in ranks[0]["discriminator"].items())
