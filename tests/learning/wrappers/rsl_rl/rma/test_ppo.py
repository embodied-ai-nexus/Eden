"""RmaPPO built by rsl_rl from the extreme-parkour teacher's runner options, over synthetic flat observations."""

from __future__ import annotations

import types

import pytest
import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version, translate_runner_dict
from eden.tasks import TASK_REGISTRY

pytestmark = pytest.mark.skipif(detect_rsl_rl_version() < MIN_RSL_RL_VERSION, reason="RmaPPO needs rsl-rl-lib >= 5.5.1")

NUM_ENVS = 8
NUM_STEPS = 4
NUM_ACTIONS = 12


def _obs(seed: int = 0) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    return TensorDict({"policy": torch.randn(NUM_ENVS, 753, generator=g)}, batch_size=[NUM_ENVS])


def _rma_ppo(critic=None, **alg_overrides):
    """rsl_rl's own ``construct_algorithm`` over the teacher task's runner options (optionally another critic)."""
    from rsl_rl.utils import resolve_callable

    options = TASK_REGISTRY.build("extreme_parkour_teacher").runner_options
    if critic is not None:
        options = options.model_copy(update={"critic": critic})
    cfg = translate_runner_dict(options.dict())
    cfg["algorithm"].update({"num_mini_batches": 2, "num_learning_epochs": 1, **alg_overrides})
    cfg["num_steps_per_env"] = NUM_STEPS
    cfg["multi_gpu"] = None
    env = types.SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS)
    return resolve_callable(cfg["algorithm"]["class_name"]).construct_algorithm(_obs(), env, cfg, "cpu")


def _iterate(alg) -> dict[str, float]:
    """One learning iteration: a rollout, returns, and the update."""
    g = torch.Generator().manual_seed(alg.iteration)
    with torch.inference_mode():
        for step in range(NUM_STEPS):
            alg.act(_obs(seed=step))
            alg.process_env_step(
                _obs(seed=step + 1),
                torch.randn(NUM_ENVS, generator=g),
                torch.zeros(NUM_ENVS, dtype=torch.long),
                {},
            )
        alg.compute_returns(_obs(seed=NUM_STEPS))
    return alg.update()


def test_the_task_trains_with_the_parkour_mirror_and_the_estimator_outside_the_ppo_optimizer():
    alg = _rma_ppo()
    assert type(alg.mirror).__name__ == "ParkourMirror"
    ppo_params = {id(p) for group in alg.optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in ppo_params for p in alg.actor.estimator.parameters())
    assert all(id(p) in ppo_params for p in alg.critic.parameters())


def test_dagger_runs_every_dagger_update_freq_iterations():
    alg = _rma_ppo(dagger_update_freq=2)
    histories = [("hist_latent" in _iterate(alg)) for _ in range(4)]
    assert histories == [True, False, True, False]
    assert alg.iteration == 4
    assert alg.counter == 6  # 4 PPO updates + 2 DAgger updates advance the priv-reg ramp


def test_an_update_trains_estimator_and_policy_and_reports_every_loss():
    alg = _rma_ppo()
    before_estimator = [p.clone() for p in alg.actor.estimator.parameters()]
    before_backbone = [p.clone() for p in alg.actor.net.backbone.parameters()]
    losses = _iterate(alg)
    for key in ("value", "surrogate", "entropy", "priv_reg", "priv_reg_coef", "estimator", "symmetry", "hist_latent"):
        assert torch.isfinite(torch.tensor(losses[key])), key
    assert any(not torch.equal(a, b) for a, b in zip(before_estimator, alg.actor.estimator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_backbone, alg.actor.net.backbone.parameters()))


def test_a_checkpoint_resumes_the_dagger_cadence_and_the_ramp():
    alg = _rma_ppo(dagger_update_freq=3)
    for _ in range(2):
        _iterate(alg)
    saved = alg.save()
    saved["iter"] = alg.iteration  # what the runner adds: the iteration to resume at

    fresh = _rma_ppo(dagger_update_freq=3)
    assert fresh.load(saved, load_cfg=None, strict=True)
    assert (fresh.iteration, fresh.counter) == (alg.iteration, alg.counter)
    for key, value in alg.actor.state_dict().items():
        assert torch.equal(fresh.actor.state_dict()[key], value), key
    assert (
        fresh.estimator_optimizer.state_dict()["state"].keys() == alg.estimator_optimizer.state_dict()["state"].keys()
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mirror_class_name": None}, r"symmetry_coef=1\.0\) needs mirror_class_name="),
        ({"dagger_update_freq": 0}, r"dagger_update_freq=0\) must be >= 1"),
    ],
    ids=["mirror_loss_without_a_mirror", "zero_dagger_frequency"],
)
def test_a_misconfigured_algorithm_says_what_to_set(overrides, message):
    with pytest.raises(ValueError, match=message):
        _rma_ppo(**overrides)


def test_per_mini_batch_advantage_normalization_is_applied_in_the_update():
    """With the flag, ``compute_returns`` leaves the advantages raw; the update must normalize each mini-batch."""
    alg = _rma_ppo(normalize_advantage_per_mini_batch=True)
    with torch.inference_mode():
        for step in range(NUM_STEPS):
            alg.act(_obs(seed=step))
            alg.process_env_step(
                _obs(seed=step + 1), 1e4 * torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS, dtype=torch.long), {}
            )
        alg.compute_returns(_obs(seed=NUM_STEPS))
    assert alg.storage.advantages.abs().max() > 100.0  # raw, as stock compute_returns leaves them under the flag
    assert abs(alg.update()["surrogate"]) < 10.0  # O(1): each mini-batch was normalized before the surrogate


def test_a_recurrent_critic_is_refused():
    from eden.options.learning.rsl_rl import RslRlCriticRNNOptions

    with pytest.raises(ValueError, match=r"feed-forward critic, got the recurrent RNNModel"):
        _rma_ppo(critic=RslRlCriticRNNOptions(hidden_dims=[16], obs_normalization=False))


def test_an_actor_state_dict_of_another_model_is_refused_whatever_strict_says():
    alg = _rma_ppo()
    saved = alg.save()
    saved["actor_state_dict"] = {"mlp.0.weight": torch.zeros(4, 4), "distribution.std_param": torch.ones(12)}
    with pytest.raises(ValueError, match=r"not an RmaActorModel's: missing \[.*'estimator\..*unexpected \['mlp\.0"):
        alg.load(saved, load_cfg=None, strict=False)


def test_a_rsl_rl_3_teacher_checkpoint_is_refused_before_anything_is_converted_or_loaded(tmp_path):
    from eden.envs.wrappers.rsl_rl_env import load_runner_checkpoint

    alg = _rma_ppo()
    v3 = {  # the rsl-rl-lib 3 stack's layout: one model_state_dict (actor.* / critic.* / std) + the estimator apart
        "model_state_dict": {
            "actor.scan_encoder.0.weight": torch.zeros(128, 132),
            "critic.0.weight": torch.zeros(512, 753),
            "std": torch.ones(NUM_ACTIONS),
        },
        "estimator_state_dict": {"0.weight": torch.zeros(128, 53)},
        "optimizer_state_dict": {},
        "iter": 100,
    }
    path = tmp_path / "model_100.pt"
    torch.save(v3, path)

    def must_not_load(*_args, **_kwargs):
        raise AssertionError("the checkpoint reached runner.load")

    cfg = translate_runner_dict(TASK_REGISTRY.build("extreme_parkour_teacher").runner_options.dict())
    runner = types.SimpleNamespace(alg=alg, cfg=cfg, device="cpu", load=must_not_load)
    with pytest.raises(RuntimeError, match=r"actor is a RmaActorModel with no 'mlp\.\*' keys .*Nothing was converted"):
        load_runner_checkpoint(runner, str(path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model_100.pt"]  # no .converted.pt sibling written
