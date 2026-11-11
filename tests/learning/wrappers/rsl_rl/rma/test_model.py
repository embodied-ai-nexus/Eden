"""RmaActorModel built by rsl_rl from the extreme-parkour teacher's runner options, over a synthetic flat observation."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version, translate_runner_dict
from eden.tasks import TASK_REGISTRY

pytestmark = pytest.mark.skipif(
    detect_rsl_rl_version() < MIN_RSL_RL_VERSION, reason="RmaActorModel needs rsl-rl-lib >= 5.5.1"
)

NUM_ENVS = 5
NUM_ACTIONS = 12


def _teacher_cfg() -> dict:
    return translate_runner_dict(TASK_REGISTRY.build("extreme_parkour_teacher").runner_options.dict())


def _obs(width: int = 753, seed: int = 0) -> TensorDict:
    g = torch.Generator().manual_seed(seed)
    return TensorDict({"policy": torch.randn(NUM_ENVS, width, generator=g)}, batch_size=[NUM_ENVS])


def _actor(obs: TensorDict, **overrides):
    from rsl_rl.utils import resolve_class

    cfg = _teacher_cfg()
    actor_cls, actor_kwargs = resolve_class({**cfg["actor"], **overrides})
    return actor_cls(obs, cfg["obs_groups"], "actor", NUM_ACTIONS, **actor_kwargs)


def test_the_option_class_names_resolve_to_the_implementations():
    from rsl_rl.utils import resolve_callable

    from eden.learning.wrappers.rsl_rl.rma import (
        RslRlRmaActorOptions,
        RslRlRmaDepthDistillationAlgorithmOptions,
        RslRlRmaDepthDistillationRunnerOptions,
        RslRlRmaPpoAlgorithmOptions,
    )
    from eden.learning.wrappers.rsl_rl.rma import distillation, model, ppo

    pins = {
        RslRlRmaActorOptions: (model.RMA_ACTOR_MODEL_CLASS_NAME, model.RmaActorModel),
        RslRlRmaPpoAlgorithmOptions: (ppo.RMA_PPO_CLASS_NAME, ppo.RmaPPO),
        RslRlRmaDepthDistillationAlgorithmOptions: (
            distillation.RMA_DEPTH_DISTILLATION_CLASS_NAME,
            distillation.RmaDepthDistillation,
        ),
        RslRlRmaDepthDistillationRunnerOptions: (
            distillation.RMA_DEPTH_DISTILLATION_RUNNER_CLASS_NAME,
            distillation.RmaDepthDistillationRunner,
        ),
    }
    for options, (class_name, implementation) in pins.items():
        assert options.model_fields["class_name"].default == class_name, options.__name__
        assert resolve_callable(class_name) is implementation


def test_the_default_forward_is_the_deployed_composition():
    """Estimator in the privileged-explicit slots, history latent — what eden inference / deploy call."""
    obs = _obs()
    actor = _actor(obs).eval()
    flat = obs["policy"]
    with torch.no_grad():
        expected = actor.net(actor.with_estimate(flat), True)
        assert torch.equal(actor(obs), expected)
        # The switches select the training paths, and each changes the output.
        assert not torch.allclose(actor(obs, estimate=False), expected)
        assert not torch.allclose(actor(obs, hist_encoding=False), expected)


def test_the_estimate_replaces_only_the_privileged_explicit_chunk_and_is_detached():
    obs = _obs()
    actor = _actor(obs)
    flat = obs["policy"]
    estimated = actor.with_estimate(flat)
    pe = actor.dims.priv_explicit_slice
    keep = torch.ones(flat.shape[1], dtype=torch.bool)
    keep[pe] = False
    assert torch.equal(estimated[:, keep], flat[:, keep])
    assert not torch.equal(estimated[:, pe], flat[:, pe])
    assert not estimated.requires_grad


def test_a_mis_sized_actor_observation_names_the_layout_and_the_fix():
    with pytest.raises(ValueError, match=r"752-d.*\[prop 53 \| scan 132.*= 753.*num_prop"):
        _actor(_obs(width=752))


def test_observation_normalization_is_refused():
    obs = _obs()
    with pytest.raises(ValueError, match="obs_normalization=False"):
        _actor(obs, obs_normalization=True)


def test_the_jit_export_scripts_and_matches_the_deployed_forward():
    obs = _obs()
    actor = _actor(obs).eval()
    scripted = torch.jit.script(actor.as_jit())
    with torch.no_grad():
        torch.testing.assert_close(scripted(obs["policy"]), actor(obs))
