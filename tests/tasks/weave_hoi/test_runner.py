"""Weave's runner preset: ``configs/track/train.yaml``'s hyperparameters, and a saved config reloading into it."""

from __future__ import annotations

import pytest

from eden.tasks.weave_hoi.runner import WEAVE_OBS_GROUPS, RslRlWeaveRunnerOptions, weave_runner_options

#: Weave's value for every SimBa field, for both networks (``configs/track/train.yaml``).
_SIMBA = {
    "hidden_dim": 512,
    "expansion": 4,
    "latent_dim": 128,
    "encoder_hidden_dims": {group: [256] for group in sorted({g for gs in WEAVE_OBS_GROUPS.values() for g in gs})},
    "default_encoder_hidden_dims": [256],
    "c_shift": 3.0,
    "activation": "swish",
    "obs_normalization": True,
    "compile": True,
}
#: Weave's PPO + Muon values.
_ALGORITHM = {
    "learning_rate": 1e-4,
    "schedule": "adaptive",
    "desired_kl": 0.01,
    "entropy_coef": 5e-4,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_param": 0.2,
    "use_clipped_value_loss": True,
    "value_loss_coef": 1.0,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "muon_weight_decay": 0.1,
}


def test_weave_runner_options_carry_the_train_yaml_hyperparameters_and_round_trip():
    ro = weave_runner_options(experiment_name="weave_hoi")
    for network, num_blocks in ((ro.actor, 1), (ro.critic, 2)):
        assert {name: getattr(network, name) for name in _SIMBA} == _SIMBA
        assert network.num_blocks == num_blocks
    assert (ro.actor.distribution_cfg.init_std, ro.actor.distribution_cfg.std_type) == (1.0, "scalar")
    assert {name: getattr(ro.algorithm, name) for name in _ALGORITHM} == _ALGORITHM
    assert (ro.num_steps_per_env, ro.obs_groups) == (32, WEAVE_OBS_GROUPS)
    assert (ro.save_interval, ro.keep_last_checkpoints) == (100, 1)
    # The narrowed runner class reloads its own dump, subclass fields included.
    assert RslRlWeaveRunnerOptions(**ro.dict()) == ro


@pytest.mark.parametrize(
    ("network", "fields"),
    [
        ("actor", [*_SIMBA, "num_blocks", "distribution_cfg"]),
        ("critic", [*_SIMBA, "num_blocks"]),
        ("algorithm", list(_ALGORITHM)),
    ],
)
def test_every_weave_value_is_set_by_the_preset_not_inherited_from_a_shared_default(network, fields):
    """The option classes live in eden.learning and serve other tasks; changing their defaults must not move Weave."""
    options = getattr(weave_runner_options(), network)
    assert set(fields) - options.model_fields_set == set()
