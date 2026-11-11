"""``eden.options.learning.rsl_rl.rl``: the runner options a task declares, as rsl_rl receives them."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eden.options.learning.rsl_rl import (
    RslRlActorCNNOptions,
    RslRlActorOptions,
    RslRlActorRNNOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version, translate_runner_dict


@pytest.mark.parametrize(
    ("options_cls", "class_name", "own_field", "foreign_field"),
    [
        (RslRlActorCNNOptions, "CNNModel", "cnn_cfg", "rnn_type"),
        (RslRlActorRNNOptions, "RNNModel", "rnn_type", "cnn_cfg"),
    ],
)
def test_each_actor_options_class_dumps_its_own_model(options_cls, class_name, own_field, foreign_field):
    """``RslRlActorCNNOptions`` once inherited the RNN bases and dumped ``class_name="RNNModel"``."""
    dumped = options_cls().dict()

    assert dumped["class_name"] == class_name
    assert own_field in dumped and foreign_field not in dumped
    assert "distribution_cfg" in dumped


def test_the_configured_std_type_reaches_the_translated_runner_dict():
    """Both dump APIs feed rsl_rl a whole ``distribution_cfg``; a narrowed one trains with rsl_rl's default
    distribution and leaves the legacy-checkpoint converter nothing to detect.
    """
    from eden.envs.wrappers.rsl_rl_env import _detect_std_type
    from eden.tasks import TASK_REGISTRY

    options = TASK_REGISTRY.get("reacher")().runner_options
    configured = options.actor.distribution_cfg.std_type

    assert _detect_std_type(translate_runner_dict(options.dict())) == configured
    assert _detect_std_type(translate_runner_dict(options.model_dump())) == configured


@pytest.mark.skipif(detect_rsl_rl_major_version() < 5, reason="rsl-rl < 5 does not splat distribution_cfg")
def test_the_distribution_cfg_constructs_under_the_installed_rsl_rl():
    """rsl-rl 5 builds ``dist_class(output_dim, **dist_cfg)``: an unset field must be absent, not ``None``
    (``std_range=None`` raised before the first iteration of every task on the default distribution).
    """
    from tests.support.rsl_rl import resolve_class

    def build(actor):
        dist_cfg = actor.dict()["distribution_cfg"]
        assert None not in dist_cfg.values(), dist_cfg
        dist_class, kwargs = resolve_class(dist_cfg)
        return dist_class(4, **kwargs)

    build(RslRlActorOptions())
    configured = RslRlActorOptions(
        distribution_cfg=RslRlGaussianDistributionOptions(init_std=0.5, std_range=(0.2, 1.0))
    )
    assert build(configured).std_range == [0.2, 1.0]


def test_a_negative_seed_is_refused_at_construction():
    """A negative seed is not a "draw one" sentinel: refused, rather than silently drawing the same value every run."""
    with pytest.raises(ValidationError, match="seed must be >= 0"):
        RslRlOnPolicyRunnerOptions(seed=-1)


@pytest.mark.parametrize("keep", [0, -1])
def test_keeping_fewer_than_one_checkpoint_is_refused_at_construction(keep):
    """Each save would delete the checkpoint it just wrote, leaving ``--resume`` nothing."""
    with pytest.raises(ValueError, match=rf"keep_last_checkpoints={keep}\).*count >= 1.*or None"):
        RslRlBaseRunnerOptions(keep_last_checkpoints=keep)


def test_a_run_keeps_only_its_newest_checkpoint_by_default():
    assert RslRlBaseRunnerOptions().keep_last_checkpoints == 1


@pytest.mark.parametrize("keep", [None, 2])
def test_keep_last_checkpoints_reaches_the_installed_runner_dict(keep):
    options = RslRlOnPolicyRunnerOptions(
        actor=RslRlActorOptions(),
        critic=RslRlCriticOptions(),
        algorithm=RslRlPpoAlgorithmOptions(),
        keep_last_checkpoints=keep,
    )
    assert translate_runner_dict(options.dict())["keep_last_checkpoints"] == keep
