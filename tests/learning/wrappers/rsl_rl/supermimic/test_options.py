"""``SuperMimicRunnerOptions``: the base runner options it narrows or refuses, and the ranges of its own."""

import pytest
from pydantic import ValidationError

from eden.learning.wrappers.rsl_rl.supermimic import (
    PolicyOptions,
    ReturnSamplerOptions,
    SuperMimicRunnerOptions,
    WorldModelOptions,
)


def test_the_runner_is_named_by_import_path():
    assert SuperMimicRunnerOptions().class_name == "eden.learning.wrappers.rsl_rl.supermimic.runner:SuperMimicRunner"


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("clip_actions", 1.0, "the world model"),
        ("obs_groups", {"actor": ["policy"]}, "actor_observation_group"),
        ("resume", True, "eden train --checkpoint"),
        ("load_run", "run_3", "eden train --checkpoint"),
    ],
)
def test_a_base_runner_option_the_runner_does_not_read_is_refused_with_its_reason(field, value, reason):
    with pytest.raises(ValidationError, match=rf"{field}=.*{reason}"):
        SuperMimicRunnerOptions(**{field: value})


def test_the_neptune_logger_names_the_loggers_the_runner_writes_to():
    with pytest.raises(ValidationError, match=r"logger='neptune' \(the runner writes to TensorBoard or W&B"):
        SuperMimicRunnerOptions(logger="neptune")


def test_the_options_are_not_an_rsl_rl_config():
    from eden.options.learning.rsl_rl.rl import RslRlOnPolicyRunnerOptions

    assert RslRlOnPolicyRunnerOptions.translates_to_rsl_rl
    assert not SuperMimicRunnerOptions.translates_to_rsl_rl


def test_the_goal_window_needs_at_least_one_frame():
    with pytest.raises(ValidationError, match=r"goal_frames \(0\)"):
        PolicyOptions(goal_frames=0)


def test_the_actor_critic_needs_a_discount_below_one():
    with pytest.raises(ValidationError, match=r"0 < gamma < 1 and 0 <= lam <= 1, got gamma=1.0, lam=0.95"):
        PolicyOptions(gamma=1.0)
    assert PolicyOptions(algorithm="bptt", gamma=1.0).gamma == 1.0


def test_an_actor_average_decay_outside_zero_to_one_is_refused():
    with pytest.raises(ValidationError, match=r"ema_decay must be in \[0, 1\), got 1.0"):
        PolicyOptions(ema_decay=1.0)
    assert PolicyOptions(ema_decay=0.995).ema_decay == 0.995


@pytest.mark.parametrize(
    "options, message",
    [
        ({"lr_multiplier": 0.0}, r"lr_multiplier must be positive, got 0.0"),
        ({"lr_decay_return": 1.5}, r"lr_decay_return is a normalized return, in \(0, 1\], got 1.5"),
        ({"lr_decay_iterations": 0}, r"lr_decay_iterations must be at least 1, got 0"),
    ],
    ids=["multiplier", "return_range", "ramp_length"],
)
def test_a_learning_rate_schedule_that_cannot_run_is_refused(options, message):
    with pytest.raises(ValidationError, match=message):
        PolicyOptions(**options)


def test_a_mirror_share_outside_the_batch_is_refused():
    with pytest.raises(ValidationError, match=r"mirror_share is a share of the batch, in \[0, 1\], got 1.5"):
        WorldModelOptions(mirror_share=1.5)


def test_each_option_of_the_recipe_can_be_set_back_to_the_papers_value_on_its_own():
    assert PolicyOptions(lr_multiplier=1.0).lr_decay_return == 0.75  # the ramp's options may stay
    assert not PolicyOptions(reference_action=False).reference_action
    assert PolicyOptions(ema_decay=0.0).ema_decay == 0.0
    assert WorldModelOptions(mirror_share=0.0).mirror_share == 0.0
    assert ReturnSamplerOptions(failure=0.0).explore == 0.35
