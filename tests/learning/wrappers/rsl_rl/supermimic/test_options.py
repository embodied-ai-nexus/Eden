"""``SuperMimicRunnerOptions``: the base runner options it narrows or refuses."""

import pytest
from pydantic import ValidationError

from eden.learning.wrappers.rsl_rl.supermimic import PolicyOptions, SuperMimicRunnerOptions


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
