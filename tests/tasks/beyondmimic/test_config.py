"""The BeyondMimic task config: its data flags, and the task built and stepped end to end."""

import pydantic
import pytest

from eden.tasks import TASK_REGISTRY
from eden.utils.configs import overrides_from_str
from tests.support.envs import build_env


@pytest.mark.parametrize("flag", ["raw_data_path", "processed_data_path"])
def test_an_empty_data_flag_raises_instead_of_resolving_the_default(flag):
    """An unset shell variable expands to ``""``; treating it as omitted would train on the sample clip."""
    with pytest.raises(ValueError, match=f"--{flag} was given an empty path"):
        TASK_REGISTRY.build("beyondmimic", **{flag: ""})


def test_the_cache_identity_opt_out_is_reachable_through_a_nested_override():
    """``load_cached``'s error advertises this override; a dotted key is the shape that must not work."""
    cfg = TASK_REGISTRY.build("beyondmimic")
    nested = "motion_options: {default: {dataset_options: {strict_cache_identity: false}}}"
    overridden = cfg.with_overrides_from_dict(overrides_from_str(nested))
    assert overridden.motion_options.default.dataset_options.strict_cache_identity is False
    with pytest.raises(pydantic.ValidationError):
        cfg.with_overrides_from_dict(
            overrides_from_str("motion_options.default.dataset_options.strict_cache_identity: false")
        )


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_builds_with_the_motion_end_resync_validator_run_and_steps():
    """Regression: the resync validator's post-build hook made ``env.build()`` raise "Scene is already built".

    BeyondMimic is the only task with ``resync_on_motion_end=True``. ``_rsi_terms`` is checked before the
    first step because the wrap path also fills that lazy cache; before any step only the hook can have.
    """
    cfg = TASK_REGISTRY.build("beyondmimic")
    cfg.env_options.num_envs = 2
    env = build_env(cfg)

    resync = env.motion_manager._resync
    assert resync.enabled
    assert resync._rsi_terms, "MotionEndResync.validate did not run as a post-build hook"

    env.reset()
    for _ in range(2):
        env.step(None)
