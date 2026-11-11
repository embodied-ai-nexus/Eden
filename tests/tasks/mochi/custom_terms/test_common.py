"""Build-time checks of the terms the Mochi deformable tasks share, on ``mochi_wipe_the_bar``."""

import pytest

from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from eden.tasks.mochi import custom_terms


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_grip_gate_on_a_missing_metric_raises_at_build():
    """GripHeld (and ReachSoftBody) read the grip metric every step, so a bad name fails the build instead."""
    config = TASK_REGISTRY.build("mochi_wipe_the_bar")
    config.env_options.num_envs = 1
    config.reward_options.held = custom_terms.GripHeld.configure(
        entity_name="sponge", grip_metric_name="jaw_grip", weight=0.25
    )
    with pytest.raises(KeyError, match=r"Metric 'jaw_grip' not found. Active metrics: \[.*'grip'"):
        RLEnvBase.from_config(config).build()
