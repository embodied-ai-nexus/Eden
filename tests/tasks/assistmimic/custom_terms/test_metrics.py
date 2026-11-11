"""AssistMimic's metrics refuse a configuration that would read 0, or the wrong track, every step."""

from __future__ import annotations

import pytest

from tests.support.envs import build_env
from tests.tasks.assistmimic.conftest import assistmimic_config


@pytest.mark.requires_genesis
def test_a_power_metric_without_dofs_is_refused_at_build(clips_file):
    cfg = assistmimic_config("assistmimic", clips_file)
    cfg.metric_options.power_caregiver.dofs_name = []
    with pytest.raises(ValueError, match=r"JointPower on entity 'caregiver': dofs_name is empty"):
        build_env(cfg)


@pytest.mark.requires_genesis
def test_an_imitation_reward_with_an_unknown_role_is_refused_at_build(clips_file):
    cfg = assistmimic_config("assistmimic", clips_file)
    cfg.metric_options.imitation_caregiver.role = "nurse"
    with pytest.raises(
        ValueError,
        match=r"PhcImitationReward on entity '[a-z]+': role 'nurse' is not one of \('caregiver', 'recipient'\)",
    ):
        build_env(cfg)
