"""The two registered AssistMimic tasks and the config edits behind ``main.py``'s diagnostics."""

from __future__ import annotations

import pytest

from eden.options.managers.rewards import RewardGroupOptions
from eden.options.robots.smpl_humanoid import PhcSmplHumanoid
from eden.tasks import TASK_REGISTRY
from eden.tasks.assistmimic.config import (
    BED_POS,
    TASK_FOR_DATASET,
    AssistMimicBedConfig,
    AssistMimicConfig,
    phc_adaptation_config,
    remove_caregiver,
    separate_pair,
)
from eden.tasks.assistmimic.robots import AssistMimicBedRecipient, AssistMimicRecipient


def test_each_dataset_names_its_registered_task():
    assert {d: TASK_REGISTRY.get(t) for d, t in TASK_FOR_DATASET.items()} == {
        "inter-x": AssistMimicConfig,
        "hhi-assist": AssistMimicBedConfig,
    }


@pytest.mark.parametrize("task", sorted(TASK_FOR_DATASET.values()))
def test_no_single_agent_runner_is_declared(task):
    """Two observation groups and two reward groups: ``eden train`` must refuse rather than mistrain."""
    cfg = TASK_REGISTRY.build(task)
    assert cfg.runner_options is None
    assert set(cfg.observation_options.__pydantic_extra__) >= {"caregiver", "recipient"}
    assert set(cfg.reward_options.__pydantic_extra__) >= {"caregiver", "recipient"}


def test_bed_task_weakens_the_recipient_further_and_checks_the_bed():
    inter_x, bed = TASK_REGISTRY.build("assistmimic"), TASK_REGISTRY.build("assistmimic_bed")
    assert type(inter_x.scene_options.recipient) is AssistMimicRecipient
    assert type(bed.scene_options.recipient) is AssistMimicBedRecipient
    assert inter_x.command_options.motion.bed_entity_name == ""
    assert bed.command_options.motion.bed_entity_name == "bed"


def test_separate_pair_moves_the_bed_with_the_recipient():
    cfg = separate_pair(TASK_REGISTRY.build("assistmimic_bed"), offset_y=3.0)
    assert cfg.command_options.motion.recipient_offset == (0.0, 3.0, 0.0)
    assert tuple(cfg.scene_options.bed.default_root_pos) == pytest.approx((BED_POS[0], BED_POS[1] + 3.0, BED_POS[2]))


def test_phc_adaptation_uses_a_full_strength_recipient_and_imitation_rewards_only():
    cfg = phc_adaptation_config(TASK_REGISTRY.build("assistmimic"))
    assert type(cfg.scene_options.recipient) is PhcSmplHumanoid
    rewards = {
        group: {name for name in terms.__pydantic_extra__ if not name.startswith("_")}
        for group, terms in cfg.reward_options.__pydantic_extra__.items()
        if isinstance(terms, RewardGroupOptions)
    }
    assert rewards == {"caregiver": {"cg_imitation", "cg_power"}, "recipient": {"rc_imitation", "rc_power"}}


def test_remove_caregiver_excuses_it_from_the_tracking_termination():
    cfg = remove_caregiver(TASK_REGISTRY.build("assistmimic"), offset_y=10.0)
    assert cfg.command_options.motion.caregiver_offset == (0.0, 10.0, 0.0)
    assert cfg.termination_options.tracking_failure.caregiver_threshold > 1e6
    assert cfg.termination_options.tracking_failure.recipient_threshold == 0.25
