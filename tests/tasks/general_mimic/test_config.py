"""Tests for ``eden.tasks.general_mimic.config``: the data and architecture TaskMods."""

from __future__ import annotations

import operator

import pytest

from eden.tasks import TASK_REGISTRY

_SONIC_FLAT = {
    "runner_options.actor.hidden_dims": [2048, 2048, 1024, 1024, 512, 512],
    "runner_options.critic.hidden_dims": [2048, 2048, 1024, 1024, 512, 512],
    "runner_options.actor.activation": "swish",  # rsl_rl's name for SiLU
    "runner_options.critic.activation": "swish",
    "runner_options.actor.distribution_cfg.init_std": 0.05,
    "runner_options.actor.distribution_cfg.std_type": "log",
}


def _read(cfg, path: str):
    return operator.attrgetter(path)(cfg)


def test_default_build_is_the_lafan1_snapshot_recipe():
    """The default must stay bit-identical to the LAFAN1 runs it was tuned on."""
    cfg = TASK_REGISTRY.build("general_mimic")
    motion = cfg.motion_options.default
    assert motion.dataset_options.name == "lafan1_retargeted_g1_dataset"
    assert motion.adaptive_motion_weights is True
    assert motion.motion_weight_update_mode == "snapshot"
    assert motion.motion_weight_uniform_mix == pytest.approx(0.1)
    assert motion.motion_weight_failure_rate_max_over_mean == pytest.approx(200.0)
    assert motion.log_per_motion_metrics is True
    assert motion.log_motion_id_to_filename is False
    for net in (cfg.runner_options.actor, cfg.runner_options.critic):
        assert (net.hidden_dims, net.activation) == ([512, 256, 128], "elu")
    assert cfg.runner_options.actor.distribution_cfg.init_std == pytest.approx(1.0)
    assert cfg.runner_options.actor.distribution_cfg.std_type == "scalar"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"motion_weight_update_mode": "compound", "motion_weight_uniform_mix": 0.0},
            {
                "motion_options.default.motion_weight_update_mode": "compound",
                "motion_options.default.motion_weight_uniform_mix": 0.0,
            },
        ),
        ({"log_per_motion_metrics": False}, {"motion_options.default.log_per_motion_metrics": False}),
        ({"log_motion_id_to_filename": True}, {"motion_options.default.log_motion_id_to_filename": True}),
        ({"dataset_name": "amass"}, {"motion_options.default.dataset_options.name": "amass_retargeted_g1_dataset"}),
        ({"arch_preset": "sonic_flat"}, _SONIC_FLAT),
        (
            {"arch_preset": "sonic_flat", "log_per_motion_metrics": False, "motion_weight_update_mode": "compound"},
            {
                **_SONIC_FLAT,
                "motion_options.default.log_per_motion_metrics": False,
                "motion_options.default.motion_weight_update_mode": "compound",
            },
        ),
    ],
    ids=["compound-mode", "no-per-motion-logging", "filename-resolver", "amass", "sonic-flat", "arch-and-data"],
)
def test_mod_flags_reach_the_config(kwargs, expected):
    cfg = TASK_REGISTRY.build("general_mimic", **kwargs)
    assert {path: _read(cfg, path) for path in expected} == expected


@pytest.mark.parametrize(("flag", "value"), [("arch_preset", "sonic_h2"), ("dataset_name", "dex_ycb")])
def test_unknown_choice_is_rejected(flag, value):
    """The Literal keeps object-manipulation datasets and unknown presets out of a humanoid task."""
    with pytest.raises(ValueError, match="not one of"):
        TASK_REGISTRY.build("general_mimic", **{flag: value})
