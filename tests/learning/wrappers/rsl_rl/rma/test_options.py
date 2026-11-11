"""The RMA option classes refuse, at construction, what the models and algorithms would trip over after the build."""

from __future__ import annotations

import pytest

from eden.learning.wrappers.rsl_rl.rma import (
    RslRlRmaActorOptions,
    RslRlRmaDepthDistillationAlgorithmOptions,
    RslRlRmaDepthDistillationRunnerOptions,
    RslRlRmaPpoAlgorithmOptions,
)
from eden.options.learning.rsl_rl import RslRlRndOptions


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"num_hist": 8}, r"num_hist=8\): the history encoder's conv stack is sized for exactly 10 frames"),
        ({"scan_encoder_dims": []}, r"scan_encoder_dims=\[\] — the last width of each is a latent width"),
        ({"obs_normalization": True}, r"obs_normalization=True\): the actor slices its flat observation by position"),
    ],
    ids=["history_length", "empty_encoder", "obs_normalization"],
)
def test_the_actor_options_refuse_a_layout_the_model_cannot_build(fields, message):
    with pytest.raises(ValueError, match=message):
        RslRlRmaActorOptions(**fields)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"priv_reg_coef_schedule": [0.0, 0.1, 2000]}, r"must be 4 numbers: \[start, end, ramp_start, ramp_length\]"),
        ({"priv_reg_coef_schedule": [0.0, 0.1, 2000, 0]}, r"ramp_length \(4th\) > 0 updates"),
        ({"dagger_update_freq": 0}, r"dagger_update_freq=0\) must be >= 1 iteration"),
        ({"symmetry_coef": -1.0}, r"symmetry_coef=-1\.0\) must be >= 0"),
        ({"symmetry_coef": 1.0}, r"symmetry_coef=1\.0\) needs mirror_class_name='module\.path:Class'"),
        ({"rnd_cfg": RslRlRndOptions()}, r"does not support \{'rnd_cfg'"),
    ],
    ids=["schedule_length", "zero_ramp", "dagger_frequency", "negative_mirror_weight", "mirror_without_class", "rnd"],
)
def test_the_ppo_options_refuse_what_rma_ppo_would_trip_over(fields, message):
    with pytest.raises(ValueError, match=message):
        RslRlRmaPpoAlgorithmOptions(**fields)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"update_interval": 0}, r"\{'update_interval': 0\} out of range"),
        ({"num_pretrain_iters": -1, "yaw_scale": 0.0}, r"\{'num_pretrain_iters': -1, 'yaw_scale': 0\.0\} out of range"),
        ({"yaw_slots": [6, 9]}, r"yaw_slots=\[6, 9\]\) must be \[start, start \+ 2\]"),
    ],
    ids=["update_interval", "several_at_once", "yaw_slots"],
)
def test_the_distillation_options_refuse_out_of_range_knobs(fields, message):
    with pytest.raises(ValueError, match=message):
        RslRlRmaDepthDistillationAlgorithmOptions(**fields)


def test_the_student_runner_options_tie_the_iteration_length_to_the_depth_cadence():
    with pytest.raises(
        ValueError, match=r"num_steps_per_env=12\) must be a positive multiple of .*update_interval \(5\)"
    ):
        RslRlRmaDepthDistillationRunnerOptions(num_steps_per_env=12)
    with pytest.raises(ValueError, match=r"yaw_slots=\[52, 54\] runs past the 53-d proprioception chunk"):
        RslRlRmaDepthDistillationRunnerOptions(
            num_steps_per_env=10, algorithm=RslRlRmaDepthDistillationAlgorithmOptions(yaw_slots=[52, 54])
        )
    RslRlRmaDepthDistillationRunnerOptions(num_steps_per_env=10)  # a multiple of the default cadence builds
