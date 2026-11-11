"""Tests for ``eden.tasks.manipulation.custom_terms.rewards``: the lift gates' resting-height source.

A gate takes a scalar ``rest_z`` or a per-env height from a metric (``rest_z_metric``, as
``manipulation_lift_ycb_rgb`` does).
"""

from __future__ import annotations

import pytest
import torch

from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.managers.metrics import MetricManagerOptions
from eden.tasks.manipulation.custom_terms.rewards import lift_height
from eden.tasks.manipulation.lift_ycb_rgb.config import ObjectRestHeight
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis


def _config(metric_options: MetricManagerOptions | None) -> EdenConfig:
    return EdenConfig(
        env_options=EnvOptions(num_envs=2),
        scene_options=SceneOptions(cube=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.0, 0.0, 0.5))),
        metric_options=metric_options,
    )


@pytest.fixture(scope="module")
def env():
    return build_env(
        _config(
            MetricManagerOptions(
                rest=ObjectRestHeight.configure(object_name="cube"),
                rest_sum=ObjectRestHeight.configure(object_name="cube", is_cumulative=True),
            )
        )
    )


def _place_cube(env, heights):
    pos = torch.zeros(len(heights), 3, device=env.device)
    pos[:, 2] = torch.tensor(heights, device=env.device)
    env.entities["cube"].set_pos(pos)


def test_lift_height_measures_from_the_per_env_metric_or_the_scalar(env):
    _place_cube(env, [0.5, 0.7])
    env.metric_manager.compute()  # the metric latches each env's current height
    _place_cube(env, [0.6, 0.75])

    torch.testing.assert_close(
        lift_height(env, object_name="cube", rest_z_metric="rest"), torch.tensor([0.1, 0.05], device=env.device)
    )
    torch.testing.assert_close(
        lift_height(env, object_name="cube", rest_z=0.5), torch.tensor([0.1, 0.25], device=env.device)
    )


@pytest.mark.parametrize(("rest_z", "rest_z_metric"), [(None, None), (0.42, "rest")])
def test_gate_requires_exactly_one_rest_z_source(env, rest_z, rest_z_metric):
    """A defaulted reference would not crash: ``rest_z=0`` opens the gate on a cube sitting on the table."""
    with pytest.raises(ValueError, match="exactly one resting-height source"):
        lift_height(env, object_name="cube", rest_z=rest_z, rest_z_metric=rest_z_metric)


def test_gate_rejects_a_cumulative_metric(env):
    """A running episode sum would push the gate out of reach."""
    with pytest.raises(ValueError, match="rest_z_metric='rest_sum' is a cumulative metric"):
        lift_height(env, object_name="cube", rest_z_metric="rest_sum")


def test_gate_rejects_an_unknown_metric_naming_the_active_ones(env):
    with pytest.raises(KeyError, match="Active metrics"):
        lift_height(env, object_name="cube", rest_z_metric="does_not_exist")


def test_gate_requires_a_metric_manager():
    no_metrics = build_env(_config(None))
    with pytest.raises(ValueError, match="rest_z_metric='rest' was requested but no metric manager"):
        lift_height(no_metrics, object_name="cube", rest_z_metric="rest")
