"""Tests for ``eden.tasks.beyondmimic.custom_terms.metrics``: every tracking metric is configured the way it is
registered, the metrics share their anchor-pose reads through ``MetricManager.compute_once``, and sharing
must not move a logged value.
"""

from __future__ import annotations

import pytest
import torch

from eden.constants import MetricDirection, MetricMode
from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env

NUM_ENVS = 4


def test_every_tracking_metric_is_a_lower_is_better_episode_mean_with_its_own_threshold():
    """Six of the ten were function terms, and ran as higher-is-better per-step terms at threshold 1.0."""
    metric_options = TASK_REGISTRY.build("beyondmimic").metric_options
    terms = {name: getattr(metric_options, name) for name in metric_options.term_keys()}

    assert {name: (o.direction, o.metric_mode, o.is_cumulative) for name, o in terms.items()} == dict.fromkeys(
        terms, (MetricDirection.LIB, MetricMode.RESET, True)
    )
    assert {name: o.success_threshold for name, o in terms.items()} == {
        "error_anchor_pos": 0.2,
        "error_anchor_rot": 0.2,
        "error_anchor_lin_vel": 0.6,
        "error_anchor_ang_vel": 3.0,
        "error_body_pos": 0.1,
        "error_body_rot": 0.4,
        "error_body_lin_vel": 0.7,
        "error_body_ang_vel": 4.0,
        "error_dofs_pos": 1.8,
        "error_dofs_vel": 15.0,
    }


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_a_shared_metric_pass_logs_what_each_term_computes_on_its_own():
    """Each term evaluated alone, outside a pass (where nothing is shared), is the independent reference."""
    cfg = TASK_REGISTRY.build("beyondmimic")
    cfg.env_options.num_envs = NUM_ENVS
    env = build_env(cfg, reset=True)
    generator = torch.Generator(device=env.device).manual_seed(0)
    with torch.inference_mode():
        for _ in range(3):
            env.step(torch.rand(NUM_ENVS, env.action_manager.total_action_dim, generator=generator, device=env.device))

    manager = env.metric_manager
    with torch.inference_mode():
        manager.reset()  # so each buffer holds exactly this pass's value, not an episode sum
        shared = {name: value.clone() for name, value in manager.compute().items()}
        for name in manager.active_terms:
            alone = manager.get_term(name).compute().float()
            assert torch.equal(shared[name], alone), (
                f"metric {name!r}: pass {shared[name].tolist()} != alone {alone.tolist()}"
            )
