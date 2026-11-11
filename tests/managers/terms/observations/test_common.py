"""The shared observation terms in ``eden/managers/terms/observations/common.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.observations.common import MetricObservation

pytestmark = pytest.mark.requires_genesis


def test_metric_observation_is_the_metric_value_with_a_feature_axis(panda_env, make_term, score_metric):
    drive = score_metric(panda_env)
    term = make_term(MetricObservation, panda_env, metric_name="score")
    drive([3.0, 0.5])
    torch.testing.assert_close(term.compute(), torch.tensor([[3.0], [0.5]]))
    # A read, not a tick: the extra calls an observation term receives in a step return the same value.
    torch.testing.assert_close(term.compute(), torch.tensor([[3.0], [0.5]]))
