"""The official LIBERO predicate geometry behind ``LiberoGoalMetric``, pinned at its boundaries.

The metric term itself runs against a real scene in ``tests/tasks/libero/test_task.py``.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks.libero.custom_terms.metrics import (
    _articulation_predicate,
    _object_on_object,
    _site_contains_object,
    _site_under_object,
)


def _identity(count):
    return torch.eye(3).expand(count, 3, 3)


def test_site_contains_object_is_strict_in_xy_and_extends_1cm_below():
    objects = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.005]])

    result = _site_contains_object(objects, torch.zeros(3, 3), _identity(3), torch.ones(3, 3))

    assert result.tolist() == [True, False, True]


def test_site_under_object_accepts_the_official_vertical_window():
    objects = torch.tensor([[0.0, 0.0, 0.001], [0.0, 0.0, 0.003], [0.0, 0.0, 0.108]])
    site_size = torch.tensor([[1.0, 1.0, 0.007]]).expand(3, 3)

    result = _site_under_object(objects, torch.zeros(3, 3), _identity(3), site_size)

    assert result.tolist() == [False, True, False]


def test_object_on_object_needs_contact_support_from_below_and_3cm_xy():
    objects = torch.tensor([[0.0, 0.0, 0.1], [0.03, 0.0, 0.1], [0.0, 0.0, -0.1], [0.0, 0.0, 0.1]])
    contact = torch.tensor([True, True, True, False])

    assert _object_on_object(objects, torch.zeros(4, 3), contact).tolist() == [True, False, False, False]


@pytest.mark.parametrize(
    ("comparison", "threshold", "reduction", "expected"),
    [
        ("lt", -0.14, "any", [True, False, False, False]),
        ("gt", 0.0, "all", [False, False, False, True]),
        ("ge", 0.0, "any", [False, True, True, True]),
    ],
)
def test_articulation_predicate_thresholds_and_reductions(comparison, threshold, reduction, expected):
    qpos = torch.tensor([[-0.15, -0.13], [-0.14, 0.01], [0.0, 0.0], [0.01, 0.02]])

    result = _articulation_predicate(qpos, comparison=comparison, threshold=threshold, reduction=reduction)

    assert result.tolist() == expected
