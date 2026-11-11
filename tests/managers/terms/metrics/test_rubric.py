"""``RubricMetricTerm`` in ``eden/managers/terms/metrics/sequence.py``: criteria reached once, in any order their dependencies allow.

A criterion whose predicate must change mid-test is "cube X is lifted", toggled with the ``lift`` fixture.
The term's value is the fraction of criteria each env has reached.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.metric_manager import MetricManager
from eden.managers.terms.metrics.sequence import RubricMetricTerm
from eden.options.managers.metrics import CriterionOptions, MetricManagerOptions, MetricTermOptions

pytestmark = pytest.mark.requires_genesis


def _lifted(cube):
    return MetricTermOptions(name="object_above_height", params={"entity_name": cube, "height": 0.5})


def _criterion(name, cube, after=()):
    return CriterionOptions(name=name, term=_lifted(cube), after=list(after))


@pytest.fixture
def rubric(panda_env, make_term, lift):
    """``rubric(*criteria)``: a built term, with both cubes down (the scene hangs them at 1 m)."""

    def build(*criteria):
        lift("cube_a", False)
        lift("cube_b", False)
        return make_term(RubricMetricTerm, panda_env, criteria=list(criteria))

    return build


def _progress(term, *expected):
    """Compute once and compare every env's reached fraction (one value, or one per env)."""
    out = term.compute()
    torch.testing.assert_close(out, torch.tensor(expected if len(expected) > 1 else expected * len(out)))


def test_independent_criteria_are_reached_in_any_order(rubric, lift):
    term = rubric(_criterion("a", "cube_a"), _criterion("b", "cube_b"))
    lift("cube_b", True)
    _progress(term, 0.5)
    lift("cube_b", False)
    lift("cube_a", True)
    _progress(term, 1.0)


def test_a_reached_criterion_stays_reached(rubric, lift):
    term = rubric(_criterion("a", "cube_a"))
    lift("cube_a", True)
    _progress(term, 1.0)
    lift("cube_a", False)
    _progress(term, 1.0)


def test_a_dependent_criterion_counts_only_after_its_dependencies(rubric, lift):
    term = rubric(_criterion("a", "cube_a"), _criterion("b", "cube_b", after=["a"]))
    lift("cube_b", True)
    _progress(term, 0.0)  # b passes, but a has not been reached
    lift("cube_a", True)
    _progress(term, 1.0)  # a and then b in the same step: b is evaluated after a


def test_each_env_reaches_its_own_criteria(rubric, lift):
    term = rubric(_criterion("a", "cube_a"), _criterion("b", "cube_b", after=["a"]))
    lift("cube_a", [True, False])
    lift("cube_b", True)
    _progress(term, 1.0, 0.0)
    assert term.reached().tolist() == [[True, True], [False, False]]


def test_reset_clears_the_selected_envs(rubric, lift):
    term = rubric(_criterion("a", "cube_a"))
    lift("cube_a", True)
    _progress(term, 1.0)
    lift("cube_a", False)
    term.reset(envs_idx=torch.tensor([0]))
    _progress(term, 0.0, 1.0)


def test_a_reset_of_every_env_or_of_a_mask_clears_those_envs(rubric, lift):
    """``reset()`` with no index (the manager's full reset) and with a bool mask (the auto-reset's ``reset_buf``)."""
    term = rubric(_criterion("a", "cube_a"))
    lift("cube_a", True)
    _progress(term, 1.0)
    lift("cube_a", False)
    term.reset(envs_idx=torch.tensor([False, True]))
    _progress(term, 1.0, 0.0)
    term.reset()
    _progress(term, 0.0)


def test_a_criterion_is_met_as_its_term_judges_success(rubric, lift):
    """A lower-is-better term is met at or below its ``success_threshold``, not above it."""
    lowered = MetricTermOptions(
        name="object_above_height",
        params={"entity_name": "cube_a", "height": 0.5},
        direction="lib",
        success_threshold=0.0,
    )
    term = rubric(CriterionOptions(name="down", term=lowered))
    lift("cube_a", [True, False])
    _progress(term, 0.0, 1.0)


def test_success_through_the_metric_manager_is_every_criterion_reached(panda_env, install_manager, lift):
    options = MetricManagerOptions(
        task=RubricMetricTerm.configure(criteria=[_criterion("a", "cube_a"), _criterion("b", "cube_b")])
    )
    manager = install_manager(panda_env, "metric_manager", MetricManager(panda_env, options))
    lift("cube_b", False)
    lift("cube_a", True)
    manager.compute()
    assert manager.success_buf.tolist() == [False, False]
    lift("cube_b", [True, False])
    manager.compute()
    assert manager.success_buf.tolist() == [True, False]


@pytest.mark.parametrize(
    ("criteria", "match"),
    [
        ([], "needs at least one criterion"),
        ([_criterion("b", "cube_b", after=["a"]), _criterion("a", "cube_a")], r"after=\['a'\].*earlier ones: none"),
        ([_criterion("a", "cube_a"), _criterion("a", "cube_b")], "'a' is used twice"),
        (
            [CriterionOptions(name="a", term=_lifted("cube_a").model_copy(update={"is_cumulative": True}))],
            "'a' has is_cumulative=True",
        ),
    ],
)
def test_malformed_criteria_are_refused_at_build(rubric, criteria, match):
    with pytest.raises(ValueError, match=match):
        rubric(*criteria)
