"""Tests for the copy metric base's contract with its subclasses."""

import pytest

from eden.tasks.art.terms import CopyMetric


@pytest.mark.parametrize("rule", [None, "ioU", "tool_name"])
def test_a_copy_metric_that_names_no_float_grade_rule_is_refused_when_defined(rule):
    """``None``: not set; ``"ioU"``: no such field; ``"tool_name"``: a field, but no threshold."""
    namespace = {} if rule is None else {"_grade_rule": rule}
    with pytest.raises(
        TypeError, match=rf"Unruled is a CopyMetric, so it must set _grade_rule .* got _grade_rule={rule!r}\."
    ):
        type("Unruled", (CopyMetric,), namespace)


def test_a_copy_metric_whose_grade_rule_defaults_to_an_integer_is_accepted():
    ruled = type("Ruled", (CopyMetric,), {"_grade_rule": "coverage", "coverage": 1})
    assert ruled._grade_rule == "coverage"
