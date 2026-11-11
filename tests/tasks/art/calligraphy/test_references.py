"""Tests for what a work is: its plan's tool, and the target the tool draws along it."""

import pytest

from eden.tasks.art.calligraphy.references import Plan, render_plan
from eden.tasks.art.calligraphy.works import WORKS
from eden.tasks.art.plans import Stroke


def test_a_plan_for_a_tool_that_does_not_write_is_refused_naming_the_writing_tools():
    stroke = Stroke("dash", ((0.0, 0.0, -0.003), (0.01, 0.0, 0.002), (0.02, 0.0, -0.003)))
    with pytest.raises(ValueError, match=r"unknown tool 'paintbrush'; the writing tools are \['brush', 'pen'\]"):
        Plan("paintbrush", (stroke,))


def test_a_rendering_is_deterministic():
    a, b = render_plan(WORKS["dai"]), render_plan(WORKS["dai"])
    assert (a.target == b.target).all() and a.path_length == b.path_length
