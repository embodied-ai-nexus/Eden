"""Tests for the art suite's workcell: one spec per tool, the tool MJCF generated from it, and the sheet's grid."""

import typing
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.tasks.art import workcell


def test_every_tool_kind_has_a_spec_and_specs_are_frozen():
    assert set(workcell.TOOLS) == set(typing.get_args(workcell.ToolKind))
    assert len({workcell.TOOLS[kind] for kind in workcell.TOOLS}) == len(workcell.TOOLS)
    with pytest.raises(TypeError):
        workcell.TOOLS["brush"].home["joint1"] = 1.0


@pytest.mark.parametrize("kind", typing.get_args(workcell.ToolKind))
def test_a_tool_s_mjcf_comes_from_its_spec_with_a_visual_only_tip(kind):
    spec = workcell.TOOLS[kind]
    body = ET.parse(workcell.tool_file(kind)).getroot().find("worldbody/body")
    assert float(body.find("inertial").get("mass")) == spec.mass
    visual = [g.get("name") for g in body.findall("geom") if g.get("contype") == "0"]
    assert visual, "the hairs, the tuft or the nib are drawn but do not collide"


def test_an_unknown_tool_is_refused_naming_the_tools():
    with pytest.raises(ValueError, match=r"unknown tool 'marker'; the tools are \['brush', 'pen', 'paintbrush'\]"):
        workcell.tool_file("marker")


@pytest.mark.parametrize("kind", ["brush", "paintbrush"])
def test_a_brush_s_hand_points_the_tool_straight_down(kind):
    approach = gu.quat_to_R(np.asarray(workcell.TOOLS[kind].hand_quat))[:, 2]  # the hand's z: the tool's axis
    np.testing.assert_allclose(approach, (0.0, 0.0, -1.0), atol=1e-9)


def test_the_brushes_share_a_home_because_their_tips_are_as_far_ahead_of_the_grip():
    brush, paintbrush = workcell.TOOLS["brush"], workcell.TOOLS["paintbrush"]
    assert brush.home == paintbrush.home
    assert brush.tip_ahead_of_grip == pytest.approx(paintbrush.tip_ahead_of_grip)
    assert workcell.tool_tip_in_attachment("brush") == pytest.approx(workcell.tool_tip_in_attachment("paintbrush"))


def test_the_sheet_grid_is_centred_on_the_sheet():
    grid = workcell.sheet_grid((0.16, 0.06), 0.0002)
    assert (grid.rows, grid.cols) == (300, 800)
    assert (grid.u0, grid.v0) == pytest.approx((-0.08, 0.03))
