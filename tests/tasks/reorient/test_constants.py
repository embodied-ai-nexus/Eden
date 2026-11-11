"""The reorient task's side-dependent constants and the cube geometry they mirror from the asset."""

import xml.etree.ElementTree as ET

import pytest

from eden.options.objects.mujoco_playground import DexCube
from eden.tasks.reorient.constants import (
    REORIENT_CUBE_EDGE,
    REORIENT_CUBE_MASS,
    cube_in_palm_for_side,
    dofs_pos_for_side,
    goal_in_palm_for_side,
)
from eden.utils.assets import get_asset_path


def test_left_keyframe_negates_exactly_the_mirrored_abduction_joints():
    """The left URDF inverts only fingers 2-5's abduction; negating more or fewer still runs, and drops the cube."""
    right, left = dofs_pos_for_side("right"), dofs_pos_for_side("left")
    assert set(left) == set(right)
    negated = {name for name in right if right[name] != 0.0 and left[name] == pytest.approx(-right[name])}
    expected = {f".*_finger{finger}_joint2" for finger in range(2, 6)}
    assert negated == expected
    for name in set(right) - expected:
        assert left[name] == pytest.approx(right[name])


@pytest.mark.parametrize("helper", [dofs_pos_for_side, cube_in_palm_for_side, goal_in_palm_for_side])
def test_side_helpers_reject_a_bad_side(helper):
    with pytest.raises(ValueError, match="side must be 'right' or 'left', got 'up'"):
        helper("up")


def test_the_cube_asset_carries_the_geometry_the_constants_declare():
    """The terms compute against the constants while the simulator reads the MJCF; either alone is silently wrong.

    The visual mesh must stay contact- and mass-free, and the box collider hidden, or Genesis draws a grey twin
    through the textured mesh.
    """
    cube = DexCube()
    assert cube.recompute_inertia is False  # it would discard the authored mass
    root = ET.parse(get_asset_path(cube.file)).getroot()

    box = root.find(".//geom[@type='box']")
    assert [float(v) for v in box.get("size").split()] == pytest.approx([REORIENT_CUBE_EDGE / 2.0] * 3)
    assert float(box.get("mass")) == pytest.approx(REORIENT_CUBE_MASS)
    assert box.get("group") == "3"

    visual = root.find(".//geom[@type='mesh']")
    assert (visual.get("contype"), visual.get("conaffinity")) == ("0", "0")
    assert float(visual.get("density")) == 0.0
