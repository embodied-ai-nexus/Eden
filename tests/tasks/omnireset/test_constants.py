"""``omnireset.constants``: every pair names props that exist, and its pinned values match the props' MJCF."""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET

import pytest

from eden.tasks.omnireset.constants import INSERTIVE_MASS, PAIRS, prop_file
from eden.utils.assets import get_asset_path


def test_an_unknown_prop_is_refused_where_the_pair_is_defined():
    with pytest.raises(
        ValueError, match=r"No OmniReset prop 'pegg'; props: \['peg', 'peg_hole', .*add 'pegg' to PROPS"
    ):
        dataclasses.replace(PAIRS["peg"], insertive="pegg")
    with pytest.raises(ValueError, match=r"No OmniReset prop 'table'"):
        prop_file("table")


@pytest.mark.parametrize("pair_name", list(PAIRS))
def test_the_insertive_props_inertial_is_the_pinned_mass_and_centre_of_mass(pair_name):
    # The prop's <inertial> was written from its colliders, as PhysX derives the centre of mass that
    # ``Pair.insertive_com`` pins for converting the reset states; the two must not drift apart.
    pair = PAIRS[pair_name]
    body = ET.parse(get_asset_path(prop_file(pair.insertive))).getroot().find("./worldbody/body")
    inertial = body.find("inertial")
    assert inertial is not None, f"{pair.insertive}'s MJCF has no <inertial>: Genesis would weigh its meshes instead"
    assert float(inertial.get("mass")) == pytest.approx(INSERTIVE_MASS)
    assert [float(v) for v in inertial.get("pos").split()] == pytest.approx(pair.insertive_com, abs=1e-6)
