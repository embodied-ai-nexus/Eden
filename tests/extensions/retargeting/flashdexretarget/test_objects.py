"""Tests for ``eden.extensions.retargeting.flashdexretarget.objects``: the object URDFs the task simulates."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from eden.extensions.retargeting.flashdexretarget.convex import decompose_objects
from eden.extensions.retargeting.flashdexretarget.objects import write_object_urdf
from eden.options.extensions.flashdexretarget import FDRCoacdConvexOptions
from tests.extensions.retargeting.flashdexretarget.conftest import write_box


@pytest.fixture
def box(tmp_path):
    write_box(tmp_path / "objects" / "box" / "visual.obj", extents=(0.1, 0.1, 0.1))
    decompose_objects(tmp_path, ["box"], FDRCoacdConvexOptions(), workers=1)
    return tmp_path / "objects" / "box"


def test_a_heavy_object_is_clamped_to_half_a_kilo_with_its_inertia_scaled_to_match(box):
    """A 10 cm cube at 800 kg/m^3 is 0.8 kg: clamped to 0.5 kg, inertia m a^2 / 6 of the clamped mass."""
    link = ET.parse(write_object_urdf(box)).getroot().find("link")
    assert float(link.find("inertial/mass").get("value")) == pytest.approx(0.5)
    assert float(link.find("inertial/inertia").get("ixx")) == pytest.approx(0.5 * 0.1**2 / 6, rel=1e-6)
    collisions = [c.find("geometry/mesh").get("filename") for c in link.findall("collision")]
    assert collisions == ["convex/0.obj"]
    assert link.find("visual/geometry/mesh").get("filename") == "visual.obj"


def test_a_scaled_object_gets_its_own_urdf_with_the_scale_on_every_mesh(box):
    path = write_object_urdf(box, scale=0.5)
    assert path.name == "object_s0.5.urdf"
    link = ET.parse(path).getroot().find("link")
    assert {m.get("scale") for m in link.iter("mesh")} == {"0.5 0.5 0.5"}
    # 5 cm cube: 0.1 kg, under the clamp.
    assert float(link.find("inertial/mass").get("value")) == pytest.approx(0.1)
