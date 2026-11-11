"""The cached YAM asset the Mochi tasks share: one copy per source file, reading that source's meshes."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from eden.tasks.mochi.common import without_collision_spheres

_MJCF = """<mujoco>
  <compiler angle="radian" meshdir="assets"/>
  <worldbody>
    <body name="finger">
      <geom name="pad" type="box" size="0.01 0.01 0.01"/>
      <geom class="sphere_collision" type="sphere" size="0.0006"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_asset(directory) -> str:
    directory.mkdir(parents=True)
    source = directory / "yam.xml"
    source.write_text(_MJCF)
    return str(source)


def test_the_copy_drops_the_collision_spheres_and_reads_meshes_from_its_source(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = _write_asset(tmp_path / "checkout_a")

    root = ET.parse(without_collision_spheres(source)).getroot()
    assert [geom.get("name") for geom in root.iter("geom")] == ["pad"]
    assert root.find("compiler").get("meshdir") == os.path.join(str(tmp_path / "checkout_a"), "assets")


def test_each_source_file_gets_its_own_copy_which_follows_it(tmp_path, monkeypatch):
    """A copy names its meshes by absolute path, so another checkout must not share it, nor an update be missed."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    first, second = _write_asset(tmp_path / "checkout_a"), _write_asset(tmp_path / "checkout_b")

    copy = without_collision_spheres(first)
    assert without_collision_spheres(first) == copy  # one path per source
    other = without_collision_spheres(second)
    assert other != copy
    assert ET.parse(other).getroot().find("compiler").get("meshdir").startswith(str(tmp_path / "checkout_b"))

    # The asset is updated in place: the next call rewrites the same copy from it.
    with open(first, "w") as f:
        f.write(
            _MJCF.replace('<geom name="pad"', '<geom name="new_pad" type="box" size="0.01 0.01 0.01"/><geom name="pad"')
        )
    assert without_collision_spheres(first) == copy
    assert [geom.get("name") for geom in ET.parse(copy).getroot().iter("geom")] == ["new_pad", "pad"]
