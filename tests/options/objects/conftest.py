"""A hermetic robosuite-shaped MJCF for the object-option tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: A robosuite-shaped object: wrapper body, one named and one unnamed dual-purpose geom, a cube texture.
ROBOSUITE_XML = """
<mujoco model="nut">
  <asset>
    <texture file="brass.png" type="cube" name="brass"/>
    <material name="bmetal" texture="brass" texrepeat="1 1" specular="1"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="ring" size="0.01 0.04 0.01" type="box" density="100" material="bmetal"/>
        <geom size="0.02 0.01 0.01" type="box" density="100" material="bmetal"/>
        <geom name="pad" size="0.01" type="sphere" rgba="1 0 0 1"/>
      </body>
      <site name="bottom_site" pos="0 0 -0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def robosuite_xml() -> str:
    """The MJCF text of :data:`ROBOSUITE_XML`."""
    return ROBOSUITE_XML


@pytest.fixture
def robosuite_object_file(tmp_path: Path) -> Path:
    """:data:`ROBOSUITE_XML` written to ``tmp_path/objects/nut.xml`` beside a flat (0.6, 0.4, 0.2) texture."""
    (tmp_path / "objects").mkdir()
    path = tmp_path / "objects" / "nut.xml"
    path.write_text(ROBOSUITE_XML, encoding="utf-8")
    Image.fromarray(np.full((8, 8, 3), (153, 102, 51), dtype=np.uint8)).save(tmp_path / "objects" / "brass.png")
    return path
