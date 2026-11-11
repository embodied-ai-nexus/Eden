"""LIBERO object and workspace options, and the runtime MJCF they resolve to.

``resolve_runtime_file`` runs the ``eden.options.objects.libero.mjcf_transforms`` pipelines over the source
descriptor, so these tests pin that module too. The workspace and bowl tests read the real LIBERO assets
(fetched on first use); the rest are hermetic.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from eden.options.objects.libero import (
    LiberoAkitaBlackBowl,
    LiberoFloorWorkspace,
    LiberoKitchenTabletopWorkspace,
    LiberoObjectOptions,
    LiberoSceneAssetOptions,
    LiberoTabletopWorkspace,
)
from eden.options.objects.mjcf_transforms import runtime_dir
from eden.utils.configs import serialize_obj_with_metadata, to_options

_WRAPPED_CUBE = """
<mujoco>
  <worldbody>
    <body>
      <body name="object">
        <geom name="cube" type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _vector(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(part) for part in element.get(attribute).split())


def _source(tmp_path: Path, name: str = "source.xml", text: str = _WRAPPED_CUBE) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_serialization_keeps_the_source_file_and_a_deleted_runtime_file_is_rebuilt(tmp_path):
    source = _source(tmp_path)
    options = LiberoObjectOptions(file=str(source), local_dir=str(tmp_path))
    runtime_path = Path(options.resolve_runtime_file())
    serialized = serialize_obj_with_metadata(options)

    # Content-addressed, but the basename survives as the suffix: morph dispatch reads the extension.
    assert runtime_path.name.endswith(source.name)
    assert serialized["file"] == str(source)

    runtime_path.unlink()
    restored = to_options(serialized, LiberoObjectOptions)
    assert restored.file == str(source)
    assert Path(restored.resolve_runtime_file()).is_file()


def test_runtime_descriptors_share_one_self_cleaning_directory(tmp_path):
    """A ``mkdtemp`` per asset leaked one ``/tmp`` entry per object, thousands across a 130-task replay."""
    runtime_paths = [
        LiberoObjectOptions(
            file=str(_source(tmp_path, f"source_{index}.xml")), local_dir=str(tmp_path)
        ).resolve_runtime_file()
        for index in range(2)
    ]

    assert len(set(runtime_paths)) == 2, "distinct sources collided on one runtime path"
    assert {Path(path).parent for path in runtime_paths} == {runtime_dir()}


def test_fixed_and_floating_variants_of_one_source_do_not_collide(tmp_path):
    source = str(_source(tmp_path))
    floating = LiberoObjectOptions(file=source, local_dir=str(tmp_path), is_fixed_base=False).resolve_runtime_file()
    fixed = LiberoObjectOptions(file=source, local_dir=str(tmp_path), is_fixed_base=True).resolve_runtime_file()

    assert floating != fixed
    assert ET.parse(floating).getroot().find("./worldbody/body/freejoint[@name='root_joint']") is not None
    assert ET.parse(fixed).getroot().find("./worldbody/body/freejoint") is None


@pytest.mark.parametrize(
    ("options_type", "expected_friction"),
    [(LiberoTabletopWorkspace, (0.6, 0.005, 0.0001)), (LiberoKitchenTabletopWorkspace, (1.0, 0.005, 0.0001))],
)
def test_table_workspace_materializes_the_canonical_libero_table(options_type, expected_friction):
    worldbody = ET.parse(options_type().resolve_runtime_file()).getroot().find("worldbody")
    table = worldbody.find("./body[@name='table']")
    assert _vector(worldbody.find("./geom[@name='floor']"), "pos") == (0.0, 0.0, 0.0)
    assert _vector(table, "pos") == (0.0, 0.0, 0.875)

    collision = table.find("./geom[@name='table_collision']")
    visual = table.find("./geom[@name='table_visual']")
    assert _vector(collision, "size") == _vector(visual, "size") == (0.5, 0.6, 0.025)
    assert _vector(collision, "friction") == expected_friction
    assert collision.get("group") == "3"
    assert (visual.get("group"), visual.get("contype"), visual.get("conaffinity")) == ("1", "0", "0")
    assert visual.get("material") == "table_texture"
    assert _vector(table.find("./site[@name='table_top']"), "pos") == (0.0, 0.0, 0.025)
    legs = [(0.4, 0.5), (-0.4, 0.5), (-0.4, -0.5), (0.4, -0.5)]
    for index, (x, y) in enumerate(legs, start=1):
        leg = table.find(f"./geom[@name='table_leg{index}_visual']")
        assert _vector(leg, "pos") == (x, y, -0.4375)
        assert _vector(leg, "size") == (0.025, 0.4375)


def test_floor_workspace_keeps_the_official_material_and_splits_visual_from_collision():
    root = ET.parse(LiberoFloorWorkspace().resolve_runtime_file()).getroot()
    material = root.find("./asset/material[@name='floorplane']")
    collision = root.find("./worldbody/body[@name='floor']/geom[@name='floor']")
    visual = root.find("./worldbody/body[@name='floor']/geom[@name='floor_vis']")

    assert (material.get("texrepeat"), material.get("texuniform")) == ("3 3", "true")
    assert collision.get("group") == "3"
    assert (visual.get("group"), visual.get("contype"), visual.get("conaffinity")) == ("1", "0", "0")
    assert visual.get("material") == "floorplane"


def test_visual_split_hides_only_a_collision_geom_with_an_exact_visual_counterpart(tmp_path):
    source = _source(
        tmp_path,
        "scene.xml",
        """
        <mujoco>
          <worldbody>
            <body name="table">
              <geom name="table_collision" type="box" pos="0 0 0" size="0.5 0.6 0.025" group="0"/>
              <geom name="table_visual" type="box" pos="0 0 0" size="0.5 0.6 0.025" group="1"
                    contype="0" conaffinity="0" material="table_texture"/>
              <geom name="leg_collision" type="box" pos="0 0 -0.5" size="0.1 0.1 0.5" group="0"/>
              <geom name="leg_visual" type="box" pos="0 0 -0.5" size="0.05 0.05 0.5" group="1"
                    contype="0" conaffinity="0" material="table_texture"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    options = LiberoSceneAssetOptions(file=str(source), local_dir=str(tmp_path))
    table = ET.parse(options.resolve_runtime_file()).getroot().find("./worldbody/body[@name='table']")

    assert table.find("./geom[@name='table_collision']").get("group") == "3"
    assert table.find("./geom[@name='table_visual']").get("group") == "1"
    assert table.find("./geom[@name='leg_collision']").get("group") == "0"  # its visual has a different size
    assert len(table.findall("./geom")) == 4


def test_libero_visual_twin_keeps_the_mass_it_has_always_added(tmp_path):
    """LIBERO passes ``massless_twin=False``: its twin inherits ``density`` and doubles the body's mass.

    Wrong, but what every LIBERO descriptor has compiled to; fixing it would change the shipped
    benchmark's dynamics. This pins the doubling as current behaviour, a guard against changing it by
    accident (sharing the transform with robosuite briefly did).
    """
    mujoco = pytest.importorskip("mujoco")
    source = _source(
        tmp_path,
        "scene.xml",
        """
        <mujoco>
          <asset><material name="wood" rgba="0.6 0.4 0.2 1"/></asset>
          <worldbody>
            <body name="slab">
              <geom name="slab_geom" type="box" size="0.2 0.2 0.02" group="1" density="500" material="wood"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    )
    before = float(mujoco.MjModel.from_xml_path(str(source)).body_mass[1:].sum())
    runtime_path = LiberoSceneAssetOptions(file=str(source), local_dir=str(tmp_path)).resolve_runtime_file()

    twin = ET.parse(runtime_path).getroot().find("./worldbody/body[@name='slab']/geom[@name='slab_geom_vis']")
    assert twin is not None, "the material geom got no visual twin, so this test proves nothing"
    after = float(mujoco.MjModel.from_xml_path(runtime_path).body_mass[1:].sum())
    assert after == pytest.approx(2.0 * before, abs=1e-6), f"{before} kg became {after} kg"


def test_akita_black_bowl_promotes_the_source_root_and_adds_a_free_joint():
    worldbody = ET.parse(LiberoAkitaBlackBowl().resolve_runtime_file()).getroot().find("worldbody")
    root = worldbody.find("./body")

    assert (root.get("name"), root.get("pos")) == ("main", "0 -0.1 0")
    assert root.find("./freejoint[@name='root_joint']") is not None
    assert worldbody.find(".//body[@name='object']") is None
    assert worldbody.find(".//site[@name='bottom_site']") is None
