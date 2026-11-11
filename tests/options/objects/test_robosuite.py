"""The robosuite object options and the runtime MJCF they resolve to."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from eden.options.objects.robosuite import (
    ROBOSUITE_OBJECT_REGISTRY,
    RobosuiteDoor,
    RobosuiteObjectOptions,
    RobosuiteSquareNut,
    prepare_robosuite_object_xml,
)


def test_prepare_pipeline_writes_free_and_fixed_variants(robosuite_object_file):
    # An absolute source path is used as-is by ``get_asset_path`` (no registry lookup, no download).
    source = str(robosuite_object_file)
    free = prepare_robosuite_object_xml(source, is_fixed_base=False, registry=None, dataset=None, local_dir=None)
    fixed = prepare_robosuite_object_xml(source, is_fixed_base=True, registry=None, dataset=None, local_dir=None)
    assert free != fixed and Path(free).suffix == ".xml"
    free_root = ET.parse(free).getroot()
    assert free_root.find("worldbody/body").get("name") == "main"
    assert free_root.find("worldbody/body/freejoint") is not None
    assert ET.parse(fixed).getroot().find("worldbody/body/freejoint") is None
    # The cube texture was flattened into the material.
    assert free_root.find("asset/texture") is None


def test_options_keep_the_source_file_and_derive_the_runtime_one(robosuite_object_file):
    options = RobosuiteObjectOptions(file=str(robosuite_object_file))
    runtime = options.resolve_runtime_file()
    assert options.file == str(robosuite_object_file), "the serialized identity must not change"
    assert runtime != options.file and Path(runtime).is_file()


def test_registry_covers_every_object_the_tasks_use():
    names = set(ROBOSUITE_OBJECT_REGISTRY.keys())
    assert {"milk", "bread", "cereal", "can", "square_nut", "round_nut", "door", "door_no_latch"} <= names
    assert {"pot_with_handles", "hammer", "hammer_payload", "lid", "transport_bin"} <= names
    assert {"pick_place_bin1", "pick_place_bin2", "tool_hang_stand", "tool_hang_frame", "tool_hang_wrench"} <= names
    for name in names:
        options = ROBOSUITE_OBJECT_REGISTRY.get(name)()
        assert options.file.startswith("objects/") and options.dataset == "eden_robosuite", name


def test_fixtures_and_articulated_objects_declare_themselves():
    assert RobosuiteSquareNut().is_fixed_base is False
    door = RobosuiteDoor()
    assert door.is_fixed_base and door.is_articulated
    assert door.dofs_name == ["hinge", "latch_joint"]


@pytest.mark.parametrize("name", ("square_nut", "round_nut", "milk", "cereal", "pot_with_handles", "hammer"))
def test_the_render_twin_adds_no_mass(name):
    """None of the robosuite assets declares an ``<inertial>``, so a twin with mass would change the object's weight."""
    mujoco = pytest.importorskip("mujoco")

    options = ROBOSUITE_OBJECT_REGISTRY.get(name)()
    source = options.fetch_asset()
    if source is None:
        pytest.skip(f"robosuite asset '{options.file}' is not available locally")
    before = float(mujoco.MjModel.from_xml_path(str(source)).body_mass[1:].sum())
    after = float(mujoco.MjModel.from_xml_path(str(options.resolve_runtime_file())).body_mass[1:].sum())
    # Each twin carries mass="1e-8" (MuJoCo rejects a massless geom on a body it derives inertia for).
    assert after == pytest.approx(before, abs=1e-6), f"{name}: {before} kg became {after} kg"
