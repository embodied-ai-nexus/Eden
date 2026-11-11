"""The RoboLab asset registry (no asset download)."""

from __future__ import annotations

from collections import Counter

from eden.options.objects.robolab import ROBOLAB_REGISTRY


def test_registry_holds_every_asset_set():
    assert Counter(key.split("/")[0] for key in ROBOLAB_REGISTRY.keys()) == {
        "basic": 4,
        "fruits_veggies": 9,
        "handal": 16,
        "hope": 27,
        "hot3d": 25,
        "objaverse": 10,
        "vomp": 196,
        "ycb": 22,
        "fixtures": 12,
    }


def test_every_entry_is_a_urdf_and_only_fixtures_are_fixed_base():
    for key in ROBOLAB_REGISTRY.keys():
        options = ROBOLAB_REGISTRY.get(key)()
        assert options.file.endswith(".urdf"), key
        assert options.is_fixed_base == key.startswith("fixtures/"), key


def test_an_entry_resolves_to_its_file_dataset_and_category():
    banana = ROBOLAB_REGISTRY.get("ycb/banana")()
    assert (banana.file, banana.dataset) == ("objects/ycb/banana/banana.urdf", "eden_robolab")
    assert type(banana).metadata.category == "fruit"
    assert ROBOLAB_REGISTRY.get("fixtures/kitchen_island")().file == "fixtures/kitchen_island/kitchen_island.urdf"
