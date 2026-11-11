"""``eden.options.scenes.empty``: the preset's ``plane`` slot is whatever the robot stands on."""

from __future__ import annotations

import pytest

from eden.options import BoxOptions, PlaneOptions, TerrainOptions
from eden.options.scenes.empty import BimanualEmptySceneOptions, EmptySceneOptions


def test_empty_scene_defaults_to_a_flat_plane():
    assert isinstance(EmptySceneOptions(robot=BoxOptions()).plane, PlaneOptions)


@pytest.mark.parametrize(
    "scene",
    [
        lambda: EmptySceneOptions(robot=BoxOptions(), plane=TerrainOptions()),
        lambda: BimanualEmptySceneOptions(right_robot=BoxOptions(), left_robot=BoxOptions(), plane=TerrainOptions()),
    ],
    ids=["single", "bimanual"],
)
def test_passing_a_terrain_replaces_the_plane_under_the_same_name(scene):
    """Everything addressing the ground by ``plane`` keeps working; no second ground entity appears."""
    options = scene()
    assert isinstance(options.plane, TerrainOptions)
    assert not any(isinstance(getattr(options, k), PlaneOptions) for k in options.keys() if k != "attachments_dict")


def test_empty_scene_rejects_plane_options_it_would_have_to_drop():
    """``is_support_enabled`` / ``surface`` configure the default plane only, so with a replacement they contradict."""
    with pytest.raises(ValueError, match="is_support_enabled / surface"):
        EmptySceneOptions(robot=BoxOptions(), plane=TerrainOptions(), is_support_enabled=True)
