"""The runtime MJCF transforms that make robosuite-style assets load correctly in Genesis.

Genesis' MJCF loader differs from MuJoCo / robosuite in three ways that made every robosuite object render
as a grey collision proxy: a geom with collision bits is never visual, ``cube`` textures get no UV mapping,
and the free joint robosuite adds at load time is absent from the files. Uses the ``robosuite_xml`` and
``robosuite_object_file`` fixtures from this directory's ``conftest.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import pytest

from eden.options.objects.mjcf_transforms import (
    add_freejoint_to_root_body,
    clone_collision_geoms_for_rendering,
    declare_radians,
    extract_wrapped_object_body,
    flatten_cube_textures,
    materialize_xml,
    remove_freejoint_from_root_body,
    transform_identity,
)

_BARE_ROOT = "<mujoco><worldbody><body name='pot'><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"


def test_angles_are_declared_in_radians_as_robosuites_world_does():
    """The object files declare no compiler and are written in radians; MuJoCo's default unit is degrees."""
    bare = ET.fromstring("<mujoco><worldbody/></mujoco>")
    assert declare_radians(bare, Path("."))
    assert [c.get("angle") for c in bare.findall("compiler")] == ["radian"]

    with_meshdir = ET.fromstring("<mujoco><compiler meshdir='meshes'/><worldbody/></mujoco>")
    assert declare_radians(with_meshdir, Path("."))
    assert [(c.get("angle"), c.get("meshdir")) for c in with_meshdir.findall("compiler")] == [("radian", "meshes")]
    assert declare_radians(with_meshdir, Path(".")) is False, "a second pass changes nothing"


def test_the_wrapper_is_unwrapped_into_one_root_body(robosuite_xml):
    root = ET.fromstring(robosuite_xml)
    assert extract_wrapped_object_body(root, Path("."), strict=False)
    assert [b.get("name") for b in root.find("worldbody").findall("body")] == ["main"]
    assert root.find("worldbody").find("site") is None, "the wrapper's sites went with it"


def test_unwrapping_leaves_a_hand_authored_root_and_rejects_an_empty_world():
    assert extract_wrapped_object_body(ET.fromstring(_BARE_ROOT), Path("."), strict=False) is False
    with pytest.raises(ValueError):
        extract_wrapped_object_body(ET.fromstring("<mujoco><worldbody/></mujoco>"), Path("."), strict=True)


@pytest.mark.parametrize("attribute,value", [("pos", "0 0 0.1"), ("quat", "0.707 0 0 0.707"), ("euler", "0 0 1.57")])
def test_a_posed_wrapper_body_is_refused_rather_than_silently_dropped(robosuite_xml, attribute, value):
    """Unwrapping deletes the wrapper, so a pose on it would move the object with nothing raised."""
    root = ET.fromstring(robosuite_xml.replace("<body>", f'<body {attribute}="{value}">', 1))
    with pytest.raises(ValueError, match="carries a pose"):
        extract_wrapped_object_body(root, Path("."), strict=False)


def test_every_collision_geom_gets_a_uniquely_named_massless_visual_twin(robosuite_xml):
    root = ET.fromstring(robosuite_xml)
    assert clone_collision_geoms_for_rendering(root, Path("."), groups=None, require_material=False)
    geoms = list(root.iter("geom"))
    visual = [g for g in geoms if g.get("contype") == "0" and g.get("conaffinity") == "0"]
    collision = [g for g in geoms if g not in visual]
    names = [g.get("name") for g in visual]
    assert len(visual) == len(collision) == len(set(names)) == 3, f"MuJoCo rejects duplicate geom names: {names}"
    assert {"ring_vis", "pad_vis"} <= set(names)
    # The twin carries the look; the original keeps the physics and is hidden (MuJoCo group 3).
    ring_vis = next(g for g in visual if g.get("name") == "ring_vis")
    assert ring_vis.get("material") == "bmetal"
    # ...and none of the mass: contype/conaffinity exclude a geom from collision, not from MuJoCo's
    # inertiafromgeom sum, so a twin inheriting `density` doubles the body's mass.
    assert ring_vis.get("density") is None and ring_vis.get("mass") == "1e-8"
    assert all(g.get("group") == "3" for g in collision)


def test_the_libero_default_clones_only_material_geoms_in_groups_1_and_2(robosuite_xml):
    assert clone_collision_geoms_for_rendering(ET.fromstring(robosuite_xml), Path(".")) is False  # all group 0


def test_a_cube_texture_becomes_the_materials_mean_colour(robosuite_xml, robosuite_object_file):
    root = ET.fromstring(robosuite_xml)
    assert flatten_cube_textures(root, robosuite_object_file.parent)
    assert root.find("asset").find("texture") is None
    material = root.find("asset").find("material")
    assert material.get("texture") is None
    assert [float(v) for v in material.get("rgba").split()] == pytest.approx([0.6, 0.4, 0.2, 1.0], abs=1e-3)


def test_a_2d_texture_is_left_for_genesis_to_map(robosuite_xml, robosuite_object_file):
    root = ET.fromstring(robosuite_xml.replace('type="cube"', 'type="2d"'))
    assert flatten_cube_textures(root, robosuite_object_file.parent) is False
    assert root.find("asset").find("material").get("texture") == "brass"


def test_the_free_joint_is_added_and_removed_once():
    root = ET.fromstring(_BARE_ROOT)
    assert add_freejoint_to_root_body(root, Path("."))
    assert add_freejoint_to_root_body(root, Path(".")) is False, "already free"
    assert remove_freejoint_from_root_body(root, Path("."))
    assert root.find("worldbody/body/freejoint") is None
    assert remove_freejoint_from_root_body(root, Path(".")) is False, "already fixed"


def test_the_runtime_cache_is_keyed_by_the_pipeline_and_its_bound_keywords(robosuite_object_file):
    """LIBERO and robosuite run the same transform over the same kind of file, differing in one keyword.

    A key blind to it would hand the second caller the first caller's descriptor: an object at twice its
    intended mass in a scene that builds and steps without complaint.
    """
    source = str(robosuite_object_file)
    common = dict(registry=None, dataset=None, local_dir=None)

    def materialize(**bound):
        clone = partial(clone_collision_geoms_for_rendering, groups=None, require_material=False, **bound)
        return materialize_xml(source, transforms=(clone,), **common)

    massless, massive = materialize(), materialize(massless_twin=False)
    assert massless != massive
    twin_mass = {
        path: [geom.get("mass") for geom in ET.parse(path).getroot().iter("geom") if geom.get("contype") == "0"]
        for path in (massless, massive)
    }
    assert all(mass == "1e-8" for mass in twin_mass[massless])
    assert all(mass is None for mass in twin_mass[massive])
    assert materialize() == massless, "the same pipeline twice is still one cached file"


def test_transform_identity_sees_through_partials_and_ignores_binding_order():
    plain = transform_identity(add_freejoint_to_root_body)
    assert plain == ("eden.options.objects.mjcf_transforms.add_freejoint_to_root_body",)
    assert transform_identity(remove_freejoint_from_root_body) != plain

    bound = partial(clone_collision_geoms_for_rendering, groups=(1, 2), massless_twin=False)
    reordered = partial(clone_collision_geoms_for_rendering, massless_twin=False, groups=(1, 2))
    assert transform_identity(bound) == transform_identity(reordered), "keyword order is not a difference"
    assert transform_identity(bound) != transform_identity(
        partial(clone_collision_geoms_for_rendering, groups=(1, 2), massless_twin=True)
    ), "a keyword that changes the output must change the identity"
    assert transform_identity(bound) != transform_identity(clone_collision_geoms_for_rendering)
