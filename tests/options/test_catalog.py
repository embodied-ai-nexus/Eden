"""The asset manifest (curation, queries, lazy ``make``), the robot registry, and name lookup."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from eden.options import robots
from eden.options.catalog import MANIFEST, ROBOT_REGISTRY, AssetManifest, RobotRegistry, find_entity_options
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions, RobotOptions

SEATING = {"sofa", "chair", "armchair", "beanbag", "side_chair", "stool"}

_opts_counter = 0


@pytest.fixture(scope="module")
def manifest():
    """The global manifest, populated from the asset families (imports classes, downloads nothing)."""
    return MANIFEST.build()


def _opts(filename: str = "x.glb") -> type[EntityOptions]:
    """A throwaway ``EntityOptions`` subclass, published under a process-unique module-level name.

    ``make()`` and curation identity resolve a class by its ``module:Class`` path, so the class must
    be importable from here and two calls must never alias one global.
    """
    global _opts_counter
    _opts_counter += 1
    name = f"Opts_{_opts_counter}"

    class Opts(EntityOptions):
        file: str = filename

    Opts.__name__ = Opts.__qualname__ = name
    globals()[name] = Opts
    return Opts


# ── the global manifest ───────────────────────────────────────────────────────


def test_make_constructs_the_class_with_its_pose_and_the_overrides(manifest):
    opts = manifest.make("sofa", default_root_pos=(1.0, 2.0, 0.0), is_fixed_base=True)
    assert tuple(opts.default_root_pos) == (1.0, 2.0, 0.0) and opts.is_fixed_base is True
    assert tuple(opts.up) == (0, 0, 1) and opts.scale == 1.0  # canonical pose from the class


def test_queries_filter_by_category_placement_and_flags(manifest):
    assert set(manifest.names(category="furniture/seating")) == SEATING
    assert set(manifest.names(category="furniture")) >= {"sofa", "desk", "cabinet"}  # prefix subtree
    assert "monitor" in manifest.names(placement="wall")
    assert "monitor" not in manifest.names(placement="surface")
    assert "basket" in manifest.names(container=True)
    assert "desk" in manifest.names(support_surface=True) and "desk" in manifest.names(tag="table")
    openable = set(manifest.names(tag="openable"))
    assert {"fridge", "kitchen_counter", "chest_of_drawers", "sliding_cabinet", "door_1", "double_door"} <= openable
    assert set(manifest.names(category="architecture/door")) <= set(manifest.names(placement="wall"))


def test_facets_tree_and_summary_describe_the_manifest(manifest):
    facets = manifest.facets()
    assert set(facets) == {"category", "placement", "tag", "support_surface", "container", "movable"}
    assert "furniture/seating" in facets["category"] and "wall" in facets["placement"]
    assert set(manifest.tree()["furniture"]["seating"]["__assets__"]) == SEATING
    records = manifest.summary()
    assert {r["name"] for r in records} >= SEATING
    json.dumps(records)  # every record serializes (tags coerced to lists)


def test_ensure_does_not_rebuild_a_populated_manifest(manifest):
    n = len(list(manifest.keys()))
    assert manifest.ensure() is manifest
    assert len(list(manifest.keys())) == n


def test_get_of_an_unknown_asset_raises(manifest):
    with pytest.raises(KeyError):
        manifest.get("does_not_exist")


def test_importing_the_catalog_does_not_populate_the_manifest():
    """``import eden`` stays cheap: no asset family is imported until ``build()``/``ensure()``.

    Probed in a fresh interpreter, since this module has already populated the global manifest.
    """
    probe = (
        "import sys, eden.options.catalog as m\n"
        "assert 'eden.options.objects.replica_cad' not in sys.modules, 'family imported eagerly'\n"
        "assert not list(m.MANIFEST.keys()), 'MANIFEST populated on import'\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ── curation on a fresh manifest ──────────────────────────────────────────────


def test_search_matches_tags_descriptions_and_categories_case_insensitively():
    m = AssetManifest("s")
    m.curate("coffee_cup", "tableware/cup", placement="surface", tags=("cup", "mug"), description="a coffee cup")(
        _opts("c.glb")
    )
    m.curate("sofa", "furniture/seating", tags=("seating",), description="a couch to sit on")(_opts("s.glb"))

    assert m.search("mug") == ["coffee_cup"]
    assert m.search("couch") == ["sofa"]
    assert m.search("tableware") == ["coffee_cup"]
    assert m.search("CUP") == ["coffee_cup"]
    assert m.search("nope") == []
    with pytest.raises(ValueError):  # a blank query would match everything
        m.search("  ")


def test_recurating_the_same_asset_is_a_no_op_and_a_different_one_collides():
    m = AssetManifest("t")
    widget = _opts("widget.glb")
    m.curate("widget", "tools/gadget", placement="surface", container=True, tags=("demo",))(widget)
    assert m.get("widget").container and m.get("widget").category_path == ("tools", "gadget")
    m.curate("widget", "tools/gadget")(widget)
    assert list(m.keys()) == ["widget"]

    with pytest.raises(ValueError):
        m.curate("widget", "a/c")(_opts("other.glb"))


def test_variants_of_one_class_are_curated_by_make_kwargs():
    """The Objaverse shape: ``make_kwargs`` selects the variant, and explicit overrides still win."""
    m = AssetManifest("var")
    apple = _opts("apple_default.glb")
    m.curate("apple_3", "food/fruit", make_kwargs={"file": "apple_3.glb"})(apple)
    m.curate("apple_5", "food/fruit", make_kwargs={"file": "apple_5.glb"})(apple)

    assert m.make("apple_3").file == "apple_3.glb"
    assert m.make("apple_3", file="other.glb").file == "other.glb"
    m.curate("apple_3", "food/fruit", make_kwargs={"file": "apple_3.glb"})(apple)  # same variant: no-op
    with pytest.raises(ValueError):  # same name and class, different kwargs: a different asset
        m.curate("apple_3", "food/fruit", make_kwargs={"file": "apple_9.glb"})(apple)


def test_movable_is_derived_by_constructing_the_options():
    """Construction sees ``model_post_init`` (the ``fixed`` alias) and curated kwargs; a field read would not."""
    m = AssetManifest("mv")
    m.curate("freebie", "x")(_opts("f.glb"))

    class FixedThing(EntityOptions):
        file: str = "g.glb"
        is_fixed_base: bool = True

    m.curate("bolted", "x")(FixedThing)
    m.curate("freed", "x", make_kwargs={"is_fixed_base": False})(FixedThing)
    m.curate("ground", "architecture/floor")(PlaneOptions)  # ships the Genesis-style fixed=True

    assert {r["name"]: r["movable"] for r in m.summary()} == {
        "freebie": True,
        "bolted": False,
        "freed": True,
        "ground": False,
    }
    assert m.names(movable=False) == ["bolted", "ground"]
    assert m.facets()["movable"] == [False, True]


def test_boolean_facets_report_only_the_values_present():
    m = AssetManifest("fb")
    m.curate("a", "x", support_surface=False, container=False)(_opts("a.glb"))
    assert m.facets()["support_surface"] == [False] and m.facets()["container"] == [False]

    m.curate("b", "x", support_surface=True, container=True)(_opts("b.glb"))
    assert m.facets()["support_surface"] == [False, True] and m.facets()["container"] == [False, True]


def test_register_is_disabled_in_favour_of_curate():
    """The inherited ``register`` would store a raw class where every query expects an ``AssetSpec``."""
    m = AssetManifest("reg")
    with pytest.raises(NotImplementedError):
        m.register(_opts("foo.glb"))
    assert not list(m.keys())


# ── ROBOT_REGISTRY ────────────────────────────────────────────────────────────


def test_exactly_the_exported_robot_classes_are_registered():
    exported = {
        name
        for name in robots.__all__
        if isinstance(getattr(robots, name, None), type) and issubclass(getattr(robots, name), RobotOptions)
    }
    assert {cls.__name__ for _, cls in ROBOT_REGISTRY} == exported


@pytest.mark.parametrize(
    "key",
    # Keys tasks and CLIs name, including the overrides for what to_snake_case would mangle
    # (tesollo_dg5_f_l, x_hand1_l, widow_x250).
    [
        "franka_emika_panda",
        "g1_dofs29",
        "g1_dofs29_no_wrist_dof",
        "oli",
        "iiwa14",
        "tesollo_dg5f_l",
        "xhand1_r",
        "widowx250",
    ],
)
def test_robot_registry_keys(key):
    assert key in ROBOT_REGISTRY


@pytest.mark.parametrize("mangled", ["tesollo_dg5_f_l", "x_hand1_l", "widow_x250"])
def test_mangled_autoderived_keys_are_replaced_by_the_overrides(mangled):
    assert mangled not in ROBOT_REGISTRY


@pytest.mark.parametrize("obj", [BoxOptions, object], ids=["entity-options", "not-options"])
def test_robot_registry_rejects_a_non_robot(obj):
    with pytest.raises(TypeError, match="RobotOptions subclasses"):
        RobotRegistry("robot_test").register(obj, name="x")


@pytest.mark.parametrize("name", ["robot/franka_emika_panda", "franka_emika_panda", "FrankaEmikaPanda"])
def test_find_entity_options_resolves_a_robot_by_namespaced_key_bare_key_or_class_name(name):
    assert find_entity_options(name).__name__ == "FrankaEmikaPanda"
