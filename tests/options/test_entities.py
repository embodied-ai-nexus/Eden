"""Entity and scene options: validation where a field is written, and editing a scene after construction."""

from __future__ import annotations

import ast
import copy
import os
import pathlib

import genesis as gs
import pytest
from pydantic import ValidationError

from eden.options import BoxOptions, EntityOptions, EnvOptions, PlaneOptions, SphereOptions
from eden.options.entities import _DOF_FIELD_NAMES, GroupedEntityOptions, SceneOptions, TerrainOptions
from eden.options.actuators import Unitree7520_14
from eden.options.materials import (
    MPMElasticMaterialOptions,
    MochiElasticMaterialOptions,
    PBDClothMaterialOptions,
    RigidMaterialOptions,
    SPHLiquidMaterialOptions,
)
from eden.options.robots import PandaFrankaHand
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

JOINTLESS = [
    pytest.param(BoxOptions, {"size": (1.0, 1.0, 1.0)}, id="Box"),
    pytest.param(SphereOptions, {"radius": 0.1}, id="Sphere"),
    pytest.param(PlaneOptions, {}, id="Plane"),
    pytest.param(TerrainOptions, {}, id="Terrain"),
]


# ── DOF fields on a shape with no joints ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls, base", JOINTLESS)
@pytest.mark.parametrize(
    "dof_kwargs",
    [
        {"dofs_name": ["a"]},
        {"default_dofs_kp": {"a": 1.0}},
        {"default_dofs_pos": {"a": 0.5}},
        {"soft_dofs_pos_limits": {"a": (0.0, 1.0)}},
        {"is_actuated": True},
        {"is_articulated": True},
    ],
    ids=lambda kwargs: next(iter(kwargs)),
)
def test_a_dof_field_on_a_jointless_shape_raises_naming_it(cls, base, dof_kwargs):
    """They used to be blanked in silence, and the entity came back unactuated."""
    (field,) = dof_kwargs
    with pytest.raises(ValueError, match=rf"no joints.*{field}|{field}.*no joints"):
        cls(**base, **dof_kwargs)


def test_one_error_names_every_offending_dof_field():
    with pytest.raises(ValueError) as excinfo:
        BoxOptions(
            size=(1.0, 1.0, 1.0), dofs_name=["a"], default_dofs_kp={"a": 1.0}, is_actuated=True, is_articulated=True
        )
    for name in ("dofs_name", "default_dofs_kp", "is_actuated", "is_articulated"):
        assert name in str(excinfo.value)


@pytest.mark.parametrize(
    "make",
    [
        # Construction marks the blanked DOF fields as set, so a ``model_fields_set`` predicate would reject these.
        lambda: copy.deepcopy(BoxOptions(size=(1.0, 1.0, 1.0))),
        lambda: copy.deepcopy(PlaneOptions()),
        lambda: copy.deepcopy(TerrainOptions()),
        # The scene deep-copies its children.
        lambda: SceneOptions(plane=PlaneOptions(), box=BoxOptions(size=(0.1, 0.1, 0.1))).box,
        # A rigid primitive keeps its constraint fields, and a non-rigid one without DOF fields is fine.
        lambda: BoxOptions(size=(1.0, 1.0, 1.0), default_geoms_friction={"x": 1.0}),
        lambda: BoxOptions(size=(1.0, 1.0, 1.0), material=MPMElasticMaterialOptions()),
    ],
)
def test_a_legal_jointless_shape_still_constructs(make):
    assert make().default_dofs_kp == {}


def test_the_dof_rule_leaves_articulated_and_grouped_entities_alone():
    """Hoisting the check into ``BaseEntityOptions`` would pass every rejection above and break these."""
    options = EntityOptions(file="x.urdf", dofs_name=["j"], default_dofs_kp={"j": 10.0}, is_actuated=True)
    assert options.default_dofs_kp == {"j": 10.0}
    assert GroupedEntityOptions(dofs_name=["j"], default_dofs_kp={"j": 10.0}).default_dofs_kp == {"j": 10.0}


def test_a_dof_field_still_raises_through_the_scene():
    with pytest.raises(ValueError, match="no joints"):
        SceneOptions(plane=PlaneOptions(), box=BoxOptions(size=(0.1, 0.1, 0.1), default_dofs_kp={"a": 1.0}))


def test_a_saved_config_with_a_primitive_reloads(tmp_path):
    """Primitives were blanked before serialization, so an existing saved config cannot trip the new check."""
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(plane=PlaneOptions(), box=BoxOptions(size=(0.1, 0.1, 0.1))),
    )
    path = tmp_path / "config.json"
    config.save_as_file(str(path))
    reloaded = EdenConfig.load_from_file(str(path)).scene_options.box
    assert (reloaded.default_dofs_kp, reloaded.dofs_name) == ({}, [])


# ── rigid-solver fields on a non-rigid material ────────────────────────────────────────────────────


@pytest.mark.parametrize("cls, base", JOINTLESS)
def test_a_constraint_field_on_a_non_rigid_primitive_raises(cls, base):
    """The non-rigid guard used to inspect DOF fields that had already been blanked, so it never fired."""
    with pytest.raises(ValueError, match="Non-rigid entities have no DOFs"):
        cls(**base, material=MPMElasticMaterialOptions(), default_geoms_friction={"x": 1.0})


@pytest.mark.parametrize("cls, base", JOINTLESS)
def test_a_dof_field_on_a_non_rigid_primitive_still_raises(cls, base):
    """Either diagnosis is correct here; what matters is that it is no longer silent."""
    with pytest.raises(ValueError, match="default_dofs_kp"):
        cls(**base, material=MPMElasticMaterialOptions(), default_dofs_kp={"a": 1.0})


def test_a_dof_field_on_a_non_rigid_file_entity_names_the_non_rigid_reason():
    with pytest.raises(ValueError, match="Non-rigid entities have no DOFs"):
        EntityOptions(file="x.obj", material=MPMElasticMaterialOptions(), default_dofs_kp={"a": 1.0})


@pytest.mark.parametrize(
    "field, value",
    [
        ("dofs_name", ["joint0"]),
        ("default_dofs_frictionloss", {"joint": 0.1}),
        ("default_geoms_friction_torsional", {"base": 0.02}),
        ("default_geoms_friction_rolling", {"base": 0.02}),
    ],
)
@pytest.mark.parametrize("material", [SPHLiquidMaterialOptions, PBDClothMaterialOptions])
def test_rigid_solver_fields_are_rejected_on_a_particle_material(field, value, material):
    with pytest.raises(ValueError, match="Rigid-solver fields"):
        EntityOptions(file="dummy.obj", material=material(), **{field: value})


def test_every_per_dof_option_is_in_the_canonical_list():
    """``RobotStateEntity`` and every other enumerating site derive from it; a missing field is not cloned."""
    assert {name for name in EntityOptions.model_fields if "dofs" in name} == set(_DOF_FIELD_NAMES)


# ── support-surface fields ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "options_cls, extra", [(BoxOptions, {"size": (1.0, 1.0, 1.0)}), (PlaneOptions, {}), (TerrainOptions, {})]
)
def test_support_fields_are_accepted_by_every_rigid_options_class(options_cls, extra):
    options = options_cls(is_support_enabled=True, support_shrink=0.2, support_links_name=["table_top"], **extra)
    assert (options.support_shrink, options.support_links_name) == (0.2, ["table_top"])


@pytest.mark.parametrize(
    "make",
    [
        lambda m: BoxOptions(size=(1.0, 1.0, 1.0), material=m, is_support_enabled=True),
        lambda m: BoxOptions(size=(1.0, 1.0, 1.0), material=m, support_shrink=0.2),
        # TerrainOptions always builds a rigid Terrain, so its own check is the only one.
        lambda m: TerrainOptions(material=m, is_support_enabled=True),
        # A grouped entity would dispatch a non-rigid material to a particle entity.
        lambda m: GroupedEntityOptions(material=m, support_shrink=0.2),
        # Rebuilt from a dict, as a config reload does, with a real deviation from the default.
        lambda m: BoxOptions(**{**BoxOptions(size=(1.0, 1.0, 1.0)).model_dump(), "material": m, "support_shrink": 0.2}),
    ],
    ids=["box-enabled", "box-shrink", "terrain", "grouped", "reloaded"],
)
def test_support_fields_are_rejected_on_a_non_rigid_material(make):
    """Support extraction walks rigid collision meshes; a particle entity would silently drop the fields."""
    with pytest.raises(ValueError, match="Support-surface fields"):
        make(MPMElasticMaterialOptions())


def test_a_legal_non_rigid_entity_can_be_copied_and_restated():
    """Serialized load supplies every field, so the guard compares against defaults, not ``model_fields_set``."""
    for options in (
        EntityOptions(file="x.obj", material=MPMElasticMaterialOptions()),
        BoxOptions(size=(1.0, 1.0, 1.0), material=MPMElasticMaterialOptions()),
    ):
        copy.deepcopy(options)
    # Explicitly matching the default is not "setting" it.
    EntityOptions(file="x.obj", material=MPMElasticMaterialOptions(), support_shrink=0.1)


# ── SceneOptions: editing after construction ───────────────────────────────────────────────────────


def test_remove_entity_removes_by_name_and_names_what_is_there_otherwise():
    scene = SceneOptions(cube=BoxOptions(), ball=SphereOptions())

    scene.remove_entity("cube")

    assert "cube" not in scene.keys() and "ball" in scene.keys()
    with pytest.raises(KeyError, match=r"Entity 'cube' not found in SceneOptions; available: \['ball'\]"):
        scene.remove_entity("cube")
    with pytest.raises(KeyError, match="serialization metadata, not a removable entity"):
        scene.remove_entity("_option_type")
    scene.ghost = None  # a child stored as None is still a child
    scene.remove_entity("ghost")
    assert "ghost" not in scene.keys()


def test_remove_entity_reports_the_lock_first():
    scene = SceneOptions(arm=BoxOptions(), gripper=BoxOptions(), attachments_dict={"gripper": ("arm", "attachment")})
    scene.lock()

    with pytest.raises(RuntimeError, match="Cannot remove entity 'arm' because SceneOptions is locked"):
        scene.remove_entity("arm")
    assert "arm" in scene.keys()


def test_remove_entity_keeps_attachments_consistent():
    """A parent with a welded child is refused; removing the child takes its weld along, not the source's copy."""
    scene = SceneOptions(
        arm=BoxOptions(), gripper=BoxOptions(), cube=BoxOptions(), attachments_dict={"gripper": ("arm", "attachment")}
    )

    with pytest.raises(ValueError, match=r"Cannot remove entity 'arm': \['gripper'\] are attached to it"):
        scene.remove_entity("arm")
    source = scene.model_copy()  # shallow: shares attachments_dict
    scene.remove_entity("gripper")
    scene.remove_entity("arm")

    assert scene.attachments_dict == {}
    assert source.attachments_dict == {"gripper": ("arm", "attachment")}
    assert set(scene.keys()) == {"attachments_dict", "cube"}


def test_a_composite_is_removed_whole_and_only_whole():
    scene = SceneOptions(
        robot=PandaFrankaHand(),
        table=BoxOptions(),
        cam=BoxOptions(),
        attachments_dict={"robot": ("table", "box_baselink"), "cam": ("gripper", "hand")},
    )

    with pytest.raises(KeyError, match=r"'gripper' is a body of the composite 'robot' \(PandaFrankaHand\)"):
        scene.remove_entity("gripper")
    with pytest.raises(ValueError, match=r"Cannot remove entity 'robot': \['cam'\] are attached to it"):
        scene.remove_entity("robot")
    scene.remove_entity("cam")
    scene.remove_entity("robot")  # takes the composite's own attachment onto the table along

    assert set(scene.keys()) == {"attachments_dict", "table"} and scene.attachments_dict == {}


class _SceneWithDeclaredEntities(SceneOptions):
    """The ``DroidSceneOptions`` shape: entities declared as fields, not added by name."""

    robot: PandaFrankaHand = PandaFrankaHand()


def test_a_declared_entity_is_not_removable():
    scene = _SceneWithDeclaredEntities(cube=BoxOptions())

    with pytest.raises(KeyError, match="'robot' is a declared field of _SceneWithDeclaredEntities"):
        scene.remove_entity("robot")
    with pytest.raises(KeyError, match=r"'gripper' is a body of the composite 'robot'.*cannot be removed either"):
        scene.remove_entity("gripper")
    scene.remove_entity("cube")
    assert "cube" not in scene.keys()


def test_a_config_round_trips_after_remove_entity(tmp_path):
    config = EdenConfig(scene_options=SceneOptions(cube=BoxOptions(size=(0.1, 0.1, 0.1)), ball=SphereOptions()))
    config.scene_options.remove_entity("ball")
    path = os.path.join(tmp_path, "removed.json")

    config.save_as_file(path, lock_config=False)
    loaded = EdenConfig.load_from_file(path)

    assert loaded.scene_options.dict() == config.scene_options.dict()


def test_a_splat_options_saved_before_857_gets_a_migration_message():
    """``splat_options`` moved to the config root; in a scene it would otherwise be blamed as a bad entity."""
    from eden.options.splat import SplatOptions

    with pytest.raises(ValueError, match="not a scene field"):
        SceneOptions(splat_options=SplatOptions())


# ── the morph= escape hatch ────────────────────────────────────────────────────────────────────────


def test_primitives_take_a_prebuilt_morph_override_and_file_entities_refuse_one():
    """File-based entities never read ``options.morph``, so accepting it would ignore it silently."""
    box = BoxOptions(size=(0.1, 0.1, 0.1), morph={"pos": (1.0, 2.0, 3.0)})

    assert box._collect_morph_kwargs()["pos"] == (1.0, 2.0, 3.0)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EntityOptions(file="robot.urdf", morph={"pos": (1.0, 2.0, 3.0)})
    with pytest.raises(ValidationError, match="Morph or a kwargs dict"):
        BoxOptions(size=(0.1, 0.1, 0.1), morph=42)


# ── grouped, entity and terrain validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("grouped", "match"),
    [
        ([TerrainOptions()], None),
        ([EntityOptions(file="a.urdf", is_articulated=True), BoxOptions(size=(0.1, 0.1, 0.1))], "is_articulated"),
        (
            [
                EntityOptions(file="a.urdf", is_articulated=True, is_actuated=True),
                EntityOptions(file="b.urdf", is_articulated=True, is_actuated=False),
            ],
            "is_actuated",
        ),
    ],
    ids=["terrain-variant", "mixed-articulated", "mixed-actuated"],
)
def test_grouped_entities_must_be_alike(grouped, match):
    with pytest.raises(ValueError, match=match):
        GroupedEntityOptions(grouped_entities=grouped)


def _urdf(file: str, **fields) -> EntityOptions:
    return EntityOptions(file=file, is_articulated=True, dofs_name=["joint_a", "joint_b"], **fields)


def test_each_variant_takes_its_own_dof_values_else_the_groups():
    """Each variant keeps its own gains instead of the group copying the first variant's over all of them."""
    group = GroupedEntityOptions(
        grouped_entities=[_urdf("a.urdf", default_dofs_kp={"*": 500.0}), _urdf("b.urdf"), _urdf("c.urdf")],
        default_dofs_kp={"*": 50.0},
    )

    assert group.variant_values()["default_dofs_kp"] == [{"*": 500.0}, {"*": 50.0}, {"*": 50.0}]
    assert group.shared_dof_fields()["dofs_name"] == ["joint_a", "joint_b"]
    # Resolved on read, never copied: the group's own fields, and so its dump, hold only what was set on it.
    assert (group.dofs_name, group.default_dofs_kp) == ([], {"*": 50.0})


def test_a_shared_dof_field_comes_from_the_first_variant_that_sets_one():
    unnamed = EntityOptions(file="a.urdf", is_articulated=True)

    group = GroupedEntityOptions(grouped_entities=[unnamed, _urdf("b.urdf")])

    assert group.shared_dof_fields()["dofs_name"] == ["joint_a", "joint_b"]


def test_each_variant_takes_its_own_density_and_friction_else_the_groups():
    group = GroupedEntityOptions(
        material=RigidMaterialOptions(friction=0.5, gravity_compensation=1.0),
        grouped_entities=[
            BoxOptions(material=RigidMaterialOptions(rho=25.0, friction=2.0)),
            BoxOptions(default_geoms_friction={"*": 1.5}),
        ],
    )

    materials = group.variant_materials
    assert [(m.rho, m.friction, m.gravity_compensation) for m in materials] == [(25.0, 2.0, 1.0), (200.0, 0.5, 1.0)]
    assert group.variant_values()["default_geoms_friction"] == [{}, {"*": 1.5}]


class _FixedBox(BoxOptions):
    is_fixed_base: bool = True


class _CorkBox(BoxOptions):
    material: RigidMaterialOptions = RigidMaterialOptions(rho=120.0)


def test_a_variant_classs_own_default_counts_as_its_value():
    """A catalog class's default (LIBERO's ``is_fixed_base=True``, Polaris' ``rho``) is not in ``model_fields_set``,
    yet it counts as the variant's value rather than being silently dropped.
    """
    assert [m.rho for m in GroupedEntityOptions(grouped_entities=[_CorkBox(), BoxOptions()]).variant_materials] == [
        120.0,
        200.0,
    ]
    with pytest.raises(ValueError, match=r"grouped_entities\[0\] \(_FixedBox\) has is_fixed_base=True"):
        GroupedEntityOptions(grouped_entities=[_FixedBox(), _FixedBox()])
    assert GroupedEntityOptions(is_fixed_base=True, grouped_entities=[_FixedBox(), BoxOptions()]).is_fixed_base


class _SpecBox(BoxOptions):
    dofs_spec = {"*": Unitree7520_14}


class _SpecUrdf(EntityOptions):
    dofs_spec = {"*": Unitree7520_14}


@pytest.mark.parametrize(
    ("variants", "match"),
    [
        ([BoxOptions(), BoxOptions(color=(1.0, 0.0, 0.0))], r"grouped_entities\[1\] \(BoxOptions\) has color="),
        (
            [BoxOptions(), BoxOptions(material=SPHLiquidMaterialOptions())],
            r"has a SPHLiquidMaterialOptions material, but the group builds every variant with its own "
            r"RigidMaterialOptions",
        ),
        (
            [BoxOptions(material=RigidMaterialOptions(rho=25.0, gravity_compensation=1.0))],
            r"\{'gravity_compensation': 1.0\} differ from the group's \{'gravity_compensation': 0.0\}.*keep only "
            r"\['rho'\] on the variant's material",
        ),
        (
            [_urdf("a.urdf"), EntityOptions(file="b.urdf", is_articulated=True, dofs_name=["joint_b"])],
            r"grouped_entities\[1\] \(EntityOptions\) has dofs_name=\['joint_b'\], but every variant of a "
            r"heterogeneous entity shares one dofs_name \(dofs_name=\['joint_a', 'joint_b'\], taken from "
            r"grouped_entities\[0\]",
        ),
        (
            [_urdf("a.urdf"), _SpecUrdf(file="b.urdf", is_articulated=True)],
            r"grouped_entities\[1\] \(_SpecUrdf\) has dofs_spec \['\*'\]",
        ),
    ],
    ids=["color", "material-type", "material-field", "dofs-name", "dofs-spec"],
)
def test_a_variant_value_only_the_group_applies_is_refused_not_dropped(variants, match):
    """Genesis takes one surface and one material per entity, and the DOF structure is shared."""
    with pytest.raises(ValueError, match=match):
        GroupedEntityOptions(grouped_entities=variants)


def test_a_primitive_variant_brings_no_actuator_specs_from_its_class():
    """A primitive blanks its class's ``dofs_spec`` on the instance, as it would as an entity of its own."""
    assert GroupedEntityOptions(grouped_entities=[_SpecBox(), _SpecBox()]).shared_dofs_spec == {}
    assert GroupedEntityOptions(grouped_entities=[_SpecUrdf(file="a.urdf", is_articulated=True)]).shared_dofs_spec == {
        "*": Unitree7520_14
    }


class _GainedUrdf(EntityOptions):
    """A catalog-style class whose gains are a class default, not a value anyone passes in."""

    is_articulated: bool = True
    dofs_name: list[str] = ["joint_a", "joint_b"]
    default_dofs_kp: dict[str, float] = {"*": 50.0}


class _HeavyBox(BoxOptions):
    material: RigidMaterialOptions = RigidMaterialOptions(rho=120.0, gravity_compensation=1.0)


def test_a_class_default_that_overrides_the_groups_value_is_refused():
    """The group's gains would reach only the variants without a class default of their own."""
    with pytest.raises(
        ValueError,
        match=r"sets default_dofs_kp=\{'\*': 500.0\}, but grouped_entities\[0\] \(_GainedUrdf\) overrides it with "
        r"default_dofs_kp=\{'\*': 50.0\}, its class's default.*Fix: pass default_dofs_kp=\{'\*': 500.0\} to the "
        r"variant to take the group's value. To keep its class's value instead, move default_dofs_kp off the group",
    ):
        GroupedEntityOptions(
            default_dofs_kp={"*": 500.0}, grouped_entities=[_GainedUrdf(file="a.urdf"), _urdf("b.urdf")]
        )


def test_a_class_default_material_that_overrides_the_groups_is_refused_with_a_fix_that_keeps_its_other_fields():
    group_material = RigidMaterialOptions(rho=500.0, gravity_compensation=1.0)
    take_group = r"material=RigidMaterialOptions\(rho=500.0, gravity_compensation=1.0\)"

    with pytest.raises(ValueError, match=rf"sets material.rho=500.0, but .*\(_HeavyBox\).*Fix: pass {take_group}"):
        GroupedEntityOptions(material=group_material, grouped_entities=[_HeavyBox(), BoxOptions()])

    # The fix, as the message spells it.
    fixed = GroupedEntityOptions(
        material=group_material,
        grouped_entities=[_HeavyBox(material=RigidMaterialOptions(rho=500.0, gravity_compensation=1.0)), BoxOptions()],
    )
    assert [m.rho for m in fixed.variant_materials] == [500.0, 500.0]


def test_a_value_of_the_variants_own_overrides_the_groups_on_purpose():
    group = GroupedEntityOptions(
        default_dofs_kp={"*": 500.0},
        grouped_entities=[_GainedUrdf(file="a.urdf", default_dofs_kp={"*": 80.0}), _urdf("b.urdf")],
    )

    assert group.variant_values()["default_dofs_kp"] == [{"*": 80.0}, {"*": 500.0}]


def test_passing_the_class_default_again_still_reads_as_inherited():
    """Read off the value, not ``model_fields_set``, so a dump (every field passed back in) reads the same."""
    with pytest.raises(ValueError, match=r"\(_GainedUrdf\) overrides it with default_dofs_kp=\{'\*': 50.0\}"):
        GroupedEntityOptions(
            default_dofs_kp={"*": 500.0},
            grouped_entities=[_GainedUrdf(file="a.urdf", default_dofs_kp={"*": 50.0}), _urdf("b.urdf")],
        )


def test_an_override_that_adds_a_group_value_a_class_default_would_skip_is_refused():
    """``eden train --override`` dumps the config and re-validates it: every variant field comes back as passed in."""
    bank = GroupedEntityOptions(grouped_entities=[_GainedUrdf(file="a.urdf"), _urdf("b.urdf")])
    config = EdenConfig(scene_options=SceneOptions(bank=bank))

    with pytest.raises(ValueError, match=r"\(_GainedUrdf\) overrides it with default_dofs_kp=\{'\*': 50.0\}"):
        config.with_overrides_from_dict({"scene_options": {"bank": {"default_dofs_kp": {"*": 500.0}}}})


def test_a_class_default_equal_to_the_groups_value_is_no_override():
    group = GroupedEntityOptions(default_dofs_kp={"*": 50.0}, grouped_entities=[_GainedUrdf(file="a.urdf")])

    assert group.variant_values()["default_dofs_kp"] == [{"*": 50.0}]


def test_a_saved_group_with_a_variant_exception_loads_back(tmp_path):
    group = GroupedEntityOptions(
        default_dofs_kp={"*": 500.0},
        grouped_entities=[_GainedUrdf(file="a.urdf", default_dofs_kp={"*": 80.0}), _urdf("b.urdf")],
    )
    path = str(tmp_path / "config.json")
    EdenConfig(scene_options=SceneOptions(bank=group)).save_as_file(path, lock_config=False)

    loaded = EdenConfig.load_from_file(path).scene_options.bank
    assert loaded == group
    assert loaded.variant_values()["default_dofs_kp"] == [{"*": 80.0}, {"*": 500.0}]


def test_a_group_value_every_variant_overrides_is_refused():
    """It would apply to none of them; one variant left to take it is enough."""
    with pytest.raises(ValueError, match=r"sets \{'default_geoms_friction': \{'\*': 1.0\}\}, but every one of its 2"):
        GroupedEntityOptions(
            default_geoms_friction={"*": 1.0},
            grouped_entities=[BoxOptions(default_geoms_friction={"*": f}) for f in (2.0, 3.0)],
        )
    with pytest.raises(ValueError, match=r"sets \{'material.rho': 500.0\}"):
        GroupedEntityOptions(
            material=RigidMaterialOptions(rho=500.0),
            grouped_entities=[BoxOptions(material=RigidMaterialOptions(rho=rho)) for rho in (25.0, 50.0)],
        )

    GroupedEntityOptions(
        default_geoms_friction={"*": 1.0},
        grouped_entities=[BoxOptions(default_geoms_friction={"*": 2.0}), BoxOptions()],
    )


@pytest.mark.parametrize(
    "group",
    [
        GroupedEntityOptions(is_fixed_base=True, grouped_entities=[BoxOptions(), BoxOptions()]),
        GroupedEntityOptions(default_geoms_friction={"*": 2.0}, color=(1.0, 0.0, 0.0), grouped_entities=[BoxOptions()]),
        GroupedEntityOptions(
            material=RigidMaterialOptions(friction=0.5),
            grouped_entities=[BoxOptions(material=RigidMaterialOptions(rho=25.0)), BoxOptions()],
        ),
        GroupedEntityOptions(grouped_entities=[_urdf("a.urdf", default_dofs_kp={"*": 500.0}), _urdf("b.urdf")]),
    ],
    ids=["fixed-base", "geom-friction-and-color", "variant-density", "articulated"],
)
def test_a_saved_group_loads_back(group, tmp_path):
    """A saved config passes every variant field back in, so a variant's unset defaults must not read as its own."""
    path = str(tmp_path / "config.json")
    EdenConfig(scene_options=SceneOptions(bank=group)).save_as_file(path, lock_config=False)

    assert EdenConfig.load_from_file(path).scene_options.bank == group


def test_a_group_saved_when_it_copied_its_first_variants_dof_fields_loads_back():
    """A save from a group that copied its first variant's DOF fields onto itself holds the gains of a bank whose
    variants all carry the same ones on the group too; every variant overriding the group with that same value is not
    a group value that applies to none of them.
    """
    variants = [_urdf(file, default_dofs_kp={"*": 500.0}) for file in ("a.urdf", "b.urdf")]

    group = GroupedEntityOptions(
        grouped_entities=variants, dofs_name=["joint_a", "joint_b"], default_dofs_kp={"*": 500.0}
    )

    assert group.variant_values()["default_dofs_kp"] == [{"*": 500.0}, {"*": 500.0}]


def test_an_override_that_swaps_an_articulated_bank_applies_the_new_banks_dof_fields():
    """An override dumps, merges and re-validates; nothing of the old bank survives on the group."""

    def urdf(file: str, kp: float, joint: str) -> EntityOptions:
        return EntityOptions(file=file, is_articulated=True, dofs_name=[joint], default_dofs_kp={"*": kp})

    config = EdenConfig(
        scene_options=SceneOptions(
            bank=GroupedEntityOptions(grouped_entities=[urdf("a.urdf", 500.0, "j"), urdf("b.urdf", 500.0, "j")])
        )
    )
    new_bank = [urdf("c.urdf", 50.0, "k").model_dump(), urdf("d.urdf", 50.0, "k").model_dump()]

    bank = config.with_overrides_from_dict(
        {"scene_options": {"bank": {"grouped_entities": new_bank}}}
    ).scene_options.bank

    assert (bank.shared_dof_fields()["dofs_name"], bank.variant_values()["default_dofs_kp"]) == (
        ["k"],
        [{"*": 50.0}, {"*": 50.0}],
    )


def test_parallel_up_and_front_vectors_are_refused():
    with pytest.raises(ValueError, match="parallel"):
        EntityOptions(file="test.urdf", up=(0, 0, 1), front=(0, 0, 1))


def test_terrain_options_are_a_fixed_unactuated_body():
    terrain = TerrainOptions()

    assert (terrain.is_fixed_base, terrain.is_articulated, terrain.is_actuated) == (True, False, False)


def test_an_entity_added_to_the_scene_after_construction_is_built():
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(plane=PlaneOptions(), box=BoxOptions(size=(0.2, 0.2, 0.2))),
    )
    config.scene_options.sphere = SphereOptions(radius=0.1)
    env = build_env(config)
    assert set(env.entities) == {"plane", "box", "sphere"}
    assert isinstance(env.entities["sphere"].morph, gs.morphs.Sphere)


@pytest.mark.parametrize("file", ["objects/cup/mesh.obj", "objects/cup/MESH.GLB"])
def test_a_mesh_file_takes_a_scale_per_axis(file):
    assert EntityOptions(file=file, scale=(0.09, 0.07, 0.09)).scale == (0.09, 0.07, 0.09)


@pytest.mark.parametrize("file", ["objects/mochi/duck_1899.node", "objects/mochi/duck.vtk"])
def test_a_tetrahedral_mesh_file_takes_a_scale_per_axis(file):
    # The Mochi soft-body loader scales a tetrahedral mesh's vertices per axis, as a surface mesh's.
    options = EntityOptions(file=file, scale=(0.09, 0.07, 0.09), material=MochiElasticMaterialOptions())
    assert options.scale == (0.09, 0.07, 0.09)


@pytest.mark.parametrize("file", ["objects/cup/model.urdf", "objects/cup/model.xml", "objects/cup/model.usd"])
def test_an_articulated_asset_refuses_a_scale_per_axis_but_takes_an_even_one(file):
    with pytest.raises(ValueError, match=r"scale=\(0.09, 0.07, 0.09\).*file '.*' loads as a URDF / MJCF / USD"):
        EntityOptions(file=file, scale=(0.09, 0.07, 0.09))
    assert EntityOptions(file=file, scale=(0.5, 0.5, 0.5)).scale == (0.5, 0.5, 0.5)


# --- local_dir: resolved at load time, never baked in ------------------------------------------
#
# Both rules are source scans of eden/ and examples/: breaking either one still downloads (into the asset
# root instead of the family's subdirectory, or into one machine's path), so no behavioural test fails.

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _is_raw_local_dir(node: ast.AST) -> bool:
    """``<x>.local_dir`` read, or ``getattr(<x>, "local_dir", ...)``."""
    if isinstance(node, ast.Attribute):
        return node.attr == "local_dir" and isinstance(node.ctx, ast.Load)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "local_dir"
    )


def _raw_local_dir_arguments(source: str) -> list[int]:
    """The line of every ``local_dir=`` argument built from a raw ``local_dir`` read."""
    return [
        node.value.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.keyword)
        and node.arg == "local_dir"
        and any(_is_raw_local_dir(part) for part in ast.walk(node.value))
    ]


def _baked_local_dir_defaults(source: str) -> list[tuple[int, str]]:
    """``(line, class name)`` of every class body that gives ``local_dir`` a default other than ``None``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign):
                targets, value = [statement.target], statement.value
            elif isinstance(statement, ast.Assign):
                targets, value = statement.targets, statement.value
            else:
                continue
            named = any(isinstance(target, ast.Name) and target.id == "local_dir" for target in targets)
            if named and value is not None and not (isinstance(value, ast.Constant) and value.value is None):
                found.append((statement.lineno, node.name))
    return found


def _scanned_sources() -> list[pathlib.Path]:
    return sorted(path for top in ("eden", "examples") for path in (_REPO / top).rglob("*.py"))


@pytest.mark.parametrize(
    ("snippet", "flagged"),
    [
        ("get_asset_path(f, local_dir=options.local_dir)", True),
        ("get_asset_path(f, local_dir=self.local_dir or get_assets_dir())", True),
        ('get_asset_path(f, local_dir=getattr(options, "local_dir", None))', True),
        ("kwargs = dict(dataset=d, local_dir=self.local_dir)", True),
        ("get_asset_path(f, local_dir=options.resolve_local_dir())", False),
        ("get_asset_path(f, local_dir=None)", False),
        ("local_dir = args.local_dir or get_cache_dir()", False),
    ],
)
def test_the_raw_local_dir_scan_tells_a_raw_read_from_a_resolved_one(snippet, flagged):
    assert bool(_raw_local_dir_arguments(snippet)) is flagged


@pytest.mark.parametrize(
    ("snippet", "flagged"),
    [
        ('class A:\n    local_dir: str = os.path.join(get_assets_dir(), "ycb")', True),
        ("class A:\n    local_dir: str | None = ROOT", True),
        ("class A:\n    local_dir = ROOT", True),
        ("class A:\n    local_dir: str | None = None", False),
        ("class A:\n    local_dir: str | None", False),
        ('class A:\n    assets_subdir: ClassVar[str] = "ycb"', False),
    ],
)
def test_the_baked_default_scan_tells_a_computed_default_from_none(snippet, flagged):
    assert bool(_baked_local_dir_defaults(snippet)) is flagged


def test_nothing_fetches_with_a_raw_local_dir():
    """An unset local_dir means the family's assets_subdir, which only resolve_local_dir() adds.

    Passed raw, the None fetches into the asset root instead, and everything still works, so only this
    scan notices. Call ``options.fetch_asset(file)`` (or the policy's), or pass ``resolve_local_dir()``.

    It sees only a raw ``local_dir`` read. A call that leaves ``local_dir`` out, or passes ``None``, also
    fetches into the root without failing, and passes this scan: a pass does not mean every fetch lands in
    its family's folder.
    """
    offenders = [
        f"{path.relative_to(_REPO)}:{line}"
        for path in _scanned_sources()
        for line in _raw_local_dir_arguments(path.read_text())
    ]
    assert offenders == []


def test_no_class_bakes_a_local_dir_default():
    """A default computed at import pins one machine's path into saved configs and ignores EDEN_ASSETS_DIR.

    An asset family names its subdirectory with ``assets_subdir`` instead; a policy leaves ``local_dir`` None.
    """
    offenders = [
        f"{path.relative_to(_REPO)}:{line} ({name})"
        for path in _scanned_sources()
        for line, name in _baked_local_dir_defaults(path.read_text())
    ]
    assert offenders == []


def test_an_unset_local_dir_is_the_family_subdirectory_of_the_current_asset_root(
    user_assets_dir, monkeypatch, tmp_path
):
    from eden.options.objects.ycb import YCB002MasterChefCan

    can = YCB002MasterChefCan()
    assert can.resolve_local_dir() == str(user_assets_dir / "ycb")
    assert EntityOptions(file="robot.xml").resolve_local_dir() == str(user_assets_dir)

    monkeypatch.setenv("EDEN_ASSETS_DIR", str(tmp_path / "shared"))
    assert can.resolve_local_dir() == str(tmp_path / "shared" / "ycb")


def test_a_saved_config_carries_no_default_local_dir_and_an_explicit_one_wins(user_assets_dir):
    from eden.options.objects.ycb import YCB002MasterChefCan

    assert YCB002MasterChefCan().model_dump()["local_dir"] is None
    assert YCB002MasterChefCan(local_dir="/data/ycb").resolve_local_dir() == "/data/ycb"


def test_fetch_asset_reads_the_family_subdirectory_or_an_explicit_local_dir(user_assets_dir, tmp_path):
    """Pre-seeded files: a cached file is returned without a Hub request (a .txt has no dependencies)."""
    from eden.options.objects.ycb import YCB002MasterChefCan

    cached = user_assets_dir / "ycb" / "notes" / "can.txt"
    cached.parent.mkdir(parents=True)
    cached.write_text("can")
    assert YCB002MasterChefCan().fetch_asset("notes/can.txt") == str(cached)

    explicit = tmp_path / "mine" / "notes" / "can.txt"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("can")
    assert YCB002MasterChefCan(local_dir=str(tmp_path / "mine")).fetch_asset("notes/can.txt") == str(explicit)


def test_fetch_family_asset_reads_the_family_defaults_without_validating_an_entity(user_assets_dir):
    """A catalogue or background file is fetched through its family with no options instance, whatever its type."""
    from pydantic import model_validator

    from eden.options.objects.simpler import SimplerObjectOptions

    class RefusesEveryEntity(SimplerObjectOptions):
        @model_validator(mode="after")
        def _refuse(self):
            raise ValueError("built as an entity")

    cached = user_assets_dir / "simpler" / "real_inpainting" / "background.txt"
    cached.parent.mkdir(parents=True)
    cached.write_text("pixels")

    assert RefusesEveryEntity.fetch_family_asset("real_inpainting/background.txt") == str(cached)


def test_fetch_family_asset_refuses_a_family_with_no_hub_dataset():
    """Without a dataset there is nothing to fetch from: raise, rather than return ``None`` for a ``Path()`` to hit."""
    from eden.options.objects.calvin import CalvinObjectOptions

    class LocalOnly(CalvinObjectOptions):
        dataset: str | None = None

    with pytest.raises(ValueError, match=r"LocalOnly\.fetch_family_asset\('table\.png'\).*dataset=None"):
        LocalOnly.fetch_family_asset("table.png")
