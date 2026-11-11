"""``eden.options.materials``: deformable material options build the matching Genesis material.

The Mochi section also covers the option-level guards that turn a build-time MochiSolver error (or a silently
ignored field) into a validation error at the config: entity-class dispatch, ``EnvOptions.mochi_options``,
``RodOptions`` and the all-Mochi scene rule.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import pytest
from pydantic import ValidationError

from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi.morphs import Rod
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import _entity_class_for_options
from eden.options import RodOptions
from eden.options.entities import (
    BoxOptions,
    EntityOptions,
    GroupedEntityOptions,
    PlaneOptions,
    SceneOptions,
    SphereOptions,
    TerrainOptions,
)
from eden.options.envs import EnvOptions
from eden.options.materials import (
    KinematicMaterialOptions,
    MochiElasticMaterialOptions,
    MochiMaterialOptions,
    MochiRigidMaterialOptions,
    MochiRodMaterialOptions,
    MochiShellMaterialOptions,
    MPMElasticMaterialOptions,
    MPMElastoPlasticMaterialOptions,
    MPMLiquidMaterialOptions,
    MPMSandMaterialOptions,
    PBDClothMaterialOptions,
    PBDElasticMaterialOptions,
    PBDLiquidMaterialOptions,
    PBDParticleMaterialOptions,
    RigidMaterialOptions,
    SPHLiquidMaterialOptions,
)

MATERIALS = [
    (MPMElasticMaterialOptions, gs.materials.MPM.Elastic),
    (MPMElastoPlasticMaterialOptions, gs.materials.MPM.ElastoPlastic),
    (MPMLiquidMaterialOptions, gs.materials.MPM.Liquid),
    (MPMSandMaterialOptions, gs.materials.MPM.Sand),
    (SPHLiquidMaterialOptions, gs.materials.SPH.Liquid),
    (PBDLiquidMaterialOptions, gs.materials.PBD.Liquid),
    (PBDClothMaterialOptions, gs.materials.PBD.Cloth),
    (PBDElasticMaterialOptions, gs.materials.PBD.Elastic),
    (PBDParticleMaterialOptions, gs.materials.PBD.Particle),
]


@pytest.mark.parametrize(("options_cls", "genesis_cls"), MATERIALS, ids=lambda c: c.__name__)
def test_material_forwards_every_set_field_to_genesis(options_cls, genesis_cls):
    """Each declared field reaches the Genesis material, and a ``None`` default is dropped rather than forwarded.

    Several Genesis materials (``gs.materials.MPM.*``) reject an explicit ``sampler=None`` although ``None`` is
    their own default, so forwarding it would make the default-constructed options unusable.
    """
    options = options_cls(rho=321.0) if "rho" in options_cls.model_fields else options_cls()
    material = options.to_genesis_material()
    assert isinstance(material, genesis_cls)
    forwarded = {
        name: value
        for name in options_cls.model_fields
        if name != "use_visual_raycasting" and (value := getattr(options, name)) is not None
    }
    assert {name: getattr(material, name) for name in forwarded} == forwarded


@pytest.mark.parametrize("field", ["friction_torsional", "friction_rolling"])
def test_a_negative_rigid_friction_axis_is_rejected_at_construction(field):
    """Genesis would reject it only at build; the field constraint moves that to the config."""
    with pytest.raises(ValidationError, match="greater_than_equal"):
        RigidMaterialOptions(**{field: -1.0})


# --- Mochi (implicit multi-physics) materials -----------------------------------------------------------------

MOCHI_MATERIAL_CASES = [
    (MochiRigidMaterialOptions, mochi_materials.Rigid),
    (MochiElasticMaterialOptions, mochi_materials.Elastic),
    (MochiShellMaterialOptions, mochi_materials.Shell),
    (MochiRodMaterialOptions, mochi_materials.Rod),
]


@pytest.mark.parametrize("opt_cls, gs_cls", MOCHI_MATERIAL_CASES)
def test_mochi_material_roundtrips_to_genesis(opt_cls, gs_cls):
    """to_genesis_material() builds the matching gs.materials.Mochi class with every set field forwarded."""
    opt = opt_cls()
    mat = opt.to_genesis_material()
    assert isinstance(mat, gs_cls)
    for field, value in opt.dict().items():
        if field == "use_visual_raycasting":
            continue
        # ``None`` fields are dropped so the fork derives them (shell / rod stiffness) — a shell
        # resolves them at construction, a rod only once it knows the morph radius.
        if value is not None:
            assert getattr(mat, field) == value


def test_mochi_base_material_is_abstract():
    with pytest.raises(ValueError, match="abstract"):
        MochiMaterialOptions()


def test_mochi_rigid_material_has_no_coupler_fields():
    """Mochi.Rigid rejects the IPC/SAP coupler fields, so the wrapper must not carry them."""
    fields = MochiRigidMaterialOptions.model_fields
    assert "coup_type" not in fields and "gravity_compensation" not in fields and "needs_coup" not in fields
    assert "has_gravity" in fields


def test_mochi_entity_dispatch():
    from eden.entities.mochi import MochiRigidEntity, MochiSoftEntity

    assert _entity_class_for_options(BoxOptions(material=MochiRigidMaterialOptions())) is MochiRigidEntity
    for material in (MochiElasticMaterialOptions(), MochiShellMaterialOptions()):
        assert _entity_class_for_options(BoxOptions(material=material)) is MochiSoftEntity
    rod = RodOptions(points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material=MochiRodMaterialOptions())
    assert _entity_class_for_options(rod) is MochiSoftEntity


def test_env_options_accepts_mochi_options():
    env_opts = EnvOptions(num_envs=1, mochi_options=MochiOptions(n_newton_iterations=8))
    assert isinstance(env_opts.mochi_options, MochiOptions)
    assert EnvOptions(num_envs=1).mochi_options is None


def test_env_options_rejects_differentiable_mochi():
    with pytest.raises(ValueError, match="requires_grad"):
        EnvOptions(num_envs=1, requires_grad=True, mochi_options=MochiOptions())


@pytest.mark.requires_genesis
def test_a_mochi_material_in_a_differentiable_env_raises_at_its_entity():
    """Without mochi_options, EnvOptions cannot see the Mochi scene: its first Mochi entity refuses instead."""
    from eden.envs.base import EnvBase

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, requires_grad=True),
        scene_options=SceneOptions(plane=PlaneOptions(material=MochiRigidMaterialOptions())),
    )
    with pytest.raises(
        ValueError, match=r"entity 'plane' takes a Mochi material.*requires_grad=True.*not differentiable"
    ):
        env.build()


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda: BoxOptions(size=(0.1, 0.1, 0.1), material=MochiRigidMaterialOptions(), visualize_contact=True),
            id="primitive",
        ),
        pytest.param(
            lambda: EntityOptions(
                file="objects/cup/mesh.obj", material=MochiShellMaterialOptions(), visualize_contact=True
            ),
            id="file",
        ),
        pytest.param(
            lambda: RodOptions(
                points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material=MochiRodMaterialOptions(), visualize_contact=True
            ),
            id="rod",
        ),
    ],
)
def test_visualize_contact_is_refused_on_a_mochi_entity_at_validation(make):
    with pytest.raises(
        ValueError, match=r"visualize_contact=True cannot be used with Mochi.*Leave visualize_contact=False"
    ):
        make()


def test_rod_options_accepts_array_and_list_and_serializes():
    points = np.stack([np.linspace(0.0, 1.0, 5), np.zeros(5), np.zeros(5)], axis=-1)
    rod = RodOptions(points=points, radius=0.02, material=MochiRodMaterialOptions())
    morph = rod.get_morph()
    assert isinstance(morph, Rod)
    assert morph.points.shape == (5, 3) and morph.radius == 0.02
    restored = RodOptions.model_validate_json(rod.model_dump_json())
    assert np.allclose(np.asarray(restored.points), points)
    assert RodOptions(points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material=MochiRodMaterialOptions()).get_morph()


def test_rod_options_requires_rod_material_and_two_nodes():
    with pytest.raises(ValueError, match="MochiRodMaterialOptions"):
        RodOptions(points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], material=MochiElasticMaterialOptions())
    with pytest.raises(ValueError, match="n_nodes"):
        RodOptions(points=[[0.0, 0.0, 0.0]], material=MochiRodMaterialOptions())


def test_mochi_soft_material_rejects_dof_fields():
    with pytest.raises(ValueError, match="Rigid-solver fields"):
        SphereOptions(material=MochiElasticMaterialOptions(), dofs_name=["joint0"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_dofs_frictionloss": {"joint1": 0.1}},
        {"default_dofs_force_limits": {"joint1": 10.0}},
        {"default_geoms_sol_params": {".*": (0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)}},
    ],
)
def test_mochi_rigid_rejects_fields_the_solver_never_reads(kwargs):
    with pytest.raises(ValueError, match="MochiSolver"):
        EntityOptions(file="panda.xml", material=MochiRigidMaterialOptions(), **kwargs)


def test_mochi_rigid_rejects_support_surface():
    with pytest.raises(ValueError, match="Support-surface"):
        BoxOptions(material=MochiRigidMaterialOptions(), is_support_enabled=True)


def test_mochi_material_rejected_on_terrain_and_groups():
    with pytest.raises(ValueError, match="terrain"):
        TerrainOptions(material=MochiRigidMaterialOptions())
    with pytest.raises(ValueError, match="heterogeneous"):
        GroupedEntityOptions(grouped_entities=[BoxOptions(material=MochiRigidMaterialOptions())])


@pytest.mark.parametrize("material", [None, RigidMaterialOptions(), MochiRigidMaterialOptions()])
@pytest.mark.parametrize("file", ["objects/mochi/duck_1899.node", "objects/mochi/duck.VTK"])
def test_tet_mesh_file_requires_elastic_material(file, material):
    with pytest.raises(ValueError, match=r"is a tetrahedral mesh.*Pass material=MochiElasticMaterialOptions"):
        EntityOptions(file=file, material=material)
    # The one material that loads it; the suffix matches in any case, as at build (tests/entities/test_mochi.py).
    EntityOptions(file=file, material=MochiElasticMaterialOptions())


def test_scene_rejects_mixing_mochi_with_other_solvers():
    with pytest.raises(ValueError, match="plane"):
        SceneOptions(plane=PlaneOptions(), cube=BoxOptions(material=MochiRigidMaterialOptions()))
    # Kinematic (visual-only) entities are the one exception Genesis allows.
    SceneOptions(
        marker=BoxOptions(material=KinematicMaterialOptions()),
        cube=BoxOptions(material=MochiRigidMaterialOptions()),
    )
