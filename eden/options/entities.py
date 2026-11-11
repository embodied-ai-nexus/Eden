"""Entity configuration options (rigid/primitive/grouped) and metadata.

:class:`EntityOptions` describes an entity to add to the scene (file/morph, default root
pose, material). ``PrimitiveOptions`` is an abstract base — use its concrete subclasses
(``BoxOptions`` / ``SphereOptions`` / ``CylinderOptions`` / ``PlaneOptions``) for rigid
primitives. For deformables, position/orientation are baked in via ``default_root_pos``
and ``default_root_quat`` (the runtime ``ParticleEntity.set_pos`` / ``set_quat`` raise).
:class:`GroupedEntityOptions` supports heterogeneous batching of entities that share a
joint structure.
"""

from __future__ import annotations

import copy
import functools
from typing import ClassVar, Literal, TypeAlias

import genesis as gs
import numpy as np
from genesis.options.morphs import Morph
from genesis.typing import FArrayType, PositiveVec2FType, UnitVec3FType, UnitVec4FType, Vec2FType, Vec3FType
from pydantic import Field, field_validator

from eden.engine.solvers.mochi.morphs import Rod
from eden.options.actuators import ActuatorSpecOptions
from eden.options.materials import (
    FEMClothMaterialOptions,
    FEMElasticMaterialOptions,
    FEMMuscleMaterialOptions,
    KinematicMaterialOptions,
    MaterialLike,
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
from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions
from eden.options.surfaces import SurfaceLike
from eden.types import VisMode
from eden.utils.assets import get_asset_path
from eden.utils.misc import resolve_local_dir
from eden.utils.terrains import FlatTerrain

# Default Hugging Face location used to resolve entity assets by ``file``.
# Centralized here so it is obvious and overridable per-entity via ``registry`` / ``dataset``.
DEFAULT_ASSET_REGISTRY = "Kashu7100"
DEFAULT_ASSET_DATASET = "eden_assets"
#: Tetrahedral meshes (tetgen ``.node`` + sibling ``.ele``, legacy ASCII VTK); only a ``MochiElasticMaterialOptions``
#: entity reads them.
TET_MESH_FILE_SUFFIXES = (".node", ".vtk")
#: Files an entity loads as a mesh (``gs.morphs.Mesh``), the only morph that scales each axis on its own.
MESH_FILE_SUFFIXES = (".obj", ".stl", ".ply", ".glb", ".gltf") + TET_MESH_FILE_SUFFIXES

# Which solver — and therefore which Entity subclass — a material implies. These two are the
# single source of truth for that mapping and are public because
# ``eden.envs.base._entity_class_for_options`` dispatches on them: adding a deformable
# material here cannot silently fall through to RigidEntity there. Add new materials to
# whichever of the two owns them, never to the private aggregate below.
#
# Both are consumed only via ``isinstance``, so their order carries no meaning.
PARTICLE_MATERIAL_TYPES = (
    MPMElasticMaterialOptions,
    MPMElastoPlasticMaterialOptions,
    MPMLiquidMaterialOptions,
    MPMSandMaterialOptions,
    SPHLiquidMaterialOptions,
    PBDLiquidMaterialOptions,
    PBDClothMaterialOptions,
    PBDElasticMaterialOptions,
    PBDParticleMaterialOptions,
)

FEM_MATERIAL_TYPES = (
    FEMElasticMaterialOptions,
    FEMClothMaterialOptions,
    FEMMuscleMaterialOptions,
)

# The Mochi solver owns two entity families: rigid/articulated bodies keep the whole rigid-entity
# API (``MochiRigidEntity``), while solids, shells and rods are vertex-based (``MochiSoftEntity``).
MOCHI_RIGID_MATERIAL_TYPES = (MochiRigidMaterialOptions,)

MOCHI_SOFT_MATERIAL_TYPES = (
    MochiElasticMaterialOptions,
    MochiShellMaterialOptions,
    MochiRodMaterialOptions,
)

# Everything the rigid solver does not own — every material whose entity is a bag of vertices with
# no rigid pose, so it has no DOFs and no post-build ``set_quat``. Derived from the tuples above,
# which stay the contract; read by the validators here and by ``Entity._create_morph_from_options``,
# which bakes the orientation into the morph for exactly these materials.
NON_RIGID_MATERIAL_TYPES = PARTICLE_MATERIAL_TYPES + FEM_MATERIAL_TYPES + MOCHI_SOFT_MATERIAL_TYPES

_DOF_FIELD_NAMES = (
    "dofs_name",
    "default_dofs_pos",
    "default_dofs_vel",
    "default_dofs_stiffness",
    "default_dofs_damping",
    "default_dofs_armature",
    "default_dofs_frictionloss",
    "default_dofs_spring_ref",
    "default_dofs_kp",
    "default_dofs_kd",
    "default_dofs_pos_limits",
    "soft_dofs_pos_limits",
    "default_dofs_force_limits",
)

# The two ``_DOF_FIELD_NAMES`` entries ``RigidEntity.post_build`` never resolves to a per-DOF
# tensor: ``dofs_name`` is the name list itself, and ``default_dofs_spring_ref`` is folded into
# the position servo (``RigidEntity._apply_dofs_spring_ref``) rather than stored as one.
_UNRESOLVED_DOF_FIELD_NAMES = ("dofs_name", "default_dofs_spring_ref")

#: The per-DOF fields ``RigidEntity.post_build`` *may* resolve to a
#: ``(1 | num_envs, num_dofs[, 2])`` tensor — not the fields a built entity is guaranteed to
#: carry as one. Resolution is conditional: a kinematic articulated entity gets only
#: ``default_dofs_pos`` and ``default_dofs_pos_limits``, a non-actuated one no
#: ``default_dofs_kp``/``_kd``, and one that is neither articulated nor actuated none of them
#: (``_clear_dof_properties`` blanks those to ``None``). Every name here still answers an
#: attribute read on the entity either way, because they are mirrored config fields — with the
#: raw configured value where nothing was resolved — so a consumer that iterates this list must
#: check the type rather than assume a tensor.
#:
#: Public and *derived*, so that the sites which have to enumerate them —
#: ``RigidEntity._clear_dof_properties`` and the connect-time clone in
#: ``RobotStateEntity`` — cover a field added above without a second edit. Restating the list
#: at those sites is how a property silently goes back to falling through to the sim entity.
#: A new field that is **never** resolved to a tensor belongs in ``_UNRESOLVED_DOF_FIELD_NAMES``.
RESOLVED_DOF_TENSOR_FIELD_NAMES = tuple(n for n in _DOF_FIELD_NAMES if n not in _UNRESOLVED_DOF_FIELD_NAMES)

# Rigid-solver-only fields that are not per-DOF (constraint solver parameters). Public because
# ``RigidEntity`` enforces the same list against a *kinematic* entity, which has no contacts or
# constraints either — one list, so a new field is guarded at both sites.
RIGID_CONSTRAINT_FIELD_NAMES = (
    "default_equalities_sol_params",
    "default_geoms_sol_params",
    "default_geoms_friction",
    "default_geoms_friction_torsional",
    "default_geoms_friction_rolling",
)

# Rigid-entity fields the MochiSolver has no counterpart for. It never reads a DOF force range or
# friction loss and has no constraint solver, so accepting these would silently do nothing.
MOCHI_RIGID_UNSUPPORTED_FIELD_NAMES = (
    "default_dofs_frictionloss",
    "default_dofs_force_limits",
) + RIGID_CONSTRAINT_FIELD_NAMES

# What a ``GroupedEntityOptions`` variant may hold besides its morph. A variant "holds" a value when it
# differs from the field's ``BaseEntityOptions`` default, whether passed in or its catalog class's default: not
# ``model_fields_set``, since a saved config passes every field back in.
#
# Per variant: the entity applies a variant's own value to the envs holding it, and the group's to every variant
# without one. The geom fields go over the variant's own collision geoms, which Genesis builds per variant; the
# per-DOF fields become one row per env (``RigidEntity._init_dofs_property``).
_PER_VARIANT_FIELD_NAMES = (
    "default_geoms_sol_params",
    "default_geoms_friction",
    "default_geoms_friction_torsional",
    "default_geoms_friction_rolling",
    *RESOLVED_DOF_TENSOR_FIELD_NAMES,
)
# The material fields a variant may hold: density, applied after the build as a per-env mass, and the friction
# coefficients, written over the variant's geoms. Genesis builds every variant with the group's one material, so
# any other material field is the group's alone.
_PER_VARIANT_MATERIAL_FIELD_NAMES = ("rho", "friction", "friction_torsional", "friction_rolling")
# Shared: one value for every variant, the group's own or, when it sets none, that of the first variant that sets
# one. ``dofs_name`` is the DOF structure every variant must have; a spring reference moves the DOF onto a position
# servo.
_SHARED_DOF_FIELD_NAMES = _UNRESOLVED_DOF_FIELD_NAMES
# Group only: applied once from the group, so a variant's own value is refused. Genesis takes one surface and one
# visualization mode per entity, the base joint is structure, and Eden applies the collision mask, the support
# surface and the equality parameters to the whole entity.
_GROUP_ONLY_FIELD_NAMES = (
    "surface",
    "color",
    "visualize_contact",
    "vis_mode",
    "collision_link_patterns",
    "collision_friction",
    "is_fixed_base",
    "is_support_enabled",
    "support_sample_method",
    "support_minimum_clearance",
    "support_shrink",
    "support_num_sample_points",
    "support_grid_size",
    "support_links_name",
    "default_equalities_sol_params",
)


def _unconfigured(name: str) -> object:
    """The value of entity field ``name`` when nothing configures it: its ``BaseEntityOptions`` default."""
    return BaseEntityOptions.model_fields[name].get_default(call_default_factory=True)


def _has_own_value(options: BaseEntityOptions, name: str) -> bool:
    """Whether ``options`` holds a value of its own for entity field ``name`` (see ``_PER_VARIANT_FIELD_NAMES``)."""
    return getattr(options, name) != _unconfigured(name)


# Constraint solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power);
# see the MuJoCo "Solver parameters" docs. The fixed-length tuple makes Pydantic reject
# a wrong element count at construction time.
SolParamsType = tuple[float, float, float, float, float, float, float]


def _validate_no_dofs_for_non_rigid_material(opts) -> None:
    """Raise ValueError if rigid-solver fields are set on a non-rigid (particle/FEM) entity.

    Must be called *before* ``_set_dofs_unactuated``, which blanks the DOF half of what this
    inspects and would leave the check vacuous.
    """
    if not isinstance(opts.material, NON_RIGID_MATERIAL_TYPES):
        return
    set_fields = [name for name in _DOF_FIELD_NAMES + RIGID_CONSTRAINT_FIELD_NAMES if getattr(opts, name, None)]
    if set_fields:
        raise ValueError(
            f"Rigid-solver fields {set_fields} cannot be used with {type(opts.material).__name__}. "
            f"Non-rigid entities have no DOFs or constraint-solver parameters."
        )


def _validate_mochi_rigid_fields(opts) -> None:
    """Raise ValueError if a Mochi rigid entity declares fields the MochiSolver cannot honor.

    Force ranges, friction loss and constraint-solver parameters are never read by the Mochi
    solver (drive forces are not clamped, joints have no dry friction, contact is a penalty).
    Genesis raises on the corresponding setters, but only at build — reject the options here so
    the message names the field where it was written.
    """
    if not isinstance(opts.material, MOCHI_RIGID_MATERIAL_TYPES):
        return
    set_fields = [name for name in MOCHI_RIGID_UNSUPPORTED_FIELD_NAMES if getattr(opts, name, None)]
    if set_fields:
        raise ValueError(
            f"{set_fields} cannot be used with {type(opts.material).__name__}: the MochiSolver clamps no drive "
            f"force, models no joint friction loss and has no constraint solver. Contact behaviour is set through "
            f"the material's contact fields (`friction`, `penalty_coefficient`, ...) or "
            f"`MochiRigidEntity.set_contact_params`."
        )


def _validate_mochi_visualize_contact(opts) -> None:
    """Raise ValueError if a Mochi entity asks for ``visualize_contact``: it draws the RigidSolver's contacts only.

    The MochiSolver refuses it too, but only as the entity is added at build, after earlier entities have loaded.
    """
    if opts.visualize_contact and isinstance(opts.material, MochiMaterialOptions):
        raise ValueError(
            f"visualize_contact=True cannot be used with {type(opts.material).__name__}: it draws the RigidSolver's "
            f"contacts, and a Mochi entity has none. Leave visualize_contact=False."
        )


def _validate_no_support_for_non_rigid_material(opts) -> None:
    """Raise ValueError if support-surface fields are set on a non-rigid (particle/FEM/Mochi) entity.

    Support extraction walks a *rigid* entity's collision meshes (``RigidEntity.post_build`` ->
    ``SupportSurfaceMixin._prepare_support``), so ``is_support_enabled`` and the ``support_*``
    sampling knobs are rigid-solver-only — a particle/FEM entity would silently drop them.

    Unlike ``RIGID_CONSTRAINT_FIELD_NAMES``, the guarded set is *derived* from the options' own
    fields (``is_support_enabled`` plus every ``support_*`` field) rather than kept as an exported
    tuple: this validator is the only consumer, and ``RigidEntity`` gates support entirely on the
    single ``is_support_enabled`` flag, so there is no second site that needs a stable list.
    Deriving it also guards a ``support_*`` field a subclass adds without a matching edit here.

    Membership can use neither of the two obvious predicates:

    * **truthiness** — every field but ``is_support_enabled`` has a non-empty default
      (``"uniform"``, ``0.05``, ``0.1``, ``300``, ``0.01``), so it would fire for every entity;
    * **``model_fields_set``** — a saved config contains every serialized field, so validated
      load marks defaults and caller choices alike as explicitly set. The set also includes
      fields assigned by construction hooks, and therefore does not reliably encode caller
      intent across all supported construction paths.

    So compare against each field's own default instead: a value that differs from it can only
    have been chosen by the caller, and a deep-copied clean entity still matches.
    """
    # Mochi rigid bodies have DOFs but no collider/AABB API for support extraction to walk.
    if not isinstance(opts.material, NON_RIGID_MATERIAL_TYPES + MOCHI_RIGID_MATERIAL_TYPES):
        return
    set_fields = [
        name
        for name, field in type(opts).model_fields.items()
        if (name == "is_support_enabled" or name.startswith("support_"))
        and getattr(opts, name, None) != field.get_default(call_default_factory=True)
    ]
    if set_fields:
        raise ValueError(
            f"Support-surface fields {set_fields} cannot be used with {type(opts.material).__name__}. "
            f"Support extraction reads a rigid entity's collision meshes; only rigid entities "
            f"(RigidEntity / Terrain) can be placement targets."
        )


# Blanked by ``_set_dofs_unactuated`` alongside ``_DOF_FIELD_NAMES``; a caller that sets either
# on a shape with no joints has made the same mistake.
_UNACTUATED_FLAG_NAMES = ("is_articulated", "is_actuated")


def _reject_dofs_on_never_articulated(opts) -> None:
    """Raise ValueError if DOF fields are set on a shape that can never have joints.

    Primitives and terrains are single rigid bodies, so ``_set_dofs_unactuated`` blanks every
    DOF field on them; without this check that would silently discard whatever the caller wrote: a
    box given ``default_dofs_kp`` would come back unactuated, with nothing pointing at the config.

    Predicated on the value being **non-empty**, not on ``model_fields_set``. Two reasons:

    * a saved config contains every serialized field, so validated load makes
      ``model_fields_set`` all-ones and a predicate based on it would reject every loaded
      primitive even when all DOF fields merely carry their defaults;
    * ``_set_dofs_unactuated`` itself assigns these fields, and pydantic's ``__setattr__`` adds
      them to ``model_fields_set`` too, so it is already polluted by the time anything downstream
      could look.

    Every DOF field on every primitive/terrain options class defaults to empty (``[]`` / ``{}`` /
    ``False``), so a non-empty value can only have come from the caller. ``_set_dofs_unactuated``
    still runs afterwards, so a future subclass that legitimately carries a non-empty default is
    blanked rather than becoming an error.
    """
    set_fields = [name for name in _DOF_FIELD_NAMES + _UNACTUATED_FLAG_NAMES if getattr(opts, name, None)]
    if set_fields:
        raise ValueError(
            f"DOF fields {set_fields} cannot be used with {type(opts).__name__}: a primitive or "
            f"terrain is a single rigid body with no joints, so these would be discarded. Use a "
            f"mesh/URDF entity (EntityOptions) if the object needs articulated DOFs."
        )


class MetadataOptions(DynamicConfigurableOptions):
    """
    Metadata options for Entity, used to store additional metadata about the entity.

    Parameters
    ----------
    <key>: Any
        The value of the metadata.
    """


class BaseEntityOptions(ConfigurableOptions):
    """
    Shared options for entities.

    Parameters
    ----------
    default_root_pos : array-like[float, float, float]
        Default root position.
    default_root_quat : array-like[float, float, float, float]
        Default root orientation as quaternion in (w, x, y, z).
    is_fixed_base : bool
        Whether the entity base is fixed.
    is_articulated : bool
        Whether the entity has articulation.
    is_actuated : bool
        Whether the entity is actuated.
    material : MaterialLike | None
        Material used for the entity.
    surface : SurfaceOptions | Surface | None
        Surface appearance used for the entity. Prefer an Eden :class:`SurfaceOptions`
        (e.g. ``PlasticSurfaceOptions(color=(1.0, 0.0, 0.0))``) in configs constructed
        before ``gs.init()``; a raw ``gs.surfaces.*`` instance is also accepted.
    color : array-like[float] | None
        Convenience shortcut to color the entity. When set (and ``surface`` is not),
        a plastic surface with this color (RGB or RGBA) is created at build time.
    visualize_contact : bool
        Whether to visualize contacts involving this entity.
    vis_mode : VisMode
        Visualization mode for the entity.
    visualization : bool
        Whether to visualize the entity.
    collision : bool
        Whether this entity participates in collision checking.
    contype : int
        32-bit collision-filter bitmask. Two geoms can collide only if the ``contype`` of one
        and the ``conaffinity`` of the other share a set bit. Defaults to 0xFFFF.
    conaffinity : int
        32-bit collision-filter bitmask. See ``contype``. Defaults to 0xFFFF.
    collision_link_patterns : list of str, optional
        Regex/glob patterns matched against link names. When set, only geoms on
        matching links keep collision; geoms on all other links are disabled
        (``contype = conaffinity = 0``). Use to restrict an entity's collisions
        to specific parts — e.g. a parallel-jaw gripper's fingertip pads
        (``[".*pad"]``) so objects are grasped at the pads instead of wedging
        against the inner linkage. Applied before the scene is built (Genesis
        bakes the collision-pair list at build time). ``None`` keeps the morph's
        default per-geom collision.
    collision_friction : float, optional
        Friction coefficient applied to the geoms kept by
        ``collision_link_patterns`` (e.g. high pad friction for stable grasps).
    batch_fixed_verts : bool
        Whether fixed-geometry vertices are batched for per-env transforms.
    is_support_enabled : bool
        Whether support-point computation is enabled.
    support_sample_method : Literal["uniform", "grid"]
        The method to sample the support points. Defaults to "uniform".
    support_minimum_clearance : float
        The minimum clearance to be considered for support points. Defaults to 0.05.
    support_shrink : float
        The shrink factor to be applied to the support points. Defaults to 0.1.
    support_num_sample_points : int
        The number of support points to sample per square meter of polygon. Defaults to 300.
    support_grid_size : float
        The size of the grid to sample the support points when ``support_sample_method``
        is "grid". Defaults to 0.01.
    support_links_name : list[str]
        Exact link names (URDF link names / MJCF body names) whose geometry should
        be considered when extracting support surfaces. Empty (default) means all
        links contribute. Use this on multi-link assets
        (e.g. the Riverway franka station) to restrict placement targets to a
        specific surface like ``["station_table_top"]`` so events don't spawn
        objects onto auxiliary cart / gantry tops.
    dofs_name : list[str]
        Controllable DOF names.
    dofs_spec : ClassVar[dict[str, type[ActuatorSpecOptions]]]
        Class-level default actuator specification map for DOFs.
    default_dofs_pos : dict[str, float]
        Default DOF positions.
    default_dofs_vel : dict[str, float]
        Default DOF velocities.
    default_dofs_kp : dict[str, float]
        Default DOF position gains.
    default_dofs_kd : dict[str, float]
        Default DOF velocity gains.
    default_dofs_stiffness : dict[str, float]
        Default DOF stiffness values.
    default_dofs_damping : dict[str, float]
        Default DOF damping values.
    default_dofs_armature : dict[str, float]
        Default DOF armature values.
    default_dofs_frictionloss : dict[str, float]
        Default DOF dry-friction (stiction) torques. Values below the drive torque leave
        actuation unaffected while absorbing sub-threshold oscillation (contact chatter).
    default_dofs_spring_ref : dict[str, float]
        Re-anchor a DOF's passive spring (``default_dofs_stiffness`` / the file's joint
        ``stiffness``) at this reference position instead of the initial qpos. Genesis
        ignores MJCF ``springref`` and anchors joint springs at ``qpos0``, which silently
        drops any preload (e.g. the Robotiq 2F85's spring links). Implemented at build by
        moving the spring to a position servo: sim stiffness is zeroed and the DOF gets
        ``kp = stiffness`` with the reference as its (persistent) position target — do not
        also drive these DOFs from an action term.
    default_dofs_pos_limits : dict[str, array-like[float, float]]
        Hard DOF position limits.
    soft_dofs_pos_limits : dict[str, array-like[float, float]]
        Soft DOF position limits.
    default_dofs_force_limits : dict[str, float]
        Default DOF force limits.
    default_equalities_sol_params : SolParamsType | None
        Constraint solver parameters ``(timeconst, dampratio, dmin, dmax, width, mid,
        power)`` applied to **all** of the entity's equality constraints (loop closures,
        joint couplings) at build. Note genesis clamps ``timeconst`` to ``2*substep_dt``,
        so a stiff value needs a matching ``sim_dt``/``sim_substeps``.
    default_geoms_sol_params : dict[str, SolParamsType]
        Constraint solver parameters for the collision geoms of links matching each
        key (exact name, glob, or regex — same semantics as ``collision_link_patterns``).
        Note genesis mixes a contact pair's parameters as the *mean* of its two geoms'
        values (MuJoCo geom ``priority`` is not honored), so the partner geom should be
        stiffened too for the pair to reach the requested stiffness.
    default_geoms_friction : dict[str, float]
        Friction for the collision geoms of links matching each key (same pattern semantics
        as ``default_geoms_sol_params``), applied at build — e.g. a dexterous hand's
        fingertips (``{".*_DP": 1.5}``) while the whole hand keeps colliding. Unlike sol
        params the partner geom needs no matching value, since genesis combines a pair's
        friction as ``max(a, b)``. Prefer this over ``collision_friction``, which only covers
        the geoms kept by ``collision_link_patterns`` and disables collision elsewhere.
    default_geoms_friction_torsional : dict[str, float]
        Torsional friction (spin resistance about the contact normal, in meters) for the collision
        geoms of links matching each key — same pattern and ``max(a, b)`` pair semantics as
        ``default_geoms_friction``. Use it when only *part* of an entity needs the strength the
        material-wide ``RigidMaterialOptions.friction_torsional`` would apply everywhere, e.g. a
        dexterous hand whose fingertips resist an object twisting in the grasp more than its palm
        does. Written after the material coefficient, so it wins on the geoms it matches. Requires
        ``EnvOptions.enable_torsional_friction``.
    default_geoms_friction_rolling : dict[str, float]
        Rolling friction (resistance about the two contact tangent axes, in meters) for the collision
        geoms of links matching each key. Same semantics as ``default_geoms_friction_torsional``, but
        requires ``EnvOptions.enable_rolling_friction`` and costs two extra constraint axes per
        contact instead of one.
    """

    material: MaterialLike | None = None
    surface: SurfaceLike | None = None
    color: FArrayType | None = None
    visualize_contact: bool = False
    vis_mode: VisMode = "visual"
    metadata: ClassVar[MetadataOptions | None] = None
    visualization: bool = True
    collision: bool = True
    contype: int = 0xFFFF
    conaffinity: int = 0xFFFF
    collision_link_patterns: list[str] | None = None
    collision_friction: float | None = None
    batch_fixed_verts: bool = False
    # Support-surface sampling lives on the *base* options: ``is_support_enabled`` is honored for
    # every rigid entity (``RigidEntity.post_build``), so a primitive or a terrain used as a table
    # must be able to configure the sampling too — not just switch it on.
    is_support_enabled: bool = False
    support_sample_method: Literal["uniform", "grid"] = "uniform"
    support_minimum_clearance: float = 0.05
    support_shrink: float = 0.1
    support_num_sample_points: int = 300
    support_grid_size: float = 0.01
    support_links_name: list[str] = []  # empty = all links contribute
    default_root_pos: Vec3FType = (0.0, 0.0, 0.0)
    default_root_quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    is_fixed_base: bool = False
    is_articulated: bool = False
    is_actuated: bool = False
    dofs_name: list[str] = []
    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = {}
    default_dofs_pos: dict[str, float] = {}
    default_dofs_vel: dict[str, float] = {}
    default_dofs_kp: dict[str, float] = {}
    default_dofs_kd: dict[str, float] = {}
    default_dofs_stiffness: dict[str, float] = {}
    default_dofs_damping: dict[str, float] = {}
    default_dofs_armature: dict[str, float] = {}
    default_dofs_frictionloss: dict[str, float] = {}
    default_dofs_spring_ref: dict[str, float] = {}
    default_dofs_pos_limits: dict[str, Vec2FType] = {}
    soft_dofs_pos_limits: dict[str, Vec2FType] = {}
    default_dofs_force_limits: dict[str, float] = {}
    default_equalities_sol_params: SolParamsType | None = None
    default_geoms_sol_params: dict[str, SolParamsType] = {}
    default_geoms_friction: dict[str, float] = {}
    default_geoms_friction_torsional: dict[str, float] = {}
    default_geoms_friction_rolling: dict[str, float] = {}

    def _set_dofs_unactuated(self):
        """Set DOF fields to unactuated defaults."""
        # ``dofs_spec`` is a ClassVar (Pydantic does not treat it as a field, so it cannot be
        # assigned through normal validation/``__setattr__``). ``object.__setattr__`` shadows it
        # on *this instance* only — assigning the class attribute would corrupt every other
        # instance/subclass that shares the same default.
        object.__setattr__(self, "dofs_spec", {})
        self.dofs_name = []
        self.default_dofs_pos = {}
        self.default_dofs_vel = {}
        self.default_dofs_stiffness = {}
        self.default_dofs_damping = {}
        self.default_dofs_armature = {}
        self.default_dofs_frictionloss = {}
        self.default_dofs_spring_ref = {}
        self.default_dofs_kp = {}
        self.default_dofs_kd = {}
        self.default_dofs_pos_limits = {}
        self.soft_dofs_pos_limits = {}
        self.default_dofs_force_limits = {}
        self.is_articulated = False
        self.is_actuated = False


class EntityOptions(BaseEntityOptions):
    """
    Entity options for Eden, wrapping Genesis' Entity related options.

    Parameters
    ----------
    up: array-like[int, int, int]
        The object's up direction in its own coordinate system (use `gs view` to visualize that coordinate
        system). On loading, the object is rotated so this axis becomes world +z.
    front: array-like[int, int, int]
        The object's front direction in its own coordinate system (use `gs view` to visualize that coordinate
        system). On loading, the object is rotated so this axis becomes world +x.
    local_dir: str or None
        Where the asset is cached. Unset, it is the family's ``assets_subdir`` under the asset root, resolved at
        load time by :meth:`resolve_local_dir`; :meth:`fetch_asset` downloads into it.
    file: str
        Path to the entity file.
    scale: float or tuple[float, float, float]
        Scale of the entity: uniform, or along each axis of the asset's frame (a scanned mesh
        measured to a real object's size, say). Unequal axes need a mesh ``file``
        (:data:`MESH_FILE_SUFFIXES`); URDF, MJCF and USD assets scale uniformly.
    collision: bool
        Whether the entity needs to be considered for collision checking. Defaults to True.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Defaults to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 1000. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as an integer ranging from 0
        to 8. 0 is lossless. 2 preserves all features of the original geometry. 5 may significantly alter the
        original geometry if necessary. 8 does what needs to be done at all costs. Defaults to 3.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh will be decomposed into multiple convex components if the convex hull is not
        sufficient to meet the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_object_error_threshold : float, optional
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : float, optional
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threshold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_threshold : float, optional
        The threshold for the CoACD algorithm. Defaults to 0.05.
    coacd_preprocess_resolution : int, optional
        The resolution for the CoACD algorithm. Defaults to 100.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to [].
    align : bool | None, optional
        Whether Genesis re-frames a floating root link onto its centre of mass and principal axes. ``None``
        (default) lets Genesis decide: on for a basic rigid object (one free root body, nothing articulated under
        it), off otherwise. An aligned link's centre of mass and inertia belong to the solver and cannot be written
        after build, so set ``False`` on an object whose inertial properties a task randomizes per env (a COM
        offset, a per-env inertia); its free joint then sits at the authored body frame, as in MuJoCo, so for a
        single-body free object ``get_pos()`` equals ``get_pos(relative=True)`` (and likewise ``get_quat``). The
        cost: the unaligned free body loses its diagonal mass block (linear and angular motion couple through the
        COM offset), the numerical-stability benefit alignment exists for.
    default_armature : float, optional
        Rotor inertia Genesis fills in at ``scene.build()`` for every DOF of a body moved by a single hinge or slide
        joint whose armature the asset leaves at zero: every URDF joint (URDF has no armature element) and any MJCF
        joint that omits ``armature`` or authors it as ``0``. Ball, free and multi-joint bodies are skipped, and a
        nonzero authored value is kept. 0.0 disables the fill. Defaults to 0.1, usually far too high for
        low-stiffness joints, so set ``default_dofs_armature`` from the source asset instead of inheriting it.

    is_fixed_base: bool
        Whether the entity is fixed in the environment. A URDF, USD or mesh asset is built to match. An MJCF is
        not — Genesis has no ``fixed`` switch for it — so declare what the file says: ``False`` when its base
        body carries a ``<freejoint/>``, ``True`` when it has no joint or any other joint of its own (hinge,
        slide or ball). A mismatch raises at build.
    is_articulated: bool
        Whether the entity is articulated.
    is_actuated: bool
        Whether the entity is actuated.
    default_root_pos: array-like[float, float, float]
        The default position of the loaded entity's root link. Will not take effect if the entity is attached to
        another entity.
    default_root_quat: array-like[float, float, float, float]
        The default orientation of the loaded entity's root link (w, x, y, z). Will not take effect if the entity
        is attached to another entity.
    dofs_name: list[str]
        The names of the DOFs to be treated as controllable DOFs.
    dofs_spec : ClassVar[dict[str, type[ActuatorSpecOptions]]]
        Class-level default actuator specification map for entity DOFs.
    default_dofs_pos: dict[str, float]
        The default positions of the DOFs of the entity.
    default_dofs_vel: dict[str, float]
        The default velocities of the DOFs of the entity.
    default_dofs_kp: dict[str, float]
        The default control position gains of the DOFs of the entity.
    default_dofs_kd: dict[str, float]
        The default control velocity gains of the DOFs of the entity.
    default_dofs_stiffness: dict[str, float]
        The default stiffnesses of the DOFs of the entity. If provided, the value will be used over the stiffness
        in dofs_spec.
    default_dofs_damping: dict[str, float]
        The default dampings of the DOFs of the entity. If provided, the value will be used over the damping in
        dofs_spec.
    default_dofs_armature: dict[str, float]
        The default armatures of the DOFs of the entity. If provided, the value will be used over the armature in
        dofs_spec.
    default_dofs_pos_limits: dict[str, array-like[float, float]]
        The position limits of the DOFs of the entity.
    soft_dofs_pos_limits: dict[str, array-like[float, float]]
        The position limits of the DOFs of the entity that are treated as soft DOFs.
    """

    # NOTE: default up (+z) and front direction (+x) in object coordinate system
    # NOTE: this is used to set the default up and front direction for the entity on loading
    # NOTE: init_quat will be the offset from this default direction
    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    # NOTE: easier access to morph options with default values
    file: str = ""
    registry: str = DEFAULT_ASSET_REGISTRY
    dataset: str = DEFAULT_ASSET_DATASET
    local_dir: str | None = None
    scale: float | tuple[float, float, float] = 1.0
    decimate: bool = True
    decimate_face_num: int = 1000
    decimate_aggressiveness: int = 3
    convexify: bool = True
    # When True, ignore the file's authored mass/inertia and recompute them from the
    # collision geometry * the material density (``rho``). Needed for meshes authored at
    # unit scale whose ``<mass>`` would otherwise be scaled by ``scale**3`` into a
    # near-zero (unstable) value. Default False keeps the asset's own inertia.
    recompute_inertia: bool = False
    decompose_object_error_threshold: float = 0.15
    decompose_robot_error_threshold: float = float("inf")  # Any objects with joint
    coacd_threshold: float = 0.05
    coacd_preprocess_resolution: int = 100
    links_to_keep: list[str] = []  # only used for URDF
    align: bool | None = None
    default_armature: float = 0.1

    # NOTE: no ``forward_vec`` field — the forward direction is exactly ``front``.
    # The runtime entity derives its ``forward_vec`` from ``front`` (see eden/entities/base.py).

    # Whether ``resolve_runtime_file`` writes ``is_fixed_base`` into an MJCF's root joint. Genesis has no
    # ``fixed`` switch for MJCF, so a plain MJCF's base is whatever the file says and the entity raises when
    # the flag disagrees; a family that rewrites its MJCF from the flag (the robosuite / LIBERO objects) sets
    # this, so a caller that pins or frees entities generically (``eden view`` / ``eden arrange``) knows the
    # flag reaches the asset.
    mjcf_follows_is_fixed_base: ClassVar[bool] = False

    # The subdirectory of the asset root (``eden.utils.misc.get_assets_dir``) that an unset ``local_dir`` means:
    # an asset family may name its own (``"ycb"``, ``"libero"``, ...); one that leaves it empty uses the root.
    assets_subdir: ClassVar[str] = ""

    def resolve_local_dir(self) -> str:
        """Where this entity's asset is cached: ``local_dir``, else ``assets_subdir`` under the asset root.

        Resolved on each call, never stored: ``EDEN_ASSETS_DIR`` then moves every default, and a saved
        config's ``local_dir`` stays unset. To fetch a file, call :meth:`fetch_asset`.
        """
        return resolve_local_dir(self.local_dir, self.assets_subdir)

    def fetch_asset(self, file: str | None = None) -> str:
        """The local path of ``file`` (default: ``self.file``) from this entity's Hub dataset, downloaded if missing.

        Fetches with this entity's ``registry``, ``dataset`` and :meth:`resolve_local_dir`. Call this rather
        than ``get_asset_path(..., local_dir=options.local_dir)``: an unset ``local_dir`` passed on raw would
        fetch into the asset root instead of the family's ``assets_subdir``, and nothing would fail.
        """
        return get_asset_path(
            self.file if file is None else file,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )

    @classmethod
    def fetch_family_asset(cls, file: str) -> str:
        """The local path of ``file`` from this family's default Hub dataset, downloaded if missing.

        For a file the family's dataset ships that no entity is built from (a task catalogue's BDDL file, a
        background image), so there is no options instance to call :meth:`fetch_asset` on: fetches with the
        class's default ``registry``/``dataset``/``local_dir``, the last resolved like :meth:`resolve_local_dir`.
        Nothing is validated as an entity, so the file may be of any type.

        Raises
        ------
        ValueError
            When the family declares no Hub ``registry``/``dataset`` default, so there is no dataset to fetch from.
        """
        fields = cls.model_fields
        registry, dataset = fields["registry"].default, fields["dataset"].default
        if registry is None or dataset is None:
            raise ValueError(
                f"{cls.__name__}.fetch_family_asset({file!r}): the family has no Hub dataset to fetch from "
                f"(registry={registry!r}, dataset={dataset!r}). Give {cls.__name__} registry and dataset defaults, "
                f"or open a local file directly instead of fetching it."
            )
        return get_asset_path(
            file,
            registry=registry,
            dataset=dataset,
            local_dir=resolve_local_dir(fields["local_dir"].default, cls.assets_subdir),
        )

    def resolve_runtime_file(self) -> str:
        """Return the asset descriptor consumed while building the runtime entity.

        Subclasses may derive a loadable descriptor from the serialized source asset.
        The returned path is runtime state and must not be written back to ``file``.
        """
        return self.file

    def model_post_init(self, context):
        super().model_post_init(context)

        # Validate up and front are not parallel
        up_norm = np.linalg.norm(self.up)
        front_norm = np.linalg.norm(self.front)
        if up_norm > 0 and front_norm > 0:
            up_unit = np.array(self.up) / up_norm
            front_unit = np.array(self.front) / front_norm
            dot = np.abs(np.dot(up_unit, front_unit))
            if dot > 0.99:  # Nearly parallel
                raise ValueError("Up and front directions must not be parallel")

        if (
            isinstance(self.scale, tuple)
            and len(set(self.scale)) > 1
            and not self.file.lower().endswith(MESH_FILE_SUFFIXES)
        ):
            raise ValueError(
                f"scale={self.scale} scales each axis differently, which only a mesh file {MESH_FILE_SUFFIXES} "
                f"supports; file '{self.file}' loads as a URDF / MJCF / USD asset, scaled uniformly. Load the "
                f"asset's mesh as file=, or pass one number."
            )

        if self.file.lower().endswith(TET_MESH_FILE_SUFFIXES) and not isinstance(
            self.material, MochiElasticMaterialOptions
        ):
            # Only the Mochi solid entity reads a volume mesh, in place of tetgen; every other loader expects a surface.
            got = "no material, so the default rigid one" if self.material is None else type(self.material).__name__
            raise ValueError(
                f"file '{self.file}' is a tetrahedral mesh, which only a MochiElasticMaterialOptions entity can load "
                f"(got {got}). Pass material=MochiElasticMaterialOptions(...), or "
                f"export a surface mesh ('.obj', '.stl', ...) for another material."
            )

        if self.material is not None:
            _validate_no_dofs_for_non_rigid_material(self)
            _validate_no_support_for_non_rigid_material(self)
            _validate_mochi_rigid_fields(self)
            _validate_mochi_visualize_contact(self)


class PrimitiveOptions(BaseEntityOptions):
    """
    Base options for primitive shapes.

    Parameters
    ----------
    default_root_pos : array-like[float, float, float]
        Default root position.
    default_root_quat : array-like[float, float, float, float]
        Default root orientation as quaternion in (w, x, y, z).
    fixed: bool
        Whether the entity is fixed.
    is_articulated : bool
        Whether the entity has articulation.
    is_actuated : bool
        Whether the entity is actuated.
    material : MaterialLike | None
        Material used for the entity.
    surface : SurfaceOptions | Surface | None
        Surface appearance used for the entity. Prefer an Eden :class:`SurfaceOptions`
        in configs constructed before ``gs.init()``.
    color : array-like[float] | None
        Convenience shortcut to color the entity (RGB or RGBA) without a full surface.
    visualize_contact : bool
        Whether to visualize contacts involving this entity.
    vis_mode : VisMode
        Visualization mode for the entity.
    collision : bool
        Whether this entity participates in collision checking.
    batch_fixed_verts : bool
        Whether fixed-geometry vertices are batched for per-env transforms.
    is_support_enabled : bool
        Whether support-point computation is enabled.
    """

    fixed: bool = False
    # Optional pre-built genesis morph override (advanced/deformable escape hatch), consumed only
    # by :meth:`_collect_morph_kwargs` below — hence declared here, not on ``BaseEntityOptions``,
    # so file-based entities (``EntityOptions``/``TerrainOptions``) reject it loudly instead of
    # silently ignoring it. A real field so extra="forbid" still accepts ``morph=``; the plain-mode
    # validator below replaces pydantic's schema validation, keeping the raw dict/Morph exactly as
    # passed (no dict->Morph coercion) while still rejecting wrong types at construction time.
    morph: Morph | dict | None = None

    @field_validator("morph", mode="plain")
    @classmethod
    def _keep_morph_raw(cls, value):
        if value is None or isinstance(value, (dict, Morph)):
            return value
        # ValueError (not TypeError): pydantic only wraps ValueError into a ValidationError.
        raise ValueError(f"morph override must be a genesis Morph or a kwargs dict, got {type(value).__name__}")

    def _collect_morph_kwargs(self, **kwargs) -> dict[str, object]:
        """Collect kwargs forwarded to Genesis morph constructors."""
        kwargs.update(
            pos=self.default_root_pos,
            quat=self.default_root_quat,
            fixed=self.is_fixed_base,
            batch_fixed_verts=self.batch_fixed_verts,
            collision=self.collision,
            contype=self.contype,
            conaffinity=self.conaffinity,
            visualization=self.visualization,
        )
        if self.morph is not None:
            if isinstance(self.morph, dict):
                kwargs.update(**self.morph)
            elif isinstance(self.morph, Morph):
                kwargs.update(**self.morph.model_dump())
            else:
                raise TypeError(f"morph override must be a genesis Morph or a kwargs dict, got {type(self.morph)}")
        return kwargs

    def model_post_init(self, context):
        super().model_post_init(context)
        if type(self) is PrimitiveOptions:
            raise ValueError(
                "PrimitiveOptions is an abstract base. Use one of: "
                "PlaneOptions, BoxOptions, SphereOptions, CylinderOptions."
            )
        # Unify: either ``fixed`` or ``is_fixed_base`` opts the entity in.
        # ``is_fixed_base`` is the canonical Eden field; ``fixed`` is the
        # Genesis-style spelling (``gs.morphs.Box(fixed=True)``).
        if self.fixed or self.is_fixed_base:
            self.fixed = True
            self.is_fixed_base = True

        # Both checks must precede the clear: it blanks exactly what they inspect.
        if self.material is not None:
            _validate_no_dofs_for_non_rigid_material(self)
            _validate_no_support_for_non_rigid_material(self)
            _validate_mochi_rigid_fields(self)
            _validate_mochi_visualize_contact(self)
        _reject_dofs_on_never_articulated(self)

        self._set_dofs_unactuated()


class PlaneOptions(PrimitiveOptions):
    fixed: bool = True
    normal: UnitVec3FType = (0.0, 0.0, 1.0)
    plane_size: PositiveVec2FType = (1e3, 1e3)
    tile_size: PositiveVec2FType = (1.0, 1.0)

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Plane(
            **self._collect_morph_kwargs(
                normal=self.normal,
                plane_size=self.plane_size,
                tile_size=self.tile_size,
            ),
        )


class BoxOptions(PrimitiveOptions):
    size: Vec3FType = (0.1, 0.1, 0.1)

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Box(
            **self._collect_morph_kwargs(
                size=self.size,
            ),
        )


class SphereOptions(PrimitiveOptions):
    radius: float = 0.1

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Sphere(
            **self._collect_morph_kwargs(
                radius=self.radius,
            ),
        )


class CylinderOptions(PrimitiveOptions):
    radius: float = 0.1
    height: float = 0.1

    def get_morph(self) -> gs.morphs.Primitive:
        return gs.morphs.Cylinder(
            **self._collect_morph_kwargs(
                radius=self.radius,
                height=self.height,
            ),
        )


class RodOptions(PrimitiveOptions):
    """
    A discrete elastic rod on a polyline (``eden.engine.solvers.mochi.morphs.Rod``), for the Mochi solver only.

    The morph is the centerline; the material must be a
    :class:`~eden.options.materials.MochiRodMaterialOptions`, which derives the rod's stiffness from
    ``radius`` unless given explicitly. Clamp a rod after build by fixing its first two nodes
    (``env.entities[name].set_vertices_fixed([0, 1])``). ``default_root_pos`` / ``default_root_quat``
    place the whole polyline; ``is_fixed_base`` has no meaning for a rod and is rejected.

    Parameters
    ----------
    points : array-like[n_nodes, 3]
        Centerline nodes (at least two, consecutive nodes distinct), in the rod's own frame.
    radius : float
        Circular cross-section radius (m). Default is 0.01.
    n_cross_section_segments : int
        Tube tessellation used for rendering only. Default is 12.
    is_closed_loop : bool
        Connect the last node back to the first. Default is False.
    """

    points: list[tuple[float, float, float]]
    radius: float = 0.01
    n_cross_section_segments: int = 12
    is_closed_loop: bool = False

    @field_validator("points", mode="before")
    @classmethod
    def _points_as_list(cls, value):
        """Accept an ``(n, 3)`` array-like and store plain tuples so the config serializes."""
        array = np.asarray(value, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 2:
            raise ValueError(f"RodOptions.points must be an (n_nodes >= 2, 3) array, got shape {array.shape}")
        return [tuple(float(v) for v in row) for row in array]

    def get_morph(self) -> Rod:
        # ``eden.engine.solvers.mochi.morphs.Rod`` inherits only ``Morph`` (no ``fixed`` / ``contype`` /
        # ``conaffinity`` / ``batch_fixed_verts``), and genesis options forbid extras, so the generic
        # ``_collect_morph_kwargs`` would be rejected — assemble the accepted subset by hand.
        kwargs: dict[str, object] = {
            "points": np.asarray(self.points, dtype=float),
            "radius": self.radius,
            "n_cross_section_segments": self.n_cross_section_segments,
            "is_closed_loop": self.is_closed_loop,
            "pos": self.default_root_pos,
            "quat": self.default_root_quat,
            "collision": self.collision,
            "visualization": self.visualization,
        }
        if isinstance(self.morph, dict):
            kwargs.update(**self.morph)
        elif isinstance(self.morph, Morph):
            kwargs.update(**self.morph.model_dump())
        return Rod(**kwargs)

    def model_post_init(self, context):
        super().model_post_init(context)
        if not isinstance(self.material, MochiRodMaterialOptions):
            # ValueError (not TypeError): pydantic only wraps ValueError into a ValidationError.
            raise ValueError(  # noqa: TRY004
                f"RodOptions requires material=MochiRodMaterialOptions(...), got "
                f"{type(self.material).__name__ if self.material is not None else None}. A rod is simulated by the "
                f"MochiSolver only."
            )
        if self.is_fixed_base:
            raise ValueError("RodOptions has no base to fix; clamp nodes after build with `set_vertices_fixed`.")


class TerrainTermOptions(DynamicConfigurableOptions):
    """TerrainGenerator options for Eden, configure by `TerrainGenerator.configure()`."""


class TerrainOptions(BaseEntityOptions):
    """
    Terrain entity options for Eden, wrapping Genesis' Terrain related options.

    Note that all dofs/articulation-related fields are set to empty/False and is_fixed_base is always set to True.

    Parameters
    ----------
    horizontal_scale : float
        The size of each cell in the subterrain in meters. Defaults to 0.25.
    vertical_scale : float
        The height of each step in the subterrain in meters. Defaults to 0.005.
    subterrain_size : tuple of float
        The size of each subterrain in meters. Defaults to (32.0, 32.0), which holds every
        built-in parkour course at its shipped defaults (they run out to ~29 m). On a shorter
        strip the parkour generators pull the final goal back into the walkable area, and
        raise once the course cannot fit at all (see
        ``eden/utils/terrains/parkour_terrains.py``); tuned courses fit far shorter lanes —
        ``eden/tasks/extreme_parkour`` runs 16 x 4 m.
    uv_scale : float or tuple of float or None
        How many times the surface texture repeats across the whole terrain, as
        ``(x, y)``; a scalar applies to both axes. ``None`` (the default) means auto:
        one repeat per axis for a custom surface, and for ``use_checker_surface`` the
        count that renders 1 m checker squares. Give a tuple when the terrain is not
        square and the texture must not stretch with it — a single figure spreads one
        repeat over each axis's full extent, so a 160 x 40 m terrain would render
        squares 4x wider than they are deep.
    randomize : bool
        Whether to randomize the subterrains that involve randomness. Defaults to False.
    terrain_generators : list[list[TerrainTermOptions]]
        Terrain generator configuration grid.
    use_checker_surface : bool
        Whether to render checkerboard surface instead of a custom surface.
    material : MaterialLike | None
        Material used for terrain.
    surface : SurfaceOptions | Surface | None
        Surface appearance used for terrain.
    visualize_contact : bool
        Whether to visualize contacts involving terrain.
    vis_mode : VisMode
        Visualization mode for terrain.
    is_support_enabled : bool
        Whether support-point computation is enabled.
    default_root_pos : array-like[float, float, float]
        Default root position.
    default_root_quat : array-like[float, float, float, float]
        Default root orientation as quaternion in (w, x, y, z).
    """

    horizontal_scale: float = 0.25  # meter size of each cell in the subterrain
    vertical_scale: float = 0.005  # meter height of each step in the subterrain
    subterrain_size: Vec2FType = (32.0, 32.0)  # meter (holds every built-in parkour course, ~29 m)
    uv_scale: float | Vec2FType | None = None  # texture repeats across (x, y); None = auto
    randomize: bool = False
    terrain_generators: list[list[TerrainTermOptions]] = [
        [
            FlatTerrain.configure(),
        ],
    ]
    use_checker_surface: bool = False

    def model_post_init(self, context):
        super().model_post_init(context)

        self.is_fixed_base = True

        # Both checks must precede the clear: it blanks exactly what they inspect.
        if isinstance(self.material, MochiMaterialOptions):
            # ValueError (not TypeError): pydantic only wraps ValueError into a ValidationError.
            raise ValueError(  # noqa: TRY004
                f"TerrainOptions cannot carry {type(self.material).__name__}: the MochiSolver does not accept "
                f"terrain morphs. Build the ground from a fixed BoxOptions / PlaneOptions instead."
            )
        if self.material is not None:
            _validate_no_dofs_for_non_rigid_material(self)
            # ``_entity_class_for_options`` returns ``Terrain`` for any ``TerrainOptions``
            # regardless of material, so this guard is the only thing that catches a non-rigid
            # material carrying support-* fields that Terrain could never read.
            _validate_no_support_for_non_rigid_material(self)
        _reject_dofs_on_never_articulated(self)

        self._set_dofs_unactuated()

        # Validation
        if self.use_checker_surface and self.surface is not None:
            raise ValueError("Cannot use checker surface with a custom surface.")
        if self.uv_repeats is not None and not all(r > 0.0 for r in self.uv_repeats):
            raise ValueError(
                f"TerrainOptions.uv_scale must be positive; got {self.uv_scale}. It counts texture "
                "repeats across the terrain, so zero or negative collapses or mirrors the mapping."
            )

    @property
    def uv_repeats(self) -> tuple[float, float] | None:
        """``uv_scale`` as an explicit (x, y) pair, or ``None`` when it is left on auto.

        The scalar and pair spellings differ only here; everything downstream reads the pair.
        """
        if self.uv_scale is None:
            return None
        if isinstance(self.uv_scale, (tuple, list)):
            return float(self.uv_scale[0]), float(self.uv_scale[1])
        return float(self.uv_scale), float(self.uv_scale)


class GroupedEntityOptions(BaseEntityOptions):
    """Grouped entity used for heterogeneous batching.

    Supports both non-articulated entities (Mesh, Primitive) and articulated entities (URDF)
    that share the same joint structure (same number of links, joints, and DOFs).

    Each variant contributes its morph (file or shape, mesh processing, its own ``default_root_pos`` /
    ``default_root_quat``), and may hold its own value for some of the group's fields. A variant holds a
    value when it differs from the field's default, whether passed in or its catalog class's default.

    - **Per variant** (``variant_values()``, ``variant_materials``): the geom fields
      (``default_geoms_friction*``, ``default_geoms_sol_params``), the per-DOF ones (``default_dofs_*`` and
      ``soft_dofs_pos_limits``) and the material's ``rho`` / ``friction*``. Each env takes its variant's own
      value, else the group's. Variants that differ in a simulated DOF field (gains, stiffness, damping,
      armature, frictionloss, force limits) need ``EnvOptions(batch_dofs_info=True)``. A per-variant ``rho``
      rescales each env's mass after the build, so it needs a mesh or primitive variant, whose mass Genesis
      computes from its density.
    - **Shared** (``shared_dof_fields()``, ``shared_dofs_spec``): ``dofs_name`` and
      ``default_dofs_spring_ref``, the group's own or, when it sets none, that of the first variant that sets
      one; a variant's own different value is refused. So is a variant class's ``dofs_spec`` that differs
      from another's.
    - **Group only**: every other field (``surface``, ``color``, ``is_fixed_base``, the support and collision
      mask fields, the other material fields): a variant's own different value is refused.

    A variant overrides the group's own value for a per-variant field only with a value other than its class's
    default: one equal to the class default is read as inherited and refused, as the group's value would
    silently skip it. The group's value is also refused when every variant overrides it, as it would apply to
    none of them. ``check_variants`` runs these checks; the entity runs it again before the build.

    Parameters
    ----------
    grouped_entities : list[EntityOptions | PrimitiveOptions]
        A list of entity configurations to be grouped.
    is_fixed_base : bool
        Whether grouped entities should be treated as fixed base.
    dofs_name : list[str]
        Controllable DOF names shared by the group.
    dofs_spec : ClassVar[dict[str, type[ActuatorSpecOptions]]]
        Class-level default actuator specification map for grouped DOFs.
    default_dofs_pos : dict[str, float]
        Default grouped DOF positions.
    default_dofs_vel : dict[str, float]
        Default grouped DOF velocities.
    default_dofs_kp : dict[str, float]
        Default grouped DOF position gains.
    default_dofs_kd : dict[str, float]
        Default grouped DOF velocity gains.
    default_dofs_stiffness : dict[str, float]
        Default grouped DOF stiffness values.
    default_dofs_damping : dict[str, float]
        Default grouped DOF damping values.
    default_dofs_armature : dict[str, float]
        Default grouped DOF armature values.
    default_dofs_pos_limits : dict[str, array-like[float, float]]
        Hard grouped DOF position limits.
    soft_dofs_pos_limits : dict[str, array-like[float, float]]
        Soft grouped DOF position limits.
    default_dofs_force_limits : dict[str, float]
        Default grouped DOF force limits.
    """

    # TODO: add TerrainOptions support
    grouped_entities: list[EntityOptions | PrimitiveOptions] = Field(default_factory=list)

    def model_post_init(self, context):
        super().model_post_init(context)

        # Validation for grouped entities
        for entity in self.grouped_entities:
            if not isinstance(entity, (EntityOptions, PrimitiveOptions)):
                raise ValueError(
                    f"Invalid entity type: {type(entity)}, expected one of: "
                    f"EntityOptions, PrimitiveOptions, got {type(entity)}"
                )

        # Detect articulated/actuated flags from grouped entities.
        # All grouped entities must agree on is_articulated and is_actuated.
        any_articulated = any(getattr(e, "is_articulated", False) for e in self.grouped_entities)
        any_actuated = any(getattr(e, "is_actuated", False) for e in self.grouped_entities)

        if any_articulated:
            # Validate all entities are articulated (must share same joint structure)
            for e in self.grouped_entities:
                if not getattr(e, "is_articulated", False):
                    raise ValueError(
                        "All grouped entities must have the same is_articulated flag. "
                        "Cannot mix articulated and non-articulated entities."
                    )

        if any_actuated:
            # Validate all entities are actuated
            for e in self.grouped_entities:
                if not getattr(e, "is_actuated", False):
                    raise ValueError(
                        "All grouped entities must have the same is_actuated flag. "
                        "Cannot mix actuated and non-actuated entities."
                    )

        self.is_articulated = any_articulated
        self.is_actuated = any_actuated

        # The MochiSolver refuses heterogeneous (list) morphs outright. Checked before the variants, whose
        # material checks would otherwise point at moving the Mochi material onto the group.
        if any(
            isinstance(m, MochiMaterialOptions) for m in (self.material, *(e.material for e in self.grouped_entities))
        ):
            raise ValueError(
                "GroupedEntityOptions cannot carry a Mochi material: the MochiSolver has no heterogeneous morphs."
            )

        self.check_variants()

        # A group with a non-rigid material dispatches to a particle/FEM entity
        # (``_entity_class_for_options``), which can never be a support target — guard the
        # support-* fields here as the other entity-options classes do, so they are not
        # silently discarded at build.
        if self.material is not None:
            _validate_no_support_for_non_rigid_material(self)

    def check_variants(self) -> None:
        """Refuse what the variants hold that the entity cannot apply as configured (raises ``ValueError``).

        Run at construction, and again by the entity before the scene is built: a TaskMod or any other code
        that assigns a field of the group or of a variant after construction never re-runs
        ``model_post_init``, so its edit would otherwise go unchecked.
        """
        self._refuse_variant_group_only_values()
        self._refuse_variant_shared_dof_values()
        for i, variant in enumerate(self.grouped_entities):
            if variant.material is not None:
                self._refuse_variant_material(i, variant.material)
        self._refuse_implicit_variant_overrides()
        self._refuse_unused_group_values()

    def variant_values(self) -> dict[str, list]:
        """Each variant's value for every per-variant field: its own, else the group's."""
        return {
            name: [getattr(v if _has_own_value(v, name) else self, name) for v in self.grouped_entities]
            for name in _PER_VARIANT_FIELD_NAMES
        }

    @property
    def variant_materials(self) -> list[MaterialLike]:
        """Each variant's material: the group's, with the variant's own ``rho`` and friction coefficients."""
        group = self._resolved_material
        materials = []
        for variant in self.grouped_entities:
            own = _own_material_values(variant.material, _PER_VARIANT_MATERIAL_FIELD_NAMES)
            materials.append(group.model_copy(update=own) if own else group)
        return materials

    def shared_dof_fields(self) -> dict[str, object]:
        """The DOF fields every variant shares (``dofs_name``, ``default_dofs_spring_ref``).

        Each is the group's own value, else that of the first variant that sets one. Resolved on read rather
        than copied onto the group: once a config is dumped a copy cannot be told from a value the group set, so
        an override that swaps the bank would keep the old one's.
        """
        return {name: getattr(self._shared_dof_source(name), name) for name in _SHARED_DOF_FIELD_NAMES}

    @property
    def shared_dofs_spec(self) -> dict[str, type[ActuatorSpecOptions]]:
        """The actuator specs every variant shares: a variant's own ``dofs_spec``, else the group's."""
        specs = self._variant_dofs_specs()
        return specs[0] if specs else self.dofs_spec

    @property
    def _resolved_material(self) -> MaterialLike:
        """The material the group builds with: its own, or ``RigidMaterialOptions()``, as the entity resolves
        ``None``.
        """
        return self.material if self.material is not None else RigidMaterialOptions()

    def _shared_dof_source(self, name: str) -> BaseEntityOptions:
        """Where shared DOF field ``name`` comes from: the group if it sets one, else the first variant that does."""
        if _has_own_value(self, name):
            return self
        return next((v for v in self.grouped_entities if _has_own_value(v, name)), self)

    def _variant_dofs_specs(self) -> list[dict[str, type[ActuatorSpecOptions]]]:
        """Each variant's actuator specs: its own ``dofs_spec``, else the group's.

        Read off the instance, not the class: a non-actuated variant shadows its class's specs with ``{}``
        (``_set_dofs_unactuated``), as it would as an entity of its own.
        """
        return [v.dofs_spec or self.dofs_spec for v in self.grouped_entities]

    def _refuse_variant_group_only_values(self) -> None:
        """Refuse a variant's own value for a field only the group applies (``_GROUP_ONLY_FIELD_NAMES``)."""
        for i, variant in enumerate(self.grouped_entities):
            for name in _GROUP_ONLY_FIELD_NAMES:
                mine, group = getattr(variant, name), getattr(self, name)
                if _has_own_value(variant, name) and mine != group:
                    raise ValueError(
                        f"GroupedEntityOptions {_where(i, variant)} has {name}={mine!r} (passed in, or its class's "
                        f"default), but a heterogeneous entity applies {name} once for all its variants, from the "
                        f"group ({name}={group!r}): {_GROUP_ONLY_REASONS[name]}. Fix: set {name}={mine!r} on the "
                        f"GroupedEntityOptions itself if every variant shares it, or give the variant the group's "
                        f"value (remove it, or pass {name}={group!r} to override its class's default); an object "
                        f"that needs its own value has to be an entity of its own."
                    )

    def _refuse_variant_shared_dof_values(self) -> None:
        """Refuse a variant whose own shared DOF field (or actuator specs) differs from the one every variant takes."""
        for name in _SHARED_DOF_FIELD_NAMES:
            source = self._shared_dof_source(name)
            applied = getattr(source, name)
            origin = (
                "the group's own"
                if source is self
                else f"taken from grouped_entities[{self.grouped_entities.index(source)}], the first variant that sets "
                f"one, as the group sets none"
            )
            for i, variant in enumerate(self.grouped_entities):
                mine = getattr(variant, name)
                if _has_own_value(variant, name) and mine != applied:
                    raise ValueError(
                        f"GroupedEntityOptions {_where(i, variant)} has {name}={mine!r}, but every variant of a "
                        f"heterogeneous entity shares one {name} ({name}={applied!r}, {origin}). Fix: give every "
                        f"variant the same {name}, or set it once on the GroupedEntityOptions and remove it from the "
                        f"variants."
                    )
        specs = self._variant_dofs_specs()
        for i, spec in enumerate(specs):
            if spec != specs[0]:
                raise ValueError(
                    f"GroupedEntityOptions {_where(i, self.grouped_entities[i])} has dofs_spec "
                    f"{sorted(spec)} but grouped_entities[0] ({type(self.grouped_entities[0]).__name__}) has "
                    f"{sorted(specs[0])}: a heterogeneous entity applies one set of actuator specs to every "
                    f"variant. Fix: group variants whose option classes share a dofs_spec, or set the per-variant "
                    f"gains through default_dofs_kp / default_dofs_kd instead."
                )

    def _refuse_variant_material(self, i: int, mine: MaterialLike) -> None:
        """Refuse a variant material that differs from the group's in more than its per-variant fields."""
        group = self._resolved_material
        where = _where(i, self.grouped_entities[i])
        if type(mine) is not type(group):
            raise ValueError(
                f"GroupedEntityOptions {where} has a {type(mine).__name__} material, but the group builds every "
                f"variant with its own {type(group).__name__}: Genesis takes one material per entity. Fix: give the "
                f"variant the group's material type (or none), or set material={type(mine).__name__}(...) on the "
                f"GroupedEntityOptions if every variant shares it."
            )
        shared = [name for name in type(mine).model_fields if name not in _PER_VARIANT_MATERIAL_FIELD_NAMES]
        differing = {
            name: value for name, value in _own_material_values(mine, shared).items() if value != getattr(group, name)
        }
        if differing:
            own = _own_material_values(mine, _PER_VARIANT_MATERIAL_FIELD_NAMES)
            keep = (
                f"keep only {sorted(own)} on the variant's material" if own else "remove the material from the variant"
            )
            raise ValueError(
                f"GroupedEntityOptions {where} has a material (passed in, or its class's default) whose "
                f"{differing} differ from the group's { ({name: getattr(group, name) for name in differing}) }, but "
                f"Genesis builds every variant with the group's one material, so only "
                f"{list(_PER_VARIANT_MATERIAL_FIELD_NAMES)} may differ per variant. Fix: put the shared fields on "
                f"GroupedEntityOptions(material=...) and {keep}."
            )

    def _refuse_implicit_variant_overrides(self) -> None:
        """Refuse a variant whose class default, rather than a value of its own, overrides the group's own value.

        A variant's own value wins over the group's for a per-variant field, which is how one object gets an
        exception. A value the variant only inherits from its catalog class (a robot class's gains, an object
        class's material) would win just the same, silently: the group's value then reaches only some of its
        variants. A variant value equal to its class's default is read as inherited.

        Compared against the class default rather than ``model_fields_set``: a dump passes every field back in,
        so an override (``with_overrides_from_dict``) or a reload would read every inherited value as passed in.
        The class survives the dump (``_option_class_``), so this reads the same however the config was made.
        The price is that a variant cannot keep its class's value by passing it again; the group's value then
        goes on the other variants instead, which the message says.
        """
        group_material = self._resolved_material
        group_material_own = _own_material_values(self.material, _PER_VARIANT_MATERIAL_FIELD_NAMES)
        for i, variant in enumerate(self.grouped_entities):
            fields = type(variant).model_fields
            # name -> (the variant's value, the group's value, the fix that takes the group's)
            conflicts = {
                name: (mine, group, f"{name}={group!r}")
                for name in _PER_VARIANT_FIELD_NAMES
                if _has_own_value(self, name)
                and _has_own_value(variant, name)
                and (mine := getattr(variant, name)) != (group := getattr(self, name))
                and mine == fields[name].get_default(call_default_factory=True)
            }
            material = variant.material
            if material is not None and material == fields["material"].get_default(call_default_factory=True):
                variant_own = _own_material_values(material, _PER_VARIANT_MATERIAL_FIELD_NAMES)
                for name, group in group_material_own.items():
                    if name in variant_own and (mine := variant_own[name]) != group:
                        # The class's material written out with the group's value, so its other fields survive.
                        own = _own_material_values(material, type(material).model_fields)
                        conflicts[f"material.{name}"] = (
                            mine,
                            getattr(group_material, name),
                            f"material={_material_repr(material, {**own, name: group})}",
                        )
            if not conflicts:
                continue
            name, (mine, group, take_group) = next(iter(conflicts.items()))
            raise ValueError(
                f"GroupedEntityOptions sets {name}={group!r}, but {_where(i, variant)} overrides it with "
                f"{name}={mine!r}, its class's default, so the group's value would silently not reach that variant "
                f"(every such field on it: {sorted(conflicts)}). Fix: pass {take_group} to the variant to take the "
                f"group's value. To keep its class's value instead, move {name.split('.')[0]} off the group and onto "
                f"each of the other variants: a variant value equal to its class's default is read as inherited, so "
                f"passing it again does not mark it as intended."
            )

    def _refuse_unused_group_values(self) -> None:
        """Refuse a group value for a per-variant field that every variant overrides, which would apply to none."""
        if not self.grouped_entities:
            return
        group_material = self._resolved_material
        unused = {}
        for name in _PER_VARIANT_FIELD_NAMES:
            value = getattr(self, name)
            if _has_own_value(self, name) and all(
                _has_own_value(v, name) and getattr(v, name) != value for v in self.grouped_entities
            ):
                unused[name] = value
        group_own = _own_material_values(self.material, _PER_VARIANT_MATERIAL_FIELD_NAMES)
        for name, value in group_own.items():
            if all(
                name in (own := _own_material_values(v.material, _PER_VARIANT_MATERIAL_FIELD_NAMES))
                and own[name] != value
                for v in self.grouped_entities
            ):
                unused[f"material.{name}"] = getattr(group_material, name)
        if unused:
            raise ValueError(
                f"GroupedEntityOptions sets {unused}, but every one of its {len(self.grouped_entities)} variants "
                f"sets its own value for {'that field' if len(unused) == 1 else 'those fields'}, so the group's "
                f"applies to none of them. Fix: remove it from the group, or remove it from the variants that "
                f"should take it."
            )


# Why ``GroupedEntityOptions`` refuses a variant's own value for each group-only field.
_GROUP_ONLY_REASONS = {
    **dict.fromkeys(
        ("surface", "color", "visualize_contact", "vis_mode"),
        "Genesis takes one surface and one visualization mode per entity",
    ),
    **dict.fromkeys(
        ("collision_link_patterns", "collision_friction"), "Eden applies the collision mask to the whole entity"
    ),
    "is_fixed_base": "the base joint is part of the structure every variant shares",
    **dict.fromkeys(
        (
            "is_support_enabled",
            "support_sample_method",
            "support_minimum_clearance",
            "support_shrink",
            "support_num_sample_points",
            "support_grid_size",
            "support_links_name",
        ),
        "Eden extracts one support surface for the whole entity",
    ),
    "default_equalities_sol_params": "every variant shares the equality constraints of the first one's asset",
}


def _where(i: int, variant: BaseEntityOptions) -> str:
    """How an error names variant ``i`` of a ``GroupedEntityOptions``."""
    return f"grouped_entities[{i}] ({type(variant).__name__})"


def _material_repr(material: MaterialLike, fields: dict[str, object]) -> str:
    """``material``'s class called with ``fields``, as a fix an error message can hand over to type."""
    return f"{type(material).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


def _own_material_values(material: MaterialLike | None, names) -> dict[str, object]:
    """The fields among ``names`` that ``material`` holds a value of its own for: one other than its class default."""
    if material is None:
        return {}
    fields = type(material).model_fields
    return {
        name: getattr(material, name)
        for name in names
        if name in fields and getattr(material, name) != fields[name].get_default(call_default_factory=True)
    }


class RobotOptions(EntityOptions):
    """Robot entity options for Eden."""

    is_actuated: bool = True
    is_articulated: bool = True
    actuated_dofs_name: ClassVar[list[str]] = []
    #: Whether some class from this one up declared ``actuated_dofs_name`` (rather than having it filled with
    #: every DOF).
    _actuated_dofs_name_declared: ClassVar[bool] = False
    ee_links_name: ClassVar[list[str]] = []
    #: Links this robot stands on — a biped's ankle-roll links, a quadruped's four feet — in
    #: whatever order reads naturally for the robot. Which links those are is a property of the
    #: *asset*, so it belongs here rather than being restated by each consumer: the deploy-time
    #: base-linear-velocity estimator hangs its leg odometry and contact detection on it
    #: (``DeploymentBase.feet_links`` overrides it for a one-off).
    #:
    #: Empty means "this robot has no feet in that sense", which covers more than arms: a
    #: wheeled base whose ground contact rolls rather than plants (``Go2W``) breaks the leg
    #: odometry's stationary-contact assumption, so it declares none and the estimator refuses
    #: rather than returning a quietly biased velocity.
    #:
    #: A tuple, not a list, because robots share one module-level constant between this and
    #: their ``links_to_keep``: pydantic deep-copies a *field* default per instance, but a
    #: ClassVar is the shared object itself, so a mutable one would let an in-place edit on
    #: one robot reach its siblings.
    feet_links_name: ClassVar[tuple[str, ...]] = ()

    #: Links a simulated bring-up gantry hangs the robot from (``eden deploy --mujoco --gantry``),
    #: declared by robots that have a sensible rig point. For a humanoid this is the pair of
    #: shoulders: two ropes resist roll as a couple, which one rope over the centre cannot.
    #: Prefer the most proximal link of each shoulder — a joint's frame origin does not translate
    #: when that joint rotates, so a mount at the origin stays put on the torso while the arm
    #: swings. Empty means no declared rig point; the gantry then hangs from the base body.
    gantry_links_name: ClassVar[tuple[str, ...]] = ()

    #: Two-finger-gripper facts, declared by gripper assets (``FrankaHand``, ``Robotiq2f85``) and read
    #: through :class:`~eden.options.robots.composite.ManipulatorOptions`, and by arms whose asset
    #: carries its own fingers (``WidowX250``):
    #: ``tcp_offset`` is the tool point — the midpoint of the fingertip pads with the gripper open —
    #: in the gripper's base-link frame, or for an arm with its own fingers in the frame of
    #: ``ee_links_name[0]`` (m); ``stroke`` is the pad-to-pad opening at ``open_dofs_pos`` (m), zero at
    #: ``close_dofs_pos``. ``None`` means the asset declares neither; a rig that needs them then
    #: raises naming the robot instead of guessing.
    tcp_offset: ClassVar[tuple[float, float, float] | None] = None
    stroke: ClassVar[float | None] = None

    #: Coarsest ``sim_dt / sim_substeps`` (s) this asset's constraints stay stiff at, declared by
    #: assets whose MJCF encodes a closed loop — Genesis clamps every ``solref`` ``timeconst`` to
    #: ``2 * substep_dt``, so a four-bar gripper stepped above its limit goes soft and slides off
    #: what it holds while the scene still builds and steps. ``None`` means no limit declared.
    max_substep_dt: ClassVar[float | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Default ``actuated_dofs_name`` to *all* DOFs when a robot subclass doesn't set it.

        Runs once per ``RobotOptions`` subclass at class-definition time. It only fills the
        default when neither the subclass nor any ancestor declared ``actuated_dofs_name`` — an
        explicit subset is never overwritten, including one a shared base declares for its
        left/right leaves (``_InspireRH56`` for ``InspireRH56_L`` / ``_R``). The default mirrors
        ``dofs_name``'s declared default, i.e. every DOF is actuated unless the robot says
        otherwise; a value this hook filled is not a declaration, so a subclass that changes
        ``dofs_name`` gets its own.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if "actuated_dofs_name" in cls.__dict__:
            cls._actuated_dofs_name_declared = True
            return
        if cls._actuated_dofs_name_declared:
            return  # inherited from the ancestor that declared it
        dofs_field = cls.model_fields.get("dofs_name")
        if dofs_field is not None and isinstance(dofs_field.default, list):
            cls.actuated_dofs_name = list(dofs_field.default)


@functools.lru_cache(maxsize=1)
def _ghost_defaults() -> dict:
    return dict(
        material=KinematicMaterialOptions(),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ColorTexture(color=(0.4, 0.7, 1.0)),
            opacity_texture=gs.textures.ColorTexture(color=(0.5,)),
        ),
    )


class _LazyGhostDefaults:
    """Lazy proxy so ``gs.surfaces.Default()`` isn't called at import time."""

    def __iter__(self):
        return iter(_ghost_defaults())

    def keys(self):
        return _ghost_defaults().keys()

    def __getitem__(self, key):
        return _ghost_defaults()[key]


GHOST_DEFAULTS = _LazyGhostDefaults()


# Weld wiring value: child entity -> (parent entity, parent link) mounts the child at its own
# ``default_root_pos`` / ``default_root_quat``; the 4-tuple form appends an (attach_offset, attach_quat)
# that overrides the mounting transform relative to the parent link (see ``RigidEntity.attach_to``).
AttachmentSpec: TypeAlias = tuple[str, str] | tuple[str, str, Vec3FType, UnitVec4FType]


class CompositeEntityOptions(ConfigurableOptions):
    """A composite of several entities added to a scene as a single unit.

    A composite is a real, validated Options (it round-trips through the config system
    and declares its own fields). It stays in :class:`SceneOptions` as-is — a task adds
    it by name like any other entity, ``SceneOptions(robot=Droid(...))`` — and the env
    **build** loop expands it into ordinary entities + weld wiring (see
    ``RLEnvBase`` scene construction). A composite adds no runtime behaviour of its own
    beyond "these named entities + these welds", which ``env.entities`` + ``attach_to``
    already provide, so there is deliberately no composite *runtime* entity.

    Subclasses describe the rig by implementing :meth:`entity_parts` (the ordered
    ``(name, sub-entity Options)`` pairs) and, when parts are welded, :meth:`attachments`;
    :meth:`entity_options` keys those parts by name, refusing a collision that would otherwise
    cost the rig a body silently. The scene placement key must equal :attr:`primary_name` (the
    root body), so the key is meaningful (not silently discarded) and downstream terms can
    reference the sub-entities by the composite's own names. Concrete composites — an arm +
    gripper (a manipulator), a humanoid + hands — live in ``eden/options/robots/composite``.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        """Reject subclasses that bypass the duplicate-name guard.

        A subclass overriding ``entity_options`` would win normal Python dispatch and silently
        skip the validation there. Fail at class definition instead: every composite must expose
        its pre-dict sequence through :meth:`entity_parts`, while ``entity_options`` remains the
        framework-owned checked conversion.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if "entity_options" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must implement entity_parts(), not entity_options(); "
                "entity_options() performs mandatory duplicate-name validation."
            )

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        # Run the duplicate-name check at construction, so a colliding rig fails where it was
        # written — for every composite shape, not just the ones that remember to. The check
        # itself lives in entity_options(), which re-runs it on every call to also catch a
        # ``*_name`` reassigned after construction.
        self.entity_options()

    def entity_parts(self) -> list[tuple[str, "BaseEntityOptions"]]:
        """The named sub-entities this composite is built from, in declaration order (uncopied).

        A *list*, not a dict, because this is the only place two bodies sharing a name are still
        visible — :meth:`entity_options` collapses them. Implement this rather than
        :meth:`entity_options` so the duplicate check there covers every composite shape.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement entity_parts().")

    def entity_options(self) -> dict[str, "BaseEntityOptions"]:
        """The named sub-entities keyed by name (uncopied), and where a name collision is caught.

        Framework-owned: subclasses implement :meth:`entity_parts` instead, so this check cannot
        be bypassed by method dispatch.

        Two bodies under one name collapse to a single entry, leaving the composite one body
        short with nothing downstream noticing: :attr:`entity_names`, :meth:`expand` and the
        cross-composite check in :meth:`SceneOptions.model_post_init` all read *this* dict, so
        none of them can see what has already been overwritten. Checked here, on every call, not
        only at construction (:meth:`model_post_init` calls this) — these Options do not
        ``validate_assignment``, so a ``*_name`` reassigned after construction would slip past a
        construction-time-only guard.
        """
        parts = self.entity_parts()
        names = [name for name, _ in parts]
        collisions = sorted({name for name in names if names.count(name) > 1})
        if collisions:
            raise ValueError(
                f"{type(self).__name__} gives two bodies the same name {collisions}, but a "
                f"composite's entity names ARE the scene's entity names, so each body needs its "
                f"own; got {names}. Set the colliding '*_name' field(s) to something distinct."
            )
        return dict(parts)

    def attachments(self) -> dict[str, AttachmentSpec]:
        """Weld wiring: child entity -> (parent entity, parent link[, mount pos, mount quat]).

        Default: none. See :attr:`SceneOptions.attachments_dict` for the optional offset form.
        """
        return {}

    @property
    def entity_names(self) -> list[str]:
        """Scene entity names this composite produces."""
        return list(self.entity_options().keys())

    @property
    def primary_name(self) -> str:
        """The root body's name — the scene placement key must equal this."""
        children = set(self.attachments())
        roots = [name for name in self.entity_options() if name not in children]
        if len(roots) != 1:
            raise ValueError(
                f"{type(self).__name__} must have exactly one root entity (one not welded to "
                f"another); found roots {roots}. A composite needs a single primary body."
            )
        return roots[0]

    def expand(self) -> tuple[dict[str, "BaseEntityOptions"], dict[str, tuple[str, str]]]:
        """Return ``(named_entities, attachments)`` for the scene build loop.

        The sub-entity Options are deep-copied so one composite instance can be placed in
        several scenes (or reused from the registry) without the scenes aliasing — and
        mutating — the same Options. (``copy.deepcopy`` — not pydantic ``model_copy`` —
        because genesis' Options define ``__deepcopy__(self, memo)``.)
        """
        entities = {name: copy.deepcopy(opts) for name, opts in self.entity_options().items()}
        return entities, dict(self.attachments())


class SceneOptions(DynamicConfigurableOptions):
    """
    Scene options for Eden, specifying the entities to construct the scene.

    Parameters
    ----------
    attachments_dict: dict[str, tuple[str, str] | tuple[str, str, attach_offset, attach_quat]]
        A dictionary of entity_name -> (attach_to_entity_name, attach_to_link_name), optionally with a
        mounting offset: entity_name -> (attach_to_entity_name, attach_to_link_name, attach_offset, attach_quat).
        e.g., {"gripper": ("arm", "attachment")} or
        {"gripper": ("arm", "attachment", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0))}.
        Without an offset, the attached entity's ``default_root_pos`` / ``default_root_quat`` act as
        the mounting transform relative to the parent link. The optional ``attach_offset`` (xyz) and
        ``attach_quat`` (wxyz) override it (see ``RigidEntity.attach_to``).
    <entity_name>: EntityOptionsLike
        An entity configuration (entity, primitive, terrain, grouped or composite) for the given name.
    """

    attachments_dict: dict[str, AttachmentSpec] = {}

    def model_post_init(self, context):
        super().model_post_init(context)

        # The declared field (attachments_dict) lives on the instance,
        # not in __pydantic_extra__, so iterating extras gives us only the
        # user-provided named entity kwargs.
        extra = self.__pydantic_extra__ or {}

        # Validate entity types and that the scene is self-consistent. Composites
        # (e.g. a Droid manipulator) are kept as-is and round-trip through the config;
        # the env build loop expands them into their sub-entities + welds. Here we only
        # check, cheaply and without expanding, that: the placement key names the
        # composite's primary body, and no two entities (plain or composite-produced)
        # share a name. (Entity/DOF handle resolution stays deferred to build.)
        owner_of: dict[str, str] = {}  # final entity name -> scene key that produces it
        for key, entity in extra.items():
            if key.startswith("_option_"):
                continue
            if isinstance(entity, CompositeEntityOptions):
                if key != entity.primary_name:
                    raise ValueError(
                        f"Composite placed under key '{key}', but its primary entity is "
                        f"'{entity.primary_name}'. Place it under that key (e.g. "
                        f"'{entity.primary_name}=...'), or set the composite's names so its "
                        f"primary body matches the key."
                    )
                names = entity.entity_names
            elif isinstance(entity, BaseEntityOptions):
                names = [key]
            elif key == "splat_options":
                # splat_options lives on the config root, because it selects the env class and
                # `env_class_for_config` cannot see into the scene. Extras land in this entity loop, so
                # without this branch a misplaced one fails as an "invalid entity type" — loud, but
                # pointing at entity validation rather than at where the field belongs.
                raise ValueError(
                    "scene_options.splat_options is not a scene field: splat options live on the config "
                    "root, because they select the environment class. Set `splat_options=...` on an "
                    "EdenSplatConfig (or EdenNyxSplatConfig) instead of on scene_options."
                )
            else:
                raise ValueError(
                    f"Invalid entity type: {type(entity)}, expected BaseEntityOptions or "
                    f"CompositeEntityOptions (or subclass), got {type(entity)}"
                )
            for name in names:
                if name in owner_of:
                    raise ValueError(
                        f"Entity name '{name}' (from '{key}') collides with the entity from "
                        f"'{owner_of[name]}'. Give composites distinct names, e.g. a second "
                        f"manipulator as Droid(arm_name='left', gripper_name='left_gripper')."
                    )
                owner_of[name] = key

        _validate_mochi_exclusivity(self.named_entities())

    def named_entities(self) -> dict[str, BaseEntityOptions]:
        """Every entity the scene will build, keyed by final entity name (composites expanded)."""
        out: dict[str, BaseEntityOptions] = {}
        for key, entity in (self.__pydantic_extra__ or {}).items():
            if key.startswith("_option_"):
                continue
            if isinstance(entity, CompositeEntityOptions):
                out.update(entity.entity_options())
            elif isinstance(entity, BaseEntityOptions):
                out[key] = entity
        return out

    def remove_entity(self, name: str) -> None:
        """Remove the entity placed under scene key ``name``, together with its own attachment.

        The supported way for a task variant to drop an entity (see
        ``eden/tasks/robosuite/variants.py::drop_object``). Only an entity added by
        name (an extra) is removable; one declared as a field of a ``SceneOptions`` subclass
        (``DroidSceneOptions.robot``) is part of that scene class. An ``attachments_dict`` entry
        that welds the removed entity onto a parent goes with it. An entity that *other* entities
        are welded onto cannot be removed — remove the children first. Nothing else that names
        the entity is touched — manager terms, cameras, sensors — so update those too, otherwise
        they fail to resolve the entity at build time.

        Parameters
        ----------
        name : str
            Scene key of the entity. A composite is removed whole, under its placement key; one of
            its bodies cannot be removed on its own.

        Raises
        ------
        RuntimeError
            If the scene options are locked (checked before anything else).
        KeyError
            If the scene has no entity ``name`` (the message lists the removable entities), or
            ``name`` cannot be removed: a declared field, or a body of a composite (the message
            says which).
        ValueError
            If another entity is attached to the one being removed.
        """
        try:
            entity = self._removable_child(name, noun="entity")
        except KeyError as err:
            # The generic "not found" would be wrong for an entity that exists but is not an extra.
            reason = self._unremovable_entity_reason(name)
            if reason is None:
                raise
            raise KeyError(reason) from err
        # A composite produces several scene entities (arm + gripper); all of them go with it.
        removed_names = set(entity.entity_names) if isinstance(entity, CompositeEntityOptions) else {name}
        dependents = sorted(
            child
            for child, spec in self.attachments_dict.items()
            if spec[0] in removed_names and child not in removed_names
        )
        if dependents:
            raise ValueError(
                f"Cannot remove entity '{name}': {dependents} are attached to it via attachments_dict. "
                f"Remove those entities first."
            )

        self._remove_named_child(name, noun="entity")
        # Rebuilt, not edited in place: a shallow model_copy() shares the dict with its source scene,
        # which would silently lose its own welds.
        self.attachments_dict = {
            child: spec for child, spec in self.attachments_dict.items() if child not in removed_names
        }

    def _unremovable_entity_reason(self, name: str) -> str | None:
        """Why the scene entity ``name``, which is not an extra, cannot be removed; ``None`` if there is none.

        Parameters
        ----------
        name : str
            The name :meth:`remove_entity` was given.

        Returns
        -------
        str | None
            The error message, or ``None`` when ``name`` is simply not in the scene.
        """
        scene = type(self).__name__
        declared = type(self).model_fields
        if name in declared:
            return (
                f"'{name}' is a declared field of {scene}, not an entity added by name, so it cannot be "
                f"removed: a declared entity is part of the scene class. To go without it, build the scene "
                f"from a SceneOptions class that does not declare it."
            )
        # A composite's bodies are scene entities by name but not scene keys, wherever the composite is.
        placed = {key: getattr(self, key) for key in declared} | (self.__pydantic_extra__ or {})
        for key, value in placed.items():
            if isinstance(value, CompositeEntityOptions) and name in value.entity_names:
                prefix = (
                    f"Entity '{name}' is a body of the composite '{key}' ({type(value).__name__}), not a scene "
                    f"key of its own, and a composite is only removed whole — removing one body would leave "
                    f"the rig incomplete."
                )
                if key in declared:
                    return f"{prefix} '{key}' is a declared field of {scene}, so the rig cannot be removed either."
                return f"{prefix} To drop the whole rig, call remove_entity('{key}')."
        return None


def _validate_mochi_exclusivity(entities: dict[str, BaseEntityOptions]) -> None:
    """Raise ValueError when a scene's entities (:meth:`SceneOptions.named_entities`) mix Mochi with another solver.

    The MochiSolver assembles every body and the penalty contact into one Newton system and
    Genesis refuses to build it beside another active solver (``Simulator.build``) — but only at
    ``scene.build()``, after every asset has been loaded. Kinematic (visual-only) entities are the
    one exception Genesis allows. A missing material counts as rigid, because
    ``Entity._parse_material`` defaults it to ``RigidMaterialOptions``.
    """
    mochi = [name for name, e in entities.items() if isinstance(e.material, MochiMaterialOptions)]
    if not mochi:
        return
    others = [
        name
        for name, e in entities.items()
        if not isinstance(e.material, (MochiMaterialOptions, KinematicMaterialOptions))
    ]
    if others:
        raise ValueError(
            f"Entities {mochi} use the MochiSolver, which cannot run beside another physics solver, but "
            f"{others} carry a non-Mochi material (a missing material defaults to RigidMaterialOptions). Give "
            f"every entity a Mochi material — e.g. `material=MochiRigidMaterialOptions()` on the ground plane — or "
            f"a KinematicMaterialOptions."
        )


#: Anything a :class:`SceneOptions` accepts under an entity name — a composite included, since
#: it is placed like one entity and expanded at build time.
EntityOptionsLike: TypeAlias = (
    EntityOptions | PrimitiveOptions | TerrainOptions | GroupedEntityOptions | CompositeEntityOptions
)
"""Union of the user-facing entity options classes."""
