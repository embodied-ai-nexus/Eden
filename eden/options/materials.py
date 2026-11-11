"""Material options for rigid, MPM, SPH, FEM, PBD, and Mochi entities."""

from __future__ import annotations

from typing import ClassVar, Literal, TypeAlias

import genesis as gs
from pydantic import Field

from eden.engine.solvers.mochi import materials as mochi_materials
from eden.options.options import ConfigurableOptions


class MaterialOptions(ConfigurableOptions):
    """
    Options for Genesis materials.

    Subclasses set ``_genesis_material_cls`` to the corresponding ``gs.materials.*``
    class; ``to_genesis_material`` then forwards every declared field through.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        If True, the entity's visual mesh is included in the raycaster BVH used by
        Lidar / DepthCamera sensors, and by a viewer plugin casting against visual
        geometry (``MeshPointSelectorPlugin(use_visual_geom=True)``). Required for
        non-colliding background meshes (e.g. photogrammetry GLBs with
        ``collision=False``) to be hit by lidar rays. Must be set before
        ``scene.build()``. Default is False.
    """

    use_visual_raycasting: bool = False

    _genesis_material_cls: ClassVar[type | None] = None

    def to_genesis_material(self):
        cls = self._genesis_material_cls
        if cls is None:
            raise NotImplementedError(f"{type(self).__name__} does not implement to_genesis_material()")
        # Drop None-valued fields so Genesis applies its own (often lazily-resolved)
        # defaults rather than seeing an explicit ``None``. Several Genesis materials
        # reject an explicit ``sampler=None`` even though it is their ``__init__``
        # default (e.g. ``gs.materials.MPM.Elastic``), so forwarding None would make
        # ``MPMElasticMaterialOptions()`` unusable.
        return cls(**{k: v for k, v in self.dict().items() if v is not None})


class KinematicMaterialOptions(MaterialOptions):
    """Options for Genesis kinematic materials."""

    _genesis_material_cls: ClassVar[type] = gs.materials.Kinematic


class RigidMaterialOptions(MaterialOptions):
    """
    Options for Genesis rigid materials.

    Parameters
    ----------
    rho : float, optional
        The density of the material used to compute mass. Default is 200.0.
    friction : float, optional
        Friction coefficient within the rigid solver. If None, a default of 1.0 may be used or parsed from file.
    friction_torsional : float, optional
        Torsional friction coefficient: how strongly a contact resists relative spin about its normal.
        Expressed in **meters** — it is the effective contact-patch radius, and it bounds the resisting
        torque at ``friction_torsional * normal_force``. Requires
        ``EnvOptions.enable_torsional_friction`` (Eden raises at build if the switch is off) and costs
        one extra constraint axis per contact. If None, parsed from the asset when available (MJCF
        ``friction[1]``, only for geoms declaring ``condim >= 4``), otherwise genesis' 0.005.
        Default is None.
    friction_rolling : float, optional
        Rolling friction coefficient: how strongly a contact resists rolling about its two tangent
        axes. In meters like ``friction_torsional``. Requires ``EnvOptions.enable_rolling_friction``
        (which itself requires ``enable_torsional_friction``) and costs two extra constraint axes per
        contact. If None, parsed from the asset when available (MJCF ``friction[2]``, only for geoms
        declaring ``condim >= 6``), otherwise genesis' 0.0001. Default is None.
    needs_coup : bool, optional
        Whether the material participates in coupling with other solvers. Default is True.
    coup_friction : float, optional
        Friction used during coupling. Must be non-negative. Default is 0.1.
    coup_softness : float, optional
        Softness of coupling interaction. Must be non-negative. Default is 0.002.
    coup_restitution : float, optional
        Restitution coefficient in collision coupling. Should be between 0 and 1. Default is 0.0.
    sdf_cell_size : float, optional
        Cell size in SDF grid in meters. Defines grid resolution. Default is 0.005.
    sdf_min_res : int, optional
        Minimum resolution of the SDF grid. Must be at least 16. Default is 32.
    sdf_max_res : int, optional
        Maximum resolution of the SDF grid. Must be >= sdf_min_res. Default is 128.
    gravity_compensation : float, optional
        Compensation factor for gravity. 1.0 cancels gravity. Default is 0.
    coup_type : str or None, optional
        Coupling mode for this entity. Only used by the IPC coupler. Requires ``needs_coup=True``.
        If None, auto-selected based on entity type: ``'external_articulation'`` for fixed-base
        articulated robots, ``'two_way_soft_constraint'`` for floating-base robots, and
        ``'ipc_only'`` for non-articulated objects. Valid values:
            - 'two_way_soft_constraint': Two-way soft coupling.
            - 'external_articulation': Joint-level coupling for articulated bodies. Joint positions will be coupled at
            the DOF level.
            - 'ipc_only': IPC controls entity, transforms copied to Genesis (one-way). Only supported by rigid
            non-articulated objects.
        Default is None.
    coup_links : list of str or None, optional
        Link names to include in coupling. When set, only the named links participate
        in coupling; other links are excluded. Only supported with needs_coup=True and
        ``two_way_soft_constraint`` type in IPC. Default is None.
    enable_coup_collision : bool, optional
        Whether coupler collision is enabled for this entity's links. Only used by the IPC coupler.
        Unlike ``needs_coup=False`` (which removes the entity from the coupler entirely), setting this to
        False keeps the entity in the coupler for coupling forces but disables contact response. Default is True.
    coup_collision_links : list of str or None, optional
        Link names whose geoms participate in coupler collision. Only used by the IPC coupler.
        Only effective when ``enable_coup_collision=True``. If None, all coupled links have collision.
        When set, only the named links get coupler collision; other links are marked no-collision.
        Default is None.
    contact_resistance : float or None, optional
        IPC coupling contact resistance/stiffness override for this entity. ``None`` means use
        ``IPCCouplerOptions.contact_resistance``. Default is None.
    """

    rho: float = 200.0
    friction: float | None = None
    friction_torsional: float | None = Field(default=None, ge=0.0)
    friction_rolling: float | None = Field(default=None, ge=0.0)
    needs_coup: bool = True
    coup_friction: float = 0.1
    coup_softness: float = 0.002
    coup_restitution: float = 0.0
    sdf_cell_size: float = 0.005
    sdf_min_res: int = 32
    sdf_max_res: int = 128
    gravity_compensation: float = 0.0
    coup_type: Literal["two_way_soft_constraint", "external_articulation", "ipc_only"] | None = None
    coup_links: list[str] | None = None
    enable_coup_collision: bool = True
    coup_collision_links: list[str] | None = None
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.Rigid


class MPMElasticMaterialOptions(MaterialOptions):
    """
    Options for Genesis MPM elastic materials.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 3e5.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    model: str, optional
        Stress model ('corotation', 'neohooken'). Default is 'corotation'.
    """

    E: float = 3e5
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    sampler: str | None = None
    model: Literal["corotation", "neohooken"] = "corotation"

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.Elastic


class MPMElastoPlasticMaterialOptions(MaterialOptions):
    """Options for an MPM elasto-plastic material.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    yield_lower: float, optional
        Lower bound for the yield clamp (ignored if using von Mises). Default is 2.5e-2.
    yield_higher: float, optional
        Upper bound for the yield clamp (ignored if using von Mises). Default is 4.5e-2.
    use_von_mises: bool, optional
        Whether to use von Mises yield criterion. Default is True.
    von_mises_yield_stress: float, optional
        Yield stress for von Mises criterion. Default is 10000.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    sampler: str | None = None
    yield_lower: float = 2.5e-2
    yield_higher: float = 4.5e-2
    use_von_mises: bool = True
    von_mises_yield_stress: float = 10000.0

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.ElastoPlastic


class MPMLiquidMaterialOptions(MaterialOptions):
    """Options for an MPM liquid material.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    viscous: bool, optional
        Whether the liquid is viscous. Simply set mu to zero when non-viscous. Default is False.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    viscous: bool = False
    sampler: str | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.Liquid


class MPMSandMaterialOptions(MaterialOptions):
    """Options for an MPM sand (granular) material.

    Parameters
    ----------
    E: float, optional
        Young's modulus. Default is 1e6.
    nu: float, optional
        Poisson ratio. Default is 0.2.
    rho: float, optional
        Density (kg/m^3). Default is 1000.
    lam: float, optional
        The first Lame's parameter. Default is None, computed by E and nu.
    mu: float, optional
        The second Lame's parameter. Default is None, computed by E and nu.
    friction_angle: float, optional
        Internal friction angle of the granular material (degrees). Default is 45.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). Note that 'pbs' is only supported on Linux x86 for now. Defaults
        to 'pbs' on supported platforms, 'random' otherwise.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    friction_angle: float = 45.0
    sampler: str | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.MPM.Sand


class SPHLiquidMaterialOptions(MaterialOptions):
    """Options for an SPH liquid material.

    Parameters
    ----------
    rho: float, optional
        The rest density (kg/m^3) the material tends to maintain in equilibrium. Default is 1000.
    stiffness: float, optional
        State stiffness (N/m^2): how pressure increases with compression. Default is 50000.0.
    exponent: float, optional
        State exponent: how nonlinearly pressure scales with density. Larger values mean a stiffer response to
        compression. Default is 7.0.
    mu: float, optional
        The viscosity of the liquid. Default is 0.005.
    gamma: float, optional
        The surface tension of the liquid: how strongly the material "clumps" together at boundaries. Default is 0.01.
    sampler: str, optional
        Particle sampler ('pbs', 'regular', 'random'). 'pbs' is only supported on Linux x86. Default is 'regular':
        SPH is sensitive to the initial particle distribution, which sets the initial density and pressure fields.
        Irregular samplers ('pbs', 'random') introduce local density fluctuations at initialization, which cause
        large spurious pressure forces and can make the simulation unstable or diverge.
    """

    rho: float = 1000.0
    stiffness: float = 50_000.0
    exponent: float = 7.0
    mu: float = 0.005
    gamma: float = 0.01
    sampler: str = "regular"

    _genesis_material_cls: ClassVar[type] = gs.materials.SPH.Liquid


class PBDLiquidMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) liquid materials.

    Particle-based fluid simulated with the PBD solver. The PBD solver's
    simulation domain (``lower_bound``/``upper_bound``) and iteration counts are
    configured via ``EnvOptions.pbd_options`` (a ``gs.options.PBDOptions``). If
    left unset, genesis uses its PBD defaults (a ``(-100, -100, 0)``..``(100,
    100, 100)`` domain) — set it to bound the domain to the working volume.
    Position/orientation are baked into the morph at creation time
    (``set_pos``/``set_quat`` raise on the resulting entity).

    Parameters
    ----------
    rho : float, optional
        Rest density of the fluid. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'random', 'regular'). 'pbs' is only supported on
        Linux x86. Default is 'pbs'.
    density_relaxation : float, optional
        Relaxation factor for the density (incompressibility) constraint. Larger
        values enforce incompressibility more strongly per iteration. Default is 0.2.
    viscosity_relaxation : float, optional
        Relaxation factor for the XSPH viscosity constraint. Larger values make the
        fluid more viscous. Default is 0.01.
    """

    rho: float = 1000.0
    sampler: Literal["pbs", "random", "regular"] = "pbs"
    density_relaxation: float = 0.2
    viscosity_relaxation: float = 0.01

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Liquid


class PBDClothMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) cloth materials.

    Thin-shell cloth simulated as a particle/edge network with the PBD solver.
    Built from a triangle mesh morph (e.g. ``EntityOptions(file="meshes/cloth.obj")``).
    The PBD solver iteration counts are configured via ``EnvOptions.pbd_options``
    (genesis defaults are used if unset). Individual particles can be pinned via the
    underlying Genesis entity's ``fix_particles`` / ``find_closest_particle``
    (accessible through ``entity.entity``).

    Parameters
    ----------
    rho : float, optional
        Surface density of the cloth. Default is 4.0.
    static_friction : float, optional
        Static friction coefficient for particle contact. Default is 0.15.
    kinetic_friction : float, optional
        Kinetic friction coefficient for particle contact. Default is 0.15.
    stretch_compliance : float, optional
        Compliance (inverse stiffness) of the stretch constraint. Smaller values
        make the cloth less stretchy. Default is 1e-7.
    bending_compliance : float, optional
        Compliance (inverse stiffness) of the bending constraint. Smaller values
        make the cloth resist folding more. Default is 1e-5.
    stretch_relaxation : float, optional
        Relaxation factor for the stretch constraint per iteration. Default is 0.3.
    bending_relaxation : float, optional
        Relaxation factor for the bending constraint per iteration. Default is 0.1.
    air_resistance : float, optional
        Air drag applied to the cloth particles. Default is 0.001.
    """

    rho: float = 4.0
    static_friction: float = 0.15
    kinetic_friction: float = 0.15
    stretch_compliance: float = 1e-7
    bending_compliance: float = 1e-5
    stretch_relaxation: float = 0.3
    bending_relaxation: float = 0.1
    air_resistance: float = 0.001

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Cloth


class PBDElasticMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) elastic (soft-body) materials.

    Volumetric soft body simulated with the PBD solver, governed by stretch,
    bending, and volume-preservation constraints. The PBD solver is configured
    via ``EnvOptions.pbd_options`` (genesis defaults are used if unset).

    Parameters
    ----------
    rho : float, optional
        Material density. Default is 1000.0.
    static_friction : float, optional
        Static friction coefficient for particle contact. Default is 0.15.
    kinetic_friction : float, optional
        Kinetic friction coefficient for particle contact. Default is 0.15.
    stretch_compliance : float, optional
        Compliance (inverse stiffness) of the stretch constraint. Default is 0.0.
    bending_compliance : float, optional
        Compliance (inverse stiffness) of the bending constraint. Default is 0.0.
    volume_compliance : float, optional
        Compliance (inverse stiffness) of the volume-preservation constraint.
        Default is 0.0.
    stretch_relaxation : float, optional
        Relaxation factor for the stretch constraint per iteration. Default is 0.1.
    bending_relaxation : float, optional
        Relaxation factor for the bending constraint per iteration. Default is 0.1.
    volume_relaxation : float, optional
        Relaxation factor for the volume constraint per iteration. Default is 0.1.
    """

    rho: float = 1000.0
    static_friction: float = 0.15
    kinetic_friction: float = 0.15
    stretch_compliance: float = 0.0
    bending_compliance: float = 0.0
    volume_compliance: float = 0.0
    stretch_relaxation: float = 0.1
    bending_relaxation: float = 0.1
    volume_relaxation: float = 0.1

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Elastic


class PBDParticleMaterialOptions(MaterialOptions):
    """
    Options for Genesis PBD (Position-Based Dynamics) free-particle materials.

    Bare particles driven by the PBD solver with no internal constraints (collide
    and respond to gravity only). The PBD solver is configured via
    ``EnvOptions.pbd_options`` (genesis defaults are used if unset).

    Parameters
    ----------
    rho : float, optional
        Material density. Default is 1000.0.
    sampler : str, optional
        Particle sampler ('pbs', 'random', 'regular'). 'pbs' is only supported on
        Linux x86. Default is 'pbs'.
    """

    rho: float = 1000.0
    sampler: Literal["pbs", "random", "regular"] = "pbs"

    _genesis_material_cls: ClassVar[type] = gs.materials.PBD.Particle


class FEMElasticMaterialOptions(MaterialOptions):
    """
    Options for Genesis FEM elastic materials.

    Used for volumetric deformable bodies (tetrahedralized meshes) simulated
    with the FEM solver. When an IPC coupler is active, these entities
    participate in penetration-free frictional contact.

    Parameters
    ----------
    E : float, optional
        Young's modulus controlling stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio describing volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    model : str, optional
        Constitutive model ('linear', 'stable_neohookean', 'linear_corotated'). Default is 'linear'.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float or None, optional
        IPC contact resistance/stiffness override. None uses the coupler global default. Default is None.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"
    friction_mu: float = 0.1
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.FEM.Elastic


class FEMClothMaterialOptions(MaterialOptions):
    """
    Options for Genesis FEM cloth (thin shell) materials.

    Used for cloth, fabric, and thin flexible surfaces simulated as 2D shells
    in the IPC backend. Requires an IPC coupler and GPU backend.

    Parameters
    ----------
    E : float, optional
        Young's modulus (Pa) controlling stiffness. Default is 1e4.
    nu : float, optional
        Poisson ratio. Default is 0.49 (nearly incompressible).
    rho : float, optional
        Material density (kg/m^3). Default is 200.
    thickness : float, optional
        Shell thickness in meters. Default is 0.001 (1mm).
    bending_stiffness : float or None, optional
        Bending resistance. None disables bending. Default is None.
    model : str, optional
        FEM material model. Default is 'stable_neohookean'.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float or None, optional
        IPC contact resistance/stiffness override. None uses the coupler global default. Default is None.
    """

    E: float = 1e4
    nu: float = 0.49
    rho: float = 200.0
    thickness: float = 0.001
    bending_stiffness: float | None = None
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "stable_neohookean"
    friction_mu: float = 0.1
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.FEM.Cloth


class FEMMuscleMaterialOptions(MaterialOptions):
    """
    Options for Genesis FEM muscle materials.

    Extends FEM elastic material with muscle activation support for
    soft-body actuation.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    model : str, optional
        Constitutive model. Default is 'linear'.
    n_groups : int, optional
        Number of muscle groups. Default is 1.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    contact_resistance : float or None, optional
        IPC contact resistance/stiffness override. Default is None.
    """

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    model: Literal["linear", "stable_neohookean", "linear_corotated"] = "linear"
    n_groups: int = 1
    friction_mu: float = 0.1
    contact_resistance: float | None = None

    _genesis_material_cls: ClassVar[type] = gs.materials.FEM.Muscle


_MOCHI_CONTACT_DOC = """
    friction : float, optional
        Coulomb friction coefficient. Combined per contact pair as the geometric mean of the two
        bodies' values. Default is 0.5.
    penalty_coefficient : float, optional
        Contact pressure per unit penetration (Pa/m) once the smooth ramp is fully active. The
        default 1e9 makes the Newton system ill-conditioned in single precision — run Mochi scenes
        with ``en.init(precision="64")``, or lower it (light cloth uses 1e7). Default is 1e9.
    penalty_smoothing_half_distance : float, optional
        Half-width (m) of the smooth pressure ramp; the full ramp spans twice this. Taken from the
        collider side of a pair. Default is 5e-3.
    penalty_threshold : float, optional
        Signed distance (m) at which contact pressure starts to build; positive values form a
        contact skin above the surface. Taken from the collider side of a pair. Default is 1e-3.
    friction_falloff_vel : float, optional
        Sliding speed (m/s) below which Coulomb friction is regularized. Default is 1e-2.
    viscous_friction : float, optional
        Tangential viscous friction coefficient (s/m). Default is 0.0.
    normal_viscous_damping : float, optional
        Normal impact damping (s/m), the restitution knob: the coefficient of restitution ``e`` at
        impact speed ``v`` follows from ``c = (1-e)(1+4.5e)/(e(1+8e/3))/v``. Default is 0.0.
    max_alignment_normals : float, optional
        Cosine threshold above which a contact whose normal aligns with the collider normal is
        dropped (escape hatch for embedded bodies). Taken from the collider side. Default is 0.0.
    has_gravity : bool, optional
        Whether gravity acts on this body. Default is True.
    contact_layer : str, optional
        Contact layer name; pairs of layers are switched with
        ``env.mochi_solver.enable_layer_contact``. Default is ``"default"``.
"""


class MochiMaterialOptions(MaterialOptions):
    """
    Base of every Mochi (implicit multi-physics solver) material.

    Not usable directly — pick :class:`MochiRigidMaterialOptions`, :class:`MochiElasticMaterialOptions`,
    :class:`MochiShellMaterialOptions` or :class:`MochiRodMaterialOptions`. A scene holding one Mochi
    entity must hold *only* Mochi (and kinematic) entities: the MochiSolver assembles every body and
    the penalty contact into one Newton system and refuses to run beside another physics solver.
    The shared contact parameters below are combined per contact pair as documented on each field.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See :class:`MaterialOptions`. Default is False.
    """

    __doc__ = (__doc__ or "") + _MOCHI_CONTACT_DOC

    friction: float = 0.5
    penalty_coefficient: float = 1e9
    penalty_smoothing_half_distance: float = 5e-3
    penalty_threshold: float = 1e-3
    friction_falloff_vel: float = 1e-2
    viscous_friction: float = 0.0
    normal_viscous_damping: float = 0.0
    max_alignment_normals: float = 0.0
    has_gravity: bool = True
    contact_layer: str = "default"

    def model_post_init(self, context):
        super().model_post_init(context)
        if type(self) is MochiMaterialOptions:
            raise ValueError(
                "MochiMaterialOptions is abstract; use MochiRigidMaterialOptions, MochiElasticMaterialOptions, "
                "MochiShellMaterialOptions or MochiRodMaterialOptions."
            )


class MochiRigidMaterialOptions(MochiMaterialOptions):
    """
    Options for Mochi rigid and articulated bodies (``eden.engine.solvers.mochi.materials.Rigid``).

    Deliberately *not* a :class:`RigidMaterialOptions`: the Mochi solver has no coupler, so the
    ``coup_*`` / ``gravity_compensation`` / ``needs_coup`` fields do not exist here (Genesis rejects
    them on ``Mochi.Rigid``; gravity is switched per body with ``has_gravity``). Entities carrying
    this material build as :class:`~eden.entities.mochi.MochiRigidEntity`, which keeps the whole
    rigid-entity API (DOFs, PD control, IK, ``attach_to``) minus force ranges, friction loss and
    constraint-solver parameters, which the solver never reads.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See :class:`MaterialOptions`. Default is False.
    rho : float, optional
        Density (kg/m^3) used to derive mass and inertia when the asset gives none; overrides
        per-geometry densities declared in the asset. Default is 1000.0.
    collider_type : {"auto", "plane", "sphere", "box", "sdf", "none"}, optional
        How *other* bodies collide against this one: an analytic plane/sphere/box, a signed
        distance field of the mesh, or nothing. ``"auto"`` picks from the morph. An analytic field needs
        the matching primitive morph (the solver raises at build otherwise). Default is "auto".
    sdf_cell_size : float, optional
        Cell size (m) of the SDF grid built for ``collider_type="sdf"``. Default is 2.5e-3.
    sdf_min_res : int, optional
        Minimum SDF grid resolution. Default is 32.
    sdf_max_res : int, optional
        Maximum SDF grid resolution. Default is 128.
    """

    __doc__ = (__doc__ or "") + _MOCHI_CONTACT_DOC

    rho: float = 1000.0
    collider_type: Literal["auto", "plane", "sphere", "box", "sdf", "none"] = "auto"
    sdf_cell_size: float = 2.5e-3
    sdf_min_res: int = 32
    sdf_max_res: int = 128

    _genesis_material_cls: ClassVar[type] = mochi_materials.Rigid


class MochiElasticMaterialOptions(MochiMaterialOptions):
    """
    Options for Mochi volumetric soft bodies (``eden.engine.solvers.mochi.materials.Elastic``).

    Linear tetrahedra with a stable neo-Hookean, Saint Venant-Kirchhoff or linear material, solved
    fully implicitly together with contact. Box / Sphere / Cylinder / Mesh morphs are tetrahedralized
    by tetgen (pass ``morph={"maxvolume": ..., "nobisect": False}`` on the primitive to refine); a
    ``.node`` (tetgen) or ``.vtk`` (legacy ASCII unstructured grid) file is read as-is. Entities
    build as :class:`~eden.entities.mochi.MochiSoftEntity`.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See :class:`MaterialOptions`. Default is False.
    E : float, optional
        Young's modulus (Pa). Default is 1e5.
    nu : float, optional
        Poisson ratio. Default is 0.45.
    rho : float, optional
        Density (kg/m^3). Default is 1000.0.
    model : {"stable_neohookean", "stvk", "linear"}, optional
        Constitutive model. Default is "stable_neohookean".
    mass_damping : float, optional
        Rayleigh mass-proportional damping (1/s). Default is 0.0.
    stiffness_damping : float, optional
        Rayleigh stiffness-proportional damping (s). Default is 0.0.
    collider_type : {"auto", "sdf", "none"}, optional
        How other bodies collide against this body: a signed distance field of the rest shape
        queried through the deformed tetrahedra (no contact skin — a few millimeters of
        interpenetration remain), or nothing. Default is "auto".
    """

    __doc__ = (__doc__ or "") + _MOCHI_CONTACT_DOC

    E: float = 1e5
    nu: float = 0.45
    rho: float = 1000.0
    model: Literal["stable_neohookean", "stvk", "linear"] = "stable_neohookean"
    mass_damping: float = 0.0
    stiffness_damping: float = 0.0
    collider_type: Literal["auto", "sdf", "none"] = "auto"

    _genesis_material_cls: ClassVar[type] = mochi_materials.Elastic


class MochiShellMaterialOptions(MochiMaterialOptions):
    """
    Options for Mochi thin shells / cloth (``eden.engine.solvers.mochi.materials.Shell``).

    The surface triangles of a Mesh (or primitive) morph become a Saint Venant-Kirchhoff membrane
    plus a Koiter-type bending term, both thickness-integrated from ``E``, ``nu``, ``rho`` and
    ``thickness`` unless overridden. Shells act as colliders through spheres of ``collider_radius``
    at their vertices, which only other point-cloud bodies (cloths, rods) and — with
    ``self_contact`` — the shell itself feel; rigid bodies and solids touch the shell through its own
    contact samples. Consistently wound meshes are required. Light cloth has little inertia: use
    ``penalty_coefficient=1e7``. Entities build as :class:`~eden.entities.mochi.MochiSoftEntity`.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See :class:`MaterialOptions`. Default is False.
    E : float, optional
        Young's modulus (Pa). Default is 1e4.
    nu : float, optional
        Poisson ratio. Default is 0.3.
    rho : float, optional
        Density (kg/m^3); areal density is ``rho * thickness``. Default is 200.0.
    thickness : float, optional
        Shell thickness (m). Default is 1e-3.
    membrane_mu : float or None, optional
        Membrane shear stiffness override; None derives ``thickness * mu``. Default is None.
    membrane_lambda : float or None, optional
        Membrane Lame lambda override (plane stress); None derives it. Default is None.
    bending_alpha : float or None, optional
        Bending stiffness override, ``D * nu`` when None. Default is None.
    bending_beta : float or None, optional
        Bending stiffness override, ``D * (1 - nu)`` when None. Default is None.
    mass_damping : float, optional
        Rayleigh mass-proportional damping (1/s). Default is 0.0.
    stiffness_damping : float, optional
        Rayleigh stiffness-proportional damping (s). Default is 0.0.
    collider_type : {"auto", "point_cloud", "none"}, optional
        Whether the shell carries vertex spheres other point-cloud bodies collide with. Default is "auto".
    collider_radius : float, optional
        Radius (m) of the vertex spheres. Default is 1e-2.
    self_contact : bool, optional
        Let the shell's samples collide with its own vertex spheres. Default is False.
    self_contact_exclusion_ratio : float, optional
        Vertices within ``collider_radius * ratio`` (plus the penalty threshold) of a sample in the
        rest configuration are excluded from self-contact. Default is 1.5.
    """

    __doc__ = (__doc__ or "") + _MOCHI_CONTACT_DOC

    E: float = 1e4
    nu: float = 0.3
    rho: float = 200.0
    thickness: float = 1e-3
    membrane_mu: float | None = None
    membrane_lambda: float | None = None
    bending_alpha: float | None = None
    bending_beta: float | None = None
    mass_damping: float = 0.0
    stiffness_damping: float = 0.0
    collider_type: Literal["auto", "point_cloud", "none"] = "auto"
    collider_radius: float = 1e-2
    self_contact: bool = False
    self_contact_exclusion_ratio: float = 1.5

    _genesis_material_cls: ClassVar[type] = mochi_materials.Shell


class MochiRodMaterialOptions(MochiMaterialOptions):
    """
    Options for Mochi discrete elastic rods (``eden.engine.solvers.mochi.materials.Rod``).

    A Kirchhoff rod (stretch, bend, twist; no shear) on the polyline of a
    :class:`~eden.options.entities.RodOptions` morph — the only morph this material accepts. The
    five stiffness / inertia quantities derive from ``E``, ``nu``, ``rho`` and the rod radius unless
    given explicitly. Clamp a rod by fixing its first two nodes. Rods collide through samples on
    their centerline and act as colliders through node spheres of the rod radius, which only other
    point-cloud bodies (and the rod itself under ``self_contact``) feel. Entities build as
    :class:`~eden.entities.mochi.MochiSoftEntity`.

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        See :class:`MaterialOptions`. Default is False.
    E : float, optional
        Young's modulus (Pa). Default is 1e7.
    nu : float, optional
        Poisson ratio (shear modulus ``G = E / (2 (1 + nu))``). Default is 0.3.
    rho : float, optional
        Density (kg/m^3). Default is 1000.0.
    axial_stiffness : float or None, optional
        ``E * A`` override. Default is None.
    flexural_stiffness : float or None, optional
        ``E * I`` override. Default is None.
    torsional_stiffness : float or None, optional
        ``G * J`` override. Default is None.
    linear_density : float or None, optional
        ``rho * A`` override (kg/m). Default is None.
    linear_rotational_inertia : float or None, optional
        ``rho * J`` override (kg m). Default is None.
    mass_damping : float, optional
        Rayleigh mass-proportional damping (1/s). Default is 0.0.
    stiffness_damping : float, optional
        Rayleigh stiffness-proportional damping (s). Default is 0.0.
    collider_type : {"auto", "point_cloud", "none"}, optional
        Whether the rod carries node spheres other point-cloud bodies collide with. Default is "auto".
    self_contact : bool, optional
        Let the rod's samples collide with its own node spheres. Default is False.
    self_contact_exclusion_ratio : float, optional
        Nodes within ``radius * ratio`` (plus the penalty threshold) of a sample in the rest
        configuration are excluded from self-contact; e.g. 3 for segments twice the radius. Default is 1.5.
    """

    __doc__ = (__doc__ or "") + _MOCHI_CONTACT_DOC

    E: float = 1e7
    nu: float = 0.3
    rho: float = 1000.0
    axial_stiffness: float | None = None
    flexural_stiffness: float | None = None
    torsional_stiffness: float | None = None
    linear_density: float | None = None
    linear_rotational_inertia: float | None = None
    mass_damping: float = 0.0
    stiffness_damping: float = 0.0
    collider_type: Literal["auto", "point_cloud", "none"] = "auto"
    self_contact: bool = False
    self_contact_exclusion_ratio: float = 1.5

    _genesis_material_cls: ClassVar[type] = mochi_materials.Rod


MaterialLike: TypeAlias = (
    RigidMaterialOptions
    | KinematicMaterialOptions
    | MPMElasticMaterialOptions
    | MPMElastoPlasticMaterialOptions
    | MPMLiquidMaterialOptions
    | MPMSandMaterialOptions
    | SPHLiquidMaterialOptions
    | PBDLiquidMaterialOptions
    | PBDClothMaterialOptions
    | PBDElasticMaterialOptions
    | PBDParticleMaterialOptions
    | FEMElasticMaterialOptions
    | FEMClothMaterialOptions
    | FEMMuscleMaterialOptions
    | MochiRigidMaterialOptions
    | MochiElasticMaterialOptions
    | MochiShellMaterialOptions
    | MochiRodMaterialOptions
)
"""Union of every concrete material options class."""
