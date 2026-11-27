"""Physics audit of one entity: build it alone on a ground plane and check what Genesis made of the asset.

``eden audit`` (handler in :mod:`eden.utils.cli.audit`: a task's scene entity, an entity name or an asset file)
calls :func:`audit_entity`. The audit
reads the *built* model, so it sees what the solver will simulate rather than what the asset file says: the
armature Genesis filled in, the masses after scaling, the collision geometry after convexification.

Static checks (one build, no stepping):

- ``collision``: the entity has collision geometry at all.
- ``mass``: every movable link that carries collision geometry has a positive mass, and none is below
  :attr:`AuditThresholds.min_link_mass`.
- ``mass_ratio``: across every DOF-carrying joint between two movable links, the heavier link is at most
  :attr:`AuditThresholds.max_mass_ratio` times the lighter one. Larger ratios make the mass matrix ill-conditioned.
- ``armature``: the hinge and slide DOFs whose armature Genesis filled in from ``default_armature`` because the
  asset authored none (every URDF joint). See ``docs/guides/entities.md``, "Armature defaults and the URDF 0.1 trap".
- ``joint_limits``: hinge and slide DOFs with an empty range; unlimited ones are listed for information.

Dynamic checks:

- ``self_contact``: pairs of links of the entity whose collision shapes touch at its default pose, plus the shape
  pairs Genesis drops from collision for good because they overlap at the asset's zero pose (the candidate pairs a
  second build with ``enable_neutral_collision=True`` keeps and the audited build does not), per link pair.
- ``settle`` (free base): drop the entity from :attr:`AuditThresholds.drop_height` above the plane and simulate.
  Fails on non-finite state, a link faster than the explosion limits, falling through the plane, or ground
  penetration deeper than :attr:`AuditThresholds.max_penetration` over the final
  :attr:`AuditThresholds.rest_window` (the transient on impact is reported, not graded); warns when it is still
  moving at the end.
- ``hold`` (fixed base): command the default pose with the built PD gains and simulate. Fails on non-finite state
  or a DOF faster than :attr:`AuditThresholds.max_dof_speed`; warns when a DOF drifts further than
  :attr:`AuditThresholds.max_hold_drift`.

The scene runs at the physics settings of the ``env_options`` given (a task's, for ``eden audit --task``), else
at :class:`~eden.options.EnvOptions`' defaults, with one env; the report records them under ``physics``.

Build errors are not caught: a non-physical inertia tensor or a URDF ``<inertial><density>`` raises from
Genesis with its own message, which is the finding.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions
    from eden.options.entities import EntityOptions

Status = Literal["pass", "warn", "fail", "skip"]

#: Name the audited entity takes in the scene.
AUDIT_ENTITY_NAME = "asset"

#: Time ``settle`` allows, after the entity first reaches the plane, for the landing to damp out before the graded
#: :attr:`AuditThresholds.rest_window` may begin, in s. An unpowered robot folds and bounces for a few tenths of a
#: second; its overshoot then is not a property of the asset.
LANDING_SECONDS = 0.5

#: Standard gravity, for the fall time of :attr:`AuditThresholds.drop_height`, in m/s^2.
_GRAVITY = 9.81

#: File suffixes ``eden audit`` loads as a raw asset path: the ones ``Entity`` builds a rigid morph from
#: (``eden/entities/base.py``): URDF, MJCF (``.xml``), USD and the surface meshes of ``MESH_FILE_SUFFIXES``.
#: ``tests/utils/test_asset_audit.py`` pins the mesh part against that constant.
ASSET_FILE_SUFFIXES = (".urdf", ".xml", ".usd", ".usda", ".usdc", ".usdz", ".obj", ".stl", ".ply", ".glb", ".gltf")


@dataclass(frozen=True)
class AuditThresholds:
    """Limits the checks compare against. The defaults suit tabletop objects and robot arms in SI units."""

    #: Seconds simulated by ``settle`` and ``hold``.
    seconds: float = 2.0
    #: Gap between the entity's lowest point and the plane at the start of ``settle``, in m.
    drop_height: float = 0.05
    #: A movable link with collision geometry lighter than this warns, in kg.
    min_link_mass: float = 1e-4
    #: Heavier-to-lighter mass ratio across a DOF joint above which ``mass_ratio`` warns.
    max_mass_ratio: float = 100.0
    #: Deepest ground penetration ``settle`` accepts over the final :attr:`rest_window`, in m.
    max_penetration: float = 0.01
    #: Link linear speed treated as an explosion, in m/s.
    max_link_speed: float = 20.0
    #: Link angular speed treated as an explosion, in rad/s.
    max_link_ang_speed: float = 200.0
    #: Over the last :attr:`rest_window` seconds of ``settle``, link speed below this counts as at rest, in m/s.
    rest_speed: float = 0.05
    #: Over the last :attr:`rest_window` seconds of ``settle``, angular speed below this counts as at rest, in rad/s.
    rest_ang_speed: float = 0.5
    #: Length of the window at the end of ``settle`` that must be at rest, in s.
    rest_window: float = 0.25
    #: Largest distance a DOF may drift from its commanded default during ``hold``, in rad or m.
    max_hold_drift: float = 0.05
    #: DOF speed treated as an explosion during ``hold``, in rad/s or m/s.
    max_dof_speed: float = 50.0

    def __post_init__(self) -> None:
        not_positive = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if not getattr(self, f.name) > 0.0]
        if not_positive:
            raise ValueError(f"AuditThresholds: every limit must be a positive number; got {', '.join(not_positive)}.")
        if self.seconds < self.min_seconds:
            raise ValueError(
                f"AuditThresholds: seconds={self.seconds!r} is too short to grade settle: the drop from "
                f"drop_height={self.drop_height} m takes {self.fall_seconds:.2f} s, the landing is given "
                f"{LANDING_SECONDS} s to damp, and the graded rest_window={self.rest_window} s must come after both. "
                f"Pass seconds >= {self.min_seconds:.2f} (eden audit --seconds {math.ceil(self.min_seconds * 10) / 10}), "
                "or shorten rest_window."
            )

    @property
    def fall_seconds(self) -> float:
        """Time to fall :attr:`drop_height` from rest, in s."""
        return math.sqrt(2.0 * self.drop_height / _GRAVITY)

    @property
    def min_seconds(self) -> float:
        """Shortest :attr:`seconds` whose graded :attr:`rest_window` starts after the drop and the landing, in s."""
        return self.fall_seconds + LANDING_SECONDS + self.rest_window


@dataclass(frozen=True)
class AuditCheck:
    """The outcome of one check."""

    name: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditReport:
    """Every check run on one target, in the order they ran."""

    target: str
    checks: tuple[AuditCheck, ...]
    #: The physics settings the audit scene ran at: ``sim_dt``, ``sim_substeps``, ``integrator``.
    physics: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Whether any check failed."""
        return any(check.status == "fail" for check in self.checks)

    def check(self, name: str) -> AuditCheck:
        """The check called ``name``."""
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(f"no audit check named {name!r}; this report has {[c.name for c in self.checks]}")

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable form of the report."""
        return {
            "target": self.target,
            "failed": self.failed,
            "physics": dict(self.physics),
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self, path: str) -> None:
        """Write :meth:`as_dict` to ``path`` as indented JSON."""
        import json

        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=2)

    def render(self) -> str:
        """One line per check, status first, under a header naming the target and the physics."""
        width = max(len(check.name) for check in self.checks)
        physics = ", ".join(f"{key}={value}" for key, value in self.physics.items())
        lines = [f"Audit of {self.target}" + (f" ({physics})" if physics else "")]
        lines += [f"  {check.status.upper():<4}  {check.name:<{width}}  {check.message}" for check in self.checks]
        counts = {
            status: sum(check.status == status for check in self.checks) for status in ("pass", "warn", "fail", "skip")
        }
        lines.append("  " + ", ".join(f"{n} {status}" for status, n in counts.items() if n))
        return "\n".join(lines)


def options_for_file(path: str, *, fixed_base: bool | None = None) -> EntityOptions:
    """Entity options that load the asset file at ``path`` as it is, for auditing.

    ``fixed_base`` pins (``True``) or frees (``False``) the base. ``None`` reads an MJCF's own declaration (a
    top-level body with a free joint is free) and frees anything else, so a URDF arm needs ``fixed_base=True``.
    """
    from eden.options.entities import MESH_FILE_SUFFIXES, EntityOptions

    if not os.path.isfile(path):
        raise FileNotFoundError(f"eden audit: no such asset file: {path!r}.")
    if not path.lower().endswith(ASSET_FILE_SUFFIXES):
        raise ValueError(
            f"eden audit: {path!r} is not an asset file Eden loads; expected one of {', '.join(ASSET_FILE_SUFFIXES)}."
        )
    if fixed_base is None:
        fixed_base = path.lower().endswith(".xml") and not mjcf_has_free_root(path)
    # A fixed base is what ``hold`` runs on, and it needs the file's PD gains, which load only on an actuated entity.
    # A mesh is one rigid body, so only a description file (URDF, MJCF, USD) can be articulated or actuated.
    articulated = not path.lower().endswith(MESH_FILE_SUFFIXES)
    return EntityOptions(
        file=os.path.abspath(path),
        is_fixed_base=fixed_base,
        is_articulated=articulated,
        is_actuated=articulated and fixed_base,
    )


def mjcf_has_free_root(path: str) -> bool:
    """Whether a top-level body of the MJCF at ``path`` carries a free joint (``<freejoint/>`` or ``type="free"``)."""
    import xml.etree.ElementTree as ET

    worldbody = ET.parse(path).getroot().find("worldbody")
    if worldbody is None:
        return False
    for body in worldbody.findall("body"):
        if body.find("freejoint") is not None:
            return True
        if any(joint.get("type") == "free" for joint in body.findall("joint")):
            return True
    return False


def check_auditable(options, target: str) -> None:
    """Raise unless ``options`` builds one rigid entity: a composite rig or a non-rigid entity cannot be audited."""
    from eden.options.entities import CompositeEntityOptions, EntityOptions, PrimitiveOptions
    from eden.options.materials import RigidMaterialOptions

    if isinstance(options, CompositeEntityOptions):
        parts = ", ".join(repr(type(o).__name__) for o in options.entity_options().values())
        raise TypeError(
            f"eden audit: {target!r} is a composite rig, which builds several entities welded together. Audit "
            f"its arm and its gripper separately: {parts}."
        )
    if not isinstance(options, (EntityOptions, PrimitiveOptions)):
        raise TypeError(
            f"eden audit: {target!r} is a {type(options).__name__}; only rigid entities (EntityOptions or a "
            "primitive) can be audited."
        )
    material = getattr(options, "material", None)
    if material is not None and not isinstance(material, RigidMaterialOptions):
        raise TypeError(
            f"eden audit: {target!r} uses {type(material).__name__}; the audit scene is a rigid ground plane, so "
            "only entities with a RigidMaterialOptions (or no material) can be audited."
        )


def audit_entity(
    options: EntityOptions,
    *,
    env_options: EnvOptions | None = None,
    target: str | None = None,
    thresholds: AuditThresholds | None = None,
) -> AuditReport:
    """Build ``options`` alone on a ground plane and run every check on it.

    Eden must be initialized (``eden.init()``). ``options`` is not modified. Composite rigs are refused: audit
    the arm and the gripper as separate entities.

    Parameters
    ----------
    options : EntityOptions
        The entity to audit, as a task would declare it.
    env_options : EnvOptions, optional
        The physics to audit under, usually a task's ``config.env_options``: its ``sim_dt``, ``sim_substeps``,
        integrator and solver settings apply, with one env, no CUDA graph and no gradients. Defaults to
        ``EnvOptions()``.
    target : str, optional
        Label for the report. Defaults to the options class name.
    thresholds : AuditThresholds, optional
        Limits for the checks. Defaults to :class:`AuditThresholds`.

    Returns
    -------
    AuditReport
        One :class:`AuditCheck` per check, in the order they ran.
    """
    thresholds = thresholds or AuditThresholds()
    target = target or type(options).__name__
    check_auditable(options, target)

    env = _build_audit_env(options.model_copy(deep=True), env_options)
    try:
        checks = _run_checks(env, options, env_options, thresholds)
        physics = {
            "sim_dt": env.env_options.sim_dt,
            "sim_substeps": env.env_options.sim_substeps,
            "integrator": env.env_options.integrator,
        }
    finally:
        env.destroy()
    return AuditReport(target=target, checks=tuple(checks), physics=physics)


def _run_checks(env: EnvBase, options, env_options: EnvOptions | None, thresholds: AuditThresholds) -> list[AuditCheck]:
    entity = env.entities[AUDIT_ENTITY_NAME]
    checks = [
        _check_collision(entity),
        _check_mass(entity, thresholds),
        _check_mass_ratio(entity, thresholds),
        _check_armature(entity, options),
        _check_joint_limits(entity),
        _check_self_contact(env, entity, options, env_options),
    ]
    if entity.has_immobile_base:
        checks.append(AuditCheck("settle", "skip", "fixed base: nothing to drop"))
        checks.append(_check_hold(env, entity, thresholds))
        return checks
    if entity.geoms:
        checks.append(_check_settle(env, entity, thresholds))
    else:
        checks.append(AuditCheck("settle", "skip", "no collision geometry to land on (see collision)"))
    hint = " (pass --fixed-base to run it on an arm)" if _rotor_dofs(entity) else ""
    checks.append(AuditCheck("hold", "skip", f"free base: holding a pose needs a fixed base{hint}"))
    return checks


# --- scene ----------------------------------------------------------------------------------------


def _build_audit_env(
    options, env_options: EnvOptions | None = None, *, enable_neutral_collision: bool = False
) -> EnvBase:
    """One env, a ground plane and the entity, at ``env_options``' physics and the substep a closed-loop gripper needs.

    ``enable_neutral_collision=True`` keeps the self-collision pairs that overlap at the zero pose; otherwise the
    setting of ``env_options`` stands.
    """
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions
    from eden.options.entities import PlaneOptions

    # One env on whatever backend Eden was initialized with: a captured CUDA graph or gradients serve training only.
    update: dict[str, Any] = {"num_envs": 1, "cuda_graph_options": None, "requires_grad": False}
    if enable_neutral_collision:
        update["enable_neutral_collision"] = True
    env_options = (env_options or EnvOptions()).model_copy(deep=True, update=update)
    max_substep_dt = getattr(type(options), "max_substep_dt", None)
    if isinstance(max_substep_dt, float):
        env_options.sim_substeps = max(env_options.sim_substeps, math.ceil(env_options.sim_dt / max_substep_dt - 1e-9))
    env = EnvBase(
        env_options=env_options,
        scene_options=SceneOptions(plane=PlaneOptions(), **{AUDIT_ENTITY_NAME: options}),
        show_viewer=False,
    )
    env.build()
    env.reset()
    return env


def _rotor_joints(entity: RigidEntity) -> list:
    """The entity's hinge and slide joints, in Genesis' joint order."""
    import genesis as gs

    return [joint for joint in entity.joints if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]


def _rotor_dofs(entity: RigidEntity) -> list[int]:
    """Local DOF indices of the entity's hinge and slide joints."""
    return [dof for joint in _rotor_joints(entity) for dof in joint.dofs_idx_local]


def _steps(env: EnvBase, seconds: float) -> int:
    return max(1, math.ceil(seconds / env.scene.dt))


def _names(items, limit: int = 8) -> str:
    items = list(items)
    shown = ", ".join(str(item) for item in items[:limit])
    return shown + (f", ... ({len(items) - limit} more)" if len(items) > limit else "")


# --- static checks --------------------------------------------------------------------------------


def _check_collision(entity: RigidEntity) -> AuditCheck:
    n_geoms = len(entity.geoms)
    if n_geoms == 0:
        return AuditCheck(
            "collision",
            "fail",
            "no collision geometry: the entity passes through the ground and everything else. Check that the "
            "asset declares collision shapes and that the options do not set collision=False.",
        )
    return AuditCheck(
        "collision", "pass", f"{n_geoms} collision geoms on {entity.num_links} links", {"n_geoms": n_geoms}
    )


def _check_mass(entity: RigidEntity, thresholds: AuditThresholds) -> AuditCheck:
    masses = entity.get_links_inertial_mass()[0].tolist()
    movable = [(link.name, m) for link, m in zip(entity.links, masses) if not link.is_fixed and len(link.geoms) > 0]
    massless = [name for name, m in movable if m <= 0.0]
    light = [f"{name} ({m:.2e} kg)" for name, m in movable if 0.0 < m < thresholds.min_link_mass]
    details = {"total_mass": sum(masses), "links": dict(zip((link.name for link in entity.links), masses))}
    if massless:
        return AuditCheck(
            "mass",
            "fail",
            f"movable links with collision geometry and no mass: {_names(massless)}. Give them a <mass> in the "
            "asset, or set recompute_inertia=True to derive it from the collision geometry.",
            details,
        )
    if light:
        return AuditCheck(
            "mass",
            "warn",
            f"links lighter than {thresholds.min_link_mass} kg, which stiff contacts can destabilize: {_names(light)}",
            details,
        )
    return AuditCheck("mass", "pass", f"total {sum(masses):.4g} kg over {entity.num_links} links", details)


def _check_mass_ratio(entity: RigidEntity, thresholds: AuditThresholds) -> AuditCheck:
    masses = entity.get_links_inertial_mass()[0].tolist()
    offset = entity.link_start
    pairs = []
    for link in entity.links:
        parent = link.parent_idx - offset
        if parent < 0 or link.n_dofs == 0 or entity.links[parent].is_fixed:
            continue
        lo, hi = sorted((masses[link.idx_local], masses[parent]))
        if lo > 0.0:
            pairs.append((hi / lo, entity.links[parent].name, link.name))
    if not pairs:
        return AuditCheck("mass_ratio", "skip", "no DOF joint between two movable links")
    worst = max(pairs)
    over = [
        f"{parent}/{child} ({ratio:.0f}x)"
        for ratio, parent, child in sorted(pairs, reverse=True)
        if ratio > thresholds.max_mass_ratio
    ]
    details = {"worst_ratio": worst[0], "worst_pair": [worst[1], worst[2]]}
    if over:
        return AuditCheck(
            "mass_ratio",
            "warn",
            f"mass ratio across a joint above {thresholds.max_mass_ratio:.0f}x, which slows convergence and can "
            f"jitter: {_names(over)}",
            details,
        )
    return AuditCheck("mass_ratio", "pass", f"largest ratio {worst[0]:.1f}x ({worst[1]}/{worst[2]})", details)


def _check_armature(entity: RigidEntity, options) -> AuditCheck:
    """Name the hinge/slide DOFs whose armature came from ``default_armature`` rather than the asset or the options."""
    from eden.utils.string import resolve_matching_names_values

    default = getattr(options, "default_armature", None)
    joints = [joint for joint in _rotor_joints(entity) if joint.n_dofs == 1]
    if not joints:
        return AuditCheck("armature", "skip", "no hinge or slide joints")
    built = entity.get_dofs_armature([joint.dofs_idx_local[0] for joint in joints])[0].tolist()
    configured_idx, _, _ = resolve_matching_names_values(
        dict(getattr(options, "default_dofs_armature", {}) or {}),
        [joint.name for joint in joints],
        literal_names=entity.joints_name,
    )
    configured = {joints[i].name for i in configured_idx}
    filled = [
        joint.name
        for joint, value in zip(joints, built)
        if joint.name not in configured
        and not joint.desc.dofs_armature.any()
        and default
        and math.isclose(value, default, rel_tol=1e-6)
    ]
    details = {"armature": {joint.name: value for joint, value in zip(joints, built)}, "filled": filled}
    if filled:
        return AuditCheck(
            "armature",
            "warn",
            f"{len(filled)} of {len(joints)} DOFs carry the default_armature={default} Genesis fills in when the "
            f"asset authors none: {_names(filled)}. Set default_dofs_armature from the source asset or datasheet, "
            "or default_armature=0.0 (docs/guides/entities.md, 'Armature defaults and the URDF 0.1 trap').",
            details,
        )
    return AuditCheck("armature", "pass", f"{len(joints)} DOFs, armature from the asset or the options", details)


def _check_joint_limits(entity: RigidEntity) -> AuditCheck:
    dofs = _rotor_dofs(entity)
    if not dofs:
        return AuditCheck("joint_limits", "skip", "no hinge or slide joints")
    names = [joint.name for joint in _rotor_joints(entity) for _ in joint.dofs_idx_local]
    lower, upper = (bound[0].tolist() for bound in entity.get_dofs_limit(dofs))
    empty = [f"{n} [{lo:.3g}, {hi:.3g}]" for n, lo, hi in zip(names, lower, upper) if lo >= hi]
    unlimited = [n for n, lo, hi in zip(names, lower, upper) if math.isinf(lo) or math.isinf(hi)]
    details = {"limits": {n: [lo, hi] for n, lo, hi in zip(names, lower, upper)}, "unlimited": unlimited}
    if empty:
        return AuditCheck(
            "joint_limits",
            "warn",
            f"DOFs whose range is empty, so they cannot move: {_names(empty)}. A fixed joint authored as a hinge "
            "should be a fixed joint.",
            details,
        )
    note = f"; unlimited: {_names(unlimited)}" if unlimited else ""
    return AuditCheck("joint_limits", "pass", f"{len(dofs)} DOFs with a non-empty range{note}", details)


# --- dynamic checks -------------------------------------------------------------------------------


def _link_pair(env: EnvBase, entity: RigidEntity, geom_a: int, geom_b: int) -> tuple[str, str] | None:
    """The sorted link names of two scene-global geoms, if both are on different links of ``entity``; else ``None``."""
    geoms = env.rigid_solver.geoms
    link_a, link_b = geoms[geom_a].link, geoms[geom_b].link
    start, end = entity.link_start, entity.link_end
    if link_a.idx == link_b.idx or not (start <= link_a.idx < end and start <= link_b.idx < end):
        return None
    return tuple(sorted((link_a.name, link_b.name)))


def _link_pairs(env: EnvBase, entity: RigidEntity, geom_pairs) -> set[tuple[str, str]]:
    """Name pairs of two different links of ``entity``, from an ``(n, 2)`` array of scene-global geom indices."""
    return {pair for a, b in geom_pairs.tolist() if (pair := _link_pair(env, entity, a, b)) is not None}


def _collision_pairs(env: EnvBase):
    """The scene's collision candidate geom pairs, after Genesis' adjacency and neutral-overlap filters.

    Genesis keeps them only in the private ``Collider._valid_collision_pairs``; it logs, but does not store, the
    pairs it dropped. ``tests/utils/test_asset_audit.py`` pins the attribute.
    """
    return env.rigid_solver.collider._valid_collision_pairs


def _check_self_contact(env: EnvBase, entity: RigidEntity, options, env_options: EnvOptions | None) -> AuditCheck:
    """Link pairs touching at the default pose, and the shape pairs Genesis dropped from collision at build.

    Genesis drops, for good, every self-collision shape pair it finds overlapping at the asset's zero pose
    (``EnvOptions.enable_neutral_collision=False``, the default): those shapes pass through each other in every task
    that keeps the default. The neutral filter is the only difference between ``env`` and a second build with
    ``enable_neutral_collision=True``, and both number their geoms alike, so the candidate pairs only that build keeps
    are exactly the dropped ones. They are reported per link pair, as dropped of all candidate shape pairs.
    """
    if entity.num_links < 2:
        return AuditCheck("self_contact", "skip", "single link")
    env.reset()
    live = sorted(_link_pairs(env, entity, entity.detect_collision()))
    probe = _build_audit_env(options.model_copy(deep=True), env_options, enable_neutral_collision=True)
    try:
        if len(probe.rigid_solver.geoms) != len(env.rigid_solver.geoms):
            raise RuntimeError(
                f"eden audit: the probe build has {len(probe.rigid_solver.geoms)} geoms and the audited build "
                f"{len(env.rigid_solver.geoms)}; their collision pairs cannot be compared. This is a bug in the audit."
            )
        candidates = {tuple(pair) for pair in _collision_pairs(probe).tolist()}
    finally:
        probe.destroy()
    kept = {tuple(pair) for pair in _collision_pairs(env).tolist()}
    total: dict[tuple[str, str], int] = {}
    dropped_count: dict[tuple[str, str], int] = {}
    for geom_a, geom_b in candidates:
        pair = _link_pair(env, entity, geom_a, geom_b)
        if pair is None:
            continue
        total[pair] = total.get(pair, 0) + 1
        if (geom_a, geom_b) not in kept:
            dropped_count[pair] = dropped_count.get(pair, 0) + 1
    dropped = sorted(dropped_count)

    details = {
        "pairs": [list(p) for p in live],
        "dropped_pairs": [list(p) for p in dropped],
        "dropped_shape_pairs": {f"{a}/{b}": [dropped_count[(a, b)], total[(a, b)]] for a, b in dropped},
    }
    note = ""
    if dropped:
        described = (f"{a}/{b} ({dropped_count[(a, b)]} of {total[(a, b)]} shape pairs)" for a, b in dropped)
        note = (
            f" Genesis also dropped shape pairs that overlap at the asset's zero pose, so those shapes never "
            f"collide: {_names(described)} (EnvOptions.enable_neutral_collision=True keeps them)."
        )
    if live:
        return AuditCheck(
            "self_contact",
            "warn",
            f"links in contact at the default pose: {_names(f'{a}/{b}' for a, b in live)}. Overlapping collision "
            f"shapes push the joints apart every step; trim the shapes or move the default pose.{note}",
            details,
        )
    return AuditCheck("self_contact", "pass", f"no link touches another at the default pose.{note}", details)


def _check_settle(env: EnvBase, entity: RigidEntity, thresholds: AuditThresholds) -> AuditCheck:
    """Drop the free entity onto the plane and watch it come to rest."""
    import torch

    env.reset()
    lowest = float(entity.get_AABB()[0, 0, 2])
    pos = entity.get_pos().clone()
    pos[:, 2] += thresholds.drop_height - lowest
    entity.set_pos(pos)
    entity.zero_all_dofs_velocity()

    plane = env.entities["plane"]
    n_steps = _steps(env, thresholds.seconds)
    rest_steps = min(n_steps, _steps(env, thresholds.rest_window))
    max_speed = max_ang_speed = peak_penetration = rest_penetration = 0.0
    tail_speed = tail_ang_speed = 0.0
    for step in range(n_steps):
        env.scene.step()
        lin = entity.get_links_vel()[0].norm(dim=-1).max().item()
        ang = entity.get_links_ang()[0].norm(dim=-1).max().item()
        if not (math.isfinite(lin) and math.isfinite(ang) and torch.isfinite(entity.get_pos()).all()):
            env.reset()
            return AuditCheck("settle", "fail", f"state became non-finite at step {step} ({step * env.scene.dt:.3f} s)")
        max_speed, max_ang_speed = max(max_speed, lin), max(max_ang_speed, ang)
        contacts = entity.get_contacts(with_entity=plane)
        depth = contacts["penetration"][0]
        if "valid_mask" in contacts:
            depth = depth[contacts["valid_mask"][0]]
        deepest = depth.max().item() if depth.numel() else 0.0
        peak_penetration = max(peak_penetration, deepest)
        # Graded only once the drop is over: an unpowered robot landing at several m/s overshoots for a few
        # steps on any contact model, which says nothing about the asset.
        if step >= n_steps - rest_steps:
            tail_speed, tail_ang_speed = max(tail_speed, lin), max(tail_ang_speed, ang)
            rest_penetration = max(rest_penetration, deepest)
    top = float(entity.get_AABB()[0, 1, 2])
    env.reset()

    details = {
        "seconds": thresholds.seconds,
        "max_link_speed": max_speed,
        "max_link_ang_speed": max_ang_speed,
        "peak_penetration": peak_penetration,
        "rest_penetration": rest_penetration,
        "final_speed": tail_speed,
        "final_ang_speed": tail_ang_speed,
    }
    if top < 0.0:
        return AuditCheck("settle", "fail", "fell through the ground plane", details)
    if max_speed > thresholds.max_link_speed or max_ang_speed > thresholds.max_link_ang_speed:
        return AuditCheck(
            "settle",
            "fail",
            f"exploded: peak link speed {max_speed:.3g} m/s, {max_ang_speed:.3g} rad/s (limits "
            f"{thresholds.max_link_speed} m/s, {thresholds.max_link_ang_speed} rad/s). Check mass, inertia and "
            "the collision shapes for overlaps.",
            details,
        )
    if rest_penetration > thresholds.max_penetration:
        return AuditCheck(
            "settle",
            "fail",
            f"rests {rest_penetration * 1000:.1f} mm deep in the ground over the last {thresholds.rest_window} s "
            f"(limit {thresholds.max_penetration * 1000:.0f} mm). Thin or concave collision shapes, or a very heavy "
            "body on soft contacts, do this.",
            details,
        )
    if tail_speed > thresholds.rest_speed or tail_ang_speed > thresholds.rest_ang_speed:
        return AuditCheck(
            "settle",
            "warn",
            f"still moving after {thresholds.seconds} s: {tail_speed:.3g} m/s, {tail_ang_speed:.3g} rad/s over the "
            f"last {thresholds.rest_window} s. A round body may still be rolling and an unpowered robot may "
            "still be folding; jitter in place points at contact or inertia problems.",
            details,
        )
    return AuditCheck(
        "settle",
        "pass",
        f"at rest after {thresholds.seconds} s, {rest_penetration * 1000:.1f} mm deep (peak {max_speed:.2f} m/s and "
        f"{peak_penetration * 1000:.1f} mm on impact)",
        details,
    )


def _check_hold(env: EnvBase, entity: RigidEntity, thresholds: AuditThresholds) -> AuditCheck:
    """Command the default pose with the built PD gains and measure how far each DOF drifts."""
    import torch

    dofs = _rotor_dofs(entity)
    if not dofs:
        return AuditCheck("hold", "skip", "no hinge or slide joints")
    names = [joint.name for joint in _rotor_joints(entity) for _ in joint.dofs_idx_local]
    env.reset()
    kp = entity.get_dofs_kp(dofs)[0]
    driven = [i for i, gain in enumerate(kp.tolist()) if gain > 0.0]
    if not driven:
        return AuditCheck(
            "hold",
            "skip",
            "every DOF has kp = 0, so nothing holds a pose. Set default_dofs_kp / default_dofs_kd to audit the gains.",
        )
    dofs = [dofs[i] for i in driven]
    names = [names[i] for i in driven]
    target = entity.get_dofs_pos(dofs).clone()
    entity.control_dofs_pos(target, dofs)

    max_drift = torch.zeros(len(dofs), device=target.device)
    max_speed = 0.0
    for step in range(_steps(env, thresholds.seconds)):
        env.scene.step()
        pos, vel = entity.get_dofs_pos(dofs)[0], entity.get_dofs_vel(dofs)[0]
        if not (torch.isfinite(pos).all() and torch.isfinite(vel).all()):
            env.reset()
            return AuditCheck("hold", "fail", f"state became non-finite at step {step} ({step * env.scene.dt:.3f} s)")
        max_drift = torch.maximum(max_drift, (pos - target[0]).abs())
        max_speed = max(max_speed, vel.abs().max().item())
    env.reset()

    drift = dict(zip(names, max_drift.tolist()))
    details = {"seconds": thresholds.seconds, "max_drift": drift, "max_dof_speed": max_speed, "n_driven": len(dofs)}
    if max_speed > thresholds.max_dof_speed:
        return AuditCheck(
            "hold",
            "fail",
            f"exploded: a DOF reached {max_speed:.3g} (limit {thresholds.max_dof_speed}) while holding its default "
            "pose. Check armature, kd and the inertia of the distal links.",
            details,
        )
    drifting = [
        f"{n} ({d:.3g})" for n, d in sorted(drift.items(), key=lambda kv: -kv[1]) if d > thresholds.max_hold_drift
    ]
    if drifting:
        return AuditCheck(
            "hold",
            "warn",
            f"DOFs drift more than {thresholds.max_hold_drift} from the commanded default pose: {_names(drifting)}. "
            "The gains do not hold the arm against gravity, or the default pose sits on a joint limit.",
            details,
        )
    worst = max(drift.values())
    return AuditCheck("hold", "pass", f"{len(dofs)} DOFs held within {worst:.2g} for {thresholds.seconds} s", details)
