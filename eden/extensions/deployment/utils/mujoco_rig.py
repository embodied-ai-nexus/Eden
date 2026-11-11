"""One MuJoCo world holding a whole composite rig -- every part, welded as the env welds them -- and its props.

The engine half of :class:`~eden.extensions.deployment.mujoco_composite.MujocoCompositeDeployment`,
kept free of any Eden ``env`` so it can be built from asset paths and poses alone, as
:class:`~eden.extensions.deployment.utils.mujoco_sim.MujocoSim` is for one robot.

Each part is loaded from its own asset and attached to the world (the root) or to its parent
part's body (a welded part) at a given pose, under a ``"<part>/"`` name prefix so two assets that
both name a ``base_link`` cannot collide. The assets' own actuators are dropped: every joint is
driven by the same per-substep PD law ``MujocoSim`` applies (``qfrc_applied``), with the gains each
part's command carries. Props -- a fixed box (a table), a free body from a URDF/MJCF (a tool) --
are added as bodies of their own, and their poses are read back as a mocap system would report
them.

Commands are buffered per part (:meth:`set_command`) and the physics advances only in
:meth:`step`, so every part's command of a tick acts on the same substeps -- the rig moves as one
robot, whichever order its parts' sends arrived in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import eden as en
from eden.extensions.deployment.utils.mujoco_sim import MujocoFloor, _resolve_meshdir

_IMU_SITE = "__eden_imu_site__"
_IMU_GYRO = "__eden_imu_gyro__"
_IMU_ACC = "__eden_imu_acc__"


@dataclass
class RigPart:
    """One jointed part of the rig: what to load, where it sits, and the physics its env entity runs with.

    ``pos`` / ``quat`` (wxyz) place the asset's root body: in the world for the root part, in the
    frame of ``parent``'s body ``parent_body`` for a welded part. ``armature`` / ``damping`` are
    per DOF in ``dofs_name`` order; ``link_friction`` maps a body name to its geoms' sliding
    friction, over ``friction`` for the rest. ``link_sol_params`` maps a body name to the contact
    parameters of each of its geoms in the env, an ``(n_geoms, 7)`` array of Genesis' ``(timeconst,
    dampratio, dmin, dmax, width, mid, power)`` -- MuJoCo's ``solref`` and ``solimp`` in one -- as
    stated, before any floor (see :meth:`MujocoRig._apply_contacts`); a body it does not name keeps
    its asset's.
    """

    name: str
    asset_path: str
    dofs_name: list[str]
    pos: np.ndarray
    quat: np.ndarray
    parent: str | None = None
    parent_body: str | None = None
    floating: bool = False
    armature: np.ndarray | None = None
    damping: np.ndarray | None = None
    gravity_compensation: float = 0.0
    friction: float | None = None
    link_friction: dict[str, float] = field(default_factory=dict)
    link_sol_params: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RigProp:
    """A non-robot body of the scene: a box (``half_size``) or a rigid body loaded from a URDF / MJCF ``asset_path``.

    ``free`` is whether the env built it with a free joint; the asset's own free joint, if any, is
    replaced by that. ``inertial`` is the whole rigid body's inertia as the env built it -- mass, centre
    of mass and 3x3 inertia tensor about it, both in the frame of the asset's root body -- which a free
    prop carries on its root body in MuJoCo, every other body of it massless. MuJoCo would otherwise
    derive it from geometry at its own 1000 kg/m^3 where Genesis uses the material's ``rho`` (200 by
    default): a box would weigh 5x. ``sol_params`` are its geoms' contact parameters in the env, one
    row per geom, as :attr:`RigPart.link_sol_params`.
    """

    name: str
    pos: np.ndarray
    quat: np.ndarray
    free: bool
    asset_path: str | None = None
    half_size: np.ndarray | None = None
    friction: float | None = None
    inertial: tuple[float, np.ndarray, np.ndarray] | None = None
    sol_params: np.ndarray | None = None


def _sol_params_rows(sol_params) -> np.ndarray:
    """Genesis' contact parameters as an ``(n, 7)`` float64 array, one row per geom; refused when not 7 wide."""
    rows = np.asarray(sol_params, dtype=np.float64)
    rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
    if rows.ndim != 2 or rows.shape[1] != 7 or rows.shape[0] == 0:
        raise ValueError(
            "sol_params must be one or more rows of 7 values (timeconst, dampratio, dmin, dmax, width, mid, power); "
            f"got shape {np.asarray(sol_params).shape}."
        )
    return rows


def _solref_solimp(sol_params) -> tuple[list[float], list[float]]:
    """One geom's 7 Genesis contact parameters as MuJoCo's ``solref`` (timeconst, dampratio) and ``solimp`` (5)."""
    rows = _sol_params_rows(sol_params)
    if rows.shape[0] != 1:
        raise ValueError(f"_solref_solimp converts one geom's sol_params; got {rows.shape[0]} rows.")
    values = [float(v) for v in rows[0]]
    return values[:2], values[2:]


def _same_rows(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two sets of parameter rows are the same set, to float32 precision (Genesis stores them in float32)."""

    def covered(rows: np.ndarray, by: np.ndarray) -> bool:
        return all(np.isclose(row, by, rtol=1e-5, atol=1e-9).all(axis=1).any() for row in rows)

    return covered(a, b) and covered(b, a)


#: MuJoCo's ``mjMINVAL``: its compiler refuses a full inertia whose smallest eigenvalue is below this.
_MJ_MINVAL = 1e-15


def _inertia_refused(moments: np.ndarray, *, full: bool) -> str | None:
    """Why MuJoCo's compiler refuses a body with these principal moments, or ``None`` when it compiles it.

    Its two checks on an inertia an asset states: a full tensor's eigenvalues must reach
    ``mjMINVAL`` ("inertia must have positive eigenvalues"), and any body's principal moments must
    satisfy the triangle inequality, strictly as compared ("inertia must satisfy A + B >= C").
    """
    a, b, c = np.sort(moments)
    if full and a < _MJ_MINVAL:
        return "not positive definite"
    if a + b < c:
        return "A + B < C"
    return None


def _repair_inertias(spec, asset_path: str) -> None:
    """Replace every stated inertia MuJoCo's compiler refuses with an isotropic one of the same trace.

    Genesis builds them; MuJoCo refuses the whole model (see :func:`_inertia_refused`). The eigenvalue
    test is MuJoCo's threshold, not ``> 0``: a rank-1 tensor's zero eigenvalues come out of LAPACK
    as +-1e-27, signed by the machine. The RoboTera XHand1 URDFs give a dozen massless dummy links
    (``*_ee_link``, mg-scale) every inertia entry ``1e-11``: a rank-1 tensor. A URDF states every
    inertia as a full tensor (``fullinertia``), an MJCF ``diaginertia`` as principal moments
    (``inertia``). The isotropic replacement keeps the mass, the centre of mass and the trace -- what
    MuJoCo's own ``balanceinertia`` does for the triangle inequality -- so it changes nothing those
    bodies contribute at that scale.

    Those XHand links hang on fixed joints. Genesis merges a fixed link into its parent (all but a
    robot's ``links_to_keep``, there the fingertip frames, which stay welded links), and MuJoCo welds
    the body to its parent: in both engines its inertia only adds to its parent's rigid body, so the
    warning says which repaired bodies are welded. The asset is what to fix; this keeps it running.
    """
    repaired, welded = {}, []
    for body in spec.bodies:
        full = np.asarray(body.fullinertia, dtype=np.float64)
        if full.size == 6 and np.isfinite(full).all():
            ixx, iyy, izz, ixy, ixz, iyz = full
            tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
            why, trace = _inertia_refused(np.linalg.eigvalsh(tensor), full=True), float(np.trace(tensor))
        elif body.explicitinertial:
            moments = np.asarray(body.inertia, dtype=np.float64)
            why, trace = _inertia_refused(moments, full=False), float(moments.sum())
        else:
            continue  # derived from the geoms by MuJoCo itself
        if why is None:
            continue
        moment = max(trace / 3.0, 1e-12)
        body.fullinertia = [float("nan")] * 6
        body.inertia = [moment, moment, moment]
        repaired[body.name] = why
        if not body.joints:
            welded.append(body.name)
    if repaired:
        weld = (
            "all welded to their parent, so each only adds to its parent's rigid body, in Genesis too"
            if len(welded) == len(repaired)
            else f"{len(welded)} welded to their parent; the others ({sorted(set(repaired) - set(welded))}) move "
            "on joints of their own, where the change is the body's own rotational inertia"
        )
        en.logger.warning(
            f"[deploy] {os.path.basename(asset_path)}: MuJoCo cannot compile the inertia of {repaired}; each is "
            f"replaced by an isotropic one of the same mass and trace ({weld}). To run on the asset's own, give "
            "those links an inertia MuJoCo accepts in the asset."
        )


def _absolutize_meshes(spec, asset_path: str) -> None:
    """Make every mesh path absolute, so attaching the spec elsewhere cannot re-root them under another meshdir.

    Each path is resolved where it exists: under the asset's ``meshdir``, beside the asset, or by
    file name anywhere below it. The first two disagree for an asset that declares a ``meshdir``
    *and* writes it into its mesh paths (the SharPa Wave URDF: ``meshdir="assets"`` with
    ``assets/visual/...``), which plain MuJoCo cannot compile -- Genesis resolves it beside the asset.
    """
    root = os.path.dirname(os.path.abspath(asset_path))
    meshdir = os.path.join(root, spec.meshdir) if spec.meshdir else root
    by_name: dict[str, str] | None = None
    for mesh in spec.meshes:
        if not mesh.file or os.path.isabs(mesh.file):
            continue
        found = next(
            (p for p in (os.path.join(meshdir, mesh.file), os.path.join(root, mesh.file)) if os.path.isfile(p)), None
        )
        if found is None:
            if by_name is None:
                by_name = {name: os.path.join(d, name) for d, _, names in os.walk(root) for name in names}
            found = by_name.get(os.path.basename(mesh.file))
        if found is None:
            raise FileNotFoundError(
                f"Mesh {mesh.file!r} of {asset_path} is neither under its meshdir {meshdir!r}, beside it, nor anywhere "
                f"below {root!r}."
            )
        mesh.file = os.path.normpath(found)
    spec.meshdir = ""


def _at_frame_origin(body) -> None:
    """Clear ``body``'s own pose in its asset: the frame it is attached under carries its whole world pose.

    The poses this rig is given are where the env has each root body, the asset's own offset of it
    (a ``<body pos="0 0 1">`` at the top of an MJCF) included -- keeping that offset too would apply it twice.
    """
    body.pos = [0.0, 0.0, 0.0]
    body.quat = [1.0, 0.0, 0.0, 0.0]
    body.alt.type = type(body.alt.type).mjORIENTATION_QUAT  # the quaternion, not an euler / axis-angle spec


#: MuJoCo's STL decoder refuses a mesh with more faces than this.
_STL_MAX_FACES = 200_000


def _mujoco_decodes(path: str) -> bool:
    """Whether MuJoCo can load mesh ``path``: a binary STL of at most 200k faces, or any other format it reads.

    MuJoCo rejects an ASCII STL and a binary one over its face limit, and a binary STL states its
    face count in its 84-byte header -- so this reads the header, not the mesh.
    """
    if not path.lower().endswith(".stl"):
        return True
    with open(path, "rb") as f:
        header = f.read(84)
    if len(header) < 84:
        return False
    faces = int.from_bytes(header[80:84], "little")
    binary = os.path.getsize(path) == 84 + 50 * faces
    return binary and 1 <= faces <= _STL_MAX_FACES


def _mesh_name(mesh) -> str:
    """A mesh's name as geoms reference it: its own, or -- unnamed, as an MJCF ``<mesh file=...>`` is -- its file's stem."""
    return mesh.name or os.path.splitext(os.path.basename(mesh.file))[0]


def _load(asset_path: str):
    """``asset_path`` as a spec to attach: mesh paths absolute, visual meshes MuJoCo cannot decode dropped.

    A geom that collides with nothing (``contype = conaffinity = 0``: a URDF's ``<visual>``, an
    MJCF's ``visual`` class) takes no part in the simulation but is what the viewer draws -- the
    iiwa's 15 meshes; without them only its collision primitives remain, in a group the viewer
    hides. So it stays, made massless (it must not change a body's inertia where an asset derives
    that from its geoms), unless MuJoCo cannot load its mesh at all (the SharPa Wave's visual STLs
    are over MuJoCo's 200k-face limit): then it goes, with the meshes only it used.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(asset_path)
    _resolve_meshdir(spec, asset_path)
    _absolutize_meshes(spec, asset_path)
    undecodable = {_mesh_name(mesh) for mesh in spec.meshes if mesh.file and not _mujoco_decodes(mesh.file)}
    for geom in list(spec.geoms):
        if geom.contype == 0 and geom.conaffinity == 0:
            if geom.meshname in undecodable:
                spec.delete(geom)
            else:
                geom.density = 0.0
    used = {geom.meshname for geom in spec.geoms if geom.meshname}
    for mesh in list(spec.meshes):
        if _mesh_name(mesh) not in used:
            spec.delete(mesh)
    if undecodable & used:
        bad = sorted(mesh.file for mesh in spec.meshes if _mesh_name(mesh) in undecodable & used)
        raise ValueError(
            f"{asset_path}: MuJoCo cannot decode the colliding mesh(es) {bad} (an ASCII STL, or over its "
            f"{_STL_MAX_FACES}-face limit). Decimate them, or convert them to binary STL / OBJ."
        )
    return spec


def root_body_name(asset_path: str) -> str:
    """The name of the single body under ``asset_path``'s ``<worldbody>`` -- the body a rig attaches it by."""
    import mujoco

    bodies = mujoco.MjSpec.from_file(asset_path).worldbody.bodies
    if len(bodies) != 1:
        raise ValueError(
            f"{asset_path} has {len(bodies)} bodies under <worldbody> ({[b.name for b in bodies]}); a rig part or prop "
            "must be one kinematic tree under a single root body."
        )
    return bodies[0].name


class MujocoRig:
    """A composite rig and its props in one MuJoCo model, driven part by part by a joint-space PD.

    Parameters
    ----------
    parts:
        The rig's jointed parts, every welded part after the part it hangs off.
    props:
        Scene bodies to simulate alongside the rig.
    sim_dt:
        Integration timestep; ``None`` keeps MuJoCo's default.
    integrator, cone:
        MuJoCo integrator (``"implicitfast"``, ``"implicit"``, ``"euler"``, ``"rk4"``) and friction
        cone (``"pyramidal"``, ``"elliptic"``).
    floor:
        The ground plane to add (:class:`~eden.extensions.deployment.utils.mujoco_sim.MujocoFloor`, by
        default MuJoCo's at ``z = 0``); ``None`` adds none.
    gravity:
        World gravity, m/s^2.
    """

    def __init__(
        self,
        parts: list[RigPart],
        props: list[RigProp] = (),
        *,
        sim_dt: float | None = None,
        integrator: str = "implicitfast",
        cone: str = "elliptic",
        floor: MujocoFloor | None = MujocoFloor(),  # frozen, so one shared default is safe
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        if not parts:
            raise ValueError("MujocoRig needs at least one part.")
        self.parts = {part.name: part for part in parts}
        self.props = {prop.name: prop for prop in props}
        self.root = parts[0].name
        self.floor = floor
        spec = mujoco.MjSpec()
        spec.option.gravity = list(gravity)
        spec.option.integrator = _enum(mujoco.mjtIntegrator, "mjINT_", integrator)
        spec.option.cone = _enum(mujoco.mjtCone, "mjCONE_", cone)
        if sim_dt is not None:
            spec.option.timestep = float(sim_dt)
        if floor is not None:
            floor.add_to(spec)
        for part in parts:
            self._attach_part(spec, part)
        # Every joint is driven by the PD in :meth:`step`; an asset's own actuators (the iiwa MJCF's
        # position servos) would pull its joints to ``ctrl = 0`` on top of it.
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        for prop in props:
            self._add_prop(spec, prop)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self._apply_contacts()
        self._build_index_maps()
        # The quantities MuJoCo derives at compile from what was just edited (subtree masses, the
        # constraint solver's per-DOF weights) are recomputed from it.
        mujoco.mj_setConst(self.model, self.data)
        self._armature0 = self.model.dof_armature.copy()
        n = len(self._v_idx)
        self._target_pos = np.zeros(n)
        self._target_vel = np.zeros(n)
        self._kp = np.zeros(n)
        self._kd = np.zeros(n)
        self._tau_ff = np.zeros(n)
        self._last_tau = np.zeros(n)

    # -- construction --------------------------------------------------------

    def _attach_part(self, spec, part: RigPart) -> None:
        mujoco = self._mujoco
        child = _load(part.asset_path)
        _repair_inertias(child, part.asset_path)
        bodies = child.worldbody.bodies
        if len(bodies) != 1:
            raise ValueError(
                f"Part '{part.name}' ({part.asset_path}) has {len(bodies)} bodies under <worldbody> "
                f"({[b.name for b in bodies]}); a rig part must be one kinematic tree under a single root body."
            )
        root = bodies[0]
        _at_frame_origin(root)
        if not part.floating:
            # Welded (or a fixed base): the asset's own free joint, if it declares one, is the weld's to replace.
            for joint in list(root.joints):
                if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                    child.delete(joint)
        if part.floating:
            if part.parent is not None:
                raise ValueError(f"Part '{part.name}' is welded to '{part.parent}' and cannot also float free.")
            if not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in root.joints):
                root.add_freejoint()
            root.add_site(name=_IMU_SITE, pos=[0.0, 0.0, 0.0])
        if part.parent is None:
            frame = spec.worldbody.add_frame(pos=list(part.pos), quat=list(part.quat))
        else:
            parent = spec.body(f"{part.parent}/{part.parent_body}")
            if parent is None:
                raise ValueError(
                    f"Part '{part.name}' welds to body '{part.parent_body}' of part '{part.parent}', which its asset "
                    f"does not have. Bodies of '{part.parent}': {self.part_bodies(spec, part.parent)}."
                )
            frame = parent.add_frame(pos=list(part.pos), quat=list(part.quat))
        spec.attach(child, prefix=f"{part.name}/", frame=frame)
        prefix = f"{part.name}/"
        for body in spec.bodies:
            if body.name.startswith(prefix):
                body.gravcomp = float(part.gravity_compensation)
        if part.floating:
            site = spec.site(f"{prefix}{_IMU_SITE}")
            for name, kind in (
                (_IMU_GYRO, mujoco.mjtSensor.mjSENS_GYRO),
                (_IMU_ACC, mujoco.mjtSensor.mjSENS_ACCELEROMETER),
            ):
                spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)

    @staticmethod
    def part_bodies(spec, part: str) -> list[str]:
        prefix = f"{part}/"
        return sorted(b.name[len(prefix) :] for b in spec.bodies if b.name.startswith(prefix))

    def _add_prop(self, spec, prop: RigProp) -> None:
        mujoco = self._mujoco
        if prop.asset_path is None:
            body = spec.worldbody.add_body(name=prop.name, pos=list(prop.pos), quat=list(prop.quat))
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(prop.half_size))
            if prop.free:
                body.add_freejoint()
            self._set_prop_inertial(spec, prop, root=prop.name)
            return
        child = _load(prop.asset_path)
        _repair_inertias(child, prop.asset_path)
        bodies = child.worldbody.bodies
        if len(bodies) != 1:
            raise ValueError(f"Prop '{prop.name}' ({prop.asset_path}) must have a single root body; has {len(bodies)}.")
        _at_frame_origin(bodies[0])
        # Free or fixed as the env built it, whatever the asset says: its own free joint (an MJCF object's
        # ``<freejoint/>``) would float a prop the env holds still, or make a free one's a second free joint.
        for joint in list(bodies[0].joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                child.delete(joint)
        if prop.free:
            bodies[0].add_freejoint()
        root = f"{prop.name}/{bodies[0].name}"
        frame = spec.worldbody.add_frame(pos=list(prop.pos), quat=list(prop.quat))
        spec.attach(child, prefix=f"{prop.name}/", frame=frame)
        self._set_prop_inertial(spec, prop, root=root)

    @staticmethod
    def _set_prop_inertial(spec, prop: RigProp, root: str) -> None:
        """Give a free prop the env's inertia, explicitly, before compile: all of it on its root body ``root``.

        A prop is one rigid body (no joints of its own), so its root body can carry the whole of it;
        its other, welded bodies are made massless. Set in the spec rather than on the compiled model,
        which MuJoCo has already specialised to the inertial frames the asset gave.
        """
        if not prop.free or prop.inertial is None:
            return
        mass, com, inertia = prop.inertial
        inertia = 0.5 * (np.asarray(inertia, dtype=np.float64) + np.asarray(inertia, dtype=np.float64).T)
        for body in spec.bodies:
            if body.name != root and body.name.startswith(f"{prop.name}/"):
                body.explicitinertial = True
                body.mass = 0.0
                body.inertia = [0.0, 0.0, 0.0]
        body = spec.body(root)
        body.explicitinertial = True
        body.mass = float(mass)
        body.ipos = [float(v) for v in com]
        # The tensor below is in the body frame. MuJoCo would still turn it by the asset's own
        # inertial orientation (a URDF ``<inertial><origin rpy>``), so that goes.
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        body.ialt.type = type(body.ialt.type).mjORIENTATION_QUAT
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]

    def _apply_contacts(self) -> None:
        """Every geom's contact as the env simulates it: sliding friction, solver parameters, and how a pair mixes them.

        A part geom takes its link's friction (``link_friction``, else the part's ``friction``) and
        solver parameters (``link_sol_params``); a body with no joint of its own (a fingertip's
        elastomer pad) takes its parent's, as Genesis merges a fixed child into the link it hangs
        off. A prop's geoms take the prop's, and the floor the ground's.

        Genesis mixes a contact's solver parameters as the plain mean of its two geoms' and its
        friction as their max, which MuJoCo does too only while neither geom outranks the other
        (``priority``) or weighs more (``solmix``) -- an asset's own (a Robotiq 2F-85's pads carry
        ``priority="1"``) are reset to the neutral ones. Both then floor the mixed time constant
        (Genesis at twice its substep, MuJoCo at twice its timestep: ``refsafe``), so each geom
        carries its own value as stated, never floored on its own -- a per-geom floor would raise
        the mix of a pair already above it.
        """
        mujoco = self._mujoco
        model = self.model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]

        outranked = sorted(
            {names[model.geom_bodyid[g]] or "world" for g in np.flatnonzero(model.geom_priority != 0)}
            | {names[model.geom_bodyid[g]] or "world" for g in np.flatnonzero(model.geom_solmix != 1.0)}
        )
        if outranked:
            en.logger.info(
                f"[deploy] MuJoCo rig: the geoms of {outranked} carry an asset priority / solmix, reset so their "
                "contacts mix as Genesis mixes them (the mean of both geoms' parameters, the max of their friction)."
            )
        model.geom_priority[:] = 0
        model.geom_solmix[:] = 1.0

        def genesis_link(body: int, prefix: str, table: dict) -> str | None:
            """The link ``table`` names for ``body``: the body itself, or the fixed ancestor Genesis merges it into."""
            while names[body].startswith(prefix):
                link = names[body][len(prefix) :]
                if link in table:
                    return link
                if model.body_jntnum[body] > 0:
                    return None
                body = int(model.body_parentid[body])
            return None

        link_geoms: dict[tuple[str, str], list[int]] = {}
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            part = next((p for p in self.parts.values() if names[body].startswith(f"{p.name}/")), None)
            if part is None:
                continue
            prefix = f"{part.name}/"
            link = genesis_link(body, prefix, part.link_friction)
            friction = part.friction if link is None else part.link_friction[link]
            if friction is not None:
                model.geom_friction[geom, 0] = float(friction)
            link = genesis_link(body, prefix, part.link_sol_params)
            if link is not None:
                link_geoms.setdefault((part.name, link), []).append(geom)
        for (part, link), geoms in link_geoms.items():
            self._set_sol_params(geoms, self.parts[part].link_sol_params[link], f"{part}/{link}")
        for name, prop in self.props.items():
            geoms = [
                g
                for b in self._prop_bodies(name)
                for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b] + model.body_geomnum[b]))
            ]
            if prop.friction is not None:
                model.geom_friction[geoms, 0] = float(prop.friction)
            if prop.sol_params is not None:
                self._set_sol_params(geoms, prop.sol_params, name)
        if self.floor is not None:
            self.floor.apply_contact(model)

    def _set_sol_params(self, geoms: list[int], sol_params, label: str) -> None:
        """Give MuJoCo geoms ``geoms`` (one Genesis link's, or one prop's) the solver parameters the env's carry.

        One value across the link -- what ``set_contact_sol_params`` writes, and every URDF's -- goes
        on every one of them. Values that differ per geom are an MJCF asset's own, which MuJoCo
        loaded from the same asset: no geom of one engine maps onto one of the other's (Genesis splits
        and convexifies meshes), so those stay the asset's, checked to be the same set of values.
        """
        model = self.model
        rows = np.unique(_sol_params_rows(sol_params), axis=0)
        if rows.shape[0] == 1:
            solref, solimp = _solref_solimp(rows[0])
            model.geom_solref[geoms] = solref
            model.geom_solimp[geoms] = solimp
            return
        colliding = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
        mujoco_rows = np.unique(np.concatenate([model.geom_solref[colliding], model.geom_solimp[colliding]], 1), axis=0)
        if not _same_rows(rows, mujoco_rows):
            en.logger.warning(
                f"[deploy] MuJoCo rig: the geoms of '{label}' carry {rows.shape[0]} different contact parameter sets "
                f"in the env ({rows.tolist()}) and others in MuJoCo ({mujoco_rows.tolist()}); its contacts run on "
                "the asset's MuJoCo values. To carry the env's, give the link one value (set_contact_sol_params / "
                "default_geoms_sol_params)."
            )

    def _prop_bodies(self, name: str) -> list[int]:
        mujoco = self._mujoco
        prop = self.props[name]
        if prop.asset_path is None:
            return [self._body_id(name)]
        prefix = f"{name}/"
        names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(self.model.nbody))
        return [b for b, n in enumerate(names) if n.startswith(prefix)]

    def _build_index_maps(self) -> None:
        mujoco = self._mujoco
        model = self.model
        joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt)}
        self._slices: dict[str, slice] = {}
        q_idx, v_idx = [], []
        for name, part in self.parts.items():
            missing = [dof for dof in part.dofs_name if f"{name}/{dof}" not in joints]
            if missing:
                have = sorted(j[len(name) + 1 :] for j in joints if j and j.startswith(f"{name}/"))
                raise ValueError(f"Part '{name}': the MuJoCo model has no joints {missing}. Its joints: {have}.")
            start = len(v_idx)
            for dof in part.dofs_name:
                j = joints[f"{name}/{dof}"]
                q_idx.append(int(model.jnt_qposadr[j]))
                v_idx.append(int(model.jnt_dofadr[j]))
            self._slices[name] = slice(start, len(v_idx))
            if part.armature is not None:
                model.dof_armature[v_idx[start:]] = np.asarray(part.armature, dtype=np.float64)
            if part.damping is not None:
                model.dof_damping[v_idx[start:]] = np.asarray(part.damping, dtype=np.float64)
        self._q_idx = np.array(q_idx, dtype=np.int64)
        self._v_idx = np.array(v_idx, dtype=np.int64)
        self._root_body = self._body_id(f"{self.root}/{self._root_body_name()}")
        self._free = {}
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                body = int(model.jnt_bodyid[j])
                self._free[body] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
        self._prop_body = {name: self._prop_root(name) for name in self.props}
        self._sensor = {}
        for name in (_IMU_GYRO, _IMU_ACC):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                self._sensor[name] = (int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))

    def _root_body_name(self) -> str:
        part = self.parts[self.root]
        return _load(part.asset_path).worldbody.bodies[0].name

    def _prop_root(self, name: str) -> int:
        prop = self.props[name]
        if prop.asset_path is None:
            return self._body_id(name)
        return self._body_id(f"{name}/{_load(prop.asset_path).worldbody.bodies[0].name}")

    def _body_id(self, name: str) -> int:
        mujoco = self._mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"No body named {name!r} in the rig's MuJoCo model.")
        return int(bid)

    # -- configuration -------------------------------------------------------

    def set_implicit_damping(self, kd: np.ndarray) -> None:
        """Each actuated DOF's armature: its own plus ``kd * sim_dt``, as :meth:`MujocoSim.set_implicit_damping`."""
        kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        if kd.shape[0] != len(self._v_idx):
            raise ValueError(f"kd has {kd.shape[0]} entries, expected {len(self._v_idx)} (every part's DOFs).")
        self.model.dof_armature[self._v_idx] = self._armature0[self._v_idx] + kd * self.sim_dt

    # -- state and commands --------------------------------------------------

    def reset(self, dofs_pos: dict[str, np.ndarray], prop_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Place every part's joints and every prop, at rest; the root's free base keeps its built pose."""
        data = self.data
        self._mujoco.mj_resetData(self.model, data)
        for name, pos in dofs_pos.items():
            data.qpos[self._q_idx[self._slices[name]]] = np.asarray(pos, dtype=np.float64)
        for name, (pos, quat) in prop_poses.items():
            body = self._prop_body[name]
            if body in self._free:
                adr = self._free[body][0]
                data.qpos[adr : adr + 3] = pos
                data.qpos[adr + 3 : adr + 7] = quat
        self._last_tau[:] = 0.0
        self._kp[:] = self._kd[:] = 0.0
        self._mujoco.mj_forward(self.model, data)

    def set_command(self, part: str, target_pos, target_vel, kp, kd, tau_ff) -> None:
        """Buffer ``part``'s PD command; it acts from the next :meth:`step`."""
        s = self._slices[part]
        self._target_pos[s] = target_pos
        self._target_vel[s] = 0.0 if target_vel is None else target_vel
        self._kp[s] = kp
        self._kd[s] = kd
        self._tau_ff[s] = 0.0 if tau_ff is None else tau_ff

    def step(self, n_substeps: int) -> None:
        """Advance ``n_substeps`` under every part's buffered command, the PD recomputed each substep."""
        mujoco = self._mujoco
        data = self.data
        for _ in range(int(n_substeps)):
            q = data.qpos[self._q_idx]
            dq = data.qvel[self._v_idx]
            tau = self._tau_ff + self._kp * (self._target_pos - q) + self._kd * (self._target_vel - dq)
            data.qfrc_applied[self._v_idx] = tau
            mujoco.mj_step(self.model, data)
            self._last_tau = tau

    def read_part(self, part: str) -> dict[str, np.ndarray]:
        """``part``'s joints, and for the root part its base: IMU gyro / accelerometer, and world pose and velocity."""
        s = self._slices[part]
        data = self.data
        out = {
            "dofs_pos": np.array(data.qpos[self._q_idx[s]], dtype=np.float64),
            "dofs_vel": np.array(data.qvel[self._v_idx[s]], dtype=np.float64),
            "dofs_torque": np.array(self._last_tau[s], dtype=np.float64),
        }
        if part == self.root:
            body = self._root_body
            out["base_pos"] = np.array(data.xpos[body], dtype=np.float64)
            out["base_quat"] = np.array(data.xquat[body], dtype=np.float64)
            if body in self._free:
                adr = self._free[body][1]
                out["base_lin_vel"] = np.array(data.qvel[adr : adr + 3], dtype=np.float64)
            for key, name in (("base_ang_vel", _IMU_GYRO), ("base_lin_acc", _IMU_ACC)):
                if name in self._sensor:
                    adr, dim = self._sensor[name]
                    out[key] = np.array(data.sensordata[adr : adr + dim], dtype=np.float64)
        return out

    def prop_body_id(self, name: str) -> int:
        """The MuJoCo body id of prop ``name``'s root body, the body that carries its pose."""
        return self._prop_body[name]

    def part_body_ids(self) -> list[int]:
        """The MuJoCo body ids of every part's bodies: the rig itself, without the floor or the props."""
        mujoco = self._mujoco
        prefixes = tuple(f"{name}/" for name in self.parts)
        names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(self.model.nbody))
        return [b for b, n in enumerate(names) if n.startswith(prefixes)]

    def prop_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose (pos, wxyz quat) of a prop's root body -- its asset's own root frame."""
        body = self._prop_body[name]
        return np.array(self.data.xpos[body], dtype=np.float64), np.array(self.data.xquat[body], dtype=np.float64)

    def body_pose(self, part: str, body: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose of ``part``'s body ``body`` as of the last :meth:`step`, :meth:`reset` or :meth:`set_dofs_pos`."""
        bid = self._body_id(f"{part}/{body}")
        return np.array(self.data.xpos[bid], dtype=np.float64), np.array(self.data.xquat[bid], dtype=np.float64)

    def set_dofs_pos(self, part: str, dofs_pos) -> None:
        """Write ``part``'s joint positions and recompute the kinematics, nothing else (a kinematics check)."""
        self.data.qpos[self._q_idx[self._slices[part]]] = np.asarray(dofs_pos, dtype=np.float64)
        self._mujoco.mj_forward(self.model, self.data)

    def bodies_of(self, part: str) -> list[str]:
        mujoco = self._mujoco
        prefix = f"{part}/"
        names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody))
        return [n[len(prefix) :] for n in names if n and n.startswith(prefix)]


def _enum(table, prefix: str, name: str):
    value = getattr(table, f"{prefix}{name.upper()}", None)
    if value is None:
        options = sorted(m[len(prefix) :].lower() for m in dir(table) if m.startswith(prefix))
        raise ValueError(f"Unknown MuJoCo option {name!r}; expected one of {options}.")
    return value
