"""Standalone MuJoCo physics wrapper for sim-to-sim deployment.

This is the engine half of the :class:`~eden.extensions.deployment.mujoco.MujocoDeployment`
backend, kept free of any Eden ``env`` dependency so it can be built and unit-tested from
nothing more than an asset path and a list of DOF names.

The wrapper mirrors what a real robot's low-level controller exposes over the wire, but
against a MuJoCo model instead of hardware:

- **Actuation** is a per-joint PD law identical to Unitree's ``unitree_mujoco`` bridge,
  ``tau = tau_ff + kp * (q* - q) + kd * (dq* - dq)``, applied through ``qfrc_applied`` (so
  the asset does *not* need to declare ``<actuator>`` elements — Eden's training MJCFs
  don't). The law is re-evaluated every physics substep, not held across the control
  period, which is what keeps a stiff PD stable.
- **Proprioception** is read from an IMU site placed on the floating-base body plus the
  joint ``qpos``/``qvel``. The IMU exposes a gyro, an accelerometer (specific force,
  so it reads ``+g`` when the robot is supported, like a real IMU), and a frame
  quaternion — matching the fields a Unitree ``LowState`` carries.

Because the training MJCF is authored for Genesis (zero joint armature/damping, no ground
plane, no free joint on some assets), the wrapper reconstructs a runnable MuJoCo scene: at
build time it injects a ground plane and adds a free joint to the base body when the asset
lacks one, and :meth:`MujocoSim.set_implicit_damping` adds ``kd * dt`` to each actuated DOF's
armature. That last term is the MuJoCo analogue of Genesis' implicit PD damping (integrated
implicitly by its implicit integrators, folded into the armature by ``RigidEntity.post_build``
under ``euler``) — without it a hand-rolled PD on a zero-armature model diverges.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


# Names of the geom / site / sensors this wrapper injects, prefixed so they can't collide
# with anything the asset already declares.
_FLOOR_GEOM = "__eden_floor__"
_IMU_SITE = "__eden_imu_site__"
_IMU_GYRO = "__eden_imu_gyro__"
_IMU_ACC = "__eden_imu_acc__"
_IMU_QUAT = "__eden_imu_quat__"


def _resolve_meshdir(spec, asset_path: str) -> None:
    """Point ``spec.meshdir`` at the directory the asset's meshes actually live in.

    MuJoCo's URDF importer strips the directory off every mesh reference, so a
    ``filename="./assets/base_link.STL"`` reaches ``compile()`` as the bare
    ``base_link.STL`` and is looked up under ``meshdir`` -- empty by default, i.e. the
    model file's own directory. A robot URDF that keeps its meshes in a subdirectory
    (LimX Oli's ``assets/``, the ROS convention ``meshes/``) therefore fails to compile
    with ``Error opening file '<mesh>'`` unless it carries a ``<mujoco><compiler
    meshdir=.../></mujoco>`` block, which most vendor URDFs do not.

    Locate the subdirectory holding the meshes and name it. Assets that already resolve
    -- every Eden training MJCF, and any URDF whose meshes sit beside it -- are left
    untouched, as is a model that sets ``meshdir`` itself.

    Parameters
    ----------
    spec:
        The ``mujoco.MjSpec`` just parsed from ``asset_path``, edited in place.
    asset_path:
        Path the spec was loaded from; mesh lookup is relative to its directory.
    """
    import os

    if spec.meshdir:
        return
    names = [os.path.basename(mesh.file) for mesh in spec.meshes if mesh.file]
    if not names:
        return
    root = os.path.dirname(os.path.abspath(asset_path))
    if os.path.isfile(os.path.join(root, names[0])):
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        if names[0] in filenames:
            spec.meshdir = os.path.relpath(dirpath, root)
            return


@dataclass(frozen=True)
class MujocoFloor:
    """The ground plane a MuJoCo sim adds: through ``pos``, facing ``normal`` (world frame), with the env's ground contact.

    ``friction`` is its sliding friction and ``sol_params`` its 7 Genesis contact solver parameters
    ``(timeconst, dampratio, dmin, dmax, width, mid, power)`` -- MuJoCo's ``solref`` and ``solimp``;
    ``None`` keeps MuJoCo's default. The default is MuJoCo's plain floor at ``z = 0``. Read off a
    built env by :func:`~eden.extensions.deployment.utils.mujoco_scene.read_floor`.

    ``caveat`` says how the floor falls short of the scene's ground (several grounds, a terrain's
    shape). The deployment says it once the floor is in the model, and says nothing when an asset
    keeps a plane of its own instead.
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    friction: float | None = None
    sol_params: tuple[float, ...] | None = None
    caveat: str | None = None

    def __post_init__(self) -> None:
        if self.sol_params is not None and len(self.sol_params) != 7:
            raise ValueError(
                "MujocoFloor.sol_params is 7 values (timeconst, dampratio, dmin, dmax, width, mid, power); got "
                f"{len(self.sol_params)}: {self.sol_params}."
            )

    def add_to(self, spec) -> None:
        """Add the floor geom to a MuJoCo spec, at its pose (a MuJoCo plane faces its local +z)."""
        import mujoco

        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, np.asarray(self.normal, dtype=np.float64))
        spec.worldbody.add_geom(
            name=_FLOOR_GEOM,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            pos=list(self.pos),
            quat=quat.tolist(),
        )

    def apply_contact(self, model) -> None:
        """Give the compiled floor geom its friction and solver parameters (where stated)."""
        import mujoco

        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM)
        if self.friction is not None:
            model.geom_friction[geom, 0] = float(self.friction)
        if self.sol_params is not None:
            model.geom_solref[geom] = self.sol_params[:2]
            model.geom_solimp[geom] = self.sol_params[2:]


def divides(control_dt: float, sim_dt: float) -> bool:
    """Whether ``sim_dt`` divides the control period into a whole number of steps (to float rounding)."""
    ratio = control_dt / sim_dt
    return ratio >= 1.0 - 1e-6 and math.isclose(ratio, round(ratio), rel_tol=1e-6)


def fitting_timestep(control_dt: float, cap: float) -> float:
    """The largest timestep of at most ``cap`` that divides the control period: ``control_dt / ceil(control_dt / cap)``."""
    return control_dt / math.ceil(control_dt / cap - 1e-9)


def steps_per_tick(control_dt: float, sim_dt: float) -> int:
    """The MuJoCo steps a lockstep control tick advances: ``control_dt / sim_dt``, refused unless a whole number.

    A tick advances a whole number of steps, so a timestep that does not divide the control period
    would run the physics on a clock ahead of or behind the policy's -- and the prop camera's latency
    and ``policy_seconds`` with it.
    """
    if not sim_dt > 0.0:
        raise ValueError(f"MuJoCo's timestep (sim_dt) must be > 0, got {sim_dt}.")
    if divides(control_dt, sim_dt):
        return round(control_dt / sim_dt)
    fit = fitting_timestep(control_dt, sim_dt)
    if sim_dt > control_dt:
        problem = f"is longer than the control period {control_dt:g} s, so no tick can end on a step"
    else:
        steps = round(control_dt / sim_dt)
        problem = (
            f"does not divide the control period {control_dt:g} s ({control_dt / sim_dt:.4g} steps a tick): each tick "
            f"would simulate {steps * sim_dt:g} s, so the physics would run "
            f"{'ahead of' if steps * sim_dt > control_dt else 'behind'} the policy's clock"
        )
    raise ValueError(
        f"MuJoCo's timestep {sim_dt:g} s (sim_dt) {problem}. Give one that divides it, e.g. "
        f"`eden deploy --mujoco_sim_dt {fit:g}` (the largest at most {sim_dt:g} s that does). Rounding the step count "
        "instead is what drifts the clock."
    )


class MujocoSim:
    """A single-robot MuJoCo world driven by a joint-space PD controller.

    Parameters
    ----------
    asset_path:
        Filesystem path to an MJCF (``.xml``) or URDF (``.urdf``) model of the robot.
    dofs_name:
        Actuated joint names, in the order the caller will supply commands and expect
        state back. Every name must be a hinge/slide joint in the model.
    sim_dt:
        MuJoCo integration timestep in seconds. ``None`` keeps the asset's own timestep
        (or MuJoCo's 0.002 default). The deployment picks the number of substeps per
        control step from this and the control period.
    floating_base:
        Whether the robot's base is free. When ``True`` and the asset has no free joint,
        one is added to the base body so the robot can fall/translate. When ``False`` the
        base is welded to the world (manipulators): ``base_pos`` / ``base_lin_vel`` read zero
        and the IMU reads the welded body's fixed orientation.
    floor:
        The ground plane to add if the asset has none (:class:`MujocoFloor`, by default MuJoCo's at
        ``z = 0``); ``None`` adds none.
    integrator:
        MuJoCo integrator: ``"implicitfast"`` (default, stable for stiff PD), ``"implicit"``,
        ``"euler"``, or ``"rk4"``.
    gravity:
        World gravity vector, m/s². Defaults to ``(0, 0, -9.81)``.
    base_body_name:
        Name of the body to treat as the floating base / IMU mount. Defaults to the first
        body under ``worldbody`` (the robot root for single-robot assets).
    """

    def __init__(
        self,
        asset_path: str,
        dofs_name: list[str],
        *,
        sim_dt: float | None = None,
        floating_base: bool = True,
        floor: MujocoFloor | None = MujocoFloor(),  # frozen, so one shared default is safe
        integrator: str = "implicitfast",
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        base_body_name: str | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.dofs_name = list(dofs_name)
        self.num_dofs = len(self.dofs_name)
        self.floating_base = floating_base

        spec = mujoco.MjSpec.from_file(asset_path)
        _resolve_meshdir(spec, asset_path)
        self._prepare_spec(spec, floating_base, floor, integrator, gravity, base_body_name, sim_dt)
        self.model = spec.compile()
        if self.floor is not None:
            self.floor.apply_contact(self.model)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)

        self._build_index_maps()
        # The asset's own armature on the actuated DOFs, which :meth:`set_implicit_damping` adds to.
        self._armature0 = self.model.dof_armature[self._v_idx].copy()
        self._last_tau = np.zeros(self.num_dofs, dtype=np.float64)
        self._gantry: dict | None = None

    # -- model construction ------------------------------------------------

    def _prepare_spec(self, spec, floating_base, floor, integrator, gravity, base_body_name, sim_dt) -> None:
        """Edit the loaded spec in place: base free joint, floor, IMU site+sensors, solver options."""
        mujoco = self._mujoco

        top_bodies = spec.worldbody.bodies
        if not top_bodies:
            raise ValueError(f"MuJoCo asset has no bodies under <worldbody>; cannot deploy: {spec.modelname!r}")
        base_body = self._find_body(top_bodies, base_body_name)
        self._base_body_name = base_body.name

        if floating_base and not self._has_free_joint(base_body):
            base_body.add_freejoint()

        # The floor this sim added, if any: an asset with a plane of its own keeps that one.
        self.floor = floor if floor is not None and not self._has_plane_geom(spec) else None
        if self.floor is not None:
            self.floor.add_to(spec)

        # IMU site on the base body + gyro / accelerometer / frame-quat sensors. We always
        # add our own (uniquely named) rather than hunt for the asset's, so proprioception
        # is identical across robots regardless of how each MJCF names its sensors.
        base_body.add_site(name=_IMU_SITE, pos=[0.0, 0.0, 0.0])
        spec.add_sensor(name=_IMU_GYRO, type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=_IMU_SITE)  # fmt: skip
        spec.add_sensor(name=_IMU_ACC, type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=_IMU_SITE)  # fmt: skip
        spec.add_sensor(name=_IMU_QUAT, type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=_IMU_SITE)  # fmt: skip

        spec.option.gravity = list(gravity)
        spec.option.integrator = self._integrator_enum(integrator)
        if sim_dt is not None:
            spec.option.timestep = float(sim_dt)

    def _find_body(self, bodies, name: str | None):
        if name is None:
            return bodies[0]
        for body in bodies:
            if body.name == name:
                return body
        available = [b.name for b in bodies]
        raise ValueError(f"base_body_name={name!r} not found under <worldbody>; available: {available}")

    def _has_free_joint(self, body) -> bool:
        free = self._mujoco.mjtJoint.mjJNT_FREE
        return any(jnt.type == free for jnt in body.joints)

    def _has_plane_geom(self, spec) -> bool:
        plane = self._mujoco.mjtGeom.mjGEOM_PLANE
        return any(geom.type == plane for geom in spec.geoms)

    def _integrator_enum(self, name: str):
        mujoco = self._mujoco
        table = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }
        try:
            return table[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown integrator {name!r}; expected one of {sorted(table)}.") from None

    def _build_index_maps(self) -> None:
        mujoco = self._mujoco
        model = self.model

        qpos_adr: dict[str, int] = {}
        dof_adr: dict[str, int] = {}
        free_qpos = free_dof = None
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                free_qpos = int(model.jnt_qposadr[j])
                free_dof = int(model.jnt_dofadr[j])
                continue
            qpos_adr[name] = int(model.jnt_qposadr[j])
            dof_adr[name] = int(model.jnt_dofadr[j])

        missing = [n for n in self.dofs_name if n not in qpos_adr]
        if missing:
            raise ValueError(
                f"MuJoCo model is missing actuated joints {missing}. Available hinge/slide joints: {sorted(qpos_adr)}."
            )
        self._q_idx = np.array([qpos_adr[n] for n in self.dofs_name], dtype=np.int64)
        self._v_idx = np.array([dof_adr[n] for n in self.dofs_name], dtype=np.int64)
        self._free_qpos = free_qpos
        self._free_dof = free_dof
        self._base_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._base_body_name))

        self._sensor_adr = {}
        for name in (_IMU_GYRO, _IMU_ACC, _IMU_QUAT):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._sensor_adr[name] = (int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))

    # -- configuration -----------------------------------------------------

    def set_implicit_damping(self, kd: np.ndarray) -> None:
        """Set each actuated DOF's armature to the asset's plus ``kd * sim_dt`` (implicit velocity damping).

        Mirrors Genesis' ``RigidEntity.post_build``, which adds the same term under the ``euler``
        integrator (its implicit integrators integrate the PD damping implicitly instead).
        Absolute, not cumulative: call it again whenever ``kd`` changes -- a deploy policy-slot
        switch ramps the gains -- and the armature follows the new ``kd`` instead of stacking on
        the old one. The armature is a model property, so the caller has to keep a
        concurrent stepper out while it is written.
        """
        kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        if kd.shape[0] != self.num_dofs:
            raise ValueError(f"kd has {kd.shape[0]} entries, expected {self.num_dofs}.")
        self.model.dof_armature[self._v_idx] = self._armature0 + kd * self.sim_dt

    # -- state / actuation -------------------------------------------------

    def reset(
        self,
        dofs_pos: np.ndarray,
        base_pos: np.ndarray | None = None,
        base_quat: np.ndarray | None = None,
    ) -> None:
        """Place the robot at a joint pose (and optional base pose) and zero all velocities.

        ``base_quat`` is (w, x, y, z), matching Eden's convention and MuJoCo's free-joint
        quaternion layout. Ignored when the base is welded (fixed base).
        """
        mujoco = self._mujoco
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        if self.floating_base and self._free_qpos is not None:
            if base_pos is not None:
                self.data.qpos[self._free_qpos : self._free_qpos + 3] = np.asarray(base_pos, dtype=np.float64)
            quat = np.array([1.0, 0.0, 0.0, 0.0]) if base_quat is None else np.asarray(base_quat, dtype=np.float64)
            self.data.qpos[self._free_qpos + 3 : self._free_qpos + 7] = quat
        self.data.qpos[self._q_idx] = np.asarray(dofs_pos, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        self._last_tau[:] = 0.0

    def set_gantry(
        self,
        *,
        enabled: bool,
        mounts: Sequence[tuple[str | None, tuple[float, float, float]]] = ((None, (0.0, 0.0, 0.25)),),
        height: float | None = None,
        stiffness: float = 5000.0,
        damping: float = 200.0,
        slack: float = 0.0,
        support: float = 1.0,
    ) -> None:
        """Hang the robot from an overhead hoist by one or more simulated ropes.

        The bring-up rig: a hoist above the robot that catches it, so a biped can be held
        upright while a controller is commissioned instead of falling every time the operator
        pauses. Each mount is a tension-only spring-damper between a fixed world anchor and a
        point on a body -- **not** weight cancellation, which is the tempting shortcut and the
        wrong physics: a uniform upward force through the centre of mass makes the robot lighter
        without making it any more upright.

        **Two mounts beat one.** A single rope rights the robot only through its own ``r x F``
        moment arm, so it needs an attachment well above the centre of mass and still lets the
        robot roll about the rope. A pair at the shoulders resists roll as a *couple* -- the
        shoulder that drops stretches its rope and is pulled back -- which is why real humanoid
        rigs use a spreader bar, and why :attr:`RobotOptions.gantry_links_name` names two links.

        Tension-only means each rope pulls and never pushes, so the robot is free to move toward
        the anchor and is caught on the way down. ``slack`` > 0 makes it a safety tether that
        does nothing until the robot drops that far.

        Parameters
        ----------
        enabled:
            ``False`` removes the ropes (and clears any force they were applying).
        mounts:
            ``(body_name, attach)`` pairs; ``body_name=None`` means the floating base and
            ``attach`` is in that body's frame, in metres. Each gets its own anchor directly
            above it, so a symmetric pair hangs level.
        height:
            World height of every anchor. Defaults to 1 m above each attachment point, which for
            a level pair is the same height for both. Rest lengths are fixed from this at rig
            time, so :meth:`adjust_anchor` afterwards raises and lowers the robot like a hoist.
        stiffness, damping:
            Spring-damper of each rope, N/m and N.s/m. Per rope, not shared across them.
        slack:
            Extra rope beyond each anchor-to-attachment distance at setup, in metres.
        support:
            Scale on the resulting forces, ``0``-``1``. Ramp down to hand the robot's weight back
            to its own controller without re-rigging.
        """
        if not enabled:
            self._gantry = None
            self.data.xfrc_applied[:] = 0.0
            return
        if not mounts:
            raise ValueError("set_gantry(enabled=True) needs at least one mount.")
        ropes = []
        for body, attach in mounts:
            body_id = self._base_body_id if body is None else self._body_id(body)
            attach = np.asarray(attach, dtype=np.float64)
            attach_w = self._attach_point_world(body_id, attach)
            anchor_z = attach_w[2] + 1.0 if height is None else float(height)
            anchor = np.array([attach_w[0], attach_w[1], anchor_z])
            ropes.append(
                {
                    "draw": None,
                    "body_id": body_id,
                    "attach": attach,
                    "anchor": anchor,
                    "rest": float(np.linalg.norm(anchor - attach_w)) + float(slack),
                }
            )
        self._gantry = {
            "ropes": ropes,
            "stiffness": float(stiffness),
            "damping": float(damping),
            "support": float(support),
        }

    def _body_id(self, name: str) -> int:
        """Resolve a body name, listing what the model has when it is not there."""
        mujoco = self._mujoco
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            available = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
            raise ValueError(
                f"No body named {name!r} in the MuJoCo model. Available: {sorted(filter(None, available))}"
            )
        return int(bid)

    @property
    def gantry_anchor_height(self) -> float | None:
        """Height of the gantry anchors (their mean), or ``None`` when nothing is rigged."""
        if self._gantry is None:
            return None
        return float(np.mean([r["anchor"][2] for r in self._gantry["ropes"]]))

    def adjust_anchor(self, delta_z: float) -> float | None:
        """Raise (+) or lower (-) every anchor together, and return the new height.

        The rest lengths do not change, so the robot follows: raising takes up rope and lifts it,
        lowering pays rope out and sets it down. Moving the anchors as one keeps a shoulder pair
        level -- the rig is a spreader bar, not two independent winches.
        """
        if self._gantry is None:
            return None
        for rope in self._gantry["ropes"]:
            rope["anchor"][2] += float(delta_z)
        return self.gantry_anchor_height

    def _attach_point_world(self, body_id: int, attach: np.ndarray) -> np.ndarray:
        """World position of ``attach`` expressed in ``body_id``'s frame."""
        mujoco = self._mujoco
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3, 3) @ attach

    def _apply_gantry(self) -> None:
        """Write this substep's rope wrenches to the attached bodies' ``xfrc_applied``."""
        g = self._gantry
        if g is None:
            return
        mujoco = self._mujoco
        # Zero the bodies this rig loads before accumulating into them: two mounts may share a
        # body and must sum, so the writes below are ``+=`` and would otherwise pile up across
        # calls. Clearing here (rather than in the caller) keeps one call self-contained.
        for rope in g["ropes"]:
            self.data.xfrc_applied[rope["body_id"]] = 0.0
        for rope in g["ropes"]:
            bid = rope["body_id"]
            attach_w = self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3) @ rope["attach"]
            delta = rope["anchor"] - attach_w
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                continue
            direction = delta / dist
            # ``xfrc_applied`` acts at the body's CoM (``xipos``), and ``mj_objectVelocity``
            # reports the linear velocity there too -- so both the moment arm and the point
            # velocity are measured from the CoM, not the body frame origin (they differ for
            # any real link).
            arm = attach_w - self.data.xipos[bid]
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            vel = vel6[3:] + np.cross(vel6[:3], arm)
            stretch = dist - rope["rest"]
            # Tension-only: a rope pulls along its length and never pushes.
            tension = max(g["stiffness"] * stretch - g["damping"] * float(vel @ direction), 0.0)
            force = tension * g["support"] * direction
            # ``+=``: two mounts on one body (or a repeated link) must sum, not overwrite.
            self.data.xfrc_applied[bid, :3] += force
            self.data.xfrc_applied[bid, 3:] += np.cross(arm, force)
            # Last-drawn state for the viewer overlay. Kept on the rope rather than recomputed
            # from qpos later, so what is drawn is exactly the wrench that was applied.
            rope["draw"] = (rope["anchor"].copy(), attach_w.copy(), force.copy())

    def draw_gantry(self, scene, *, force_scale: float = 1.0e-3) -> None:
        """Draw each rope and the force it is pulling with into a viewer scene.

        Two marks per rope: a thin line along the rope itself (grey when slack, so a tether that
        is not doing anything reads as such at a glance) and an arrow at the attachment point
        along the force, scaled by ``force_scale``. The arrow is the useful one -- it is the
        wrench that was actually applied this substep, not a re-derivation, so what is on screen
        is what the robot felt.

        No-op when nothing is rigged, and it stops adding geoms when the scene is full rather
        than overrunning ``maxgeom``.

        Parameters
        ----------
        scene:
            An ``mjvScene`` to append to — the passive viewer's ``user_scn``.
        force_scale:
            Metres of arrow per newton. The default draws a 1 kN pull as 1 m.
        """
        mujoco = self._mujoco
        if self._gantry is None:
            return
        slack_rgba = np.array([0.55, 0.55, 0.60, 0.8], dtype=np.float32)
        taut_rgba = np.array([0.15, 0.85, 1.0, 0.9], dtype=np.float32)
        force_rgba = np.array([1.0, 0.45, 0.1, 0.95], dtype=np.float32)
        for rope in self._gantry["ropes"]:
            drawn = rope.get("draw")
            if drawn is None:
                continue
            anchor, attach_w, force = drawn
            magnitude = float(np.linalg.norm(force))
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                taut_rgba if magnitude > 0.0 else slack_rgba,
            )  # fmt: skip
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, anchor, attach_w)
            scene.ngeom += 1
            if magnitude <= 0.0 or scene.ngeom >= scene.maxgeom:
                continue
            tip = attach_w + force * force_scale
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), force_rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.014, attach_w, tip)
            scene.ngeom += 1

    def step_pd(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        tau_ff: np.ndarray,
        n_substeps: int,
    ) -> None:
        """Advance ``n_substeps`` physics steps under a joint PD law recomputed each substep.

        ``tau = tau_ff + kp * (target_pos - q) + kd * (target_vel - dq)`` on the actuated
        DOFs, written to ``qfrc_applied``. Recomputing per substep (rather than holding the
        torque across the whole control period) is what keeps a stiff PD numerically stable.
        """
        mujoco = self._mujoco
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_vel = np.asarray(target_vel, dtype=np.float64)
        kp = np.asarray(kp, dtype=np.float64)
        kd = np.asarray(kd, dtype=np.float64)
        tau_ff = np.asarray(tau_ff, dtype=np.float64)
        for _ in range(int(n_substeps)):
            q = self.data.qpos[self._q_idx]
            dq = self.data.qvel[self._v_idx]
            tau = tau_ff + kp * (target_pos - q) + kd * (target_vel - dq)
            self.data.qfrc_applied[self._v_idx] = tau
            self._apply_gantry()
            mujoco.mj_step(self.model, self.data)
        self._last_tau = tau

    def read(self) -> dict[str, np.ndarray]:
        """Return the current proprioceptive state, mirroring a robot's low-level feedback.

        Keys: ``base_pos`` (world), ``base_quat`` (w,x,y,z world), ``base_lin_vel`` (world),
        ``base_ang_vel`` (gyro, body frame), ``base_lin_acc`` (accelerometer specific force,
        body frame), ``dofs_pos``, ``dofs_vel``, ``dofs_torque`` (last commanded PD torque).
        Base fields are ground truth from the free joint / IMU; ``base_pos`` and
        ``base_lin_vel`` are exact here (a real robot would have to estimate them).
        """
        data = self.data
        g_adr, g_dim = self._sensor_adr[_IMU_GYRO]
        a_adr, a_dim = self._sensor_adr[_IMU_ACC]
        q_adr, q_dim = self._sensor_adr[_IMU_QUAT]
        base_ang_vel = np.array(data.sensordata[g_adr : g_adr + g_dim], dtype=np.float64)
        base_lin_acc = np.array(data.sensordata[a_adr : a_adr + a_dim], dtype=np.float64)
        base_quat = np.array(data.sensordata[q_adr : q_adr + q_dim], dtype=np.float64)

        if self.floating_base and self._free_qpos is not None:
            base_pos = np.array(data.qpos[self._free_qpos : self._free_qpos + 3], dtype=np.float64)
            base_lin_vel = np.array(data.qvel[self._free_dof : self._free_dof + 3], dtype=np.float64)
        else:
            base_pos = np.zeros(3)
            base_lin_vel = np.zeros(3)

        return {
            "base_pos": base_pos,
            "base_quat": base_quat,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "base_lin_acc": base_lin_acc,
            "dofs_pos": np.array(data.qpos[self._q_idx], dtype=np.float64),
            "dofs_vel": np.array(data.qvel[self._v_idx], dtype=np.float64),
            "dofs_torque": self._last_tau.copy(),
        }

    def robot_body_ids(self) -> list[int]:
        """The MuJoCo body ids of the robot: the base body's tree, without the world or any other body the asset has."""
        root = self.model.body_rootid[self._base_body_id]
        return np.flatnonzero(self.model.body_rootid == root).tolist()

    def lowest_geom_z(self, exclude_floor: bool = True) -> float:
        """World-z of the lowest geom *centre* (not surface) at the current pose (for spawn-height fitting)."""
        mujoco = self._mujoco
        self._mujoco.mj_forward(self.model, self.data)
        floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM)
        zs = [
            float(self.data.geom_xpos[g][2]) for g in range(self.model.ngeom) if not (exclude_floor and g == floor_gid)
        ]
        return min(zs) if zs else 0.0
