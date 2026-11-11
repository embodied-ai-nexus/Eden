"""MuJoCo sim-to-sim of a composite rig, through the composite deployment itself.

``eden deploy --mujoco`` on a scene whose ``--entity_name`` is a composite rig (``KukaSharpa``,
``FrankaXHand``, ``OliWuji``) runs :class:`MujocoCompositeDeployment`: the very
:class:`~eden.extensions.deployment.composite.CompositeDeployment` that drives the rig's hardware --
one backend per part, each on its own thread, one state machine, the parts' readings joined into one
observation -- with every part on a :class:`MujocoPartDeployment` in place of its hardware backend.
The parts share one MuJoCo model of the whole rig (:class:`~eden.extensions.deployment.utils.mujoco_rig.MujocoRig`)
plus the scene's props (a table, a tool), so a manipulation policy acts on a simulated object and
observes it.

The MuJoCo model is assembled from the built Genesis env, so it is the rig the policy trained on:

- each part's own asset, welded where the env welds it -- the mount read off the env's forward
  kinematics, not re-derived -- and checked link by link against that kinematics at connect;
- each part's armature and joint damping as built, its material friction (``default_geoms_friction``
  per link), its links' contact solver parameters as built (a ``set_contact_sol_params`` startup
  event included) and gravity compensation;
- each prop at the pose the env spawned it, with its friction and contact parameters.

Each tick every part writes its command to the model on its own thread, then the composite steps the
physics once for the control period (lockstep, as ``MujocoDeployment``), reads the parts back on
their threads, and writes each prop's pose into its env entity -- what a mocap system does on
hardware -- before the observation is computed. Where hardware has a camera instead (an AprilTag
pose estimate), ``prop_pos_noise`` / ``prop_rot_noise`` / ``prop_camera_hz`` / ``prop_latency_s``
make what the policy reads of each free prop one: noisy, late and at the camera's rate (the exact
pose stays in the state as ``<prop>.true_pos`` / ``<prop>.true_quat`` for ``--debug_record``).
``record_video`` renders the rollout from MuJoCo, as a single robot's backend does, and the base's
``policy_seconds`` ends a headless run. The env is read into the rig's description by
:mod:`~eden.extensions.deployment.utils.mujoco_scene`; this class keeps the lifecycle, the checks
and the camera.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

import eden as en
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY, DeploymentBase
from eden.extensions.deployment.composite import CompositeDeployment
from eden.extensions.deployment.utils.mujoco_scene import (
    EnvPart,
    entity_pose,
    f64,
    is_kinematic,
    kinematic_markers,
    link_pose,
    marker_poses,
    read_floor,
    rig_parts,
    rig_props,
    say_floor_caveat,
    unsupported_prop,
)
from eden.extensions.deployment.utils.mujoco_sim import fitting_timestep, steps_per_tick
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.options.entities import CompositeEntityOptions, PlaneOptions, RobotOptions, TerrainOptions

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.extensions.deployment.utils.mujoco_rig import MujocoRig, RigPart
    from eden.extensions.deployment.utils.mujoco_video import MujocoVideo
    from eden.options.extensions.deployment import DeploymentOptions


@DEPLOYMENT_REGISTRY.register()
class MujocoPartDeployment(DeploymentBase):
    """One part of a composite rig on the rig's shared MuJoCo model -- the sim's stand-in for a part's hardware backend.

    Built only by :class:`MujocoCompositeDeployment`, which binds it to the model at connect. Its
    wire is the model: :meth:`send_payload` buffers the part's PD command (the composite steps the
    physics once every part has sent), :meth:`read_state` reads the part's joints, and for the rig's
    root its base -- the IMU, and the ground-truth pose and velocity of a free base when
    ``provide_ground_truth_base``, as ``MujocoDeployment`` gives a single robot.
    """

    provide_ground_truth_base: bool = True
    # The robot is a simulation, so a headless run may press its own INIT and START.
    allow_unattended_start: ClassVar[bool] = True

    def __init__(self, env: RLEnvBase, options: DeploymentOptions) -> None:
        # Before the base constructor, which asks :meth:`_publishes_base_lin_vel` already.
        self._rig: MujocoRig | None = None
        super().__init__(env, options)

    def bind(self, rig: MujocoRig) -> None:
        """Wire this part to the rig's model (:meth:`MujocoCompositeDeployment.connect`)."""
        self._rig = rig

    def _require_rig(self) -> MujocoRig:
        if self._rig is None:
            raise RuntimeError(
                f"MujocoPartDeployment '{self.entity_name}' has no MuJoCo model: it is one part of a rig, built and "
                "connected by MujocoCompositeDeployment (`eden deploy --mujoco` on a composite rig), never on its own."
            )
        return self._rig

    def connect(self) -> None:
        self._require_rig()

    def close(self) -> None:
        self._teardown_common()

    def _is_free_root(self) -> bool:
        rig = self._require_rig()
        return self.entity_name == rig.root and rig.parts[rig.root].floating

    def _publishes_base_lin_vel(self) -> bool:
        return bool(self.provide_ground_truth_base) and self._rig is not None and self._is_free_root()

    def read_state(self) -> RobotState:
        r = self._require_rig().read_part(self.entity_name)
        extra: dict[str, Any] = {}
        if self.provide_ground_truth_base and self._is_free_root():
            extra["base_pos"] = r["base_pos"]
            extra["base_lin_vel"] = r["base_lin_vel"]
        # A welded part has no IMU; its base is read off the forward kinematics through the weld.
        return RobotState(
            stamp=time.time(),
            base_quat=r.get("base_quat", np.array([1.0, 0.0, 0.0, 0.0])),
            base_ang_vel=r.get("base_ang_vel", np.zeros(3)),
            base_lin_acc=r.get("base_lin_acc", np.zeros(3)),
            dofs_pos=r["dofs_pos"],
            dofs_vel=r["dofs_vel"],
            dofs_torque=r["dofs_torque"],
            extra=extra,
        )

    def send_payload(self, payload: RobotCommand) -> None:
        kp = self.default_dof_kp if payload.dofs_kp is None else payload.dofs_kp
        kd = self.default_dof_kd if payload.dofs_kd is None else payload.dofs_kd
        # Each part low-passes its own slice, as its hardware backend does on its wire (``target_lpf``), and
        # leaves what it sent in the payload, which the composite joins into the rig's (``--debug_record``).
        payload.dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))
        self._require_rig().set_command(
            self.entity_name, payload.dofs_pos, payload.dofs_vel, kp, kd, payload.dofs_torque
        )


@DEPLOYMENT_REGISTRY.register()
class MujocoCompositeDeployment(CompositeDeployment):
    """A composite rig deployed onto MuJoCo: :class:`CompositeDeployment` with a :class:`MujocoPartDeployment` per part.

    Options (in addition to :class:`CompositeDeployment`'s):

    sim_dt: float | None
        MuJoCo timestep. ``None`` (default) takes the largest step of at most 2 ms that divides the
        control period exactly, so each tick advances the physics by exactly one control period. A
        given one must divide it too (``eden deploy --mujoco_sim_dt``): one that does not is refused.
    integrator: str
        ``"implicitfast"`` (default), ``"implicit"``, ``"euler"`` or ``"rk4"``.
    cone: str
        Friction cone, ``"elliptic"`` (default -- grasps hold far better on it) or ``"pyramidal"``.
    props: tuple[str, ...] | None
        Scene entities to simulate besides the rig (``eden deploy --mujoco_props``). ``None``
        (default) takes every colliding entity of the scene that is not the rig or the ground. Each
        must be a box primitive or a rigid body loaded from a URDF / MJCF, without joints of its own:
        only its pose reaches the env, which a mesh file's (MuJoCo cannot load one as a body) or an
        articulated object's joints would not. Anything else is refused, naming it. A kinematic entity
        (a goal marker: not simulated, no contacts) is never a prop; ``record_video`` draws it.
    prop_pos_noise, prop_rot_noise: float
        A camera's error on every free prop's pose the policy reads: per-axis white noise on the
        position (m, standard deviation), and a rotation about a random axis by a normal angle (rad,
        standard deviation). Default 0. With either on, the quaternion's sign is random too, as a
        pose estimator's is. Only what the policy observes changes; MuJoCo's physics does not, and the
        exact pose stays in the state as ``<prop>.true_pos`` / ``<prop>.true_quat``.
    prop_camera_hz: float | None
        The camera's frame rate: a free prop's observed pose updates at it and holds between frames,
        on the simulation's clock (a tick that sends nothing advances neither). ``None`` (default):
        every control tick.
    prop_latency_s: float
        How old a frame is when it arrives, rounded half up to whole control ticks (with a warning
        when that changes it). Default 0.
    prop_noise_seed: int | None
        Seed for the camera model's noise, so a run can be repeated. ``None`` draws a fresh one.
    record_video: str | None
        Write the rollout from the policy's first START on to this mp4, rendered offscreen from MuJoCo
        (two views side by side, one frame per control tick, a STOP's holds included so the video
        never skips). Headless, it needs ``MUJOCO_GL=egl``.
    kinematics_tolerance: float
        Largest link-pose disagreement (metres plus radians) between the MuJoCo model and the env's
        forward kinematics accepted at connect. Above it the assembly is wrong and connect raises.
    show_viewer: bool
        Open MuJoCo's passive viewer on the rig.
    """

    sim_dt: float | None = None
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    props: tuple[str, ...] | None = None
    prop_pos_noise: float = 0.0
    prop_rot_noise: float = 0.0
    prop_camera_hz: float | None = None
    prop_latency_s: float = 0.0
    prop_noise_seed: int | None = None
    record_video: str | None = None
    kinematics_tolerance: float = 1.0e-4
    show_viewer: bool = False
    allow_unattended_start: ClassVar[bool] = True
    operator_guide_title: ClassVar[str] = (
        "MUJOCO RIG SIM-TO-SIM DEPLOY CONTROLS  (keyboard — same sequence as the robot)"
    )

    def __init__(self, env: RLEnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)
        self._rig: MujocoRig | None = None
        self._viewer = None
        self._n_substeps = 1
        self._damped_kd: np.ndarray | None = None
        # The yaw the props were last written into the env under (the env spawned them under none).
        self._props_yaw = 0.0
        self._prop_names = self._resolve_props()
        # Where each prop spawned, captured before any tick writes the simulated pose over it.
        self._prop_home = {name: entity_pose(self._env, name) for name in self._prop_names}
        self._camera = _PropCamera(
            pos_noise=self.prop_pos_noise,
            rot_noise=self.prop_rot_noise,
            hz=self.prop_camera_hz,
            latency_ticks=self._latency_ticks(),
            control_dt=self.control_dt,
            seed=self.prop_noise_seed,
        )
        self._video: MujocoVideo | None = None
        # Control periods the physics has advanced since the reset: the camera's clock.
        self._sim_ticks = 0

    def _latency_ticks(self) -> int:
        """``prop_latency_s`` in whole control ticks, rounded half up, with a warning when that changes it."""
        ticks = math.floor(self.prop_latency_s / self.control_dt + 0.5)
        if abs(ticks * self.control_dt - self.prop_latency_s) > 1e-9:
            en.logger.warning(
                f"[deploy] prop_latency_s={self.prop_latency_s:g} s is not a whole number of control ticks "
                f"({self.control_dt:g} s): the camera's frames arrive {ticks} ticks "
                f"({ticks * self.control_dt:g} s) late."
            )
        return ticks

    def _resolve_options(self) -> None:
        """Reject a camera model or a timestep that cannot be (see :meth:`DeploymentBase._resolve_options`)."""
        super()._resolve_options()
        if self.sim_dt is not None:
            steps_per_tick(self.control_dt, self.sim_dt)
        for name in ("prop_pos_noise", "prop_rot_noise", "prop_latency_s"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}.")
        if self.prop_camera_hz is not None and not self.prop_camera_hz > 0.0:
            raise ValueError(f"prop_camera_hz must be > 0, or None; got {self.prop_camera_hz}.")

    @staticmethod
    def _part_backend(rig: CompositeEntityOptions, name: str, options: RobotOptions) -> type[DeploymentBase]:
        # Every part runs on the rig's MuJoCo model, whatever hardware backend its robot is bound to.
        return MujocoPartDeployment

    # -- the props -----------------------------------------------------------

    def _resolve_props(self) -> list[str]:
        scene = self._env.config.scene_options
        parts = set(self.robot_options.entity_names)
        candidates = {}
        for key in scene.keys():
            options = getattr(scene, key)
            if key not in self._env.entities or key in parts:  # not an entity (``attachments_dict``), or a part
                continue
            if isinstance(options, (CompositeEntityOptions, PlaneOptions, TerrainOptions)):  # the rig, or the ground
                continue
            candidates[key] = options
        if self.props is None:
            chosen = [
                name
                for name, options in candidates.items()
                if getattr(options, "collision", True) and not is_kinematic(options)
            ]
        else:
            unknown = [name for name in self.props if name not in candidates]
            if unknown:
                raise ValueError(
                    f"props names {unknown}, which are not scene entities besides the rig and the ground. "
                    f"Candidates: {sorted(candidates)}."
                )
            chosen = list(self.props)
        unsupported = {
            name: why for name in chosen if (why := unsupported_prop(self._env.entities[name], candidates[name]))
        }
        if unsupported:
            others = [name for name in chosen if name not in unsupported]
            if others:
                fix = f"simulate the others only, `--mujoco_props {' '.join(others)}` (props={tuple(others)!r})"
            else:
                # Nothing else to keep: the bare flag (no names) is the rig alone.
                fix = "simulate the rig alone, `--mujoco_props` with no names (props=())"
            them = "them" if len(unsupported) > 1 else "it"
            raise NotImplementedError(
                f"MujocoCompositeDeployment cannot simulate {unsupported}: a prop must be a box primitive or a rigid "
                f"body from a URDF / MJCF, with no joints of its own (only its pose is written back to the env). "
                f"To run anyway, {fix}, or remove {them} from the scene."
            )
        return chosen

    # -- the model ------------------------------------------------------------

    def _rig_parts(self) -> list[RigPart]:
        return rig_parts(
            self.robot_options,
            {
                name: EnvPart(part._state_entity._sim_entity, list(part.dofs_name), part.robot_options)
                for name, part in self.parts.items()
            },
        )

    def _check_contact_floor(self) -> None:
        """Warn when a contact time constant reaches MuJoCo below a floor the two engines put in different places.

        Both floor a contact's mixed time constant: Genesis at twice its substep, MuJoCo at twice its
        own timestep (``refsafe``). Where those differ, a contact whose mix lies below the higher
        floor is stiffer or softer here than in training -- possible only if some geom's own time
        constant lies below it.
        """
        env_options = self._env.config.env_options
        genesis_floor = 2.0 * env_options.sim_dt / env_options.sim_substeps
        mujoco_floor = 2.0 * self._rig.sim_dt
        if math.isclose(genesis_floor, mujoco_floor, rel_tol=1e-9):
            return
        model = self._rig.model
        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        lowest = float(model.geom_solref[colliding, 0].min()) if colliding.any() else math.inf
        if lowest >= max(genesis_floor, mujoco_floor) * (1.0 - 1e-6):  # Genesis stores them in float32
            return
        en.logger.warning(
            f"[deploy] A contact time constant as low as {lowest:g} s reaches MuJoCo, which floors a contact's at "
            f"twice its timestep ({mujoco_floor:g} s) where Genesis floors it at twice its substep ({genesis_floor:g} "
            f"s): such a contact is {'stiffer' if mujoco_floor < genesis_floor else 'softer'} here than in training. "
            f"To floor it alike, run MuJoCo at Genesis' substep: `eden deploy --mujoco_sim_dt {genesis_floor / 2.0:g}` "
            f"(MujocoCompositeDeployment sim_dt={genesis_floor / 2.0:g})."
        )

    def _check_kinematics(self) -> None:
        """Every link both engines have, at the env's current joint pose, must sit where the env's FK puts it."""
        from eden.utils.geom import quat_error_magnitude

        rig = self._rig
        for name, part in self.parts.items():
            sim = part._state_entity._sim_entity
            q = f64(sim.get_dofs_pos()).reshape(-1)
            rig.set_dofs_pos(name, [q[sim.dofs_name.index(dof)] for dof in part.dofs_name])
        worst, where, compared = 0.0, None, 0
        for name, part in self.parts.items():
            sim = part._state_entity._sim_entity
            genesis_links = {link.name for link in sim.links}
            for body in rig.bodies_of(name):
                if body not in genesis_links:
                    continue  # a fixed child Genesis merged into its parent link
                gpos, gquat = link_pose(sim, body)
                mpos, mquat = rig.body_pose(name, body)
                err = float(np.linalg.norm(mpos - gpos.numpy())) + float(
                    quat_error_magnitude(torch.as_tensor(mquat)[None], gquat[None])[0]
                )
                compared += 1
                if err > worst:
                    worst, where = err, f"{name}/{body}"
        if worst > self.kinematics_tolerance:
            raise ValueError(
                f"The MuJoCo model of {type(self.robot_options).__name__} disagrees with the env's forward kinematics: "
                f"link '{where}' is {worst:.3g} (m + rad) off, over the {self.kinematics_tolerance:g} allowed "
                f"(kinematics_tolerance), across {compared} links. A part's asset or its weld does not match the "
                "env's; check that every part's asset is the one the env built from."
            )
        en.logger.info(
            f"[deploy] MuJoCo rig kinematics match the env's: {compared} links, worst {worst:.2g} (m + rad)."
        )

    # -- lifecycle ------------------------------------------------------------

    def connect(self) -> None:
        from eden.extensions.deployment.utils.mujoco_rig import MujocoRig

        # The largest step of at most 2 ms that divides the control period, unless one is given.
        sim_dt = fitting_timestep(self.control_dt, 0.002) if self.sim_dt is None else float(self.sim_dt)
        self._rig = MujocoRig(
            self._rig_parts(),
            rig_props(self._env, {name: self._prop_home[name] for name in self._prop_names}),
            sim_dt=sim_dt,
            integrator=self.integrator,
            cone=self.cone,
            floor=read_floor(self._env),
        )
        say_floor_caveat(self._rig.floor)
        self._check_kinematics()
        self._check_contact_floor()
        self._n_substeps = steps_per_tick(self.control_dt, self._rig.sim_dt)
        for part in self.parts.values():
            part.bind(self._rig)
        super().connect()
        self._reset_sim()
        en.logger.info(
            f"[deploy] MuJoCo rig sim-to-sim: parts {list(self.parts)}, props {self._prop_names} | "
            f"sim_dt={self._rig.sim_dt:.6g}s x {self._n_substeps} substeps/step | integrator={self.integrator} | "
            f"cone={self.cone} | lockstep with the control loop"
        )
        if self.show_viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._rig.model, self._rig.data)
        if self.record_video is not None:
            from eden.extensions.deployment.utils.mujoco_video import MujocoVideo

            # Framed on the rig's bodies and its free props where they start.
            rig = self._rig
            framed = [rig.prop_body_id(name) for name, prop in rig.props.items() if prop.free] + rig.part_body_ids()
            self._video = MujocoVideo(
                rig.model,
                rig.data,
                self.record_video,
                fps=1.0 / self.control_dt,
                framed_bodies=framed,
                markers=kinematic_markers(self._env),
            )

    def _reset_sim(self) -> None:
        dofs = {name: self.default_dof_pos[self._part_slices[name]] for name in self.parts}
        self._rig.reset(dofs, {name: self._prop_home[name] for name in self._prop_names})
        self._camera.reset()
        self._sim_ticks = 0
        self._fold_implicit_damping()

    def _fold_implicit_damping(self) -> None:
        """Set the armature's implicit damping to the active slot's ``default_dof_kd``.

        The armature models the damping the training robot folded in, so it follows the slot's gains.
        """
        self._rig.set_implicit_damping(self.default_dof_kd)
        self._damped_kd = self.default_dof_kd.copy()

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._video is not None:
                self._video.close()
                self._video = None
            if self._viewer is not None:
                try:
                    self._viewer.close()
                except Exception:  # noqa: BLE001 — best-effort on the way down
                    pass
                self._viewer = None

    def _prepare_reset(self) -> None:
        super()._prepare_reset()
        if self._rig is not None:
            self._reset_sim()

    # -- the tick -------------------------------------------------------------

    def send_payload(self, payload: RobotCommand) -> None:
        """Every part writes its command (each on its thread), then the physics advances one control period."""
        if self._rig is None:
            raise RuntimeError("MujocoCompositeDeployment: connect() must be called before send_payload().")
        if not np.array_equal(self.default_dof_kd, self._damped_kd):
            self._fold_implicit_damping()
        super().send_payload(payload)
        self._rig.step(self._n_substeps)
        self._sim_ticks += 1
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        if self._video is not None:
            self._video.tick(self.policy_active, marker_poses(self._env, self._video.markers))

    def read_state(self) -> RobotState:
        state = super().read_state()
        for name in self._prop_names:
            pos, quat = self._rig.prop_pose(name)
            if self._rig.props[name].free and not self._camera.exact:
                # What the policy reads is the camera's; the exact pose stays beside it for a post-mortem.
                state.extra[f"{name}.true_pos"], state.extra[f"{name}.true_quat"] = pos, quat
                pos, quat = self._camera.observe(name, self._sim_ticks, pos, quat)
            state.extra[f"{name}.pos"] = pos
            state.extra[f"{name}.quat"] = quat
        return state

    def _sync_state(self, state: RobotState) -> None:
        """Sync the parts, then pose each moving prop's env entity where MuJoCo has it (a mocap read of the scene).

        In the frame the rig's links are in: a floating root's state is yaw-aligned at START
        (``auto_yaw_align``), which turns its links into the trained heading, so the props turn
        with it -- or every robot-relative object observation would be off by that yaw. A fixed
        root's links are never moved (its sim entity is not posed), so its props stay in the world.
        """
        from eden.utils.geom import quat_mul, transform_by_quat

        super()._sync_state(state)
        device = self._env.device
        root = self._state_entity
        yaw = 0.0 if root.has_immobile_base else root.yaw_offset
        remove = torch.tensor([[math.cos(-0.5 * yaw), 0.0, 0.0, math.sin(-0.5 * yaw)]], device=device)
        # A fixed prop never moves in MuJoCo: it is rewritten only when the frame it is shown in turns.
        turned, self._props_yaw = yaw != self._props_yaw, yaw
        for name in self._prop_names:
            if f"{name}.pos" not in state.extra or not (self._rig.props[name].free or turned):
                continue
            pos = torch.as_tensor(state.extra[f"{name}.pos"], dtype=torch.float32, device=device)[None]
            quat = torch.as_tensor(state.extra[f"{name}.quat"], dtype=torch.float32, device=device)[None]
            if yaw != 0.0:
                pos, quat = transform_by_quat(pos, remove), quat_mul(remove, quat)
            entity = self._env.entities[name]
            entity.set_pos(pos, relative=True)
            entity.set_quat(quat, relative=True)


class _PropCamera:
    """A pose camera on the free props: what the policy reads of each instead of MuJoCo's exact pose.

    :meth:`observe` is handed the exact pose and the simulation tick it is at (control periods since
    the reset). A frame is taken at ``hz`` (every tick when ``None``) of the pose ``latency_ticks``
    ticks ago, with its error -- ``pos_noise`` per axis, a rotation by a ``rot_noise``-normal angle
    about a random axis, and a random quaternion sign once either is on -- and held until the next
    frame. Observing a tick again (a control tick that sent nothing, so the physics stood still)
    returns the same reading: the camera runs on the simulation's clock, not on the reads.
    """

    def __init__(
        self,
        *,
        pos_noise: float,
        rot_noise: float,
        hz: float | None,
        latency_ticks: int,
        control_dt: float,
        seed: int | None,
    ) -> None:
        self.pos_noise, self.rot_noise, self.hz = pos_noise, rot_noise, hz
        self.latency_ticks, self.control_dt = int(latency_ticks), control_dt
        self._rng = np.random.default_rng(seed)
        self.reset()

    @property
    def exact(self) -> bool:
        return self.pos_noise == 0.0 and self.rot_noise == 0.0 and self.hz is None and self.latency_ticks == 0

    def reset(self) -> None:
        # Kept per prop, though every prop is observed on the same tick, so each prop's camera stands alone:
        # a prop first observed later starts its own history and frame count rather than sharing another's.
        self._history: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._held: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_tick: dict[str, int] = {}
        self._frames: dict[str, int] = {}

    def observe(self, name: str, tick: int, pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.exact:
            return pos, quat
        if self._last_tick.get(name) != tick:
            self._last_tick[name] = tick
            self._advance(name, tick, pos, quat)
        held_pos, held_quat = self._held[name]
        return held_pos.copy(), held_quat.copy()

    def _advance(self, name: str, tick: int, pos: np.ndarray, quat: np.ndarray) -> None:
        """Take in the pose at a new tick, and the camera's next frame if one arrives at it."""
        from eden.utils.geom import quat_mul

        history = self._history.setdefault(name, [])
        history.append((np.array(pos, dtype=np.float64), np.array(quat, dtype=np.float64)))
        del history[: -(self.latency_ticks + 1)]
        frame = tick if self.hz is None else math.floor(tick * self.control_dt * self.hz + 1e-9)
        if self._frames.get(name) != frame:  # a new camera frame arrives
            self._frames[name] = frame
            seen_pos, seen_quat = history[0]  # the oldest kept: latency_ticks ago, or the first pose seen
            if self.pos_noise > 0.0:
                seen_pos = seen_pos + self._rng.normal(0.0, self.pos_noise, size=3)
            if self.rot_noise > 0.0:
                axis = self._rng.normal(size=3)
                angle = self._rng.normal(0.0, self.rot_noise)
                error = np.concatenate([[np.cos(0.5 * angle)], np.sin(0.5 * angle) * axis / np.linalg.norm(axis)])
                seen_quat = quat_mul(torch.as_tensor(error)[None], torch.as_tensor(seen_quat)[None])[0].numpy()
            if self.pos_noise > 0.0 or self.rot_noise > 0.0:
                seen_quat = seen_quat * self._rng.choice((-1.0, 1.0))
            self._held[name] = (seen_pos, seen_quat / np.linalg.norm(seen_quat))
