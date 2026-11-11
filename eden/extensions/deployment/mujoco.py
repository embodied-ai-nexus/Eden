"""MuJoCo sim-to-sim deployment backend.

A :class:`~eden.extensions.deployment.base.DeploymentBase` whose "robot" is a MuJoCo
simulation instead of hardware. It runs the *exact* deploy control loop
(``eden.utils.cli.tasks._deploy_control_loop``) — policy loading, the ``RobotStateEntity``
swap, observation assembly, command terms — but reads state from and writes torques to a
MuJoCo model rather than a Unitree DDS wire. That makes it a faithful sim-to-sim harness
for validating a Genesis-trained policy against an independent physics engine before it
ever touches a real robot.

Inspired by Unitree's ``unitree_mujoco`` bridge (github.com/unitreerobotics/unitree_mujoco),
which stands up a MuJoCo world speaking the Unitree ``LowState``/``LowCmd`` DDS protocol.
This backend takes the simpler, robot-agnostic route: it drives MuJoCo directly (no DDS,
no ``unitree_sdk2py``), so it works for any robot Eden can describe — the two Unitree
families here (G1, Go2) and anything else with a ``RobotOptions`` and a loadable
MJCF/URDF. The engine details live in
:class:`~eden.extensions.deployment.utils.mujoco_sim.MujocoSim`; this class is the glue
between it and the Eden env.

Selected at deploy time with ``eden deploy ... --mujoco`` (see ``deploy_task``), which
overrides the usual robot→backend binding so any task/robot can be routed through MuJoCo.

Physics can run in **lockstep** with the control loop (the default: each command advances
the simulation exactly one control period, so a rollout is deterministic and a late policy
costs nothing) or **asynchronously** (``--async`` / ``async_physics=True``: the simulation
advances on its own thread in wall-clock time, like a robot's controller, so a late policy
is late for real). See :class:`~eden.extensions.deployment.utils.realtime_physics.RealtimePhysics`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import time

import numpy as np

import eden as en
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY, DeploymentBase
from eden.extensions.deployment.utils.menagerie import MENAGERIE_DATASET, MENAGERIE_REGISTRY, menagerie_path_for
from eden.extensions.deployment.utils.mujoco_scene import (
    kinematic_markers,
    marker_poses,
    read_floor,
    say_floor_caveat,
)
from eden.extensions.deployment.utils.mujoco_sim import (
    MujocoFloor,
    MujocoSim,
    divides,
    fitting_timestep,
    steps_per_tick,
)
from eden.extensions.deployment.utils.realtime_physics import PDCommand, RealtimePhysics
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.utils.assets import get_asset_path

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.extensions.deployment.utils.mujoco_video import MujocoVideo
    from eden.options.extensions.deployment import DeploymentOptions


@DEPLOYMENT_REGISTRY.register()
class MujocoDeployment(DeploymentBase):
    """Deploy a policy against a MuJoCo simulation of the robot (sim-to-sim).

    Configurable options (in addition to :class:`DeploymentBase`'s):

    mjcf_path: str | None
        Explicit model file to load into MuJoCo, bypassing ``mjcf_source`` resolution.
        Point this at any hand-authored MuJoCo scene for one-off sim-to-sim runs.
    mjcf_source: str
        Which model to load when ``mjcf_path`` is unset (default ``"eden"``):

        - ``"eden"``: the robot's own Genesis training asset (``robot_options.file``).
          Always available; the MuJoCo backend reconstructs armature/floor for it.
        - ``"menagerie"``: a community-tuned `MuJoCo Menagerie
          <https://github.com/google-deepmind/mujoco_menagerie>`_ model (with the
          armature/damping/contact params MuJoCo needs), fetched from an HF dataset — the
          higher-fidelity, more independent sim-to-sim target. Errors if the robot has no
          Menagerie mapping or the download fails. Selected by ``eden deploy --menagerie``.
        - ``"auto"``: Menagerie when the robot is mapped and the download succeeds,
          otherwise fall back to the Eden asset. Not reachable from the CLI, which keeps
          the loaded model deterministic.
    menagerie_registry: str
        HF org for the Menagerie dataset (default ``"Kashu7100"``).
    menagerie_dataset: str
        HF dataset name mirroring MuJoCo Menagerie (default ``"mujoco_menagerie"``).
    sim_dt: float | None
        MuJoCo integration timestep. ``None`` keeps the asset's timestep, or in lockstep, when it
        does not divide the control period, takes the largest step under it that does. In
        lockstep each step advances ``control_dt / sim_dt`` substeps, so a given ``sim_dt`` must
        divide the control period: one that does not is refused rather than rounded, which would
        run the physics on a clock ahead of or behind the policy's.
    integrator: str
        MuJoCo integrator (``"implicitfast"`` default; also ``"implicit"``, ``"euler"``,
        ``"rk4"``). ``implicitfast`` keeps a stiff joint PD stable.
    add_floor: bool
        Add a ground plane if the model has none (default True). It stands where the scene's ground
        plane does, raised or tilted alike, with its friction
        (:func:`~eden.extensions.deployment.utils.mujoco_scene.read_floor`), but keeps MuJoCo's
        default solver parameters: the robot's own geoms keep its asset's, and a contact's mix of a
        Genesis value and a MuJoCo one would be neither engine's.
    provide_ground_truth_base: bool
        Feed MuJoCo's exact base position and linear velocity to the observation pipeline
        (default True). Unlike hardware — where these must be estimated — the simulator
        knows them, so policies that observe base position / linear velocity work without
        ``--estimate_lin_vel``. Set False to emulate a hardware-blind base.

        Combining it with ``estimate_lin_vel`` is the useful sim-to-sim rehearsal of the
        estimator, not a conflict: the estimate is installed *after* the ground-truth
        reading, so it is what the policy observes, while the truth stays alongside it in
        ``state.extra["base_lin_vel"]`` for the estimate
        (``state.extra["base_lin_vel_estimate"]``) to be scored against.
    imu_acc_noise, imu_gyro_noise: float
        Standard deviation of white noise added to every accelerometer (m/s^2) and gyro (rad/s)
        reading, per axis. Default 0: MuJoCo's IMU is otherwise exact, which hides how an
        IMU-driven estimate behaves on hardware. Under ``estimate_lin_vel`` an exact accelerometer
        barely drifts even with no foot contact, so a contact dropout that sends the real robot's
        estimate to -0.5 m/s stays at 0.03 m/s in sim. Turn these on to rehearse the
        estimator against a hardware-like IMU.
    imu_acc_bias, imu_gyro_bias: tuple[float, float, float]
        Constant bias added to the accelerometer (m/s^2) and gyro (rad/s) readings, body frame.
        Default zero.
    imu_noise_seed: int | None
        Seed for the IMU noise generator, so a noisy run can be repeated. ``None`` (default)
        draws a fresh seed.
    show_viewer: bool
        Open MuJoCo's native passive viewer to watch the sim-to-sim rollout.
    record_video: str | None
        Write the rollout from the policy's first START on to this mp4, rendered offscreen from MuJoCo
        (:class:`~eden.extensions.deployment.utils.mujoco_video.MujocoVideo`: two views, one frame per
        control tick, a STOP's holds included). Lockstep only. Headless, it needs ``MUJOCO_GL=egl``.
    gantry: bool
        Hang the robot from a simulated overhead hoist -- the sim counterpart of the bring-up
        rig, so a biped can be held upright while a controller is commissioned instead of
        falling every time control pauses. See
        :meth:`~eden.extensions.deployment.utils.mujoco_sim.MujocoSim.set_gantry` for the rope
        model; it is a tension-only spring-damper to a fixed anchor, attached above the centre
        of mass so it produces a real restoring torque rather than just cancelling weight.

        Re-rigged on every :meth:`reset` from wherever the robot then stands, so the anchor
        follows a re-placed robot instead of yanking it back to the first episode's spot.
    gantry_body: tuple[str, ...]
        Bodies the ropes are tied to, one rope each. Empty (default) uses the robot's own
        :attr:`~eden.options.entities.RobotOptions.gantry_links_name` — the two shoulders for a
        humanoid that declares them — and falls back to the floating base for a robot that
        declares none.
    gantry_attach: tuple[float, float, float] | None
        Attachment point within each rope's body frame, metres. ``None`` (default) picks it from
        how the robot is rigged: the link origin for declared mount links, which for a shoulder
        sits on the pitch axis at the top of the torso, or 25 cm up for a single base rope,
        which needs the height to have any moment arm at all.
    gantry_height: float | None
        World height of the anchor. Defaults to 1 m above the attachment point.
    gantry_step: float
        Metres the operator's ``+`` / ``-`` keys move the anchor per press.
    gantry_draw: bool
        Draw each rope and the force it is pulling with in the ``--vis`` viewer (default on).
        Seeing the pull is most of the value of a simulated rig: a rope that is quietly carrying
        the robot looks exactly like a robot that is standing, right up until you unrig it.
    gantry_force_scale: float
        Metres of force arrow per newton (default 1e-3, so 1 kN draws as 1 m).
    gantry_stiffness, gantry_damping:
        Rope spring-damper, N/m and N.s/m.
    gantry_slack: float
        Metres of slack. ``0`` (default) is taut and holds station; a positive value is a
        safety tether that does nothing until the robot drops that far.
    gantry_support: float
        Scale on the rope force, ``0``-``1``. Ramp down to hand weight back to the controller.
    init_steps: int
        Control cycles the INIT ramp takes to reach the start pose — the sim-to-sim
        counterpart of the hardware backends' INIT state, inherited from
        :class:`DeploymentBase` along with its default of 250 (5 s at 50 Hz). The ramp runs
        when the operator presses INIT, or on its own in a headless run. The policy does
        not drive straight from the reset pose: this backend runs the same INIT/START
        sequence the robot does.
    init_target: str
        Inherited from :class:`DeploymentBase`; this backend is the one that also accepts
        ``"motion_rsi"``, which places joints *and* base at the tracked reference's frame 0
        on reset. ``"motion"`` and ``"default"`` are the same joint-space ramps here as on
        the wire.
    async_physics: bool
        Run the simulation on its own thread in wall-clock time (default False). Off, physics
        advances ``control_dt / sim_dt`` substeps inside each ``send_payload`` and then
        waits for the next command, so the control loop's timing has no effect on the rollout
        -- deterministic, and the right mode for comparing two runs. On, the simulation is
        the robot: it steps at ``1 / sim_dt`` whether or not a new command has arrived,
        ``send_payload`` only replaces the command it holds, and ``read_state`` returns the
        latest state it has published. A policy that overruns the control period therefore
        leaves the robot executing its previous command for the overrun, and observes a
        correspondingly older state -- the behaviour hardware has and lockstep cannot show.
        Selected by ``eden deploy --mujoco --async``.

    The simulated robot has no command latency. A task's
    :class:`~eden.managers.modifiers.actions.actuators.ActionDelay` models the Genesis actuator's
    latency, and no deployment backend applies it: each command acts from the substep it
    arrives on. A policy trained against a delay therefore runs here with less latency than it trained
    with. This matters most for a fixed delay (both ends of the range equal), which the policy expects
    on every step. ``--async`` adds only the control loop's real lateness, not the trained delay.
    ``eden inference --async`` is the run that keeps the delay.

    Being robot-agnostic, this backend knows nothing robot-specific — including the feet the
    ``estimate_lin_vel`` filter needs. It does not have to: the robot declares them itself
    (``RobotOptions.feet_links_name``), so ``eden deploy --mujoco --estimate_lin_vel`` needs no
    further flag for any robot that names its feet.
    """

    mjcf_path: str | None = None
    mjcf_source: str = "eden"
    menagerie_registry: str = MENAGERIE_REGISTRY
    menagerie_dataset: str = MENAGERIE_DATASET
    sim_dt: float | None = None
    integrator: str = "implicitfast"
    add_floor: bool = True
    provide_ground_truth_base: bool = True
    gantry: bool = False
    gantry_body: tuple[str, ...] = ()
    gantry_attach: tuple[float, float, float] | None = None
    gantry_height: float | None = None
    gantry_step: float = 0.05
    gantry_draw: bool = True
    gantry_force_scale: float = 1.0e-3
    gantry_stiffness: float = 5000.0
    gantry_damping: float = 200.0
    gantry_slack: float = 0.0
    gantry_support: float = 1.0
    imu_acc_noise: float = 0.0
    imu_gyro_noise: float = 0.0
    imu_acc_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_noise_seed: int | None = None
    show_viewer: bool = False
    record_video: str | None = None
    supports_base_placement: ClassVar[bool] = True
    # The robot is a simulation, so a headless run may press its own INIT and START.
    allow_unattended_start: ClassVar[bool] = True
    operator_guide_title: ClassVar[str] = "MUJOCO SIM-TO-SIM DEPLOY CONTROLS  (keyboard — same sequence as the robot)"
    async_physics: bool = False

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)
        # A note, not a check, so it stays out of _resolve_options.
        if self.estimate_lin_vel and self.provide_ground_truth_base:
            en.logger.info(
                "[deploy] MuJoCo sim-to-sim: the policy observes the *estimated* base_lin_vel "
                "(estimate_lin_vel overrides the ground-truth reading); the simulator's truth "
                "rides along in state.extra['base_lin_vel'] for comparison. Set "
                "provide_ground_truth_base=False to also blind the base position."
            )
        self._sim: MujocoSim | None = None
        self._physics: RealtimePhysics | None = None
        # The kd the armature currently folds in (``MujocoSim.set_implicit_damping``); set at connect.
        self._damped_kd: np.ndarray | None = None
        self._viewer = None
        self._video: MujocoVideo | None = None
        self._n_substeps = 1
        self._base_pos0 = np.asarray(self.robot_options.default_root_pos, dtype=np.float64)
        self._base_quat0 = np.asarray(self.robot_options.default_root_quat, dtype=np.float64)
        self._imu_rng = np.random.default_rng(self.imu_noise_seed)

    def _resolve_options(self) -> None:
        """Reject a negative IMU noise level, or a lockstep ``sim_dt`` that does not divide the control period."""
        super()._resolve_options()
        for name in ("imu_acc_noise", "imu_gyro_noise"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} is a standard deviation and must be >= 0, got {getattr(self, name)}.")
        if self.sim_dt is not None and not self.async_physics:
            steps_per_tick(self.control_dt, self.sim_dt)
        if self.record_video is not None and self.async_physics:
            raise ValueError(
                "record_video renders a frame each lockstep tick, and async_physics steps MuJoCo on its own thread, "
                "under the render. Drop one: --mujoco_record or --async."
            )

    # -- lifecycle ---------------------------------------------------------

    def _eden_asset(self) -> str:
        return self.robot_options.fetch_asset()

    def _resolve_asset(self) -> str:
        """Pick the MuJoCo model file per ``mjcf_path`` / ``mjcf_source`` (see class docstring)."""
        if self.mjcf_path:
            return self.mjcf_path
        if self.mjcf_source not in ("auto", "eden", "menagerie"):
            raise ValueError(f"mjcf_source must be 'auto', 'eden', or 'menagerie'; got {self.mjcf_source!r}.")

        if self.mjcf_source in ("auto", "menagerie"):
            men_path = menagerie_path_for(self.robot_options)
            if men_path is not None:
                try:
                    asset = get_asset_path(men_path, registry=self.menagerie_registry, dataset=self.menagerie_dataset)
                    en.logger.info(f"[deploy] MuJoCo sim-to-sim using Menagerie model: {men_path}")
                    return asset
                except Exception as exc:  # noqa: BLE001 — network/asset failure; auto degrades to Eden asset
                    if self.mjcf_source == "menagerie":
                        raise
                    en.logger.warning(
                        f"[deploy] Menagerie model '{men_path}' unavailable ({exc}); "
                        "falling back to the Eden training asset."
                    )
            elif self.mjcf_source == "menagerie":
                raise ValueError(
                    f"No Menagerie mapping for {type(self.robot_options).__name__}. "
                    "Drop --menagerie to use the robot's own training asset, or add a mapping "
                    "in deployment/utils/menagerie.py."
                )
        return self._eden_asset()

    def connect(self) -> None:
        asset = self._resolve_asset()
        floor = None
        if self.add_floor:
            # The ground's pose and friction; MuJoCo's solver parameters, as the robot's geoms keep the asset's.
            floor = dataclasses.replace(read_floor(self._env), sol_params=None)
        self._sim = self._load_sim(asset, self.sim_dt, floor)
        if self.sim_dt is None and not self.async_physics and not divides(self.control_dt, self._sim.sim_dt):
            # Lockstep ends every tick on a step: an asset's timestep off the control period gives way to
            # the largest under it that divides it, as a composite rig's default does.
            fit = fitting_timestep(self.control_dt, self._sim.sim_dt)
            en.logger.info(
                f"[deploy] MuJoCo sim-to-sim: the asset's timestep {self._sim.sim_dt:g} s does not divide the control "
                f"period {self.control_dt:g} s; running at {fit:g} s, the largest under it that does "
                "(--mujoco_sim_dt sets another)."
            )
            self._sim = self._load_sim(asset, fit, floor)
        say_floor_caveat(self._sim.floor)
        # Genesis integrates the PD damping implicitly (its implicit integrators) or folds kd*dt into
        # the armature (``euler``, RigidEntity.post_build); match that here, or a hand-rolled PD on the
        # zero-armature training MJCF diverges.
        self._fold_implicit_damping()
        if self.async_physics:  # the physics thread keeps its own clock, at 1 / sim_dt
            self._n_substeps = max(1, round(self.control_dt / self._sim.sim_dt))
        else:
            self._n_substeps = steps_per_tick(self.control_dt, self._sim.sim_dt)
        self._sim.reset(self.default_dof_pos, self._base_pos0, self._base_quat0)
        mode = "asynchronous, physics on its own thread" if self.async_physics else "lockstep with the control loop"
        en.logger.info(
            f"[deploy] MuJoCo sim-to-sim: {asset} | sim_dt={self._sim.sim_dt:g}s "
            f"× {self._n_substeps} substeps/step | integrator={self.integrator} | {mode}"
        )
        if self.show_viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._sim.model, self._sim.data)
        if self.async_physics:
            overlay = self._draw_gantry_overlay if self.gantry and self.gantry_draw else None
            self._physics = RealtimePhysics(
                self._sim, self._hold_default_pose(), viewer=self._viewer, viewer_overlay=overlay
            )
            self._physics.start()
        if self.record_video is not None:
            from eden.extensions.deployment.utils.mujoco_video import MujocoVideo

            self._video = MujocoVideo(
                self._sim.model,
                self._sim.data,
                self.record_video,
                fps=1.0 / self.control_dt,
                framed_bodies=self._sim.robot_body_ids(),
                markers=kinematic_markers(self._env),
            )

    def _load_sim(self, asset: str, sim_dt: float | None, floor: MujocoFloor | None) -> MujocoSim:
        return MujocoSim(
            asset,
            self.dofs_name,
            sim_dt=sim_dt,
            floating_base=not self.robot_options.is_fixed_base,
            floor=floor,
            integrator=self.integrator,
        )

    def _fold_implicit_damping(self) -> None:
        """Set the armature's implicit damping to the active slot's ``default_dof_kd`` (see :meth:`send_payload`).

        The caller excludes the physics thread, if there is one.
        """
        self._sim.set_implicit_damping(self.default_dof_kd)
        self._damped_kd = self.default_dof_kd.copy()

    def _hold_default_pose(self) -> PDCommand:
        """The command the asynchronous physics holds before (and between) the policy's: the default pose."""
        return PDCommand.hold(self.default_dof_pos, self.default_dof_kp, self.default_dof_kd)

    @contextmanager
    def _sim_exclusive(self) -> Iterator[None]:
        """Touch the simulator outside a command: excludes the asynchronous physics thread, if any.

        Lockstep physics steps only inside :meth:`send_payload`, on this thread, so there is
        nothing to exclude; asynchronous physics is stepping concurrently, and a rope rigged
        or cranked mid-burst would be read half-written.
        """
        if self._physics is None:
            yield
        else:
            with self._physics.exclusive():
                yield

    def _draw_gantry_overlay(self, viewer) -> None:
        """Redraw the gantry ropes and their forces into the viewer's user scene."""
        # user_scn persists between syncs, so rebuild it each frame rather than append.
        viewer.user_scn.ngeom = 0
        self._sim.draw_gantry(viewer.user_scn, force_scale=self.gantry_force_scale)

    def close(self) -> None:
        if self._physics is not None:
            # Stop the robot's clock before anything else, and let it grade itself: a physics
            # thread that fell behind wall time silently stretched every control period.
            self._physics.stop()
            log = en.logger.warning if self._physics.reanchors else en.logger.info
            log(self._physics.summary())
            self._physics = None
        self._teardown_common()
        if self._video is not None:
            self._video.close()
            self._video = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:  # noqa: BLE001 — best-effort on the way down
                pass
            self._viewer = None
        self._sim = None

    def _prepare_reset(self) -> None:
        """Put the simulated robot back at its default pose: the start of :meth:`DeploymentBase.reset`.

        The reset then runs the same INIT/START/STOP/KILL sequence the hardware backends do, so a
        sim-to-sim rollout rehearses the real deployment procedure rather than a shortcut through
        it: the robot boots in STOP holding its pose (every hold tick steps MuJoCo, so it stands
        there under PD exactly as the robot does on its feet), and nothing moves until the operator
        presses INIT and then START.
        """
        assert self._sim is not None, "connect() must be called before reset()."
        if self._physics is not None:
            self._physics.reset(self.default_dof_pos, self._base_pos0, self._base_quat0, hold=self._hold_default_pose())
        else:
            self._sim.reset(self.default_dof_pos, self._base_pos0, self._base_quat0)

    def _after_reset_read(self) -> None:
        """Place the robot (``init_target="motion_rsi"``), capture its yaw and rig the gantry, on the reset's read.

        ``"motion_rsi"`` teleports the robot to the clip's first frame here, after the reset
        rewound the clip and before the operator takes over; that is the one sim-only step, and
        the hold that follows lets contacts settle while it waits for INIT. The operator state
        machine holds the clock at frame 0 until START, so the clip still begins at the handover.
        """
        if self.init_target == "motion_rsi":
            self._place_at_motion_start()
        self.state = self.read_state()
        if self.auto_yaw_align and self.state.base_quat is not None:
            yaw = self._state_entity.capture_yaw_offset(self.state.base_quat)
            en.logger.info(f"[deploy] MuJoCo captured base yaw offset: {np.degrees(yaw):.2f} deg")
        self._rig_gantry()

    def _rig_gantry(self) -> None:
        """(Re-)hang the robot from the overhead rope, or clear it when disabled.

        Rigged here rather than at ``connect`` so the anchor is set from where the robot
        actually stands after this reset -- including an ``init_target='motion_rsi'``
        placement, which would otherwise leave the rope anchored over the previous pose.
        """
        assert self._sim is not None, "connect() must be called before reset()."
        mounts = self._gantry_mounts()
        with self._sim_exclusive():
            self._sim.set_gantry(
                enabled=self.gantry,
                mounts=mounts,
                height=self.gantry_height,
                stiffness=self.gantry_stiffness,
                damping=self.gantry_damping,
                slack=self.gantry_slack,
                support=self.gantry_support,
            )
        if self.gantry:
            where = ", ".join(f"{body or 'base'}{list(attach)}" for body, attach in mounts)
            en.logger.info(
                f"[deploy] Gantry rigged: {len(mounts)} rope(s) on {where}, anchor at "
                f"{self._sim.gantry_anchor_height:.3f} m, "
                f"k={self.gantry_stiffness:g} N/m, c={self.gantry_damping:g} N.s/m, "
                f"slack={self.gantry_slack:g} m, support={self.gantry_support:g}. "
                "SIM-ONLY — it is the rig, not the robot."
            )

    def rig_gantry(self, enabled: bool) -> None:
        """Rig or release the gantry mid-run, anchored over where the robot stands now.

        The switch a caller driving its own loop needs to hand the robot's weight back to the
        controller (and to catch it again); ``gantry`` itself only sets how a reset rigs it.
        """
        self.gantry = enabled
        self._rig_gantry()
        if not enabled and self._viewer is not None:
            # Nothing redraws the overlay once the gantry is off; drop the last-drawn ropes.
            self._viewer.user_scn.ngeom = 0

    def _gantry_mounts(self) -> tuple[tuple[str | None, tuple[float, float, float]], ...]:
        """Where the ropes tie on: ``--gantry_body``, else the robot's declared mount links.

        A humanoid that names its shoulders is hung from both — two ropes resist roll as a
        couple, which a single rope over the centre cannot. A robot that declares none hangs
        from its base by one rope, raised 25 cm so it has a moment arm at all.
        """
        bodies: tuple[str | None, ...] = tuple(self.gantry_body) or tuple(self.robot_options.gantry_links_name)
        if not bodies:
            return ((None, self.gantry_attach or (0.0, 0.0, 0.25)),)
        attach = self.gantry_attach or (0.0, 0.0, 0.0)
        return tuple((body, attach) for body in bodies)

    def _adjust_gantry(self, sign: float) -> None:
        """Crank the rig up (+) or down (-) by ``gantry_step``, live, mid-run."""
        if self._sim is None or not self.gantry:
            super()._adjust_gantry(sign)
            return
        with self._sim_exclusive():
            height = self._sim.adjust_anchor(sign * self.gantry_step)
        if height is not None:
            direction = "raised" if sign > 0 else "lowered"
            en.logger.info(f"[deploy] Gantry {direction} to {height:.3f} m.")

    def _motion_carries_base(self) -> bool:
        """True when the clip behind ``entity_name`` actually supplies a free base pose.

        This cannot be asked of the per-step buffer. ``MotionReferenceData`` allocates every
        field through ``make_empty``, so a dataset that publishes no base still reads back as
        ``base_pos_w`` zeros and an identity ``base_quat_w`` -- never ``None``. Placing a
        floating base at those zeros drops the robot through the floor at the world origin.
        :func:`assert_motion_field_present` inspects the source ``BatchMotionSequence`` instead,
        and honours its ``is_free_floating`` flag.

        The entity is already known to be tracked when this runs, so the only objections left
        for that helper to raise are the two being asked about -- the dataset does not populate
        the field, or the sequence is flagged fixed-base. Both mean "joints only", which is a
        supported start, not an error; the reason is logged rather than swallowed.
        """
        from eden.utils.motion_reference.lookup import assert_motion_field_present

        try:
            for field in ("base_pos_w", "base_quat_w"):
                assert_motion_field_present(self._env, self.entity_name, field)
        except ValueError as exc:
            en.logger.info(f"[deploy] init_target='motion_rsi': the clip carries no base pose ({exc}) — joints only.")
            return False
        return True

    def _motion_start_state(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None:
        """Frame-0 joint pose (wire order) + base pose of the tracked reference, or ``None``.

        ``None`` when nothing in this env tracks ``entity_name`` -- no motion manager at all, or
        one whose terms target other entities -- so the caller can fall back to the default-pose
        ramp. The base pose is ``None`` when the clip supplies none (see
        :meth:`_motion_carries_base`); the caller must then leave the base where the reset put it
        rather than pass that ``None`` on to ``MujocoSim.reset``, which reads it as the origin.
        """
        dofs = self._motion_start_dofs()
        if dofs is None:
            return None
        from eden.utils.motion_reference.lookup import current_motion_frame

        ref = self._motion_manager.data[self.entity_name]
        base_pos = base_quat = None
        if self._motion_carries_base():
            base_pos = current_motion_frame(ref.base_pos_w)[0].detach().cpu().numpy().astype(np.float64)
            base_quat = current_motion_frame(ref.base_quat_w)[0].detach().cpu().numpy().astype(np.float64)
        return dofs, base_pos, base_quat

    def _place_at_motion_start(self) -> None:
        """Teleport joints *and* base to the reference's frame 0 (``init_target='motion_rsi'``).

        Reference-state initialization, and the only sim-only move in the deploy path: a clip
        that starts crouched (an Oli reference clip sits 0.48 m below the standing reset
        height) is not reachable by folding the legs while the base stays put, and ramping the
        joints into it open-loop simply topples. Nothing on a robot can place its own base, so
        the hardware backends reject this target -- see
        :meth:`~eden.extensions.deployment.base.DeploymentBase._validate_init_target`.

        The operator hold that follows serves as the settle; the INIT ramp is unrelated and
        drives to ``init_target`` through the wire like any other backend.
        """
        start = self._motion_start_state()
        if start is None:
            en.logger.info("[deploy] init_target='motion_rsi' but this env tracks no motion; keeping the reset pose.")
            return
        dofs, base_pos, base_quat = start
        where = "" if base_pos is None else f", base at {np.round(base_pos, 3).tolist()}"
        # MujocoSim.reset reads a None base as the world origin, so keep the reset pose instead.
        base_pos = self._base_pos0 if base_pos is None else base_pos
        base_quat = self._base_quat0 if base_quat is None else base_quat
        if self._physics is not None:
            # Through the robot's clock, which excludes its thread and restarts its schedule;
            # it then holds frame 0 until the operator's first command, as the STOP hold does.
            hold = PDCommand.hold(dofs, self.default_dof_kp, self.default_dof_kd)
            self._physics.reset(dofs, base_pos, base_quat, hold=hold)
        else:
            self._sim.reset(dofs, base_pos, base_quat)
        en.logger.info(f"[deploy] MuJoCo: placed at motion frame 0{where}.")

    def _publishes_base_lin_vel(self) -> bool:
        # Mirrors the ``extra`` branch of :meth:`read_state`: ground truth for a floating base.
        return bool(self.provide_ground_truth_base) and not self.robot_options.is_fixed_base

    def read_state(self) -> RobotState:
        assert self._sim is not None, "connect() must be called before read_state()."
        if self._physics is not None:
            # The latest state the physics thread published, stamped when it was published:
            # in this mode the state has an age, as a wire's does.
            r = self._physics.read()
            stamp = r["stamp"]
        else:
            r = self._sim.read()
            stamp = time.time()
        extra: dict[str, Any] = {}
        if self.provide_ground_truth_base and not self.robot_options.is_fixed_base:
            # The simulator knows the true base pose/velocity; hand them to the
            # observation pipeline so base-position / base-lin-vel terms work without
            # an estimator (RobotStateEntity yaw-aligns them like any world-frame read).
            extra["base_pos"] = r["base_pos"]
            extra["base_lin_vel"] = r["base_lin_vel"]
        return RobotState(
            stamp=stamp,
            base_quat=r["base_quat"],  # wxyz
            base_ang_vel=self._corrupt_imu(r["base_ang_vel"], self.imu_gyro_noise, self.imu_gyro_bias),  # body (gyro)
            base_lin_acc=self._corrupt_imu(r["base_lin_acc"], self.imu_acc_noise, self.imu_acc_bias),  # specific force
            dofs_pos=r["dofs_pos"],
            dofs_vel=r["dofs_vel"],
            dofs_torque=r["dofs_torque"],
            extra=extra,
        )

    def _corrupt_imu(self, reading: np.ndarray, noise: float, bias: tuple[float, float, float]) -> np.ndarray:
        """Add the configured bias and white noise to one exact IMU reading (see ``imu_acc_noise``)."""
        out = reading + np.asarray(bias, dtype=np.float64)
        if noise > 0.0:
            out = out + self._imu_rng.normal(0.0, noise, size=out.shape)
        return out

    def send_payload(self, payload: RobotCommand) -> None:
        assert self._sim is not None, "connect() must be called before send_payload()."
        # As every hardware backend does on its wire, so a --target_lpf rehearsed here is the one the robot gets.
        payload.dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))
        kp = payload.dofs_kp if payload.dofs_kp is not None else self.default_dof_kp
        kd = payload.dofs_kd if payload.dofs_kd is not None else self.default_dof_kd
        # The armature models the damping Genesis folded in at build from the robot the active slot
        # trained on, so it follows the slot's gains (``default_dof_kd``, which a slot switch ramps
        # through ``set_gains``) -- not a payload's own kd, such as the INIT ramp's
        # ``init_dofs_kd_scale``, which was never part of any training robot.
        refold = not np.array_equal(self.default_dof_kd, self._damped_kd)
        if self._physics is not None:
            # Hand the command to the robot's clock; it takes effect at the thread's next
            # substep, and the physics keeps running whether or not another one follows.
            command = PDCommand.of(payload.dofs_pos, payload.dofs_vel, kp, kd, payload.dofs_torque)
            if not refold:
                self._physics.set_command(command)
                return
            # One exclusive section for both, so no burst runs the new damping with the old
            # command or the old damping with the new one.
            with self._physics.exclusive():
                self._fold_implicit_damping()
                self._physics.set_command(command)
            return
        if refold:
            self._fold_implicit_damping()
        self._sim.step_pd(
            target_pos=payload.dofs_pos,
            target_vel=payload.dofs_vel,
            kp=kp,
            kd=kd,
            tau_ff=payload.dofs_torque,
            n_substeps=self._n_substeps,
        )
        if self._video is not None:
            self._video.tick(self.policy_active, marker_poses(self._env, self._video.markers))
        if self._viewer is not None and self._viewer.is_running():
            if self.gantry and self.gantry_draw:
                self._draw_gantry_overlay(self._viewer)
            self._viewer.sync()
