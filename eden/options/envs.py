"""Top-level environment configuration options (sim, solvers, batching)."""

from typing import ClassVar, Literal

import genesis as gs
from genesis.typing import Vec2FType, Vec3FType
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt

from eden.engine.solvers.mochi.options import MochiOptions
from eden.options.options import ConfigurableOptions


class CudaGraphOptions(ConfigurableOptions):
    """How ``EnvOptions.cuda_graph_options`` captures, validates and re-captures the graded phase's CUDA graph.

    The graph is captured after ``warmup_steps`` eager steps. That step and the next ``validation_steps`` are each
    run eagerly too and compared bit for bit with the graph over every tensor the phase leaves behind: a term the
    graph cannot reproduce raises, naming the tensors that differ, rather than training on stale values. A change
    to a graded term's configured fields (a curriculum moving a termination threshold), to the env's ``dt``,
    ``max_episode_length`` or ``training``, or a tensor the graph reads rebound to a new one, re-captures and
    re-validates; reward weights are read from the device and need none. Reading ``env.common_step_counter``
    during the capture raises, naming the term's line. Not watched: any other Python value a term reads from
    outside its own fields (another manager's term, a module global), which the graph replays as it was at
    capture.

    Parameters
    ----------
    warmup_steps: int
        Eager graded steps before the first capture. A term builds its lazy caches (index tensors, resolved
        link ids) on its first calls, and building one inside a capture is a host-to-device copy, which fails.
        These steps already read contacts padded, as the capture will, so a cache sized by the contact width
        fits it. Defaults to 2.
    validation_steps: int
        How many replays after each capture are checked against an eager run of the same step. A validated
        replay runs the graph before the eager phase, which is what catches work a capture could not record (a
        kernel launched outside torch's stream), so at least one is required. Each costs about one eager graded
        phase plus a copy of the tensors the phase writes. Defaults to 3.
    recapture_window: int
        The span, in graded steps, that ``max_recaptures_per_window`` counts re-captures over. Defaults to 100.
    max_recaptures_per_window: int
        Re-captures allowed within ``recapture_window`` steps; one more raises, naming what changed. A one-off
        change (a curriculum switching stage) re-captures once, but something that changes a baked-in value on
        nearly every step would re-capture and re-validate every step, which costs more than the eager phase.
        Defaults to 2.
    """

    warmup_steps: PositiveInt = 2
    validation_steps: PositiveInt = 3
    recapture_window: PositiveInt = 100
    max_recaptures_per_window: NonNegativeInt = 2


class EnvOptions(ConfigurableOptions):
    """Top-level environment configuration (sim, solvers, batching).

    Parameters
    ----------
    num_envs: int
        Number of environments to run in parallel.
    sim_dt: float
        Simulation time step.
    sim_substeps: int
        Number of simulation substeps.
    decimation: int
        Physics steps per control (env) step. The control step is `sim_dt * decimation`.
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    episode_length_s: float
        Length of the episode in seconds.
    use_gjk_collision: bool
        Whether to use GJK collision detection. Defaults to True. If False, MPR+GJK will be used.
    solver: Literal["newton", "cg"]
        Constraint solver (Newton or conjugate gradient).
    integrator: Literal["approximate_implicitfast", "implicitfast", "euler"]
        Rigid-body integrator. ``"approximate_implicitfast"`` (default) is Genesis' own fast scheme.
        ``"implicitfast"`` and ``"euler"`` are MuJoCo's integrators of the same name, for tasks ported
        from a MuJoCo stack: ``"implicitfast"`` integrates joint damping and the position actuators'
        velocity gain implicitly, as MuJoCo does, and ``"euler"`` integrates the PD controller
        explicitly, which is why an actuated entity then folds ``kd * substep_dt`` into its armature.
        Only damping Genesis itself applies is ever implicit: a torque-mode action term
        (``control_dofs_force``, e.g. ``ExplicitPDController``) computes ``-kd * qdot`` itself, and it
        stays explicit under all three (see "Implicit Damping" in ``docs/guides/entities.md``).
    impratio: float | None
        Ratio of frictional to normal constraint impedance at contacts (MuJoCo's ``impratio``); must be
        positive. ``None`` (default) lets Genesis resolve it: 100 with the elliptic cone, 1 otherwise.
    constraint_timeconst: float | None
        Time constant for the constraint solver. If None, it will be computed as max(0.01, 2 * sim_dt / sim_substeps).
    max_collision_pairs: int
        Maximum number of collision pairs.
    iterations: int
        Maximum constraint-solver iterations.
    tolerance: float | None
        Convergence tolerance for the constraint solver, as a ratio against the scene's
        free-motion cost. If None, Genesis resolves it from the floating-point precision
        (1e-6 for single, 1e-9 for double). Defaults to None — pinning a literal here means
        a Genesis release that re-scales what the ratio is quoted against (as
        https://github.com/Genesis-Embodied-AI/Genesis/pull/3194 did) silently moves Eden's convergence gate.
    ls_iterations: int
        Maximum line-search iterations per constraint-solver iteration.
    ls_tolerance: float
        Line-search convergence tolerance.
    enable_self_collision: bool
        Whether to enable self collision detection. Defaults to True.
    enable_neutral_collision: bool
        Keep self-collision pairs that already overlap in an entity's neutral configuration (qpos0). Defaults to
        False, where Genesis drops every such pair for good. Its overlap test is a voxel approximation, so it can
        drop a pair that only comes within millimetres at qpos0 -- set True to keep the MJCF's full collision set.
        The switch is **scene-wide** (a rigid-solver option, not a per-entity one): True keeps the pairs that
        overlap at qpos0 on *every* articulated entity in the scene, so a second robot whose collision meshes
        genuinely interpenetrate at qpos0 then fights itself from the first step.
    batch_links_info: bool
        Whether to batch links information. Defaults to False.
    batch_joints_info: bool
        Whether to batch joints information. Defaults to False.
    batch_dofs_info: bool
        Whether to batch DOFs information. Defaults to True.
    enable_multi_contact: bool
        Whether to enable multi-contact detection. Defaults to True.
    enable_torsional_friction: bool
        Opt-in torsional friction: whether contacts also resist relative spin about their
        normal (one extra constraint axis per contact). Defaults to False. Enable it when spin
        resistance matters (e.g. a peg twisting in a hole, an object twisting inside a grasp).
        This switch only adds the axis — its strength is per geom, from
        ``RigidMaterialOptions.friction_torsional`` or ``default_geoms_friction_torsional``, and
        an MJCF geom that does not declare ``condim >= 4`` parses to 0, i.e. no spin resistance
        until a coefficient is set. See ``docs/guides/entities.md``.
    enable_rolling_friction: bool
        Opt-in rolling friction: whether contacts also resist rolling about their tangent axes
        (two extra constraint axes per contact). Requires ``enable_torsional_friction``. Strength
        is per geom like the torsional axis, from ``RigidMaterialOptions.friction_rolling`` or
        ``default_geoms_friction_rolling``. Defaults to False. Enable it when rolling resistance
        matters (e.g. a ball settling on a plane).
    friction_cone: Literal["pyramidal", "elliptic"]
        Shape of the friction cone. ``"pyramidal"`` (default) is robust and easy to solve;
        ``"elliptic"`` is the exact isotropic cone, harder to solve but the only one that
        supports ``contact_resolution="signorini"``. Unsupported with the noslip solver
        (``noslip_iterations > 0``) or ``requires_grad``.
    contact_resolution: Literal["convex", "signorini"] | None
        How a contact's normal force and friction force are resolved against each other.
        ``"signorini"`` bounds friction against the normal force the contact has developed, so
        sliding never inflates it — a body launched horizontally decelerates at mu * g instead of
        lifting off — at the cost of extra solver iterations. It requires
        ``friction_cone="elliptic"`` and ``solver="newton"``. ``"convex"`` poses the contact as a
        single convex program, which converges more predictably on stiff scenes but lets fast
        sliding buy normal force. Defaults to None, which resolves to ``"signorini"`` with the
        elliptic cone and the Newton solver, and ``"convex"`` otherwise.
    noslip_iterations: int
        Number of iterations for the noslip solver. Defaults to 0 (disabled).
        noslip is a post-processing step after the main solver to suppress slip/drift.
        Recommended to set this value to 5 for manipulation tasks or when slip/drift is a big problem.

    env_spacing: array-like[float, float]
        Spacing between environments. Defaults to (0.0, 0.0).
    center_envs_at_origin: bool
        Whether to center environments at the origin. Defaults to False.
    show_link_frame: bool
        Whether to show link frames. Defaults to False.
    show_world_frame: bool
        Whether to show world frames. Defaults to False.
    show_camera_frustums: bool
        Whether to show camera frustums. Defaults to False.
    segmentation_level: Literal["entity", "link", "geom"]
        Level of segmentation. Defaults to "entity".
    show_FPS: bool
        Whether to show FPS for profiling. Defaults to False.
    enable_default_keybinds: bool
        Whether to enable the default viewer keyboard shortcuts. Defaults to True.
    background_color: array-like[float, float, float]
        RGB background color (each component in [0, 1]) for the viewer/renderer. Defaults to (0.04, 0.08, 0.12).
    rasterizer_shadows: bool
        Whether Rasterizer viewers and camera sensors render shadows. Defaults to True.
    update_sensors_every_substep: bool
        If False (default), Genesis's ``SensorManager.step()`` runs only on the
        last sub-step of the env decimation loop. Intermediate sub-step sensor
        readings would be discarded by the next call before anything reads
        them (``observation_manager.compute()`` reads from the sensor cache
        once at the end of ``step()``), so updating them is pure waste — for
        ray-casting depth cameras at perceptive_mimic scale this is a ~2.3×
        end-to-end ``env.step()`` speedup at decimation=4.

        The skip auto-disables on a per-step basis when any sensor declares
        Genesis-level ``SensorOptions.history_length`` or ``delay`` (i.e. the
        SensorManager allocates timeline / return rings) — those rings need
        sim-rate fresh writes to stay correct, so leaving the optimisation on
        by default is safe even for sensor-history users.

        Set True to force per-sub-step updates unconditionally (mostly useful
        for an A/B comparison against per-sub-step updates).

    record_final_observations: bool
        Snapshot pre-reset observations for done envs into ``extras['final_observations']``.

        Off by default. When on, ``RLEnvBase.step()`` runs one extra
        ``observation_manager.compute(update_history=False)`` every step. The
        returned ``extras`` dict gains a ``'final_observations'`` entry shaped
        ``(num_envs, *obs_shape)`` per group; rows for non-done envs are zero,
        and on a no-termination step every row is zero. For groups with
        ``history_length > 0`` the row is history-shaped with the freshly-
        computed post-physics frame in the most-recent slot (composed via
        ``CircularBuffer.peek_buffer``; the canonical once-per-step advance
        still happens in the post-reset compute).

        **No per-step GPU→CPU sync.** Both the gating logic and the snapshot
        copy stay on-device. Cost: one extra
        ``observation_manager.compute(update_history=False)`` per step plus
        one ``torch.where`` per observation group.

        **rsl_rl PPO does not consume this**, by design — rsl_rl 5.x reads
        only the ``"time_outs"``, ``"log"`` and ``"episode"`` ``extras`` keys. rsl_rl bootstraps
        time-out V-targets from ``V(s_t)`` via ``extras['time_outs']`` (a
        deliberate approximation; not something this flag changes). Leaving
        this flag off in pure rsl_rl runs costs nothing.

        Turn it on for off-policy algorithms whose replay buffers need the
        actual ``s'`` for time-out transitions (FastSAC), eval-recording
        callbacks that segment trajectories on done and need the real terminal
        frame for visualization / per-trajectory metrics, or a future
        Holosoma-style PPO runner that bootstraps from ``V(s_{t+1}^{phys})``
        instead of rsl_rl's ``V(s_t)`` proxy.

    cuda_graph_options: CudaGraphOptions | None = None
        Replay the graded phase of ``RLEnvBase.step`` — ``metric_manager.compute``, the observation
        terms' ``post_physics_refresh``, ``termination_manager.compute`` and ``reward_manager.compute`` —
        from a CUDA graph instead of launching its kernels one by one; ``None`` (the default)
        runs it eagerly. ``CudaGraphOptions()`` turns it on with the default capture and validation
        settings, documented there.

        These managers only read the post-physics state and write their own buffers, but they do it in
        hundreds of small kernels, and on a host-bound step the launches are the cost: on BeyondMimic at
        4096 envs the phase takes ~3 ms of host time eagerly, against ~5 µs for the replay plus ~70 µs for
        the checks before it (see ``docs/guides/environments.md``). CUDA only (a CPU or Metal
        backend raises at build), and only an ``RLEnvBase`` has a graded phase (any other env raises at
        construction). A training-throughput switch: the replay commands (``eden inference``,
        ``eden export``, ``eden deploy``) reset it in a run's config (see :attr:`training_only_fields`).

    coupler_options : gs.options.BaseCouplerOptions | None = None
        The options configuring the coupler. Defaults to None (LegacyCouplerOptions).

    fem_options: gs.options.FEMOptions | None = None
        The options configuring the fem_solver. Defaults to None.
    mpm_options: gs.options.MPMOptions | None = None
        The options configuring the mpm_solver. Defaults to None.
    sph_options: gs.options.SPHOptions | None = None
        The options configuring the sph_solver. Defaults to None.
    pbd_options: gs.options.PBDOptions | None = None
        The options configuring the pbd_solver (Position-Based Dynamics: liquids,
        cloth, soft bodies, free particles). Defaults to None.
    mochi_options: MochiOptions | None = None
        The options configuring the mochi_solver (Meta's implicit multi-physics solver:
        rigid/articulated bodies, tetrahedral solids, shells and rods in one Newton system
        with penalty contact). A scene with a Mochi entity may hold only Mochi and kinematic
        entities, and cannot be differentiable. Leave ``MochiOptions.dt`` unset — the solver
        substep is ``sim_dt / sim_substeps`` like every other solver. Run with
        ``en.init(precision="64")`` unless every ``penalty_coefficient`` is lowered.
        Defaults to None.
    """

    num_envs: int = 4096
    sim_dt: float = 0.005
    sim_substeps: int = 1
    decimation: int = 4
    requires_grad: bool = False
    episode_length_s: float = 20.0
    use_gjk_collision: bool | None = True
    solver: Literal["newton", "cg"] = "newton"
    integrator: Literal["approximate_implicitfast", "implicitfast", "euler"] = "approximate_implicitfast"
    impratio: PositiveFloat | None = None
    constraint_timeconst: float | None = None

    max_collision_pairs: int = 50
    multiplier_collision_broad_phase: int = 8
    iterations: int = 50
    tolerance: float | None = None
    ls_iterations: int = 50
    ls_tolerance: float = 1e-2
    enable_self_collision: bool = True
    enable_neutral_collision: bool = False
    batch_links_info: bool = False
    batch_joints_info: bool = False
    batch_dofs_info: bool = True
    enable_multi_contact: bool = True
    enable_torsional_friction: bool = False
    enable_rolling_friction: bool = False
    friction_cone: Literal["pyramidal", "elliptic"] = "pyramidal"
    contact_resolution: Literal["convex", "signorini"] | None = None
    noslip_iterations: int = 0
    use_hibernation: bool = False

    env_separate_rigid: bool = False
    env_spacing: Vec2FType = (0.0, 0.0)
    center_envs_at_origin: bool = False
    show_link_frame: bool = False
    show_world_frame: bool = False
    show_camera_frustums: bool = False
    segmentation_level: Literal["entity", "link", "geom"] = "entity"

    show_FPS: bool = False
    enable_default_keybinds: bool = True
    background_color: Vec3FType = (0.04, 0.08, 0.12)
    rasterizer_shadows: bool = True

    record_final_observations: bool = False
    update_sensors_every_substep: bool = False
    cuda_graph_options: CudaGraphOptions | None = None

    coupler_options: gs.options.BaseCouplerOptions | None = None
    fem_options: gs.options.FEMOptions | None = None
    mpm_options: gs.options.MPMOptions | None = None
    sph_options: gs.options.SPHOptions | None = None
    pbd_options: gs.options.PBDOptions | None = None
    mochi_options: MochiOptions | None = None

    #: Fields that only speed up or instrument a training run and that a replay of the run's own config resets to
    #: their defaults (:meth:`reset_training_only_fields`): a replay gains nothing from them, and may run where
    #: they cannot (``cuda_graph_options`` on a host without an NVIDIA GPU raises at build).
    training_only_fields: ClassVar[tuple[str, ...]] = ("cuda_graph_options",)

    def reset_training_only_fields(self) -> list[str]:
        """Reset every field in :attr:`training_only_fields` to its default; return the names of those it changed."""
        changed = []
        for name in self.training_only_fields:
            default = type(self).model_fields[name].get_default(call_default_factory=True)
            if getattr(self, name) != default:
                setattr(self, name, default)
                changed.append(name)
        return changed

    def model_post_init(self, context):
        super().model_post_init(context)

        assert self.num_envs > 0, "`num_envs` should be greater than 0"

        if self.mochi_options is not None and self.requires_grad:
            raise ValueError(
                "`requires_grad` must be False when using MochiOptions: the MochiSolver is not differentiable."
            )

        if isinstance(self.coupler_options, gs.options.IPCCouplerOptions):
            if self.batch_dofs_info:
                raise ValueError("`batch_dofs_info` must be False when using IPCCouplerOptions.")
            if self.num_envs > 1:
                raise ValueError(
                    "IPC coupler does not support per-env reset (num_envs > 1). "
                    "Use num_envs=1 until per-env reset via libuipc StateAccessor is implemented."
                )
