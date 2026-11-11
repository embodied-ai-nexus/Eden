"""Core vectorized environment base classes: EnvBase, RLEnvBase, and MimicRLEnvBase.

The class hierarchy mirrors the config hierarchy in :mod:`eden.utils.configs`:

- :class:`EnvBase` — scene, entities, sensors, and the action/observation/event managers.
- :class:`RLEnvBase` — adds reward, termination, command, and curriculum managers plus the RL ``step``.
- :class:`MimicRLEnvBase` — adds the motion manager for motion-imitation tasks.

Build environments with ``RLEnvBase.from_config(cfg)`` (not ``from_cfg``); the ``config``
property round-trips that config back, preserving its concrete class and the fields no env
parameter carries. In :meth:`RLEnvBase.step`, auto-reset happens
**after** reward/termination but **before** observation, so the returned observations
reflect the fresh post-reset state for done environments.
"""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager, nullcontext
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Self, TypeVar, get_args, get_origin

import genesis as gs
import numpy as np
import torch
from genesis.options.renderers import BatchRenderer, Rasterizer, RendererOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import qd_from_torch, qd_to_torch
from genesis.typing import Vec2FType

import eden as en
from eden.constants import EventMode
from eden.entities.camera import Camera
from eden.engine.solvers.mochi.error_codes import MochiErrorCode
from eden.entities.sensor import Sensor
from eden.envs.draw_debug_mixin import DrawDebugMixin
from eden.envs.graded_graph import BakedHostReadError, GradedPhaseGraph
from eden.options import (
    ActionManagerOptions,
    BatchRendererOptions,
    CameraOptions,
    CamerasOptions,
    CommandManagerOptions,
    CompositeEntityOptions,
    CurriculumManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    MotionManagerOptions,
    ObservationManagerOptions,
    RecorderManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SensorOptions,
    SensorsOptions,
    TerminationManagerOptions,
    TerrainOptions,
)
from eden.options._metadata import OPTION_METADATA_KEYS
from eden.types import VecEnvReset, VecEnvStep
from eden.utils.configs import EdenConfig, EdenMimicConfig, EdenRLConfig
from eden.utils.misc import (
    invert_segmentation_idx_dict,
    resolve_named_entity,
    sanitize_envs_idx,
    sharing_env_indices,
)
from eden.utils.random import set_random_seed, validate_seed

if TYPE_CHECKING:
    from genesis.engine.couplers import IPCCoupler, LegacyCoupler, SAPCoupler
    from genesis.engine.solvers import RigidSolver

    from eden.engine.solvers.mochi.mochi_solver import MochiSolver
    from eden.entities.base import Entity
    from eden.entities.camera import Camera
    from eden.managers import CommandManager, CurriculumManager, MetricManager
    from eden.options.entities import EntityOptionsLike
    from eden.utils.configs import EdenMimicConfig, EdenRLConfig


_STARTUP, _RESET, _INTERVAL = EventMode

# `EnvOptions` spells the rigid-solver enums as strings so a config round-trips as plain
# data; these are the only place that vocabulary meets the Genesis enums.
_CONSTRAINT_SOLVERS = {"newton": gs.constraint_solver.Newton, "cg": gs.constraint_solver.CG}
_FRICTION_CONES = {"pyramidal": gs.friction_cone.pyramidal, "elliptic": gs.friction_cone.elliptic}
_INTEGRATORS = {
    "approximate_implicitfast": gs.integrator.approximate_implicitfast,
    "implicitfast": gs.integrator.implicitfast,
    "euler": gs.integrator.Euler,
}
# None lets the solver pick: 'signorini' with the elliptic cone and the Newton solver,
# 'convex' otherwise. Asking for 'signorini' without both raises there.
_CONTACT_RESOLUTIONS = {
    None: None,
    "convex": gs.contact_resolution.convex,
    "signorini": gs.contact_resolution.signorini,
}

#: The config leaf an env class is parameterized by — ``EnvBase[EdenRLConfig]`` and friends.
TConfig = TypeVar("TConfig", bound=EdenConfig)
_R = TypeVar("_R")


#: The solver errors that leave an env's state corrupt but its scene sound, which a reset of the env recovers:
#: a NaN contact, force or acceleration and a diverged implicit solve. Every other error a solver registers is a
#: misconfiguration of the scene, which a reset must not hide: a contact capacity overflow, an unsupported collision
#: pair, or a NaN gradient under ``requires_grad`` (a rollout too stiff to differentiate, which the message says how to
#: shorten).
_RECOVERABLE_SOLVER_ERRORS = (
    ErrorCode.INVALID_CONTACT_NAN
    | ErrorCode.INVALID_FORCE_NAN
    | ErrorCode.INVALID_ACC_NAN
    | MochiErrorCode.MOCHI_DIVERGED
)


def _misconfiguration_errors(solver) -> int:
    """The bits of the errors ``solver`` registered that are a misconfiguration of the scene, not a corrupt state.

    Only a registered error has a message the check raises; a bit no kernel registered would be carried past every
    reset without ever raising, so it is left to the reset like a corrupt state.
    """
    registered = 0
    for code in _solver_errno_attribute(solver, "_error_messages"):
        registered |= int(code)
    return registered & ~_RECOVERABLE_SOLVER_ERRORS


def _latched_errors(solver) -> torch.Tensor:
    """The error flags ``solver`` latched per env since its last reset (see ``Solver.update_errno``)."""
    return _solver_errno_attribute(solver, "_errno_latched")


def _solver_errno_attribute(solver, name: str):
    """A private attribute of Genesis' per-env error flags (``Solver.register_errors``), which EnvBase reads to keep
    the flags of a misconfiguration across a reset.
    """
    try:
        return getattr(solver, name)
    except AttributeError:
        raise AttributeError(
            f"{type(solver).__name__} has no {name!r}, of the per-env error flags EnvBase keeps across a reset (see "
            "`EnvBase._reset_scene_and_events`): the Genesis pin in pyproject.toml most likely moved. Re-sync "
            "eden/envs/base.py against the new Genesis `Solver` errno API, or pin Genesis back."
        ) from None


def _zero_applied_control_force(solver, envs_idx: slice | torch.Tensor | None) -> None:
    """Zero the control force ``solver`` reports for ``envs_idx`` (all of them if None): no step has applied one since.

    ``get_dofs_control_force`` reads ``dofs.qf_applied``, which a scene reset leaves at the last force of the episode
    that ended. ``envs_idx`` is a slice, env ids or a bool mask over the envs, written without a GPU->CPU sync.

    The write bypasses the state changes Genesis records for a differentiable rollout (``mutates``), which holds only
    because ``qf_applied`` is an output of a step: the next one overwrites it before anything reads it.
    """
    if not solver.is_active or solver.n_dofs == 0:
        return
    field = solver.dyn_state.dofs.qf_applied
    forces = qd_to_torch(field, copy=not gs.use_zerocopy)  # (n_dofs, n_envs), the layout of the field
    if envs_idx is None or isinstance(envs_idx, slice):
        forces[:, envs_idx if envs_idx is not None else slice(None)] = 0.0
    elif envs_idx.dtype == torch.bool:
        forces.masked_fill_(envs_idx[None, :], 0.0)
    else:
        forces.index_fill_(1, envs_idx.long(), 0.0)
    if not gs.use_zerocopy:
        qd_from_torch(field, forces)
    elif gs.backend == gs.metal:
        torch.mps.synchronize()


def _skip_errno_check(blocking: bool = True) -> None:
    pass


@contextmanager
def _errno_check_deferred(sim) -> Iterator[None]:
    """Skip the check of the solvers' error flags that ``sim.step()`` opens with (see ``EnvBase._run_sim_steps``).

    The flags stay latched in the solvers until their envs are reset, so the next check that runs raises what is left.
    On CPU the step clears the simulator's flag that one is pending along with the skipped check, and sets it again
    from the latched flags once the step is over. The behaviour is pinned by ``tests/envs/test_base.py``.
    """
    sim.check_errno = _skip_errno_check  # an instance attribute, which `Simulator.step` calls instead of the method
    try:
        yield
    finally:
        del sim.check_errno


def _entity_class_for_options(options):
    """Return the appropriate Entity subclass for the given options."""
    from eden.entities.fem import FEMEntity
    from eden.entities.mochi import MochiRigidEntity, MochiSoftEntity
    from eden.entities.particle import ParticleEntity
    from eden.entities.rigid import RigidEntity
    from eden.entities.terrain import Terrain
    from eden.options.entities import (
        FEM_MATERIAL_TYPES,
        MOCHI_RIGID_MATERIAL_TYPES,
        MOCHI_SOFT_MATERIAL_TYPES,
        PARTICLE_MATERIAL_TYPES,
    )

    if isinstance(options, TerrainOptions):
        return Terrain
    material = getattr(options, "material", None)
    if isinstance(material, PARTICLE_MATERIAL_TYPES):
        return ParticleEntity
    if isinstance(material, FEM_MATERIAL_TYPES):
        return FEMEntity
    if isinstance(material, MOCHI_SOFT_MATERIAL_TYPES):
        return MochiSoftEntity
    if isinstance(material, MOCHI_RIGID_MATERIAL_TYPES):
        return MochiRigidEntity
    return RigidEntity


class EnvBase(DrawDebugMixin, RBC, Generic[TConfig]):
    """
    Base class for all environments.

    Parameters
    ----------
    env_options: EnvOptions
        Environment options.
    scene_options: SceneOptions
        Scene options.
    observation_options: ObservationManagerOptions
        Observation manager options.
    event_options: EventManagerOptions
        Event manager options.
    action_options: ActionManagerOptions
        Action manager options.
    metric_options: MetricManagerOptions
        Metric manager options.
    recorder_options: RecorderManagerOptions
        Recorder manager options.
    cameras_options: CamerasOptions
        Cameras options.
    sensors_options: SensorsOptions
        Sensors options.
    renderer_options: RendererOptions
        Renderer options.
    show_viewer: bool
        Whether to show the viewer.
    eval_mode: bool
        Whether to run in eval mode.
    **kwargs
        Forwarded to ``gs.Scene``, overriding the scene options built from ``env_options`` (and its extra fields).
    """

    #: This env's config leaf, bound from the ``EnvBase[T]`` type argument by
    #: :meth:`__init_subclass__` — never written by hand.
    _CONFIG_CLASS: type[EdenConfig] = EdenConfig

    # Managers a subclass resets OUTSIDE the ``_manager_order`` loop, bracketing it — reset
    # before the scene reset, post_reset before the loop. Declared so ``_validate_manager_order``
    # knows they are handled rather than forgotten. Empty here; see ``MimicRLEnvBase``.
    _BRACKETED_MANAGERS: frozenset[str] = frozenset()

    # Managers ``_load_managers`` drops to ``None`` when their options declare no terms — exactly
    # the nullable ones, which carry a class-level ``None`` default for the same reason (below).
    # Every other manager is always constructed.
    _OPTIONAL_MANAGERS: frozenset[str] = frozenset({"metric_manager"})

    #: Whether :meth:`step` grades its post-physics state in a phase ``EnvOptions.cuda_graph_options`` can
    #: replay from a CUDA graph. Only ``RLEnvBase`` does; any other env refuses the option at construction.
    _HAS_GRADED_PHASE: ClassVar[bool] = False

    #: :meth:`compute_once`'s memo: a dict only while a pass is open (:meth:`compute_pass`), None otherwise.
    _pass_cache: dict[Hashable, object] | None = None

    #: :attr:`fixed_shape_queries`, set only by :meth:`fixed_shape_query_scope`.
    _fixed_shape_queries: bool = False

    #: The metric manager, or ``None`` when the task declares no metric terms — which is what
    #: :meth:`_load_managers` leaves behind, and what every consumer in the tree (and the idiom
    #: taught in ``docs/guides/terms.md``) tests for with ``if env.metric_manager is not None``.
    #: Declared here so that idiom also holds *before* build: assignment happens inside
    #: ``_load_managers``, so without a class-level default the documented ``None`` check would raise
    #: ``AttributeError`` instead — from a term's or a task's own ``build()``, the one place it is
    #: most likely to be called. ``RLEnvBase`` defaults its two other nullable managers
    #: (:attr:`~RLEnvBase.command_manager`, :attr:`~RLEnvBase.curriculum_manager`) for the same
    #: reason. The managers that are *always* constructed — action/observation/event/recorder, and
    #: reward/termination — are deliberately left undefaulted: none of them is ever ``None``, so
    #: reading one before build is a mistake that should still raise.
    metric_manager: MetricManager | None = None

    #: The only config fields an env does not own; no env constructor takes them, so
    #: :attr:`config` carries them over from the source config rather than reading them live.
    #: An exclusion rather than an inclusion list, so a field added to a config later is owned by
    #: default — that way a mistake raises, where an inclusion list would report a stale value.
    _NON_RUNTIME_CONFIG_FIELDS: ClassVar[frozenset[str]] = frozenset({"runner_options", "language_instructions"})

    def __init_subclass__(cls, **kwargs):
        """Bind ``_CONFIG_CLASS`` from the ``EnvBase[SomeConfig]`` type argument.

        ``__orig_bases__`` is read off ``cls.__dict__``, not via ``getattr``: it is inherited, so
        ``getattr`` would make every plain subclass re-derive from its parent's bases and stamp a
        ``_CONFIG_CLASS`` of its own, and "declares its own" would stop meaning "is a config leaf".
        """
        super().__init_subclass__(**kwargs)
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            if not (isinstance(origin, type) and issubclass(origin, EnvBase)):
                continue
            args = get_args(base)
            if not args or isinstance(args[0], TypeVar):
                continue
            if not (isinstance(args[0], type) and issubclass(args[0], EdenConfig)):
                # The TypeVar bound is not enforced at runtime, and an unchecked binding only
                # surfaces as a missing-`model_fields` AttributeError on the first `config` access.
                raise TypeError(
                    f"{cls.__name__} is parameterized as {origin.__name__}[{args[0]!r}], but the type "
                    f"argument must be an EdenConfig subclass."
                )
            cls._CONFIG_CLASS = args[0]
            cls._check_leaf_is_constructible()
            break

    @classmethod
    def _check_leaf_is_constructible(cls) -> None:
        """Raise unless every runtime field of the bound leaf is a declared ``__init__`` parameter.

        :meth:`from_config` forwards :meth:`_runtime_option_fields` straight into ``cls(...)``, and
        :attr:`config` reads the same names back off the env, so a field with no parameter of that
        name breaks both directions — and neither says which field, or why. The forward is caught
        by ``__init__``'s ``**kwargs``, which hands it to ``gs.Scene`` and surfaces as
        ``Scene.__init__() got an unexpected keyword argument`` naming neither the config nor the
        env; the read is a bare ``AttributeError``.

        The usual way in is parameterizing on a *task* config rather than on the leaf the env
        reads: ``EnvBase[ReacherConfig]`` sweeps in every task-specific field, where
        ``EnvBase[EdenRLConfig]`` (the leaf ``ReacherEnv`` inherits) sweeps in none and lets
        :attr:`config` recover them from the recorded source instead.

        Checked at class creation because binding the leaf is what creates the obligation — the
        earliest well-defined point, and one that costs a signature read per env *class*.
        """
        declared = inspect.signature(cls.__init__).parameters
        missing = [name for name in cls._runtime_option_fields() if name not in declared]
        if missing:
            raise TypeError(
                f"{cls.__name__} is parameterized as EnvBase[{cls._CONFIG_CLASS.__name__}], but "
                f"{cls._CONFIG_CLASS.__name__} declares {missing}, which {cls.__name__}.__init__ does "
                f"not. from_config() forwards every field of the config leaf into the constructor and "
                f"env.config reads them back off the env, so each one needs a parameter of the same "
                f"name. Either declare them, or parameterize on the leaf this env actually reads and "
                f"let env.config recover the task-specific fields from the source config."
            )

    @classmethod
    def _runtime_option_fields(cls) -> tuple[str, ...]:
        """Config fields :attr:`config` reads off the live env, in the config's own field order."""
        return tuple(name for name in cls._CONFIG_CLASS.model_fields if name not in cls._NON_RUNTIME_CONFIG_FIELDS)

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        # The config ``from_config`` built this env from, if any. Direct construction
        # leaves it None and :attr:`config` synthesizes a base config instead.
        self._source_config: EdenConfig | None = None

        # Handling of default arguments
        self.env_options = env_options or EnvOptions()
        self.scene_options = scene_options or SceneOptions()
        self.observation_options = observation_options or ObservationManagerOptions()
        self.event_options = event_options or EventManagerOptions()
        self.action_options = action_options or ActionManagerOptions()
        self.metric_options = metric_options or MetricManagerOptions()
        self.recorder_options = recorder_options or RecorderManagerOptions()
        self.cameras_options = cameras_options or CamerasOptions()
        self.sensors_options = sensors_options or SensorsOptions()
        self.renderer_options = renderer_options or Rasterizer()

        if self.env_options.cuda_graph_options is not None and not self._HAS_GRADED_PHASE:
            raise ValueError(
                f"EnvOptions(cuda_graph_options=...) replays the graded phase of RLEnvBase.step (metrics, "
                f"terminations, rewards), but {type(self).__name__} is not an RLEnvBase and has no such phase, "
                "so the option would do nothing. Set cuda_graph_options=None, or build an RL task (an "
                "EdenRLConfig) if graded steps were intended."
            )

        if isinstance(self.renderer_options, BatchRenderer):
            en.logger.info(f"You are using {type(self.renderer_options).__name__} as the renderer options.")
            en.logger.info("Consider adding lights to the scene by `env.scene.add_light()`.")

        self.headless = not show_viewer
        # `eval_mode` only flips training/randomization off; the env count is
        # always `num_envs` so an explicit override can't be silently shadowed.
        self.training = not eval_mode
        self.num_envs = self.env_options.num_envs

        self.sim_dt = self.env_options.sim_dt
        self.dt = self.env_options.decimation * self.env_options.sim_dt
        self._sim_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.obs_buf = None
        self.reset_info = None
        # Sanitized index of the envs the last ``_reset_idx`` call actually reset;
        # consumed (and cleared) by ``_dispatch_post_reset`` at the public entry points.
        self._last_reset_envs_idx: slice | torch.Tensor | None = None

        # hooks
        self._pre_build_hooks: list[Callable[[], None]] = []
        self._post_build_hooks: list[Callable[[], None]] = []
        # Called around every scene.step() of step(): before it, then after it (see register_substep_callback).
        self._pre_substep_callbacks: list[Callable[[], None]] = []
        self._substep_callbacks: list[Callable[[], None]] = []
        self._post_build_hooks_drained = False

        # create scene
        # Build default scene kwargs from env_options fields
        scene_kwargs = dict(
            sim_options=gs.options.SimOptions(
                dt=env_options.sim_dt,
                substeps=env_options.sim_substeps,
                requires_grad=env_options.requires_grad,
            ),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(1 / env_options.sim_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                enable_default_keybinds=env_options.enable_default_keybinds,
            ),
            vis_options=gs.options.VisOptions(
                # Genesis renamed ``env_separate_rigid`` to ``split_envs`` and warns on every build that
                # passes the old keyword, so Eden passes the new one.
                split_envs=env_options.env_separate_rigid,
                show_link_frame=env_options.show_link_frame,
                show_world_frame=env_options.show_world_frame,
                show_cameras=env_options.show_camera_frustums,
                segmentation_level=env_options.segmentation_level,
                background_color=env_options.background_color,
                shadow=env_options.rasterizer_shadows,
            ),
            rigid_options=gs.options.RigidOptions(
                # No ``dt=`` here: a solver dt is now an *authored* integration interval, and
                # authoring the step dt while also passing ``SimOptions.substeps > 1`` is a
                # conflict Genesis rejects. The rigid solver derives its rate from
                # ``SimOptions(dt, substeps)`` like every other solver.
                constraint_solver=_CONSTRAINT_SOLVERS[env_options.solver],
                constraint_timeconst=env_options.constraint_timeconst
                if env_options.constraint_timeconst is not None
                else max(0.01, 2 * env_options.sim_dt / env_options.sim_substeps),
                use_gjk_collision=env_options.use_gjk_collision,
                max_collision_pairs=env_options.max_collision_pairs,
                multiplier_collision_broad_phase=env_options.multiplier_collision_broad_phase,
                iterations=env_options.iterations,
                tolerance=env_options.tolerance,
                ls_iterations=env_options.ls_iterations,
                ls_tolerance=env_options.ls_tolerance,
                enable_collision=True,
                enable_self_collision=env_options.enable_self_collision,
                enable_neutral_collision=env_options.enable_neutral_collision,
                enable_joint_limit=True,
                use_hibernation=env_options.use_hibernation,
                batch_links_info=env_options.batch_links_info,
                batch_joints_info=env_options.batch_joints_info,
                batch_dofs_info=env_options.batch_dofs_info,
                enable_multi_contact=env_options.enable_multi_contact,
                enable_torsional_friction=env_options.enable_torsional_friction,
                enable_rolling_friction=env_options.enable_rolling_friction,
                friction_cone=_FRICTION_CONES[env_options.friction_cone],
                integrator=_INTEGRATORS[env_options.integrator],
                impratio=env_options.impratio,
                contact_resolution=_CONTACT_RESOLUTIONS[env_options.contact_resolution],
                noslip_iterations=env_options.noslip_iterations,
            ),
            coupler_options=env_options.coupler_options,
            fem_options=env_options.fem_options,
            mpm_options=env_options.mpm_options,
            sph_options=env_options.sph_options,
            pbd_options=env_options.pbd_options,
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=env_options.show_FPS,
            ),
            renderer=renderer_options,
            show_viewer=show_viewer,
        )

        # Allow extra fields on EnvOptions to override auto-constructed defaults
        extras = env_options.__pydantic_extra__ or {}
        internal_keys = set(OPTION_METADATA_KEYS)
        for key, value in extras.items():
            if key not in internal_keys:
                scene_kwargs[key] = value

        # Explicit kwargs take highest priority
        scene_kwargs.update(kwargs)

        self.scene = gs.Scene(**scene_kwargs)
        # Genesis ships no MochiSolver: Eden hosts one, installed on the scene's first Mochi entity.
        self._mochi_solver: MochiSolver | None = None
        # The misconfiguration flags of the envs a reset or a setter cleared, per solver, which the next control step
        # raises (see `get_solver_error_envs_mask`), and whether any were kept since. Allocated on the first use.
        self._misconfigurations: list[tuple[Any, int, torch.Tensor]] | None = None
        self._has_kept_misconfigurations = False

        # gs-madrona bakes the light count at scene.build(), so BatchRenderer
        # lights must be registered up front. Only the eden BatchRendererOptions
        # subclass carries a ``lights`` field — the raw genesis BatchRenderer has
        # none, and lights declared elsewhere would silently never render.
        if isinstance(self.renderer_options, BatchRendererOptions):
            for light in self.renderer_options.lights:
                self.scene.add_light(**light)

        entity_data: dict[str, EntityOptionsLike] = {}
        composite_attachments: dict[str, tuple[str, str]] = {}
        for key in self.scene_options.keys():
            if key == "attachments_dict":
                continue
            value = getattr(self.scene_options, key)
            if isinstance(value, CompositeEntityOptions):
                # A composite (e.g. a Droid manipulator) is stored in the scene as one
                # unit and unfolds here into its sub-entities + welds. The scene stays the
                # source of truth (it round-trips as the composite); expansion is local to
                # the build. Name collisions / primary-key were checked in SceneOptions.
                sub_entities, sub_attachments = value.expand()
                entity_data.update(sub_entities)
                composite_attachments.update(sub_attachments)
            else:
                entity_data[key] = value

        # Composite welds join the scene's own attachments; both feed the topo sort below
        # and the attach loop that welds children onto parents.
        attachments_dict = {**(self.scene_options.attachments_dict or {}), **composite_attachments}
        if attachments_dict:
            # Order so each attachment parent is immediately followed by its attached
            # subtree. genesis' ``attach`` requires a welded child to be instantiated
            # *consecutively* with its parent (the merged pair is simulated as one
            # kinematic tree whose DOFs must form one contiguous range), so no other
            # DOF-carrying entity may be created between a parent and its child. A plain
            # "parents before children" topological order is not enough: free entities
            # (a table, a cube) declared between the arm and its welded gripper would sit
            # in the middle and genesis rejects the weld. Each entity has at most one
            # parent (``attachments_dict`` is keyed by child), so the graph is a forest —
            # emit each root in declaration order depth-first so a parent and its welded
            # descendants stay adjacent.
            children_of: dict[str, list[str]] = defaultdict(list)
            attached_children: set[str] = set()
            for child_entity_name, spec in attachments_dict.items():
                children_of[spec[0]].append(child_entity_name)
                attached_children.add(child_entity_name)

            ordered_keys: list[str] = []
            seen: set[str] = set()

            def _emit_subtree(name: str) -> None:
                if name in seen:  # defensive: a malformed cycle must not recurse forever
                    return
                seen.add(name)
                ordered_keys.append(name)
                for child in children_of.get(name, []):
                    _emit_subtree(child)

            for name in entity_data.keys():
                if name not in attached_children:  # roots: entities not welded onto anything
                    _emit_subtree(name)

            # A key left unemitted means a child whose parent is missing, or a cycle among
            # attached entities (no root to reach them from).
            if len(ordered_keys) != len(entity_data):
                missing = {spec[0] for spec in attachments_dict.values() if spec[0] not in entity_data}
                if missing:
                    raise ValueError(f"attachments_dict references nonexistent entities: {missing}")
                raise ValueError(f"Cycle detected in attachments_dict: {attachments_dict}")
        else:
            ordered_keys = list(entity_data.keys())

        self.entities: dict[str, Entity] = {}
        for entity_name in ordered_keys:
            entity_options: EntityOptionsLike = entity_data[entity_name]
            entity_cls = _entity_class_for_options(entity_options)
            entity = entity_cls(env=self, options=entity_options)
            entity._name = entity_name
            self.entities[entity_name] = entity

        for entity_name, spec in attachments_dict.items():
            attach_to_entity_name, attach_to_link_name = spec[0], spec[1]
            # Optional 4-tuple form carries a (pos, quat) mounting offset; the 2-tuple form
            # falls back to the child's default_root_pos/quat (see RigidEntity.attach_to).
            attach_offset, attach_quat = (spec[2], spec[3]) if len(spec) >= 4 else (None, None)
            self.entities[entity_name].attach_to(
                self.entities[attach_to_entity_name],
                attach_to_link_name,
                attach_offset=attach_offset,
                attach_quat=attach_quat,
            )

        self.cameras: dict[str, Camera] = {}
        self._setup_cameras()

        self.sensors: dict[str, Sensor] = {}
        self._setup_sensors()

    @classmethod
    def _check_config_pairing(cls, cfg: EdenConfig) -> None:
        """Raise unless ``cfg`` is the config leaf this env class reads.

        :meth:`from_config` forwards the option groups this env's own :attr:`_CONFIG_CLASS`
        declares, so a config from a *different* leaf is only ever partly built:
        ``RLEnvBase.from_config(EdenMimicConfig(...))`` drops ``motion_options`` and trains
        against no motion reference at all, and ``MimicRLEnvBase.from_config(EdenRLConfig(...))``
        reads a group the config does not declare. Unchecked, the first fails silently and the
        second as an ``AttributeError`` naming neither the config nor the fix.

        Both directions are the same question: is ``cfg``'s leaf this env's leaf? Task subclasses
        on either side are ordinary — ``ReacherEnv(RLEnvBase)`` built from a
        ``ReacherConfig(EdenRLConfig)`` shares the ``RLEnvBase``/``EdenRLConfig`` leaf — so it is
        asked of the leaves, not the exact classes: ``cls`` must be (or subclass) the env
        :func:`env_class_for_config` dispatches ``cfg`` to, and ``cfg`` must be an instance of
        the config class ``cls`` reads.

        The two ways that can fail need different advice, so they raise separately. When the env is
        too *narrow* for the config, naming the env that owns the config is the fix. When the env is
        too *specific* — including a third-party ``EnvBase[MyConfig]`` subclass handed a plain
        leaf — it is not: :func:`env_class_for_config` cannot recover ``cls``, so pointing at the
        env it does return would answer a question the caller did not ask. That case leads with the
        config to pass instead.

        Parameters
        ----------
        cfg: EdenConfig
            The configuration object ``from_config`` was handed.
        """
        expected_env = env_class_for_config(cfg)

        # The env is too narrow for the config: everything past this env's leaf is dropped.
        if not issubclass(cls, expected_env):
            raise TypeError(
                f"{cls.__name__}.from_config() reads only the option groups "
                f"{cls._CONFIG_CLASS.__name__} declares, so it cannot build {type(cfg).__name__} — "
                f"whatever that config carries beyond them would be dropped with no error. Build it "
                f"with {expected_env.__name__}.from_config(cfg), or with "
                f"env_class_for_config(cfg).from_config(cfg) when the config class is not known "
                f"statically."
            )

        # The env is too specific for the config: it reads a group the config does not declare.
        if not isinstance(cfg, cls._CONFIG_CLASS):
            raise TypeError(
                f"{cls.__name__}.from_config() reads the option groups {cls._CONFIG_CLASS.__name__} "
                f"declares, and {type(cfg).__name__} does not carry them. Pass a config whose leaf is "
                f"{cls._CONFIG_CLASS.__name__}, or build this one with "
                f"{expected_env.__name__}.from_config(cfg)."
            )

    @classmethod
    def from_config(cls, cfg: TConfig, *, show_viewer: bool = False, eval_mode: bool = False) -> Self:
        """
        Create an environment from a configuration object.

        Inherited by every env class rather than reimplemented per leaf: a hand-copied list of
        option groups drops a forgotten group in silence, the config fully valid and the env
        simply never told. Reading the field list off :meth:`_runtime_option_fields` instead makes
        forwarding a property of the *structure*: that is the same list :attr:`config` reads back
        off the live env, so the two directions are symmetric by construction, and a group added
        to a config leaf later is forwarded without anyone remembering to. Task-specific fields
        are not swept in — the list comes from this env's leaf, not from ``type(cfg)`` — and
        :attr:`config` carries them across via the recorded source instead.

        Parameters
        ----------
        cfg: TConfig
            The configuration object. Must be the leaf this env class reads — see
            :meth:`_check_config_pairing` — with every field a TaskMod binds filled in, see
            :meth:`~eden.utils.configs.EdenConfig.check_bindings`.
        show_viewer: bool
            Whether to show the viewer.
        eval_mode: bool
            Whether to run in eval mode.

        Returns
        -------
        env: Self
            The constructed environment, **unbuilt** — call ``build()`` yourself. The training
            wrappers' ``from_config`` builds and resets; this one does not.
        """
        cls._check_config_pairing(cfg)
        cfg.check_bindings()
        env = cls(
            **{name: getattr(cfg, name) for name in cls._runtime_option_fields()},
            show_viewer=show_viewer,
            eval_mode=eval_mode,
        )
        # Keeps `env.config` faithful to its source: the concrete config class plus the
        # fields no env parameter carries (runner_options, language_instructions, and any
        # task-specific ones). See :attr:`config`.
        env._source_config = cfg

        return env

    @gs.assert_unbuilt
    def _setup_cameras(self):
        for camera_name in self.cameras_options.keys():
            camera_options: CameraOptions = getattr(self.cameras_options, camera_name)
            camera = Camera(env=self, options=camera_options, key=camera_name)
            self.cameras[camera_name] = camera

    @gs.assert_unbuilt
    def _setup_sensors(self):
        for sensor_name in self.sensors_options.keys():
            sensor_options: SensorOptions = getattr(self.sensors_options, sensor_name)
            sensor = Sensor(env=self, options=sensor_options)
            self.sensors[sensor_name] = sensor

    @gs.assert_unbuilt
    def register_pre_build_hook(self, hook: Callable[[], None]) -> None:
        self._pre_build_hooks.append(hook)

    def register_post_build_hook(self, hook: Callable[[], None]) -> None:
        """Queue ``hook`` to run once at the end of :meth:`post_build_setup`.

        Parameters
        ----------
        hook : Callable[[], None]
            Zero-argument callable, invoked after every entity/camera ``post_build()`` and
            after the managers are loaded.

        Raises
        ------
        RuntimeError
            If :meth:`post_build_setup` already drained the queue, so ``hook`` would be
            registered too late to ever run.
        """
        if self._post_build_hooks_drained:
            raise RuntimeError(
                f"register_post_build_hook({getattr(hook, '__qualname__', hook)}) was called after "
                "post_build_setup() drained the post-build hooks, so the hook would never run. "
                "Register it from __init__, from a pre-build hook, or from a manager's "
                "construction (which happens inside post_build_setup, before the drain)."
            )
        self._post_build_hooks.append(hook)

    def register_substep_callback(self, callback: Callable[[], None], *, before: bool = False) -> None:
        """Run ``callback`` around every ``scene.step()`` of :meth:`step` (``decimation`` times per control step).

        For a term that has to act on the simulation state *between* sim steps — clamp what the solver just did,
        before the next step integrates it — which neither the action manager (it writes controls *before* each
        step) nor a per-control-step hook can do. A ``before`` callback runs right before ``scene.step()``, after the
        action manager has applied its controls; the rest run right after it. Each runs in registration order, with
        no arguments, on every env; keep them cheap and sync-free, since they sit on the physics hot path.

        One ``scene.step()`` advances ``EnvOptions.sim_dt``; Genesis' own ``sim_substeps`` run inside it, so a
        callback never sees them.

        A setter that overwrites an env's state clears the error flags its solver latched for it (the MochiSolver's
        and Genesis' particle solvers' do), so a callback calling one would hide the error of that sim step. While a
        callback is registered, the env keeps the flags of a misconfiguration after every sim step, which the next
        control step still raises; a corrupt state the setter overwrote is gone with its flag, as Genesis intends.

        Parameters
        ----------
        callback : Callable[[], None]
            Zero-argument callable.
        before : bool, optional
            Run it before each ``scene.step()`` instead of after. Defaults to False.
        """
        (self._pre_substep_callbacks if before else self._substep_callbacks).append(callback)

    def _kept_misconfigurations(self) -> list[tuple[Any, int, torch.Tensor]]:
        """``(solver, misconfiguration codes, flags kept across a reset)`` of each solver whose errors Eden reads.

        Allocated once, after the build: every solver registers its errors as it builds (``Solver.register_errors``;
        the NaN gradient of ``requires_grad`` in ``Solver.build``), so the codes read then are all there will be.
        """
        if self._misconfigurations is None:
            solvers = [solver for solver in (self.rigid_solver, self._mochi_solver) if solver is not None]
            self._misconfigurations = [
                (solver, _misconfiguration_errors(solver), torch.zeros_like(_latched_errors(solver)))
                for solver in solvers
            ]
        return self._misconfigurations

    def _keep_misconfigurations(self) -> None:
        """Keep the misconfiguration flags the solvers latched, before a reset or a setter clears them."""
        for solver, codes, kept in self._kept_misconfigurations():
            kept.bitwise_or_(_latched_errors(solver) & codes)
        self._has_kept_misconfigurations = True

    def _raise_kept_misconfigurations(self, sim) -> None:
        """Hand the solvers back the misconfiguration flags a reset or a setter cleared, which the ``scene.step()``
        about to run then raises with the option to change (on CUDA, whose check does not wait for the device, the next
        one).

        Sync-free: the flags join those the solver latched, and Genesis' own check raises them. On CPU that check runs
        only while the simulator records a flag as pending, which a deferred sim step after the setter cleared the
        flags left unset, so it is set again here. Skipped when nothing was kept since the last control step, which a
        step without a reset or a substep callback is.
        """
        if not self._has_kept_misconfigurations:
            return
        for solver, _, kept in self._kept_misconfigurations():
            _latched_errors(solver).bitwise_or_(kept)
            kept.zero_()
        sim._has_errno_flags = True  # read on CPU only: the check finds nothing to raise if no flag was kept
        self._has_kept_misconfigurations = False

    def _run_sim_steps(self, *, apply_actions: bool) -> None:
        """The ``decimation`` sim steps of one control step, each bracketed by the substep callbacks.

        Genesis raises the error flags its solvers latched at the start of a ``scene.step()``, so that an env reset
        in between carries on. The envs a solver flags are reset once the control step is over, so only its
        first sim step checks them: a later one would raise for the whole scene on an env about to be reset.
        """
        sim = self.scene._sim
        for i in range(self.env_options.decimation):
            self._sim_step_counter += 1
            if apply_actions:
                self.action_manager.apply_actions()
            for callback in self._pre_substep_callbacks:
                callback()
            if i == 0:  # after anything that could call a setter, right before the step that checks the flags
                self._raise_kept_misconfigurations(sim)
            with nullcontext() if i == 0 else _errno_check_deferred(sim):
                self._scene_step_with_sensor_gating(is_last_substep=(i == self.env_options.decimation - 1))
            if self._substep_callbacks or self._pre_substep_callbacks:
                self._keep_misconfigurations()  # before a callback's setter clears them (see register_substep_callback)
            for callback in self._substep_callbacks:
                callback()

    @gs.assert_unbuilt
    def build(
        self,
        env_spacing: Vec2FType | None = None,
        center_envs_at_origin: bool | None = None,
    ):
        """Build the underlying Genesis scene and finalize all entities.

        Runs :meth:`pre_build_setup`, builds the vectorized scene, then runs
        :meth:`post_build_setup`. Must be called once before stepping the environment.

        Parameters
        ----------
        env_spacing : tuple[float, float], optional
            Spacing between environments in the xy-plane. Falls back to
            ``env_options.env_spacing`` when None.
        center_envs_at_origin : bool, optional
            Whether to center the grid of environments at the world origin.
            Falls back to ``env_options.center_envs_at_origin`` when None.
        """
        env_spacing = env_spacing if env_spacing is not None else self.env_options.env_spacing
        center_envs_at_origin = (
            center_envs_at_origin if center_envs_at_origin is not None else self.env_options.center_envs_at_origin
        )
        self.pre_build_setup()
        self.scene.build(
            n_envs=self.num_envs,
            env_spacing=env_spacing,
            center_envs_at_origin=center_envs_at_origin,
        )
        self.post_build_setup()
        self._verify_build()

    @gs.assert_unbuilt
    def pre_build_setup(self):
        for entity in self.entities.values():
            entity.pre_build()

        for camera in self.cameras.values():
            camera.pre_build()

        for sensor in self.sensors.values():
            sensor.pre_build()

        for hook in self._pre_build_hooks:
            hook()

    @gs.assert_built
    def post_build_setup(self):
        for entity in self.entities.values():
            entity.post_build()

        for camera in self.cameras.values():
            camera.post_build()

        self._load_managers()
        self._validate_manager_order()

        # Managers register post-build hooks from their __init__ (MotionManager's motion-end
        # resync validator), so this must stay below _load_managers(). Iterating the live list lets
        # a hook queue another one; after this point registering is a loud error.
        for hook in self._post_build_hooks:
            hook()
        self._post_build_hooks_drained = True

        # NOTE: save the post setup state for reset
        self.reset_state = self.scene.get_state()

    @gs.assert_built
    def _verify_build(self):
        if self.rigid_solver.n_equalities > 0:
            if self.env_options.sim_dt / self.env_options.sim_substeps > 0.004:
                en.logger.warning(
                    "Modify the sim_dt or sim_substeps to make effective dt smaller than 0.004 for stability when "
                    "there are closed-link represented by equality constraints. Current sim_dt: "
                    f"{self.env_options.sim_dt}, sim_substeps: {self.env_options.sim_substeps}"
                )

        # NOTE: check entities
        from eden.entities.fem import FEMEntity
        from eden.entities.mochi import MochiRigidEntity, MochiSoftEntity
        from eden.entities.particle import ParticleEntity

        for name, entity in self.entities.items():
            # Skip mass check for non-rigid entities (MPM/SPH/FEM/Mochi soft) — their mass is
            # determined by material density and vertex/particle count, not a single rigid mass.
            # Also skip kinematic entities which have no meaningful rigid mass.
            if isinstance(entity, (ParticleEntity, FEMEntity, MochiSoftEntity)) or isinstance(
                entity.material, en.materials.KinematicMaterialOptions
            ):
                continue
            # ``has_immobile_base``, not ``is_fixed_base``: a welded child is carried by its
            # parent link, so a sub-gram link mass is fine there and the 0.01 kg floor below
            # would silently change the rig's dynamics.
            if entity.has_immobile_base:
                continue
            # ``get_mass`` is a 0-dim tensor, or ``(num_envs,)`` when link info is batched
            # (``batch_links_info=True`` or a heterogeneous entity).
            masses = entity.get_mass()
            if (masses < 0.001).any():
                if isinstance(entity, MochiRigidEntity):
                    # Mochi link masses are fixed at build (``set_links_mass`` raises), so the
                    # patch below is not available — the density is the only knob.
                    raise ValueError(
                        f"{name} has mass {float(masses.min()):.2e} kg, which the MochiSolver cannot patch at "
                        f"build. Raise `MochiRigidMaterialOptions.rho` or give the asset an explicit mass."
                    )
                en.logger.warning(
                    f"""{name} has mass of {masses.min()} kg, potentially leads to instability in simulation.
                    Setting mass to 0.01 kg."""
                )
                entity.set_mass(0.01)

    def destroy(self):
        # Cameras first, so a recording still running is finalized while the scene it renders
        # is alive (see Camera.destroy). The scene is destroyed even when one of them fails:
        # a lost video is recoverable, whereas a scene left alive keeps the Quadrants SNode
        # trees every build spends, and the process runs out of them after ~100 builds.
        error = None
        for name, camera in self.cameras.items():
            try:
                camera.destroy()
            except Exception as e:  # noqa: BLE001 - teardown continues; the error is re-raised below
                en.logger.warning(f"Tearing down camera '{name}' failed: {e}")
                if error is None:
                    error = e

        self.scene.destroy()

        if error is not None:
            raise error

    def _manager_factories(self) -> dict[str, Callable[[], Any]]:
        """Map each manager attribute name to the zero-argument factory that constructs it.

        The construction half of :meth:`_manager_order`, which :meth:`_load_managers` walks — so
        every manager is *built* in the same producer→consumer position it is reset in, and a term
        may resolve an upstream manager's handle in its own ``build()`` rather than deferring to the
        first ``compute()``. Unlike ``_manager_order`` this composes: the slots are keyed, so a
        subclass merges its own into ``super()``'s instead of restating the whole thing.

        A manager that brackets the loop is constructed by the override that brackets it
        (:meth:`MimicRLEnvBase._load_managers`) and must NOT appear here — ``_load_managers``
        rejects a factory that no position in the order would ever call. Manager imports stay
        inside the method to break the manager↔env import cycle.
        """
        from eden.managers import (
            ActionManager,
            EventManager,
            MetricManager,
            ObservationManager,
            RecorderManager,
        )

        return {
            "action_manager": lambda: ActionManager(self, self.action_options),
            "metric_manager": lambda: MetricManager(self, self.metric_options),
            "event_manager": lambda: EventManager(self, self.event_options),
            "observation_manager": lambda: ObservationManager(self, self.observation_options),
            "recorder_manager": lambda: RecorderManager(self, self.recorder_options),
        }

    @gs.assert_built
    def _load_managers(self):
        """Construct every manager in :meth:`_manager_order`, then fire the STARTUP events.

        One declared order drives all three walks — construction here, ``reset()`` in
        :meth:`_reset_idx`, ``post_reset()`` in :meth:`_post_reset_idx` — so the position a manager
        holds is the position it holds everywhere. Every upstream manager a term resolves therefore
        exists during its ``build()``, and a bad handle raises at env-construction time rather than
        on the first ``compute()``/``reset()``.

        An optional manager (:attr:`_OPTIONAL_MANAGERS`) whose options declare no terms is left as
        ``None``, which is what every consumer's ``is not None`` check tests for.
        """
        factories = self._manager_factories()
        order = self._manager_order()
        if missing := sorted(set(order) - set(factories)):
            raise RuntimeError(
                f"{type(self).__name__}._manager_factories() has no factory for: {', '.join(missing)} — "
                f"named by _manager_order() but nothing constructs them. Add a factory, or drop them "
                f"from the order."
            )
        if uncalled := sorted(set(factories) - set(order)):
            raise RuntimeError(
                f"{type(self).__name__}._manager_factories() declares factories no position in "
                f"_manager_order() would call: {', '.join(uncalled)}. Add them to the order, or — for a "
                f"manager that brackets the loop — construct it in the _load_managers override that "
                f"brackets it and keep it out of this map. Order declared: {', '.join(order)}."
            )

        for name in order:
            manager = factories[name]()
            if name in self._OPTIONAL_MANAGERS and not manager._terms:
                manager = None
            setattr(self, name, manager)

        if _STARTUP in self.event_manager.available_modes:
            self.event_manager.compute(mode=_STARTUP)

    @gs.assert_built
    def summary(self) -> str:
        """Return a human-readable summary of the environment and its managers."""
        msg = "Environment Summary:\n"
        msg += f"Number of environments: {self.num_envs}\n"
        msg += f"Step time step: {self.dt} s\n"
        msg += f"Simulation time step: {self.sim_dt} s\n"
        msg += f"Decimation: {self.env_options.decimation}\n"

        msg += "Managers:\n"
        msg += self.action_manager.summary()
        msg += self.event_manager.summary()
        msg += self.observation_manager.summary()
        if self.metric_manager is not None:
            msg += self.metric_manager.summary()
        msg += self.recorder_manager.summary()
        return msg

    @staticmethod
    def seed(seed: int) -> int:
        """Seed all RNGs used by the environment.

        Applies exactly the seed given: no DDP rank offset, and no write-back to any
        config. Seeding a *run* is :func:`eden.utils.random.apply_run_seed`'s job — the
        training wrappers call it from ``from_config``, before the env is built.

        Parameters
        ----------
        seed : int
            The seed to set. Must be non-negative; ``-1`` is not a "draw one"
            sentinel (see :func:`eden.utils.random.validate_seed`).

        Returns
        -------
        int
            The seed that was actually applied.

        Raises
        ------
        ValueError
            If ``seed`` is negative.
        """
        seed = validate_seed(seed)
        en.logger.info(f"Setting seed: {seed}")
        set_random_seed(seed)
        return seed

    @cached_property
    def _all_indices(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device)

    @property
    def is_built(self) -> bool:
        """Whether the scene has been built."""
        return self.scene._is_built

    @property
    def coupler(self) -> LegacyCoupler | SAPCoupler | IPCCoupler:
        return self.scene.sim.coupler

    @cached_property
    def _has_ipc_coupler(self) -> bool:
        """Whether the scene uses an IPC coupler (requires full-scene reset)."""
        return isinstance(self.env_options.coupler_options, gs.options.IPCCouplerOptions)

    @property
    def rigid_solver(self) -> RigidSolver:
        return self.scene.rigid_solver

    @property
    def mochi_solver(self) -> MochiSolver:
        """The Mochi implicit multi-physics solver of this scene's Mochi entities.

        See :mod:`eden.engine.solvers.mochi`.

        Raises
        ------
        RuntimeError
            If no entity of the scene takes a Mochi material, so no solver was installed.
        """
        if self._mochi_solver is None:
            raise RuntimeError(
                "This env has no MochiSolver: no entity of its scene takes a Mochi material "
                "(MochiRigidMaterialOptions, MochiElasticMaterialOptions, MochiShellMaterialOptions or "
                "MochiRodMaterialOptions). Give one of them a Mochi material, or don't read `env.mochi_solver` here."
            )
        return self._mochi_solver

    def _install_mochi_solver(self, subject: str) -> MochiSolver:
        """The scene's MochiSolver, installed with ``EnvOptions.mochi_options`` the first time an entity needs it.

        ``subject`` names that entity for the error a differentiable env raises. ``EnvOptions`` rejects
        ``requires_grad`` itself only when ``mochi_options`` is set; a Mochi material alone installs the solver too,
        so this is the first point that sees both, before any later entity has loaded.
        """
        if self._mochi_solver is None:
            if self.env_options.requires_grad:
                raise ValueError(
                    f"{subject} takes a Mochi material, but the env has EnvOptions(requires_grad=True) and the "
                    f"MochiSolver is not differentiable. Set requires_grad=False, or give every entity a non-Mochi "
                    f"material."
                )
            from eden.engine.solvers.mochi.hosting import install_mochi_solver

            self._mochi_solver = install_mochi_solver(self.scene, self.env_options.mochi_options)
        return self._mochi_solver

    def get_solver_error_envs_mask(self) -> torch.Tensor:
        """Envs a physics solver flagged since their last reset, ``(num_envs,)`` bool.

        The rigid solver's errors (a NaN contact force) and, in a Mochi scene, the Mochi solver's (a diverged
        implicit solve). The termination manager truncates these envs, and the reward and metric managers drop what
        their terms read of them, so no value of the corrupt state reaches the learner; the reset clears the flag
        before the next error check would raise for the whole scene.

        A misconfiguration of the scene (a contact capacity overflow, an unsupported collision pair, a NaN gradient
        under ``requires_grad``; see ``_RECOVERABLE_SOLVER_ERRORS``) is reset and dropped the same way, but the env
        keeps its flag across the reset, and the next control step raises it with the message that names the option to
        change (on CUDA, whose check does not wait for the device, the one after).
        """
        mask = self.rigid_solver.get_error_envs_mask()
        if self._mochi_solver is not None:
            mask = mask | self._mochi_solver.get_error_envs_mask()
        return mask

    @property
    def segmentation_idx_dict(self) -> dict[int, int | tuple[int, ...]]:
        """Genesis' ``segmentation id -> scene element`` map.

        The element is a bare ``entity_idx`` at ``segmentation_level="entity"`` and an
        ``(entity_idx, link_idx, ...)`` tuple at ``"link"`` / ``"geom"``. To go the other
        way round use :attr:`entity_segmentation_ids`, not a manual inversion.
        """
        return self.scene.segmentation_idx_dict

    @property
    def entity_segmentation_ids(self) -> dict[int, list[int]]:
        """``entity_idx -> [segmentation ids]``, the inverse of :attr:`segmentation_idx_dict`.

        An entity owns one id per link/geom at ``segmentation_level="link"``/``"geom"``, so this
        is the accessor to mask an entity out of a segmentation image — indexing
        ``segmentation_idx_dict`` the other way round only ever finds one of them. Recomputed per
        access so it tracks a rebuilt scene rather than going stale.
        """
        return invert_segmentation_idx_dict(self.segmentation_idx_dict)

    @property
    def device(self) -> torch.device:
        """The torch device the environment's tensors live on."""
        return gs.device

    @property
    def config(self) -> EdenConfig:
        """The config describing this environment — a *round trip* of the config it was built from.

        Built via :meth:`from_config`, this returns a new instance of the **source config's own
        class** (``ReacherConfig``, not ``EdenRLConfig``), carrying every field the env never
        receives — ``runner_options``, ``language_instructions``, task-specific fields — verbatim,
        with every field :attr:`_CONFIG_CLASS` declares except :attr:`_NON_RUNTIME_CONFIG_FIELDS`
        read off the live env, so runtime mutations (and any ``None`` the env replaced with a
        default) are reflected. That is what makes ``env.config.save_as_json()`` write a file that
        reloads into the same task and rebuilds the same env — synthesizing a base config here
        instead would drop the runner and task fields with no error, since no env parameter
        carries them.

        Constructed **directly** (``EnvBase(env_options=..., ...)``), there is no source config to
        round-trip: this synthesizes a plain :attr:`_CONFIG_CLASS` from the runtime options only,
        and the fields the env does not own take their class defaults.

        The returned root is a fresh, unlocked object, but its option children are the *live* ones
        (not copies) — mutating them mutates the environment. Use ``env.config.clone()`` for an
        independent copy.
        """
        runtime_options = {name: getattr(self, name) for name in self._runtime_option_fields()}
        source = self._source_config
        if source is None:
            return self._CONFIG_CLASS(**runtime_options)
        # ``model_construct``, not ``type(source)(...)``: a task config may define a custom
        # ``__init__`` that rebuilds its fields from scratch, which would discard exactly the
        # runtime state this property exists to report (see docs/guides/task-config.md).
        # ``from_config`` read every runtime option off ``source``, so its class declares them all.
        fields = {name: getattr(source, name) for name in type(source).model_fields}
        fields.update(runtime_options)
        return type(source).model_construct(**fields)

    @property
    def gravity(self) -> torch.Tensor:
        """The gravity vector applied by the rigid solver."""
        return self.rigid_solver.get_gravity()

    def set_global_sol_params(self, sol_params) -> None:
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: tuple[float] | list[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        self.rigid_solver.set_global_sol_params(sol_params)

    def set_sol_params(
        self,
        sol_params,
        geoms_idx=None,
        envs_idx=None,
        *,
        joints_idx=None,
        eqs_idx=None,
    ) -> None:
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: tuple[float] | list[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        geoms_idx: array_like, optional
            Scene-global indices of the geoms to set parameters for — ``geom.idx``, not a position in
            ``entity.geoms`` (that would be ``geom.idx - entity.geom_start``). If None, all geoms are used.
            Defaults to None.
        envs_idx: array_like, optional
            Indices of the environments to set parameters for. If None, all environments are used. Defaults to None.
        joints_idx: array_like, optional
            Indices of the joints to set parameters for. If None, no joints are targeted. Defaults to None.
        eqs_idx: array_like, optional
            Indices of the equality constraints to set parameters for. If None, none are targeted. Defaults to None.
        """
        self.rigid_solver.set_sol_params(
            sol_params,
            geoms_idx=geoms_idx,
            envs_idx=envs_idx,
            joints_idx=joints_idx,
            eqs_idx=eqs_idx,
        )

    def compute_once(self, key: Hashable, producer: Callable[[], _R]) -> _R:
        """``producer()``, evaluated at most once per pass for each ``key``.

        For terms that derive different values from one intermediate — a body-position and a body-rotation
        error from the same anchor-relative transform, say — so a pass computes it once rather than once per
        term. A pass (:meth:`compute_pass`) spans code that only reads the simulation, so the first result holds
        for the rest of it: each manager's term loop opens one, and ``RLEnvBase.step`` opens one across its whole
        graded phase (metrics, the observation tick, terminations, rewards), which grades one post-physics
        state. The memo is dropped when the pass ends, and the next pass starts empty. Outside a pass (a term's
        ``compute()`` called on its own) ``producer`` simply runs, so a result is never served stale. Every term
        sharing ``key`` receives the same object: treat it as read-only.

        Parameters
        ----------
        key : Hashable
            Names the intermediate, including every argument it depends on (entity, link names, ...): two calls
            with one key, from any term of any manager, are assumed to want the same value.
        producer : Callable[[], T]
            Computes the intermediate.

        Returns
        -------
        T
            ``producer()``'s result, from this pass's memo when an earlier term already produced it.
        """
        cache = self._pass_cache
        if cache is None:
            return producer()
        if key not in cache:
            cache[key] = producer()
        return cache[key]

    @contextmanager
    def compute_pass(self) -> Iterator[None]:
        """Open a :meth:`compute_once` pass for the duration of the block, unless one is already open.

        A pass opened inside another joins it and leaves it open on exit, so a manager's term loop run inside
        the graded phase shares the phase's memo.
        """
        if self._pass_cache is not None:
            yield
            return
        self._pass_cache = {}
        try:
            yield
        finally:
            self._pass_cache = None

    @property
    def fixed_shape_queries(self) -> bool:
        """Whether a query must take its fixed-shape, device-only form now: inside :meth:`fixed_shape_query_scope`.

        A query with a data-dependent size (a contact query trimmed to the most contacts any env has) reads that
        size back to the host and is shaped by it. Its fixed-shape form (the padded contact query, see
        :meth:`~eden.entities.base.Entity._query_contacts`) does neither.
        """
        return self._fixed_shape_queries

    @contextmanager
    def fixed_shape_query_scope(self) -> Iterator[None]:
        """Make every query in the block take its fixed-shape form (:attr:`fixed_shape_queries`).

        The CUDA graph of the graded phase (``EnvOptions.cuda_graph_options``) runs each graded step in one, the
        warm-up, the eager validation runs and the capture alike: a capture cannot record a host read, and the
        runs it compares bit for bit must reduce over the same shapes. A scope opened inside another leaves it
        open on exit.
        """
        previous = self._fixed_shape_queries
        self._fixed_shape_queries = True
        try:
            yield
        finally:
            self._fixed_shape_queries = previous

    def get_entity(self, name: str) -> "Entity":
        """Return the entity registered under ``name``, or raise naming the ones that exist.

        Parameters
        ----------
        name : str
            The entity's name.

        Returns
        -------
        Entity
            The matching entity.

        Raises
        ------
        KeyError
            If no entity is registered under ``name``.
        """
        return resolve_named_entity(self.entities, name)

    def get_entities(self, names: str | list[str]) -> list["Entity"]:
        """
        Get entities by name(s) with regex pattern matching support.

        This method uses `re.match()` which matches patterns from the beginning
        of the entity name. To match entities containing a pattern anywhere,
        prefix with `.*` (e.g., `".*arm"` to match entities containing "arm"; add `$` to anchor the end).

        Parameters
        ----------
        names : str | list[str]
            A single regex pattern string or a list of regex pattern strings
            to match against entity names. Patterns are matched from the
            beginning of entity names.

        Returns
        -------
        list[Entity]
            A list of entities whose names match any of the provided regex patterns.

        Examples
        --------
        >>> # Match all entities starting with "robot"
        >>> env.get_entities("robot.*")
        >>> # Match entities containing "_<number>" (won't match "robot" without a suffix)
        >>> env.get_entities(".*_[0-9]+")
        >>> # Match multiple patterns
        >>> env.get_entities(["robot_[0-9]+", "box.*"])
        >>> # Exact match
        >>> env.get_entities("^robot$")  # or simply "robot$"
        """
        # Normalize input to a list
        patterns = [names] if isinstance(names, str) else names

        matched_entities = []
        for entity_name in self.entities.keys():
            for pattern in patterns:
                if re.match(pattern, entity_name):
                    matched_entities.append(self.entities[entity_name])
                    break  # Avoid adding the same entity multiple times

        return matched_entities

    def get_camera(self, name: str) -> "Camera":
        """Return the camera registered under ``name``.

        Parameters
        ----------
        name : str
            The camera's name.

        Returns
        -------
        Camera
            The matching camera.
        """
        return self.cameras[name]

    def reset(
        self,
        envs_idx: slice | torch.Tensor | None = None,
        *,
        seed: int | None = None,
    ) -> VecEnvReset:
        """Reset the given environments and return their fresh observations.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor, optional
            The environments to reset. If None, all environments are reset.
        seed : int, optional
            If given, seed the RNGs before resetting.

        Returns
        -------
        VecEnvReset
            A ``(observations, extras)`` tuple, where ``extras`` carries reset
            logging info under the ``"log"`` key.
        """
        if envs_idx is None:
            envs_idx = slice(None)
        if seed is not None:
            self.seed(seed)

        self.recorder_manager._on_reset_started(envs_idx)
        with sharing_env_indices(envs_idx):
            self.reset_info = self._reset_idx(envs_idx)
            self._dispatch_post_reset()
        self.recorder_manager._on_reset_finished(envs_idx=envs_idx)
        self.clear_debug_objects()

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, {"log": self.reset_info}

    def step(self, action: torch.Tensor | dict[str, torch.Tensor] | None = None) -> VecEnvStep:
        """Advance the environment by one control step.

        Applies ``action`` through the action manager, runs ``decimation`` physics
        sub-steps, and computes observations. ``EnvBase`` itself has no reward or
        termination; subclasses (:class:`RLEnvBase`) extend this.

        Parameters
        ----------
        action : torch.Tensor | dict[str, torch.Tensor], optional
            The action to apply. If None, no action is applied (the sim still steps).

        Returns
        -------
        VecEnvStep
            A ``(obs, reward, terminated, timeouts, info)`` tuple. ``EnvBase``
            has no reward or termination, so ``reward``, ``terminated``, and
            ``timeouts`` are always ``None``.
        """
        if action is not None:
            self.action_manager.compute(action)

        self.recorder_manager._on_step_started()

        self._run_sim_steps(apply_actions=action is not None)

        self.episode_length_buf += 1

        # NOTE: cameras are deliberately NOT rendered here — the camera API is
        # pull-based, and `start_recording` captures one frame per `render()`
        # call. See `Camera.start_recording`.

        if self.metric_manager is not None:
            self.metric_manager.compute()

        # Phase 7: the once-per-step tick for stateful observation terms, on the post-physics
        # state and after the metrics so a term may integrate one. See
        # ObservationManager.post_physics_refresh for the contract.
        self.observation_manager.post_physics_refresh()

        # Close the recorded step here — below the phases that grade it, above the ones that
        # prepare the next — which is the position it holds in ``RLEnvBase.step`` (the two bodies
        # agree phase-for-phase, pinned by ``tests/envs/test_base.py``). Closing it after the
        # INTERVAL events would corrupt every recorder term that reads simulation state in
        # ``record_post_step()``: an INTERVAL-mode randomizer (``PushByVelocity`` writes the base
        # DOFs outright) perturbs the robot for the *next* step, so ``SysIDRecorder`` would log
        # "after action A the velocity was V" with a V no physics had produced.
        self.recorder_manager._on_step_finished()

        if _INTERVAL in self.event_manager.available_modes:
            self.event_manager.compute(mode=_INTERVAL, dt=self.dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        # NOTE: to match the RLEnvBase.step() return type for consistency
        return (self.obs_buf, None, None, None, {"log": self.reset_info})

    def _scene_step_with_sensor_gating(self, is_last_substep: bool) -> None:
        """Call ``scene.step()``, optionally gating sensor updates on intermediate sub-steps.

        Genesis ``SensorManager.step`` is skipped on intermediate sub-steps when
        ``env_options.update_sensors_every_substep`` is False (the default).

        Intermediate sub-step sensor readings are overwritten by the last
        sub-step's update before anything reads them — ``observation_manager
        .compute()`` runs once at the end of ``step()``. Skipping the
        redundant per-sub-step kernel work (visual-AABB updates, raycast,
        cache population) yields a ~2.3× speedup for perceptive_mimic at
        decimation=4. ``_sensors_by_type`` is the dispatch dict the manager
        loops over; swapping it to empty short-circuits the expensive per-
        sensor-class branch while leaving timeline ring rotations intact
        (rings live in a separate ``_measured_timeline_ring`` dict).

        Safety: when any sensor declares Genesis-level ``delay`` or
        ``history_length``, its ring slots beyond slot 0 need fresh writes every
        sub-step or the delay/history reads return stale, duplicated samples.
        Such a sensor makes this fall back to the per-sub-step path (see
        :meth:`_compute_sensor_substep_skip_safe`), so the optimisation stays safe
        to leave on by default. Ring presence alone is not the signal: Genesis
        allocates rings even for default sensors.
        """
        if is_last_substep or self.env_options.update_sensors_every_substep:
            self.scene.step()
            return

        sim = self.scene._sim
        if not hasattr(sim, "_sensor_manager"):
            self.scene.step()
            return

        sm = sim._sensor_manager
        # Decided once (the sensor options do not change after build), then a single attribute read.
        cached = getattr(self, "_sensor_substep_skip_safe", None)
        if cached is None:
            cached = self._compute_sensor_substep_skip_safe()
            self._sensor_substep_skip_safe = cached
        if not cached:
            self.scene.step()
            return

        saved = sm._sensors_by_type
        sm._sensors_by_type = {}
        try:
            self.scene.step()
        finally:
            sm._sensors_by_type = saved

    def _compute_sensor_substep_skip_safe(self) -> bool:
        """Return True iff every sensor has Genesis-level ``delay == 0`` and ``history_length == 0``.

        Sensors with non-default delay or history consume ring slots beyond slot 0
        in their post-process / delay pipeline, and those slots only stay coherent if
        fresh writes happen on every sub-step. Anything else is safe to skip.
        """
        for sensor in self.sensors.values():
            inner = getattr(getattr(sensor, "_options", None), "sensor", None)
            if inner is None:
                continue
            if getattr(inner, "delay", 0.0) or getattr(inner, "history_length", 0):
                return False
        return True

    def _reset_scene_and_events(self, envs_idx: slice | torch.Tensor | None) -> slice | torch.Tensor | None:
        """Shared ``_reset_idx`` prologue: reset the scene and fire RESET-mode events.

        Sanitizes ``envs_idx``, resets the Genesis scene for those envs (honoring the
        MPM-no-slice and IPC-full-scene-only constraints), then runs RESET-mode events.
        Returns the sanitized ``envs_idx`` that the caller's per-manager resets and
        ``episode_length_buf`` zeroing should use, or ``None`` when no envs need
        resetting (caller should early-return an empty info dict).
        """
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return None
        # Genesis MPM coupler kernels don't accept slice; convert to index tensor.
        if isinstance(envs_idx, slice):
            _scene_envs_idx = self._all_indices[envs_idx]
        else:
            _scene_envs_idx = envs_idx

        # IPC coupler only supports full-scene reset (envs_idx=None).
        # TODO: implement per-env IPC reset via libuipc StateAccessor
        _reset_envs_idx = None if self._has_ipc_coupler else _scene_envs_idx
        # The reset clears every error flag of the reset envs, and so may a setter the RESET events or the managers
        # call on them (the MochiSolver's do). Those of a misconfiguration are kept here meanwhile, for the next
        # control step to raise (see `_raise_kept_misconfigurations`).
        self._keep_misconfigurations()
        self.scene.reset(state=self.reset_state, envs_idx=_reset_envs_idx)
        for solver in (self.rigid_solver, self._mochi_solver):
            if solver is not None:
                _zero_applied_control_force(solver, _reset_envs_idx)

        if _RESET in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.env_options.decimation
            self.event_manager.compute(
                mode=_RESET, envs_idx=envs_idx, global_env_step_count=env_step_count, n_envs=n_envs
            )
        return envs_idx

    def _reset_idx(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, Any]:
        """Reset ``envs_idx``: scene + RESET events, then every manager in :meth:`_manager_order`.

        The first half of a reset; :meth:`_post_reset_idx` is the second and walks the same
        order. Returns the merged per-manager logging extras. Stashes the sanitized index for
        :meth:`_dispatch_post_reset`, which the public entry points call right after this
        returns — so a subclass that brackets this call with extra work (the mimic env's
        motion-buffer refill) completes before any ``post_reset`` hook runs.
        """
        self._last_reset_envs_idx = None
        envs_idx = self._reset_scene_and_events(envs_idx)
        if envs_idx is None:
            return dict()

        reset_info = self._reset_managers(envs_idx)
        # After the loop, never inside it: metric_manager.reset and the episode-length curricula
        # normalize the finished episode by episode_length_buf, so every reset() above must still
        # see the old lengths.
        self.episode_length_buf[envs_idx] = 0
        self._last_reset_envs_idx = envs_idx

        return reset_info

    def _reset_managers(self, envs_idx: slice | torch.Tensor) -> dict[str, Any]:
        """The manager half of :meth:`_reset_idx`: every manager's ``reset()``, in :meth:`_manager_order`."""
        reset_info = dict()
        for name in self._manager_order():
            manager = getattr(self, name)
            if manager is not None:  # optional managers (metric, command, curriculum)
                reset_info.update(manager.reset(envs_idx))
        return reset_info

    def _dispatch_post_reset(self) -> None:
        """Run :meth:`_post_reset_idx` once per completed reset.

        Called by the public entry points (:meth:`reset`, :meth:`RLEnvBase.step`) right
        after ``_reset_idx`` returns — i.e. after the FULL reset sequence of the leaf
        class, since a subclass override of ``_reset_idx`` completes before this runs
        (e.g. the mimic env's post-reset motion-buffer refill). Consumes the ``envs_idx``
        that ``_reset_idx`` stashed; costs nothing on steps that reset no envs.
        """
        envs_idx, self._last_reset_envs_idx = self._last_reset_envs_idx, None
        if envs_idx is None:
            return
        self._post_reset_idx(envs_idx)

    def _post_reset_idx(self, envs_idx: slice | torch.Tensor) -> None:
        """Second half of a reset: fan :meth:`ManagerTermBase.post_reset` out over :meth:`_manager_order`.

        The overridable counterpart of :meth:`_reset_idx`, so a subclass whose manager
        brackets the first half brackets this one the same way
        (:meth:`MimicRLEnvBase._post_reset_idx`).
        """
        for name in self._manager_order():
            manager = getattr(self, name)
            if manager is not None:
                manager.post_reset(envs_idx)

    def _manager_order(self) -> tuple[str, ...]:
        """Name the managers, in order, producers before consumers — construction and both reset halves.

        One order drives three walks: :meth:`_load_managers` (constructing each manager, so a term
        can resolve an upstream handle in its own ``build()``), :meth:`_reset_idx` (each manager's
        ``reset()``) and :meth:`_post_reset_idx` (each term's ``post_reset()``). Four constraints
        bind it, all met here: ``action_manager`` before ``event_manager`` (checked by
        :meth:`_validate_manager_order`); ``metric_manager`` and the curricula anywhere in the loop,
        since ``episode_length_buf`` is zeroed only after it; observation and recorder terms last, as
        pure consumers; and every build-time handle lookup pointing backwards (commands before the
        reward/observation terms that resolve them, metrics before reward/termination, reward and
        termination before the curricula). Attribute names, discovered by the ``_manager`` suffix, so
        the validator can check the tuple against the managers this env owns; restated per class
        rather than composed (:meth:`RLEnvBase._manager_order`), while the factories that go with
        them do compose (:meth:`_manager_factories`); a manager that brackets the loop instead of
        sitting in it goes in :attr:`_BRACKETED_MANAGERS`. Why this order, the DR-draw consequence,
        and the bracket: ``docs/guides/environments.md``, "Manager order".
        """
        return (
            "action_manager",
            "metric_manager",
            "event_manager",
            "observation_manager",
            "recorder_manager",
        )

    def _validate_manager_order(self) -> None:
        """Fail at build if :meth:`_manager_order` is incomplete, inconsistent, or breaks a constraint.

        Writing the order out per class keeps it readable but lets the tuples drift: add a
        manager to :meth:`EnvBase._manager_order` and every subclass override silently drops
        it from both halves of the reset. The symptom would be exactly what ``post_reset``
        exists to prevent — a freshly reset env observing the dead episode's caches, with
        nothing raised — so pin it at the earliest point every manager exists instead.

        A manager assigned outside :meth:`_load_managers`' walk is what the ``owned`` check below
        catches; the mirror case — declared in :meth:`_manager_factories` but named by no position
        in the order, so never constructed at all — raises inside ``_load_managers`` itself, before
        anything can read the missing attribute.
        """
        # Discovery by suffix is the contract (see _manager_order): every manager the env holds
        # is an instance attribute named ``<something>_manager``, and nothing else is.
        owned = {name for name in vars(self) if name.endswith("_manager")}
        order = self._manager_order()
        bracketed = self._BRACKETED_MANAGERS
        problems = []
        if missing := sorted(owned - set(order) - bracketed):
            problems.append(f"never reset or dispatched to: {', '.join(missing)}")
        if unknown := sorted(set(order) - owned):
            problems.append(f"named but not owned by this env: {', '.join(unknown)}")
        if repeated := sorted({name for name in order if order.count(name) > 1}):
            problems.append(f"listed more than once: {', '.join(repeated)}")
        if both := sorted(set(order) & bracketed):
            problems.append(f"both in the loop and declared bracketed: {', '.join(both)}")
        if phantom := sorted(bracketed - owned):
            problems.append(f"declared bracketed but not owned by this env: {', '.join(phantom)}")
        if {"action_manager", "event_manager"} <= set(order) and order.index("event_manager") < order.index(
            "action_manager"
        ):
            problems.append(
                "event_manager must come after action_manager (reset-mode DR terms write the action modifiers' "
                "buffers in reset() so they survive action_manager.reset(); see _ResetModeActuatorDR)"
            )
        if problems:
            raise RuntimeError(
                f"{type(self).__name__}._manager_order() is out of sync with the env's managers — "
                f"{'; '.join(problems)}. Managers owned: {', '.join(sorted(owned))}. "
                f"Order declared: {', '.join(order)}."
                + (f" Bracketed: {', '.join(sorted(bracketed))}." if bracketed else "")
            )


class RLEnvBase(EnvBase[EdenRLConfig]):
    """
    Base class for all RL environments.

    Parameters
    ----------
    env_options: EnvOptions
        Environment options.
    scene_options: SceneOptions
        Scene options.
    observation_options: ObservationManagerOptions
        Observation manager options.
    reward_options: RewardManagerOptions
        Reward manager options.
    termination_options: TerminationManagerOptions
        Termination manager options.
    event_options: EventManagerOptions
        Event manager options.
    curriculum_options: CurriculumManagerOptions
        Curriculum manager options.
    command_options: CommandManagerOptions
        Command manager options.
    action_options: ActionManagerOptions
        Action manager options.
    metric_options: MetricManagerOptions
        Metric manager options.
    recorder_options: RecorderManagerOptions
        Recorder manager options.
    cameras_options: CamerasOptions
        Cameras options.
    sensors_options: SensorsOptions
        Sensors options.
    renderer_options: RendererOptions
        Renderer options.
    show_viewer: bool
        Whether to show the viewer.
    eval_mode: bool
        Whether to run in eval mode.
    **kwargs
        Forwarded to ``gs.Scene`` (see :class:`EnvBase`).
    """

    #: The command manager, or ``None`` when the task declares no command terms; the curriculum
    #: manager likewise. Both are assigned in :meth:`_load_managers` and set back to ``None`` when
    #: empty, and every consumer (this class's own ``step``/``reset``, and the deployment stack)
    #: tests ``is not None`` — so, exactly like :attr:`EnvBase.metric_manager`, they carry a
    #: class-level ``None`` default so that idiom also holds *before* build.
    command_manager: CommandManager | None = None
    curriculum_manager: CurriculumManager | None = None

    #: The CUDA graph :meth:`step` replays its graded phase from (``EnvOptions.cuda_graph_options``); ``None``
    #: runs it eagerly. Created in :meth:`post_build_setup`, once the managers exist.
    _graded_graph: GradedPhaseGraph | None = None

    #: True inside :meth:`graded_capture_scope`, so that :attr:`common_step_counter` can refuse a read the graph
    #: would bake in.
    _graded_capture_active: bool = False

    _OPTIONAL_MANAGERS: frozenset[str] = EnvBase._OPTIONAL_MANAGERS | {"command_manager", "curriculum_manager"}

    _HAS_GRADED_PHASE: ClassVar[bool] = True

    @property
    def common_step_counter(self) -> int:
        """Control steps taken since construction, one count shared by every env.

        Reading it while the graded phase is captured (``EnvOptions.cuda_graph_options``) raises
        :class:`~eden.envs.graded_graph.BakedHostReadError`: the graph would replay the count it read on every
        later step.
        """
        if self._graded_capture_active:
            raise BakedHostReadError(
                "env.common_step_counter was read while the graded phase (metrics, the observation tick, "
                "terminations, rewards) was being captured into a CUDA graph. The graph replays the value it read "
                "at capture on every later step, so a branch on it (an update_interval cadence, a schedule) would "
                "stay on the branch the capture took. Keep the cadence on the device (a step-count tensor advanced "
                "in place, and torch.where on the due envs), read the counter outside the graded phase, or set "
                "EnvOptions(cuda_graph_options=None); catching this error would leave the phase unreplayable."
            )
        return self._common_step_counter

    @common_step_counter.setter
    def common_step_counter(self, value: int) -> None:
        self._common_step_counter = value

    @contextmanager
    def graded_capture_scope(self) -> Iterator[None]:
        """Refuse reads of :attr:`common_step_counter` in the block, which captures the graded phase.

        :class:`~eden.envs.graded_graph.GradedPhaseGraph` captures in one: the graph would replay the count a
        term read there on every later step. Leaving it restores the guard as it found it, so a nested scope
        leaves the outer one open.
        """
        previous = self._graded_capture_active
        self._graded_capture_active = True
        try:
            yield
        finally:
            self._graded_capture_active = previous

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        reward_options: RewardManagerOptions | None = None,
        termination_options: TerminationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        curriculum_options: CurriculumManagerOptions | None = None,
        command_options: CommandManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            env_options=env_options,
            scene_options=scene_options,
            observation_options=observation_options,
            event_options=event_options,
            action_options=action_options,
            metric_options=metric_options,
            recorder_options=recorder_options,
            cameras_options=cameras_options,
            sensors_options=sensors_options,
            renderer_options=renderer_options,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            **kwargs,
        )
        # Handling of default arguments
        self.reward_options = reward_options or RewardManagerOptions()
        self.termination_options = termination_options or TerminationManagerOptions()
        self.curriculum_options = curriculum_options or CurriculumManagerOptions()
        self.command_options = command_options or CommandManagerOptions()

        self.max_episode_length_s = self.env_options.episode_length_s
        if not np.isfinite(self.max_episode_length_s) or self.max_episode_length_s <= 0.0:
            # inf/NaN would die two lines below with an opaque OverflowError,
            # and <= 0 would truncate every env on step one;
            # `max_episode_length` feeds integer buffers and the rsl_rl runner.
            raise ValueError(
                f"env_options.episode_length_s must be a positive finite value, got "
                f"{self.max_episode_length_s}. For unbounded episodes, keep a finite horizon "
                "and omit the `time_out` termination term instead — truncation then never fires."
            )
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        self.common_step_counter = 0

        # Lazily allocated zero-initialized cache for ``extras['final_observations']``.
        # See ``EnvOptions.record_final_observations`` and ``RLEnvBase.step()``.
        self._final_obs_cache: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    def _manager_factories(self) -> dict[str, Callable[[], Any]]:
        """Add the RL managers to the base map; :meth:`_manager_order` decides when each is built.

        Written in that order for reading only — ``_load_managers`` walks the declared order, not
        this dict. The dependencies it satisfies are real and all point the same way: command terms
        are resolved at build by reward and observation terms, metric terms by reward and
        termination terms, and reward *and* termination terms by the curricula.
        """
        from eden.managers import (
            CommandManager,
            CurriculumManager,
            RewardManager,
            TerminationManager,
        )

        return super()._manager_factories() | {
            "command_manager": lambda: CommandManager(self, self.command_options),
            "termination_manager": lambda: TerminationManager(self, self.termination_options),
            "reward_manager": lambda: RewardManager(self, self.reward_options),
            "curriculum_manager": lambda: CurriculumManager(self, self.curriculum_options),
        }

    @gs.assert_built
    def summary(self) -> str:
        """Return a human-readable summary, including RL managers (reward, termination, etc.)."""
        msg = super().summary()
        if self.command_manager is not None:
            msg += self.command_manager.summary()
        if self.curriculum_manager is not None:
            msg += self.curriculum_manager.summary()
        msg += self.reward_manager.summary()
        msg += self.termination_manager.summary()
        return msg

    @property
    def unwrapped(self) -> "RLEnvBase":
        """The underlying environment (returns ``self``; provided for wrapper compatibility)."""
        return self

    def step(
        self,
        action: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> VecEnvStep:
        """Advance the RL environment by one control step.

        Applies ``action``, runs ``decimation`` physics sub-steps, then computes
        metrics, terminations, and rewards. Done environments are auto-reset
        **after** reward/termination but **before** observation, so the returned
        observations reflect the fresh post-reset state for those environments.

        Parameters
        ----------
        action : torch.Tensor | dict[str, torch.Tensor], optional
            The action to apply. If None, no action is applied (the sim still steps).

        Returns
        -------
        VecEnvStep
            A ``(obs, reward, terminated, timeouts, info)`` tuple.
        """
        if action is not None:
            self.action_manager.compute(action)

        self.recorder_manager._on_step_started()

        # Advance command references BEFORE physics (no-op unless a command term
        # overrides pre_physics_advance, e.g. MotionCommand). Advance-then-grade:
        # advancing the reference before the physics sub-steps makes the post-
        # physics robot get graded against the temporally-aligned reference frame
        # (matching MotionManager) instead of a one-control-step-stale one. Any
        # wrap-teleport therefore precedes physics, keeping forward kinematics fresh.
        if self.command_manager is not None:
            self.command_manager.pre_physics_advance()

        self._run_sim_steps(apply_actions=action is not None)

        # Update env counters.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Refresh command terms' cached robot state on the just-computed post-physics
        # state, so metric/termination/reward grade the CURRENT step (S_N) rather than
        # the previous one. Matches IsaacLab/mjlab. See CommandManager.post_physics_refresh
        # for the ordering contract.
        if self.command_manager is not None:
            self.command_manager.post_physics_refresh()

        # NOTE: cameras are deliberately NOT rendered here — the camera API is
        # pull-based, and `start_recording` captures one frame per `render()`
        # call. See `Camera.start_recording`.

        # Phases 6-9: metrics, the observation tick, terminations and rewards, from a CUDA graph when
        # EnvOptions.cuda_graph_options asks for one.
        if self._graded_graph is not None:
            self.reset_buf, self.reward_buf = self._graded_graph()
        else:
            self.reset_buf, self.reward_buf = self._compute_graded()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_timeouts = self.termination_manager.timeouts
        # Gate on is_recording so `.any()` doesn't force a GPU→CPU sync every step during training.
        if self.recorder_manager.is_recording:
            if self.metric_manager is not None and hasattr(self.metric_manager, "success_buf") and self.reset_buf.any():
                self.recorder_manager.set_episode_success(
                    self.reset_buf, self.metric_manager.success_buf[self.reset_buf]
                )
        self.recorder_manager._on_step_finished(self.reset_buf)

        # Snapshot pre-reset observations for done envs (opt-in via
        # ``EnvOptions.record_final_observations``). Must run before
        # ``_reset_idx`` since ``scene.reset`` wipes the done envs' state.
        if self.env_options.record_final_observations:
            final_obs = self._record_final_observations()
        else:
            final_obs = None

        # Always call _reset_idx — bool-mask ops are no-ops for all-False. The mask's ids are taken once
        # here, for every consumer that gathers the resetting rows, and its count gates the reset: one
        # GPU->CPU sync for the whole reset rather than one per consumer.
        with sharing_env_indices(self.reset_buf):
            self.reset_info = self._reset_idx(self.reset_buf)
            # Terms rebuild per-step caches from the final post-reset state, before the
            # recorder snapshots new-episode data and observations read them below.
            self._dispatch_post_reset()

        # Record new-episode data after reset, such as initial state snapshots.
        self.recorder_manager._on_reset_finished(envs_idx=self.reset_buf)

        # Curricula are frozen at eval. The gate lives in CurriculumManager, not here, so that it
        # covers reset() as well — see CurriculumManager.compute for why difficulty must not move
        # during an eval rollout, and why the freeze has to be all-or-nothing.
        if self.curriculum_manager is not None:
            self.curriculum_manager.compute()

        if self.command_manager is not None:
            self.command_manager.compute(dt=self.dt)

        if _INTERVAL in self.event_manager.available_modes:
            self.event_manager.compute(mode=_INTERVAL, dt=self.dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        if not self.training and self.command_manager is not None:
            self.command_manager.draw_vis()

        extras: dict[str, Any] = {"log": self.reset_info}
        if final_obs is not None:
            extras["final_observations"] = final_obs

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_timeouts,
            extras,
        )

    @gs.assert_built
    def post_build_setup(self):
        super().post_build_setup()
        if self.env_options.cuda_graph_options is not None:
            self._graded_graph = GradedPhaseGraph(self, self._compute_graded, self.env_options.cuda_graph_options)

    def _compute_graded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The graded phase of :meth:`step` (phases 6-9), eagerly: ``(reset_buf, reward_buf)``.

        Everything here reads the post-physics state and writes only the managers' own buffers, which is what
        lets :class:`~eden.envs.graded_graph.GradedPhaseGraph` replay it.
        """
        # One compute_once pass over the whole phase: it grades one post-physics state, so an intermediate
        # a metric, an observation tick, a termination or a reward derives from it (an anchor-relative warp,
        # say) is computed once for all of them.
        with self.compute_pass():
            if self.metric_manager is not None:
                self.metric_manager.compute()

            # Phase 7: tick stateful observation terms once, on the post-physics state and after the
            # command caches (5) and metrics (6) they may read, above the graded layer and the
            # auto-reset. compute() at phase 19 is a read and is NOT this tick — see
            # ObservationManager.post_physics_refresh.
            self.observation_manager.post_physics_refresh()

            reset_buf = self.termination_manager.compute()
            reward_buf = self.reward_manager.compute(dt=self.dt)
        return reset_buf, reward_buf

    def _compute_final_observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Return the pre-reset observation dict for snapshotting (hook).

        Default: full-batch ``observation_manager.compute(update_history=False)``.
        ``update_history=False`` is critical — the post-reset compute at the
        end of ``step()`` advances the history buffer once with
        ``update_history=True``; advancing here too would skip a frame on
        non-done envs. For history-bearing groups the manager still computes
        the fresh post-physics frame and exposes it in the most-recent slot of
        the returned tensor (via ``CircularBuffer.peek_buffer``), without
        mutating the underlying buffer. Subclass to swap in motion-frame
        indices or skip groups whose terms are unsafe to evaluate against the
        post-physics state.
        """
        return self.observation_manager.compute(update_history=False)

    def _record_final_observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Snapshot ``_compute_final_observations`` into the cached ``_final_obs_cache``.

        Only the rows for ``self.reset_buf`` are kept.

        Allocates the cache lazily; rebuilds the cache when the source
        tensor's shape or dtype changes (e.g. when the env is rebuilt at a
        different ``num_envs``). The slice-copy from ``snapshot[reset_buf]``
        is required, not just an allocation optimization — for
        history-bearing groups ``compute(update_history=False)`` returns a
        view into ``CircularBuffer.buffer`` that the post-reset
        ``compute(update_history=True)`` mutates.
        """
        snapshot = self._compute_final_observations()
        return self._copy_into_final_obs_cache(snapshot, self._final_obs_cache, self.reset_buf)

    @classmethod
    def _copy_into_final_obs_cache(
        cls,
        source: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        cache: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Sync-free: bool-mask advanced indexing (``cached[mask] = value[mask]``)
        # internally calls ``nonzero()`` on CUDA to size the gather/scatter and
        # forces a GPU→CPU sync — exactly the cost the design avoids elsewhere
        # (see ``EnvOptions.record_final_observations``). ``torch.where`` over a
        # broadcastable bool mask has fully static shapes and stays on-device.
        for key, value in source.items():
            if isinstance(value, dict):
                sub_cache = cache.get(key)
                if not isinstance(sub_cache, dict):
                    sub_cache = {}
                    cache[key] = sub_cache
                cls._copy_into_final_obs_cache(value, sub_cache, mask)
                continue

            cached = cache.get(key)
            if (
                not isinstance(cached, torch.Tensor)
                or cached.shape != value.shape
                or cached.dtype != value.dtype
                or cached.device != value.device
            ):
                cached = torch.zeros_like(value)
                cache[key] = cached
            mask_view = mask.view(mask.shape[0], *([1] * (value.ndim - 1)))
            zero = torch.zeros((), dtype=value.dtype, device=value.device)
            torch.where(mask_view, value, zero, out=cached)
        return cache

    def _manager_order(self) -> tuple[str, ...]:
        """Interleave the RL managers, keeping producers ahead of their cache consumers.

        Commands are the upstream reference every graded layer reads; metric terms feed
        termination and reward; the curriculum reads the graded outcome last — so a reward-driven
        curriculum must consume *published* state (``PenaltyCurriculum(tracker_name=...)`` →
        ``get_state()``), not ``RewardManager``'s accumulators, which are already cleared.
        Restated rather than spliced because these slot *between* the base entries;
        :meth:`_validate_manager_order` keeps the restatement honest.
        """
        return (
            "action_manager",
            "command_manager",
            "metric_manager",
            "termination_manager",
            "reward_manager",
            "curriculum_manager",
            "event_manager",
            "observation_manager",
            "recorder_manager",
        )


class MimicRLEnvBase(RLEnvBase, EnvBase[EdenMimicConfig]):
    """Base class for all Mimic RL environments.

    The extra ``EnvBase[EdenMimicConfig]`` base is how this leaf re-binds its config: ``RLEnvBase``
    is itself instantiable so it binds ``EdenRLConfig`` outright rather than staying generic, which
    leaves it unsubscriptable — a plain ``MimicRLEnvBase(RLEnvBase)`` would inherit that binding.
    """

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        reward_options: RewardManagerOptions | None = None,
        termination_options: TerminationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        curriculum_options: CurriculumManagerOptions | None = None,
        command_options: CommandManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        motion_options: MotionManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            env_options=env_options,
            scene_options=scene_options,
            observation_options=observation_options,
            reward_options=reward_options,
            termination_options=termination_options,
            event_options=event_options,
            curriculum_options=curriculum_options,
            command_options=command_options,
            action_options=action_options,
            metric_options=metric_options,
            recorder_options=recorder_options,
            cameras_options=cameras_options,
            sensors_options=sensors_options,
            renderer_options=renderer_options,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            **kwargs,
        )

        self.motion_options = motion_options or MotionManagerOptions()

    @gs.assert_built
    def _load_managers(self):
        from eden.managers.motion_manager import MotionManager

        # NOTE: motion_manager must be created BEFORE super()._load_managers()
        # because observation/reward/termination terms may reference it during init.
        self.motion_manager = MotionManager(self, self.motion_options)

        super()._load_managers()

    @gs.assert_built
    def summary(self) -> str:
        """Return a human-readable summary, including the motion manager."""
        msg = super().summary()
        msg += self.motion_manager.summary()
        return msg

    def step(
        self,
        action: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> VecEnvStep:
        """Advance the motion-imitation environment by one control step.

        Updates the motion reference for the current step, then delegates to
        :meth:`RLEnvBase.step`.

        Parameters
        ----------
        action : torch.Tensor | dict[str, torch.Tensor], optional
            The action to apply. If None, no action is applied (the sim still steps).

        Returns
        -------
        VecEnvStep
            A ``(obs, reward, terminated, timeouts, info)`` tuple.
        """
        self.motion_manager.compute(dt=self.dt)
        return super().step(action)

    # The motion manager brackets the manager loop instead of sitting in it (see
    # _reset_idx / _post_reset_idx); declared so _validate_manager_order knows it is handled.
    _BRACKETED_MANAGERS: frozenset[str] = frozenset({"motion_manager"})

    def _reset_idx(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, Any]:
        # Two-phase motion-manager reset bracketing super()._reset_idx (it cannot be a slot in
        # _manager_order: phase 1 must precede the scene reset itself):
        #   Phase 1: sample motion selection + roll back _timestamp. Must run
        #            FIRST so RSI events inside super()._reset_idx can read the
        #            freshly-sampled init state via
        #            ``motion_manager.get_reset_reference_state``.
        #   Phase 2: refill ``_data`` around the scene reset and RESET events
        #            (``_reset_scene_and_events`` below): at the RSI events' init-state
        #            request when the window starts at the current frame, which then
        #            seeds them from its first frame, else right after the events.
        #            Every manager reset after them sees the new episode's window, so
        #            reading ``motion_manager.data`` there raises (``_reset_managers``).

        if isinstance(envs_idx, slice):
            envs_idx = self.motion_manager._all_indices[envs_idx]

        reset_info = self.motion_manager.reset(envs_idx)
        reset_info.update(super()._reset_idx(envs_idx))
        return reset_info

    def _reset_scene_and_events(self, envs_idx: slice | torch.Tensor | None) -> slice | torch.Tensor | None:
        # Phase 2 of ``_reset_idx``: the motion window is filled by the time the manager resets run.
        with self.motion_manager.resetting(envs_idx):
            return super()._reset_scene_and_events(envs_idx)

    def _reset_managers(self, envs_idx: slice | torch.Tensor) -> dict[str, Any]:
        # The window already holds the new episode here, so a manager reset grading the ended one
        # reading ``motion_manager.data`` would grade the wrong episode: it raises instead.
        with self.motion_manager.managers_resetting():
            return super()._reset_managers(envs_idx)

    def _post_reset_idx(self, envs_idx: slice | torch.Tensor) -> None:
        # The bracket's other half: refresh motion caches before every manager that may consume
        # them — motion first here mirrors motion_manager.reset() running first in _reset_idx.
        self.motion_manager.post_reset(envs_idx)
        super()._post_reset_idx(envs_idx)


def env_class_for_config(cfg: EdenConfig) -> type[EnvBase]:
    """Return the env class that builds ``cfg``.

    Maps each config leaf to its env class, mirroring the parallel hierarchies:

    * ``EdenMimicConfig`` → ``MimicRLEnvBase``
    * ``EdenSplatRLConfig`` → ``SplatRLEnvBase``
    * ``EdenSplatConfig``   → ``SplatEnvBase``
    * ``EdenRLConfig``      → ``RLEnvBase``
    * ``EdenNyxSplatConfig`` → ``NyxSplatEnvBase``
    * ``EdenConfig``        → ``EnvBase``

    The single *dispatch* point for config→env-class, shared by the ``from_config``
    wrappers, the CLI play path, and the smoke harness. The configs form an inheritance chain
    (``EdenMimicConfig``, ``EdenSplatRLConfig`` ⊂ ``EdenRLConfig`` ⊂ ``EdenConfig``, and
    ``EdenSplatConfig`` / ``EdenNyxSplatConfig`` ⊂ ``EdenConfig``), so the pairs are checked
    most-specific first — ``EdenSplatRLConfig`` before the generic ``EdenRLConfig`` branch that
    would otherwise shadow it, and ``EdenSplatConfig`` before the ``EdenConfig`` fallback.

    The splat env classes are imported lazily: they pull optional renderer dependencies, and a
    config that is not a splat config must not require them. A splat config that *is* passed
    while those dependencies are missing gets the ``ImportError`` — that is the actionable
    failure, and far better than a silent downgrade to ``RLEnvBase``, which would build an env
    with no splat renderer and no error.

    The reverse direction is not stated here: each env declares its own leaf as
    :attr:`EnvBase._CONFIG_CLASS`, following the definition-site convention Eden uses for
    tasks rather than keeping a second central index. Deduplicating the two would mean
    deriving one by reflection over ``__subclasses__`` — import-order dependent — so they are
    pinned to each other by
    ``tests/envs/test_base.py::test_every_leaf_accepts_its_own_config_and_only_the_two_keywords``
    instead. A new config leaf must therefore update both, and that test says so if it doesn't.
    """
    from eden.utils.configs import (
        EdenMimicConfig,
        EdenNyxSplatConfig,
        EdenRLConfig,
        EdenSplatConfig,
        EdenSplatRLConfig,
    )

    if isinstance(cfg, EdenNyxSplatConfig):
        from eden.envs.nyx_splat import NyxSplatEnvBase

        return NyxSplatEnvBase
    # Checked before EdenRLConfig below: EdenSplatRLConfig subclasses it, so the generic RL branch
    # would otherwise shadow it (and drop splat_options silently).
    if isinstance(cfg, EdenSplatRLConfig):
        from eden.envs.splat import SplatRLEnvBase

        return SplatRLEnvBase
    # Checked before the EdenConfig fallback below (it is an EdenConfig, not an EdenRLConfig).
    if isinstance(cfg, EdenSplatConfig):
        from eden.envs.splat import SplatEnvBase

        return SplatEnvBase

    for cfg_cls, env_cls in (
        (EdenMimicConfig, MimicRLEnvBase),
        (EdenRLConfig, RLEnvBase),
        (EdenConfig, EnvBase),
    ):
        if isinstance(cfg, cfg_cls):
            return env_cls
    raise TypeError(f"No env class for config of type {type(cfg).__name__}")
