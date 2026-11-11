"""Locomotion observation terms (gait phase, projected gravity, base height, height scan)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
from genesis.typing import Vec2FType
import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import base_height_above_ground, gravity_vec, projected_gravity, resolve_terrain
from eden.options.managers.observations import ObservationTermOptions
from eden.utils.geom import transform_by_quat_yaw_approx
from eden.utils.misc import resolve_named_entity
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.managers.terms.commands.gait import GaitPhaseCommand


@OBSERVATION_TERM_REGISTRY.register()
def sin_phase(env: EnvBase, *, command_name: str) -> torch.Tensor:
    """Sine of the gait phase for each foot. Shape (num_envs, num_feet)."""
    gait_cmd: GaitPhaseCommand = env.command_manager.get_term(command_name)
    return torch.sin(gait_cmd.command)


@OBSERVATION_TERM_REGISTRY.register()
def cos_phase(env: EnvBase, *, command_name: str) -> torch.Tensor:
    """Cosine of the gait phase for each foot. Shape (num_envs, num_feet)."""
    gait_cmd: GaitPhaseCommand = env.command_manager.get_term(command_name)
    return torch.cos(gait_cmd.command)


@OBSERVATION_TERM_REGISTRY.register()
def base_height(
    env: EnvBase,
    *,
    entity_name: str = "robot",
    terrain_entity_name: str | None = None,
    relative: bool = False,
) -> torch.Tensor:
    """Height of the entity's base above the ground, shape ``(num_envs, 1)``.

    The locomotion counterpart of Isaac Lab's ``base_pos_z``, and the observation half of
    :func:`~eden.managers.terms.terminations.locomotion.root_height_below_minimum` — the two read the
    same :func:`~eden.managers.terms.utils.base_height_above_ground` helper, so an observation and a
    termination configured alike cannot disagree about what "height" means.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str, optional
        Entity whose base height is reported. Defaults to ``"robot"``.
    terrain_entity_name : str | None, optional
        Terrain entity to measure *against*. ``None`` (the default) reports the world-frame ``z``,
        which is the height above the ground only on flat terrain; naming a
        :class:`~eden.entities.terrain.Terrain` subtracts the terrain height under the base's ``xy``
        so the feature means the same thing everywhere on a heightfield.
    relative : bool, optional
        Frame the base pose is read in — see
        :func:`~eden.managers.terms.observations.proprio.base_pos`. Defaults to ``False`` (the solver
        frame), matching the other base-pose observations and
        :func:`~eden.managers.terms.rewards.locomotion.base_height_l2`.
    """
    terrain = resolve_terrain(env.entities, terrain_entity_name)
    height = base_height_above_ground(resolve_named_entity(env.entities, entity_name), terrain, relative=relative)
    return height.unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
class ProjectedGravity(ObservationTerm):
    """Gravity direction expressed in the entity's base frame.

    Caches only the constant world-frame gravity vector in :meth:`build`; the entity handle is
    resolved each :meth:`compute` via ``resolve_entity`` so the deployment's sim→``RobotStateEntity``
    swap is picked up automatically (a cached handle would keep reading the simulated robot on hardware).
    """

    entity_name: str = "robot"

    def build(self) -> None:
        self._grav = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        return projected_gravity(self.resolve_entity(self.entity_name).get_quat(), self._grav)


@OBSERVATION_TERM_REGISTRY.register()
class ScaledCommands(ObservationTerm):
    """Command vector multiplied by a per-component scale; prefer the built-in scale=[...] modifier.

    For plain per-component scaling prefer the built-in ``scale=[...]`` post-compute modifier
    (``ObsTerm.configure(func=..., scale=[2.0, 1.0])``). ``command_scale`` is the per-component
    multiplier (named to avoid colliding with the inherited post-compute ``scale`` modifier). Both
    the command ref and the scale tensor are resolved once in :meth:`build`.

    **The swap is exact only when the term configures no ``noise`` and no ``clip``.** This class
    scales inside ``compute()``, so the manager's noise and clip see the *scaled* command; the
    ``scale`` modifier is applied *after* both (``compute_group`` runs noise -> clip -> scale). So
    migrating a term that also sets ``noise`` divides the effective noise std by the per-component
    factor, and one that sets ``clip`` reinterprets the bounds from scaled units into raw ones.
    Rescale those two alongside the swap, or keep this class.
    """

    command_name: str = ""
    command_scale: list[float] = []

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)
        self._scale = torch.tensor(self.command_scale, device=self.device, dtype=torch.float32)

    def compute(self) -> torch.Tensor:
        return torch.mul(self._command.command, self._scale)


@OBSERVATION_TERM_REGISTRY.register()
class VelocityTrackingErrorIntegral(ObservationTerm):
    """Running integral of the base-velocity tracking error ``(N, 3)``.

    The worked example of a stateful observation term.

    ``∫ (commanded − measured) dt`` over ``[vx, vy, wz]`` in the base frame, against a velocity
    command laid out ``[vx, vy, wz]`` (:class:`~eden.managers.terms.commands.velocity.UniformVelocityCommand`).
    Zero at reset, exactly ``k`` increments after ``k`` control steps, and clamped per component to
    ``±max_abs`` so a persistent error (a robot that cannot reach the commanded speed) saturates the
    feature instead of growing without bound — the anti-windup clamp of a PI controller.

    This term exists as much to be copied as to be used. It is the reference for per-step state on an
    observation term, which the manager's ``compute()`` cannot hold because that call is a *read* and
    runs several times a step (:class:`~eden.managers.observation_manager.ObservationTerm`):

    * :meth:`build` resolves the command (a bad name raises at env construction), touches the entity,
      and allocates the buffer — outside any inference-mode region, so every hook may write it.
    * :meth:`post_physics_refresh` is the once-per-step tick: it reads the post-physics robot and adds
      the increment **in place**. It runs at phase 7 — after the command caches and metrics, above the
      graded layer and the auto-reset — so the
      final-observation snapshot carries this step's increment and ``reset()`` zeroes it for the envs
      that restart before their first fresh observation. The deploy loop ticks it once per control tick.
    * :meth:`reset` zeroes the rows of the restarting envs.
    * :meth:`compute` returns the buffer, live — like a metric buffer, not a copy; the manager's
      snapshot paths copy what they keep. Nothing else.

    The entity is resolved in the hook, not cached in ``build`` — the deployment swaps
    ``env.entities[name]`` for a ``RobotStateEntity`` and a cached handle would keep integrating the
    simulated robot on hardware. ``self._env.dt`` is the control dt, so the integral is in
    ``m`` / ``rad`` regardless of decimation.
    """

    entity_name: str = "robot"
    command_name: str = "velocity_command"
    max_abs: float = 1.0

    def build(self) -> None:
        # ``getattr``: a plain ``EnvBase`` has no command manager attribute at all, and that is the
        # same misconfiguration as an RL env with none — one message for both.
        command_manager = getattr(self._env, "command_manager", None)
        if command_manager is None:
            raise ValueError(f"{term_subject(self)} requires a command manager with a '{self.command_name}' term.")
        # ``get_term`` raises naming the available commands; touching the entity does the same for
        # ``entity_name``. Both fail here, at env construction, rather than on the first tick.
        self._command = command_manager.get_term(self.command_name)
        self.resolve_entity(self.entity_name)
        # The command buffer exists at build (the command manager is constructed first), so its width
        # is checked here: a narrower command would otherwise fail on the first tick as a broadcasting
        # error in ``post_physics_refresh``, not at env construction.
        command_dim = self._command.command.shape[-1]
        if command_dim < 3:
            raise ValueError(
                f"{term_subject(self)}: command '{self.command_name}' publishes {command_dim} component(s); this term "
                f"integrates the first three as [vx, vy, wz] and needs at least 3 (UniformVelocityCommand's layout)."
            )
        if self.max_abs <= 0.0:
            raise ValueError(f"{term_subject(self)}: max_abs must be positive, got {self.max_abs}.")
        self._integral = torch.zeros(self.num_envs, 3, device=self.device)

    def post_physics_refresh(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        measured = torch.cat([entity.get_vel(frame="body")[:, :2], entity.get_ang(frame="body")[:, 2:3]], dim=-1)
        # In place: the buffer was allocated in build(), and this may run under inference mode.
        self._integral.add_((self._command.command[:, :3] - measured) * self._env.dt).clamp_(
            -self.max_abs, self.max_abs
        )

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> None:
        self._integral[slice(None) if envs_idx is None else envs_idx] = 0.0

    def compute(self) -> torch.Tensor:
        return self._integral


@OBSERVATION_TERM_REGISTRY.register()
class GridScanDots(ObservationTerm):
    """Height scan of the terrain at the given xy coordinates.

    This is specifically used for Terrain entity. For the general LiDAR sensor, use the LiDAR observation term.

    Parameters
    ----------
    entity_name : str, optional
        The name of the entity where the grid is "attached" to. Defaults to "robot".
    terrain_entity_name : str, optional
        The name of the terrain entity. Defaults to "terrain".
    measured_points_x : list[float], optional
        The x coordinates (m) of the measured points, in the base's yaw-aligned frame. Defaults to the values in
        the class.
    measured_points_y : list[float], optional
        The y coordinates (m) of the measured points, in the base's yaw-aligned frame. Defaults to the values in
        the class.
    flatten_measured_points : bool, optional
        Whether to flatten the heights per env. Defaults to True: shape
        ``(num_envs, len(measured_points_x) * len(measured_points_y))``. If False, the shape is
        ``(num_envs, len(measured_points_x), len(measured_points_y))``.
    offset_from_base : bool, optional
        If True (default), return each terrain height relative to the base height (``terrain_z - base_z``),
        with NaN zeroed and the result clamped to ``clamp_offset_range``. If False, return the absolute
        world-frame terrain heights, unclamped.
    clamp_offset_range : tuple[float, float], optional
        The ``(min, max)`` clamp (m) applied when ``offset_from_base`` is True. Defaults to ``(-5.0, 5.0)``.
    """

    entity_name: str = "robot"
    terrain_entity_name: str = "terrain"
    measured_points_x: list[float] = [
        -0.8,
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ]
    measured_points_y: list[float] = [
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    flatten_measured_points: bool = True
    offset_from_base: bool = True
    clamp_offset_range: Vec2FType = (-5.0, 5.0)

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env, options)
        self._grid_buffer: torch.Tensor | None = None

    def build(self) -> None:
        # Cache only grid buffers; resolve entities lazily in compute() (see ManagerTermBase.resolve_entity).
        entity = self.resolve_entity(self.entity_name)
        if entity.has_immobile_base:
            raise ValueError(
                f"{term_subject(self)} needs an entity whose base moves through the world; "
                f"it has an immobile base (fixed, or welded onto another entity)."
            )
        _ = self.resolve_entity(self.terrain_entity_name)  # fail fast if the terrain entity is missing
        self._grid_buffer = torch.zeros(
            (
                self._env.num_envs,
                len(self.measured_points_x),
                len(self.measured_points_y),
                3,
            ),
            device=gs.device,
            dtype=gs.tc_float,
        )
        measured_points_x = torch.tensor(self.measured_points_x, device=gs.device, dtype=gs.tc_float)
        measured_points_y = torch.tensor(self.measured_points_y, device=gs.device, dtype=gs.tc_float)
        grid_x, grid_y = torch.meshgrid(measured_points_x, measured_points_y, indexing="ij")
        self._grid_buffer[..., 0] = grid_x
        self._grid_buffer[..., 1] = grid_y
        self.num_measured_points = grid_x.numel()
        self.num_measured_points_x = len(self.measured_points_x)
        self.num_measured_points_y = len(self.measured_points_y)
        self.grid = torch.zeros(
            (
                self._env.num_envs,
                len(self.measured_points_x),
                len(self.measured_points_y),
                2,
            ),
            device=gs.device,
            dtype=gs.tc_float,
        )

    def compute(self, *args, **kwargs):
        base_pose = self.resolve_entity(self.entity_name).get_qpos(slice(0, 7))
        self.grid = transform_by_quat_yaw_approx(
            self._grid_buffer.view(-1, 3),
            base_pose[:, 3:7].view(-1, 4).repeat_interleave(self.num_measured_points, dim=0),
        ).view(-1, self.num_measured_points_x, self.num_measured_points_y, 3)[..., :2]
        self.grid += base_pose[:, :2].view(-1, 1, 1, 2)
        scan_dots = self.resolve_entity(self.terrain_entity_name).get_height(self.grid)  # B, h_out, w_out

        if self.offset_from_base:
            scan_dots -= base_pose[:, 2].view(-1, 1, 1)
            min_clamp, max_clamp = self.clamp_offset_range
            scan_dots = torch.nan_to_num(scan_dots, nan=0.0, posinf=max_clamp, neginf=min_clamp).clamp(
                min=min_clamp, max=max_clamp
            )

        if self.flatten_measured_points:
            return scan_dots.view(-1, self.num_measured_points)
        else:
            return scan_dots
