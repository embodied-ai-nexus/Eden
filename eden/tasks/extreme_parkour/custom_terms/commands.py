"""Goal-following parkour command term (Extreme Parkour, Cheng et al. 2023).

:class:`ParkourCommand` owns the per-env terrain placement (which cell of the
level x type grid each env is on, and the cell numbering) and the goal-waypoint
tracking state (current goal index, world-frame relative target, target yaw).
It reads the goal grid the :class:`~eden.entities.terrain.Terrain` entity
exposes after build (``goals`` / ``goals_count`` / ``subterrain_origins``).

Reset ordering contract
-----------------------
Eden fires reset-mode events *before* ``command_manager.reset`` (the inverse of
IsaacLab). The spawn event
(:class:`~eden.tasks.extreme_parkour.custom_terms.events.ResetParkourRootState`)
therefore first asks its reset-configuration sampler -- by default the terrain-level
curriculum (:class:`~eden.tasks.extreme_parkour.custom_terms.curricula.TerrainLevelCurriculum`)
-- for the envs' new cells, applies them via the public
:meth:`ParkourCommand.assign_terrain_cells` (which re-gathers the per-env cell
buffers), and only then writes the robot root pose. :meth:`ParkourCommand.reset`
verifies this happened (via a per-env step stamp) and raises if the event is
missing from ``event_options``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm
from eden.utils.geom import wrap_to_pi
from eden.utils.misc import sanitize_envs_idx
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.terrain import Terrain
    from eden.envs.base import RLEnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class ParkourCommand(CommandTerm):
    """SE(2) velocity command driven by terrain goal waypoints.

    ``command`` is ``(num_envs, 3)`` = ``(vx, vy=0, wz)``: ``vx`` is resampled
    uniformly every ``resampling_time_range`` seconds and ``wz`` is a
    proportional heading controller ``clip(wrap_to_pi(heading_target - yaw) *
    heading_control_stiffness, -1, 1)`` with a deadband (``ang_vel_clip``).

    Beyond the command itself the term exposes (all world-frame, refreshed by
    :meth:`post_physics_refresh` each control step):

    - ``target_pos_rel`` (N, 2): current goal xy minus root xy,
    - ``target_yaw`` / ``next_target_yaw`` (N,): yaw towards the current / next goal,
    - ``cur_goal_idx`` / ``env_goals_count`` (N,): goal progress,
    - ``env_is_flat`` (N,) bool: whether the env's cell is a flat (obstacle-suppressed) course,
    - ``terrain_levels`` / ``terrain_types`` (N,): the env's *applied* cell placement in the
      terrain grid, decided at reset by the spawn event's reset-configuration sampler (by default
      ``TerrainLevelCurriculum``, which moves an env between rows and keeps the column dealt it here;
      see ``ResetParkourRootState``),
    - ``dis_to_start`` (N,): distance walked from the spawn point (feeds the level curriculum).
    """

    entity_name: str = "robot"
    terrain_entity_name: str = "terrain"
    resampling_time_range: tuple[float, float] = (6.0, 6.0)
    lin_vel_x_range: tuple[float, float] = (0.3, 0.8)
    heading_range: tuple[float, float] = (-1.6, 1.6)
    heading_control_stiffness: float = 0.8
    lin_vel_clip: float = 0.2
    ang_vel_clip: float = 0.4
    next_goal_threshold: float = 0.2
    reach_goal_delay: float = 0.1
    num_future_goal_obs: int = 2
    spawn_offset_x: float = 7.0

    def __init__(self, env: RLEnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        # All buffers are allocated in build() — their shapes depend on the terrain goal grid.

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        # Fail fast on both handles. The robot is touched here but resolved lazily in the
        # hot paths (resolve_entity), so a deployment entity swap is picked up.
        _ = self._env.entities[self.entity_name]
        terrain: Terrain = self._env.entities[self.terrain_entity_name]

        goals = terrain.goals  # (rows, cols, max_goals, 3); raises if the grid produced no goals
        self._rows, self._cols = int(goals.shape[0]), int(goals.shape[1])
        # Pad each cell's goal list with `num_future_goal_obs` repeats of the last goal so
        # `cur_goal_idx + 1` gathers stay in-bounds even after the last real goal is reached.
        future_pad = goals[:, :, -1:].expand(-1, -1, self.num_future_goal_obs, -1)
        self._cell_goals = torch.cat([goals, future_pad], dim=2).contiguous()  # (rows, cols, G, 3)
        self._cell_goals_count = terrain.goals_count  # (rows, cols) long
        self._cell_origins = terrain.subterrain_origins  # (rows, cols, 3)

        # Per-cell "flat course" mask from the terrain generator grid (ParkourHurdleTerrain /
        # ParkourGapTerrain carry a `flat` flag; other generators default to False).
        generators = terrain._options.terrain_generators
        is_flat_cell = torch.tensor(
            [[bool(getattr(cell, "flat", False)) for cell in row] for row in generators],
            dtype=torch.bool,
            device=self.device,
        )
        if is_flat_cell.shape != (self._rows, self._cols):
            raise ValueError(
                f"{term_subject(self)}: terrain generator grid shape {tuple(is_flat_cell.shape)} does not match "
                f"the goal grid shape {(self._rows, self._cols)}."
            )
        self._cell_is_flat = is_flat_cell

        num_envs, device = self.num_envs, self.device
        # IsaacLab-style column distribution: env i walks column floor(i / (N / cols)).
        self.terrain_types = (
            torch.div(torch.arange(num_envs, device=device), num_envs / self._cols, rounding_mode="floor")
            .long()
            .clamp(max=self._cols - 1)
        )
        # Placeholder levels until the first spawn, where the sampler's cells are applied. TerrainLevelCurriculum
        # keeps the columns dealt above for the whole run; a whole-cell sampler (uniform, SGS) replaces both.
        self.terrain_levels = torch.zeros(num_envs, dtype=torch.long, device=device)

        num_goals = self._cell_goals.shape[2]
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        self.spawn_pos = torch.zeros(num_envs, 2, device=device)
        self.env_goals = torch.zeros(num_envs, num_goals, 3, device=device)
        self.env_goals_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_is_flat = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cur_goal_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reach_goal_timer = torch.zeros(num_envs, device=device)
        self.target_pos_rel = torch.zeros(num_envs, 2, device=device)
        self.target_yaw = torch.zeros(num_envs, device=device)
        self.next_target_yaw = torch.zeros(num_envs, device=device)
        self.vel_command = torch.zeros(num_envs, 3, device=device)
        self.heading_target = torch.zeros(num_envs, device=device)
        self.dis_to_start = torch.zeros(num_envs, device=device)
        self._terrain_resample_stamp = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._spawn_offset = torch.tensor([self.spawn_offset_x, 0.0], device=device)
        # Goal-reach dwell threshold in control steps.
        self._reach_delay_steps = self.reach_goal_delay / self._env.dt

        self.stats = {
            "terrain_level": torch.zeros(num_envs, device=device),
            "current_goal_idx": torch.zeros(num_envs, device=device),
            "distance_walked": torch.zeros(num_envs, device=device),
        }

        # Fill the per-env cell buffers so pre-reset consumers (obs warm-compute) see valid data.
        self._refresh_env_cells(slice(None))
        self._update_goal_targets()

    # ------------------------------------------------------------- properties

    @property
    def command(self) -> torch.Tensor:
        """The (vx, vy, wz) velocity command, shape (num_envs, 3)."""
        return self.vel_command

    @property
    def num_levels(self) -> int:
        """Number of difficulty rows in the terrain grid."""
        return self._rows

    @property
    def num_types(self) -> int:
        """Number of course columns in the terrain grid."""
        return self._cols

    @property
    def num_cells(self) -> int:
        """Number of cells in the terrain grid: the size of the reset-configuration set."""
        return self._rows * self._cols

    def cell_index(self, levels: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """Number grid cells row-major, ``level * num_types + type``; :meth:`cell_level` / :meth:`cell_type` decode it."""
        return levels * self._cols + types

    def cell_level(self, cells: torch.Tensor) -> torch.Tensor:
        """The difficulty row of each cell numbered by :meth:`cell_index`."""
        return cells // self._cols

    def cell_type(self, cells: torch.Tensor) -> torch.Tensor:
        """The course column of each cell numbered by :meth:`cell_index`."""
        return cells % self._cols

    @property
    def goal_progress(self) -> torch.Tensor:
        """``(num_envs,)``: the fraction of its course's goals each env has reached, in [0, 1].

        1 is the end of the course: the ``all_goals_reached`` termination and the ``parkour_goal_progress``
        success metric both read it here, so the two cannot disagree. A course with no goals is complete. Clamped
        because, with that termination off, an env keeps advancing through the padded slots past its last goal.

        Computed on every read (twice a step: the metric, then the termination), not cached: a few elementwise
        kernels, against a buffer that would have to be refreshed wherever ``cur_goal_idx`` changes (the goal
        advance, a reset, a direct write) to stay right.
        """
        count = self.env_goals_count
        progress = self.cur_goal_idx.float() / count.clamp(min=1).float()
        return torch.where(count > 0, progress, torch.ones_like(progress)).clamp(max=1.0)

    # ------------------------------------------------------------ cell buffers

    def _refresh_env_cells(self, envs_idx: slice | torch.Tensor) -> None:
        """Re-gather the per-env cell buffers from ``terrain_levels`` / ``terrain_types``."""
        levels = self.terrain_levels[envs_idx]
        types = self.terrain_types[envs_idx]
        origins = self._cell_origins[levels, types]
        self.env_origins[envs_idx] = origins
        self.spawn_pos[envs_idx] = origins[:, :2] - self._spawn_offset
        self.env_goals[envs_idx] = self._cell_goals[levels, types]
        self.env_goals_count[envs_idx] = self._cell_goals_count[levels, types]
        self.env_is_flat[envs_idx] = self._cell_is_flat[levels, types]

    def _gather_goal(self, goal_idx: torch.Tensor) -> torch.Tensor:
        """Gather each env's goal at ``goal_idx``; (num_envs,) long -> (num_envs, 3)."""
        return torch.gather(self.env_goals, 1, goal_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    def _update_goal_targets(self) -> None:
        """Recompute ``target_pos_rel`` / ``target_yaw`` / ``next_target_yaw`` from the current root pose.

        NOTE (here and in the other refresh methods): buffers are updated with ``copy_``, never
        rebound — ``env.step`` runs under ``torch.inference_mode()``, and a buffer rebound to a
        step-computed (inference) tensor could no longer be scatter-written by ``reset()``.
        """
        root_xy = self.resolve_entity(self.entity_name).get_pos()[:, :2]
        last_idx = self.env_goals.shape[1] - 1
        cur_goal = self._gather_goal(self.cur_goal_idx)
        next_goal = self._gather_goal((self.cur_goal_idx + 1).clamp(max=last_idx))
        self.target_pos_rel.copy_(cur_goal[:, :2] - root_xy)  # WORLD frame
        self.target_yaw.copy_(torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0]))
        next_rel = next_goal[:, :2] - root_xy
        self.next_target_yaw.copy_(torch.atan2(next_rel[:, 1], next_rel[:, 0]))

    # -------------------------------------------------------------- step hooks

    def post_physics_refresh(self) -> None:
        """Advance goal progress on the post-physics root pose, then refresh the goal targets."""
        root_xy = self.resolve_entity(self.entity_name).get_pos()[:, :2]
        cur_goal = self._gather_goal(self.cur_goal_idx)
        reached = torch.norm(root_xy - cur_goal[:, :2], dim=1) < self.next_goal_threshold
        timer = torch.where(reached, self.reach_goal_timer + 1.0, torch.zeros_like(self.reach_goal_timer))
        advance = timer > self._reach_delay_steps
        last_idx = self.env_goals.shape[1] - 1
        self.cur_goal_idx.copy_(torch.where(advance, (self.cur_goal_idx + 1).clamp(max=last_idx), self.cur_goal_idx))
        self.reach_goal_timer.copy_(torch.where(advance, torch.zeros_like(timer), timer))

        self._update_goal_targets()
        self.dis_to_start.copy_(torch.norm(root_xy - self.spawn_pos, dim=1))

        self.stats["terrain_level"].copy_(self.terrain_levels.float())
        self.stats["current_goal_idx"].copy_(self.cur_goal_idx.float())
        self.stats["distance_walked"].copy_(self.dis_to_start)

    def _update_command(self) -> None:
        yaw = self.resolve_entity(self.entity_name).get_heading()
        wz = torch.clip(wrap_to_pi(self.heading_target - yaw) * self.heading_control_stiffness, -1.0, 1.0)
        wz = wz * (wz.abs() > self.ang_vel_clip)
        self.vel_command[:, 2] = wz

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        # Full-batch draws kept where `envs_idx` is set, which needs no GPU->CPU sync (see
        # CommandTerm._resample_command).
        r = torch.empty(self.num_envs, device=self.device)
        vx = r.uniform_(*self.lin_vel_x_range)
        vx = vx * (vx.abs() > self.lin_vel_clip)  # a fresh tensor: `r` is redrawn below
        self.vel_command[:, 0] = torch.where(envs_idx, vx, self.vel_command[:, 0])
        self.vel_command[envs_idx, 1] = 0.0
        self.heading_target[:] = torch.where(envs_idx, r.uniform_(*self.heading_range), self.heading_target)

    # ------------------------------------------------------- terrain placement

    def assign_terrain_cells(self, envs_idx: slice | torch.Tensor, cells: torch.Tensor) -> None:
        """Move the selected envs to their cells and re-gather the cell buffers.

        Called by the spawn reset event (``ResetParkourRootState``) BEFORE it writes the root pose —
        Eden runs reset events before ``command_manager.reset``. ``cells`` is the sampler's full
        ``(num_envs,)`` assignment, numbered by :meth:`cell_index`; only the ``envs_idx`` rows are
        applied. ``envs_idx`` must already be sanitized.
        """
        cells = cells[envs_idx]
        self.terrain_levels[envs_idx] = self.cell_level(cells)
        self.terrain_types[envs_idx] = self.cell_type(cells)
        self._refresh_env_cells(envs_idx)
        self.cur_goal_idx[envs_idx] = 0
        self.reach_goal_timer[envs_idx] = 0.0
        self._terrain_resample_stamp[envs_idx] = self._env.common_step_counter

    # ------------------------------------------------------------------- reset

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, float]:
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs)
        stale = self._terrain_resample_stamp[envs_idx] != self._env.common_step_counter
        if bool(stale.any()):
            raise RuntimeError(
                "the reset_parkour_root_state event must run before ParkourCommand.reset — add "
                "ResetParkourRootState (mode='reset') to event_options so terrain resampling and the "
                "respawn precede the command reset."
            )
        # The spawn event already wrote the fresh root pose; recompute the goal targets against it.
        self._update_goal_targets()
        self.dis_to_start[envs_idx] = 0.0
        return super().reset(envs_idx)
