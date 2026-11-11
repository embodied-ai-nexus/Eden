"""Terrain-level curriculum for the Extreme Parkour teacher task.

:class:`TerrainLevelCurriculum` owns the per-env difficulty level and the
promotion rule (walked distance vs. commanded distance). It lives in the
curriculum subsystem — visible in the curriculum summary table and logged via
the ``Curriculum/`` extras — but, unlike a passive curriculum term, its level
update must run *inside* the reset event window (before the spawn event writes
the root pose), which is earlier than ``CurriculumManager.reset``. It is
therefore a :class:`~eden.managers.resets.sampling.ResetConfigSampler` over the
terrain grid's cells: the spawn event
(:class:`~eden.tasks.extreme_parkour.custom_terms.events.ResetParkourRootState`)
binds it and calls ``sample()``, and the
:class:`~eden.tasks.extreme_parkour.custom_terms.commands.ParkourCommand`
applies the cells it returns.
"""

from __future__ import annotations

import torch

from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY, CurriculumTerm
from eden.managers.resets.sampling import ResetConfigSampler
from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand


@CURRICULUM_TERM_REGISTRY.register()
class TerrainLevelCurriculum(CurriculumTerm, ResetConfigSampler):
    """Walked-distance terrain-level curriculum (Extreme Parkour rule).

    On each reset of an env: promote when the distance walked from spawn exceeds
    ``level_up_frac`` of the commanded distance (``vx * episode_length_s``), demote
    below ``level_down_frac`` of it, and send envs that solved the top level to a
    uniformly random level. The very first spawn keeps the random init level
    (``U[0, max_init_terrain_level]``) — there is no walked-distance evidence yet.
    Each env keeps the course column ``ParkourCommand`` dealt it at build, so the cell it
    proposes is ``cell_index(new level, that column)``.

    Bind it only from ``ResetParkourRootState``: the cells it proposes come from the command's
    per-env columns and walked distance, so they mean nothing in another set of the same size.
    ``bind`` checks the size, and the event checks that both name the same command.

    The level an env's episode runs on is not stored here: it is the row of the sampler's
    :attr:`assignment`. Only levels no episode has started on yet are held, for an env's
    next first spawn: the build-time random ones and the ones a checkpoint load restores.

    ``compute()`` exposes ``mean_terrain_level`` through the curriculum-manager
    state (logged as ``Curriculum/<term>/mean_terrain_level`` on reset).
    ``state_dict()`` carries the per-env levels. After a load, each env's next reset keeps its
    loaded level (the episode then in flight ran on a pre-load level, so its walked distance
    says nothing about the loaded one), and the curriculum moves on from there.
    """

    command_name: str = "parkour_command"
    max_init_terrain_level: int = 2
    level_up_frac: float = 0.8
    level_down_frac: float = 0.4

    def build(self) -> None:
        # The command manager is constructed before the curriculum manager, so the
        # command term is resolvable (and fail-fast) here.
        command = self._env.command_manager.get_term(self.command_name)
        if not isinstance(command, ParkourCommand):
            raise TypeError(
                f"TerrainLevelCurriculum: command term '{self.command_name}' is a "
                f"{type(command).__name__}, expected ParkourCommand."
            )
        self._command = command
        self._num_levels = command.num_levels

        max_init = min(self.max_init_terrain_level, self._num_levels - 1)
        # The level each env's next first spawn places it on; every other reset reads the level from assignment.
        self._first_spawn_levels = torch.randint(0, max_init + 1, (self.num_envs,), device=self.device)
        self._require_bound_at_build(self._env)

    @property
    def command(self) -> ParkourCommand:
        """The command term whose walked distance and columns decide the levels."""
        return self._command

    @property
    def terrain_levels(self) -> torch.Tensor:
        """``(num_envs,)`` long: each env's level.

        The level of its running episode, or for an env whose next reset is a first spawn, the level that reset
        will place it on. Derived on every read, so it is read-only: a write into the returned tensor changes
        nothing. To set levels, ``load_state_dict`` them.
        """
        # _first_spawn_levels is read only where first_spawn holds: after the build-time spawn it goes stale
        # until a load writes it again, which also sets first_spawn for every env.
        return torch.where(self.first_spawn, self._first_spawn_levels, self._command.cell_level(self.assignment))

    def _on_bind(self, num_configs: int) -> None:
        if num_configs != self._command.num_cells:
            raise ValueError(
                f"TerrainLevelCurriculum proposes cells of the {self._num_levels} x {self._command.num_types} "
                f"terrain grid ({self._command.num_cells} cells), but {self._consumer} bound it to "
                f"{num_configs} configurations. Fix: bind it from the parkour spawn event, whose set is the grid."
            )

    def _draw(self, selected: torch.Tensor) -> torch.Tensor:
        """Advance the curriculum for the ``selected`` envs and propose every env's cell.

        Runs from the spawn event BEFORE the root pose is rewritten, so ``dis_to_start`` still
        reflects the finished episode. Computed for every env and committed under the mask, so the
        reset costs no host sync here.
        """
        command = self._command
        first_spawn = self.first_spawn
        threshold = command.command[:, 0] * self._env.max_episode_length_s
        move_up = command.dis_to_start > self.level_up_frac * threshold
        move_down = command.dis_to_start < self.level_down_frac * threshold
        delta = torch.where(first_spawn, torch.zeros_like(move_up, dtype=torch.long), move_up.long() - move_down.long())
        levels = self.terrain_levels + delta
        # Envs that solved the top level restart at a random level; never go below 0.
        random_levels = torch.randint(0, self._num_levels, levels.shape, device=self.device)
        levels = torch.where(levels >= self._num_levels, random_levels, levels).clamp(min=0)
        return command.cell_index(levels, command.terrain_types)

    def compute(self) -> dict[str, torch.Tensor]:
        return {"mean_terrain_level": self.terrain_levels.float().mean()}

    def _sampler_state(self) -> dict:
        """Each env's terrain level; the column is not carried (``ParkourCommand`` deals it at build)."""
        return {"num_levels": self._num_levels, "terrain_levels": self.terrain_levels.detach().cpu().clone()}

    def _load_sampler_state(self, state: dict) -> None:
        """Restore the levels. Under a different ``num_envs``, each env draws one of the saved levels.

        The saved levels are a distribution over difficulty, not a per-env identity (envs respawn on load),
        so a run resumed with more or fewer envs keeps the level mix it had reached. The base then makes
        every env's next reset a first spawn, which keeps its loaded level.
        """
        if state.get("num_levels") != self._num_levels:
            raise ValueError(
                f"TerrainLevelCurriculum: the checkpoint holds levels of a {state.get('num_levels')}-row terrain "
                f"grid, this env's has {self._num_levels} rows. Fix: resume with the terrain grid the checkpoint "
                "was trained on. Do NOT clamp the saved levels into range: a level indexes a difficulty row, and "
                "on another grid that row is a different course."
            )
        levels = torch.as_tensor(state.get("terrain_levels", ()), dtype=torch.long).flatten().to(self.device)
        # A host read, once per load: the checkpoint is not on the step path.
        if levels.numel() == 0 or int(levels.min()) < 0 or int(levels.max()) >= self._num_levels:
            found = f"{levels.numel()} levels" if levels.numel() == 0 else f"levels {sorted(set(levels.tolist()))}"
            raise ValueError(
                f"TerrainLevelCurriculum: the checkpoint's terrain_levels hold {found}, expected at least one "
                f"level in [0, {self._num_levels - 1}]. It was not written by TerrainLevelCurriculum.state_dict() "
                "for this grid. Fix: load the state this term saved, or leave the curriculum at its build-time "
                "levels."
            )
        if levels.numel() != self.num_envs:
            levels = levels[torch.randint(levels.numel(), (self.num_envs,), device=self.device)]
        self._first_spawn_levels.copy_(levels)
