"""Event terms for the Extreme Parkour tasks.

``ResetParkourRootState`` must be the FIRST reset-mode event in
``event_options``: it drives the reset-time terrain placement
(its reset-configuration sampler, by default ``TerrainLevelCurriculum``, →
:meth:`ParkourCommand.assign_terrain_cells`) and respawns the robot at the cell's
spawn point, and ``ParkourCommand.reset`` verifies (via a step stamp) that this
happened before the command reset ran.

``RandomizeCameraMountPitch`` is the student task's camera-extrinsics domain
randomization: a per-env pitch offset baked into the depth camera's attach
transform at startup.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import trans_quat_to_T

from eden.constants import EventMode
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.managers.resets.sampling import ResetConfigSampler, resolve_reset_config_sampler
from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand
from eden.tasks.extreme_parkour.custom_terms.curricula import TerrainLevelCurriculum
from eden.utils.geom import quat_from_euler_xyz, quat_mul
from eden.utils.misc import is_env_mask, sanitize_envs_idx
from eden.utils.sample import sample_uniform
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase

_STARTUP, _RESET, _INTERVAL = EventMode


def _get_parkour_command(env: EnvBase, command_name: str, requester: str) -> ParkourCommand:
    manager = getattr(env, "command_manager", None)
    if manager is None:
        raise ValueError(f"{requester} requires a command manager with a '{command_name}' term.")
    try:
        command = manager.get_term(command_name)
    except KeyError:
        raise ValueError(
            f"{requester}: command term '{command_name}' not found. Available: {manager.active_terms}"
        ) from None
    if not isinstance(command, ParkourCommand):
        raise TypeError(
            f"{requester}: command term '{command_name}' is a {type(command).__name__}, expected ParkourCommand."
        )
    return command


@EVENT_TERM_REGISTRY.register()
class ResetParkourRootState(EventTerm):
    """Respawn the robot at its terrain cell's spawn point, facing +x (the course direction).

    Orchestrates the reset-time terrain placement, then writes the root pose: xy = the cell's
    spawn point (cell center minus ``spawn_offset_x`` along x), z = the robot's default root
    height, identity orientation. Base velocities need no explicit zeroing — ``scene.reset``
    (which runs before reset-mode events) already restored the zero-velocity initial state; only
    the pose is overwritten here.

    ``sampler`` names the curriculum term that decides each resetting env's cell, over a
    configuration set of every cell of the grid (numbered by :meth:`ParkourCommand.cell_index`):

    - ``"terrain_level"``, a :class:`TerrainLevelCurriculum` (the shipped task): it moves the env up
      or down a difficulty row from its walked distance; the env keeps the course column
      ``ParkourCommand`` dealt it at build.
    - any other :class:`~eden.managers.resets.sampling.ResetConfigSampler` term, e.g.
      ``SuccessGuidedSampling``: it chooses the course as well as the difficulty. A fixed column per
      env would split the set into per-column pools the sampler could not move experience between,
      and the SGS paper samples terrain type and level jointly.
    - ``None``: every cell uniformly (the paper's "uniform" baseline).

    ``TerrainLevelCurriculum`` names its command itself (its own ``command_name``), as every
    Eden term names the terms it reads: it resolves the command in its ``build()``, before any
    event exists to hand it over. So the two must name the same command term, and ``build()``
    checks that they do. Equal cell counts would otherwise let the curriculum bind and then
    follow another command's walked distance.

    The event is the orchestrator because it is the only reset-window hook that runs
    before the pose write; the event manager is built after both the command and
    curriculum managers, so both terms are resolvable (fail-fast) at build time.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    command_name: str = "parkour_command"
    sampler: str | None = "terrain_level"

    def build(self) -> None:
        from eden.entities.rigid import RigidEntity

        entity = self._env.entities[self.entity_name]
        if not isinstance(entity, RigidEntity):
            raise TypeError(f"{term_subject(self)} requires a RigidEntity, got {type(entity).__name__}.")
        self._command = _get_parkour_command(self._env, self.command_name, type(self).__name__)
        # The grid shape in the label puts it in every sampler error, e.g. a checkpoint saved for another grid.
        grid = f"{self._command.num_levels} x {self._command.num_types} terrain grid"
        consumer = f"ResetParkourRootState(entity_name='{self.entity_name}') over the {grid}"
        self._sampler: ResetConfigSampler = resolve_reset_config_sampler(self._env, self.sampler, consumer=consumer)
        # A check, not a dispatch: see the class docstring for why both terms name the command.
        if isinstance(self._sampler, TerrainLevelCurriculum) and self._sampler.command is not self._command:
            raise ValueError(
                f"{consumer}: sampler='{self.sampler}' is a TerrainLevelCurriculum on command term "
                f"'{self._sampler.command_name}', but this event places envs through command term "
                f"'{self.command_name}', so the levels would follow another command's walked distance. "
                f"Fix: TerrainLevelCurriculum.configure(command_name='{self.command_name}')."
            )
        self._sampler.bind(self._command.num_cells, consumer=consumer)
        self._identity_quat = torch.zeros(self._env.num_envs, 4, device=self.device)
        self._identity_quat[:, 0] = 1.0  # wxyz

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        # 1. Cell decision + placement BEFORE the pose write (spawn_pos may change).
        self._command.assign_terrain_cells(envs_idx, self._sampler.sample(envs_idx))

        entity: RigidEntity = self.resolve_entity(self.entity_name)
        default_pos = entity.default_root_pos
        if default_pos.dim() == 1:
            default_pos = default_pos.unsqueeze(0)
        new_pos = torch.empty(self._env.num_envs, 3, device=self.device)
        new_pos[:, :2] = self._command.spawn_pos
        new_pos[:, 2] = default_pos.expand(self._env.num_envs, -1)[:, 2]

        # set_pos / set_quat bool-mask shape contracts differ:
        # set_pos wants the full buffer for a bool mask; set_quat always wants the subset.
        is_bool_mask = is_env_mask(envs_idx)
        new_pos_arg = new_pos if is_bool_mask else new_pos[envs_idx]
        entity.set_pos(new_pos_arg, envs_idx=envs_idx)
        entity.set_quat(self._identity_quat[envs_idx], envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeCameraMountPitch(EventTerm):
    """Per-env pitch offset on an attached camera's mount (startup camera-extrinsics DR).

    Samples one pitch delta per env from ``pitch_range_deg`` (degrees, positive pitches the
    view DOWN — the source's sign convention) and rebuilds the camera's attach transform as
    ``T(offset, Ry(delta) ∘ q_nominal)``, batched ``(num_envs, 4, 4)``. ``Ry`` acts in the
    mount (base-link) frame; the nominal mount has a level horizon, so this equals pitching
    about the camera's own right axis. The nominal ``attach_offset`` / ``attach_quat`` come
    from the camera's own options — this term never duplicates the mount calibration.
    """

    mode: EventMode = _STARTUP
    camera_name: str = "depth"
    pitch_range_deg: tuple[float, float] = (-5.0, 5.0)

    def build(self) -> None:
        camera = self._env.cameras.get(self.camera_name)
        if camera is None:
            raise ValueError(
                f"RandomizeCameraMountPitch: camera '{self.camera_name}' not found. "
                f"Available cameras: {sorted(self._env.cameras)}"
            )
        options = camera._options
        if options.attach_entity_name == "" or options.attach_offset is None or options.attach_quat is None:
            raise ValueError(
                f"RandomizeCameraMountPitch: camera '{self.camera_name}' must be attached with an "
                "explicit attach_offset/attach_quat."
            )
        self._camera = camera
        num_envs, device = self.num_envs, self.device
        self._nominal_pos = torch.tensor(options.attach_offset, device=device).repeat(num_envs, 1)
        self._nominal_quat = torch.tensor(options.attach_quat, device=device).expand(num_envs, 4)
        self._pitch_range_rad = tuple(math.radians(v) for v in self.pitch_range_deg)
        # Persistent per-env pitch deltas so a partial-env invocation keeps the others' mounts.
        self._pitch_delta = torch.zeros(num_envs, device=device)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        envs_idx = sanitize_envs_idx(envs_idx, self.num_envs)
        deltas = sample_uniform(*self._pitch_range_rad, (self.num_envs,), device=self.device)
        self._pitch_delta[envs_idx] = deltas[envs_idx]
        zeros = torch.zeros_like(self._pitch_delta)
        pitch_quat = quat_from_euler_xyz(zeros, self._pitch_delta, zeros)
        mount_quat = quat_mul(pitch_quat, self._nominal_quat)
        self._camera.set_attach_transform(trans_quat_to_T(self._nominal_pos, mount_quat))
