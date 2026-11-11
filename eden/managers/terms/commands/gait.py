"""Gait phase command term for bipedal locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class GaitPhaseCommand(CommandTerm):
    """Command generator that maintains a cyclic gait phase for each foot.

    Generates a per-foot phase signal that advances at a configurable frequency.
    The left foot starts at phase 0 and the right foot starts at ``phase_offset``
    (default ``pi`` for an alternating gait).

    The ``command`` property returns the raw phase angles (num_envs, num_feet).
    Use the ``sin_phase`` / ``cos_phase`` observation terms to feed these into the
    policy.

    Parameters
    ----------
    gait_frequency: float
        Gait cycle frequency in Hz (cycles per second). Default 1.25 Hz.
    phase_offset: float
        Phase offset between left and right foot in radians. Default pi (alternating).
    num_feet: int
        Number of feet. Default 2.
    velocity_command_name: str | None
        Optional name of a velocity-style command term (e.g. ``UniformVelocityCommand``)
        whose ``command`` tensor has shape ``(num_envs, ≥3)`` interpreted as
        ``[lin_x, lin_y, ang_z]`` in the base frame. When set, the gait phase is
        pinned to ``stand_phase_value`` for environments whose commanded
        velocities are below ``stand_velocity_threshold`` — so downstream
        alternating-gait rewards stop enforcing a step cycle while the policy is
        commanded to stand. ``None`` (default) keeps the phase always
        cycling.
    stand_velocity_threshold: float
        Velocity-magnitude threshold (m/s for linear, rad/s for angular) below
        which a command is treated as "stand". Default 0.01.
    stand_phase_value: float
        Phase value (radians) assigned to *all* feet for stand-mode envs.
        Setting both feet to the same phase means the gait reward stops
        penalizing alternation while standing. Default ``pi`` matches the
        common contact-schedule formulation where ``cos(phase) > 0`` is stance.
    """

    gait_frequency: float = 1.25
    phase_offset: float = np.pi
    num_feet: int = 2
    velocity_command_name: str | None = None
    stand_velocity_threshold: float = 0.01
    stand_phase_value: float = float(np.pi)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        self._phase = torch.zeros(self.num_envs, self.num_feet, device=self.device)
        # Initialize with offsets (left=0, right=phase_offset, ...)
        for i in range(self.num_feet):
            self._phase[:, i] = i * self.phase_offset
        self._stand_shape_validated = False

    @property
    def command(self) -> torch.Tensor:
        """The current gait phase per foot. Shape (num_envs, num_feet)."""
        return self._phase

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        # Reset phase on episode reset
        for i in range(self.num_feet):
            self._phase[envs_idx, i] = i * self.phase_offset

    def _apply_stand_override(self) -> None:
        """If a velocity command is configured, pin the phase for stand-mode envs."""
        cmd = self._env.command_manager.get_command(self.velocity_command_name)
        if not self._stand_shape_validated:
            if cmd.ndim != 2 or cmd.shape[0] != self.num_envs or cmd.shape[1] < 3:
                raise ValueError(
                    f"GaitPhaseCommand: velocity_command_name={self.velocity_command_name!r} "
                    f"resolved to a command of shape {tuple(cmd.shape)}; expected (num_envs, >=3) "
                    "with columns [lin_x, lin_y, ang_z]."
                )
            self._stand_shape_validated = True
        lin_below = torch.linalg.norm(cmd[:, :2], dim=-1) < self.stand_velocity_threshold
        ang_below = cmd[:, 2].abs() < self.stand_velocity_threshold
        stand_mask = lin_below & ang_below
        # NOTE: assign unconditionally — a guarded ``stand_mask.any()`` would
        # force a blocking GPU→CPU sync every step. A scalar written through the
        # mask is a masked fill: sync-free, and a no-op for an all-False mask.
        self._phase[stand_mask, :] = self.stand_phase_value

    def _update_command(self) -> None:
        dt = self._env.dt
        phase_increment = 2.0 * np.pi * self.gait_frequency * dt
        self._phase.add_(phase_increment)
        # Wrap to [-pi, pi] in-place
        self._phase.add_(np.pi).remainder_(2.0 * np.pi).sub_(np.pi)
        if self.velocity_command_name is not None:
            self._apply_stand_override()
