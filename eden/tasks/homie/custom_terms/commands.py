"""The HOMIE velocity + base-height command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class HomieCommand(CommandTerm):
    r"""A 4-D command ``[lin_vel_x, lin_vel_y, ang_vel_z, base_height]``, base frame, drawn tri-modally.

    OpenHomie's ``_resample_commands`` (``legged_robot.py``): on each resample an env draws
    ``x ~ U(0, 1)`` and lands in one of three regimes, so walking and squatting are never commanded
    together:

    - **squat** (``x < height_regime_threshold``, 1/3): height in ``default_height + height_offset_range``
      (``[0.24, 0.74]`` m by default), zero velocity;
    - **stand** (otherwise, up to ``vel_regime_threshold``, 1/6): ``default_height``, zero velocity;
    - **walk** (``x > vel_regime_threshold``, 1/2): sampled velocities at ``default_height``.

    The height is an absolute target in metres, so the posture rewards gate on ``command[:, 3] >= 0.735``.

    Parameters
    ----------
    entity_name : str
        The commanded robot.
    lin_vel_x_range, lin_vel_y_range, ang_vel_z_range : tuple[float, float]
        Walk-regime velocity ranges (m/s, m/s, rad/s).
    height_offset_range : tuple[float, float]
        Squat-regime offset added to ``default_height`` (m).
    default_height : float
        Standing / walking base-height target (m).
    height_regime_threshold, vel_regime_threshold : float
        The cut points of ``x`` between the squat / stand / walk regimes.
    """

    entity_name: str = "robot"
    lin_vel_x_range: tuple[float, float] = (-0.8, 1.2)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-0.8, 0.8)
    height_offset_range: tuple[float, float] = (-0.5, 0.0)
    default_height: float = 0.74
    height_regime_threshold: float = 1.0 / 3.0
    vel_regime_threshold: float = 0.5

    def __init__(self, env: EnvBase, options: CommandTermOptions) -> None:
        super().__init__(env, options)
        self.command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.command_b[:, 3] = self.default_height

    def build(self) -> None:
        super().build()
        self.resolve_entity(self.entity_name)

    @property
    def command(self) -> torch.Tensor:
        """``[vx, vy, wz, base_height]``, shape ``(num_envs, 4)``."""
        return self.command_b

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        n, device = self.num_envs, self.device
        regime = torch.rand(n, device=device)
        walk = (regime > self.vel_regime_threshold).float()
        squat = (regime < self.height_regime_threshold).float()
        drawn = torch.stack(
            [
                torch.empty(n, device=device).uniform_(*self.lin_vel_x_range) * walk,
                torch.empty(n, device=device).uniform_(*self.lin_vel_y_range) * walk,
                torch.empty(n, device=device).uniform_(*self.ang_vel_z_range) * walk,
                torch.empty(n, device=device).uniform_(*self.height_offset_range) * squat + self.default_height,
            ],
            dim=1,
        )
        self.command_b.copy_(torch.where(envs_idx[:, None], drawn, self.command_b))

    def _update_command(self) -> None:
        # The resample already zeroes the velocity of squatting / standing envs; no heading control.
        pass
