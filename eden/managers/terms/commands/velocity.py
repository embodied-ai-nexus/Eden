"""SE(2) velocity command terms for locomotion: sampled, held, and operator-driven."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import eden as en
from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm
from eden.utils.geom import wrap_to_pi

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class UniformVelocityCommand(CommandTerm):
    r"""SE(2) velocity command sampled from a uniform distribution.

    The command comprises a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the `heading_command` attribute is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.
    """

    entity_name: str = ""
    heading_command: bool = False
    heading_control_stiffness: float = 1.0
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0
    lin_vel_x_range: tuple[float, float] = (0.0, 0.0)
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    ang_vel_z_range: tuple[float, float] = (0.0, 0.0)
    heading_range: tuple[float, float] | None = None

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        # check configuration
        if self.heading_command and self.heading_range is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but `heading_range` is None."
            )

        self.robot: Entity | None = None
        # -- command: x vel, y vel, yaw vel (the heading target is kept separately)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

    def build(self) -> None:
        self.robot = self.resolve_entity(self.entity_name)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.resampling_time_range}\n"
        msg += f"\tHeading command: {self.heading_command}\n"
        if self.heading_command:
            msg += f"\tHeading probability: {self.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def _resample_command(self, envs_idx: torch.Tensor):
        # Full-batch draws kept where `envs_idx` is set. `torch.where` rather than writing through
        # the mask, which would cost two GPU->CPU syncs a line (see CommandTerm._resample_command).
        r = torch.empty(self._env.num_envs, device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[:, 0] = torch.where(envs_idx, r.uniform_(*self.lin_vel_x_range), self.vel_command_b[:, 0])
        # -- linear velocity - y direction
        self.vel_command_b[:, 1] = torch.where(envs_idx, r.uniform_(*self.lin_vel_y_range), self.vel_command_b[:, 1])
        # -- ang vel yaw - rotation around z
        self.vel_command_b[:, 2] = torch.where(envs_idx, r.uniform_(*self.ang_vel_z_range), self.vel_command_b[:, 2])
        # heading target
        if self.heading_command:
            self.heading_target[:] = torch.where(envs_idx, r.uniform_(*self.heading_range), self.heading_target)
            is_heading = r.uniform_(0.0, 1.0) <= self.rel_heading_envs
            self.is_heading_env[:] = torch.where(envs_idx, is_heading, self.is_heading_env)
        # update standing envs
        is_standing = r.uniform_(0.0, 1.0) <= self.rel_standing_envs
        self.is_standing_env[:] = torch.where(envs_idx, is_standing, self.is_standing_env)

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.heading_command:
            # For every env, kept where `is_heading_env` is set: resolving the heading envs to
            # indices first would be a `nonzero()` GPU->CPU sync every control step, to skip work
            # the default `rel_heading_envs=1.0` needs for every env anyway.
            heading_error = wrap_to_pi(self.heading_target - self.robot.get_heading())
            heading_ang_vel = torch.clip(
                self.heading_control_stiffness * heading_error,
                min=self.ang_vel_z_range[0],
                max=self.ang_vel_z_range[1],
            )
            self.vel_command_b[:, 2] = torch.where(self.is_heading_env, heading_ang_vel, self.vel_command_b[:, 2])
        # Enforce standing (i.e., zero velocity command) for standing envs. A scalar written
        # through a bool mask is a masked fill, which needs no sync.
        self.vel_command_b[self.is_standing_env] = 0.0


@COMMAND_TERM_REGISTRY.register()
class ConstantVelocityCommand(CommandTerm):
    r"""SE(2) velocity command held at a fixed value instead of resampled.

    Same command as :class:`UniformVelocityCommand` -- ``(vx, vy, wz)`` in the robot's
    base frame -- with the random generator taken out: the value is whatever
    :attr:`command_value` says, or whatever :meth:`set_command` was last given, and it
    never changes on its own.

    This is the term a deployment substitutes for the trained one. A policy deployed
    from its training config keeps resampling the command it was *trained* against,
    which on a robot reads as the machine wandering off on its own; holding a value the
    operator chose is what a deploy actually wants. It is equally the right term for an
    evaluation that has to hold one command for a whole episode.

    The value is clamped into ``lin_vel_x_range`` / ``lin_vel_y_range`` /
    ``ang_vel_z_range``, which a deploy fills from the term it replaces, so the held
    command cannot leave the distribution the policy was trained on. A *degenerate*
    range — ``move_forward``'s ``lin_vel_x_range=(0.5, 0.5)`` — pins its axis: the policy
    only ever saw one velocity there, so that axis is not controllable and the clamp
    holds the single trained value whatever it is asked for. The clamp is logged.
    """

    entity_name: str = ""
    command_value: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)
    # Nothing to resample, but ``CommandTerm._resample`` draws from this range, so it
    # cannot be None. One resample happens on the first compute and none after.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        self.robot: Entity | None = None
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._limits_key: tuple[tuple[float, float], ...] | None = None
        self._low: torch.Tensor | None = None
        self._high: torch.Tensor | None = None

    def build(self) -> None:
        # Resolve the named entity and *keep* it, exactly as UniformVelocityCommand does:
        # consumers reach through the command term for it (``eden/managers/terms/metrics/
        # error.py`` reads ``command.robot.get_vel()``), so a substitute that only resolved
        # and discarded would not be interchangeable with the term it replaces. Resolving
        # here also means a bad name fails at setup rather than on the robot. An unnamed
        # entity is legitimate: a command term standing alone, driving no particular entity,
        # is what the task-less deploy path builds.
        if self.entity_name:
            self.robot = self.resolve_entity(self.entity_name)
        self._seed_command()

    def _limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The current clamp bounds, rebuilt only when the range fields change.

        Read *live* rather than frozen at construction, for two reasons. A curriculum that
        widens ``lin_vel_x_range`` after build reaches this term the way it reaches the
        :class:`UniformVelocityCommand` it replaces, which reads its ranges live in
        ``_resample_command``. And the clamp and the message reporting it draw from one
        source, so they cannot disagree about which range applied — a frozen bound with a
        live message prints a range the value is *inside* and then trims it anyway.
        """
        key = (tuple(self.lin_vel_x_range), tuple(self.lin_vel_y_range), tuple(self.ang_vel_z_range))
        if key != self._limits_key:
            self._limits_key = key
            self._low = torch.tensor([lo for lo, _ in key], dtype=torch.float32, device=self.device)
            self._high = torch.tensor([hi for _, hi in key], dtype=torch.float32, device=self.device)
        return self._low, self._high

    def _seed_command(self) -> None:
        """Install :attr:`command_value` as the held command.

        A separate hook so a subclass can seed differently while still going through
        :meth:`build` — overriding ``build`` itself would mean re-stating the entity
        resolution above, and a subclass that later forgot to would push a bad
        ``entity_name`` all the way to the robot with nothing to catch it.
        """
        self.set_command(self.command_value)

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    def set_command(self, command, *, warn_on_clamp: bool = True) -> torch.Tensor:
        """Hold ``(vx, vy, wz)`` from now on, clamped into the configured ranges.

        Returns the clamped command, and warns when clamping actually bit — a value
        outside the trained ranges is worth seeing rather than silently trimming.

        Parameters
        ----------
        command
            ``(vx, vy, wz)`` in the robot's base frame.
        warn_on_clamp
            Whether to log a clamp. Off for a value nothing will hold long enough to
            matter — :class:`OperatorVelocityCommand` seeds the buffer with
            ``command_value`` only so the first observation is in range, and the device
            overwrites it on the first ``compute``.
        """
        value = torch.as_tensor(command, dtype=torch.float32, device=self.device).flatten()
        if value.numel() != 3:
            raise ValueError(f"A velocity command is (vx, vy, wz); got {value.numel()} value(s): {command}.")
        low, high = self._limits()
        clamped = torch.clamp(value, min=low, max=high)
        if warn_on_clamp and not torch.equal(clamped, value):
            # Reported from ``_limits_key``, not the fields, so the range named here is
            # always the one that was actually applied.
            vx_range, vy_range, wz_range = self._limits_key
            en.logger.warning(
                f"[commands] Velocity command {value.tolist()} is outside the trained ranges "
                f"(vx {vx_range}, vy {vy_range}, wz {wz_range}); holding {clamped.tolist()} instead."
            )
        self.vel_command_b[:] = clamped
        return clamped

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        """No-op: the command is held, not sampled."""

    def _update_command(self) -> None:
        """No-op: the held command is written by :meth:`set_command`."""


@COMMAND_TERM_REGISTRY.register()
class OperatorVelocityCommand(ConstantVelocityCommand):
    r"""SE(2) velocity command driven live by a human, from a keyboard or a gamepad.

    A :class:`ConstantVelocityCommand` whose held value is re-read from an input device
    every control step. Deploy a locomotion policy with this term substituted for its
    trained :class:`UniformVelocityCommand` and the robot goes where the operator says
    instead of where the training distribution says.

    The device emits a normalized deflection; it is scaled into the same
    ``lin_vel_*_range`` / ``ang_vel_z_range`` and clamped there, so full deflection is
    the fastest command the policy was trained on and the operator cannot ask for more.
    See :mod:`eden.utils.devices.se2` for the bindings.

    For deployment and interactive evaluation, not for training: it reads a physical
    device, so a training run would sit at whatever the sticks happen to say.
    """

    device_name: str = "keyboard"
    device_options: dict = {}

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        self._device = None
        self._scale = None

    def build(self) -> None:
        from eden.utils.devices.se2 import make_se2_device, scale_se2_deflection

        super().build()
        # The device is opened in build(), not __init__: build() is the last step of term
        # construction, so one that cannot be opened (no display for pynput, no gamepad
        # plugged in) leaves the term unusable rather than half-alive.
        self._device = make_se2_device(self.device_name, self.device_options)
        # Bound once rather than imported per control step.
        self._scale = scale_se2_deflection
        en.logger.info(
            f"[commands] Operator velocity command via {self.device_name}: "
            f"vx {self.lin_vel_x_range}, vy {self.lin_vel_y_range}, wz {self.ang_vel_z_range}."
        )

    def _seed_command(self) -> None:
        """Seed silently: ``command_value`` is a placeholder, not this term's command.

        It is only what the policy observes between ``build`` and the operator's first
        touch of the device, so reporting a clamp against a range that excludes zero would
        point at a number nothing ever holds.
        """
        self.set_command(self.command_value, warn_on_clamp=False)

    def _update_command(self) -> None:
        """Re-read the device into the command buffer.

        Runs from :meth:`CommandTerm.compute`, i.e. once per control step, after
        ``_resample`` — which is a no-op here — so the operator's reading is what every
        consumer of :attr:`command` sees this step.
        """
        deflection = self._device.get_command().to(dtype=torch.float32, device=self.device)
        self.vel_command_b[:] = self._scale(deflection, *self._limits())

    def close(self) -> None:
        """Stop the input device. Idempotent."""
        stop = getattr(self._device, "stop", None)
        if stop is not None:
            stop()
