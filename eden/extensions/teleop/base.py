"""Base classes for teleoperation operator stacks.

The interfaces a teleop stack implements, so future stacks (a different robot, a
different tracker, a different wire protocol) share one shape:

* :class:`TeleopOperator` — the operator-station lifecycle (setup -> loop -> teardown).
* :class:`TeleopInputSource` — a tracked-input device reader.
* :class:`TeleopStreamer` — a per-tick pipeline that packs and publishes one message.

:class:`TeleopOperator` is the configurable extension object, so it follows the
Eden convention shared by every other extension base — ``ConfigurableMixin``
bound to a declarative :class:`~eden.options.extensions.teleop.TeleopOptions`
(the same shape as ``DeploymentBase``): its config is mirrored onto the instance
from the options, and a concrete stack adds its device-specific parameters as
class attributes.

:class:`TeleopInputSource` and :class:`TeleopStreamer` are plain :class:`abc.ABC`
contracts instead, because they are *dependency-injected* internal components —
constructed with live callables and sockets (``send``, a socket reader), not from
a serializable options schema — so ``ConfigurableMixin`` (which exists to bind an
``Options`` instance) would not fit them. This mirrors ``sysid``'s split between
``SystemIdentificationBase`` (``ConfigurableMixin``) and its plain-ABC
``Excitation`` interface. The SONIC stack under ``teleop/sonic`` is the reference
implementation of all three.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from eden.options.extensions.teleop import TeleopOptions
from eden.utils.common import ConfigurableMixin


class TeleopInputSource(ABC):
    """A tracked-input device reader (headset / controllers / trackers).

    A concrete source owns a background thread or socket and hands the operator
    loop the latest sample without blocking. ``get_latest`` returns ``None``
    until the first frame arrives; ``disconnected`` reports whether the stream
    has since gone stale.
    """

    @abstractmethod
    def start(self) -> None:
        """Begin reading (spawn the thread / open the socket)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop reading and release the device / socket."""

    @abstractmethod
    def get_latest(self):
        """The most recent tracked-body sample, or ``None`` before the first."""

    @abstractmethod
    def get_timestamp_ns(self) -> int:
        """The device's latest timestamp in nanoseconds (a liveness / pacing signal)."""

    @abstractmethod
    def controllers(self):
        """The current controller state (buttons / sticks / triggers)."""

    @property
    @abstractmethod
    def disconnected(self) -> bool:
        """True once the input has gone stale (no fresh frames within the timeout)."""


class TeleopStreamer(ABC):
    """A per-tick operator pipeline that packs and publishes one wire message.

    A streamer owns one control mode. ``run_once`` consumes the latest input and
    calls the injected ``send`` at most once per tick, self-pacing to the mode's
    rate; its exact arguments are mode-specific, hence the loose signature.
    ``reset_yaw`` re-zeros the accumulated heading on (re)entry and
    ``on_mode_exit`` runs when the mode is left.
    """

    @abstractmethod
    def run_once(self, *args, **kwargs) -> None:
        """Advance one tick: read the latest input and maybe publish one message."""

    @abstractmethod
    def reset_yaw(self) -> None:
        """Reset the accumulated heading (called on mode entry)."""

    def on_mode_exit(self) -> None:
        """Hook run when this mode is left. Default: no-op."""


class TeleopOperator(ConfigurableMixin[TeleopOptions], ABC):
    """The operator-station lifecycle: build the stack, run the loop, tear down.

    Configured by a :class:`~eden.options.extensions.teleop.TeleopOptions` (the
    fields below are mirrored onto the instance by ``ConfigurableMixin``); a
    concrete stack adds its device-specific parameters as further class
    attributes. Construct it from an options instance built with ``configure()``.

    ``run`` is the template method — ``setup`` once, then ``step`` until
    ``should_stop`` is true, then ``teardown`` in a ``finally`` (so it runs even
    on ``KeyboardInterrupt``, which routes through ``on_interrupt``). ``setup``
    runs inside the same ``try`` — it can block for a long time (headset wait) —
    so ``teardown`` must cope with a partially built stack. The loop cadence is
    set by the streamers' own pacing inside ``step``, not by this loop.
    """

    #: Host running the robot process this operator drives (its feedback channel).
    robot_host: str = "localhost"
    #: Port the operator binds to publish its commands / pose / planner stream.
    port: int = 5556
    #: Port the robot process publishes its debug / feedback state on.
    feedback_port: int = 5557

    def run(self) -> None:
        """Set up the stack, drive the control loop, and tear down on exit."""
        try:
            self.setup()
            while not self.should_stop():
                self.step()
        except KeyboardInterrupt:
            self.on_interrupt()
        finally:
            self.teardown()

    @abstractmethod
    def setup(self) -> None:
        """Construct readers / streamers / sockets and block until inputs are live."""

    @abstractmethod
    def step(self) -> None:
        """One control tick: poll input, advance state, stream, and publish."""

    @abstractmethod
    def teardown(self) -> None:
        """Release every resource (publish socket first, then slow scenes / windows).

        Also runs after a ``setup`` that raised or was interrupted part-way, so it
        releases only what was actually built.
        """

    def should_stop(self) -> bool:
        """Whether ``run`` should leave the loop. Default: never (Ctrl-C exits)."""
        return False

    def on_interrupt(self) -> None:
        """Hook for a ``KeyboardInterrupt`` raised inside the loop. Default: no-op."""
