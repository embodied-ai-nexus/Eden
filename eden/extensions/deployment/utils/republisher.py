"""High-rate re-publishing of a deployment's held wire command.

Several robots evaluate the PD law of their **parallel-linkage** joints once per
*received command packet*, on that packet's state, rather than continuously in the motor
driver like their serial joints. On such a robot the packet rate is the damping rate: at a
50 Hz control tick the derivative term acts on a 20 ms-old velocity, which at 25 Hz is
inverted -- anti-damping -- and the linkage rings at the packet Nyquist frequency once
anything kicks it past stiction.

The fix is not a faster policy. It is to keep re-sending the *same* command between
control ticks, so the robot re-closes those joints against fresh state:
:class:`CommandRepublisherMixin` runs that loop on a background thread. Every vendor stack
that ships with such a robot does the same -- LimX's four controllers publish at 1000 Hz,
Unitree's own G1 low-level example at 500 Hz, and NVIDIA's SONIC G1 deploy runs a
dedicated 500 Hz command-writer thread beside its 50 Hz control thread.

Confirmed on LimX Oli hardware: with one packet per tick the parallel joints'
torque fits ``kp*(q_des-q) - kd*dq`` using the **previous** packet's velocity at R² 0.92-0.99
while the serial joints fit the same-instant velocity; at 1 kHz the audible waist buzz is
gone and >10 Hz torque RMS drops 23.7 -> 2.1 N·m.

**A republisher that stops working is invisible in the packet stream.** The packets it sends
are byte-identical to the ones it sent when healthy, so nothing downstream can tell a loop
that died, a loop starved of CPU, or a loop faithfully re-sending a command the control
thread stopped refreshing -- and all three silently restore the fault above. The loop
therefore reports its own health while it runs (see :meth:`CommandRepublisherMixin._check_health`)
rather than leaving it to be inferred from the end-of-run summary.

**A backend without parallel joints may use it too, for vendor-rate parity.** The Unitree Go2's
joints are all serial, so the mechanism above does not apply there; it re-sends at Unitree's
own 500 Hz low-level rate because that is what the vendor examples feed the robot, not because a
fault was seen at one packet per tick. Such a backend overrides :attr:`~CommandRepublisherMixin._PUB_DROPOUT_EFFECT`
and :attr:`~CommandRepublisherMixin._PUB_SLOW_EFFECT` so the loop's failure messages describe
its own robot rather than predicting a parallel-joint ring it cannot have.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
import time
import traceback

import eden as en


def publish_rate_configure_kwargs(value: float | None) -> dict[str, float | None]:
    """Translate an operator-supplied publish rate into ``configure()`` kwargs.

    One owner for the convention, because both ``eden deploy`` and the hardware example
    scripts have to agree on it: ``None`` means "the operator said nothing, keep the backend's
    own default", and ``0`` means "send one packet per control tick" -- the A/B against that
    default, which is the backend's ``publish_rate=None``. Conflating the two would make the
    A/B flag silently select the default it is meant to be compared against.

    Parameters
    ----------
    value : float | None
        The rate as given on the command line, or ``None`` when the flag was not passed.

    Returns
    -------
    dict[str, float | None]
        Kwargs to splat into ``configure()`` -- empty when the flag was not passed.
    """
    return {} if value is None else {"publish_rate": value or None}


class CommandRepublisherMixin(ABC):
    """Re-send the backend's held wire command at ``publish_rate`` between control ticks.

    A backend mixes this in, declares its own ``publish_rate`` option (the default rate is
    the robot's own, so it belongs with the robot), calls :meth:`_init_republisher` from
    ``__init__``, :meth:`_validate_publish_rate` from ``_resolve_options``,
    :meth:`_start_publisher` at the end of ``connect`` and :meth:`_stop_publisher` in
    ``close``, implements :meth:`_publish_held_command`, and calls
    :meth:`_note_command_published` from ``send_payload``.

    Locking: :attr:`_cmd_lock` guards the wire command. The backend's ``send_payload`` must
    hold it while it mutates and publishes, and :meth:`_publish_held_command` is always
    called with it held -- so a re-send can never observe a half-written packet, and two
    threads never write the transport concurrently.

    An :class:`~abc.ABC` rather than a plain mixin: ``@abstractmethod`` on a class built by
    ``type`` is inert (``ABCMeta`` only propagates abstracts from bases that already carry
    ``__abstractmethods__``), so a backend that forgot :meth:`_publish_held_command` would
    construct and connect fine and only fail inside the daemon thread.
    """

    #: Set by :meth:`_init_republisher`; the backend declares the ``publish_rate`` option.
    publish_rate: float | None

    #: Seconds between the loop's health checks -- the rate limit on its log lines too.
    _PUB_HEALTH_PERIOD_S: float = 1.0
    #: Fraction of ``publish_rate`` below which the loop reports itself starved.
    _PUB_STARVED_FRACTION: float = 0.5
    #: Control periods the held command may age before the loop reports the control thread gone.
    _PUB_STALE_TICKS: float = 10.0
    #: What the robot suffers once the loop dies -- the tail of the error line. Per-robot: the
    #: default is the parallel-joint fault this class exists for, which not every user has.
    _PUB_DROPOUT_EFFECT: str = "expect the parallel ankle/waist joints to ring. Stop the run"
    #: What a loop that ran below half its target rate means, in the end-of-run summary.
    _PUB_SLOW_EFFECT: str = "the parallel-joint PD ran that slowly"

    # Health bookkeeping. Declared at class level as well as bound in ``_init_republisher``,
    # so a backend built with ``__new__`` and hand-set fields (how the deployment tests
    # construct one, with no SDK or scene) still reads sane defaults.
    _pub_errors: int = 0
    _pub_last_error: str | None = None
    _pub_died: str | None = None
    _pub_degraded: bool = False
    _pub_cmd_t: float = 0.0

    def _init_republisher(self) -> None:
        """Allocate the command lock and thread bookkeeping. Call from ``__init__``."""
        # Shared between the control thread (send_payload) and the republisher thread.
        self._cmd_lock = threading.Lock()
        self._pub_thread: threading.Thread | None = None
        self._pub_stop = threading.Event()
        self._pub_count = 0
        # Stamped by the loop at its *first* packet, not at thread start: a backend may gate
        # its hook until a real command exists (the Unitree DDS backends do), and charging that dead time to
        # the achieved rate would report a rate the loop was never given the chance to hit.
        self._pub_t0 = 0.0
        self._pub_errors = 0
        self._pub_last_error = None
        self._pub_died = None
        self._pub_degraded = False
        self._pub_cmd_t = 0.0

    def _note_command_published(self) -> None:
        """Record that the control thread just refreshed the held command.

        Call from the backend's ``send_payload``, under :attr:`_cmd_lock`. It is what lets the
        loop tell "the robot is being fed a current command" from "the control thread stopped
        and I have been re-sending a frozen one at full gains" -- two states the packet stream
        itself cannot distinguish, because the packets are identical.
        """
        self._pub_cmd_t = time.perf_counter()

    @abstractmethod
    def _publish_held_command(self) -> None:
        """Write the current wire command to the transport, called with :attr:`_cmd_lock` held.

        A no-op is the right behaviour before the first command exists -- the loop starts as
        soon as the transport is up, which may be before ``send_payload`` has run once.
        """
        raise NotImplementedError

    def _validate_publish_rate(self) -> None:
        """Fail fast on a republish rate below the control rate (it would only add jitter)."""
        if self.publish_rate is None:
            return
        if self.control_freq is None:
            # ``control_freq`` is an Optional option that ``DeploymentBase`` only fills in when
            # ``control_dt > 0``. Comparing against it blind raises a bare TypeError naming
            # neither value, which is the opposite of what this check is for.
            raise ValueError(
                f"publish_rate is set ({self.publish_rate:g} Hz) but the control rate is unknown: control_freq is "
                f"None and control_dt is {self.control_dt!r}. Set one of them, or publish_rate=None to send one "
                f"packet per control tick."
            )
        if self.publish_rate <= 0.0 or self.publish_rate < self.control_freq:
            raise ValueError(
                f"publish_rate must be >= the control frequency ({self.control_freq:g} Hz) or None to send one "
                f"packet per control tick, got {self.publish_rate}"
            )

    def _start_publisher(self) -> None:
        """Start the background re-send loop (no-op when ``publish_rate`` is ``None``)."""
        if self.publish_rate is None or self._pub_thread is not None:
            return
        self._pub_stop.clear()
        self._pub_count = 0
        self._pub_t0 = 0.0  # stamped by the loop at its first packet, not here
        self._pub_errors = 0
        self._pub_last_error = None
        self._pub_died = None
        self._pub_degraded = False
        # Reset with the rest: a stamp carried over from a previous connect/close cycle would
        # read as a command minutes old and trip the staleness warning on the first check.
        self._pub_cmd_t = 0.0
        name = f"{type(self).__name__.lower()}-publisher"
        self._pub_thread = threading.Thread(target=self._publisher_loop, name=name, daemon=True)
        self._pub_thread.start()
        en.logger.info(f"[deploy] Re-publishing the wire command at {self.publish_rate:g} Hz between control ticks.")

    def _stop_publisher(self) -> None:
        """Stop and join the re-send loop, reporting the rate it actually achieved.

        The achieved rate is the diagnostic that matters: together with the control rate it is
        how often the robot re-closed its parallel joints, so a loop starved by a busy box is
        worth a warning rather than a silent regression to the behaviour this exists to avoid.
        Counts re-sends only -- the per-tick packets ``send_payload`` writes are not included,
        and the clock runs from the first re-sent packet rather than from thread start.
        """
        thread = self._pub_thread
        if thread is None:
            return
        self._pub_stop.set()
        thread.join(timeout=1.0)
        if thread.is_alive():
            # Dropping the handle anyway: the caller tears the transport down next and both
            # hooks null-check it. Say so, though -- "``_pub_thread`` is None" is supposed to
            # mean nothing is publishing, and for as long as this thread runs it does not.
            en.logger.warning("[deploy] Republisher thread did not stop within 1.0 s; it may still be publishing.")
        self._pub_thread = None

        faults: list[str] = []
        if self._pub_died is not None:
            faults.append(f"the loop died early ({self._pub_died})")
        if self._pub_errors:
            faults.append(f"{self._pub_errors} send(s) failed (last: {self._pub_last_error})")
        if self._pub_t0 == 0.0:
            # No packet ever left the loop. Either it was gated the whole time (nothing was
            # ever commanded -- normal for a connect/disconnect with no control loop between)
            # or it died before its first send, which ``faults`` already names.
            msg = "[deploy] Republisher stopped: no packets sent (no command was ever published)"
            if faults:
                en.logger.warning(f"{msg}; {'; '.join(faults)}.")
            else:
                en.logger.info(f"{msg}.")
            return
        elapsed = time.perf_counter() - self._pub_t0
        achieved = self._pub_count / elapsed if elapsed > 0 else 0.0
        msg = f"[deploy] Republisher stopped: {self._pub_count} packets in {elapsed:.1f} s = {achieved:.0f} Hz"
        if self.publish_rate is not None and achieved < self._PUB_STARVED_FRACTION * self.publish_rate:
            faults.append(f"target {self.publish_rate:g} Hz; {self._PUB_SLOW_EFFECT}")
        if faults:
            en.logger.warning(f"{msg} ({'; '.join(faults)}).")
        else:
            en.logger.info(msg)

    def _publisher_loop(self) -> None:
        """Run :meth:`_publish_until_stopped`, and make it loud if it ever ends on its own.

        Nothing downstream can see this thread stop: the robot simply goes back to one packet
        per control tick, which is the fault the class exists to prevent and which looks like a
        policy problem from every other vantage point. So the one thing this wrapper must not
        do is let the thread end quietly the way an uncaught exception in a daemon thread does.
        """
        try:
            self._publish_until_stopped()
        except BaseException as exc:  # noqa: BLE001 -- a dead republisher must be loud, not silent
            self._pub_died = f"{type(exc).__name__}: {exc}"
            en.logger.error(
                f"[deploy] Republisher thread died after {self._pub_count} packets: {self._pub_died}. "
                f"The robot is back to one packet per control tick -- {self._PUB_DROPOUT_EFFECT}.\n"
                f"{traceback.format_exc()}"
            )

    def _publish_until_stopped(self) -> None:
        """Re-send the held command at ``publish_rate`` until stopped (deadline-paced, no bursts)."""
        period = 1.0 / float(self.publish_rate)
        deadline = time.perf_counter()
        check_at = deadline + self._PUB_HEALTH_PERIOD_S
        window_t0, window_sent, window_errors = deadline, self._pub_count, 0
        while not self._pub_stop.is_set():
            with self._cmd_lock:
                try:
                    self._publish_held_command()
                except Exception as exc:  # noqa: BLE001 -- one bad write must not end the loop
                    # A transport hiccup is worth retrying at the next deadline, not worth
                    # dropping the robot's damping rate for; ``_check_health`` reports the count.
                    self._pub_errors += 1
                    self._pub_last_error = f"{type(exc).__name__}: {exc}"
                    window_errors += 1
            now = time.perf_counter()
            if self._pub_t0 == 0.0 and self._pub_count > 0:
                # First packet out. Both clocks start here, the health window included: a
                # backend may gate its hook until the first command (the Unitree DDS backends do), and a window
                # straddling that hand-over would divide real packets by a span that is mostly
                # correctly-idle time and report a loop running exactly at rate as starved.
                self._pub_t0 = now
                check_at = now + self._PUB_HEALTH_PERIOD_S
                window_t0, window_sent, window_errors = now, self._pub_count, 0
            if now >= check_at:
                self._check_health(now - window_t0, self._pub_count - window_sent, window_errors, now)
                check_at = now + self._PUB_HEALTH_PERIOD_S
                window_t0, window_sent, window_errors = now, self._pub_count, 0
            deadline += period
            delay = deadline - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                deadline = time.perf_counter()  # overran: resync instead of bursting to catch up

    def _check_health(self, window_s: float, sent: int, errors: int, now: float) -> None:
        """Report, at most once per health period, anything that has quietly undone the fix.

        Three things restore the per-packet-PD fault mid-run without changing what the packets
        look like: sends that fail, a loop starved of CPU, and a control thread that stopped
        refreshing the command while this loop dutifully re-sends a frozen one. Each is named
        here at the time it happens, and a single line says when it clears.

        Parameters
        ----------
        window_s : float
            Wall-clock seconds the window covered.
        sent : int
            Packets the loop actually put on the wire during the window.
        errors : int
            Sends that raised during the window.
        now : float
            ``time.perf_counter()`` at the end of the window.
        """
        if self._pub_t0 == 0.0:
            return  # gated before the first command -- idle by design, not degraded
        faults: list[str] = []
        if errors:
            faults.append(f"{errors} send(s) failed (last: {self._pub_last_error})")
        rate = sent / window_s if window_s > 0.0 else 0.0
        target = float(self.publish_rate)
        if rate < self._PUB_STARVED_FRACTION * target:
            faults.append(f"publishing at {rate:.0f} Hz against a {target:g} Hz target")
        # ``send_payload`` runs every control tick on every path (policy, hold and INIT alike),
        # so a held command older than a few ticks means the control thread is gone, and the
        # robot is being held at full gains on a target nothing is updating any more.
        stale_after = self._PUB_STALE_TICKS / self.control_freq if self.control_freq else float("inf")
        age = now - self._pub_cmd_t if self._pub_cmd_t > 0.0 else 0.0
        if age > stale_after:
            faults.append(f"re-sending a command {age:.1f} s old -- the control thread has stopped feeding it")
        if faults:
            self._pub_degraded = True
            en.logger.warning(f"[deploy] Republisher degraded: {'; '.join(faults)}.")
        elif self._pub_degraded:
            self._pub_degraded = False
            en.logger.info(f"[deploy] Republisher recovered: {rate:.0f} Hz against a {target:g} Hz target.")
