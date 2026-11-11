"""Fixed-rate loop limiter for deployment control loops.

A control loop that misses its period is invisible from the robot's side: the policy still
gets fresh state and the motors still get targets, only later than the policy was trained to
expect. On a 50 Hz loop a step that takes 30 ms plays the motion reference at 0.67x, stretches
every history observation, and hands each PD target to the robot 10 ms late. Nothing in the
packet stream shows it, so :class:`RateLimiter` counts its own overruns and says so while the
loop runs, rather than leaving the achieved rate to be recovered from a ``--debug_record``
afterwards.
"""

from __future__ import annotations

import time

import eden as en


class RateLimiter:
    """Pace a loop at ``frequency_hz`` and report when it cannot keep up.

    Call :meth:`sleep` once per iteration. The first call anchors the schedule and returns at
    once -- the limiter is typically constructed long before the loop starts (the deployment
    builds it in ``__init__``, then connects and waits for the operator), so the time before the
    first tick is not the loop's. From then on, an iteration that finished early sleeps out the
    remainder of the period; one that finished late (an *overrun*) returns at once and re-anchors
    the schedule to now instead of bursting to catch up.

    Parameters
    ----------
    frequency_hz : float
        Target loop rate.
    report_every_s : float
        Seconds between health reports. Within each window the overruns are counted, and if
        there were any a single warning names how many ticks were late, the worst lateness, and
        the rate actually achieved. ``0`` disables the periodic report (the counters and
        :meth:`summary` still work).
    name : str
        What the loop is, for the log lines (the deployment passes ``"[deploy] control loop"``
        so its lines carry the same prefix as the rest of the deploy output).
    """

    def __init__(self, frequency_hz: float, *, report_every_s: float = 1.0, name: str = "control loop") -> None:
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be greater than zero.")
        if report_every_s < 0.0:
            raise ValueError("report_every_s must be >= 0 (0 disables the periodic report).")
        self.frequency_hz = float(frequency_hz)
        self.name = name
        self._period_s = 1.0 / self.frequency_hz
        self._report_every_s = float(report_every_s)
        # ``None`` until the first tick anchors the schedule.
        self._next_time: float | None = None
        # Lifetime counters.
        self.ticks = 0
        self.overruns = 0
        self.worst_late_s = 0.0
        self._t0 = 0.0
        # Current report window.
        self._win_t0 = 0.0
        self._win_ticks = 0
        self._win_overruns = 0
        self._win_worst_late_s = 0.0
        self._next_report = 0.0

    def sleep(self) -> None:
        now = time.perf_counter()
        self.ticks += 1
        if self._next_time is None:
            # First tick: anchor the schedule and both clocks here.
            self._next_time = now + self._period_s
            self._t0 = now
            self._win_t0 = now
            self._next_report = now + self._report_every_s
            return
        self._win_ticks += 1
        late = now - self._next_time
        if late >= 0.0:
            # Overran: the iteration took longer than the period. Re-anchor rather than
            # sleeping zero for the next N ticks to catch up, which would burst commands.
            self.overruns += 1
            self._win_overruns += 1
            self.worst_late_s = max(self.worst_late_s, late)
            self._win_worst_late_s = max(self._win_worst_late_s, late)
            self._next_time = now + self._period_s
        else:
            time.sleep(-late)
            self._next_time += self._period_s
        if self._report_every_s > 0.0 and now >= self._next_report:
            self._report(now)

    def _report(self, now: float) -> None:
        window_s = now - self._win_t0
        if self._win_overruns:
            achieved = self._win_ticks / window_s if window_s > 0.0 else 0.0
            en.logger.warning(
                f"{self.name} overran {self._win_overruns}/{self._win_ticks} ticks in the last "
                f"{window_s:.1f} s (worst {self._win_worst_late_s * 1e3:.1f} ms late; {achieved:.1f} Hz achieved "
                f"against {self.frequency_hz:g} Hz). One iteration -- policy step plus state I/O -- is taking "
                f"longer than the {self._period_s * 1e3:.0f} ms period."
            )
        self._win_t0 = now
        self._win_ticks = 0
        self._win_overruns = 0
        self._win_worst_late_s = 0.0
        self._next_report = now + self._report_every_s

    @property
    def overrun_fraction(self) -> float:
        """Fraction of the paced ticks (every tick after the anchoring first one) that overran."""
        paced = self.ticks - 1
        return self.overruns / paced if paced > 0 else 0.0

    def summary(self) -> str:
        """One line for the end of a run: ticks, overruns, worst lateness and the achieved rate."""
        paced = self.ticks - 1
        elapsed = time.perf_counter() - self._t0 if paced > 0 else 0.0
        achieved = paced / elapsed if elapsed > 0.0 else 0.0
        return (
            f"{self.name}: {self.ticks} ticks, {self.overruns} overran ({self.overrun_fraction * 100:.0f}%), "
            f"worst {self.worst_late_s * 1e3:.1f} ms late, {achieved:.1f} Hz achieved against {self.frequency_hz:g} Hz"
        )
