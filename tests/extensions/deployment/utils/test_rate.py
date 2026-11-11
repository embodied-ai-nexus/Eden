"""``RateLimiter``: pacing the deploy loop, and counting the ticks it could not keep."""

from __future__ import annotations

import pytest

from eden.extensions.deployment.utils import rate as rate_mod
from eden.extensions.deployment.utils.rate import RateLimiter


class _Clock:
    """The wall clock: ``time.perf_counter`` / ``time.sleep``, advanced by hand."""

    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(rate_mod.time, "perf_counter", c.perf_counter)
    monkeypatch.setattr(rate_mod.time, "sleep", c.sleep)
    return c


def _run(limiter: RateLimiter, clock: _Clock, busy_s: float, n: int) -> None:
    """Simulate ``n`` iterations whose work takes ``busy_s`` each."""
    for _ in range(n):
        clock.now += busy_s
        limiter.sleep()


def test_on_time_loop_sleeps_the_remainder_and_reports_nothing(clock, eden_caplog):
    lim = RateLimiter(50.0)
    _run(lim, clock, busy_s=0.005, n=100)
    assert lim.overruns == 0 and lim.ticks == 100
    # The first tick anchors the schedule (no sleep); the other 99 sleep out the 15 ms remainder.
    assert len(clock.slept) == 99
    assert all(abs(s - 0.015) < 1e-9 for s in clock.slept)
    assert "overran" not in eden_caplog.text


def test_construction_time_is_not_charged_to_the_loop(clock):
    """The deployment builds the limiter in __init__, then connects and waits for the operator."""
    lim = RateLimiter(50.0, report_every_s=0)
    clock.now += 30.0  # connect(), INIT ramp, operator hesitation
    lim.sleep()
    assert lim.ticks == 1 and lim.overruns == 0 and clock.slept == []
    _run(lim, clock, busy_s=0.005, n=3)
    assert lim.overruns == 0 and len(clock.slept) == 3


def test_late_loop_counts_overruns_and_reanchors_instead_of_bursting(clock):
    lim = RateLimiter(50.0, report_every_s=0)
    _run(lim, clock, busy_s=0.030, n=10)
    assert lim.ticks == 10
    assert lim.overruns == 9  # the anchoring first tick is not paced, so it cannot overrun
    assert lim.overrun_fraction == 1.0
    assert clock.slept == []  # never slept, never tried to "catch up"
    assert abs(lim.worst_late_s - 0.010) < 1e-9


def test_partial_overruns_are_counted_exactly(clock):
    lim = RateLimiter(50.0, report_every_s=0)
    _run(lim, clock, busy_s=0.010, n=5)
    _run(lim, clock, busy_s=0.025, n=3)
    _run(lim, clock, busy_s=0.010, n=2)
    assert lim.ticks == 10
    assert lim.overruns == 3


def test_overruns_are_reported_once_per_window_with_the_achieved_rate(clock, eden_caplog):
    lim = RateLimiter(50.0, report_every_s=1.0, name="control loop")
    _run(lim, clock, busy_s=0.030, n=40)  # 1.2 s of a loop running at 33 Hz
    warnings = [r for r in eden_caplog.records if "overran" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    msg = warnings[0].getMessage()
    assert "control loop overran" in msg
    assert "against 50 Hz" in msg
    assert "33." in msg  # achieved rate
    assert "10.0 ms late" in msg  # worst lateness in the window


def test_report_window_resets_so_a_recovered_loop_goes_quiet(clock, eden_caplog):
    lim = RateLimiter(50.0, report_every_s=1.0)

    def n_warnings() -> int:
        return sum("overran" in r.getMessage() for r in eden_caplog.records)

    _run(lim, clock, busy_s=0.030, n=40)  # 1.2 s late
    assert n_warnings() == 1
    _run(lim, clock, busy_s=0.005, n=100)  # 2 s on time
    # The window that straddled the recovery may still report its few late ticks, once.
    assert n_warnings() <= 2
    settled = n_warnings()
    _run(lim, clock, busy_s=0.005, n=100)  # another 2 s on time: fully quiet windows say nothing
    assert n_warnings() == settled


def test_summary_names_ticks_overruns_and_achieved_rate(clock):
    lim = RateLimiter(50.0, report_every_s=0, name="control loop")
    _run(lim, clock, busy_s=0.030, n=20)
    s = lim.summary()
    assert s.startswith("control loop: 20 ticks, 19 overran (100%)")
    assert "33.3 Hz achieved against 50 Hz" in s


@pytest.mark.parametrize("kwargs", [{"frequency_hz": 0.0}, {"frequency_hz": 50.0, "report_every_s": -1.0}])
def test_rejects_bad_construction_arguments(kwargs):
    with pytest.raises(ValueError):
        RateLimiter(**kwargs)
