"""``RealtimePhysics``: stepping a real MuJoCo model at wall-clock rate on its own thread.

The scheduler is driven one ``_tick`` at a time against an injected clock, so the catch-up,
re-anchor and reporting rules are pinned exactly; the thread tests run on the real clock.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from eden.extensions.deployment.utils.mujoco_sim import MujocoSim  # noqa: E402
from eden.extensions.deployment.utils.realtime_physics import PDCommand, RealtimePhysics  # noqa: E402

KP, KD = 50.0, 2.0


class _Clock:
    """A manual clock: ``sleep`` advances it (plus an optional overshoot, as ``time.sleep`` wakes late)."""

    def __init__(self, overshoot: float = 0.0) -> None:
        self.now = 0.0
        self.overshoot = overshoot
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)
        self.now += dt + self.overshoot


def _hold(value: float = 0.0, width: int = 2) -> PDCommand:
    return PDCommand.hold(np.full(width, value), np.full(width, KP), np.full(width, KD))


@pytest.fixture
def sim(two_link_asset) -> MujocoSim:
    arm = MujocoSim(str(two_link_asset), ["j1", "j2"], floating_base=False)
    arm.set_implicit_damping(np.full(2, KD))
    arm.reset(np.zeros(2))
    return arm


def _loop(sim, clock=None, **kwargs) -> tuple[RealtimePhysics, _Clock]:
    clock = clock or _Clock()
    rt = RealtimePhysics(sim, _hold(), clock=clock.perf_counter, sleep=clock.sleep, **kwargs)
    rt._anchor(clock.now)  # what ``start`` does, without the thread
    return rt, clock


def _ticks(rt, clock, n: int) -> None:
    for _ in range(n):
        rt._tick(clock.now)


# --- pacing ---------------------------------------------------------------------------------


def test_an_on_time_loop_runs_one_substep_per_period(sim):
    rt, clock = _loop(sim)
    _ticks(rt, clock, 200)
    assert rt.substeps == 100
    assert sim.data.time == pytest.approx(100 * sim.sim_dt), "sim time equals wall time"
    assert clock.now == pytest.approx(100 * sim.sim_dt)
    assert rt.reanchors == 0
    assert rt.worst_lag_s == pytest.approx(sim.sim_dt)


def test_sleep_overshoot_is_caught_up_not_accumulated(sim):
    rt, clock = _loop(sim, _Clock(overshoot=0.0007))
    _ticks(rt, clock, 400)
    assert rt.substeps == int(clock.now / sim.sim_dt)
    assert rt.reanchors == 0


def test_a_late_wake_runs_the_overdue_substeps_in_one_burst(sim):
    """Waking five periods late (a GIL switch interval) runs five substeps at once, and the schedule is whole again."""
    rt, clock = _loop(sim)
    clock.now = 5 * sim.sim_dt + 0.0005
    rt._tick(clock.now)
    assert rt.substeps == 5 and sim.data.time == pytest.approx(5 * sim.sim_dt)
    assert rt.worst_lag_s == pytest.approx(5 * sim.sim_dt)
    assert rt.reanchors == 0

    woke = clock.now
    rt._tick(woke)
    assert clock.sleeps[-1] == pytest.approx(6 * sim.sim_dt - woke), "due at substep 6, not now + a period"


def test_a_backlog_past_max_lag_is_dropped_once_and_reported(sim, eden_caplog):
    rt, clock = _loop(sim, max_lag_s=0.05, report_every_s=0.01)
    clock.now = 0.2  # 100 substeps overdue

    rt._tick(clock.now)

    assert rt.substeps == 1 and rt.reanchors == 1 and rt.lag_steps == 99
    assert rt.worst_lag_s == pytest.approx(0.2)
    rt._tick(clock.now)
    assert clock.sleeps[-1] == pytest.approx(sim.sim_dt), "the schedule now lags wall time by what was dropped"
    _ticks(rt, clock, 20)  # the report window closes with that one event in it
    assert "fell behind wall clock 1x" in eden_caplog.text and "500 Hz" in eden_caplog.text
    rt._stopped_at = clock.now
    assert "99 substeps of sim time dropped" in rt.summary()


def test_nothing_is_reported_while_keeping_up(sim, eden_caplog):
    rt, clock = _loop(sim, report_every_s=0.1)
    _ticks(rt, clock, 400)
    assert "fell behind" not in eden_caplog.text
    rt._stopped_at = clock.now
    line = rt.summary()
    assert line.startswith("[deploy] MuJoCo physics: 200 substeps, 500 Hz achieved against 500 Hz")
    assert "worst backlog 2.0 ms" in line and "fell behind" not in line


# --- command / state hand-off -----------------------------------------------------------------


def test_a_new_command_drives_the_next_burst(sim):
    rt, clock = _loop(sim)
    rt.set_command(_hold(0.7))
    clock.now = sim.sim_dt
    rt._tick(clock.now)
    np.testing.assert_allclose(sim.read()["dofs_torque"], KP * 0.7, atol=1e-9)


def test_read_returns_the_latest_published_state_as_a_copy(sim):
    rt, clock = _loop(sim)
    before = rt.read()
    clock.now = 3 * sim.sim_dt
    rt._tick(clock.now)

    after = rt.read()
    np.testing.assert_allclose(after["dofs_pos"], sim.read()["dofs_pos"])
    assert after["stamp"] >= before["stamp"]
    after["dofs_pos"][0] = -1.0
    assert rt.read()["dofs_pos"][0] != -1.0


def test_reset_holds_the_new_pose_and_restarts_the_schedule(sim):
    rt, clock = _loop(sim)
    rt.set_command(_hold(0.7))
    clock.now = 10 * sim.sim_dt
    rt._tick(clock.now)

    rt.reset(np.full(2, 0.3), hold=_hold(0.3))

    np.testing.assert_allclose(rt.read()["dofs_pos"], 0.3)
    assert rt.substeps == 0
    rt._tick(clock.now)
    assert clock.sleeps[-1] == pytest.approx(sim.sim_dt), "nothing is owed right after a reset"
    rt._tick(clock.now)
    np.testing.assert_allclose(sim.read()["dofs_torque"], 0.0, atol=1e-9, err_msg="held at 0.3, not driven to 0.7")
    assert rt.reanchors == 0


# --- thread lifecycle ---------------------------------------------------------------------


def _wait_dead(rt, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while rt.running and time.perf_counter() < deadline:
        time.sleep(0.01)


def test_a_dead_thread_raises_on_the_control_thread(sim):
    """A physics thread that dies must not leave the control loop reading a frozen robot."""
    rt = RealtimePhysics(sim, _hold(width=3), report_every_s=0)  # a command for a 3-joint robot
    rt.start()
    _wait_dead(rt)

    assert not rt.running
    with pytest.raises(RuntimeError, match="(?s)thread died.*broadcast"):
        rt.read()
    with pytest.raises(RuntimeError, match="thread died"):
        rt.set_command(_hold())
    with pytest.raises(RuntimeError, match="thread died"), rt.exclusive():
        pass
    rt.stop()


def test_the_thread_keeps_sim_time_at_wall_time_and_stop_is_final(sim):
    rt = RealtimePhysics(sim, _hold(), report_every_s=0)
    rt.start()
    t0 = time.perf_counter()
    time.sleep(0.3)
    rt.stop()
    elapsed = time.perf_counter() - t0

    # Within 20 %: a loaded host wakes late, then catches up.
    assert sim.data.time == pytest.approx(elapsed, rel=0.2)
    assert rt.achieved_rate() == pytest.approx(500.0, rel=0.4)
    assert rt.reanchors == 0
    steps = rt.substeps
    time.sleep(0.02)
    assert rt.substeps == steps
    rt.stop()  # idempotent


def test_exclusive_holds_the_thread_off_and_the_backlog_follows(sim):
    rt = RealtimePhysics(sim, _hold(), report_every_s=0)
    rt.start()
    try:
        time.sleep(0.05)
        with rt.exclusive() as held:
            assert held is sim
            before = rt.substeps
            time.sleep(0.05)
            assert rt.substeps == before, "nothing ran behind the caller's back"
        deadline = time.perf_counter() + 1.0
        while rt.substeps < before + 20 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert rt.substeps >= before + 20, "the 25 owed substeps arrive as a burst"
        assert rt.reanchors == 0
    finally:
        rt.stop()


def test_starting_twice_raises(sim):
    rt = RealtimePhysics(sim, _hold(), report_every_s=0)
    rt.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            rt.start()
    finally:
        rt.stop()


class _Viewer:
    """The passive MuJoCo viewer's surface (a GUI window): records lock / overlay / sync in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self._lock = threading.RLock()

    def is_running(self) -> bool:
        return True

    def lock(self):
        viewer = self

        class _Held:
            def __enter__(self):
                viewer._lock.acquire()
                viewer.events.append("lock")

            def __exit__(self, *exc):
                viewer.events.append("unlock")
                viewer._lock.release()

        return _Held()

    def sync(self) -> None:
        self.events.append("sync")


def test_the_overlay_is_drawn_under_the_viewer_lock_before_each_sync(sim):
    viewer = _Viewer()
    drawn = []
    rt, clock = _loop(sim, viewer=viewer, viewer_overlay=lambda v: (drawn.append(v), viewer.events.append("overlay")))
    _ticks(rt, clock, 4)
    assert drawn and all(v is viewer for v in drawn)
    first_sync = viewer.events.index("sync")
    assert viewer.events[:first_sync][-3:] == ["lock", "overlay", "unlock"]


# --- construction ------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"max_lag_s": 0.0}, {"max_lag_s": -1.0}, {"report_every_s": -0.5}])
def test_bad_constructor_arguments_raise(sim, kwargs):
    with pytest.raises(ValueError):
        RealtimePhysics(sim, _hold(), **kwargs)


def test_a_pd_command_copies_its_arrays():
    """The thread reads a held command without a lock, so a caller's later write must not reach it."""
    pos = np.array([0.1, 0.2])
    cmd = PDCommand.hold(pos, [1.0, 1.0], [0.1, 0.1])
    pos[0] = 9.0
    np.testing.assert_allclose(cmd.target_pos, [0.1, 0.2])
    np.testing.assert_allclose(cmd.target_vel, 0.0)
    np.testing.assert_allclose(cmd.tau_ff, 0.0)
