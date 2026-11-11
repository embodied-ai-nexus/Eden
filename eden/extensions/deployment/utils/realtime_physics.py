"""Run a simulator on its own thread, in wall-clock time, the way a robot's controller runs.

A hardware deployment has two clocks that nothing keeps in step. The robot's low-level
controller closes its PD loop and publishes state at its own fixed rate whether or not the
policy is ready; the policy process reads whatever state is *latest* when it wakes, and the
command it writes takes effect whenever the controller next picks it up. A policy that
overruns its control period does not pause the robot -- the robot keeps executing the last
command, and the policy's next observation is that much older relative to what it decided
on.

The lockstep sim-to-sim path (:meth:`MujocoSim.step_pd` called from ``send_payload``) hides
all of that: physics advances exactly one control period per command, so the policy can be
arbitrarily late and the rollout is identical. :class:`RealtimePhysics` restores the two
clocks. It owns the simulator, advances it on a background thread so that simulated time
tracks wall time one-to-one, and exposes the same *latest command in, latest state out*
interface a wire would: :meth:`set_command` replaces the held command, which the thread uses
from its next substep; :meth:`read` returns the most recently published state.

**Pacing is catch-up, not fixed-period.** The thread schedules substep ``k`` at
``t0 + k * sim_dt`` and, on every wake, runs however many are overdue in one burst before
sleeping until the next is due. That keeps sim time equal to wall time through the timing
noise this thread lives with: ``time.sleep`` overshoot, and -- the large one -- the
interpreter's GIL. A CPython thread that wants the GIL back while the control thread is
inside pure-Python code waits for the switch interval, 5 ms by default, so a 500 Hz physics
thread routinely wakes two or three substeps late while the policy assembles its
observation. A burst of substeps with one held command is exactly what the lockstep path
does for a whole control period, so the dynamics are unaffected; only *when* a new command
is picked up jitters, which is also true of a real transport.

**A thread that cannot keep up says so.** If the backlog ever exceeds ``max_lag_s`` the
thread re-anchors its schedule (sim time falls behind wall time by that much, once) rather
than letting the burst grow without bound, counts the event, and reports it on the log
periodically and again in :meth:`summary` at shutdown. A thread that dies is worse than
one that is slow, because the control loop would keep reading the last published state and
see a robot frozen mid-air with no error: :meth:`read` and :meth:`set_command` therefore
raise with the thread's traceback once it has stopped for any reason other than
:meth:`stop`.

The class is engine-agnostic by duck typing -- it needs ``sim_dt``, ``step_pd``, ``read`` and
``reset`` -- so its pacing and health logic are unit-tested against a fake simulator and a
fake clock, with one real-MuJoCo test for the integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np

import eden as en


class _SteppableSim(Protocol):
    """What :class:`RealtimePhysics` needs from a simulator (:class:`MujocoSim` satisfies it)."""

    sim_dt: float

    def step_pd(self, target_pos, target_vel, kp, kd, tau_ff, n_substeps: int) -> None: ...
    def read(self) -> dict[str, np.ndarray]: ...
    def reset(self, dofs_pos, base_pos=None, base_quat=None) -> None: ...


@dataclass(frozen=True)
class PDCommand:
    """One joint-space PD command, as the thread holds it between updates.

    Frozen and array-copied at construction so the physics thread can read it without a
    lock: the control thread never mutates a held command, it replaces it.
    """

    target_pos: np.ndarray
    target_vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray

    @classmethod
    def hold(cls, pos, kp, kd) -> PDCommand:
        """A command that holds ``pos`` with zero velocity target and no feed-forward torque."""
        pos = np.array(pos, dtype=np.float64)
        return cls(
            target_pos=pos,
            target_vel=np.zeros_like(pos),
            kp=np.array(kp, dtype=np.float64),
            kd=np.array(kd, dtype=np.float64),
            tau_ff=np.zeros_like(pos),
        )

    @classmethod
    def of(cls, target_pos, target_vel, kp, kd, tau_ff) -> PDCommand:
        """Copy the five arrays into a new command."""
        return cls(
            target_pos=np.array(target_pos, dtype=np.float64),
            target_vel=np.array(target_vel, dtype=np.float64),
            kp=np.array(kp, dtype=np.float64),
            kd=np.array(kd, dtype=np.float64),
            tau_ff=np.array(tau_ff, dtype=np.float64),
        )


class RealtimePhysics:
    """Advance ``sim`` on a background thread so simulated time tracks wall-clock time.

    Parameters
    ----------
    sim:
        The simulator to drive. It is stepped only from the physics thread (and from
        :meth:`reset`, which excludes the thread while it runs).
    hold:
        The command applied from the first substep until :meth:`set_command` replaces it,
        normally the robot's default pose at its deploy gains.
    max_lag_s:
        Backlog, in seconds of simulated time, past which the thread re-anchors its schedule
        instead of catching up. The re-anchor is counted and reported. The default admits the
        one stall every run has -- the policy's first inference and warm-up hold the GIL for
        tens of milliseconds right after the thread starts -- and still bounds a catch-up burst
        to about a hundred substeps, some 10 ms of compute for a 29-DOF humanoid.
    report_every_s:
        How often the thread reports on the log while it is falling behind (``0`` disables
        the periodic report; :meth:`summary` is unaffected).
    name:
        Prefix for log lines.
    viewer:
        A ``mujoco.viewer`` passive handle to keep in sync from the physics thread, or
        ``None``. Stepping runs under the viewer's own lock so its GUI thread never reads a
        half-updated ``mjData``.
    viewer_hz:
        Rate at which the viewer is synced (the physics rate would just burn the GUI).
    viewer_overlay:
        Called with the viewer handle right before each sync, under the viewer's lock and with
        the simulator excluded, to draw into ``viewer.user_scn`` -- the deployment's gantry
        ropes, say. ``None`` draws nothing.
    clock, sleep:
        Time source and sleeper, injectable for tests. Defaults to ``time.perf_counter`` and
        ``time.sleep``.
    """

    def __init__(
        self,
        sim: _SteppableSim,
        hold: PDCommand,
        *,
        max_lag_s: float = 0.25,
        report_every_s: float = 1.0,
        name: str = "[deploy] MuJoCo physics",
        viewer: Any = None,
        viewer_hz: float = 60.0,
        viewer_overlay: Callable[[Any], None] | None = None,
        clock=time.perf_counter,
        sleep=time.sleep,
    ) -> None:
        if max_lag_s <= 0:
            raise ValueError(f"max_lag_s must be positive, got {max_lag_s}")
        if report_every_s < 0:
            raise ValueError(f"report_every_s must be non-negative, got {report_every_s}")
        self.sim = sim
        self.sim_dt = float(sim.sim_dt)
        self.max_lag_s = float(max_lag_s)
        self.report_every_s = float(report_every_s)
        self.name = name
        self._viewer = viewer
        self._viewer_period = 1.0 / float(viewer_hz)
        self._viewer_overlay = viewer_overlay
        self._clock = clock
        self._sleep = sleep

        # Command and state cross threads under one short-held lock. The step itself runs
        # under ``_step_lock`` instead, so a burst never blocks the control thread's
        # ``set_command``/``read`` for its duration; ``reset`` takes both.
        self._io_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._command: PDCommand = hold
        self._snapshot: dict[str, Any] = self._take_snapshot()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._died: str | None = None

        # Schedule and health. ``_t0`` is the wall time of substep 0; substep ``k`` is due at
        # ``_t0 + k * sim_dt``. All of these are touched only by the physics thread (and by
        # ``reset``/``start`` while it is excluded).
        self._t0 = 0.0
        self._steps = 0
        self._next_viewer_sync = 0.0
        self.reanchors = 0
        self.lag_steps = 0
        self.worst_lag_s = 0.0
        self._win_t0 = 0.0
        self._win_reanchors = 0
        self._win_worst_lag_s = 0.0
        self._next_report = 0.0
        self._started_at = 0.0
        self._stopped_at: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the physics thread. The schedule is anchored at the moment of the call."""
        if self._thread is not None:
            raise RuntimeError(f"{self.name}: already started")
        self._stop.clear()
        self._anchor(self._clock())
        self._started_at = self._t0
        self._thread = threading.Thread(target=self._run, name="mujoco-realtime-physics", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the physics thread and wait for it to exit. Safe to call more than once."""
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout_s)
        if thread.is_alive():
            en.logger.warning(f"{self.name}: thread did not exit within {timeout_s:g} s.")
        if self._stopped_at is None:
            self._stopped_at = self._clock()
        self._thread = None

    @property
    def running(self) -> bool:
        """True while the thread is alive and has not stopped or died."""
        return self._thread is not None and self._thread.is_alive() and self._died is None

    # -- control-thread interface -----------------------------------------

    def set_command(self, command: PDCommand) -> None:
        """Replace the held command; the thread applies it from its next substep."""
        self._raise_if_dead()
        with self._io_lock:
            self._command = command

    def read(self) -> dict[str, Any]:
        """Return the latest published state (a copy), with its ``stamp`` (``time.time()``)."""
        self._raise_if_dead()
        with self._io_lock:
            snap = self._snapshot
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in snap.items()}

    def reset(self, dofs_pos, base_pos=None, base_quat=None, *, hold: PDCommand) -> None:
        """Reset the simulator with the thread excluded, hold ``hold``, and re-anchor the schedule.

        Usable before :meth:`start` and while running. The thread's backlog is discarded
        rather than caught up -- the new episode starts at real time -- and is not counted as
        a lag re-anchor.
        """
        self._raise_if_dead()
        with self._step_lock:
            self.sim.reset(dofs_pos, base_pos, base_quat)
            with self._io_lock:
                self._command = hold
                self._snapshot = self._take_snapshot()
            self._anchor(self._clock())

    @contextmanager
    def exclusive(self) -> Iterator[_SteppableSim]:
        """Hold the physics thread off while the caller touches the simulator from another thread.

        For the simulator mutations a deployment makes outside a command -- rigging or cranking
        a gantry, say -- which would otherwise race the burst in progress. The schedule is
        *not* re-anchored: the substeps owed while the block ran are caught up on the next
        tick, like any other stall, so keep it short (a long hold shows up as a lag re-anchor
        and is reported as one). Use :meth:`reset` for anything that restarts the episode.
        """
        self._raise_if_dead()
        with self._step_lock:
            yield self.sim

    # -- reporting -----------------------------------------------------------

    @property
    def substeps(self) -> int:
        """Substeps run since :meth:`start` (or the last :meth:`reset`)."""
        return self._steps

    def achieved_rate(self) -> float:
        """Substeps per wall-clock second since the last anchor (``0.0`` before any substep)."""
        end = self._stopped_at if self._stopped_at is not None else self._clock()
        elapsed = end - self._t0
        return self._steps / elapsed if self._steps and elapsed > 0 else 0.0

    def summary(self) -> str:
        """One line on whether simulated time kept up with wall time over the run."""
        target = 1.0 / self.sim_dt
        line = (
            f"{self.name}: {self._steps} substeps, {self.achieved_rate():.0f} Hz achieved against {target:g} Hz, "
            f"worst backlog {self.worst_lag_s * 1e3:.1f} ms"
        )
        if self.reanchors:
            line += (
                f"; fell behind wall clock {self.reanchors}x (re-anchored past {self.max_lag_s * 1e3:.0f} ms, "
                f"{self.lag_steps} substeps of sim time dropped)"
            )
        return line

    # -- physics thread ---------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._tick(self._clock())
        except BaseException:  # noqa: BLE001 -- recorded and re-raised on the control thread
            self._died = traceback.format_exc()
            en.logger.error(f"{self.name}: thread died:\n{self._died}")
        finally:
            self._stopped_at = self._clock()

    def _tick(self, now: float) -> None:
        """One scheduler iteration at wall time ``now``: sleep until due, or run the overdue burst.

        The schedule arithmetic and the burst run under ``_step_lock`` so a concurrent
        :meth:`reset` (which re-anchors the schedule) never interleaves with them; the sleep
        happens outside it.
        """
        with self._step_lock:
            # The epsilon keeps a wake exactly at a due time from truncating to "not yet" in
            # floating point and going back to sleep for nothing.
            due = int((now - self._t0) / self.sim_dt + 1e-9)
            behind = due - self._steps
            if behind <= 0:
                wait_s = self._t0 + (self._steps + 1) * self.sim_dt - now
            else:
                self._account_lag(behind)
                if behind * self.sim_dt > self.max_lag_s:
                    behind = self._reanchor(behind)
                with self._io_lock:
                    command = self._command
                self._burst(command, behind)
                snapshot = self._take_snapshot()
                with self._io_lock:
                    self._snapshot = snapshot
                self._steps += behind
        if behind <= 0:
            self._sleep(max(0.0, wait_s))
            return

        if self._viewer is not None and now >= self._next_viewer_sync:
            self._sync_viewer()
            self._next_viewer_sync = now + self._viewer_period
        if self.report_every_s > 0 and now >= self._next_report:
            self._report(now)

    def _account_lag(self, behind: int) -> None:
        lag_s = behind * self.sim_dt
        self.worst_lag_s = max(self.worst_lag_s, lag_s)
        self._win_worst_lag_s = max(self._win_worst_lag_s, lag_s)

    def _reanchor(self, behind: int) -> int:
        """Give up all but one of the overdue substeps and shift the schedule accordingly.

        Sim time now lags wall time by the dropped amount for the rest of the run (or until
        the next reset). The alternative -- an ever-growing burst -- would hold ``_step_lock``
        for as long as the burst takes and freeze the control thread's reset behind it.
        """
        dropped = behind - 1
        self.reanchors += 1
        self._win_reanchors += 1
        self.lag_steps += dropped
        self._t0 += dropped * self.sim_dt
        return 1

    def _burst(self, command: PDCommand, n: int) -> None:
        if self._viewer is not None:
            with self._viewer.lock():
                self.sim.step_pd(command.target_pos, command.target_vel, command.kp, command.kd, command.tau_ff, n)
        else:
            self.sim.step_pd(command.target_pos, command.target_vel, command.kp, command.kd, command.tau_ff, n)

    def _take_snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = dict(self.sim.read())
        snap["stamp"] = time.time()
        return snap

    def _sync_viewer(self) -> None:
        try:
            if self._viewer.is_running():
                if self._viewer_overlay is not None:
                    # Excluded from ``reset``/``exclusive`` on the control thread, so the overlay
                    # reads a consistent ``mjData``; under the viewer's lock so the GUI does too.
                    with self._step_lock, self._viewer.lock():
                        self._viewer_overlay(self._viewer)
                self._viewer.sync()
        except Exception as exc:  # noqa: BLE001 -- the viewer is a convenience; physics must not die with it
            en.logger.warning(f"{self.name}: viewer sync failed ({exc}); no longer syncing the viewer.")
            self._viewer = None

    def _anchor(self, now: float) -> None:
        self._t0 = now
        self._steps = 0
        self._next_viewer_sync = now
        self._win_t0 = now
        self._win_reanchors = 0
        self._win_worst_lag_s = 0.0
        self._next_report = now + self.report_every_s

    def _report(self, now: float) -> None:
        if self._win_reanchors:
            window_s = now - self._win_t0
            en.logger.warning(
                f"{self.name} fell behind wall clock {self._win_reanchors}x in the last {window_s:.1f} s "
                f"(worst backlog {self._win_worst_lag_s * 1e3:.1f} ms; re-anchored past {self.max_lag_s * 1e3:.0f} ms). "
                f"Simulated time now lags real time; this host is not stepping the physics at {1.0 / self.sim_dt:g} Hz."
            )
        self._win_t0 = now
        self._win_reanchors = 0
        self._win_worst_lag_s = 0.0
        self._next_report = now + self.report_every_s

    def _raise_if_dead(self) -> None:
        if self._died is not None:
            raise RuntimeError(f"{self.name} thread died; the simulated robot is frozen. Cause:\n{self._died}")
        if self._thread is not None and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError(f"{self.name} thread exited unexpectedly; the simulated robot is frozen.")
