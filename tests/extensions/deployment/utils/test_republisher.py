"""``CommandRepublisherMixin``: re-sending the held wire command between control ticks, and saying when it stops.

The mixin is exercised through the smallest backend that uses it the way the G1 does: a wire
that records each packet, and a hook gated until the first command. The per-backend wiring
(G1 CRC, packet layout) is tested with each backend.
"""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path

import pytest

from eden.extensions.deployment.limx_oli import LimxOliDeployment
from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
from eden.extensions.deployment.unitree_go2 import UnitreeGo2Deployment
from eden.extensions.deployment.utils.republisher import CommandRepublisherMixin, publish_rate_configure_kwargs

REPUBLISHING_BACKENDS = (UnitreeG1Deployment, LimxOliDeployment, UnitreeGo2Deployment)


class _GatedBackend(CommandRepublisherMixin):
    """A backend's republisher half: ``send_payload`` publishes under the lock; the hook re-sends, gated like the G1's."""

    def __init__(self, control_freq: float | None = 50.0, publish_rate: float | None = 500.0) -> None:
        self.control_freq = control_freq
        self.control_dt = 1.0 / control_freq if control_freq else 0.0
        self.publish_rate = publish_rate
        self._init_republisher()
        self.command: float | None = None
        self.wire: list[tuple[float, bool]] = []  # (packet, whether the command lock was held)

    def send_payload(self, command: float) -> None:
        with self._cmd_lock:
            self.command = command
            self.wire.append((command, self._cmd_lock.locked()))
            self._note_command_published()

    def _publish_held_command(self) -> None:
        if self.command is None:
            return
        self.wire.append((self.command, self._cmd_lock.locked()))
        self._pub_count += 1


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return True
        time.sleep(0.002)
    return False


# --- the rate --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("control_freq", "publish_rate", "match"),
    [
        (50.0, 20.0, "must be >= the control frequency"),
        (50.0, 0.0, "must be >= the control frequency"),
        (None, 500.0, "control rate is unknown"),
        (50.0, None, None),
        (50.0, 500.0, None),
    ],
    ids=["below-control-rate", "zero", "unknown-control-rate", "off", "above"],
)
def test_the_publish_rate_is_checked_against_the_control_rate(control_freq, publish_rate, match):
    backend = _GatedBackend(control_freq=control_freq, publish_rate=publish_rate)
    if match is None:
        backend._validate_publish_rate()
    else:
        with pytest.raises(ValueError, match=match):
            backend._validate_publish_rate()


def test_zero_selects_one_packet_per_tick_and_absent_keeps_the_backend_default():
    """The A/B flag must not silently select the default it exists to be compared against."""
    assert publish_rate_configure_kwargs(None) == {}
    assert publish_rate_configure_kwargs(0.0) == {"publish_rate": None}
    assert publish_rate_configure_kwargs(250.0) == {"publish_rate": 250.0}


def test_each_backend_defaults_to_its_vendors_own_rate():
    """Unitree's g1_low_level_example and go2_stand_example (500 Hz), LimX's shipped controllers (1 kHz)."""
    assert UnitreeG1Deployment.configure(entity_name="robot").publish_rate == 500.0
    assert UnitreeGo2Deployment.publish_rate == 500.0
    assert LimxOliDeployment.publish_rate == 1000.0


# --- re-sending ---------------------------------------------------------------------------------


def test_the_held_command_is_resent_under_the_lock_until_stopped():
    backend = _GatedBackend()
    backend.send_payload(0.25)
    backend._start_publisher()
    time.sleep(0.2)
    backend._stop_publisher()

    assert len(backend.wire) >= 40, len(backend.wire)  # 100 nominal at 500 Hz; a floor for a loaded box
    assert set(backend.wire) == {(0.25, True)}
    assert backend._pub_thread is None, "stopped and joined"


def test_a_new_command_replaces_what_is_resent():
    backend = _GatedBackend()
    backend.send_payload(0.1)
    backend._start_publisher()
    time.sleep(0.05)
    backend.send_payload(0.9)
    time.sleep(0.05)
    backend._stop_publisher()

    sent = [packet for packet, _ in backend.wire]
    assert 0.1 in sent and set(sent[sent.index(0.9) :]) == {0.9}


def test_nothing_is_published_before_the_first_command_or_with_the_rate_off():
    gated = _GatedBackend()
    gated._start_publisher()
    time.sleep(0.05)
    gated._stop_publisher()
    assert gated.wire == []

    off = _GatedBackend(publish_rate=None)
    off._start_publisher()
    assert off._pub_thread is None


def test_a_backend_without_the_hook_cannot_be_constructed():
    """``@abstractmethod`` on a plain mixin is inert, so a forgotten hook used to fail only inside the thread."""

    class _NoHook(CommandRepublisherMixin):
        pass

    with pytest.raises(TypeError, match="_publish_held_command"):
        _NoHook()


# --- a republisher that stops working says so -----------------------------------------------------


def test_a_failing_send_is_counted_and_retried(eden_caplog):
    """A transport hiccup must not cost the robot its damping rate for the rest of the run."""
    backend = _GatedBackend()
    hook = backend._publish_held_command

    def flaky() -> None:
        if backend._pub_errors < 5:
            raise OSError("transient DDS write failure")
        hook()

    backend._publish_held_command = flaky
    backend.send_payload(0.25)
    backend._start_publisher()
    assert _wait_until(lambda: len(backend.wire) >= 10)
    backend._stop_publisher()

    assert backend._pub_errors == 5 and backend._pub_died is None
    assert "5 send(s) failed (last: OSError: transient DDS write failure)" in eden_caplog.text


def test_a_loop_that_dies_is_reported_at_error_level_with_its_traceback(eden_caplog):
    backend = _GatedBackend()

    def explode() -> None:
        raise RuntimeError("publisher exploded")

    backend._publish_until_stopped = explode
    backend._start_publisher()
    backend._pub_thread.join(timeout=5.0)
    backend._stop_publisher()

    [error] = [r.getMessage() for r in eden_caplog.records if r.levelname == "ERROR"]
    assert "Republisher thread died" in error and "Traceback" in error and "publisher exploded" in error
    assert "the loop died early (RuntimeError: publisher exploded)" in eden_caplog.text


def test_the_running_loop_reports_a_control_thread_that_stopped_feeding_it(eden_caplog):
    """Re-sending a frozen full-gain target looks healthy on the wire; only the command's age tells."""
    backend = _GatedBackend()
    backend._PUB_HEALTH_PERIOD_S = 0.05
    backend._PUB_STALE_TICKS = 0.5  # stale after 10 ms at 50 Hz
    backend._start_publisher()
    backend.send_payload(0.25)
    try:
        assert _wait_until(lambda: "the control thread has stopped feeding it" in eden_caplog.text)
    finally:
        backend._stop_publisher()


def test_a_starved_loop_is_reported_and_so_is_its_recovery(eden_caplog):
    backend = _GatedBackend()
    backend._pub_t0 = backend._pub_cmd_t = time.perf_counter()

    backend._check_health(window_s=1.0, sent=40, errors=0, now=time.perf_counter())
    assert "40 Hz against a 500 Hz target" in eden_caplog.text

    backend._check_health(window_s=1.0, sent=500, errors=0, now=time.perf_counter())
    assert "Republisher recovered" in eden_caplog.text


def test_no_false_starvation_warning_across_the_gate_handover(eden_caplog):
    """The gate lifts part way through a health window on every live robot; that idle span is not starvation."""
    backend = _GatedBackend()
    backend._PUB_HEALTH_PERIOD_S = 0.1
    backend._start_publisher()
    time.sleep(0.25)  # more than two health periods, correctly gated
    gate_lifted = time.perf_counter()
    try:
        for _ in range(16):
            backend.send_payload(0.25)
            time.sleep(0.02)
    finally:
        backend._stop_publisher()

    assert "degraded" not in eden_caplog.text
    assert backend._pub_t0 >= gate_lifted, "the rate clock starts at the first packet, not at thread start"
    assert len(backend.wire) > 16


def test_a_restart_inherits_no_diagnostics_from_the_last_session(eden_caplog):
    """A command stamp carried over from the last connect reads as minutes old on the first check."""
    backend = _GatedBackend()
    backend.send_payload(0.25)
    backend._start_publisher()
    time.sleep(0.05)
    backend._stop_publisher()
    backend._pub_errors, backend._pub_degraded, backend._pub_cmd_t = 7, True, 1.0

    backend._start_publisher()
    try:
        assert (backend._pub_errors, backend._pub_degraded, backend._pub_cmd_t, backend._pub_died) == (
            0,
            False,
            0.0,
            None,
        )
    finally:
        backend._stop_publisher()


def test_stop_says_so_when_the_thread_will_not_join(eden_caplog):
    backend = _GatedBackend()
    release = threading.Event()
    backend._pub_thread = threading.Thread(target=release.wait, daemon=True)
    backend._pub_thread.start()

    backend._stop_publisher()
    release.set()

    assert backend._pub_thread is None
    assert "did not stop within" in eden_caplog.text


# --- the backends stay on the one implementation -------------------------------------------------


def test_every_republishing_backend_shares_the_mixin_and_wires_its_calls():
    """A forked loop is how the G1 or the Oli quietly misses a fix made to the other.

    The wiring is a source check: each call is invisible when dropped, and connecting a backend
    needs its SDK.
    """
    shared = ("_publisher_loop", "_publish_until_stopped", "_check_health", "_start_publisher", "_stop_publisher")
    contract = {
        "__init__": "_init_republisher()",
        "_resolve_options": "_validate_publish_rate()",
        "connect": "_start_publisher()",
        "close": "_stop_publisher()",
        "send_payload": "_note_command_published()",
    }
    for backend in REPUBLISHING_BACKENDS:
        assert issubclass(backend, CommandRepublisherMixin)
        assert backend._publish_held_command is not CommandRepublisherMixin._publish_held_command
        forked = [name for name in shared if getattr(backend, name) is not getattr(CommandRepublisherMixin, name)]
        assert forked == [], backend.__name__
        for method, call in contract.items():
            # Every definition on the way down to the mixin, so a call made by a shared transport base counts.
            sources = [inspect.getsource(vars(cls)[method]) for cls in _own_classes(backend) if method in vars(cls)]
            assert any(call in source for source in sources), f"{backend.__name__}.{method} must call {call}"


def _own_classes(backend: type) -> list[type]:
    """``backend``'s MRO up to (excluding) the mixin: the backend and any transport base it shares with a sibling."""
    mro = backend.__mro__
    return list(mro[: mro.index(CommandRepublisherMixin)])


def test_every_g1_hardware_example_passes_publish_rate_through():
    """The scripts that construct the backend directly are the ones most likely to be pointed at a robot."""
    root = Path(__file__).resolve().parents[4] / "examples"
    scripts = [
        p
        for p in sorted(root.rglob("*.py"))
        if "UnitreeG1Deployment.configure(" in p.read_text() and "add_argument" in p.read_text()
    ]
    assert scripts, "no G1 hardware examples found; did they move?"
    for script in scripts:
        source = script.read_text()
        assert '"--publish-rate"' in source, script.name
        assert "publish_rate_configure_kwargs(args.publish_rate)" in source, script.name
