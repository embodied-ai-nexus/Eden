"""``eden.utils.devices.pico.bridge``: which sockets a running bridge binds for a given flag set.

The bridge runs for real as ``python -m eden.utils.devices.pico.bridge`` in a subprocess. The only stand-in
is the SDK: a fake ``xrobotoolkit_sdk`` module on the subprocess's ``PYTHONPATH``. Each test waits for the
first published pose (by then every socket is bound and the loop is running), probes the ports from
outside, then stops the bridge with SIGINT and checks it shut down cleanly.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import zmq

from tests.utils.devices.pico.conftest import write_fake_sdk_module

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STARTUP_TIMEOUT_S = 90.0  # the bridge imports eden and genesis before it binds anything
_PORT_ATTEMPTS = 3


def _free_tcp_ports(n: int) -> list[int]:
    """``n`` distinct ports that were free a moment ago (all held open together, so they differ)."""
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _port_is_bound(port: int) -> bool:
    """Whether something already holds ``port`` on all interfaces (our own bind fails)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def _accepts_connections(port: int, timeout: float = 10.0) -> bool:
    """Whether a TCP listener comes up on ``127.0.0.1:port`` within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


class _Bridge:
    """A bridge subprocess over the fake SDK."""

    def __init__(self, tmp_path: Path, flags: list[str], attempt: int) -> None:
        self.pose_port, self.frame_port, self.control_port = _free_tcp_ports(3)
        sdk_dir = tmp_path / f"attempt{attempt}" / "sdk"
        self.sdk_events = write_fake_sdk_module(sdk_dir)
        self.output = sdk_dir.parent / "bridge.log"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(sdk_dir), *filter(None, [env.get("PYTHONPATH")])])
        argv = [
            sys.executable,
            "-m",
            "eden.utils.devices.pico.bridge",
            "--fps",
            "50",
            "--pose-port",
            str(self.pose_port),
            "--frame-port",
            str(self.frame_port),
            "--control-address",
            f"127.0.0.1:{self.control_port}",
            *flags,
        ]
        with open(self.output, "wb") as out:
            self.proc = subprocess.Popen(argv, cwd=_REPO_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)

    def log(self) -> str:
        return self.output.read_text(errors="replace")

    def first_pose(self) -> dict | None:
        """The first state the bridge publishes, or None if it exits first (the log says why)."""
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.LINGER, 0)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(f"tcp://127.0.0.1:{self.pose_port}")
        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT_S
            while time.monotonic() < deadline and self.proc.poll() is None:
                if sub.poll(200):
                    return json.loads(sub.recv_string())
            return None
        finally:
            sub.close()

    def stop(self) -> int:
        """SIGINT the bridge and return its exit code; kills it (and raises) if it does not stop."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
        try:
            return self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise


@pytest.fixture
def run_bridge(tmp_path):
    """``run_bridge(flags)`` starts a bridge and returns it with its first pose; every bridge is reaped."""
    bridges: list[_Bridge] = []

    def start(flags: list[str]) -> tuple[_Bridge, dict]:
        for attempt in range(_PORT_ATTEMPTS):
            bridge = _Bridge(tmp_path, flags, attempt)
            bridges.append(bridge)
            pose = bridge.first_pose()
            if pose is not None:
                return bridge, pose
            bridge.stop()
            if "Address already in use" not in bridge.log():  # only a lost port race is worth a retry
                break
        pytest.fail(f"the bridge never published a pose; its output:\n{bridge.log()}")

    yield start

    for bridge in bridges:
        if bridge.proc.poll() is None:
            bridge.proc.kill()
            bridge.proc.wait()


@pytest.mark.parametrize("video", [False, True], ids=["no_video", "video"])
def test_bridge_binds_the_frame_and_control_ports_only_with_video(run_bridge, video):
    """--no-video leaves the frame port free: its default (5556) is the SONIC operator's command PUB."""
    bridge, pose = run_bridge([] if video else ["--no-video"])

    # The pose channel comes up either way, carrying what the SDK reports.
    assert pose["device"] == "pico" and pose["ready"] is True
    assert pose["timestamp_ns"] == 123
    assert _port_is_bound(bridge.frame_port) is video
    if video:
        assert _accepts_connections(bridge.control_port)
    else:
        assert not _port_is_bound(bridge.control_port)

    assert bridge.stop() == 0, bridge.log()
    assert bridge.sdk_events.read_text().split() == ["init", "close"]  # SIGINT ends the loop and closes the SDK
