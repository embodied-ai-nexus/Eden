"""``PicoVideoLink`` driven end to end by a fake PICO app over TCP loopback: OPEN_CAMERA, H.264, CLOSE_CAMERA."""

from __future__ import annotations

import logging
import socket
import threading
import time

import numpy as np
import pytest

from eden.utils.devices.pico import protocol as P
from eden.utils.devices.pico.video import PicoVideoLink

CAMERA_REQUEST = P.CameraRequest(
    width=640, height=240, fps=30, bitrate=1_000_000, hevc=False, render_mode=0, port=0, camera="ZED", ip="127.0.0.1"
)


class _FakePicoApp:
    """Stands in for the XRoboToolkit app: a decoder socket plus a control client."""

    def __init__(self):
        self.decoder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.decoder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.decoder.bind(("127.0.0.1", 0))
        self.decoder.listen(1)
        self.decoder_port = self.decoder.getsockname()[1]
        self.control: socket.socket | None = None
        self.stream: socket.socket | None = None

    def connect(self, control_port: int) -> None:
        self.control = socket.create_connection(("127.0.0.1", control_port), timeout=5.0)
        self.control.settimeout(5.0)

    def open_camera(self, request: P.CameraRequest) -> None:
        """Send OPEN_CAMERA, then accept the video connection Eden dials back."""
        self.control.sendall(P.encode_message(P.OPEN_CAMERA, P.encode_camera_request(request)))
        self.decoder.settimeout(5.0)
        self.stream, _ = self.decoder.accept()
        self.stream.settimeout(5.0)

    def close_camera(self) -> None:
        self.control.sendall(P.encode_message(P.CLOSE_CAMERA))

    def read_ack(self) -> str:
        header = self._recv_exactly(self.control, P.HEADER_SIZE)
        body = self._recv_exactly(self.control, P.unpack_length(header))
        command, data = P.decode_message(body)
        assert data == b""
        return command

    def read_video_frame(self) -> bytes:
        header = self._recv_exactly(self.stream, P.HEADER_SIZE)
        return self._recv_exactly(self.stream, P.unpack_length(header))

    def close(self) -> None:
        for sock in (self.stream, self.control, self.decoder):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    @staticmethod
    def _recv_exactly(sock: socket.socket, size: int) -> bytes:
        buf = bytearray()
        while len(buf) < size:
            chunk = sock.recv(size - len(buf))
            if not chunk:
                raise AssertionError(f"peer closed after {len(buf)}/{size} bytes")
            buf += chunk
        return bytes(buf)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def link_and_app():
    """A started :class:`PicoVideoLink` on a free port plus a connected fake app."""
    port = _free_port()
    link = PicoVideoLink(f"127.0.0.1:{port}", logger=logging.getLogger("test_pico"))
    link.start()
    app = _FakePicoApp()
    try:
        assert _wait_until(lambda: _port_is_open(port)), "control server never came up"
        app.connect(port)
        yield link, app, port
    finally:
        app.close()
        link.stop()


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _encode_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "PicoVideoEncode"]


def _frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_open_camera_starts_an_h264_stream(link_and_app):
    link, app, _ = link_and_app
    request = P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": app.decoder_port})
    app.open_camera(request)

    assert app.read_ack() == P.OPEN_CAMERA
    assert _wait_until(lambda: link.streaming), "link never reported streaming"

    for i in range(4):
        link.push(_frame(request.width, request.height, seed=i))
        time.sleep(0.05)

    nal = app.read_video_frame()
    assert len(nal) > 0
    # Every H.264 access unit starts with an Annex-B start code.
    assert nal.startswith(b"\x00\x00\x00\x01") or nal.startswith(b"\x00\x00\x01")
    assert link.frames_sent >= 1


def test_pushed_frames_are_rescaled_to_the_requested_size(link_and_app):
    """The app configures its decoder from the request, so the stream must match it."""
    import av

    link, app, _ = link_and_app
    request = P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": app.decoder_port})
    app.open_camera(request)
    assert app.read_ack() == P.OPEN_CAMERA
    assert _wait_until(lambda: link.streaming)

    # Half-size frames: the encoder must still emit request.width x request.height.
    for i in range(6):
        link.push(_frame(request.width // 2, request.height // 2, seed=i))
        time.sleep(0.05)

    decoder = av.CodecContext.create("h264", "r")
    decoded = []
    deadline = time.time() + 5.0
    while not decoded and time.time() < deadline:
        for packet in decoder.parse(app.read_video_frame()):
            decoded.extend(decoder.decode(packet))
    assert decoded, "no frame decoded from the stream"
    assert (decoded[0].width, decoded[0].height) == (request.width, request.height)


def test_close_camera_stops_the_stream(link_and_app):
    link, app, _ = link_and_app
    request = P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": app.decoder_port})
    app.open_camera(request)
    assert app.read_ack() == P.OPEN_CAMERA
    assert _wait_until(lambda: link.streaming)

    app.close_camera()
    assert app.read_ack() == P.CLOSE_CAMERA
    assert _wait_until(lambda: not link.streaming), "stream stayed open after CLOSE_CAMERA"

    # The encode thread must actually exit — not merely be detached — so close()'s
    # encoder flush never races a live encode(). Regression guard: before the
    # stop-before-join fix the loop only stopped after close(), so the join always
    # timed out (~1.5s) and left the thread running against the encoder.
    assert _wait_until(lambda: not _encode_threads(), timeout=2.0), "encode thread outlived CLOSE_CAMERA"

    # push() is a harmless no-op once the headset is gone.
    link.push(_frame(request.width, request.height))


def test_close_camera_teardown_is_prompt(link_and_app):
    """CLOSE_CAMERA must tear the stream down quickly, not burn the join timeout."""
    link, app, _ = link_and_app
    request = P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": app.decoder_port})
    app.open_camera(request)
    assert app.read_ack() == P.OPEN_CAMERA
    assert _wait_until(lambda: link.streaming)
    for i in range(4):  # keep frames flowing so the encode thread is actively working
        link.push(_frame(request.width, request.height, seed=i))
        time.sleep(0.05)

    start = time.perf_counter()
    app.close_camera()
    assert app.read_ack() == P.CLOSE_CAMERA
    assert _wait_until(lambda: not _encode_threads(), timeout=2.0)
    assert time.perf_counter() - start < 1.5, "teardown burned the join timeout"


def test_open_camera_with_an_unreachable_decoder_does_not_stream(link_and_app):
    link, app, _ = link_and_app
    request = P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": _free_port()})
    app.control.sendall(P.encode_message(P.OPEN_CAMERA, P.encode_camera_request(request)))

    assert app.read_ack() == P.OPEN_CAMERA
    time.sleep(0.2)
    assert not link.streaming


@pytest.mark.parametrize(
    "override",
    [{"width": 0}, {"height": -8}, {"width": 641}, {"ip": ""}, {"port": 0}],
)
def test_unusable_camera_requests_are_rejected(link_and_app, override):
    link, app, _ = link_and_app
    request = P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": app.decoder_port, **override})
    app.control.sendall(P.encode_message(P.OPEN_CAMERA, P.encode_camera_request(request)))

    assert app.read_ack() == P.OPEN_CAMERA
    time.sleep(0.2)
    assert not link.streaming


def test_malformed_control_message_does_not_kill_the_connection(link_and_app):
    link, app, _ = link_and_app
    app.control.sendall(P.encode_message(P.OPEN_CAMERA, b"garbage"))  # bad magic
    assert app.read_ack() == P.OPEN_CAMERA
    assert not link.streaming

    # The next well-formed command on the same connection still works.
    app.open_camera(P.CameraRequest(**{**CAMERA_REQUEST.__dict__, "port": app.decoder_port}))
    assert app.read_ack() == P.OPEN_CAMERA
    assert _wait_until(lambda: link.streaming)


def test_start_twice_is_rejected():
    link = PicoVideoLink(f"127.0.0.1:{_free_port()}", logger=logging.getLogger("test_pico"))
    link.start()
    try:
        with pytest.raises(RuntimeError, match="called twice"):
            link.start()
    finally:
        link.stop()


def test_stop_is_idempotent_and_joins_threads():
    link = PicoVideoLink(f"127.0.0.1:{_free_port()}", logger=logging.getLogger("test_pico"))
    link.start()
    link.stop()
    link.stop()
    assert not any(t.name.startswith("PicoVideo") for t in threading.enumerate())
