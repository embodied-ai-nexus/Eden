"""Stream Eden's rendered stereo frames to a PICO headset over the Remote Vision link.

:class:`PicoVideoLink` owns both sockets of the link documented in
:mod:`eden.utils.devices.pico.protocol`:

  * it **listens** on ``:13579`` for the ``OPEN_CAMERA`` / ``CLOSE_CAMERA``
    commands the XRoboToolkit app sends when the operator taps *Listen* / stops,
  * on ``OPEN_CAMERA`` it **connects out** to the decoder address the request
    names and pushes H.264 (or H.265) frames until the app closes the stream.

Two daemon threads do the work — one accepting control connections, one
encoding — so the caller only ever does::

    link = PicoVideoLink()
    link.start()
    while ...:
        link.push(rgb)   # (H, W, 3) uint8 RGB, left|right side-by-side
    link.stop()

``push()`` is non-blocking and latest-frame-wins: it drops into a single slot
that the encoder thread drains, so a slow encoder never stalls the sim loop and
the headset always shows the newest frame rather than a growing backlog.

Frames are encoded at the size the headset asked for; a differently sized frame
is rescaled (with a one-time warning) because the app's decoder is configured
from the ``OPEN_CAMERA`` request, not from the stream.

The encoder/sender half is ported from SIMPLE's ``src/simple/teleop/pico/``
(MIT, https://github.com/physical-superintelligence-lab/SIMPLE).
"""

from __future__ import annotations

import socket
import threading
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from eden.utils.devices.pico import protocol as P

if TYPE_CHECKING:
    import numpy.typing as npt

_ACCEPT_TIMEOUT = 1.0  # s between "should I stop?" checks while blocked on a socket
_FRAME_TIMEOUT = 0.5  # s the encoder thread waits for a new frame before re-checking
_CONNECT_TIMEOUT = 3.0  # s to reach the headset decoder before giving up (LAN link)
_DEFAULT_FPS = 30  # used when the app leaves the request's fps at 0
_DEFAULT_BITRATE = 4_000_000  # bps, likewise for an unset bitrate
_MAX_CONTROL_BODY = 64 * 1024  # a control message this large means the stream desynced


class PicoVideoLink:
    """Control server + video sender for one PICO headset.

    Parameters
    ----------
        address: ``"host:port"`` to listen on for control commands. The default
            matches the port the XRoboToolkit PICO app dials.
        logger: logger to report connection state on. Defaults to ``eden.logger``,
            which requires ``eden.init()``; the bridge passes its own so it works
            in a process that never boots Genesis.
    """

    def __init__(self, address: str = f"0.0.0.0:{P.CONTROL_PORT}", logger=None) -> None:
        host, _, port = address.rpartition(":")
        self._host = host or "0.0.0.0"
        self._port = int(port)
        self._logger = logger

        self._stop = threading.Event()
        self._control_thread: threading.Thread | None = None
        self._encode_thread: threading.Thread | None = None
        self._client: socket.socket | None = None
        self._sender: _VideoSender | None = None
        self._latest = _LatestFrame()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def streaming(self) -> bool:
        """True while a headset is connected and frames are being encoded."""
        return self._sender is not None and not self._sender.closed

    @property
    def frames_sent(self) -> int:
        """Frames encoded and written to the headset since the stream opened."""
        return self._sender.frames_sent if self._sender is not None else 0

    def start(self) -> None:
        """Start accepting control connections (returns immediately)."""
        if self._control_thread is not None:
            raise RuntimeError("PicoVideoLink.start() called twice; create a new link instead")
        self._stop.clear()
        self._control_thread = threading.Thread(target=self._serve, name="PicoVideoControl", daemon=True)
        self._control_thread.start()

    def stop(self) -> None:
        """Tear down the stream, the control connection, and both threads."""
        self._stop.set()
        self._close_stream()
        self._close_client()
        for thread in (self._control_thread, self._encode_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        self._control_thread = None
        self._encode_thread = None

    def push(self, rgb: npt.NDArray[np.uint8]) -> None:
        """Offer one rendered frame (``(H, W, 3)`` uint8 RGB) to the headset.

        Non-blocking and latest-wins; a no-op while no headset is streaming.
        """
        if self.streaming:
            self._latest.put(np.ascontiguousarray(rgb, dtype=np.uint8))

    # ------------------------------------------------------------------
    # control channel (background thread)
    # ------------------------------------------------------------------

    @property
    def _log(self):
        if self._logger is not None:
            return self._logger
        import eden as en

        return en.logger

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(_ACCEPT_TIMEOUT)
        self._log.info(f"PICO video link listening on {self._host}:{self._port}")

        try:
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                self._log.info(f"PICO app connected from {addr[0]}:{addr[1]}")
                self._handle_client(conn)
                self._close_stream()
                self._log.info("PICO app disconnected; waiting for the next connection")
        finally:
            server.close()

    def _handle_client(self, conn: socket.socket) -> None:
        """Serve one control connection until it closes or we are asked to stop."""
        self._client = conn
        conn.settimeout(_ACCEPT_TIMEOUT)
        try:
            while not self._stop.is_set():
                body = self._read_message(conn)
                if body is None:
                    return
                try:
                    command, data = P.decode_message(body)
                except ValueError as exc:
                    self._log.warning(f"malformed PICO control message ignored: {exc}")
                    continue

                if command == P.OPEN_CAMERA:
                    self._open_stream(data)
                    self._ack(conn, command)
                elif command == P.CLOSE_CAMERA:
                    # Ack before tearing down, matching the app's expected order.
                    self._ack(conn, command)
                    self._close_stream()
                else:
                    self._log.warning(f"unknown PICO control command {command!r} ignored")
        finally:
            self._close_client()

    def _read_message(self, conn: socket.socket) -> bytes | None:
        """Read one length-prefixed control body; None once the peer goes away.

        A length past :data:`_MAX_CONTROL_BODY` means the stream is out of sync
        (control messages are a few hundred bytes), so the connection is dropped
        rather than blocked waiting for bytes that will never come.
        """
        header = self._read_exactly(conn, P.HEADER_SIZE)
        if header is None:
            return None
        length = P.unpack_length(header)
        if length > _MAX_CONTROL_BODY:
            self._log.error(f"control message claims {length} bytes; dropping the desynced connection")
            return None
        return self._read_exactly(conn, length)

    def _read_exactly(self, conn: socket.socket, size: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < size:
            if self._stop.is_set():
                return None
            try:
                chunk = conn.recv(size - len(buf))
            except TimeoutError:
                continue
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _ack(self, conn: socket.socket, command: str) -> None:
        try:
            conn.sendall(P.encode_message(command))
        except OSError as exc:
            self._log.warning(f"failed to ack {command}: {exc}")

    def _close_client(self) -> None:
        conn, self._client = self._client, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # video channel
    # ------------------------------------------------------------------

    def _open_stream(self, data: bytes) -> None:
        """Handle OPEN_CAMERA: connect to the headset's decoder and start encoding."""
        if self.streaming:
            self._log.warning("OPEN_CAMERA while already streaming; ignoring the duplicate request")
            return
        try:
            request = P.decode_camera_request(data)
        except ValueError as exc:
            self._log.error(f"OPEN_CAMERA payload rejected: {exc}")
            return
        request = _sanitized(request, self._log)
        if request is None:
            return

        self._log.info(
            f"OPEN_CAMERA {request.width}x{request.height}@{request.fps}fps "
            f"{request.bitrate // 1000}kbps {'h265' if request.hevc else 'h264'} "
            f"-> {request.ip}:{request.port} (source {request.camera!r})"
        )
        try:
            self._sender = _VideoSender(request, self._log)
        except OSError as exc:
            self._log.error(f"cannot reach the headset decoder at {request.ip}:{request.port}: {exc}")
            self._sender = None
            return

        self._latest = _LatestFrame()  # drop any frame left over from the last stream
        self._encode_thread = threading.Thread(target=self._encode_loop, name="PicoVideoEncode", daemon=True)
        self._encode_thread.start()

    def _encode_loop(self) -> None:
        """Drain the frame slot into the encoder until the stream closes."""
        sender = self._sender
        while sender is not None and not sender.closed and not self._stop.is_set():
            frame = self._latest.get(timeout=_FRAME_TIMEOUT)
            if frame is not None:
                sender.send(frame)

    def _close_stream(self) -> None:
        sender, self._sender = self._sender, None
        thread, self._encode_thread = self._encode_thread, None
        # Stop the encode loop BEFORE joining, then release the encoder only once
        # the loop has exited. Joining first would deadlock the join against a loop
        # that only stops when close() runs, and would let close()'s encoder flush
        # race the still-live encode thread (PyAV CodecContext is not thread-safe).
        if sender is not None:
            sender.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_FRAME_TIMEOUT + 1.0)
        if sender is not None:
            sender.close()
            self._log.info(f"PICO video stream closed after {sender.frames_sent} frames")


def _sanitized(request: P.CameraRequest, logger) -> P.CameraRequest | None:
    """Reject a request Eden cannot honour; substitute defaults the app left unset.

    Returns the (possibly adjusted) request, or None when it is unusable — this
    is untrusted input from a device, so a bad request is logged and dropped
    rather than raised out of the control thread.
    """
    if min(request.width, request.height) <= 0 or request.width % 2 or request.height % 2:
        logger.error(f"OPEN_CAMERA asked for an unencodable {request.width}x{request.height} frame (need even, > 0)")
        return None
    if not request.ip or not 0 < request.port < 65536:
        logger.error(f"OPEN_CAMERA named an unusable decoder address {request.ip!r}:{request.port}")
        return None

    fps = request.fps if request.fps > 0 else _DEFAULT_FPS
    bitrate = request.bitrate if request.bitrate > 0 else _DEFAULT_BITRATE
    if (fps, bitrate) == (request.fps, request.bitrate):
        return request
    logger.warning(f"OPEN_CAMERA left fps/bitrate unset; encoding at {fps}fps {bitrate // 1000}kbps")
    return replace(request, fps=fps, bitrate=bitrate)


class _LatestFrame:
    """Single-slot latest-wins handoff from the sim thread to the encoder thread."""

    def __init__(self) -> None:
        self._frame: npt.NDArray[np.uint8] | None = None
        self._lock = threading.Lock()
        self._new = threading.Event()

    def put(self, frame: npt.NDArray[np.uint8]) -> None:
        with self._lock:
            self._frame = frame
        self._new.set()

    def get(self, timeout: float) -> npt.NDArray[np.uint8] | None:
        """Wait up to ``timeout`` seconds for a frame; None if none arrived."""
        if not self._new.wait(timeout):
            return None
        with self._lock:
            self._new.clear()
            return self._frame


class _VideoSender:
    """One H.264/H.265 stream to the decoder inside the PICO app.

    Parameters
    ----------
        request: the decoded ``OPEN_CAMERA`` request naming the codec, size,
            rate, and decoder address.
        logger: logger for send/encode failures.
    """

    def __init__(self, request: P.CameraRequest, logger) -> None:
        import av  # PyAV: optional 'pico' extra

        self._request = request
        self._log = logger
        self._lock = threading.Lock()
        # ``_stop`` signals the encode loop to stop touching the encoder; ``_released``
        # guards the one-time flush + socket teardown. Kept separate so close() can
        # flush safely only after the encode thread has been joined (see _close_stream).
        self._stop = threading.Event()
        self._released = False
        self.frames_sent = 0
        self._warned_resize = False

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Bound the connect so an unreachable-but-silent decoder IP can't hang the
        # control thread for the OS SYN timeout; go blocking again for sends.
        self._sock.settimeout(_CONNECT_TIMEOUT)
        self._sock.connect((request.ip, request.port))
        self._sock.settimeout(None)

        self._av = av
        self._encoder = av.CodecContext.create("libx265" if request.hevc else "libx264", "w")
        self._encoder.width = request.width
        self._encoder.height = request.height
        self._encoder.framerate = request.fps
        self._encoder.bit_rate = request.bitrate
        self._encoder.pix_fmt = "yuv420p"
        # zerolatency + ultrafast keep the glass-to-glass delay teleop-usable;
        # the short keyframe interval lets the headset recover quickly if it
        # joins late or drops a packet.
        self._encoder.options = {
            "tune": "zerolatency",
            "preset": "ultrafast",
            "x264-params": "keyint=15:min-keyint=15",
        }
        self._encoder.open()

    @property
    def closed(self) -> bool:
        """True once the stream is stopping — streaming is over and sends are no-ops."""
        return self._stop.is_set()

    def send(self, rgb: npt.NDArray[np.uint8]) -> None:
        """Encode one RGB frame and write its packets to the headset."""
        if self._stop.is_set():
            return
        height, width = rgb.shape[:2]
        if (width, height) != (self._request.width, self._request.height) and not self._warned_resize:
            self._log.warning(
                f"rescaling {width}x{height} frames to the {self._request.width}x{self._request.height} "
                "the headset requested; match your camera resolution to skip the rescale"
            )
            self._warned_resize = True

        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24")
        # One swscale pass does both the colour conversion and any rescale.
        frame = frame.reformat(width=self._request.width, height=self._request.height, format="yuv420p")
        frame.pts = None  # let the encoder stamp presentation times
        try:
            packets = self._encoder.encode(frame)
        except self._av.FFmpegError as exc:
            self._log.warning(f"PICO video encode failed, closing the stream: {exc}")
            self._stop.set()
            return
        self._write(packets)
        self.frames_sent += 1

    def stop(self) -> None:
        """Signal the encode loop to stop; does NOT release the encoder or socket.

        Cheap and thread-safe. The link calls this before joining the encode
        thread so ``close()`` can flush without racing a live ``encode()``.
        """
        self._stop.set()

    def close(self) -> None:
        """Flush the encoder and close the socket (idempotent).

        Call only after the encode thread has been joined — ``encode(None)`` here
        would otherwise race the encode thread's ``encode(frame)``.
        """
        self._stop.set()
        if self._released:
            return
        self._released = True
        try:
            self._write(self._encoder.encode(None))  # flush buffered packets
        except (OSError, self._av.FFmpegError) as exc:
            self._log.warning(f"PICO video flush failed: {exc}")
        with self._lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _write(self, packets) -> None:
        for packet in packets:
            payload = bytes(packet)
            if not payload:
                continue
            try:
                with self._lock:
                    if self._sock is None:
                        return
                    self._sock.sendall(P.pack_video_frame(payload))
            except OSError as exc:
                self._log.warning(f"PICO video send failed, closing the stream: {exc}")
                self._stop.set()
                return
