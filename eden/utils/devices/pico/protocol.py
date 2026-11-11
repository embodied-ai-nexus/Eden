"""Wire protocol of the Eden <-> PICO (XRoboToolkit) *Remote Vision* video link.

The XRoboToolkit PICO app drives both directions; Eden is the server on the
control channel and the client on the video channel:

  * **control** (headset -> Eden), TCP ``:13579``::

        [4B BE uint32 body_len][body]

    where ``body`` is a little-endian ``NetworkDataProtocol`` record::

        [4B LE int32 cmd_len][command][4B LE int32 data_len][data]

    The app sends ``OPEN_CAMERA`` (``data`` = a ``CameraRequestData`` payload,
    see below) when the operator taps *Listen*, and ``CLOSE_CAMERA`` (no data)
    when they stop. Eden acknowledges each command with the same framing,
    echoing the command name and an empty data field.

  * **video** (Eden -> headset), TCP to the ``ip``/``port`` the ``OPEN_CAMERA``
    request names — that is where the app's ``MediaDecoder`` listens::

        [4B BE uint32 frame_size][H.264 / H.265 NAL data]

``CameraRequestData`` (the ``OPEN_CAMERA`` payload) is::

        [0xCA 0xFE]        magic
        [0x01]             version
        [7 x 4B LE int32]  width, height, fps, bitrate, enable_mv_hevc,
                           render_mode, port
        [1B len + bytes]   camera   (the app's source profile, e.g. "ZED")
        [1B len + bytes]   ip       (the headset's own address)

This module is pure bytes <-> objects (stdlib only), so the framing is
unit-testable without a headset; the sockets and the encoder live in
:mod:`eden.utils.devices.pico.video`.

Ported from SIMPLE's ``src/simple/teleop/pico/protocol.py`` (MIT,
https://github.com/physical-superintelligence-lab/SIMPLE), which is where the
byte layout above was reverse-engineered from the PICO app.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

CONTROL_PORT = 13579  # TCP port the PICO app connects to for OPEN/CLOSE_CAMERA
MAGIC = b"\xca\xfe"  # CameraRequestData magic bytes
VERSION = 1  # CameraRequestData layout version
OPEN_CAMERA = "OPEN_CAMERA"
CLOSE_CAMERA = "CLOSE_CAMERA"

_FRAME_HEADER = ">I"  # both channels prefix their payload with a big-endian length
HEADER_SIZE = struct.calcsize(_FRAME_HEADER)


@dataclass(frozen=True)
class CameraRequest:
    """A decoded ``OPEN_CAMERA`` payload: what the headset wants and where to send it.

    ``width``/``height`` are the *full* side-by-side frame size (both eyes), and
    ``ip``/``port`` address the decoder listening inside the PICO app.
    """

    width: int
    height: int
    fps: int
    bitrate: int
    hevc: bool  # the request's enableMvHevc flag: encode H.265 instead of H.264
    render_mode: int
    port: int
    camera: str
    ip: str


# ----------------------------------------------------------------------------
# control channel framing
# ----------------------------------------------------------------------------


def encode_message(command: str, data: bytes = b"") -> bytes:
    """Frame one control message (length header included) ready to ``sendall``."""
    cmd = command.encode("utf-8")
    body = struct.pack("<i", len(cmd)) + cmd + struct.pack("<i", len(data)) + data
    return struct.pack(_FRAME_HEADER, len(body)) + body


def decode_message(body: bytes) -> tuple[str, bytes]:
    """Split a control-message body into ``(command, data)``.

    ``body`` is the payload *after* the 4-byte length header (see
    :func:`unpack_length`). Raises ``ValueError`` on a malformed record — the
    caller should log it and keep reading rather than tear the connection down,
    since the next message may be fine.
    """
    if len(body) < 8:
        raise ValueError(f"control body too small: {len(body)} bytes (need >= 8)")

    (cmd_len,) = struct.unpack_from("<i", body, 0)
    if cmd_len < 0 or 4 + cmd_len > len(body):
        raise ValueError(f"invalid command length {cmd_len} in a {len(body)}-byte body")
    command = body[4 : 4 + cmd_len].decode("utf-8").rstrip("\x00")

    offset = 4 + cmd_len
    if offset + 4 > len(body):
        raise ValueError(f"command {command!r} is missing its data-length field")
    (data_len,) = struct.unpack_from("<i", body, offset)
    offset += 4
    if data_len < 0 or offset + data_len > len(body):
        raise ValueError(f"invalid data length {data_len} for command {command!r}")

    return command, body[offset : offset + data_len]


def unpack_length(header: bytes) -> int:
    """Decode a 4-byte big-endian length header (control body or video frame)."""
    if len(header) != HEADER_SIZE:
        raise ValueError(f"length header must be {HEADER_SIZE} bytes, got {len(header)}")
    return int(struct.unpack(_FRAME_HEADER, header)[0])


# ----------------------------------------------------------------------------
# CameraRequestData payload
# ----------------------------------------------------------------------------


def decode_camera_request(data: bytes) -> CameraRequest:
    """Decode an ``OPEN_CAMERA`` payload. Raises ``ValueError`` if malformed."""
    if len(data) < 3 + 28:
        raise ValueError(f"camera request too small: {len(data)} bytes (need >= 31)")
    if data[:2] != MAGIC:
        raise ValueError(f"bad camera-request magic: {data[:2].hex()} (expected {MAGIC.hex()})")
    if data[2] != VERSION:
        raise ValueError(f"unsupported camera-request version: {data[2]} (expected {VERSION})")

    width, height, fps, bitrate, enable_mv_hevc, render_mode, port = struct.unpack_from("<7i", data, 3)
    camera, offset = _decode_string(data, 3 + 28)
    ip, offset = _decode_string(data, offset)
    return CameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        hevc=bool(enable_mv_hevc),
        render_mode=render_mode,
        port=port,
        camera=camera,
        ip=ip,
    )


def encode_camera_request(request: CameraRequest) -> bytes:
    """Encode a :class:`CameraRequest` back into its wire payload.

    Eden never sends this — the PICO app does. It exists so tests (and a fake
    headset client) can exercise the decoder against the documented layout.
    """
    head = MAGIC + bytes([VERSION])
    ints = struct.pack(
        "<7i",
        request.width,
        request.height,
        request.fps,
        request.bitrate,
        int(request.hevc),
        request.render_mode,
        request.port,
    )
    return head + ints + _encode_string(request.camera) + _encode_string(request.ip)


def _decode_string(buf: bytes, pos: int) -> tuple[str, int]:
    """Read one ``[1B length][utf-8 bytes]`` string, returning it and the new offset."""
    if pos >= len(buf):
        raise ValueError("truncated camera request: no room for a string length")
    length = buf[pos]
    pos += 1
    if pos + length > len(buf):
        raise ValueError(f"truncated camera request: string of {length} bytes runs past the payload")
    return buf[pos : pos + length].decode("utf-8"), pos + length


def _encode_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) > 255:
        raise ValueError(f"string is {len(raw)} bytes; the 1-byte length prefix caps it at 255")
    return bytes([len(raw)]) + raw


# ----------------------------------------------------------------------------
# video channel framing
# ----------------------------------------------------------------------------


def pack_video_frame(nal: bytes) -> bytes:
    """Frame one encoded video packet for the headset's decoder socket."""
    return struct.pack(_FRAME_HEADER, len(nal)) + nal
