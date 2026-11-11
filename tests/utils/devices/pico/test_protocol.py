"""``eden.utils.devices.pico.protocol``: the XRoboToolkit control/video framing, round-tripped and fuzzed."""

from __future__ import annotations

import struct

import pytest

from eden.utils.devices.pico import protocol as P

CAMERA_REQUEST = P.CameraRequest(
    width=640,
    height=240,
    fps=30,
    bitrate=1_000_000,
    hevc=False,
    render_mode=0,
    port=0,  # filled in per test with the fake decoder's port
    camera="ZED",
    ip="127.0.0.1",
)


@pytest.mark.parametrize(
    "command, data",
    [
        (P.CLOSE_CAMERA, b""),
        (P.OPEN_CAMERA, b"\x00\x01\x02payload"),
        ("A" * 200, bytes(range(256))),
    ],
)
def test_control_message_round_trip(command, data):
    framed = P.encode_message(command, data)
    body_len = P.unpack_length(framed[: P.HEADER_SIZE])
    body = framed[P.HEADER_SIZE :]
    assert body_len == len(body)
    assert P.decode_message(body) == (command, data)


def test_decode_message_strips_null_padding():
    """The app pads short command strings with NULs; they must not reach the caller."""
    body = struct.pack("<i", 8) + b"OPEN\x00\x00\x00\x00" + struct.pack("<i", 0)
    assert P.decode_message(body) == ("OPEN", b"")


@pytest.mark.parametrize(
    "body, reason",
    [
        (b"", "too small"),
        (b"\x00" * 7, "too small"),
        (struct.pack("<i", -1) + b"\x00" * 8, "invalid command length"),
        (struct.pack("<i", 999) + b"\x00" * 8, "invalid command length"),
        (struct.pack("<i", 4) + b"OPEN" + b"\x00\x00", "missing its data-length field"),
        (struct.pack("<i", 4) + b"OPEN" + struct.pack("<i", 12), "invalid data length"),
        (struct.pack("<i", 4) + b"OPEN" + struct.pack("<i", -3), "invalid data length"),
    ],
)
def test_decode_message_rejects_malformed(body, reason):
    with pytest.raises(ValueError, match=reason):
        P.decode_message(body)


def test_unpack_length_requires_four_bytes():
    with pytest.raises(ValueError, match="must be 4 bytes"):
        P.unpack_length(b"\x00\x00")


def test_camera_request_round_trip():
    payload = P.encode_camera_request(CAMERA_REQUEST)
    assert payload[:2] == P.MAGIC
    assert payload[2] == P.VERSION
    assert P.decode_camera_request(payload) == CAMERA_REQUEST


def test_camera_request_layout_is_the_documented_one():
    """Decode a payload built byte-by-byte from the spec, not from our encoder."""
    payload = (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 2560, 720, 60, 4_000_000, 1, 2, 12345)
        + bytes([3])
        + b"ZED"
        + bytes([11])
        + b"192.168.1.7"
    )
    request = P.decode_camera_request(payload)
    assert (request.width, request.height, request.fps, request.bitrate) == (2560, 720, 60, 4_000_000)
    assert request.hevc is True  # enableMvHevc = 1
    assert (request.render_mode, request.port) == (2, 12345)
    assert (request.camera, request.ip) == ("ZED", "192.168.1.7")


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b"\xca\xfe\x01", "too small"),
        (b"\xde\xad\x01" + b"\x00" * 28, "magic"),
        (b"\xca\xfe\x09" + b"\x00" * 28, "version"),
        (b"\xca\xfe\x01" + struct.pack("<7i", *([0] * 7)), "no room for a string length"),
        (b"\xca\xfe\x01" + struct.pack("<7i", *([0] * 7)) + bytes([9]) + b"ZED", "runs past the payload"),
    ],
)
def test_decode_camera_request_rejects_malformed(payload, reason):
    with pytest.raises(ValueError, match=reason):
        P.decode_camera_request(payload)


def test_encode_camera_request_rejects_overlong_strings():
    with pytest.raises(ValueError, match="caps it at 255"):
        P.encode_camera_request(CAMERA_REQUEST.__class__(**{**CAMERA_REQUEST.__dict__, "ip": "x" * 256}))


def test_pack_video_frame_prefixes_a_big_endian_length():
    nal = b"\x00\x00\x00\x01\x67payload"
    framed = P.pack_video_frame(nal)
    assert P.unpack_length(framed[: P.HEADER_SIZE]) == len(nal)
    assert framed[P.HEADER_SIZE :] == nal
