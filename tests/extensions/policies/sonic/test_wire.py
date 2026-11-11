"""``eden.extensions.policies.sonic.wire``: the packed-channel byte layout, the message builders/parsers, and g1_debug."""

from __future__ import annotations

import json
import struct
import time

import numpy as np
import pytest

from eden.extensions.policies.sonic import wire
from eden.extensions.policies.sonic.teleop_policy import (
    UPPER_BODY_IL_INDICES,
    VR_3POINT_DEFAULT_ORIENTATION,
    VR_3POINT_DEFAULT_POSITION,
)
from eden.options.robots.unitree_g1 import (
    G1_DEX3_LEFT_DOFS_NAME,
    G1_DEX3_RIGHT_DOFS_NAME,
    G1_DOFS29_NAME,
)
from tests.support.sonic import bind_debug_publisher


def _golden_header(fields: list[dict], version: int = 1, count: int = 1) -> bytes:
    payload = json.dumps({"v": version, "endian": "le", "count": count, "fields": fields}, separators=(",", ":"))
    return payload.encode("utf-8").ljust(wire.HEADER_SIZE, b"\x00")


def _pose_fields(n=5, **overrides):
    base = {
        "smpl_pose": np.zeros((n, 21, 3), dtype=np.float32),
        "smpl_joints": np.zeros((n, 24, 3), dtype=np.float32),
        "body_quat_w": np.zeros((n, 4), dtype=np.float32),
        "joint_pos": np.zeros((n, 29)),
        "joint_vel": np.zeros((n, 29)),
        "frame_index": np.arange(n, dtype=np.int64),
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def _parse(fields, version):
    return wire.parse_pose(wire.unpack_message(wire.pack_pose_message(fields, version=version), wire.POSE_TOPIC))


def test_command_message_bytes_match_spec():
    expected = (
        b"command"
        + _golden_header(
            [
                {"name": "start", "dtype": "u8", "shape": [1]},
                {"name": "stop", "dtype": "u8", "shape": [1]},
                {"name": "planner", "dtype": "u8", "shape": [1]},
            ]
        )
        + struct.pack("B", 1)
        + struct.pack("B", 0)
        + struct.pack("B", 1)
    )
    assert wire.build_command_message(start=True, stop=False, planner=True) == expected


def test_command_message_bytes_with_delta_heading():
    expected = (
        b"command"
        + _golden_header(
            [
                {"name": "start", "dtype": "u8", "shape": [1]},
                {"name": "stop", "dtype": "u8", "shape": [1]},
                {"name": "planner", "dtype": "u8", "shape": [1]},
                {"name": "delta_heading", "dtype": "f32", "shape": [1]},
            ]
        )
        + struct.pack("B", 0)
        + struct.pack("B", 1)
        + struct.pack("B", 0)
        + struct.pack("<f", 0.25)
    )
    assert wire.build_command_message(start=False, stop=True, planner=False, delta_heading=0.25) == expected


def test_planner_message_bytes_match_spec():
    upper = [0.1 * i for i in range(17)]
    expected_fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
        {"name": "upper_body_position", "dtype": "f32", "shape": [17]},
        {"name": "vr_position", "dtype": "f32", "shape": [9]},
        {"name": "vr_orientation", "dtype": "f32", "shape": [12]},
    ]
    payload = struct.pack("<i", 2)
    payload += struct.pack("<fff", 0.5, -0.25, 0.0)
    payload += struct.pack("<fff", 1.0, 0.0, 0.0)
    payload += struct.pack("<f", -1.0)
    payload += struct.pack("<f", -1.0)
    for value in upper:
        payload += struct.pack("<f", value)
    for value in range(9):
        payload += struct.pack("<f", float(value))
    for value in range(12):
        payload += struct.pack("<f", float(value) * 0.5)  # exact in float32
    expected = b"planner" + _golden_header(expected_fields) + payload

    msg = wire.build_planner_message(
        mode=2,
        movement=[0.5, -0.25, 0.0],
        facing=[1.0, 0.0, 0.0],
        upper_body_position=upper,
        vr_3pt_position=np.arange(9, dtype=np.float32),
        vr_3pt_orientation=np.arange(12, dtype=np.float32) * np.float32(0.5),
    )
    assert msg == expected


def test_pose_message_bytes_match_spec():
    smpl_pose = np.arange(5 * 21 * 3, dtype=np.float32).reshape(5, 21, 3)
    joint_pos = np.zeros((5, 29), dtype=np.float64)
    frame_index = np.arange(5, dtype=np.int64)
    toggle = np.array([True])
    expected = (
        b"pose"
        + _golden_header(
            [
                {"name": "smpl_pose", "dtype": "f32", "shape": [5, 21, 3]},
                {"name": "joint_pos", "dtype": "f64", "shape": [5, 29]},
                {"name": "frame_index", "dtype": "i64", "shape": [5]},
                {"name": "toggle_data_collection", "dtype": "bool", "shape": [1]},
            ],
            version=3,
        )
        + smpl_pose.tobytes()
        + joint_pos.tobytes()
        + frame_index.tobytes()
        + toggle.tobytes()
    )
    msg = wire.pack_pose_message(
        {
            "smpl_pose": smpl_pose,
            "joint_pos": joint_pos,
            "frame_index": frame_index,
            "toggle_data_collection": toggle,
        }
    )
    assert msg == expected


def test_pose_message_coerces_non_wire_dtypes_to_f32():
    # The reference packer passes f32/f64/i32/i64/bool through and casts the rest.
    msg = wire.pack_pose_message({"weird": np.arange(3, dtype=np.uint16), "skipped": [1, 2, 3]})
    parsed = wire.unpack_message(msg, wire.POSE_TOPIC)
    assert parsed.fields["weird"].dtype == np.float32
    assert "skipped" not in parsed.fields  # non-ndarray entries are dropped


def test_header_is_exactly_1280_bytes_and_nul_padded():
    msg = wire.build_command_message(True, False, True)
    header = msg[len(b"command") : len(b"command") + wire.HEADER_SIZE]
    assert len(header) == 1280
    json_part = header.split(b"\x00", 1)[0]
    assert header == json_part.ljust(1280, b"\x00")
    json.loads(json_part)  # valid JSON


def test_oversized_header_raises():
    fields = {f"field_with_a_very_long_name_{i:04d}": np.zeros(1, dtype=np.float32) for i in range(40)}
    with pytest.raises(ValueError, match="header too large"):
        wire.pack_message("pose", fields, version=3)


def test_command_round_trip():
    msg = wire.build_command_message(start=True, stop=False, planner=True, delta_heading=-0.1)
    cmd = wire.parse_command(wire.unpack_message(msg, wire.COMMAND_TOPIC))
    assert (cmd.start, cmd.stop, cmd.planner) == (True, False, True)
    assert cmd.delta_heading == pytest.approx(-0.1)

    cmd = wire.parse_command(wire.unpack_message(wire.build_command_message(False, True, False), wire.COMMAND_TOPIC))
    assert (cmd.start, cmd.stop, cmd.planner) == (False, True, False)
    assert cmd.delta_heading is None


@pytest.mark.parametrize("with_optional", [False, True])
def test_planner_round_trip(with_optional):
    kwargs = {}
    if with_optional:
        kwargs = {
            "upper_body_position": np.linspace(-1, 1, 17),
            "left_hand_position": np.full(7, 0.3),
            "right_hand_position": np.full(7, -0.3),
            "vr_3pt_position": np.arange(9),
            "vr_3pt_orientation": np.arange(12),
            "vr_3pt_compliance": [0.5, 0.5, 0.0],
        }
    msg = wire.build_planner_message(mode=3, movement=[1, 0, 0], facing=[0, 1, 0], speed=2.0, height=0.7, **kwargs)
    planner = wire.parse_planner(wire.unpack_message(msg, wire.PLANNER_TOPIC))
    assert planner.mode == 3
    np.testing.assert_allclose(planner.movement, [1, 0, 0])
    np.testing.assert_allclose(planner.facing, [0, 1, 0])
    assert planner.speed == pytest.approx(2.0)
    assert planner.height == pytest.approx(0.7)
    if with_optional:
        np.testing.assert_allclose(planner.upper_body_position, np.linspace(-1, 1, 17), rtol=1e-6)
        np.testing.assert_allclose(planner.left_hand_joints, 0.3)
        np.testing.assert_allclose(planner.vr_position, np.arange(9))
        np.testing.assert_allclose(planner.vr_compliance, [0.5, 0.5, 0.0])
        assert planner.upper_body_velocity is None
    else:
        assert planner.upper_body_position is None
        assert planner.vr_position is None


def test_pose_round_trip_preserves_dtypes_and_shapes():
    n = 5
    pose_data = {
        "smpl_pose": np.random.default_rng(0).normal(size=(n, 21, 3)).astype(np.float32),
        "smpl_joints": np.random.default_rng(1).normal(size=(n, 24, 3)).astype(np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (n, 1)),
        "joint_pos": np.random.default_rng(2).normal(size=(n, 29)),
        "joint_vel": np.zeros((n, 29)),
        "vr_position": np.arange(9, dtype=np.float32),
        "vr_orientation": np.zeros(12, dtype=np.float32),
        "frame_index": np.arange(100, 100 + n, dtype=np.int64),
        "left_trigger": np.array([0.7], dtype=np.float32),
        "heading_increment": np.array([0.02], dtype=np.float32),
        "toggle_data_collection": np.array([True]),
        "toggle_data_abort": np.array([False]),
        "left_hand_joints": np.zeros(7, dtype=np.float32),
        "right_hand_joints": np.ones(7, dtype=np.float32),
    }
    chunk = wire.parse_pose(wire.unpack_message(wire.pack_pose_message(pose_data), wire.POSE_TOPIC))
    assert chunk.version == 3
    assert chunk.joint_pos.dtype == np.float64
    assert chunk.frame_index.dtype == np.int64
    np.testing.assert_array_equal(chunk.frame_index, pose_data["frame_index"])
    np.testing.assert_allclose(chunk.smpl_pose, pose_data["smpl_pose"])
    np.testing.assert_allclose(chunk.joint_pos, pose_data["joint_pos"])
    np.testing.assert_array_equal(chunk.body_quat, pose_data["body_quat_w"])
    assert chunk.left_trigger == pytest.approx(0.7)
    assert chunk.heading_increment == pytest.approx(0.02)
    assert chunk.toggle_data_collection is True
    assert chunk.toggle_data_abort is False
    np.testing.assert_allclose(chunk.right_hand_joints, 1.0)


def test_manager_state_round_trip():
    msg = wire.build_manager_state_message(stream_mode=5, toggle_data_collection=True, toggle_data_abort=False)
    parsed = wire.unpack_message(msg, wire.MANAGER_STATE_TOPIC)
    assert parsed.fields["stream_mode"].reshape(-1)[0] == 5
    assert bool(parsed.fields["toggle_data_collection"][0]) is True
    assert bool(parsed.fields["toggle_data_abort"][0]) is False


def test_unpack_rejects_wrong_topic_and_truncation():
    msg = wire.build_command_message(True, False, True)
    with pytest.raises(ValueError, match="topic"):
        wire.unpack_message(msg, "planner")
    with pytest.raises(ValueError, match="too small"):
        wire.unpack_message(b"command" + b"\x00" * 10, "command")
    # Header declares more bytes than the payload carries.
    truncated = msg[:-1]
    with pytest.raises(ValueError, match="exceeds payload bounds"):
        wire.unpack_message(truncated, "command")


def test_unpack_ignores_trailing_bytes():
    msg = wire.build_command_message(True, False, True) + b"extra-bytes"
    cmd = wire.parse_command(wire.unpack_message(msg, wire.COMMAND_TOPIC))
    assert cmd.start is True


def test_unpack_big_endian_payload():
    values = np.array([1.5, -2.5, 3.5], dtype=">f4")
    header = json.dumps(
        {"v": 1, "endian": "be", "count": 1, "fields": [{"name": "x", "dtype": "f32", "shape": [3]}]},
        separators=(",", ":"),
    ).encode("utf-8")
    msg = b"pose" + header.ljust(wire.HEADER_SIZE, b"\x00") + values.tobytes()
    parsed = wire.unpack_message(msg, "pose")
    np.testing.assert_allclose(parsed.fields["x"], [1.5, -2.5, 3.5])
    assert parsed.fields["x"].dtype.byteorder in ("=", "<", "|")


def test_unpack_rejects_unknown_dtype():
    header = json.dumps(
        {"v": 1, "endian": "le", "count": 1, "fields": [{"name": "x", "dtype": "c64", "shape": [1]}]},
        separators=(",", ":"),
    ).encode("utf-8")
    msg = b"pose" + header.ljust(wire.HEADER_SIZE, b"\x00") + b"\x00" * 8
    with pytest.raises(ValueError, match="unsupported wire dtype"):
        wire.unpack_message(msg, "pose")


def test_receive_side_sizes_mirror_the_robot_tables():
    """The wire module keeps its field sizes literal (no robot imports); they must match what the robot consumes."""
    assert wire._UPPER_BODY_SIZE == len(UPPER_BODY_IL_INDICES)
    assert wire._HAND_SIZE == len(G1_DEX3_LEFT_DOFS_NAME) == len(G1_DEX3_RIGHT_DOFS_NAME)
    assert wire._POSE_JOINTS == len(G1_DOFS29_NAME)
    assert (wire._VR_POSITION_SIZE, wire._VR_ORIENTATION_SIZE) == (
        VR_3POINT_DEFAULT_POSITION.size,
        VR_3POINT_DEFAULT_ORIENTATION.size,
    )


def test_a_0d_field_round_trips_as_shape_1():
    """Shape [] decodes as zero elements, so a 0-d field must never be written with it — it goes out as [1]."""
    msg = wire.pack_message("command", {"stop": np.array(1, np.uint8), "start": np.array([0], np.uint8)}, version=1)
    fields = wire.unpack_message(msg, "command").fields
    assert fields["stop"].tolist() == [1] and fields["start"].tolist() == [0]  # later fields not shifted


@pytest.mark.parametrize(
    "header, match",
    [
        ([], "header must be a JSON object"),
        ({"endian": "little", "fields": []}, "endian 'little'"),
        ({"fields": {"name": "x"}}, "'fields' must be a list"),
        ({"fields": [{"name": "x", "shape": [1]}]}, "needs 'name', 'dtype' and 'shape'"),
        ({"fields": [{"name": "x", "dtype": "f32", "shape": [-1]}]}, "non-negative ints"),
        ({"v": "1", "fields": []}, "'v'/'count' must be ints"),
    ],
)
def test_unpack_reports_a_malformed_header_as_value_error(header, match):
    """The receive loop drops what unpack rejects; KeyError/AttributeError escaped the documented contract."""
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    msg = b"pose" + raw.ljust(wire.HEADER_SIZE, b"\x00") + b"\x00" * 8
    with pytest.raises(ValueError, match=match):
        wire.unpack_message(msg, "pose")


def test_pack_message_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        wire.pack_message("pose", {"x": np.zeros(2, dtype=np.complex64)}, version=1)


def test_pose_versions_required_fields():
    # v3: everything required.
    assert _parse(_pose_fields(), 3).version == 3
    for missing in ("body_quat_w", "frame_index", "smpl_joints", "smpl_pose", "joint_pos", "joint_vel"):
        with pytest.raises(ValueError, match="requires"):
            _parse(_pose_fields(**{missing: None}), 3)
    # v2: joint data optional.
    assert _parse(_pose_fields(joint_pos=None, joint_vel=None), 2).version == 2
    # v1: SMPL data optional.
    assert _parse(_pose_fields(smpl_joints=None, smpl_pose=None), 1).version == 1
    with pytest.raises(ValueError, match="joint_pos"):
        _parse(_pose_fields(smpl_joints=None, smpl_pose=None, joint_pos=None), 1)
    # 'body_quat' is accepted as an alias for 'body_quat_w'.
    aliased = _pose_fields(body_quat_w=None, body_quat=np.zeros((5, 4), dtype=np.float32))
    assert _parse(aliased, 3).body_quat.shape == (5, 4)


def test_pose_v4_token_only():
    chunk = _parse({"token_state": np.zeros(64, dtype=np.float32)}, 4)
    assert chunk.version == 4
    assert chunk.token_state.shape == (64,)
    assert chunk.frame_index is None  # optional for v4
    with pytest.raises(ValueError, match="token_state"):
        _parse(_pose_fields(), 4)


def test_pose_catch_up_field():
    assert _parse(_pose_fields(), 3).catch_up is True  # absent -> enabled
    chunk = _parse(_pose_fields(catch_up=np.array([False])), 3)
    assert chunk.catch_up is False


def test_command_with_empty_required_field_raises_value_error():
    # An empty shape decodes to zero elements; required scalars must fail as
    # ValueError (dropped by the receive thread), never IndexError.
    header = json.dumps(
        {
            "v": 1,
            "endian": "le",
            "count": 1,
            "fields": [
                {"name": "start", "dtype": "u8", "shape": []},
                {"name": "stop", "dtype": "u8", "shape": [1]},
                {"name": "planner", "dtype": "u8", "shape": [1]},
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    msg = b"command" + header.ljust(wire.HEADER_SIZE, b"\x00") + b"\x00\x01"
    with pytest.raises(ValueError, match="empty required field"):
        wire.parse_command(wire.unpack_message(msg, wire.COMMAND_TOPIC))


def test_pose_unsupported_version():
    with pytest.raises(ValueError, match="unsupported pose protocol version"):
        _parse(_pose_fields(), 7)


def test_g1_debug_publish_poll_loopback():
    publisher, free_port = bind_debug_publisher()
    poller = wire.G1DebugPoller(host="127.0.0.1", port=free_port)
    try:
        payload = {
            "index": 7,
            "body_q_measured": np.linspace(-1.0, 1.0, 29),
            "left_hand_q_measured": np.zeros(7),
            "base_quat": [1.0, 0.0, 0.0, 0.0],
        }
        received = None
        # PUB/SUB needs the subscription to propagate; retry until delivery.
        for _ in range(100):
            publisher.publish(payload)
            time.sleep(0.02)
            received = poller.get_latest()
            if received is not None:
                break
        assert received is not None, "g1_debug message never arrived over loopback"
        assert received["index"] == 7
        np.testing.assert_allclose(received["body_q_measured"], np.linspace(-1.0, 1.0, 29))
        assert received["left_hand_q_measured"] == [0.0] * 7
        assert received["base_quat"] == [1.0, 0.0, 0.0, 0.0]
    finally:
        publisher.close()
        poller.close()
