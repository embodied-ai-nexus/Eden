"""Wire protocol for SONIC whole-body teleoperation, byte-compatible with GEAR-SONIC.

Two channels connect the operator station to the robot process:

* **Packed channel** (operator → robot, PUB/SUB, default port 5556). Each ZMQ
  message is a single part::

      [topic bytes][1280-byte NUL-padded JSON header][concatenated LE binary fields]

  The header lists field names/dtypes/shapes so the payload can be decoded
  without out-of-band schema knowledge. Topics: ``command`` (start/stop/mode),
  ``planner`` (locomotion commands), ``pose`` (streamed SMPL motion windows),
  ``manager_state`` (operator state, consumed by recorders only).

* **Debug channel** (robot → operator, PUB/SUB, default port 5557). Single-part
  ``[topic bytes][msgpack map]`` messages on topic ``g1_debug`` carrying the
  robot state the operator needs for calibration and frozen-upper capture.

Both layouts match the reference stack (``zmq_planner_sender.py`` /
``zmq_packed_message_subscriber.hpp`` / ``zmq_output_handler.hpp``) byte for
byte, so either side can interoperate with the reference counterpart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

HEADER_SIZE = 1280

DEFAULT_PACKED_PORT = 5556
DEFAULT_DEBUG_PORT = 5557
DEBUG_TOPIC = "g1_debug"

COMMAND_TOPIC = "command"
PLANNER_TOPIC = "planner"
POSE_TOPIC = "pose"
MANAGER_STATE_TOPIC = "manager_state"

# Header "endian" value -> numpy byte order. The builders always write "le".
_ENDIAN_TO_BYTE_ORDER = {"le": "<", "be": ">"}

# Wire dtype string <-> numpy dtype (little-endian on the wire).
_WIRE_TO_NP = {
    "f64": "<f8",
    "f32": "<f4",
    "f16": "<f2",
    "i64": "<i8",
    "i32": "<i4",
    "i16": "<i2",
    "i8": "i1",
    "u8": "u1",
    "bool": "?",
}
_NP_TO_WIRE = {
    np.dtype(np.float64): "f64",
    np.dtype(np.float32): "f32",
    np.dtype(np.float16): "f16",
    np.dtype(np.int64): "i64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int16): "i16",
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.bool_): "bool",
}


# ---------------------------------------------------------------------------
# Packed channel: generic pack / unpack
# ---------------------------------------------------------------------------


def _build_header(fields: list[dict], version: int, count: int = 1) -> bytes:
    """Serialize the field table to the fixed-size NUL-padded JSON header."""
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"packed header too large: {len(header_json)} > {HEADER_SIZE} bytes")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def _as_wire_array(name: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    """Return the wire dtype string and a contiguous little-endian view of ``value``."""
    arr = np.asarray(value)
    wire_dtype = _NP_TO_WIRE.get(arr.dtype.newbyteorder("="))
    if wire_dtype is None:
        raise ValueError(f"field {name!r} has unsupported dtype {arr.dtype} (supported: {sorted(_WIRE_TO_NP)})")
    return wire_dtype, np.ascontiguousarray(arr, dtype=_WIRE_TO_NP[wire_dtype])


def pack_message(topic: str, fields: dict[str, np.ndarray], version: int, count: int = 1) -> bytes:
    """Pack named arrays into one packed-channel message.

    Fields are written in dict order; every value must already have a
    wire-supported dtype (use :func:`pack_pose_message` for the reference's
    lenient float32 coercion).
    """
    header_fields = []
    payload = []
    for name, value in fields.items():
        wire_dtype, arr = _as_wire_array(name, value)  # ascontiguousarray: a 0-d value goes out as shape [1]
        header_fields.append({"name": name, "dtype": wire_dtype, "shape": list(arr.shape)})
        payload.append(arr.tobytes())
    return topic.encode("utf-8") + _build_header(header_fields, version=version, count=count) + b"".join(payload)


@dataclass
class ParsedMessage:
    """One decoded packed-channel message."""

    topic: str
    version: int
    count: int
    fields: dict[str, np.ndarray]


def _malformed(detail: str) -> ValueError:
    """The error for a header that is valid JSON but not a packed-message header.

    Always ``ValueError`` — :func:`unpack_message`'s one documented error, which the
    receive loop catches to drop the message — even where a wrong type is the cause.
    """
    return ValueError(f"packed-message header {detail}")


def unpack_message(msg: bytes, topic: str) -> ParsedMessage:
    """Decode one packed-channel message published under ``topic``.

    Raises ``ValueError`` on a malformed message (wrong topic prefix, bad
    header, truncated payload); trailing bytes beyond the declared fields are
    ignored, matching the reference subscriber.
    """
    prefix = topic.encode("utf-8")
    if not msg.startswith(prefix):
        raise ValueError(f"message does not start with topic {topic!r}")
    body = msg[len(prefix) :]
    if len(body) < HEADER_SIZE:
        raise ValueError(f"packed message too small: {len(body)} < {HEADER_SIZE} header bytes")

    header_json = body[:HEADER_SIZE].split(b"\x00", 1)[0]
    try:
        header = json.loads(header_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid packed-message header JSON: {e}") from e

    if not isinstance(header, dict):
        raise _malformed(f"must be a JSON object, got {type(header).__name__}")
    endian = header.get("endian", "le")
    byte_order = _ENDIAN_TO_BYTE_ORDER.get(endian)
    if byte_order is None:
        raise _malformed(f"has endian {endian!r}; expected one of {sorted(_ENDIAN_TO_BYTE_ORDER)}")
    field_table = header.get("fields", [])
    if not isinstance(field_table, list):
        raise _malformed(f"'fields' must be a list, got {type(field_table).__name__}")
    data = body[HEADER_SIZE:]

    fields: dict[str, np.ndarray] = {}
    offset = 0
    for f in field_table:
        if not isinstance(f, dict) or not {"name", "dtype", "shape"} <= f.keys():
            raise _malformed(f"field entry {f!r} needs 'name', 'dtype' and 'shape'")
        name, dtype_str, shape = f["name"], f["dtype"], f["shape"]
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ValueError(f"field {name!r} has shape {shape!r}; expected a list of non-negative ints")
        shape = tuple(shape)
        np_dtype = _WIRE_TO_NP.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"field {name!r} has unsupported wire dtype {dtype_str!r}")
        dtype = np.dtype(np_dtype)
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder(byte_order)
        # An empty shape means zero elements on the wire (reference semantics), not a scalar.
        n_elements = int(np.prod(shape, dtype=np.int64)) if shape else 0
        nbytes = n_elements * dtype.itemsize
        if offset + nbytes > len(data):
            raise ValueError(f"field {name!r} exceeds payload bounds ({offset + nbytes} > {len(data)})")
        arr = np.frombuffer(data, dtype=dtype, count=n_elements, offset=offset)
        fields[name] = np.ascontiguousarray(arr.reshape(shape if shape else (0,))).astype(
            dtype.newbyteorder("="), copy=False
        )
        offset += nbytes

    version, count = header.get("v", 0), header.get("count", -1)
    if not (isinstance(version, int) and isinstance(count, int)):
        raise _malformed(f"'v'/'count' must be ints, got v={version!r}, count={count!r}")
    return ParsedMessage(topic=topic, version=version, count=count, fields=fields)


# ---------------------------------------------------------------------------
# Packed channel: reference message builders (byte-identical to the reference)
# ---------------------------------------------------------------------------


def build_command_message(start: bool, stop: bool, planner: bool, delta_heading: float | None = None) -> bytes:
    """Build a ``command`` message: start/stop control and managed-mode select.

    ``planner=True`` selects PLANNER mode on the robot, ``False`` selects
    STREAMED_MOTION. ``delta_heading`` is accepted for wire compatibility but
    the reference consumer ignores it.
    """
    fields = {
        "start": np.array([1 if start else 0], dtype=np.uint8),
        "stop": np.array([1 if stop else 0], dtype=np.uint8),
        "planner": np.array([1 if planner else 0], dtype=np.uint8),
    }
    if delta_heading is not None:
        fields["delta_heading"] = np.array([delta_heading], dtype=np.float32)
    return pack_message(COMMAND_TOPIC, fields, version=1)


def build_planner_message(
    mode: int,
    movement: np.ndarray,
    facing: np.ndarray,
    speed: float = -1.0,
    height: float = -1.0,
    upper_body_position: np.ndarray | None = None,
    upper_body_velocity: np.ndarray | None = None,
    left_hand_position: np.ndarray | None = None,
    right_hand_position: np.ndarray | None = None,
    vr_3pt_position: np.ndarray | None = None,
    vr_3pt_orientation: np.ndarray | None = None,
    vr_3pt_compliance: np.ndarray | None = None,
) -> bytes:
    """Build a ``planner`` message: locomotion mode + stick commands.

    The optional fields are appended only when given; their *presence* is the
    signal the robot keys on (upper-body override, hand targets, VR 3-point
    control). ``speed``/``height`` use ``-1.0`` for "mode default".
    """
    movement = np.asarray(movement, dtype=np.float32).reshape(-1)
    facing = np.asarray(facing, dtype=np.float32).reshape(-1)
    if movement.shape != (3,):
        raise ValueError("movement must have length 3")
    if facing.shape != (3,):
        raise ValueError("facing must have length 3")

    fields: dict[str, np.ndarray] = {
        "mode": np.array([mode], dtype=np.int32),
        "movement": movement,
        "facing": facing,
        "speed": np.array([speed], dtype=np.float32),
        "height": np.array([height], dtype=np.float32),
    }
    optional = {
        "upper_body_position": upper_body_position,
        "upper_body_velocity": upper_body_velocity,
        "left_hand_joints": left_hand_position,
        "right_hand_joints": right_hand_position,
        "vr_position": vr_3pt_position,
        "vr_orientation": vr_3pt_orientation,
        "vr_compliance": vr_3pt_compliance,
    }
    for name, value in optional.items():
        if value is not None:
            fields[name] = np.asarray(value, dtype=np.float32).reshape(-1)
    return pack_message(PLANNER_TOPIC, fields, version=1)


# Dtypes the reference pose packer sends as-is; everything else is cast to f32.
_POSE_PASSTHROUGH_DTYPES = frozenset(np.dtype(t) for t in (np.float32, np.float64, np.int32, np.int64, np.bool_))


def pack_pose_message(pose_data: dict[str, np.ndarray], topic: str = POSE_TOPIC, version: int = 3) -> bytes:
    """Pack a streamed-pose message; non-wire dtypes are cast to float32.

    Mirrors the reference packer: entries that are not numpy arrays are
    skipped, and any dtype outside f32/f64/i32/i64/bool is coerced to float32.
    """
    fields: dict[str, np.ndarray] = {}
    for name, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype.newbyteorder("=") not in _POSE_PASSTHROUGH_DTYPES:
            value = value.astype(np.float32)
        fields[name] = value
    return pack_message(topic, fields, version=version)


def build_manager_state_message(stream_mode: int, toggle_data_collection: bool, toggle_data_abort: bool) -> bytes:
    """Build a ``manager_state`` message (operator mode broadcast for recorders)."""
    fields = {
        "stream_mode": np.array([stream_mode], dtype=np.int32),
        "toggle_data_collection": np.array([toggle_data_collection], dtype=bool),
        "toggle_data_abort": np.array([toggle_data_abort], dtype=bool),
    }
    # version 3: the reference sends this through its pose packer's default.
    return pack_message(MANAGER_STATE_TOPIC, fields, version=3)


# ---------------------------------------------------------------------------
# Packed channel: typed parsers (robot side)
# ---------------------------------------------------------------------------


@dataclass
class CommandMessage:
    """Decoded ``command`` message."""

    start: bool
    stop: bool
    planner: bool
    delta_heading: float | None = None


def _first_finite(value: np.ndarray, name: str, topic: str) -> float:
    scalar = float(value.reshape(-1)[0])
    if not np.isfinite(scalar):
        # A NaN heading_increment would be summed into the robot's delta heading for good,
        # and a NaN flag reads as True.
        raise ValueError(f"{topic} message field {name!r} is {scalar}; the sender produced a non-finite value")
    return scalar


def _required_scalar(f: dict[str, np.ndarray], name: str, topic: str) -> float:
    value = f.get(name)
    if value is None or value.size == 0:
        raise ValueError(f"{topic} message missing or empty required field {name!r}")
    return _first_finite(value, name, topic)


def _scalar(f: dict[str, np.ndarray], name: str, topic: str, default: float = 0.0) -> float:
    value = f.get(name)
    return _first_finite(value, name, topic) if value is not None and value.size else default


# Fixed sizes of the per-message arrays, as the robot consumes them. Kept literal so this
# protocol module needs no robot tables; each mirrors the named source.
_UPPER_BODY_SIZE = 17  # teleop_policy.UPPER_BODY_IL_INDICES
_HAND_SIZE = 7  # unitree_g1.G1_DEX3_LEFT_DOFS_NAME (Dex3 motors per hand)
_VR_POSITION_SIZE = 9  # 3 points x xyz (teleop_policy.VR_3POINT_DEFAULT_POSITION)
_VR_ORIENTATION_SIZE = 12  # 3 points x wxyz (teleop_policy.VR_3POINT_DEFAULT_ORIENTATION)
_POSE_JOINTS = 29  # unitree_g1.G1_DOFS29_NAME (IsaacLab body joints per streamed frame)
_SMPL_JOINTS = 24  # smpl_joints_10frame_step1 slot: 22 body joints + 2 thumb tips
_SMPL_POSE_JOINTS = 21  # SMPL body joints in the axis-angle pose (root excluded)


def _require_finite(value: np.ndarray, name: str, topic: str) -> None:
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{topic} message field {name!r} contains NaN/inf; the sender produced a non-finite value")


def _optional_vector(f: dict[str, np.ndarray], name: str, topic: str, size: int, what: str) -> np.ndarray | None:
    """Return field ``name`` flattened to ``(size,)``, or None when absent.

    Validated here, on the receive thread, so a wrong-size field drops the message
    instead of raising later inside the robot's control loop.
    """
    value = f.get(name)
    if value is None:
        return None
    if value.size != size:
        raise ValueError(
            f"{topic} message field {name!r} has {value.size} elements (shape {tuple(value.shape)}); "
            f"expected {size} ({what})"
        )
    _require_finite(value, name, topic)
    return value.reshape(size)


def _optional_frames(
    f: dict[str, np.ndarray], name: str, num_frames: int, trailing: tuple[int, ...], what: str
) -> np.ndarray | None:
    """Return per-frame pose field ``name`` checked against ``(num_frames, *trailing)``, or None when absent."""
    value = f.get(name)
    if value is None:
        return None
    expected = (num_frames, *trailing)
    if value.shape != expected:
        raise ValueError(
            f"pose message field {name!r} has shape {tuple(value.shape)}; expected {expected} "
            f"({what}; the frame count comes from 'frame_index')"
        )
    _require_finite(value, name, "pose")
    return value


def parse_command(msg: ParsedMessage) -> CommandMessage:
    """Extract the command flags; ``start``/``stop``/``planner`` are required."""
    f = msg.fields
    delta = f.get("delta_heading")
    return CommandMessage(
        start=bool(_required_scalar(f, "start", "command")),
        stop=bool(_required_scalar(f, "stop", "command")),
        planner=bool(_required_scalar(f, "planner", "command")),
        delta_heading=float(delta.reshape(-1)[0]) if delta is not None and delta.size else None,
    )


@dataclass
class PlannerMessage:
    """Decoded ``planner`` message. Optional-field presence carries meaning."""

    mode: int
    movement: np.ndarray
    facing: np.ndarray
    speed: float
    height: float
    upper_body_position: np.ndarray | None = None
    upper_body_velocity: np.ndarray | None = None
    left_hand_joints: np.ndarray | None = None
    right_hand_joints: np.ndarray | None = None
    vr_position: np.ndarray | None = None
    vr_orientation: np.ndarray | None = None
    vr_compliance: np.ndarray | None = None


def parse_planner(msg: ParsedMessage) -> PlannerMessage:
    """Extract a planner command; ``mode``/``movement``/``facing`` are required."""
    f = msg.fields
    mode = int(_required_scalar(f, "mode", "planner"))
    movement = f.get("movement")
    facing = f.get("facing")
    if movement is None or movement.size != 3 or facing is None or facing.size != 3:
        raise ValueError("planner message requires 3-element 'movement' and 'facing' fields")
    _require_finite(movement, "movement", "planner")
    _require_finite(facing, "facing", "planner")
    return PlannerMessage(
        mode=mode,
        movement=movement.reshape(-1).astype(np.float64),
        facing=facing.reshape(-1).astype(np.float64),
        speed=_scalar(f, "speed", "planner", default=-1.0),
        height=_scalar(f, "height", "planner", default=-1.0),
        upper_body_position=_optional_vector(
            f, "upper_body_position", "planner", _UPPER_BODY_SIZE, "IsaacLab upper-body joint targets"
        ),
        upper_body_velocity=_optional_vector(
            f, "upper_body_velocity", "planner", _UPPER_BODY_SIZE, "IsaacLab upper-body joint velocities"
        ),
        left_hand_joints=_optional_vector(f, "left_hand_joints", "planner", _HAND_SIZE, "one target per Dex3 motor"),
        right_hand_joints=_optional_vector(f, "right_hand_joints", "planner", _HAND_SIZE, "one target per Dex3 motor"),
        vr_position=_optional_vector(f, "vr_position", "planner", _VR_POSITION_SIZE, "3 points x xyz"),
        vr_orientation=_optional_vector(f, "vr_orientation", "planner", _VR_ORIENTATION_SIZE, "3 points x wxyz"),
        vr_compliance=f.get("vr_compliance"),
    )


@dataclass
class PoseChunk:
    """Decoded ``pose`` message: a sliding window of streamed motion frames.

    ``frame_index`` aligns overlapping windows; per-version field requirements
    follow the reference consumer (v1: joint data; v2/v3: SMPL data, v3 adds
    joint data; v4: token-only).
    """

    version: int
    frame_index: np.ndarray | None  # (N,) global frame numbers
    body_quat: np.ndarray | None  # (N, 4) root orientation, wxyz
    joint_pos: np.ndarray | None  # (N, 29) IsaacLab order
    joint_vel: np.ndarray | None  # (N, 29)
    smpl_joints: np.ndarray | None  # (N, 24, 3) root-local positions
    smpl_pose: np.ndarray | None  # (N, 21, 3) local axis-angle
    token_state: np.ndarray | None = None  # v4 only
    vr_position: np.ndarray | None = None  # (9,)
    vr_orientation: np.ndarray | None = None  # (12,) wxyz triple
    left_hand_joints: np.ndarray | None = None  # (7,) Dex3 targets
    right_hand_joints: np.ndarray | None = None  # (7,)
    heading_increment: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0
    toggle_data_collection: bool = False
    toggle_data_abort: bool = False
    catch_up: bool = True  # False -> the merger allows unbounded stream lag
    extras: dict[str, np.ndarray] = field(default_factory=dict)


_POSE_KNOWN_FIELDS = frozenset(
    {
        "frame_index", "last_smpl_global_frames", "body_quat", "body_quat_w", "joint_pos", "joint_vel",
        "smpl_joints", "smpl_pose", "token_state", "vr_position", "vr_orientation", "left_hand_joints",
        "right_hand_joints", "heading_increment", "left_trigger", "right_trigger", "left_grip", "right_grip",
        "toggle_data_collection", "toggle_data_abort", "catch_up",
    }
)  # fmt: skip


def parse_pose(msg: ParsedMessage) -> PoseChunk:
    """Extract a pose chunk, enforcing the per-version required fields."""
    f = msg.fields
    version = msg.version
    frame_index = f.get("frame_index", f.get("last_smpl_global_frames"))
    body_quat = f.get("body_quat_w", f.get("body_quat"))

    if version == 4:  # token-only: no per-frame motion data
        if "token_state" not in f:
            raise ValueError("pose protocol v4 requires field 'token_state'")
        frame_index = None if frame_index is None else frame_index.reshape(-1)
        joint_pos = joint_vel = smpl_joints = smpl_pose = None
    elif version in (1, 2, 3):
        if body_quat is None:
            raise ValueError(f"pose protocol v{version} requires field 'body_quat' (or 'body_quat_w')")
        if frame_index is None:
            raise ValueError(f"pose protocol v{version} requires field 'frame_index'")
        if version in (2, 3) and not {"smpl_joints", "smpl_pose"} <= f.keys():
            raise ValueError(f"pose protocol v{version} requires fields 'smpl_joints' and 'smpl_pose'")
        if version in (1, 3) and not {"joint_pos", "joint_vel"} <= f.keys():
            raise ValueError(f"pose protocol v{version} requires fields 'joint_pos' and 'joint_vel'")
        frame_index = frame_index.reshape(-1)
        num_frames = frame_index.shape[0]
        if num_frames == 0:
            raise ValueError(f"pose protocol v{version} message carries zero frames ('frame_index' is empty)")
        if body_quat.shape[0] != num_frames or body_quat.shape[-1] != 4 or body_quat.ndim not in (2, 3):
            raise ValueError(
                f"pose message field 'body_quat' has shape {tuple(body_quat.shape)}; expected "
                f"({num_frames}, 4) or ({num_frames}, bodies, 4) wxyz (the frame count comes from 'frame_index')"
            )
        _require_finite(body_quat, "body_quat", "pose")
        joint_pos = _optional_frames(f, "joint_pos", num_frames, (_POSE_JOINTS,), "IsaacLab joint positions")
        joint_vel = _optional_frames(f, "joint_vel", num_frames, (_POSE_JOINTS,), "IsaacLab joint velocities")
        smpl_joints = _optional_frames(f, "smpl_joints", num_frames, (_SMPL_JOINTS, 3), "root-local SMPL joints")
        smpl_pose = _optional_frames(f, "smpl_pose", num_frames, (_SMPL_POSE_JOINTS, 3), "local axis-angle")
    else:
        raise ValueError(f"unsupported pose protocol version: {version}")

    return PoseChunk(
        version=version,
        frame_index=frame_index,
        body_quat=body_quat,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        smpl_joints=smpl_joints,
        smpl_pose=smpl_pose,
        token_state=f.get("token_state"),
        vr_position=_optional_vector(f, "vr_position", "pose", _VR_POSITION_SIZE, "3 points x xyz"),
        vr_orientation=_optional_vector(f, "vr_orientation", "pose", _VR_ORIENTATION_SIZE, "3 points x wxyz"),
        left_hand_joints=_optional_vector(f, "left_hand_joints", "pose", _HAND_SIZE, "one target per Dex3 motor"),
        right_hand_joints=_optional_vector(f, "right_hand_joints", "pose", _HAND_SIZE, "one target per Dex3 motor"),
        heading_increment=_scalar(f, "heading_increment", "pose"),
        left_trigger=_scalar(f, "left_trigger", "pose"),
        right_trigger=_scalar(f, "right_trigger", "pose"),
        left_grip=_scalar(f, "left_grip", "pose"),
        right_grip=_scalar(f, "right_grip", "pose"),
        toggle_data_collection=bool(_scalar(f, "toggle_data_collection", "pose")),
        toggle_data_abort=bool(_scalar(f, "toggle_data_abort", "pose")),
        catch_up=bool(_scalar(f, "catch_up", "pose", default=1.0)),
        extras={k: v for k, v in f.items() if k not in _POSE_KNOWN_FIELDS},
    )


# ---------------------------------------------------------------------------
# Debug channel (msgpack): robot -> operator
# ---------------------------------------------------------------------------


def _to_plain(value):
    """Convert numpy containers/scalars to msgpack-serializable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} to msgpack")


class G1DebugPublisher:
    """PUB socket for the robot's ``g1_debug`` state stream (msgpack maps).

    Socket options mirror the reference publisher: small send HWM, no linger,
    non-blocking sends (a slow subscriber drops frames instead of stalling the
    control loop).
    """

    def __init__(self, port: int = DEFAULT_DEBUG_PORT, host: str = "*") -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 10)
        self._socket.setsockopt(zmq.SNDBUF, 32768)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://{host}:{port}")

    def publish(self, payload: dict, topic: str = DEBUG_TOPIC) -> None:
        """Send one msgpack map under ``topic``; drops the frame if no buffer space."""
        import msgpack

        msg = topic.encode("utf-8") + msgpack.packb(payload, default=_to_plain, use_bin_type=True)
        try:
            self._socket.send(msg, self._zmq.DONTWAIT)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        self._socket.close()


class G1DebugPoller:
    """SUB socket that keeps only the latest ``g1_debug`` map (operator side)."""

    def __init__(self, host: str = "localhost", port: int = DEFAULT_DEBUG_PORT, topic: str = DEBUG_TOPIC) -> None:
        import zmq

        self._zmq = zmq
        self._topic = topic.encode("utf-8")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        # CONFLATE must be set before connect so only the newest message is kept.
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.connect(f"tcp://{host}:{port}")

    def get_latest(self) -> dict | None:
        """Return the newest published map, or None when nothing is pending."""
        import msgpack

        msg = None
        while True:  # drain: CONFLATE keeps one message, but be robust without it
            try:
                msg = self._socket.recv(self._zmq.DONTWAIT)
            except self._zmq.Again:
                break
        if msg is None:
            return None
        if not msg.startswith(self._topic):
            return None
        return msgpack.unpackb(msg[len(self._topic) :], raw=False)

    def close(self) -> None:
        self._socket.close()
