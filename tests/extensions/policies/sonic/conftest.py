"""Fixtures for the SONIC tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.extensions.policies.sonic.joint_maps import IL_TO_MJ
from eden.extensions.policies.sonic.motion import MotionFrames
from eden.extensions.policies.sonic.policy import (
    _DECODER_INPUT_DIM,
    _ENCODER_INPUT_DIM,
)


@pytest.fixture(scope="session")
def isaaclab_joint_names() -> list[str]:
    """IsaacLab's G1 29-DOF joint order (interleaved left/right/waist groups): the order SONIC was trained in."""
    return [
        f"{name}_joint"
        for name in (
            *("left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll", "waist_roll"),
            *("left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee", "right_knee"),
            *("left_shoulder_pitch", "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch"),
            *("left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll", "right_ankle_roll"),
            *("left_shoulder_yaw", "right_shoulder_yaw", "left_elbow", "right_elbow"),
            *("left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch"),
            *("left_wrist_yaw", "right_wrist_yaw"),
        )
    ]


@pytest.fixture
def local_sonic_policy(tmp_path, write_onnx, onnx_runtime, monkeypatch):
    """Return ``make(base_cls, decoder_nodes=None, decoder_initializers=None)``: ``base_cls`` on tiny real ONNX graphs.

    The released bundle is not available offline. The encoder passes its first 64 input
    columns through as the token, and the default decoder passes its first 29 through as the
    action, so together they hand back the encoder's first 29 input columns as the
    IsaacLab-order action. Each call rewrites the decoder, so one test can compare several.
    """
    monkeypatch.setenv("ONNX_CPU", "1")
    slice_inits = {"starts": np.array([0], np.int64), "axes": np.array([1], np.int64)}
    encoder = write_onnx(
        tmp_path / "encoder.onnx",
        [("Slice", ["obs_dict", "starts", "ends", "axes"], ["encoded_tokens"], {})],
        [("obs_dict", [1, _ENCODER_INPUT_DIM])],
        [("encoded_tokens", [1, 64])],
        {**slice_inits, "ends": np.array([64], np.int64)},
    )

    names = itertools.count()

    def make(base_cls, decoder_nodes=None, decoder_initializers=None):
        if decoder_nodes is None:  # pass-through
            decoder_nodes = [("Slice", ["obs_dict", "starts", "ends", "axes"], ["action"], {})]
            decoder_initializers = {"ends": np.array([29], np.int64)}
        decoder = write_onnx(
            # A fresh file per call: the dynamic-batch copy derived from an ONNX file is
            # cached by mtime, so rewriting one path could reuse a stale derivative.
            tmp_path / f"decoder_{next(names)}.onnx",
            decoder_nodes,
            [("obs_dict", [1, _DECODER_INPUT_DIM])],
            [("action", [1, 29])],
            {**slice_inits, **(decoder_initializers or {})},
        )
        cls = type(f"Local{base_cls.__name__}", (base_cls,), {})
        cls.use_local_checkpoint(encoder)
        cls.use_local_checkpoint(decoder, checkpoint_attr="decoder_checkpoint_path")
        return cls

    return make


def yawq(angle: float) -> np.ndarray:
    return gu.axis_angle_to_quat(np.array(float(angle)), np.array([0.0, 0.0, 1.0]))


def make_motion(name="clip", timesteps=5, *, root_yaw=0.0, quat=None, body_pos=False, encode_mode=None):
    """The shared "zeros joint_pos/vel + tiled body_quat" MotionFrames builder."""
    q = yawq(root_yaw) if quat is None else np.asarray(quat, dtype=np.float64)
    kwargs = {
        "name": name,
        "joint_pos": np.zeros((timesteps, 29)),
        "joint_vel": np.zeros((timesteps, 29)),
        "body_quat": np.tile(q, (timesteps, 1, 1)),
    }
    if body_pos:
        kwargs["body_pos"] = np.zeros((timesteps, 1, 3))
    motion = MotionFrames(**kwargs)
    if encode_mode is not None:
        motion.encode_mode = encode_mode
    return motion


def linear_qpos30(n=30, vx=0.1, dyaw=0.02):
    """30 Hz qpos rows: linear x, constant z, rotating yaw, joint columns = MuJoCo index."""
    qpos = np.zeros((n, 36))
    qpos[:, 0] = np.arange(n) * vx
    qpos[:, 2] = 0.78
    for f in range(n):
        qpos[f, 3:7] = yawq(f * dyaw)
    qpos[:, 7:] = np.arange(29)[None, :] * 0.01
    return qpos


# PlannerOnnx's 11-input signature: the dtypes and static shapes ``PlannerOnnx.generate`` must feed.
# onnxruntime rejects a missing or unknown input, a wrong dtype, and a wrong rank or static dim, so a
# graph declaring exactly these is what pins the feed contract.
PLANNER_INPUTS = (
    ("context_mujoco_qpos", "FLOAT", [1, 4, 36]),
    ("mode", "INT64", [1]),
    ("target_vel", "FLOAT", [1]),
    ("height", "FLOAT", [1]),
    ("movement_direction", "FLOAT", [1, 3]),
    ("facing_direction", "FLOAT", [1, 3]),
    ("random_seed", "INT64", [1]),
    ("has_specific_target", "INT64", [1, 1]),
    ("specific_target_positions", "FLOAT", [1, 4, 3]),
    ("specific_target_headings", "FLOAT", [1, 4]),
    ("allowed_pred_num_tokens", "INT64", [1, 11]),
)


@pytest.fixture
def write_planner_onnx(onnx_runtime, monkeypatch):
    """Return ``write(path, nodes, initializers) -> path``: an opset-13 graph with PlannerOnnx's signature.

    Inputs are :data:`PLANNER_INPUTS`; the nodes must produce ``mujoco_qpos`` (float32
    ``(1, frames, 36)``) and ``num_pred_frames`` (int64 ``(1,)``). Skips without ``onnx``.
    """
    onnx = pytest.importorskip("onnx", exc_type=ImportError)
    from onnx import TensorProto, helper, numpy_helper

    monkeypatch.setenv("ONNX_CPU", "1")

    def write(path, nodes, initializers) -> str:
        graph = helper.make_graph(
            [helper.make_node(op, ins, outs, **attrs) for op, ins, outs, attrs in nodes],
            "planner",
            [helper.make_tensor_value_info(n, getattr(TensorProto, t), s) for n, t, s in PLANNER_INPUTS],
            [
                helper.make_tensor_value_info("mujoco_qpos", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("num_pred_frames", TensorProto.INT64, [1]),
            ],
            [numpy_helper.from_array(np.asarray(value), name) for name, value in initializers.items()],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))
        return str(path)

    return write


PLANNER_NUM_FRAMES = 36  # frames the planner_onnx graph reports valid (-> 60 frames at 50 Hz)
_PLANNER_BUFFER_FRAMES = 64  # frames in the graph's qpos buffer; the rest past PLANNER_NUM_FRAMES are NaN
_QPOS_ROOT_WIDTH = 7  # a qpos row: root xyz + wxyz quaternion, then the 29 MuJoCo joints
_QPOS_WIDTH = _QPOS_ROOT_WIDTH + 29
# The command echo the planner_onnx graph writes into each valid frame, field by field
# (name, width, read back as int), starting at MuJoCo joint _ECHO_FIRST_MJ_JOINT: the arms,
# outside the lower body the idle re-adaptation reads.
_ECHO_FIELDS = (
    ("mode", 1, True), ("target_vel", 1, False), ("height", 1, False), ("movement_direction", 3, False),
    ("facing_direction", 3, False), ("random_seed", 1, True), ("context_root_z", 4, False),
)  # fmt: skip
_ECHO_FIRST_MJ_JOINT = 15


@pytest.fixture
def planner_onnx(tmp_path, write_planner_onnx):
    """A real :class:`PlannerOnnx` on a tiny graph that stands idle and echoes its command.

    The released ``planner_sonic.onnx`` is not available offline. This graph emits a 64-frame
    buffer whose first :data:`PLANNER_NUM_FRAMES` frames stand at the origin (z 0.78, identity
    root, zero legs/waist) and whose remaining frames are NaN, so a caller that ignored
    ``num_pred_frames`` would hit the resampler's NaN check. Every valid frame's arm joints carry
    the command it was fed (mode, speed, height, directions, seed, the context's 4 root heights):
    :func:`planner_command` reads it back out of a generated motion.
    """
    from eden.extensions.policies.sonic.planner import PlannerOnnx

    base = np.full((1, _PLANNER_BUFFER_FRAMES, _QPOS_WIDTH), np.nan, dtype=np.float32)
    base[0, :PLANNER_NUM_FRAMES] = 0.0
    base[0, :PLANNER_NUM_FRAMES, 2] = 0.78
    base[0, :PLANNER_NUM_FRAMES, 3] = 1.0
    echo_width = sum(n for _, n, _ in _ECHO_FIELDS)
    nodes = [
        ("Cast", ["mode"], ["mode_f"], {"to": 1}),  # 1 = FLOAT
        ("Cast", ["random_seed"], ["seed_f"], {"to": 1}),
        *[("Unsqueeze", [x, "axis1"], [f"{x}_2d"], {}) for x in ("mode_f", "target_vel", "height", "seed_f")],
        ("Slice", ["context_mujoco_qpos", "z_start", "z_end", "axis2"], ["context_z"], {}),
        ("Reshape", ["context_z", "row4"], ["context_z_2d"], {}),
        (
            "Concat",
            [
                "mode_f_2d",
                "target_vel_2d",
                "height_2d",
                "movement_direction",
                "facing_direction",
                "seed_f_2d",
                "context_z_2d",
            ],
            ["echo"],
            {"axis": 1},
        ),
        ("Pad", ["echo", "echo_pads"], ["echo_row"], {}),
        ("Unsqueeze", ["echo_row", "axis1"], ["echo_frame"], {}),
        ("Add", ["base", "echo_frame"], ["mujoco_qpos"], {}),
        ("Identity", ["num_frames"], ["num_pred_frames"], {}),
    ]
    first = _QPOS_ROOT_WIDTH + _ECHO_FIRST_MJ_JOINT  # the echo's first qpos column
    initializers = {
        "base": base,
        "axis1": np.array([1], np.int64),
        "axis2": np.array([2], np.int64),
        "z_start": np.array([2], np.int64),
        "z_end": np.array([3], np.int64),
        "row4": np.array([1, 4], np.int64),
        "echo_pads": np.array([0, first, 0, _QPOS_WIDTH - first - echo_width], np.int64),
        "num_frames": np.array([PLANNER_NUM_FRAMES], np.int64),
    }
    return PlannerOnnx(write_planner_onnx(tmp_path / "planner.onnx", nodes, initializers), device="cpu")


def planner_command(motion: MotionFrames, frame: int = -1) -> dict:
    """The command the :func:`planner_onnx` graph was fed, read back from a motion frame it generated."""
    echo = motion.joint_pos[frame][np.asarray(IL_TO_MJ)][_ECHO_FIRST_MJ_JOINT:]  # IsaacLab -> MuJoCo columns
    widths = [n for _, n, _ in _ECHO_FIELDS]
    starts = np.cumsum([0, *widths[:-1]])
    out = {}
    for (name, n, is_int), start in zip(_ECHO_FIELDS, starts):
        value = echo[start] if n == 1 else echo[start : start + n]
        out[name] = int(np.rint(value)) if is_int else value
    return out
