"""Shared fixtures for the policy tests.

* Tiny real ONNX graphs and the optional-runtime guards. Released checkpoints are not available
  offline, so a test that needs a session builds a two- or three-node graph in ``tmp_path`` and
  runs it through the real ``onnxruntime``. ``exc_type=ImportError`` on every ``importorskip``: a
  GPU wheel whose CUDA runtime is missing raises ImportError rather than ModuleNotFoundError, and
  bare ``importorskip`` errors on that from pytest 9.1.
* Golden records of the server-backed client policies (see the section below).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch


@pytest.fixture
def onnx_runtime():
    """Skip unless ``onnxruntime`` is usable; returns the module."""
    return pytest.importorskip("onnxruntime", exc_type=ImportError)


@pytest.fixture
def write_onnx():
    """Return ``write(path, nodes, inputs, outputs, initializers=())`` that saves an opset-13 graph.

    ``inputs`` / ``outputs`` are ``(name, shape)`` pairs of float32 tensors; ``initializers``
    maps a name to a numpy array. Skips when the ``onnx`` authoring package is missing.
    """
    onnx = pytest.importorskip("onnx", exc_type=ImportError)
    from onnx import TensorProto, helper, numpy_helper

    def write(path, nodes, inputs, outputs, initializers=None) -> str:
        graph = helper.make_graph(
            [helper.make_node(op, ins, outs, **attrs) for op, ins, outs, attrs in nodes],
            Path(path).stem,
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in inputs],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in outputs],
            [numpy_helper.from_array(np.asarray(value), name) for name, value in (initializers or {}).items()],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))
        return str(path)

    return write


# ---------------------------------------------------------------------------
# Server-backed client policies: golden records
# ---------------------------------------------------------------------------
#
# A golden case builds one client policy against a real env (built from the policy's own
# ``action_options()``) and a local fake policy server (tests/support/policy_servers.py), feeds a
# deterministic observation sequence to ``policy.step()``, and records
#
# * every payload the server received (inference payloads, decoded; HTTP: every POST),
# * every controller dict the policy returned to the action manager,
# * how many inference round-trips the sequence triggered (chunk scheduling),
#
# then compares that against ``tests/data/policy_client_goldens/<name>.json``. The fixtures were
# recorded from the clients before the client-layer refactor, so a match proves a change is
# behaviour-neutral; a mismatch is a changed server contract. Regenerate deliberately with
# ``EDEN_REGEN_GOLDENS=1`` and review the diff.

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "policy_client_goldens"
REGEN_GOLDENS = os.environ.get("EDEN_REGEN_GOLDENS") == "1"
GOLDEN_TASK = "golden task"


def image(h: int = 12, w: int = 16, seed: int = 0) -> torch.Tensor:
    """A deterministic ``(1, h, w, 3)`` uint8 camera frame."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, 256, size=(1, h, w, 3), dtype=np.uint8))


def vec(*values: float) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.float32)


UNIT_QUAT = (1.0, 0.0, 0.0, 0.0)
TILT_QUAT = (0.9238795, 0.0, 0.3826834, 0.0)  # 45 deg about y, wxyz


def ee8(gripper: float, quat=TILT_QUAT) -> torch.Tensor:
    """``[xyz, quat_wxyz, gripper]`` end-effector proprio."""
    return vec(0.3, -0.1, 0.2, *quat, gripper)


def joint8(gripper: float) -> torch.Tensor:
    """``[q(7), gripper]`` DROID joint proprio."""
    return vec(0.1, -0.2, 0.3, -1.5, 0.05, 1.8, 0.4, gripper)


def chunk(t: int, d: int, call: int, *, gripper_col: int | None = None, scale: float = 0.05) -> np.ndarray:
    """A bounded, call-dependent ``(t, d)`` action chunk whose gripper column alternates every 3 rows."""
    base = np.sin(np.arange(t * d, dtype=np.float32).reshape(t, d) * scale * (1 + 0.1 * call))
    if gripper_col is not None:
        base[:, gripper_col] = ((np.arange(t) // 3 + call) % 2).astype(np.float32)
    return base.astype(np.float32)


def keyed_chunk(t: int, keys: dict[str, int], call: int, *, gripper_key: str) -> dict[str, np.ndarray]:
    """``chunk`` split into GR00T-style ``{key: (1, t, width)}`` arrays."""
    flat, out, col = chunk(t, sum(keys.values()), call), {}, 0
    for key, width in keys.items():
        arr = flat[:, col : col + width]
        if key == gripper_key:
            arr = ((np.arange(t) // 3 + call) % 2).astype(np.float32).reshape(t, 1)
        out[key] = arr[np.newaxis]
        col += width
    return out


# The fixtures were recorded against an env whose DOF positions were a fixed ramp over these names
# (and dt = 1/15 s); the real envs below are seeded with the same values so integrating clients
# (DROID joint-velocity integration) see the same joint state.
_RAMP_DOFS = {
    "robot": (
        ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        + [f"joint{i}" for i in range(1, 8)]
        + [f"left_arm_joint{i}" for i in range(1, 7)]
        + [f"right_arm_joint{i}" for i in range(1, 7)]
        + ["left_finger", "right_finger"]
    ),
    "gripper": ["left_driver_joint", "right_driver_joint"],
}
_ENVS: dict[str, Any] = {}


def _robot(kind: str):
    import eden as en
    from eden.options.robots import Droid
    from eden.options.robots.aloha_vx300s import Aloha
    from eden.options.robots.libero import LiberoPanda
    from eden.options.robots.trossen_widowx250 import WidowX250

    factories = {"widowx": WidowX250, "droid": Droid, "libero": LiberoPanda, "aloha": Aloha}
    return en.scenes.Empty(robot=factories[kind]())


def policy_env(
    robot: str,
    action_options: dict,
    *,
    dt: float = 1.0 / 15.0,
    num_envs: int = 1,
    observation_options: dict | None = None,
    cam_resolution: tuple[int, int] = (16, 16),
):
    """A built scene with ``robot`` (``widowx``/``droid``/``libero``/``aloha``) and the policy's action
    terms (plus, when given, its observation groups and a ``cam_resolution`` rasterizer camera for every
    ``camera_name`` they read); cached per everything it is built from, keyed on ``model_dump()``.
    """
    from genesis.options.sensors import RasterizerCameraOptions

    from eden.options import EnvOptions
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.managers.actions import ActionManagerOptions
    from eden.options.managers.observations import ObservationManagerOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    observation_options = observation_options or {}
    key = json.dumps(
        [
            robot,
            {name: term.model_dump() for name, term in action_options.items()},
            {name: group.model_dump() for name, group in observation_options.items()},
            cam_resolution,
            dt,
            num_envs,
        ],
        sort_keys=True,
        default=str,
    )
    if key not in _ENVS:
        cameras = {}
        for group in observation_options.values():
            for name in group.term_keys():
                # Function terms (rgb_image) carry their camera in ``params``; class terms have none.
                camera = (getattr(getattr(group, name), "params", None) or {}).get("camera_name")
                if camera is not None:
                    cameras[camera] = CameraOptions(
                        cam_resolution=cam_resolution, sensor_options=RasterizerCameraOptions()
                    )
        env = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=num_envs, sim_dt=dt, decimation=1),
                scene_options=_robot(robot),
                cameras_options=CamerasOptions(**cameras),
                observation_options=ObservationManagerOptions(**observation_options),
                action_options=ActionManagerOptions(**action_options),
            )
        )
        for entity_name, names in _RAMP_DOFS.items():
            entity = env.entities.get(entity_name)
            if entity is None:
                continue
            ramp = torch.linspace(0.1, 0.7, len(names))
            pos = entity.get_dofs_pos().clone()
            for i, name in enumerate(names):
                if name in entity.dofs_name:
                    pos[:, entity.dofs_name.index(name)] = ramp[i]
            entity.set_dofs_pos(pos)
        _ENVS[key] = env
    return _ENVS[key]


def core_term(cls, env, **params) -> torch.Tensor:
    """Build one core observation term on ``env`` and return its current value."""
    term = cls(env, cls.configure(**params))
    term.build()
    return term.compute()


def normalize(obj: Any) -> Any:
    """JSON-comparable form: small arrays by value, large ones (and bytes) by shape + sha256."""
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return normalize(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        if obj.size <= 64:
            return {"shape": list(obj.shape), "dtype": str(obj.dtype), "values": normalize(obj.tolist())}
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest(),
        }
    if isinstance(obj, (bytes, bytearray)):
        return {"bytes": len(obj), "sha256": hashlib.sha256(obj).hexdigest()}
    if isinstance(obj, np.generic):
        return normalize(obj.item())
    if isinstance(obj, float):
        # NaN is a legitimate "hold" gripper command; JSON has no NaN and nan != nan, so pin it as a string.
        return "nan" if obj != obj else round(obj, 5)
    if isinstance(obj, str) and obj.startswith('{"__ndarray__"'):
        import json_numpy

        return normalize(json_numpy.loads(obj))
    return obj


def record_golden(policy, server, obs_fn: Callable[[int], dict], steps: int) -> dict:
    """Step ``policy`` through ``steps`` observations and record what reached ``server``."""
    torch.manual_seed(0)
    actions = [normalize(policy.step(obs_fn(step))) for step in range(steps)]
    return {"payloads": normalize(server.records), "actions": actions, "infer_calls": server.infer_calls}


def assert_matches_golden(name: str, got: dict) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    if REGEN_GOLDENS or not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        if not REGEN_GOLDENS:
            pytest.fail(f"Golden fixture was missing and has been written: {path}. Re-run to compare.")
    expected = json.loads(path.read_text())
    assert got == expected, f"{name}: behaviour drifted from {path}"
