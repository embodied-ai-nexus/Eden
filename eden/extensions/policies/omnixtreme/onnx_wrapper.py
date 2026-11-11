"""OmniXtreme-specific ONNX wrappers (residual + FM base policy) and FK session helpers.

The shared session layer and the generic single-IO wrapper live in
:mod:`eden.extensions.policies.onnx`.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import torch

from eden.extensions.policies import onnx as _onnx_mod
from eden.extensions.policies.onnx import (
    _OnnxWrapperBase,
    _ensure_2d,
    _env_bool,
    _make_session,
    _run_with_cpu_fallback,
)
from eden.utils.torch import to_numpy

# ``onnx.ort`` is bound lazily on first session build, so it MUST be reached through the
# module. A ``from ... import ort`` here would snapshot the pre-load ``None`` sentinel and
# never see the rebind — leaving every ``ort.`` reference below permanently None, with a
# working wheel installed.


def _make_fk_onnx_session(onnx_path: str):
    """Create the FK ONNX Runtime session (TensorRT EP if available and TENSORRT_EP/ONNX_TRT/ORT_TRT is set).

    No runtime check of its own: ``_make_session`` loads onnxruntime and raises the shared
    ImportError — the one that names both wheels and the CUDA caveat — if it is unusable.
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"FK ONNX file does not exist: {onnx_path}")
    return _make_session(onnx_path, label="FK ONNX")


def run_fk_session(cmd_term, inputs: dict) -> list:
    """Run FK ONNX session with automatic CPU fallback on CUBLAS/CUDA errors."""
    return _run_with_cpu_fallback(
        cmd_term,
        inputs,
        session_attr="fk_session",
        path_attr="_fk_onnx_path",
        flag_attr="_fk_cpu_fallback",
        label="FK ONNX",
    )


class OnnxResidualPolicyWrapper(_OnnxWrapperBase):
    """ONNX wrapper for residual policy: forward(flat_obs) -> residual_action."""

    def __init__(self, onnx_path: str, device: str | None = None):
        super().__init__(onnx_path, device=device, label="ONNX Residual")
        self._has_time = "time_step" in self._input_names

    @torch.no_grad()
    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        obs_np = _ensure_2d(to_numpy(flat_obs, np.float32))
        inputs = {"obs": obs_np}
        if self._has_time:
            inputs["time_step"] = np.zeros((obs_np.shape[0], 1), dtype=np.float32)
        return self._run(inputs, label="ONNX Residual")


class OnnxBasePolicyWrapper(_OnnxWrapperBase):
    """ONNX wrapper for FM base policy with multi-input dict interface.

    Inputs: real_obs, command_obs, real_historical_obs_raw, self_obs, optional initial_noise.
    Output: actions (torch.Tensor [B, 29]).
    """

    def __init__(self, onnx_path: str, device: str | None = None):
        super().__init__(onnx_path, device=device, label="ONNX Base")
        self._name_to_dim = {
            i.name: [d if isinstance(d, int) else None for d in i.shape] for i in self.session.get_inputs()
        }

    def _infer_feat_dim(self, name: str, fallback: int) -> int:
        dims = self._name_to_dim.get(name)
        if dims and len(dims) >= 2 and dims[1] is not None:
            return dims[1]
        return fallback

    def _action_dim(self) -> int:
        try:
            shp = self.session.get_outputs()[0].shape
            if len(shp) >= 2 and isinstance(shp[1], int):
                return int(shp[1])
        except Exception:
            pass
        return 29

    def _build_cpu_inputs(self, obs_dict: dict) -> dict[str, np.ndarray]:
        """Build numpy input dict for CPU inference path."""
        ro = _ensure_2d(to_numpy(obs_dict["real_obs"], np.float32))
        co = _ensure_2d(to_numpy(obs_dict["command_obs"], np.float32))
        rh = to_numpy(obs_dict["real_historical_obs_raw"], np.float32)
        if rh.ndim == 3:
            rh = rh.reshape(rh.shape[0], -1)
        rh = _ensure_2d(rh)

        B = ro.shape[0]
        self_obs = obs_dict.get("self_obs")
        if self_obs is None:
            so = np.zeros((B, self._infer_feat_dim("self_obs", 1)), dtype=np.float32)
        else:
            so = _ensure_2d(to_numpy(self_obs, np.float32))

        inputs = {
            "real_obs": ro,
            "command_obs": co,
            "real_historical_obs_raw": rh,
            "self_obs": so,
        }

        if "initial_noise" in self._input_names:
            init = obs_dict.get("initial_noise")
            if init is None:
                inputs["initial_noise"] = np.zeros(
                    (B, self._infer_feat_dim("initial_noise", self._action_dim())), dtype=np.float32
                )
            else:
                init_np = _ensure_2d(to_numpy(init, np.float32))
                if init_np.ndim > 2:
                    init_np = init_np.reshape(init_np.shape[0], -1)
                inputs["initial_noise"] = init_np

        return inputs

    def _forward_iobind(self, obs_dict: dict) -> torch.Tensor:
        """GPU IO-binding fast path."""
        from torch.utils.dlpack import to_dlpack

        def to_gpu(t):
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            return t.to(self.device, dtype=torch.float32).contiguous()

        ro = to_gpu(obs_dict["real_obs"])
        co = to_gpu(obs_dict["command_obs"])
        rh = obs_dict["real_historical_obs_raw"]
        if isinstance(rh, torch.Tensor) and rh.ndim == 3:
            rh = rh.reshape(rh.shape[0], -1)
        rh = to_gpu(rh)
        B = int(ro.shape[0])

        self_obs = obs_dict.get("self_obs")
        so = (
            torch.zeros((B, self._infer_feat_dim("self_obs", 1)), device=self.device, dtype=torch.float32)
            if self_obs is None
            else to_gpu(self_obs)
        )

        io = self.session.io_binding()

        def bind(name: str, t: torch.Tensor):
            # Through the module, not a from-imported name: see the import block's note.
            io.bind_ortvalue_input(name, _onnx_mod.ort.OrtValue.from_dlpack(to_dlpack(t)))

        bind("real_obs", ro)
        bind("command_obs", co)
        bind("real_historical_obs_raw", rh)
        bind("self_obs", so)

        if "initial_noise" in self._input_names:
            init = obs_dict.get("initial_noise")
            if init is not None:
                init_t = to_gpu(init)
                if init_t.ndim > 2:
                    init_t = init_t.view(init_t.shape[0], -1)
            else:
                init_t = torch.zeros(
                    (B, self._infer_feat_dim("initial_noise", self._action_dim())),
                    device=self.device,
                    dtype=torch.float32,
                )
            bind("initial_noise", init_t)

        act_dim = self._action_dim()
        out_t = torch.empty((B, act_dim), device=self.device, dtype=torch.float32)
        io.bind_ortvalue_output(self._output_names[0], _onnx_mod.ort.OrtValue.from_dlpack(to_dlpack(out_t)))
        self.session.run_with_iobinding(io)
        return out_t

    @torch.no_grad()
    def forward(self, obs_or_flat: Any) -> torch.Tensor:
        if isinstance(obs_or_flat, dict):
            real_obs = obs_or_flat.get("real_obs")
            if (
                real_obs is None
                or obs_or_flat.get("command_obs") is None
                or obs_or_flat.get("real_historical_obs_raw") is None
            ):
                raise ValueError("obs_dict missing required keys: real_obs/command_obs/real_historical_obs_raw")

            # GPU IO-binding when usable, else the numpy path.
            if self._can_iobind():
                try:
                    return self._forward_iobind(obs_or_flat)
                except Exception as e:
                    if _env_bool("ONNX_DEBUG"):
                        warnings.warn(f"[ONNX Base] IO-binding failed ({type(e).__name__}: {e}), falling back to CPU")
            return self._run(self._build_cpu_inputs(obs_or_flat), label="ONNX Base")

        # Non-dict input: every graph input is fed zeros (only the batch size is read from it).
        B = int(obs_or_flat.shape[0]) if isinstance(obs_or_flat, torch.Tensor) and obs_or_flat.ndim >= 1 else 1
        inputs = {name: np.zeros((B, self._infer_feat_dim(name, 1)), dtype=np.float32) for name in self._input_names}
        return self._run(inputs, label="ONNX Base")
