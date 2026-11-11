"""ONNX wrappers for the GEAR-SONIC encoder + decoder graphs.

The released SONIC policy ships as two single-input / single-output graphs, both
keyed on an ``obs_dict`` input:

* ``model_encoder.onnx`` — ``obs_dict`` → ``encoded_tokens`` (token state).
* ``model_decoder.onnx`` — ``obs_dict`` → ``action`` (29-DOF, IsaacLab order).

Both are thin configs over the shared
:class:`eden.extensions.policies.onnx.OnnxModule`; the session layer
(provider selection, CPU fallback) is shared there too.
"""

from __future__ import annotations

from eden.extensions.policies.onnx import OnnxModule, _env_bool


class SonicEncoder(OnnxModule):
    """``model_encoder.onnx``: concatenated encoder obs → fixed-size token tensor."""

    def __init__(self, onnx_path: str, device: str | None = None, expected_input_dim: int | None = None) -> None:
        super().__init__(
            onnx_path, "obs_dict", "encoded_tokens", device=device, label="SONIC Encoder",
            expected_input_dim=expected_input_dim, enable_iobinding=True, dynamic_batch=True,
        )  # fmt: skip


class SonicDecoder(OnnxModule):
    """``model_decoder.onnx``: ``[token_state | history obs]`` → 29-DOF action (IsaacLab order)."""

    def __init__(self, onnx_path: str, device: str | None = None, expected_input_dim: int | None = None) -> None:
        super().__init__(
            onnx_path, "obs_dict", "action", device=device, label="SONIC Decoder",
            expected_input_dim=expected_input_dim, enable_iobinding=True, dynamic_batch=True,
        )  # fmt: skip


__all__ = ["SonicDecoder", "SonicEncoder", "_env_bool"]
