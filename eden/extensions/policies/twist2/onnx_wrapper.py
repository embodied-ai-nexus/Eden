"""ONNX wrapper for the TWIST2 actor graph.

The exported TWIST2 actor is a single-input / single-output graph:
``obs`` (1, 1524) → ``actions`` (1, 29). It is a thin config over the shared
:class:`eden.extensions.policies.onnx.OnnxModule`, which owns the session layer
(provider selection, CPU fallback).

The released graph has a static batch of 1; ``dynamic_batch=True`` patches it to
a dynamic leading axis so ``num_envs > 1`` batched rollout works, and
``enable_iobinding=True`` keeps batched GPU inference on-device (both handled by
``OnnxModule``, the same path SONIC / Humanoid-GPT use).
"""

from __future__ import annotations

from eden.extensions.policies.onnx import OnnxModule
from eden.extensions.policies.twist2.constants import ACTOR_OBS_DIM, NUM_JOINTS


class Twist2Actor(OnnxModule):
    """``pretrained.onnx``: flat actor obs (1524) → 29-DOF action (MJCF order)."""

    def __init__(self, onnx_path: str, device: str | None = None) -> None:
        super().__init__(
            onnx_path, "obs", "actions", device=device, label="TWIST2 Actor",
            expected_input_dim=ACTOR_OBS_DIM, expected_output_dim=NUM_JOINTS,
            enable_iobinding=True, dynamic_batch=True,
        )  # fmt: skip


__all__ = ["Twist2Actor"]
