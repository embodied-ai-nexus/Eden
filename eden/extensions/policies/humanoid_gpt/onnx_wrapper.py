"""ONNX-runtime wrapper for the Humanoid-GPT policy (MLP or Transformer).

Both released checkpoints map ``obs`` -> ``continuous_actions`` (B, 29) with no
baked-in normalization: the MLP takes ``obs`` (B, 136), the Transformer a stacked
history (B, K, 136). ``dynamic_batch`` (the policy passes ``num_envs > 1``) patches
the static batch-1 axis for batched rollout; ``enable_iobinding`` keeps GPU
inference on-device. Both handled by the shared
:class:`~eden.extensions.policies.onnx.OnnxModule`.
See https://github.com/GalaxyGeneralRobotics/Humanoid-GPT.
"""

from __future__ import annotations

import torch

from eden.extensions.policies.humanoid_gpt.constants import ONNX_INPUT_NAME, ONNX_OUTPUT_NAME
from eden.extensions.policies.onnx import OnnxModule


class HumanoidGptOnnx(OnnxModule):
    """Map ``obs`` ((B, 136) MLP / (B, K, 136) transformer) -> ``continuous_actions``
    (B, 29). ``dynamic_batch`` patches the static batch axis (needed only for num_envs>1).
    """

    def __init__(self, onnx_path: str, device: str | None = None, *, dynamic_batch: bool = True) -> None:
        super().__init__(
            onnx_path, ONNX_INPUT_NAME, ONNX_OUTPUT_NAME, device=device, label="HumanoidGPT",
            enable_iobinding=True, dynamic_batch=dynamic_batch,
        )  # fmt: skip

    @torch.no_grad()
    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic policy mean; the graph's ``std_param`` output is ignored."""
        return self.forward(obs)
