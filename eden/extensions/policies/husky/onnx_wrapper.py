"""ONNX wrapper for the HUSKY skateboarding actor."""

from __future__ import annotations

from eden.extensions.policies.husky.constants import ACTOR_OBS_DIM, NUM_JOINTS
from eden.extensions.policies.onnx import OnnxModule


class HuskyActor(OnnxModule):
    """Released HUSKY actor: ``obs (1, 395) -> actions (1, 23)``.

    The graph embeds the empirical observation normalization (Sub/Div ahead of
    the ELU MLP), so raw scaled observations go in and raw actions come out —
    no external normalizer state.
    """

    def __init__(self, onnx_path: str, device: str | None = None) -> None:
        super().__init__(
            onnx_path,
            "obs",
            "actions",
            device=device,
            label="HUSKY Actor",
            expected_input_dim=ACTOR_OBS_DIM,
            expected_output_dim=NUM_JOINTS,
            enable_iobinding=True,
            dynamic_batch=True,
        )
