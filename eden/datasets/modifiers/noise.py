"""Noise-based dataset modifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from eden.datasets.modifiers.base import DATASET_MODIFIER_REGISTRY, DatasetModifier
from eden.utils.buffers.cbt import ConcatBatchTensor

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import BatchMotionSequence


@DATASET_MODIFIER_REGISTRY.register()
class Jitter(DatasetModifier):
    """Add additive noise to selected motion fields.

    Applies independent, per-element noise to every entry of each listed
    :class:`BatchMotionSequence` field. Randomness comes from the global torch
    RNG — seed it upstream (e.g. via ``torch.manual_seed`` in your script)
    for reproducibility.

    Quaternion fields are not rejected, but jittering them with additive noise
    breaks the unit-norm invariant and should be used carefully — prefer
    :class:`Rotate` for orientation perturbation.

    Parameters
    ----------
    fields : tuple[str, ...]
        :class:`BatchMotionSequence` attribute names to jitter, e.g.
        ``("dofs_pos",)`` or ``("base_pos_w", "links_pos_w")``.
    distribution : {"gaussian", "uniform"}
        Noise distribution.
    std : float
        Standard deviation for ``distribution="gaussian"``. Must be >= 0;
        ``0`` is a no-op.
    low, high : float
        Bounds for ``distribution="uniform"``. Require ``low <= high``;
        ``low == high`` is a no-op.
    """

    fields: tuple[str, ...] = ("dofs_pos",)
    distribution: Literal["gaussian", "uniform"] = "gaussian"
    std: float = 0.0
    low: float = 0.0
    high: float = 0.0

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        if self.distribution == "gaussian":
            if self.std < 0:
                raise ValueError(f"Jitter.std must be >= 0, got {self.std}")
            if self.std == 0.0:
                return motion_seq
        elif self.distribution == "uniform":
            if self.low > self.high:
                raise ValueError(f"Jitter.low ({self.low}) must be <= high ({self.high})")
            if self.low == self.high:
                return motion_seq
        else:
            raise ValueError(f"Jitter.distribution must be 'gaussian' or 'uniform', got {self.distribution!r}")

        for name in self.fields:
            cbt = getattr(motion_seq, name, None)
            if cbt is None:
                raise ValueError(
                    f"Jitter: field '{name}' is not populated on the motion sequence. "
                    f"Jitterable fields must be present on the loaded dataset."
                )
            if not isinstance(cbt, ConcatBatchTensor):
                raise ValueError(
                    f"Jitter: field '{name}' is a {type(cbt).__name__}, not a ConcatBatchTensor. "
                    f"Jitter only supports frame-dim fields (dofs_pos, base_pos_w, links_pos_w, etc.); "
                    f"batch-dim scalars like 'framerate' and 'buffer_length' cannot be jittered."
                )
            t = cbt.concatenated_tensor
            if self.distribution == "gaussian":
                noise = torch.randn(t.shape, dtype=t.dtype, device=t.device) * self.std
            else:
                noise = torch.empty(t.shape, dtype=t.dtype, device=t.device).uniform_(self.low, self.high)
            t.add_(noise)
        return motion_seq
