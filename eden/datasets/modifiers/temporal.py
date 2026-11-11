"""Temporal dataset modifiers.

Modifiers in this module change the number of frames per motion while keeping
the dataset's ``target_framerate`` (and therefore the simulation ``dt``)
fixed. They rebuild the underlying :class:`ConcatBatchTensor` buffers and
update ``motion_seq.buffer_length`` accordingly.
"""

from __future__ import annotations

import torch

from eden.datasets.base import _MIN_CLIP_FRAMES
from eden.datasets.modifiers.base import DATASET_MODIFIER_REGISTRY, DatasetModifier
from eden.utils.buffers.cbt import ConcatBatchTensor
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.motion_reference.utils import MOTION_FIELD_ROWS


# Velocity-like fields whose magnitude must scale with the playback-speed factor.
_VELOCITY_FIELDS: tuple[str, ...] = (
    "base_lin_vel_w",
    "base_ang_vel_w",
    "dofs_vel",
    "links_lin_vel_w",
    "links_lin_vel_b",
    "links_ang_vel_w",
    "links_ang_vel_b",
)


def _compute_resample_indices(
    old_length: int, new_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (idx_0, idx_1, blend) for resampling ``old_length`` -> ``new_length``."""
    if old_length < 2:
        raise ValueError(f"Temporal resampling requires sequences with >= 2 frames, got {old_length}")
    phase = torch.linspace(0.0, 1.0, steps=new_length, device=device, dtype=torch.float32)
    idx_float = phase * (old_length - 1)
    idx_0 = idx_float.floor().long().clamp_(max=old_length - 1)
    idx_1 = (idx_0 + 1).clamp_(max=old_length - 1)
    blend = idx_float - idx_0.to(idx_float.dtype)
    return idx_0, idx_1, blend


def _resample(seq: torch.Tensor, new_length: int, rows) -> torch.Tensor:
    """Resample a ``(T, *data_shape)`` tensor to ``new_length`` frames along its leading dim, endpoints kept.

    ``rows`` is the field's entry in :data:`MOTION_FIELD_ROWS`. A float field blends in its own dtype; a bool or int8
    one (only ever picked nearest) thresholds the float32 blend itself, which it could not hold.
    """
    old_length = seq.shape[0]
    if old_length == new_length:
        return seq.clone()
    idx_0, idx_1, blend = _compute_resample_indices(old_length, new_length, seq.device)
    if seq.is_floating_point():
        blend = blend.to(seq.dtype)
    return rows(seq, idx_0, idx_1, blend)


def _rebuild_cbt(cbt: ConcatBatchTensor, new_tensors: list[torch.Tensor]) -> ConcatBatchTensor:
    """Construct a new ConcatBatchTensor from per-sequence tensors, preserving dtype/device."""
    return ConcatBatchTensor(tensors=new_tensors, device=cbt.device, dtype=cbt.dtype)


@DATASET_MODIFIER_REGISTRY.register()
class SpeedScale(DatasetModifier):
    """Scale playback speed by resampling frames at a fixed ``target_framerate``.

    A ``speed`` of 2.0 makes the motion play twice as fast — each sequence is
    resampled to half the number of frames — while keeping the framerate (and
    therefore the simulation ``dt``) unchanged. Because covered distances are
    preserved but traversed in half the wall-clock time, every stored velocity
    is multiplied by ``speed``.

    Position-like fields are interpolated linearly; quaternion fields are
    interpolated with slerp. Body-frame fields are resampled alongside
    world-frame fields so consumers that read them stay consistent.

    The per-sequence frame count is rounded to ``max(2, round(T / speed))``.

    Parameters
    ----------
    speed : float
        Playback-speed factor. ``> 1`` fast-forwards, ``< 1`` slows down. Must be
        positive.
    """

    speed: float = 1.0

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        if self.speed == 1.0:
            return motion_seq
        if self.speed <= 0:
            raise ValueError(f"SpeedScale.speed must be > 0, got {self.speed}")

        old_sizes = motion_seq.buffer_length.tolist()
        new_sizes = [max(2, round(s / self.speed)) for s in old_sizes]

        for name, rows in MOTION_FIELD_ROWS.items():
            cbt = getattr(motion_seq, name)
            if cbt is None:
                continue
            resampled = [_resample(cbt[i], new_sizes[i], rows) for i in range(len(cbt))]
            setattr(motion_seq, name, _rebuild_cbt(cbt, resampled))

        for name in _VELOCITY_FIELDS:
            cbt = getattr(motion_seq, name)
            if cbt is None:
                continue
            cbt.concatenated_tensor.mul_(self.speed)

        motion_seq.buffer_length = torch.tensor(
            new_sizes,
            device=motion_seq.buffer_length.device,
            dtype=motion_seq.buffer_length.dtype,
        )
        return motion_seq


@DATASET_MODIFIER_REGISTRY.register()
class TrimFrames(DatasetModifier):
    """Drop leading and trailing frames from every sequence.

    Equivalent to ``DatasetBase.drop_first_n_frames`` / ``drop_last_n_frames``
    but available as a composable post-load modifier — useful for cropping
    warm-up or settling frames without re-preparing the cache.

    Note
    ----
    When the dataset has multiple ``motion_sequence_keys`` (e.g. robot + object
    in a manipulation dataset), apply the same trim to all keys to preserve
    the per-sequence length invariant across keys. Selecting a subset via
    ``motion_sequence_keys=(...)`` will desynchronize the lengths.

    Parameters
    ----------
    trim_first : int
        Number of frames to drop from the start of each sequence. Must be >= 0.
    trim_last : int
        Number of frames to drop from the end of each sequence. Must be >= 0.
    """

    trim_first: int = 0
    trim_last: int = 0

    def apply_to_sequence(self, motion_seq: "BatchMotionSequence") -> "BatchMotionSequence":
        if self.trim_first == 0 and self.trim_last == 0:
            return motion_seq
        if self.trim_first < 0 or self.trim_last < 0:
            raise ValueError(
                f"TrimFrames requires non-negative trim counts, got "
                f"trim_first={self.trim_first}, trim_last={self.trim_last}"
            )

        old_sizes = motion_seq.buffer_length.tolist()
        new_sizes = [s - self.trim_first - self.trim_last for s in old_sizes]
        for s, new_s in zip(old_sizes, new_sizes, strict=True):
            if new_s < _MIN_CLIP_FRAMES:
                raise ValueError(
                    f"TrimFrames(trim_first={self.trim_first}, trim_last={self.trim_last}) leaves {new_s} frame(s) "
                    f"in a sequence of length {s}; every clip needs at least {_MIN_CLIP_FRAMES}, because a single "
                    f"frame spans no time. Lower trim_first / trim_last."
                )

        # Every per-frame field, contact annotations included: a field left untrimmed keeps the old length and
        # desynchronises from the rest of its sequence.
        window = slice(self.trim_first, -self.trim_last or None)
        for name in BatchMotionSequence.attrs_with_frame_dim:
            cbt = getattr(motion_seq, name)
            if cbt is None:
                continue
            setattr(motion_seq, name, _rebuild_cbt(cbt, [cbt[i][window].clone() for i in range(len(cbt))]))

        motion_seq.buffer_length = torch.tensor(
            new_sizes,
            device=motion_seq.buffer_length.device,
            dtype=motion_seq.buffer_length.dtype,
        )
        return motion_seq
