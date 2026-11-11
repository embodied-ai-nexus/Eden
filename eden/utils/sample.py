"""Sampling helpers (e.g. per-env probability gating)."""

from collections.abc import Sequence

import torch

from eden.utils.torch import device_constant


def sample_uniform(
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    size: int | tuple[int, ...],
    device: str,
) -> torch.Tensor:
    """Sample uniformly within a range.

    Parameters
    ----------
    lower : torch.Tensor | float
        Lower bound of the uniform range.
    upper : torch.Tensor | float
        Upper bound of the uniform range.
    size : int | tuple[int, ...]
        Desired tensor shape.
    device : str
        Device to create the tensor on.

    Returns
    -------
    torch.Tensor
        Tensor of uniformly sampled values with shape ``size``.
    """
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_uniform_ranges(
    ranges: Sequence[Sequence[float]],
    num_samples: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample ``(num_samples, len(ranges))`` values, column ``i`` uniform in ``ranges[i] = (low, high)``.

    The same draw as ``sample_uniform(torch.tensor(lows), torch.tensor(highs), (num_samples, len(ranges)))``,
    but the bound tensors are built once per distinct ``ranges`` and reused: building them from host
    floats on every call is a blocking host->device copy, and a reset event or a command resample makes
    that call every step the auto-reset fires. Ranges a curriculum rewrites at runtime just key
    a new entry (:func:`eden.utils.torch.device_constant`).

    Parameters
    ----------
    ranges : Sequence[Sequence[float]]
        One ``(low, high)`` pair per column.
    num_samples : int
        Number of rows to draw (usually ``num_envs``).
    device : str | torch.device
        Device to create the samples on.

    Returns
    -------
    torch.Tensor
        Samples of shape ``(num_samples, len(ranges))``.
    """
    lower = device_constant([low for low, _ in ranges], device)
    upper = device_constant([high for _, high in ranges], device)
    return sample_uniform(lower, upper, (num_samples, len(ranges)), device=device)


def apply_probability(mask: torch.Tensor, probability: float) -> torch.Tensor:
    """Optionally gate a per-env boolean mask by a global Bernoulli draw.

    A single sample is drawn per call (not per env). ``probability=1.0`` is a no-op,
    ``probability=0.0`` zeroes the mask. Used by probabilistic-soft-limit termination
    terms — gating is coarse on purpose, so they behave as a stochastic safety net
    rather than a hard cap. Mirrors the convention of holosoma's locomotion termination terms.

    The intermediate path uses a 0-dim bool tensor and a tensor-level ``&``
    instead of a Python ``if`` on ``sample.item()``, so the function stays
    async on GPU (no device→host sync per call).

    Raises ``ValueError`` if ``probability`` is NaN or outside ``[0, 1]`` — silently
    zeroing every termination on a misconfiguration would mask the bug.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability!r}.")
    # Constant cases short-circuit so we don't allocate a random sample.
    if probability >= 1.0:
        return mask
    if probability <= 0.0:
        return torch.zeros_like(mask, dtype=torch.bool)
    # Tensor-level gate: 0-dim bool broadcasts to the mask shape.
    gate = torch.rand((), device=mask.device) < probability
    return mask & gate
