# Portions Copyright 2026 the Tactile Genesis authors. Derived from Tactile Genesis
# (https://github.com/neuroagents-lab/tactile-genesis, commit 3023bcd), licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Modified by the Eden authors.
"""Two-pole filters shared by the audio sources.

The coefficient helpers return the feedback pair ``(a1, a2)`` of the recurrence ``y = a1*y1 - a2*y2 + drive``
and work elementwise, so the coefficients broadcast over ``(B, n_emit, ...)``. They are ported verbatim from the
upstream fork's ``genesis/engine/audio/synthesis.py``. :func:`two_pole_block_kernels` is Eden's: it turns that
recurrence into two matrices, so a whole block is filtered without a per-sample loop.
"""

import torch


def resonator_coeffs(freq: torch.Tensor, decay: torch.Tensor, dt: float):
    """Coefficients of a two-pole resonator: a struck mode that rings at ``freq`` and dies away.

    Parameters
    ----------
    freq : torch.Tensor
        Center frequency of the mode, in Hz.
    decay : torch.Tensor
        Amplitude decay rate, in 1/s. Larger decays faster.
    dt : float
        Sample period, in seconds.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The ``(a1, a2)`` feedback coefficients.
    """
    r = torch.exp(-decay * dt)
    return 2.0 * r * torch.cos(2.0 * torch.pi * freq * dt), r * r


def bandpass_coeffs(freq: torch.Tensor, bandwidth: torch.Tensor, dt: float):
    """Coefficients of a two-pole band-pass: noise shaped into a band around ``freq``.

    Same recurrence as :func:`resonator_coeffs`; here the pole radius is set by the spectral width
    rather than a decay rate, so the filter rings continuously instead of dying away.

    Parameters
    ----------
    freq : torch.Tensor
        Center frequency of the band, in Hz.
    bandwidth : torch.Tensor
        Spectral width of the band, in Hz. Must be positive.
    dt : float
        Sample period, in seconds.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The ``(a1, a2)`` feedback coefficients.
    """
    r = torch.exp(-torch.pi * bandwidth * dt)
    return 2.0 * r * torch.cos(2.0 * torch.pi * freq * dt), r * r


def two_pole_block_kernels(a1: torch.Tensor, a2: torch.Tensor, n_samples: int):
    """Closed form of ``y[k] = a1*y[k-1] - a2*y[k-2] + drive[k]`` over one block of ``n_samples``.

    The recurrence is linear with constant coefficients, so a block is the sum of how the filter's memory rings
    on and how it answers the block's own drive::

        y[k] = free[k, 0] * y[-1] + free[k, 1] * y[-2] + sum_j forced[k, j] * drive[j]

    With ``h`` the impulse response (``h[0] = 1``, ``h[1] = a1``), ``free[k] = (h[k+1], -a2*h[k])`` and
    ``forced[k, j] = h[k-j]`` for ``j <= k``. Computed once at build, in float64, for coefficients that never
    change.

    Parameters
    ----------
    a1 : torch.Tensor
        ``(n,)`` first feedback coefficient of each filter.
    a2 : torch.Tensor
        ``(n,)`` second feedback coefficient of each filter, or a scalar shared by all of them.
    n_samples : int
        Block length ``K``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``free`` of shape ``(n, K, 2)`` and ``forced`` of shape ``(n, K, K)``, in the dtype of ``a1``.
    """
    a1_d, a2_d = torch.broadcast_tensors(a1.double(), a2.double())
    h = [torch.ones_like(a1_d), a1_d]
    for _ in range(n_samples - 1):
        h.append(a1_d * h[-1] - a2_d * h[-2])
    h = torch.stack(h, dim=-1)  # (n, K + 1)

    free = torch.stack((h[:, 1:], -a2_d[:, None] * h[:, :-1]), dim=-1)
    steps = torch.arange(n_samples, device=a1.device)
    lag = steps[:, None] - steps[None, :]  # (K, K), k - j
    forced = h[:, lag.clamp(min=0)] * (lag >= 0)
    return free.to(a1.dtype), forced.to(a1.dtype)
