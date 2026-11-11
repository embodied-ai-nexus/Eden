"""``eden.engine.sensors._dsp``: closed-form block filtering against the sample-by-sample recurrence."""

from __future__ import annotations

import pytest
import torch

from eden.engine.sensors._dsp import bandpass_coeffs, resonator_coeffs, two_pole_block_kernels

DT_SUB = 5e-4


def _two_pole_loop(a1, a2, drive, y1, y2):
    """``y = a1*y1 - a2*y2 + drive``, one sample at a time: the definition the block kernels must reproduce."""
    out = []
    for k in range(drive.shape[-1]):
        y = a1 * y1 - a2 * y2 + drive[..., k]
        y2, y1 = y1, y
        out.append(y)
    return torch.stack(out, dim=-1)


@pytest.mark.parametrize("n_samples", [1, 2, 20])
@pytest.mark.parametrize(
    "coeffs",
    [
        lambda f: resonator_coeffs(f, torch.tensor(900.0, dtype=f.dtype), DT_SUB),
        lambda f: bandpass_coeffs(f, torch.tensor(400.0, dtype=f.dtype), DT_SUB),
    ],
    ids=["resonator", "bandpass"],
)
def test_block_kernels_match_the_recurrence_across_blocks(n_samples, coeffs):
    """Filtering block by block in closed form is the same signal as one uninterrupted recurrence.

    Three consecutive blocks carry the last two outputs across each boundary the way ``ActuationVoice`` does; a
    block length of 1 makes the carried memory straddle two blocks.
    """
    gen = torch.Generator().manual_seed(0)
    freq = torch.tensor([120.0, 400.0, 800.0], dtype=torch.float64)
    n_blocks, B = 3, 2
    drive = torch.randn((B, 3, n_blocks * n_samples), generator=gen, dtype=torch.float64)
    a1, a2 = coeffs(freq)
    zeros = torch.zeros((B, 3), dtype=torch.float64)
    expected = _two_pole_loop(a1, a2, drive, zeros, zeros)

    free, forced = two_pole_block_kernels(a1, a2, n_samples)
    tail = torch.zeros((B, 3, 2), dtype=torch.float64)  # [..., 0] is y[-2], [..., 1] is y[-1]
    blocks = []
    for b in range(n_blocks):
        x = drive[..., b * n_samples : (b + 1) * n_samples]
        y = torch.einsum("ekm,bem->bek", free, tail.flip(-1)) + torch.einsum("ekj,bej->bek", forced, x)
        tail = torch.cat((tail, y), dim=-1)[..., -2:]
        blocks.append(y)
    torch.testing.assert_close(torch.cat(blocks, dim=-1), expected, rtol=1e-9, atol=1e-12)
