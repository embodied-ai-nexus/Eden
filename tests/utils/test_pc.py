"""Tests for ``eden.utils.pc.chamfer_distance``, on plain tensors."""

from __future__ import annotations

import math

import pytest
import torch

from eden.utils.pc import chamfer_distance


def _full(*shape):
    return torch.ones(*shape, dtype=torch.bool)


def test_a_cloud_is_at_zero_distance_from_itself():
    pts = torch.randn(4, 16, 3)
    torch.testing.assert_close(
        chamfer_distance(pts, pts, _full(4, 16), _full(4, 16)), torch.zeros(4), atol=1e-5, rtol=0
    )


def test_it_is_symmetric_in_its_two_clouds_and_their_masks():
    a, b = torch.randn(2, 5, 3), torch.randn(2, 7, 3)
    va = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])
    vb = torch.tensor([[True, False, True, True, True, False, True], [True] * 7])
    torch.testing.assert_close(
        chamfer_distance(a, b, va, vb, empty_distance=100.0), chamfer_distance(b, a, vb, va, empty_distance=100.0)
    )


def test_well_separated_clouds_translated_by_d_are_d_apart():
    base = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    shifted = base.clone()
    shifted[..., 0] += 0.25
    torch.testing.assert_close(chamfer_distance(base, shifted, _full(1, 2), _full(1, 2)), torch.tensor([0.25]))


def test_masked_points_are_ignored():
    a, b = torch.randn(2, 5, 3), torch.randn(2, 7, 3)
    padded = torch.cat([a, torch.full((2, 3, 3), 1e6)], dim=1)
    mask = torch.cat([_full(2, 5), torch.zeros(2, 3, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(
        chamfer_distance(padded, b, mask, _full(2, 7)), chamfer_distance(a, b, _full(2, 5), _full(2, 7))
    )


def test_empty_clouds():
    """Both empty: 0. One empty: ``empty_distance`` (0 by default), per sample."""
    a, b = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    empty = torch.zeros(2, 4, dtype=torch.bool)
    torch.testing.assert_close(chamfer_distance(a, b, empty, empty), torch.zeros(2))
    torch.testing.assert_close(chamfer_distance(a, b, _full(2, 4), empty), torch.zeros(2))
    torch.testing.assert_close(
        chamfer_distance(a, b, _full(2, 4), empty, empty_distance=100.0), torch.full((2,), 100.0)
    )
    mixed = chamfer_distance(a, b, _full(2, 4), torch.tensor([[True] * 4, [False] * 4]), empty_distance=42.0)
    assert mixed[1].item() == pytest.approx(42.0) and mixed[0].item() != pytest.approx(42.0)


def test_forward_only_is_the_mean_distance_from_the_first_cloud():
    t = torch.linspace(0.0, 1.0, 4).unsqueeze(-1)
    a = (torch.tensor([1.0, 0.0, 0.0]) * t).unsqueeze(0)
    b = (torch.tensor([1.0, 0.0, 0.0]) * t + torch.tensor([0.0, 1.0, 0.0])).unsqueeze(0)
    b = torch.cat([b, torch.tensor([[[0.5, 5.0, 0.0]]])], dim=1)  # an outlier only the backward term sees
    forward = chamfer_distance(a, b, _full(1, 4), _full(1, 5), bidirectional=False)
    assert forward.item() == pytest.approx(1.0, abs=1e-5)
    assert chamfer_distance(a, b, _full(1, 4), _full(1, 5), bidirectional=True).item() > forward.item()


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [((1, 4, 2), (1, 4, 3), "xyz tail dim"), ((2, 4, 3), (3, 4, 3), "batch shapes")],
)
def test_bad_shapes_raise(a, b, match):
    with pytest.raises(ValueError, match=match):
        chamfer_distance(torch.randn(*a), torch.randn(*b))


def test_gradients_are_finite_and_zero_through_an_empty_sample():
    """The empty-side branches contribute a constant, and must not poison neighbouring samples with NaN."""
    a = torch.randn(3, 5, 3, requires_grad=True)
    b = torch.randn(3, 6, 3, requires_grad=True)
    va = torch.tensor([[True] * 5, [False] * 5, [True] * 5])
    vb = torch.tensor([[True] * 6, [True] * 6, [False] * 6])
    chamfer_distance(a, b, va, vb, empty_distance=100.0).sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    torch.testing.assert_close(a.grad[1], torch.zeros(5, 3))
    torch.testing.assert_close(b.grad[2], torch.zeros(6, 3))
    assert a.grad[0].abs().sum() > 0


def test_matches_the_dexmachina_reference_values():
    """Hand-computed from DexMachina's ``reward_utils.chamfer_distance`` on a masked, two-sample input."""
    a = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    b = torch.tensor(
        [
            [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.1]],
            [[6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    va = torch.tensor([[True, True, True, True], [True, False, False, False]])
    vb = torch.tensor([[True, True, True, False], [True, False, False, False]])
    # Sample 0: a3 -> b0 is sqrt(1.01), every other nearest pair 0.1; sample 1: the one pair is 1.0 apart.
    forward = (0.3 + math.sqrt(1.01)) / 4.0
    expected = torch.tensor([(forward + 0.1) / 2.0, 1.0])
    torch.testing.assert_close(chamfer_distance(a, b, va, vb), expected, atol=1e-5, rtol=0)
