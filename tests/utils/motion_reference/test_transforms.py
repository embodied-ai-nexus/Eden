"""Tests for ``eden.utils.motion_reference.transforms``: the anchor yaw-warp and the anchor-frame transforms."""

from __future__ import annotations

import torch

from eden.utils.motion_reference.transforms import (
    anchor_relative_links,
    links_pos_in_anchor_frame,
    links_quat_in_anchor_frame,
)


def _rand_quats(*shape: int) -> torch.Tensor:
    """Unit wxyz quaternions of the given (leading...) shape with a trailing 4."""
    q = torch.randn(*shape, 4)
    return q / q.norm(dim=-1, keepdim=True)


def test_identity_anchor_returns_motion_bodies():
    torch.manual_seed(0)
    n, num_links = 4, 5
    anchor_pos = torch.randn(n, 3)
    anchor_quat = _rand_quats(n)
    body_pos = torch.randn(n, num_links, 3)
    body_quat = _rand_quats(n, num_links)

    pos_rel, quat_rel = anchor_relative_links(anchor_pos, anchor_quat, anchor_pos, anchor_quat, body_pos, body_quat)

    assert torch.allclose(pos_rel, body_pos, atol=1e-5)
    # Quaternions equal up to the double-cover sign: |dot| == 1.
    dot = (quat_rel * body_quat).sum(-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)


def test_single_frame_equals_k1_future_window():
    torch.manual_seed(1)
    n, num_links = 3, 4
    ra_pos, ra_quat = torch.randn(n, 3), _rand_quats(n)
    ma_pos, ma_quat = torch.randn(n, 3), _rand_quats(n)
    mb_pos, mb_quat = torch.randn(n, num_links, 3), _rand_quats(n, num_links)

    pos_s, quat_s = anchor_relative_links(ra_pos, ra_quat, ma_pos, ma_quat, mb_pos, mb_quat)
    pos_k, quat_k = anchor_relative_links(
        ra_pos.unsqueeze(1),
        ra_quat.unsqueeze(1),
        ma_pos.unsqueeze(1),
        ma_quat.unsqueeze(1),
        mb_pos.unsqueeze(1),
        mb_quat.unsqueeze(1),
    )

    assert pos_k.shape == (n, 1, num_links, 3)
    assert quat_k.shape == (n, 1, num_links, 4)
    assert torch.equal(pos_k[:, 0], pos_s)
    assert torch.equal(quat_k[:, 0], quat_s)


def test_warp_preserves_intra_body_geometry():
    """The warp is rigid (a yaw and a translation), so pairwise body distances are invariant."""
    torch.manual_seed(2)
    n, num_links = 2, 6
    ra_pos, ra_quat = torch.randn(n, 3), _rand_quats(n)
    ma_pos, ma_quat = torch.randn(n, 3), _rand_quats(n)
    mb_pos, mb_quat = torch.randn(n, num_links, 3), _rand_quats(n, num_links)

    pos_rel, _ = anchor_relative_links(ra_pos, ra_quat, ma_pos, ma_quat, mb_pos, mb_quat)

    d_in = torch.cdist(mb_pos, mb_pos)  # (N, L, L)
    d_out = torch.cdist(pos_rel, pos_rel)
    assert torch.allclose(d_in, d_out, atol=1e-5)


def test_anchor_body_maps_to_robot_xy_and_motion_z():
    """A body at the motion anchor lands on the robot anchor's xy at the motion anchor's z."""
    torch.manual_seed(3)
    n, num_links = 4, 3
    ra_pos, ra_quat = torch.randn(n, 3), _rand_quats(n)
    ma_pos, ma_quat = torch.randn(n, 3), _rand_quats(n)
    mb_pos, mb_quat = torch.randn(n, num_links, 3), _rand_quats(n, num_links)
    mb_pos[:, 0] = ma_pos  # body 0 coincides with the motion anchor

    pos_rel, _ = anchor_relative_links(ra_pos, ra_quat, ma_pos, ma_quat, mb_pos, mb_quat)

    expected = torch.stack([ra_pos[:, 0], ra_pos[:, 1], ma_pos[:, 2]], dim=-1)  # (robot xy, motion z)
    assert torch.allclose(pos_rel[:, 0], expected, atol=1e-5)


def test_nan_robot_anchor_quat_falls_back_to_motion():
    """A NaN robot anchor quaternion falls back to the identity warp instead of spreading NaN."""
    torch.manual_seed(4)
    n, num_links = 3, 4
    ra_pos, ra_quat = torch.randn(n, 3), _rand_quats(n)
    ma_pos, ma_quat = torch.randn(n, 3), _rand_quats(n)
    mb_pos, mb_quat = torch.randn(n, num_links, 3), _rand_quats(n, num_links)
    ra_quat[1] = float("nan")  # env 1 robot anchor quaternion is NaN

    pos_rel, quat_rel = anchor_relative_links(ra_pos, ra_quat, ma_pos, ma_quat, mb_pos, mb_quat)

    # No NaN leaks into any env's output.
    assert not torch.isnan(pos_rel).any()
    assert not torch.isnan(quat_rel).any()
    # The NaN env falls back to an identity warp -> its relative bodies are the motion bodies.
    assert torch.allclose(pos_rel[1], mb_pos[1], atol=1e-5)
    dot = (quat_rel[1] * mb_quat[1]).sum(-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)


def test_anchor_link_maps_to_anchor_frame_origin():
    torch.manual_seed(5)
    n, num_links = 4, 3
    anchor_pos, anchor_quat = torch.randn(n, 3), _rand_quats(n)
    link_pos, link_quat = torch.randn(n, num_links, 3), _rand_quats(n, num_links)
    link_pos[:, 0] = anchor_pos  # link 0 coincides with the anchor pose
    link_quat[:, 0] = anchor_quat

    pos_b = links_pos_in_anchor_frame(anchor_pos, anchor_quat, link_pos)
    quat_b = links_quat_in_anchor_frame(anchor_quat, link_quat)

    assert torch.allclose(pos_b[:, 0], torch.zeros(n, 3), atol=1e-5)
    # Identity quaternion up to the double-cover sign: |w| == 1, xyz == 0.
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, -1)
    assert torch.allclose(quat_b[:, 0].abs(), identity, atol=1e-5)


def test_links_pos_in_anchor_frame_inverse_rotation_direction():
    """``R⁻¹ (p - p_anchor)`` in wxyz: under a +90° yaw, world +x is the anchor's -y."""
    s = 2**-0.5
    anchor_pos = torch.zeros(1, 3)
    anchor_quat = torch.tensor([[s, 0.0, 0.0, s]])  # +90° yaw about z (wxyz)
    link_pos = torch.tensor([[[1.0, 0.0, 0.0]]])  # (n=1, L=1, 3) on world +x

    pos_b = links_pos_in_anchor_frame(anchor_pos, anchor_quat, link_pos)
    assert torch.allclose(pos_b[:, 0], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5)


def test_links_in_anchor_frame_future_window_matches_single_frame():
    torch.manual_seed(6)
    n, k, num_links = 3, 4, 5
    anchor_pos, anchor_quat = torch.randn(n, 3), _rand_quats(n)
    link_pos, link_quat = torch.randn(n, k, num_links, 3), _rand_quats(n, k, num_links)

    pos_single = torch.stack(
        [links_pos_in_anchor_frame(anchor_pos, anchor_quat, link_pos[:, i]) for i in range(k)], dim=1
    )
    quat_single = torch.stack([links_quat_in_anchor_frame(anchor_quat, link_quat[:, i]) for i in range(k)], dim=1)
    anchor_pos_k = anchor_pos[:, None, :].expand(-1, k, -1)
    anchor_quat_k = anchor_quat[:, None, :].expand(-1, k, -1)
    pos_k = links_pos_in_anchor_frame(anchor_pos_k, anchor_quat_k, link_pos)
    quat_k = links_quat_in_anchor_frame(anchor_quat_k, link_quat)

    assert torch.allclose(pos_k, pos_single, atol=1e-6)
    assert torch.allclose(quat_k, quat_single, atol=1e-6)
