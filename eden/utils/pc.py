"""Point-cloud surface-sampling helpers for entities and links."""

from __future__ import annotations
from typing import TYPE_CHECKING

import genesis as gs
from genesis.utils.geom import transform_by_trans_quat
import numpy as np
import torch
import trimesh

from eden.utils.mesh import get_link_mesh

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink


def distribute_points_proportionally(n_points: int, link_areas: list[float]) -> list[int]:
    """Distribute a total number of points among links in proportion to their surface area.

    This function uses the largest remainder method (Hamilton method) to ensure the
    total number of distributed points is exactly ``n_points``.

    Parameters
    ----------
    n_points : int
        The total number of points to be distributed.
    link_areas : list of float
        A list of the surface areas of the links.

    Returns
    -------
    list of int
        The number of points to sample from each link.

    Raises
    ------
    ValueError
        If ``n_points`` is negative, ``link_areas`` is empty, or contains negative values.
    """
    if n_points < 0:
        raise ValueError("n_points cannot be negative.")
    if not link_areas:
        raise ValueError("link_areas cannot be empty.")
    if any(area < 0 for area in link_areas):
        raise ValueError("Link areas cannot be negative.")

    areas = np.array(link_areas)
    total_area = areas.sum()

    if total_area == 0:
        # All areas zero: distribute the points as evenly as possible.
        base_points = n_points // len(areas)
        points_dist = [base_points] * len(areas)
        remainder = n_points % len(areas)
        for i in range(remainder):
            points_dist[i] += 1
        return points_dist

    # 1. Calculate the ideal (proportional) number of points for each link
    proportions = areas / total_area
    ideal_points = proportions * n_points

    # 2. Allocate the initial integer part of the points
    initial_points = ideal_points.astype(int)

    # 3. Calculate the number of remaining points to distribute
    n_remaining = n_points - initial_points.sum()

    if n_remaining > 0:
        # 4. Distribute the remaining points based on the largest remainders
        remainders = ideal_points - initial_points

        # Get the indices of the links with the largest remainders
        remainder_indices = np.argsort(remainders)[-n_remaining:]

        # Add one point to each of these links
        initial_points[remainder_indices] += 1

    assert initial_points.sum() == n_points, "The sum of n_points_per_link is not equal to n_points"

    return initial_points.tolist()


def get_entity_surface_points(
    entity: RigidEntity,
    n_points: int = 100,
    radius=None,
    use_visual_mesh=False,
):
    """Sample surface points from every link of an entity, in each link's canonical (local) frame.

    Parameters
    ----------
    entity: RigidEntity
        entity to sample points from
    n_points: int
        total number of points, split across the links in proportion to their mesh surface area
    radius: float
        minimum distance between samples (``trimesh.sample.sample_surface_even``); None derives it from area and count
    use_visual_mesh: bool
        whether to use the visual mesh or the collision mesh

    Returns
    -------
    points_list: list[torch.Tensor | None]
        per link, its sampled surface points [n_points_per_link, 3], or None for a zero-area link.
        ``sample_surface_even`` may return fewer points than requested.
    """
    meshes = []
    for link in entity.links:
        combined = get_link_mesh(link, use_visual_mesh=use_visual_mesh)
        meshes.append(combined)

    areas = [mesh.area for mesh in meshes]
    n_points_list = distribute_points_proportionally(n_points, areas)

    # sample points from the mesh with the area weight
    res = []
    for n_p, mesh in zip(n_points_list, meshes):
        if mesh.area == 0:
            res.append(None)
            continue
        points, _ = trimesh.sample.sample_surface_even(mesh, count=n_p, radius=radius)
        res.append(torch.tensor(points, dtype=gs.tc_float).to(gs.device))
    return res


def update_entity_surface_points(entity: RigidEntity, points_list: list[torch.Tensor]):
    """Transform per-link canonical surface points into world space and concatenate them.

    Parameters
    ----------
    entity: RigidEntity
        entity to update points from
    points_list: list[torch.Tensor | None]
        per-link canonical points from :func:`get_entity_surface_points` (None entries are skipped)

    Returns
    -------
    points: torch.Tensor
        the points in world space, [B, n_points, 3] for a batched scene, [n_points, 3] otherwise
    """
    res = []
    for link, points in zip(entity.links, points_list):
        if points is None:
            continue
        res.append(update_sampled_points(link, points))
    return torch.cat(res, dim=-2)


def get_link_surface_points(
    link: RigidLink,
    n_points: int = 100,
    radius=None,
    use_visual_mesh=False,
):
    """Sample surface points from a single link, in canonical space.

    Parameters
    ----------
    link: RigidLink
        link to sample points from
    n_points: int
        number of points to sample on the link
    radius: float
        minimum distance between samples (``trimesh.sample.sample_surface_even``); None derives it from area and count
    use_visual_mesh: bool
        whether to use the visual mesh or the collision mesh

    Returns
    -------
    points: torch.Tensor
        sampled points on the surface of the link in canonical space [n_points, 3]

    Note
    ----
    Apply the link's current pose to move the points to world space::

        link_points = get_link_surface_points(link, 100)  # [n_points, 3]
        update_sampled_points(link, link_points)  # [B, n_points, 3] or [n_points, 3]
    """
    combined = get_link_mesh(link, use_visual_mesh=use_visual_mesh)
    points, _ = trimesh.sample.sample_surface_even(combined, count=n_points, radius=radius)
    return torch.tensor(points, dtype=gs.tc_float).to(gs.device)


def update_sampled_points(link: RigidLink, sampled_points):
    """Transform canonical sampled points into world space using the link's current pose.

    Parameters
    ----------
    link: RigidLink
        link whose current world pose is applied
    sampled_points: torch.Tensor
        sampled points from `get_link_surface_points` [n_points_per_link, 3]

    Returns
    -------
    torch.Tensor
        the points in world space, [B, n_points_per_link, 3] for a batched scene, [n_points_per_link, 3] otherwise
    """
    # relative=False: world frame (the canonical points are transformed into world space).
    # Genesis >=1.1.1 link getters default to the user frame, which would strip the alignment.
    pos = link.get_pos(relative=False)  # B, 3 or 3
    quat = link.get_quat(relative=False)  # B, 4 or 4
    if pos.ndim == 2:
        return transform_by_trans_quat(sampled_points.unsqueeze(0), pos.unsqueeze(1), quat.unsqueeze(1))
    elif pos.ndim == 1:
        return transform_by_trans_quat(sampled_points, pos.unsqueeze(0), quat.unsqueeze(0))
    else:
        gs.raise_exception(f"Invalid input dim: {sampled_points.shape}")


def chamfer_distance(
    pts_a: torch.Tensor,
    pts_b: torch.Tensor,
    valid_a: torch.Tensor | None = None,
    valid_b: torch.Tensor | None = None,
    *,
    bidirectional: bool = True,
    empty_distance: float = 0.0,
) -> torch.Tensor:
    """Masked, batched Chamfer distance between two point clouds.

    Parameters
    ----------
    pts_a, pts_b
        ``(B, N_a, 3)`` and ``(B, N_b, 3)``. ``B`` may be any leading
        batch shape — the last two dims are points × xyz — but the
        leading shapes must match **exactly** with no broadcasting
        (``pts_a.shape[:-2] == pts_b.shape[:-2]``; mismatch raises
        ``ValueError``). ``N_a`` and ``N_b`` are independent.
    valid_a, valid_b
        ``(B, N_a)`` / ``(B, N_b)`` bool masks selecting which points to
        include. ``None`` is equivalent to all-True.
    bidirectional
        ``True`` (default) returns ``(forward + backward) / 2``. ``False``
        returns the forward-only mean (``a → nearest b``).
    empty_distance
        Returned per-direction when one side has zero valid points and the
        other is non-empty. ``0.0`` (default) treats the asymmetric case as
        "no signal"; set to a positive sentinel (e.g. ``100.0`` to mirror
        the DexMachina contact reward kernel) when missing data should
        translate into a maximum penalty.

    Returns
    -------
    Tensor with shape ``pts_a.shape[:-2]`` — i.e. the leading batch
    dimensions of the inputs are preserved (a 1-D batch yields a
    ``(B,)`` output, a 2-D batch yields ``(B1, B2)``, etc.). Both-empty
    samples are always ``0.0``; one-empty samples take
    ``empty_distance``; otherwise the standard masked Chamfer.

    Notes
    -----
    Uses Euclidean distance (``p=2``) — not squared, matching the
    DexMachina formulation in ``reward_utils.chamfer_distance``. Per-side
    means divide by the number of valid points (not the cloud capacity),
    so reductions stay invariant to padding width.
    """
    if pts_a.shape[-1] != 3 or pts_b.shape[-1] != 3:
        raise ValueError(f"chamfer_distance expects xyz tail dim, got {pts_a.shape=} {pts_b.shape=}")
    if pts_a.shape[:-2] != pts_b.shape[:-2]:
        raise ValueError(f"batch shapes must match, got {pts_a.shape[:-2]=} {pts_b.shape[:-2]=}")

    if valid_a is None:
        valid_a = pts_a.new_ones(pts_a.shape[:-1], dtype=torch.bool)
    if valid_b is None:
        valid_b = pts_b.new_ones(pts_b.shape[:-1], dtype=torch.bool)
    # Masks on the points' device as bool, so ``masked_fill(~valid_*)`` cannot hit a device mismatch;
    # ``valid_*_f`` is the float cast that masks the per-direction sums.
    valid_a = valid_a.to(device=pts_a.device, dtype=torch.bool)
    valid_b = valid_b.to(device=pts_b.device, dtype=torch.bool)
    valid_a_f = valid_a.to(pts_a.dtype)
    valid_b_f = valid_b.to(pts_a.dtype)
    n_a = valid_a_f.sum(dim=-1)  # (B,)
    n_b = valid_b_f.sum(dim=-1)  # (B,)

    dist_ab = torch.cdist(pts_a, pts_b, p=2.0)  # (B, N_a, N_b)
    # Mask invalid columns/rows with +inf so .min skips them. We keep the
    # original distance tensors (no in-place writes) so autograd is clean.
    big = torch.finfo(dist_ab.dtype).max
    dist_ab_masked_b = dist_ab.masked_fill(~valid_b.unsqueeze(-2), big)
    forward_min = dist_ab_masked_b.min(dim=-1).values  # (B, N_a)
    forward_sum = (forward_min * valid_a_f).sum(dim=-1)
    forward = forward_sum / n_a.clamp(min=1)
    # When b is fully empty, ``forward_min`` is all ``big`` and ``forward``
    # overflows to +inf — safe under the final asymmetric-empty ``where``,
    # but we zero it explicitly so the intermediate is finite and a future
    # contributor cannot accidentally invert the where order.
    forward = torch.where(n_b > 0, forward, torch.zeros_like(forward))

    if bidirectional:
        dist_ab_masked_a = dist_ab.masked_fill(~valid_a.unsqueeze(-1), big)
        backward_min = dist_ab_masked_a.min(dim=-2).values  # (B, N_b)
        backward_sum = (backward_min * valid_b_f).sum(dim=-1)
        backward = backward_sum / n_b.clamp(min=1)
        backward = torch.where(n_a > 0, backward, torch.zeros_like(backward))
    else:
        backward = torch.zeros_like(forward)

    out = (forward + backward) / 2.0 if bidirectional else forward
    one_zero = (n_a == 0) ^ (n_b == 0)
    if empty_distance != 0.0:
        out = torch.where(one_zero, out.new_full(out.shape, empty_distance), out)
    # Both-zero is already 0 by the per-direction guards above; no extra
    # ``where`` needed.
    return out
