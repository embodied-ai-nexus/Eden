"""Capsule approximations of robot links, for the self-collision constraint (Sec. III-D).

Each link's surface is approximated by one capsule along its principal axis. A pair of links
contributes a non-penetration row ``phi + n^T (J_a - J_b) dv >= -eps`` once their capsules come within
an activation distance. Pairs that touch in the reference poses (adjacent links, a hand's palm and
its finger roots) are excluded, as is usual for self-collision filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from eden.utils.geom import segment_closest_points


@dataclass
class LinkCapsules:
    """One capsule per link, in link frames.

    Attributes
    ----------
    link : np.ndarray
        ``(C,)`` rig link index of each capsule.
    a, b : np.ndarray
        ``(C, 3)`` segment endpoints in the link frame.
    radius : np.ndarray
        ``(C,)``.
    pairs : np.ndarray
        ``(P, 2)`` capsule index pairs that are checked.
    """

    link: np.ndarray
    a: np.ndarray
    b: np.ndarray
    radius: np.ndarray
    pairs: np.ndarray

    def world(self, link_pos: np.ndarray, link_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World endpoints ``(C, 3)``, ``(C, 3)``."""
        rot = link_rot[self.link]
        pos = link_pos[self.link]
        return pos + np.einsum("cab,cb->ca", rot, self.a), pos + np.einsum("cab,cb->ca", rot, self.b)


def fit_capsule(points: np.ndarray, radius_quantile: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a capsule to surface points: principal axis, radius at a quantile of the distances to it.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` surface points (link frame).
    radius_quantile : float
        Quantile of the points' distance to the axis taken as the radius; below 1 the capsule sits inside
        protruding features rather than inflating around them.

    Returns
    -------
    a, b : np.ndarray
        Segment endpoints.
    radius : float
    """
    centre = points.mean(axis=0)
    centred = points - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    s = centred @ axis
    radial = np.linalg.norm(centred - s[:, None] * axis[None], axis=1)
    radius = float(np.quantile(radial, radius_quantile))
    lo, hi = float(s.min()) + radius, float(s.max()) - radius
    if lo > hi:
        lo = hi = 0.5 * (s.min() + s.max())
    return centre + lo * axis, centre + hi * axis, radius


def build_link_capsules(
    link_points: dict[int, np.ndarray],
    link_parent: np.ndarray,
    reference_poses: list[tuple[np.ndarray, np.ndarray]],
    radius_quantile: float = 0.8,
    exclusion_margin: float = 0.005,
    link_names: list[str] | None = None,
    exclude: list[tuple[str, str]] | None = None,
) -> LinkCapsules:
    """Fit capsules and choose the link pairs to check.

    Parameters
    ----------
    link_points : dict[int, np.ndarray]
        ``{rig link index: (N, 3) surface points in its frame}`` for links with geometry.
    link_parent : np.ndarray
        Rig link parent indices.
    reference_poses : list[tuple[np.ndarray, np.ndarray]]
        Link poses (positions, rotations) at reference configurations; a pair closer than
        ``exclusion_margin`` in any of them is not checked.
    radius_quantile : float
        See :func:`fit_capsule`.
    exclusion_margin : float
        Clearance below which a pair counts as touching in a reference pose.
    link_names : list[str], optional
        Rig link names (indexed like ``link_parent``), needed by ``exclude``.
    exclude : list[tuple[str, str]], optional
        ``(regex_a, regex_b)`` link-name pairs never checked, in either order; for neighbouring links a
        capsule cannot separate (side-by-side fingers, whose capsules overlap as soon as they curl).
    """
    links = sorted(link_points)
    a, b, radius = [], [], []
    for li in links:
        pa, pb, r = fit_capsule(link_points[li], radius_quantile)
        a.append(pa)
        b.append(pb)
        radius.append(r)
    caps = LinkCapsules(np.array(links), np.array(a), np.array(b), np.array(radius), np.zeros((0, 2), dtype=np.int64))
    ii, jj = np.triu_indices(len(links), k=1)
    li, lj = caps.link[ii], caps.link[jj]
    # Adjacent links (parent/child, siblings, grandparent) always touch at their joints.
    gp = np.where(link_parent >= 0, link_parent[np.maximum(link_parent, 0)], -1)
    adjacent = (
        (link_parent[li] == lj)
        | (link_parent[lj] == li)
        | ((link_parent[li] == link_parent[lj]) & (link_parent[li] >= 0))
        | (gp[li] == lj)
        | (gp[lj] == li)
    )
    keep = ~adjacent
    if exclude:
        if link_names is None:
            raise ValueError("build_link_capsules: exclude needs link_names to match its patterns against.")
        names = np.array(link_names)
        for pat_a, pat_b in exclude:
            a = np.array([re.fullmatch(pat_a, n) is not None for n in names])
            b = np.array([re.fullmatch(pat_b, n) is not None for n in names])
            keep &= ~((a[li] & b[lj]) | (b[li] & a[lj]))
    for pos, rot in reference_poses:
        wa, wb = caps.world(pos, rot)
        ca, cb = segment_closest_points(wa[ii], wb[ii], wa[jj], wb[jj])
        clearance = np.linalg.norm(ca - cb, axis=1) - caps.radius[ii] - caps.radius[jj]
        keep &= clearance > exclusion_margin
    caps.pairs = np.stack([ii[keep], jj[keep]], axis=1)
    return caps
