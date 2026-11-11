"""Entropic optimal transport between surface point clouds (OTRetarget Sec. III-B, Eq. 3-4).

A demonstration's interaction targets live on the *human* (or demonstrated-object) surface, the
residuals on the *robot* (or substitute-object) surface. The correspondence between the two is the
argmax of an entropically regularised transport plan between the two clouds, each centred and scaled
to unit root-mean-square radius so a size mismatch does not decide the matching. Unlike a
nearest-neighbour assignment, the marginal constraints force every source sample to receive mass, so
limb extremities (fingertips, heels) are not left unmatched.

The paper's cost is the squared distance between the normalised clouds. With outward normals given, the
cost also carries ``normal_weight * |n_i - n_p|^2``: a robot phalanx is not the human one's scaled copy (a
flatter Wuji phalanx against a round human one), and position alone matches about a quarter of a robot
finger's probes to the opposite face of the human finger (pad to nail), whose demonstrated distances then
pull that face of the robot finger *away* from a grasped object. A weight of 0.5 (squared unit-normal
differences lie in [0, 4], normalised squared distances are ~1) matches 96% of the probes to a like-facing
source at unchanged coverage.
"""

from __future__ import annotations

import numpy as np
import torch


def normalize_cloud(points: np.ndarray) -> np.ndarray:
    """Centre a point cloud on its centroid and scale it to unit root-mean-square radius.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` points.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` normalised points.
    """
    points = np.asarray(points, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    rms = float(np.sqrt((centered**2).sum(axis=1).mean()))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError(
            f"normalize_cloud: a cloud of {len(points)} points has root-mean-square radius {rms!r}, so it cannot "
            f"be scaled to unit radius. Every point coincides: sample the surface with more than one distinct point."
        )
    return centered / rms


def sinkhorn(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    max_iters: int = 2000,
    tol: float = 1e-9,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Solve ``min_P <P, C> + eps * sum P (log P - 1)`` over couplings with marginals ``a``, ``b``.

    Log-domain Sinkhorn (Cuturi 2013), stable for small ``epsilon`` relative to the cost scale.

    Parameters
    ----------
    cost : np.ndarray
        ``(N, M)`` cost matrix ``C``.
    a : np.ndarray
        ``(N,)`` source marginal, positive, summing to one.
    b : np.ndarray
        ``(M,)`` target marginal, positive, summing to one.
    epsilon : float
        Entropic regularisation ``eps_ot`` (> 0).
    max_iters : int
        Iteration budget.
    tol : float
        Stop once the L1 violation of the source marginal falls below this.
    device : str | torch.device
        Where the iterations run. The caller names it (the retargeter passes its env's device) rather than this
        function picking any available GPU, so a ``--cpu`` run stays off a shared GPU.

    Returns
    -------
    np.ndarray
        ``(N, M)`` transport plan ``P*``.
    """
    if epsilon <= 0.0:
        raise ValueError(f"sinkhorn: epsilon={epsilon!r} must be > 0 (the paper uses eps_ot = 0.1).")
    scaled = -torch.as_tensor(np.asarray(cost), dtype=torch.float64, device=device) / epsilon
    log_a = torch.log(torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device))
    log_b = torch.log(torch.as_tensor(np.asarray(b), dtype=torch.float64, device=device))
    # Scaled dual potentials u = f / eps, v = g / eps.
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for it in range(max_iters):
        u = log_a - torch.logsumexp(scaled + v[None, :], dim=1)
        v = log_b - torch.logsumexp(scaled + u[:, None], dim=0)
        # After the v-update the target marginal is exact; the source one measures convergence.
        if it % 10 == 9:
            row_mass = torch.exp(torch.logsumexp(scaled + u[:, None] + v[None, :], dim=1))
            if float((row_mass - torch.exp(log_a)).abs().sum()) < tol:
                break
    return torch.exp(scaled + u[:, None] + v[None, :]).cpu().numpy()


def transport_correspondence(
    source: np.ndarray,
    target: np.ndarray,
    epsilon: float = 0.1,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    normal_weight: float = 0.0,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """For each target point, the source point it receives the most mass from (``src(p)``, Eq. 4).

    Both clouds are normalised (:func:`normalize_cloud`) before the squared-Euclidean cost is built.
    The marginals are uniform over each cloud's points, or proportional to ``*_weights`` (the surface area
    a sample stands for) when a cloud is sampled at several densities: uniform marginals would hand the
    densely sampled part a larger share of the other surface.

    Parameters
    ----------
    source : np.ndarray
        ``(N, 3)`` source cloud (the human part, or the demonstrated object), in a common pose with
        ``target`` (the T-pose for human/robot, the object frame for object/object).
    target : np.ndarray
        ``(M, 3)`` target cloud (the robot link group, or the substitute object).
    epsilon : float
        Entropic regularisation ``eps_ot``.
    source_normals, target_normals : np.ndarray, optional
        ``(N, 3)`` / ``(M, 3)`` unit outward normals, required when ``normal_weight`` > 0.
    normal_weight : float
        Weight of the normal-agreement cost ``|n_i - n_p|^2`` (0: the paper's position-only cost).
    source_weights, target_weights : np.ndarray, optional
        ``(N,)`` / ``(M,)`` positive sample weights (normalised here); uniform when omitted.
    device : str | torch.device
        Where the Sinkhorn iterations run (see :func:`sinkhorn`).

    Returns
    -------
    np.ndarray
        ``(M,)`` int indices into ``source``.
    """
    if len(source) == 0 or len(target) == 0:
        raise ValueError(
            f"transport_correspondence: cannot match a cloud of {len(source)} source points to one of {len(target)} "
            f"target points; both sides of a correspondence need surface samples."
        )
    src = normalize_cloud(source)
    tgt = normalize_cloud(target)
    cost = ((src[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=-1)
    if normal_weight > 0.0:
        if source_normals is None or target_normals is None:
            raise ValueError(
                f"transport_correspondence: normal_weight={normal_weight!r} needs source_normals and target_normals; "
                f"pass both, or normal_weight=0 for the position-only cost."
            )
        diff = np.asarray(source_normals)[:, None, :] - np.asarray(target_normals)[None, :, :]
        cost = cost + normal_weight * (diff**2).sum(axis=-1)
    a = _marginal(source_weights, len(src))
    b = _marginal(target_weights, len(tgt))
    plan = sinkhorn(cost, a, b, epsilon, device=device)
    return plan.argmax(axis=0)


def _marginal(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n,) or not np.all(weights > 0):
        raise ValueError(
            f"transport_correspondence: sample weights must be {n} positive values; got shape {weights.shape}."
        )
    return weights / weights.sum()


def group_correspondences(
    source_points: np.ndarray,
    source_labels: list[str],
    target_points: np.ndarray,
    target_labels: list[str],
    groups: list[tuple[tuple[str, ...], tuple[str, ...]]],
    epsilon: float = 0.1,
    source_normals: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    normal_weight: float = 0.0,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one transport plan per pre-assigned (source parts, target parts) group.

    Parameters
    ----------
    source_points : np.ndarray
        ``(N, 3)`` source samples (human probes in the rest pose).
    source_labels : list[str]
        ``N`` part names (the human joint each probe lies on).
    target_points : np.ndarray
        ``(M, 3)`` target samples (robot probes in the T-pose, same frame as the source).
    target_labels : list[str]
        ``M`` part names (the robot link each probe rides on).
    groups : list[tuple[tuple[str, ...], tuple[str, ...]]]
        ``((source parts...), (target parts...))``; a target part may appear in one group only.
    epsilon : float
        Entropic regularisation.
    source_normals, target_normals : np.ndarray, optional
        Unit outward normals of the samples (see :func:`transport_correspondence`).
    normal_weight : float
        Weight of the normal-agreement cost.
    source_weights, target_weights : np.ndarray, optional
        Sample weights (the area each sample stands for, see :func:`transport_correspondence`).
    device : str | torch.device
        Where the Sinkhorn iterations run (see :func:`sinkhorn`).

    Returns
    -------
    src : np.ndarray
        ``(M,)`` source index per target sample, ``-1`` where the target's part is in no group.
    group : np.ndarray
        ``(M,)`` group index per target sample, ``-1`` likewise.
    """
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)
    src = np.full(len(target_points), -1, dtype=np.int64)
    group_of = np.full(len(target_points), -1, dtype=np.int64)
    seen: dict[str, int] = {}
    for gi, (source_parts, target_parts) in enumerate(groups):
        for part in target_parts:
            if part in seen:
                raise ValueError(
                    f"group_correspondences: target part {part!r} is in groups {seen[part]} and {gi}; "
                    f"each robot link may be matched to one group of human parts only."
                )
            seen[part] = gi
        s_idx = np.flatnonzero(np.isin(source_labels, source_parts))
        t_idx = np.flatnonzero(np.isin(target_labels, target_parts))
        if len(s_idx) == 0 or len(t_idx) == 0:
            no_source = [p for p in source_parts if p not in set(source_labels.tolist())]
            no_target = [p for p in target_parts if p not in set(target_labels.tolist())]
            raise ValueError(
                f"group_correspondences: group {gi} {source_parts} <-> {target_parts} has {len(s_idx)} source and "
                f"{len(t_idx)} target samples; every group needs samples on both sides. Without samples: source "
                f"parts {no_source}, target parts {no_target} (a target link has none when it has no geometry or "
                f"probe_exclude_links drops it). Sampled source parts: {sorted(set(source_labels.tolist()))}; "
                f"sampled target parts: {sorted(set(target_labels.tolist()))}."
            )
        src[t_idx] = s_idx[
            transport_correspondence(
                source_points[s_idx],
                target_points[t_idx],
                epsilon,
                None if source_normals is None else source_normals[s_idx],
                None if target_normals is None else target_normals[t_idx],
                normal_weight,
                None if source_weights is None else source_weights[s_idx],
                None if target_weights is None else target_weights[t_idx],
                device,
            )
        ]
        group_of[t_idx] = gi
    return src, group_of
