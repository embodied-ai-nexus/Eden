"""Pure-tensor contact-aggregation helpers (backend of ``RigidEntity.get_grouped_contacts``).

These have no dependency on the entity abstraction, so they live with the other
math-only helpers (next to :mod:`eden.utils.pc`) and can be unit-tested without
importing :mod:`eden.entities.rigid` (which pulls in the support-surface mixin
and triggers ``genesis.utils.array_class`` top-level code that requires
``gs.init()``).

Scaling note: the ``(B, C, n_a, n_b)`` broadcast in
:func:`aggregate_grouped_contacts` is fine at link-level pair counts (≤ ~30 per
side) but must become a ``scatter_add`` before geom-level use (~900 pairs).
"""

from __future__ import annotations

import weakref

import torch

#: Pre-resolved index tensors whose bounds were already checked, by ``id``:
#: ``id -> (weakref, num_links, _version)``. The weakref guards against a recycled ``id`` once the tensor is
#: gone, and ``_version`` against an in-place write since the check.
_BOUNDS_CHECKED: dict[int, tuple[weakref.ref, int, int]] = {}


def resolve_local_link_idx(
    idx: torch.Tensor | list[int] | None,
    num_links: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalise a local-link-index argument into a 1-D ``long`` tensor on ``device``.

    Parameters
    ----------
    idx : torch.Tensor | list[int] | None
        Local link indices; ``None`` is expanded to ``torch.arange(num_links)``.
    num_links : int
        Exclusive upper bound for valid indices.
    device : torch.device
        Device the returned tensor must live on.

    Hot-path note: when ``idx`` is already a 1-D ``long`` tensor on
    ``device``, this returns it directly with no copy or H2D transfer.
    Callers that hit the per-step path (reward / observation terms)
    should pre-resolve the indices once at ``build()`` time and pass the
    cached tensor each step — see the docstring on
    :meth:`RigidEntity.get_grouped_contacts`.
    """
    if idx is None:
        return torch.arange(num_links, device=device)
    if (
        isinstance(idx, torch.Tensor)
        and idx.dtype == torch.long
        and idx.device == device
        and idx.ndim == 1
        and idx.numel() > 0
    ):
        # Fast path: caller pre-resolved at build time. Skip dtype/device
        # checks (they are no-ops here) and the ``.tolist()`` round-trip
        # in the bounds error message. Bounds are verified on the first call
        # for this tensor only: reading the check back is a GPU->CPU sync, and
        # the same cached tensor comes back every step.
        # The version counter is part of the check, so an in-place write re-checks it; an inference tensor
        # tracks none and is checked on every call.
        try:
            version = idx._version
        except RuntimeError:
            version = None
        checked = _BOUNDS_CHECKED.get(id(idx))
        if checked is not None and checked[0]() is idx and checked[1:] == (num_links, version):
            return idx
        if (idx < 0).any() or (idx >= num_links).any():
            raise IndexError(f"link index out of range [0, {num_links}): {idx.tolist()}")
        if version is not None:
            _BOUNDS_CHECKED[id(idx)] = (
                weakref.ref(idx, lambda _, key=id(idx), cache=_BOUNDS_CHECKED: cache.pop(key, None)),
                num_links,
                version,
            )
        return idx
    t = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx)
    t = t.to(device=device, dtype=torch.long)
    if t.ndim != 1:
        raise ValueError(f"link index must be 1-D, got shape {tuple(t.shape)}")
    if t.numel() == 0:
        raise ValueError("link index cannot be empty")
    if (t < 0).any() or (t >= num_links).any():
        raise IndexError(f"link index out of range [0, {num_links}): {t.tolist()}")
    return t


def aggregate_grouped_contacts(
    *,
    link_a: torch.Tensor,
    link_b: torch.Tensor,
    position: torch.Tensor,
    force: torch.Tensor,
    valid_mask: torch.Tensor,
    a_global: torch.Tensor,
    b_global: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Force-norm-weighted per-link-pair contact aggregation.

    Parameters
    ----------
    link_a, link_b : torch.Tensor
        ``(B, C)`` global link indices of each contact's two sides, as
        returned by the parallelised Genesis ``get_contacts`` (env-leading).
    position : torch.Tensor
        ``(B, C, 3)`` world-frame contact positions.
    force : torch.Tensor
        ``(B, C, 3)`` contact force vectors; only their norms are used.
    valid_mask : torch.Tensor
        ``(B, C)`` bool mask of real (non-padding) contacts.
    a_global, b_global : torch.Tensor
        1-D global link indices selecting which links populate axes 1 and 2
        of the output. Must be disjoint (see Raises). Both Genesis
        orderings (a→A, b→B) and (a→B, b→A) are folded into the same pair so
        the caller never has to care which side Genesis labelled "a".

    Genesis's ``get_contacts(with_entity=other)`` does **not** canonicalize
    contact-pair ordering: its ``valid_mask`` is built as
    ``(a∈self ∨ b∈self) ∧ (a∈other ∨ b∈other)``
    (see ``RigidEntity.get_contacts`` in genesis), so both orderings reach
    this function — handling both is required, not optional.

    Returns
    -------
    dict with:

    - ``position`` : ``(B, n_a, n_b, 3)`` — force-norm-weighted mean
      contact position; zero where ``valid`` is False.
    - ``valid`` : ``(B, n_a, n_b)`` bool — True iff at least one
      contact pair contributed a **non-zero** force-norm sum
      (computed as ``force_norm_sum > 0``, matching the DexMachina
      reference). Pairs whose contacts all carry zero force — e.g. the
      first frame after make/break or quasi-static initial overlap —
      are reported invalid because no force-weighted mean position is
      defined for them. ``valid_strict_geometry = pair_mask.any(...)``
      is **not** exposed; callers needing geometric-only contact
      detection should bool-AND the raw ``get_contacts(...)["valid_mask"]``
      themselves.
    - ``force_norm_sum`` : ``(B, n_a, n_b)`` — Σ ||f_i|| over contacts
      contributing to the pair (the denominator used to weight the
      position mean). Note this is the sum of magnitudes, **not** the
      magnitude of the summed force vector — opposing-direction
      contacts do not cancel here.

    Raises
    ------
    ValueError
        If ``a_global`` and ``b_global`` overlap (checked for CPU tensors
        only — on-device groups skip the check to avoid a per-step GPU sync;
        there disjointness is a caller-guaranteed precondition, which
        ``RigidEntity.get_grouped_contacts`` establishes structurally by
        rejecting self-contact and offsetting per-entity link ranges).
    """
    # Disjointness guard, CPU groups only: an `if` on a CUDA bool would force a
    # GPU->host sync every call on the per-step contact-reward path — where the
    # condition is structurally impossible anyway (RigidEntity.get_grouped_contacts
    # rejects self-contact and distinct entities have disjoint global link
    # ranges). Direct callers (tests, offline tools) typically pass CPU tensors
    # and get the loud failure.
    if a_global.device.type == "cpu" and torch.isin(a_global, b_global).any():
        raise ValueError(
            "a_global and b_global must be disjoint link-index groups — a link on "
            "both sides would double-count each of its contacts into two output "
            "cells. For self-contact queries use RigidEntity.get_contacts(with_entity=<entity>)."
        )
    la = link_a.unsqueeze(-1)  # (B, C, 1)
    lb = link_b.unsqueeze(-1)  # (B, C, 1)
    # Each match tensor below is (B, C, n_a) or (B, C, n_b); they combine into
    # the full (B, C, n_a, n_b) pair mask via outer-AND on the membership axes.
    a_in_self = (la == a_global.view(1, 1, -1)).unsqueeze(-1)  # (B, C, n_a, 1)
    b_in_other = (lb == b_global.view(1, 1, -1)).unsqueeze(-2)  # (B, C, 1,  n_b)
    a_in_other = (la == b_global.view(1, 1, -1)).unsqueeze(-2)  # (B, C, 1,  n_b)
    b_in_self = (lb == a_global.view(1, 1, -1)).unsqueeze(-1)  # (B, C, n_a, 1)
    pair_mask = (a_in_self & b_in_other) | (b_in_self & a_in_other)
    pair_mask = pair_mask & valid_mask[..., None, None]  # (B, C, n_a, n_b)

    force_norm = force.norm(dim=-1)  # (B, C)
    # Select, never multiply by the mask: the padding rows past a env's real contacts carry whatever the solver's
    # buffer held, NaN included, and NaN * 0 is NaN -- one stale slot would poison the whole pair's sum (a NaN reward
    # while the thresholded contact flags, NaN > x being False, stay finite).
    weight = torch.where(
        pair_mask,
        force_norm.unsqueeze(-1).unsqueeze(-1),
        torch.zeros((), dtype=force_norm.dtype, device=force_norm.device),
    )
    weighted_pos = torch.where(
        pair_mask.unsqueeze(-1),
        position.unsqueeze(-2).unsqueeze(-2),
        torch.zeros((), dtype=position.dtype, device=position.device),
    )
    weighted_pos_sum = (weighted_pos * weight.unsqueeze(-1)).sum(dim=1)  # (B, n_a, n_b, 3)
    force_norm_sum = weight.sum(dim=1)  # (B, n_a, n_b)
    valid = force_norm_sum > 0
    mean_pos = torch.where(
        valid.unsqueeze(-1),
        weighted_pos_sum / force_norm_sum.unsqueeze(-1).clamp(min=1e-12),
        torch.zeros_like(weighted_pos_sum),
    )
    return {"position": mean_pos, "valid": valid, "force_norm_sum": force_norm_sum}
