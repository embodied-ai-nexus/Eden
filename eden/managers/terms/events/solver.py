"""Constraint-solver parameter events (contact stiffness tuning)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.managers.event_manager import EVENT_TERM_REGISTRY

if TYPE_CHECKING:
    import torch

    from eden.envs.base import EnvBase


@EVENT_TERM_REGISTRY.register()
def set_contact_sol_params(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    sol_params: tuple[float, ...],
    entity_links: dict[str, list[str] | None],
) -> None:
    """Set constraint-solver (solref/solimp) parameters on selected entity geoms.

    A ``STARTUP`` event: applies ``sol_params`` — ``(timeconst, dampratio, dmin,
    dmax, width, mid, power)``, see the MuJoCo "Solver parameters" docs — to the
    collision geoms of the named entities, leaving every other geom (and all
    joint/limit constraints) at defaults. ``entity_links`` maps entity name to
    the link names whose geoms to modify (``None`` = all of the entity's geoms).

    Note genesis mixes a contact pair's parameters as the mean of its two geoms'
    values, so a stiffened geom mildly stiffens *every* pair it participates in
    (e.g. an object also contacts the table), not only the pair of interest.

    ``envs_idx`` must be ``None``: genesis stores geom sol_params unbatched, so a
    per-env selection would be silently ignored (the write always applies to all
    envs) — rejected loudly here to keep the term out of per-env randomization.
    """
    if len(sol_params) != 7:
        raise ValueError(
            "sol_params must have 7 elements (timeconst, dampratio, dmin, dmax, width, mid, power); "
            f"got {len(sol_params)}"
        )
    if not (envs_idx is None or (isinstance(envs_idx, slice) and envs_idx == slice(None))):
        raise ValueError(
            "set_contact_sol_params only supports envs_idx=None: geom sol_params are unbatched in genesis, "
            "so a per-env selection would silently apply to ALL envs."
        )
    geoms_idx: list[int] = []
    for entity_name, links in entity_links.items():
        geoms = env.get_entity(entity_name).geoms
        if links is None:
            matched = [g.idx for g in geoms]
        else:
            link_names = {g.link.name for g in geoms}
            unmatched = [name for name in links if name not in link_names]
            if unmatched:
                raise ValueError(
                    f"entity '{entity_name}' has no collision geoms on links {unmatched}; "
                    f"available links with geoms: {sorted(link_names)}"
                )
            matched = [g.idx for g in geoms if g.link.name in links]
        if not matched:
            raise ValueError(f"entity '{entity_name}' has no collision geoms to apply sol_params to")
        geoms_idx += matched
    env.set_sol_params(sol_params, geoms_idx=geoms_idx)
