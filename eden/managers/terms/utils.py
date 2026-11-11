"""Shared helpers for manager terms.

Small building blocks reused across observation / reward term ``build()`` and ``compute()`` so the same
idiom (gravity projection, link-index resolution, base height over the ground, per-DOF std from a regex dict)
isn't re-implemented per term. Resolve-once helpers (``resolve_link_idx`` / ``resolve_terrain`` /
``resolve_per_dof_std`` / ``resolve_metric_manager``) are meant to be called in ``build()`` — a function term,
which has none, calls them per step; the math helpers (``gravity_vec`` / ``projected_gravity`` /
``base_height_above_ground``) are cheap enough for the hot path.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.utils.misc import resolve_named_entity

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase
    from eden.managers.metric_manager import MetricManager, MetricTerm

MetricT = TypeVar("MetricT", bound="MetricTerm")


def gravity_vec(num_envs: int, device) -> torch.Tensor:
    """World-frame down vector ``[0, 0, -1]`` expanded to ``(num_envs, 3)``.

    Cache once in ``build()``; pass to :func:`projected_gravity` as the gravity input and/or use directly as
    the upright reference vector. Built on the device rather than copied from a host list, so a function term
    that has no ``build()`` can call it per step without a GPU->CPU sync.
    """
    down = torch.zeros(3, device=device, dtype=torch.float32)
    down[2:].fill_(-1.0)  # a slice fill; `down[2] = -1.0` would copy a host scalar, which syncs
    return down.expand(num_envs, -1)


def projected_gravity(quat: torch.Tensor, grav: torch.Tensor) -> torch.Tensor:
    """Express world-frame ``grav`` in the frame of ``quat`` (wxyz).

    ``quat`` is ``(..., 4)`` and ``grav`` is the matching world-frame vector (e.g. from :func:`gravity_vec`).
    It is ``inv_transform_by_quat`` with the operands in the opposite order.
    """
    return inv_transform_by_quat(grav, quat)


def resolve_metric_manager(env: EnvBase, *, term: str, metric_name: str) -> MetricManager:
    """Return ``env.metric_manager`` after checking that it exists and that ``metric_name`` is one of its terms.

    For a metric pull term's ``build()``: the metric manager is constructed ahead of the termination, reward
    and observation managers (``RLEnvBase._manager_order``), so these mistakes surface at env construction
    rather than on the first step. ``term`` is the pulling term's registry key. An empty ``metric_name`` —
    the class terms' default, i.e. a ``configure()`` that forgot it — raises a ``ValueError`` saying so; an
    unknown one raises the manager's own ``KeyError``. Both list the active metrics.
    """
    if env.metric_manager is None:
        raise ValueError(
            f"{term}(metric_name='{metric_name}') requires a metric term, but no metric manager is configured."
        )
    if not metric_name:
        raise ValueError(
            f"{term}: metric_name is not set — pass metric_name=<one of the active metrics> to configure(). "
            f"Active metrics: {env.metric_manager.active_terms}"
        )
    env.metric_manager.get_value(metric_name)
    return env.metric_manager


def resolve_metric_term(env: EnvBase, *, term: str, metric_name: str, expected: type[MetricT]) -> MetricT:
    """The metric term ``metric_name``, checked to be an ``expected``: for a term whose ``build()`` reads a task's
    metric (its ink, its grade) through the metric's own attributes rather than its value.

    Fails as :func:`resolve_metric_manager` does for a missing manager or name, and with a ``TypeError`` naming the
    metrics that are an ``expected`` when ``metric_name`` is some other metric. ``term`` is the reading term's name.
    """
    metrics = resolve_metric_manager(env, term=term, metric_name=metric_name)
    metric = metrics.get_term(metric_name)
    if not isinstance(metric, expected):
        matching = [name for name in metrics.active_terms if isinstance(metrics.get_term(name), expected)]
        raise TypeError(
            f"{term} reads a {expected.__name__} metric, but metric '{metric_name}' is a {type(metric).__name__}. "
            f"Set metric_name to one of the {expected.__name__} metrics: {matching}."
        )
    return metric


def soft_dof_pos_violation(dofs_pos: torch.Tensor, soft_limits: torch.Tensor) -> torch.Tensor:
    """Per-DOF distance outside the soft ``[lower, upper]`` position band (``>= 0``, zero inside).

    ``dofs_pos`` is ``(..., num_dofs)`` and ``soft_limits`` is ``(..., num_dofs, 2)`` (``[..., 0]`` lower,
    ``[..., 1]`` upper). Sum over the DOF axis for a scalar penalty; compare ``> 0`` for a limit-exceeded mask.
    """
    lower = -(dofs_pos - soft_limits[..., 0]).clip(max=0.0)
    upper = (dofs_pos - soft_limits[..., 1]).clip(min=0.0)
    return lower + upper


def parallel_jaw_is_grasping(
    entity: Entity,
    contact_info: dict[str, torch.Tensor],
    left_gripper_link_idx: int,
    right_gripper_link_idx: int,
    gripper_idx_local: list[int],
    force_threshold: float,
    object_link_idx: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-env bool ``(num_envs,)``: both gripper pads carry contact force aligned with the closing
    direction above ``force_threshold`` — a parallel-jaw grasp.

    Detects a grasp even when the fingers can't fully close (an object is held between them). Shared by
    the ``IsGrasping`` metric/termination and the ``ParallelJawWelding`` action, so both apply the same
    force test. They differ only in which contacts it sums:

    * ``object_link_idx=None`` (``IsGrasping``): each pad's force over all its contacts, so a grasp
      across two links of one multi-link object counts. So do pads pressing two different objects.
    * ``object_link_idx`` given (``ParallelJawWelding``, once per candidate link): only contacts with
      that link, so both pads must press the one link that would be welded.

    ``contact_info`` is ``entity.get_contacts(exclude_self_contact=True)``. ``gripper_idx_local`` is
    ``[left_pad, right_pad]`` (local link indices) and defines the left→right closing direction; the pad
    link indices (``entity.get_link(...).idx``) are the global indices matched against the contact pairs.
    """
    link_a = contact_info["link_a"]
    link_b = contact_info["link_b"]
    force_a = contact_info["force_a"]
    force_b = contact_info["force_b"]
    valid_mask = contact_info["valid_mask"]

    # Net contact force on each pad, summed over contacts (from whichever side of the pair is the pad).
    # When an object link is supplied, require both pad forces to come from that same candidate. This
    # prevents contacts with two different objects from being combined into a false grasp.
    if object_link_idx is None:
        left_a = link_a == left_gripper_link_idx
        left_b = link_b == left_gripper_link_idx
        right_a = link_a == right_gripper_link_idx
        right_b = link_b == right_gripper_link_idx
    else:
        left_a = (link_a == left_gripper_link_idx) & (link_b == object_link_idx)
        left_b = (link_b == left_gripper_link_idx) & (link_a == object_link_idx)
        right_a = (link_a == right_gripper_link_idx) & (link_b == object_link_idx)
        right_b = (link_b == right_gripper_link_idx) & (link_a == object_link_idx)

    left_forces = torch.sum(force_a * (left_a & valid_mask)[..., None], dim=1)
    left_forces += torch.sum(force_b * (left_b & valid_mask)[..., None], dim=1)
    right_forces = torch.sum(force_a * (right_a & valid_mask)[..., None], dim=1)
    right_forces += torch.sum(force_b * (right_b & valid_mask)[..., None], dim=1)

    # Closing direction (left pad → right pad); project each pad's net force onto its inward direction.
    gripper_pos = entity.get_links_pos(ls_idx_local=gripper_idx_local)
    close_dir = gripper_pos[:, 1] - gripper_pos[:, 0]
    close_dir = close_dir / close_dir.norm(dim=1, keepdim=True).clamp(min=1e-8)
    left_proj = torch.bmm(left_forces.unsqueeze(1), close_dir.unsqueeze(2)).reshape(-1)
    right_proj = torch.bmm(right_forces.unsqueeze(1), (-close_dir).unsqueeze(2)).reshape(-1)

    return (left_proj.abs() > force_threshold) & (right_proj.abs() > force_threshold)


def resolve_link_idx(
    entity: Entity,
    names: list[str],
    device,
    *,
    preserve_order: bool = True,
    term: str | None = None,
    field: str = "links_name",
) -> torch.Tensor:
    """Resolve link name patterns to a local-index ``long`` tensor (call once in ``build()``).

    ``preserve_order=True`` matches ``RigidEntity.find_named_links_idx_local``'s default.

    The entity must be a rigid one. Named links are a ``RigidEntity`` API, and so is every reason to
    ask for their indices (``get_links_pos`` / ``get_links_net_contact_force``), so a particle or FEM
    entity named here has neither the links nor the per-link readings the caller is about to take.
    Checking it at this one chokepoint is what spares each link term its own copy — the same reason
    :func:`resolve_terrain` checks ``get_height`` rather than leaving it to every height term.

    An *empty* selection always raises. A pattern matching nothing already raises from the inner
    ``resolve_matching_names``, but an empty ``names`` matches nothing *without* raising, and every
    caller is per-link by design: the silent result is a zero-width observation, a never-firing mask,
    or — for the ``feet_link_names`` rewards, whose field defaults to ``[]`` — a penalty that
    reads zero for a whole training run. That is the config mistake this helper exists to catch, so
    the check is not opt-in; a caller that genuinely tolerates an empty selection branches before
    calling, as :class:`~eden.managers.terms.observations.contact.GroupedContactFlags` does for its
    optional partner links. ``term`` (the calling term's class name) and ``field`` (the config field
    the patterns came from — not ``links_name`` on every term) only label the error.
    """
    prefix = f"{term}: " if term else ""
    if not hasattr(entity, "find_named_links_idx_local"):
        raise TypeError(
            f"{prefix}entity '{entity.name}' is a {type(entity).__name__}, which has no named links (no "
            f"find_named_links_idx_local) and so reports no per-link pose or contact force. Selecting links by "
            f"name needs a rigid entity (eden.entities.rigid.RigidEntity); a particle or FEM entity is made of "
            f"particles, not links. "
            f"Check that entity_name points at the robot rather than at a deformable object in the scene."
        )
    _, link_idx = entity.find_named_links_idx_local(names, preserve_order=preserve_order)
    if not link_idx:
        available = [link.name for link in entity.links]
        raise ValueError(
            f"{prefix}{field}={names} matched no links on entity '{entity.name}'. This term is per-link, so an "
            f"empty selection is not a disabled term — it is a zero-width observation, a never-firing mask, or a "
            f"penalty that reads zero for a whole training run. "
            f"Set {field} to the link patterns you meant; to not use the term at all, remove it from the config "
            f"rather than leaving its link selection empty. Available links: {available}"
        )
    return torch.tensor(link_idx, dtype=torch.long, device=device)


def resolve_terrain(entities: dict, name: str | None) -> Entity | None:
    """Resolve an optional ``terrain_entity_name`` to an entity that can report ground height.

    ``None`` passes through as ``None`` — measure against the world-frame ``z``, which is what a flat
    scene wants and what Isaac Lab's height terms do. A name is looked up with
    :func:`~eden.utils.misc.resolve_named_entity`, so a typo lists the scene's entities, and the result
    is then checked for the one thing :func:`base_height_above_ground` needs from it: ``get_height``.
    Without that check the plausible mistake — naming the flat ground plane rather than a
    :class:`~eden.entities.terrain.Terrain` — reaches the hot path and surfaces as an ``AttributeError``
    on the first step, which for a termination term means mid-run rather than at env construction.
    """
    if name is None:
        return None
    terrain = resolve_named_entity(entities, name)
    if not hasattr(terrain, "get_height"):
        raise TypeError(
            f"terrain_entity_name='{name}' names a {type(terrain).__name__}, which reports no ground height "
            f"(no get_height). It must be a Terrain (eden.entities.terrain.Terrain). A flat scene needs no "
            f"terrain entity at all — leave terrain_entity_name unset to measure the world-frame z."
        )
    return terrain


def base_height_above_ground(entity: Entity, terrain: Entity | None = None, *, relative: bool = False) -> torch.Tensor:
    """Height of an entity's base link above the ground, ``(num_envs,)``.

    With ``terrain=None`` this is the world-frame ``z`` of the base — correct on a flat plane and
    what Isaac Lab's ``root_height_below_minimum`` / ``base_pos_z`` measure. Pass a
    :class:`~eden.entities.terrain.Terrain` to subtract the terrain height sampled under the base's
    ``xy`` instead, which is what makes a height threshold meaningful on rough ground: on a slope the
    world ``z`` of a perfectly-postured robot tracks the hill, so a fixed floor on it fires at the
    bottom and never at the top.

    ``relative`` selects the frame the base pose is read in, exactly as on
    :func:`~eden.managers.terms.observations.proprio.base_pos`: ``False`` (the default, and what
    :func:`~eden.managers.terms.rewards.locomotion.base_height_l2` uses) is the solver frame, which
    Genesis re-origins onto the body's inertial frame; ``True`` is the frame the asset was authored
    in — the frame ``default_root_pos`` and Isaac Lab's ``root_pos_w`` are in. The two differ by the
    base link's centre-of-mass offset, so a threshold tuned in one frame is wrong in the other.
    """
    pos = entity.get_pos(relative=relative)
    if terrain is None:
        return pos[:, 2]
    # get_height wants a (B, h_out, w_out, 2) sample grid; one point per env is the 1x1 case.
    ground = terrain.get_height(pos[:, :2].reshape(-1, 1, 1, 2)).reshape(-1)
    return pos[:, 2] - ground


def resolve_per_dof_std(
    std_dict: dict[str, float],
    robot: Entity,
    device,
    *,
    default: float | None = None,
    field: str = "std",
) -> torch.Tensor:
    """Per-DOF std tensor from a ``regex -> value`` dict over the robot's DOF names.

    Every DOF must be matched by some pattern (later patterns override earlier ones); DOFs no pattern
    matches raise a ``ValueError`` naming them, so a std map written for one robot cannot silently
    mis-weight another's extra DOFs. An explicit ``".*"`` entry is the opt-in for "everything else gets
    this value"; passing ``default=`` instead fills unmatched DOFs with that value without raising.
    The reverse is deliberately tolerated: a pattern matching no DOF is NOT an error, so one std map
    can be shared across robots and carry entries for DOFs only some of them have (e.g. ``.*head.*``
    in a map used by both the G1 and the LimX Oli). ``field`` names the offending config field in the
    error. Call once in ``build()``.
    """
    dofs_name = robot.dofs_name
    fill = 0.0 if default is None else default  # placeholder only: unmatched DOFs raise below when default is None
    result = torch.full((len(dofs_name),), fill, device=device, dtype=torch.float32)
    matched = [False] * len(dofs_name)
    for pattern, value in std_dict.items():
        for i, name in enumerate(dofs_name):
            if re.match(pattern, name):
                result[i] = value
                matched[i] = True
    if default is None and not all(matched):
        unmatched = [name for name, hit in zip(dofs_name, matched) if not hit]
        raise ValueError(
            f"{field} does not cover {len(unmatched)} of {len(dofs_name)} DOFs on '{robot.name}': "
            f"{', '.join(unmatched)}\n"
            f"patterns tried: {list(std_dict)}\n"
            f"Add a pattern for them, or '.*': <value> as an explicit catch-all."
        )
    return result
