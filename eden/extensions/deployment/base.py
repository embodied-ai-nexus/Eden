"""Base class and registry for real-robot deployment backends."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Optional
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Sequence
import os
import time

import numpy as np
import torch

import eden as en
from eden.options.entities import CompositeEntityOptions, RobotOptions
from eden.options.extensions.deployment import DeploymentOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry
from eden.utils.string import first_few, resolve_matching_names_values
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.extensions.deployment.utils.record import StreamingRecorder
from eden.extensions.deployment.utils.state import RobotCommand, RobotState, RobotStateEntity


if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.action_manager import ActionTerm
    from eden.utils.configs import EdenRLConfig


DEPLOYMENT_REGISTRY = Registry("DEPLOYMENT")


# Robot → deployment binding. Populated at import time by each deployment
# module (e.g. ``unitree_g1.py`` declares ``bind_deployment(G1Dofs29, ...)``).
# ``eden deploy --task <name>`` resolves the deployment class from the robot
# bound to the task's scene entity via :func:`resolve_deployment_for`.
_ROBOT_DEPLOYMENT_BINDINGS: dict[type, str] = {}


def bind_deployment(robot_cls: type, deployment_name: str) -> None:
    """Register ``robot_cls`` as driven by the named deployment in :data:`DEPLOYMENT_REGISTRY`.

    Resolution walks the robot's MRO, so subclasses inherit the binding
    unless they register their own override. Call this once per robot class
    in the deployment module that supports it; collisions are not detected
    (the last binding wins).
    """
    _ROBOT_DEPLOYMENT_BINDINGS[robot_cls] = deployment_name


def resolve_deployment_for(robot_options: RobotOptions | CompositeEntityOptions) -> type[DeploymentBase]:
    """Return the deployment class registered for ``type(robot_options)``.

    Walks the MRO to honor inheritance (a subclass with no binding falls
    back to its parent's). Raises ``KeyError`` if no ancestor is bound.

    A composite rig (:class:`~eden.options.entities.CompositeEntityOptions` — an arm with a hand
    welded on, a humanoid with hands) resolves to ``composite_deployment``, which drives each of
    its parts through that part's own binding.
    """
    if isinstance(robot_options, CompositeEntityOptions):
        return DEPLOYMENT_REGISTRY.get("composite_deployment")
    cls = type(robot_options)
    for ancestor in cls.__mro__:
        if ancestor in _ROBOT_DEPLOYMENT_BINDINGS:
            name = _ROBOT_DEPLOYMENT_BINDINGS[ancestor]
            return DEPLOYMENT_REGISTRY.get(name)
    available = sorted({k.__name__ for k in _ROBOT_DEPLOYMENT_BINDINGS}) or ["<none>"]
    raise KeyError(
        f"No deployment registered for robot {cls.__name__}. "
        f"Available bindings: {available}. Add one with "
        f"``bind_deployment({cls.__name__}, '<deployment_name>')``."
    )


def scene_entity_options(scene_options: Any, entity_name: str) -> Any:
    """The options the env entity ``entity_name`` is built from, composite rigs expanded.

    A composite (``robot=FrankaXHand(...)``) sits in the scene under its primary part's name and is
    expanded into its parts only when the env builds, so the env entity ``robot`` is the rig's arm
    and ``hand`` is not a scene entry at all. Both are looked up inside the composite here, as the
    env build expands it. (``eden deploy`` deploys the whole rig when the scene entry under
    ``--entity_name`` is a composite; that is a decision about *what* to deploy, made by the caller.)

    Raises ``ValueError`` naming the entities there are when none is called ``entity_name``.
    """
    available: list[str] = []
    for key in scene_options.keys():
        if key == "attachments_dict":
            continue
        value = getattr(scene_options, key)
        if isinstance(value, CompositeEntityOptions):
            parts = value.entity_options()
            if entity_name in parts:
                return parts[entity_name]
            available += list(parts)
        elif key == entity_name:
            return value
        else:
            available.append(key)
    raise ValueError(f"Scene has no entity named '{entity_name}'. Entities: {sorted(set(available))}.")


def resolve_dofs_values(data: dict[str, float], dofs_name: list[str]) -> np.ndarray:
    """Resolve a pattern-keyed per-DOF dict into an array aligned with ``dofs_name``.

    :func:`resolve_matching_names_values` with ``preserve_order=True`` returns
    matches in *query-pattern* order, so the values must be scattered through
    the returned indices — zipping them positionally with ``dofs_name``
    scrambles any dict with wildcard keys. DOFs not covered by any pattern
    resolve to 0.0 (with a warning: a zero kp/kd reaching a real robot means a
    limp joint).
    """
    idx, _, values = resolve_matching_names_values(data, dofs_name, preserve_order=True)
    resolved = np.zeros(len(dofs_name), dtype=np.float64)
    resolved[idx] = values
    if len(idx) < len(dofs_name):
        missing = [name for i, name in enumerate(dofs_name) if i not in set(idx)]
        en.logger.warning(f"[deployment] No value matched for DOFs {missing}; defaulting to 0.0.")
    return resolved


def resolve_pinned_gains(robot: Any, dofs_name: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    """A policy's pinned gains on the DOFs of ``dofs_name`` it names, as ``({dof: kp}, {dof: kd})``.

    A ``default_dofs_kp`` / ``default_dofs_kd`` that ``robot`` leaves empty (``G1Dofs23``'s,
    which trains on the asset's gains), or a ``robot`` with no ``dofs_name`` at all (``None``, a
    composite), resolves to ``{}``: it declares no gains, and the deployed robot's apply.
    :meth:`DeploymentBase.wire_gains` builds a slot's wire gains from this.

    Needs no deployment, so ``eden deploy`` runs it for every slot before constructing one: a
    raise here must land before the entity swap and before an operator device is opened.

    Raises ``ValueError`` when a non-empty gain map leaves out a joint of ``dofs_name`` the
    robot has (and when a pattern in it matches none of the robot's joints): see "Each slot
    runs at its own PD gains" in ``docs/guides/deployment.md``.
    """
    robot_dofs = getattr(robot, "dofs_name", None)
    if robot_dofs is None:
        return {}, {}
    robot_dofs = list(robot_dofs)
    deployed = set(dofs_name)
    driven = [name for name in robot_dofs if name in deployed]
    resolved: list[dict[str, float]] = []
    for field in ("default_dofs_kp", "default_dofs_kd"):
        pinned = getattr(robot, field, None)
        if not pinned:  # declares no gains: the deployed robot's
            resolved.append({})
            continue
        idx, _, values = resolve_matching_names_values(pinned, robot_dofs, preserve_order=True)
        covered = {robot_dofs[j]: float(value) for j, value in zip(idx, values, strict=True)}
        if uncovered := [name for name in driven if name not in covered]:
            owner = type(robot).__name__
            raise ValueError(
                f"{owner}'s {field} gives no value for {first_few(uncovered)}, which this deployment drives. A "
                "slot runs at its pinned robot's gains; a joint they leave out took the asset's own gain in "
                "training, which a deployment cannot read. Give every joint a value in the policy's "
                f"robot_options(), e.g. {field}={{**{owner}().{field}, ...}} (pydantic replaces the whole dict, "
                "it does not merge). Do not fall back to 0 or to the deployed robot's gain: either is a guess on "
                "a live robot."
            )
        resolved.append({name: covered[name] for name in driven})
    return resolved[0], resolved[1]


def as_name_tuple(value: Collection[str] | str, field: str, consequence: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of names, refusing a bare string.

    Every name-collection option (``feet_links``, ``ignore_action_terms``,
    ``disable_joints``) has the same trap: a string is a perfectly good collection of
    *characters*, so passing one silently degenerates into one entry per character rather
    than raising. ``consequence`` says what that would do for this particular field.
    """
    if isinstance(value, str):
        raise TypeError(
            f"{field} must be a collection of names, got the bare string {value!r}: {consequence} — pass ({value!r},)."
        )
    return tuple(value)


def resolve_dof_index_map(
    dofs_name: Sequence[str],
    index_map: dict[str, int],
    *,
    owner: str,
    what: str = "SDK motor index",
    error: type[Exception] = ValueError,
    hint: str = "",
) -> list[int]:
    """Map each deployment DOF onto its index in the vendor SDK's motor array.

    Every hardware backend addresses motors by a fixed vendor index, not by the config's
    DOF order, so each resolves ``dofs_name`` through its own name→index table. A name the
    table doesn't know must raise at construction with the offending joints named, not as a
    bare ``KeyError`` from a stack deep inside the backend.

    Parameters
    ----------
    dofs_name: Sequence[str]
        Deployment DOF order to resolve.
    index_map: dict[str, int]
        The backend's joint-name → vendor-index table.
    owner: str
        Backend class name, for the error message.
    what: str
        Noun for the index being resolved (the XHand's is a finger index, not a motor one).
    error: type[Exception]
        Exception class to raise; Go2 uses ``NotImplementedError`` because its unmapped
        joints mean an unbuilt backend rather than a typo.
    hint: str
        Replaces the default "known joints" listing when a backend has something more
        useful to say.
    """
    unmapped = [name for name in dofs_name if name not in index_map]
    if unmapped:
        raise error(f"{owner} has no {what} for joints {unmapped}. " + (hint or f"Known joints: {sorted(index_map)}."))
    return [index_map[name] for name in dofs_name]


def _payload_gather_error(term: ActionTerm) -> str | None:
    """Why the payload cannot gather joint targets out of ``term``, or ``None`` if it can.

    :meth:`DeploymentBase.action_to_payload` reads ``term.processed_action[0, src_idx]``, one
    position per DOF, so it needs a ``(num_envs, len(term.dofs_name))`` buffer. That is a
    *shape* question, not a DOF-count one: ``ActionTerm.processed_action`` guarantees only the
    leading ``num_envs`` dimension ("both the units and the trailing width are term-specific"),
    so the buffer is what decides whether the gather is even expressible. A welding term's
    ``(num_envs,)`` boolean would raise ``IndexError: too many indices`` at the first hardware
    step; a zero-DOF term left with the base class's ``(num_envs, 0)`` buffer gathers nothing
    and is harmless.
    """
    processed = term.processed_action
    if processed is None:
        return "its processed_action buffer is None — the term was never built"
    if processed.ndim != 2 or processed.shape[1] != len(term.dofs_name):
        return (
            f"its processed_action is {tuple(processed.shape)}, not (num_envs, "
            f"{len(term.dofs_name)}) — one joint target per DOF it controls"
        )
    return None


def build_term_action_maps(
    terms: dict[str, ActionTerm],
    dofs_name: Sequence[str],
    device: torch.device | str,
    ignore_action_terms: Collection[str] = (),
    *,
    entity_name: str,
) -> list[tuple[ActionTerm, torch.Tensor, torch.Tensor]]:
    """Map each action term's local DOF order onto deployment DOF order.

    Returns one ``(term, src_idx, dst_idx)`` triple per term that contributes joint
    targets, such that ``term.processed_action[0, src_idx]`` belongs at deployment DOFs
    ``dst_idx``. Carrying the term inside the triple is deliberate: re-zipping a positional
    list against ``terms`` would be silently misaligned by any filtering here.

    Terms that contribute nothing are dropped rather than returned with empty index
    tensors — but "contributes nothing" covers four different situations, which differ in
    how loudly they have to be said:

    - A term driving an entity other than ``entity_name`` — the hand of an arm + dex-hand
      rig, which a scene expands into *separate* entities, when only the arm is deployed —
      is not this robot's business, and is dropped with a log line. Which term belongs to
      which robot is an **entity** question, not a DOF-name one: two entities built from
      the same asset share joint names, so matching on names alone would quietly write the
      other one's targets into this robot's command slots. A term dropped here whose DOF names
      *do* occur on the deployed robot is logged at WARNING rather than INFO — seen from in
      here, a mis-declared ``entity_name`` looks exactly like a second robot.
    - A term on the deployed entity, some or all of whose DOFs are not in ``dofs_name``.
      A term resolves its DOFs inside its own entity's declared ``dofs_name``
      (:meth:`ActionTerm.build`), which is the list this argument comes from, so the two
      having drifted apart is an anomaly, not an arm-only deployment: those DOFs go
      uncommanded and are logged at WARNING. (A term declaring no DOFs at all is a
      different thing and stays silent — ``_payload_gather_error`` treats its
      ``(num_envs, 0)`` buffer as harmless, and there is nothing to name.)
    - A term the payload cannot gather from at all (see ``_payload_gather_error`` below) but
      whose effect the real robot produces on its own — ``term.sim_surrogate``, e.g.
      ``ParallelJawWelding`` standing in for gripper friction. There is nothing to send,
      so it is dropped with a log line.
    - Any other term the payload cannot gather from commands something in its own right
      (``SuctionCupWelding`` is the vacuum bit) and :class:`RobotCommand` has no channel
      for it. Sending it is not possible, and the gather would raise ``IndexError`` at the
      first hardware step. Raise here instead, at deployment construction, so the
      operator learns before the robot is enabled.

    Those four are the *term's* side of a mismatch. The robot's side — DOFs of ``dofs_name``
    that no returned map reaches — is only visible once every term has been placed, so it is
    warned about after the loop. It is not a cosmetic gap: ``action_to_payload`` writes only
    the slots these maps name into a zero-initialised buffer, so an unreached DOF is commanded
    to 0.0 at the robot's default stiffness, which on hardware is a joint driven to zero rather
    than one left alone. The warning is hedged because an extension-policy slot can fill those
    slots through :meth:`DeploymentBase.write_dofs_targets`, which is invisible from here.

    Parameters
    ----------
    terms: dict[str, ActionTerm]
        The action manager's terms, keyed by term name.
    dofs_name: Sequence[str]
        Deployment DOF order — the layout of the ``RobotCommand`` arrays.
    device: torch.device | str
        Device for the returned index tensors.
    ignore_action_terms: Collection[str]
        Names of terms in the last group above that the caller drives over its own
        channel, and therefore accepts will not appear in the payload. Every name is
        checked against ``terms``: one that is not a term, or that names a term the
        payload *does* carry, raises — listing it there would read as a guarantee this
        cannot give. "Does carry" is the same test the loop applies, entity included, so
        acknowledging another entity's term is allowed: that term is dropped anyway, and
        rejecting it would contradict the drop.
    entity_name: str
        Scene entity being deployed (``DeploymentBase.entity_name``). Terms driving any
        other entity are not sent; see the first group above.

    Returns
    -------
    list[tuple[ActionTerm, torch.Tensor, torch.Tensor]]
        ``(term, src_idx, dst_idx)`` for every contributing term.

    Raises
    ------
    TypeError
        If ``ignore_action_terms`` is a bare string.
    ValueError
        If ``ignore_action_terms`` names something other than a term the payload cannot
        carry, or if such a term is neither a ``sim_surrogate`` nor listed there.
    """
    ignore_action_terms = as_name_tuple(
        ignore_action_terms, "ignore_action_terms", "membership would match substrings rather than term names"
    )
    gather_errors = {term_name: _payload_gather_error(term) for term_name, term in terms.items()}

    unknown = sorted(name for name in ignore_action_terms if name not in terms)
    if unknown:
        raise ValueError(
            f"ignore_action_terms names {unknown}, which are not action terms of the deployed "
            f"config. Action terms: {sorted(terms)}."
        )
    # "does carry" is the whole test below, not just gatherability: a joint term of another
    # entity gathers fine and is still dropped (by the entity filter), so rejecting it here
    # would contradict what this same function is about to do with it.
    carried = sorted(
        name for name in ignore_action_terms if gather_errors[name] is None and terms[name].entity_name == entity_name
    )
    if carried:
        raise ValueError(
            f"ignore_action_terms names {carried}, whose joint targets the payload does carry. "
            f"Listing a term there does not stop it being sent — it only acknowledges a command "
            f"RobotCommand has no channel for — so this would read as a guarantee it cannot give. "
            f"To leave joints undriven use disable_joints (kp=kd=0 on the robot); to stop "
            f"commanding them at all, remove the term from the deployed config."
        )

    dofs_name_to_idx = {name: i for i, name in enumerate(dofs_name)}
    maps: list[tuple[ActionTerm, torch.Tensor, torch.Tensor]] = []
    for term_name, term in terms.items():
        gather_error = gather_errors[term_name]
        if gather_error is not None:
            if term.sim_surrogate:
                en.logger.info(
                    f"[deployment] Action term {term_name!r} ({type(term).__name__}) is a simulation "
                    f"surrogate; the robot produces its effect itself, so it is not sent."
                )
                continue
            if term_name in ignore_action_terms:
                en.logger.info(
                    f"[deployment] Action term {term_name!r} ({type(term).__name__}) is not sent — "
                    f"acknowledged via ignore_action_terms."
                )
                continue
            raise ValueError(
                f"Action term {term_name!r} ({type(term).__name__}) cannot feed a joint-position "
                f"payload: {gather_error}. RobotCommand carries only dofs_pos / dofs_vel / "
                f"dofs_torque / dofs_kp / dofs_kd, so deploying this config would drop that command "
                f"silently on the way to the robot. Either remove the term from the deployed config, "
                f"or drive it over a channel of your own and acknowledge that it is not sent — "
                f"`eden deploy --ignore_action_terms {term_name}`, or "
                f"ignore_action_terms=({term_name!r},) when constructing the deployment. A term "
                f"whose effect the real robot produces on its own should instead declare "
                f"`_sim_surrogate_ = True` (see ActionTerm.sim_surrogate)."
            )
        if not term.dofs_name:
            # Declares no DOFs: its (num_envs, 0) buffer gathers nothing (``_payload_gather_error``),
            # so there is neither anything to send nor any DOF to name.
            continue
        src_idx = []
        dst_idx = []
        unmapped = []
        for j, name in enumerate(term.dofs_name):
            if name in dofs_name_to_idx:
                src_idx.append(j)
                dst_idx.append(dofs_name_to_idx[name])
            else:
                unmapped.append(name)
        if term.entity_name != entity_name:
            # Another entity's term. Dropping it is right whatever its DOFs are called — but a
            # name overlap is far more likely a term pointed at the wrong entity than a second
            # robot on the rig.
            shared = [term.dofs_name[j] for j in src_idx]
            if shared:
                en.logger.warning(
                    f"[deployment] Action term {term_name!r} ({type(term).__name__}) drives entity "
                    f"{term.entity_name!r}, not the deployed {entity_name!r}, so it is not sent — yet "
                    f"{len(shared)} of its DOF(s) carry names the deployed robot also has, so check "
                    f"this is a second robot and not a mis-declared entity_name: {first_few(shared)}"
                )
            else:
                en.logger.info(
                    f"[deployment] Action term {term_name!r} ({type(term).__name__}) drives entity "
                    f"{term.entity_name!r}, not the deployed {entity_name!r}, so its "
                    f"{len(term.dofs_name)} DOF(s) are not sent: {first_few(term.dofs_name)}"
                )
            continue
        followers = list(getattr(term, "mimic_follower_dofs_name", ()))
        if followers:
            # The followers are commanded from their drivers' targets in simulation, and a coupled hand
            # (Inspire's six-command firmware) couples them itself — sending them would either be rejected
            # or double-drive the linkage. Dropping them is not implemented yet, so refuse rather than guess.
            raise ValueError(
                f"Action term {term_name!r} ({type(term).__name__}) on the deployed {entity_name!r} uses "
                f"`mimic_dofs`: {first_few(followers)} are commanded from their drivers' targets in "
                f"simulation. Deploying such a term is not supported yet — hardware that couples them in "
                f"firmware (e.g. a real Inspire hand, six commands) must not be sent the follower columns, "
                f"and this map cannot drop them. Deploy an entity whose terms have no `mimic_dofs`."
            )
        # ``unmapped`` is the *term's* side of the mismatch — DOFs it drives that the deployed
        # robot does not declare. Name that side here; the robot's own side (DOFs no term
        # reaches at all) is a property of every map together, so it is reported after the loop.
        if not src_idx:
            en.logger.warning(
                f"[deployment] Action term {term_name!r} ({type(term).__name__}) drives the deployed "
                f"{entity_name!r}, but none of the {len(unmapped)} DOF(s) it controls are in that "
                f"robot's declared dofs_name, so nothing is sent for it: {first_few(unmapped)}"
            )
            continue
        if unmapped:
            en.logger.warning(
                f"[deployment] Action term {term_name!r} ({type(term).__name__}) controls "
                f"{len(unmapped)} DOF(s) missing from the declared dofs_name of the deployed "
                f"{entity_name!r}; the term still commands the rest, and these targets are "
                f"dropped: {first_few(unmapped)}"
            )
        maps.append(
            (
                term,
                torch.tensor(src_idx, dtype=torch.long, device=device),
                torch.tensor(dst_idx, dtype=torch.long, device=device),
            )
        )

    # The other side of the mismatch, and the one the robot feels. ``_processed_action`` is
    # allocated as zeros and ``action_to_payload`` writes only the slots these maps name, so a
    # deployment DOF no term reaches is not "left alone" — it ships position target 0.0 at the
    # robot's default stiffness, i.e. a joint driven hard to zero. Hedged rather than asserted
    # because a deploy can also fill those slots out of band: an extension-policy slot writes
    # straight into the same buffer via ``write_dofs_targets``, and this function cannot see it.
    covered = {slot for _, _, dst_idx in maps for slot in dst_idx.tolist()}
    unreached = [name for i, name in enumerate(dofs_name) if i not in covered]
    if unreached:
        en.logger.warning(
            f"[deployment] {len(unreached)} DOF(s) of the deployed {entity_name!r} are reached by no "
            f"action term. Unless a slot writes them directly (see write_dofs_targets), they are "
            f"commanded to position 0.0 at the robot's default stiffness rather than left "
            f"undriven — use disable_joints (kp=kd=0) for that: {first_few(unreached)}"
        )
    return maps


def cached_refs(terms: Iterable[Any], target: Any, *, what: str, remedy: str) -> list[tuple[Any, str]]:
    """Every ``(term, attr)`` among ``terms`` whose attribute *is* ``target``.

    The scan half of every deploy-time swap — the sim entity for a
    :class:`~eden.extensions.deployment.utils.state.RobotStateEntity`, a task's command
    generator for one an operator drives. Both replace an object the env already handed to
    terms, and a term that cached the old one keeps reading it: stale sim state in the first
    case, the random training command in the second, in both cases silently.

    Matching is by **object identity**, so a cached handle is caught whatever the attribute
    is named (``entity`` / ``_entity`` / ``robot`` / ``_command`` / …), not only one literally
    named ``entity``.
    A reference held in a form that cannot be reassigned — a bound method, or membership in
    a top-level list/tuple/set/dict — raises instead, as does a ``__slots__`` term that
    cannot be introspected at all: the point is to turn a would-be silent stale read into a
    loud failure at deploy init. (References nested deeper — inside a helper object, a
    ``functools.partial``, or a closure — are out of reach and would still read stale; the
    current term tree has none.)

    Writes nothing, so the caller can decide a whole swap is applicable before applying any
    of it — which is what lets a construction that raises later roll the swap back cleanly
    rather than leave it half-applied.

    ``what`` names the swapped object in the error messages ("the simulated entity"),
    ``remedy`` says how to make a term repointable.
    """
    refs: list[tuple[Any, str]] = []
    unrepointable: list[str] = []
    for term in terms:
        term_dict = getattr(term, "__dict__", None)
        if term_dict is None:
            raise RuntimeError(
                f"[deploy] Cannot scan {type(term).__name__} for cached references to {what} "
                "(it defines __slots__, so it has no __dict__). If it caches one it would keep "
                f"reading the pre-swap object on the real robot. {remedy}"
            )
        for attr, val in term_dict.items():
            if val is target:
                refs.append((term, attr))
            elif getattr(val, "__self__", None) is target:
                unrepointable.append(f"{type(term).__name__}.{attr} (bound method of {what})")
            elif isinstance(val, (list, tuple, set)) and any(v is target for v in val):
                unrepointable.append(f"{type(term).__name__}.{attr} ({what} inside a {type(val).__name__})")
            elif isinstance(val, dict) and any(v is target for v in val.values()):
                unrepointable.append(f"{type(term).__name__}.{attr} ({what} inside a dict)")

    if unrepointable:
        raise RuntimeError(
            f"[deploy] Term(s) hold {what} in a form the deployment cannot repoint: "
            f"{unrepointable}. They would keep reading the pre-swap object on the real robot. "
            f"{remedy}"
        )
    return refs


#: How :func:`cached_refs` describes the entity swap in its errors.
_ENTITY_WHAT = "the simulated entity"
_ENTITY_REMEDY = (
    "Expose the entity as a plain attribute (resolved from `env.entities[name]`) so the swap "
    "can repoint it, or resolve it lazily each compute()."
)


def cached_entity_refs(obs_terms: Iterable[Any], entity: Any) -> list[tuple[Any, str]]:
    """:func:`cached_refs` for the sim-entity swap. See :meth:`DeploymentBase._repoint_entity_refs`."""
    return cached_refs(obs_terms, entity, what=_ENTITY_WHAT, remedy=_ENTITY_REMEDY)


def with_owned_modifiers(terms: Iterable[Any]) -> list[Any]:
    """``terms`` plus every :class:`ActionModifier` they own, flattened.

    A modifier caches the entity exactly as a term does —
    :class:`~eden.managers.modifiers.actions.actuators.EnvelopeClip` keeps it and re-reads
    ``get_dofs_pos`` / ``get_dofs_vel`` on every ``modify_processed_action`` to clip the
    target to a torque envelope — but it hangs off the term as ``_modifier``, one level
    deeper than :func:`cached_refs` looks. Scanning it as a term of its own is what brings it
    inside the swap instead of leaving it reading the simulated robot on hardware.

    Recursive because
    :class:`~eden.managers.modifiers.actions.actuators.Compose` holds its children in a list,
    and by ``isinstance`` rather than by attribute name so a modifier that composes another
    under any name is still reached.
    """
    from eden.managers.modifiers import ActionModifier

    def _walk(modifier: Any) -> Iterator[Any]:
        yield modifier
        for value in vars(modifier).values():
            children = value if isinstance(value, (list, tuple, set)) else (value,)
            for child in children:
                if isinstance(child, ActionModifier):
                    yield from _walk(child)

    scanned = list(terms)
    for term in list(scanned):
        modifier = getattr(term, "_modifier", None)
        if isinstance(modifier, ActionModifier):
            scanned.extend(_walk(modifier))
    return scanned


class DeploymentBase(ConfigurableMixin[DeploymentOptions], ABC):
    """
    Base class for real-robot deployment backends.

    Attributes
    ----------
    entity_name : str
        The name of the entity to deploy
    control_dt : float | None
        The control time step
    control_freq : float | None
        The control frequency
    decimation : int | None
        The decimation factor
    sync : bool
        Whether to synchronize the control loop
    connect_timeout_s : float
        The timeout for connecting to the robot
    deploy_kp_kd_scale : float
        The scale of the Kp and Kd for the deployment.
        Recommend to try from 0.2 (20%), 0.5 (50%), 0.8 (80%), 1.0 (100%)
    auto_yaw_align : bool
        If True (default), capture the robot's IMU yaw at START (the MuJoCo
        backend also captures it on :meth:`reset`'s read) and install it on
        the state wrapper so subsequent observations are computed in the
        yaw-aligned world frame the policy was trained in. Disable when an
        external estimator (e.g. mocap) already provides a yaw-consistent
        base orientation.
    estimate_lin_vel : bool
        If True, run an onboard contact-aided Kalman filter
        (:class:`~eden.extensions.deployment.utils.lin_vel_estimator.LinVelKalmanEstimator`,
        the legged_control / BeyondMimic estimator) fusing IMU acceleration
        with leg-kinematics odometry, and feed the estimated base linear
        velocity to the observation pipeline — ``entity.get_vel()`` /
        ``base_lin_vel`` terms return the estimate instead of zeros.
        Requires ``feet_links``.
    feet_links : tuple[str, ...]
        Foot link names for the velocity estimator's leg odometry and contact
        detection. Defaults to the robot's own ``RobotOptions.feet_links_name``
        (e.g. G1's ankle-roll links), which is where that knowledge belongs — the
        feet are a property of the asset, not of the transport, so the estimator
        works the same on hardware and through the robot-agnostic MuJoCo backend.
        Set this (``--feet_links``) only to estimate off a different pair of links,
        or for a robot whose options declare none.
    lin_vel_estimator_options : dict
        Extra keyword overrides for the estimator (noise parameters,
        ``contact_force_threshold``, ...). See ``LinVelKalmanEstimator``.
    ignore_action_terms : tuple[str, ...]
        Names of action terms whose command :class:`RobotCommand` has no channel for
        (a suction/welding term) and which are therefore driven over a channel of the
        operator's own. Listing a term here acknowledges that its command is *not*
        sent to the robot; leaving it unlisted raises at construction rather than
        dropping it silently on hardware. Every name is validated against the deployed
        config's action terms, and naming a term the payload *does* carry raises — the
        flag cannot stop a joint term from being sent (use ``disable_joints`` for
        that). Joint terms never need listing: one driving another entity than
        ``entity_name``, and one whose effect the robot produces on its own
        (``ActionTerm.sim_surrogate``), are both skipped without a flag. See
        :func:`build_term_action_maps`.
    disable_joints : tuple[str, ...]
        Joint names or ``fnmatch`` patterns (e.g. ``"left_wrist_*"``,
        ``"*_ankle_*_joint"``) whose kp/kd are forced to 0 in every command, so
        those joints go free/limp on the robot. The policy still observes them
        and emits targets; the motors just ignore them (zero gains). For bench
        debugging (isolate a buzzing joint, free the arms) — do NOT disable
        load-bearing leg joints on an unsupported robot, it will collapse.
    gain_ramp_steps : int
        Control steps over which :meth:`set_gains` moves the commanded kp/kd to a new
        policy slot's (default 50 = 1 s at 50 Hz), linearly from whatever is being sent at
        the switch. ``0`` jumps. A slot switch hands the robot the gains *that* policy was
        trained at -- SONIC's 7520-22 hip pitch is ~2.4x Holosoma's 7520-14 -- and a
        stiffness step of that size can knock a standing robot over.
    debug_record : bool
        If True, record per control step the command actually sent (target
        positions + kp/kd) and the measured robot proprioception (joint pos /
        vel / torque, base quat / ang-vel / lin-acc). For offline diagnosis
        (e.g. commanded-vs-measured tracking, torque saturation, limit-cycle
        frequency). Every numeric ``state.extra`` entry rides along under an ``x_``
        prefix; with ``estimate_lin_vel`` that includes the estimate
        (``x_base_lin_vel_estimate``) and, per foot, the contact latch (``x_contacts``)
        and the solved contact force (``x_forces``, flattened ``(n_feet * 3,)``).

        **Streamed to disk as it is produced**, not held until :meth:`close`: the runs
        worth recording are the ones that end badly, and those are the runs where
        ``close`` may never finish. See
        :class:`~eden.extensions.deployment.utils.record.StreamingRecorder` — a killed
        run leaves a chunk directory that
        :func:`~eden.extensions.deployment.utils.record.load_deploy_record` reads exactly
        like the merged file a clean run produces.
    debug_record_path : str | None
        Destination ``.npz`` for ``debug_record``. Defaults to
        ``<log_dir>/deploy_record.npz``.
    debug_record_flush_steps : int
        Steps per streamed chunk (default 100 — two seconds at 50 Hz), i.e. the most that
        a kill can cost. Lower it to lose less and write more often; the writing itself is
        off the control thread either way.
    command_overrides : dict
        Command terms to substitute at deploy time, ``{term_name: CommandTermOptions}``.
        The env's own term is built and stepped by the config it was trained with; a
        deployment can replace it with one configured for a robot with an operator
        standing next to it — most usefully a task's random
        :class:`~eden.managers.terms.commands.velocity.UniformVelocityCommand` for a
        :class:`~eden.managers.terms.commands.velocity.ConstantVelocityCommand` that
        holds one value, or an
        :class:`~eden.managers.terms.commands.velocity.OperatorVelocityCommand` a human
        drives from a keyboard or gamepad. This is the command-manager counterpart of
        the entity swap: the substitute is an ordinary command term, so
        ``command_manager.compute`` drives it with no special case anywhere in the
        deploy loop, and everything reading the command is unchanged. Names are
        validated against the env's active command terms, and the replacement's command
        must have the same shape as the term it replaces. Applied transactionally with
        the entity swap and rolled back with it.
    operator_keyboard : bool
        Open the terminal keyboard reader (``s`` / ``x`` / ``i`` / ``k`` + Enter) as an operator
        input. Default True. Turn it off in a script that reads the keyboard itself -- a pynput
        teleop device sees every key typed into the terminal too, so a teleop key and a command
        key would land on the same line (``s`` is both START and a teleop axis on the WASD
        bindings) -- and give the operator commands to :meth:`request_init` / :meth:`request_start`
        / :meth:`request_stop` from that device instead.
    target_lpf : float
        First-order low-pass on the commanded joint targets, applied to every
        payload via :meth:`_apply_target_lpf`: ``sent = a*new + (1-a)*prev`` with
        ``a = target_lpf`` in (0, 1]. ``1.0`` (default) disables it. Lower values
        smooth high-frequency command content (e.g. a single-frame policy's
        half-control-rate limit cycle / motor whine) while leaving kp/kd — and
        thus the holding stiffness — untouched. Adds command lag (~``1/(a*freq)``).
    init_target : str
        Where the INIT ramp drives the joints before control is handed to the policy.
        ``"default"`` (default) targets ``RobotOptions.default_dofs_pos`` -- the robot's own
        stance, and the only target that is safe from an arbitrary starting pose.
        ``"motion"`` targets frame 0 of the tracked reference clip instead, so a motion-imitation
        policy starts from the pose it was trained to continue from rather than 100+ degrees away.
        Both are joint-space ramps over the wire, so hardware and ``--mujoco`` do the same thing.

        ``"motion"`` is NOT automatically safe: it drives the legs to whatever shape frame 0 has
        while the base is held by nothing but the joint PD. For a clip whose first frame is a
        deep crouch that is an open-loop squat with no balance behind it, and it can topple --
        it has in MuJoCo sim-to-sim. Ramp slowly (``init_steps``), support the
        robot on the first attempt, and prefer entering through a balance policy slot.

        ``"motion_rsi"`` additionally teleports the base to the clip's frame-0 pose. That is
        reference-state initialization and is **sim-only** -- backends that drive real hardware
        reject it, since nothing on a robot can place its own base.
    init_steps : int
        Control cycles the INIT ramp takes from the pose measured at the press to the
        ``init_target`` pose (default 250, 5 s at 50 Hz), so a lying or folded robot rises instead
        of snapping there under full PD from the first tick.
    init_dofs_kp_scale : float
        Scale on the kp sent during that ramp (default 1.0).
    init_dofs_kd_scale : float
        Scale on the kd sent during that ramp (default 1.0).
    on_motion_end : str
        What a motion-imitation deploy does once the clip reaches its final frame.
        ``"continue"`` (default) keeps handing the policy's actions to the robot -- the
        reference holds at its last frame and the policy tracks a target that has stopped moving.
        ``"stand"`` disengages the policy and ramps the joints to the robot's default stance over
        ``stand_steps``, then holds it.

        ``"stand"`` is joint-space only: it detects the end through the same motion clock
        that runs on hardware and sends the same ``RobotCommand`` the wire already carries,
        so a real robot and ``--mujoco`` execute the identical manoeuvre. Nothing about it is
        sim-only. It is NOT unconditionally safe, though: it drives to the default stance from
        wherever the clip left the robot, so it suits a clip that ends near a stand and wants
        a longer ``stand_steps`` -- or an operator STOP instead --
        for a clip that ends crouched, mid-step, or airborne.
        ``"switch"`` hands control to another loaded policy slot (``motion_end_slot``) instead,
        through the same request queue the operator's remote/keyboard uses. That is the only
        one of the three that actually *balances*: a fixed joint-space command holds a shape,
        not a stand, so a biped under ``"stand"`` topples within a couple of seconds. Point it
        at a velocity-tracking / loco policy at zero command.
    stand_steps : int
        Control cycles the ``on_motion_end="stand"`` ramp takes to reach the default stance
        (default 100 = 2 s at 50 Hz). Longer is gentler on hardware.
    motion_end_slot : int
        1-based policy slot ``on_motion_end="switch"`` hands to, numbered as the operator keys
        ``1``-``9`` are. Default 1. The request is dropped with a warning if no such slot is
        loaded, exactly as an out-of-range operator press would be.
    policy_seconds : float | None
        End the run once the policy has driven the robot this long since its latest START, as the
        operator's KILL does (``eden deploy --mujoco_seconds``): the headless counterpart of the
        unattended INIT and START, so a rollout with no operator ends by itself. Only the ticks the
        policy drove count -- not STOP's or INIT's, nor START's handover hold -- and a STOP starts
        the count afresh. Only a simulator (:attr:`allow_unattended_start`) takes it; on a robot the
        operator ends the run. ``None`` (default): until the operator ends it.
    """

    entity_name: str = "robot"
    control_dt: float | None = None
    control_freq: float | None = None
    decimation: int | None = None
    sync: bool = True
    connect_timeout_s: float = 5.0
    deploy_kp_kd_scale: float = 1.0
    auto_yaw_align: bool = True
    estimate_lin_vel: bool = False
    feet_links: tuple[str, ...] = ()
    lin_vel_estimator_options: dict = {}
    ignore_action_terms: tuple[str, ...] = ()
    disable_joints: tuple[str, ...] = ()
    gain_ramp_steps: int = 50
    init_target: str = "default"
    init_steps: int = 250
    init_dofs_kp_scale: float = 1.0
    init_dofs_kd_scale: float = 1.0
    on_motion_end: str = "continue"
    stand_steps: int = 100
    motion_end_slot: int = 1
    debug_record: bool = False
    debug_record_path: str | None = None
    debug_record_flush_steps: int = 100
    command_overrides: dict = {}
    target_lpf: float = 1.0
    operator_keyboard: bool = True
    policy_seconds: float | None = None

    def __init__(self, env: RLEnvBase, options: DeploymentOptions) -> None:
        super().__init__(options)
        if not (0.0 < self.target_lpf <= 1.0):
            raise ValueError(f"target_lpf must be in (0, 1] (1.0 disables), got {self.target_lpf}")
        if self.gain_ramp_steps < 0:
            raise ValueError(f"gain_ramp_steps must be >= 0 (0 jumps), got {self.gain_ramp_steps}")
        if self.policy_seconds is not None and not self.policy_seconds > 0.0:
            raise ValueError(f"policy_seconds must be > 0, or None; got {self.policy_seconds}.")
        if self.policy_seconds is not None and not self.allow_unattended_start:
            raise ValueError(
                f"policy_seconds ends a run by itself, which only a simulator may: {type(self).__name__} drives a "
                "robot, whose run the operator ends (KILL). Drop policy_seconds (--mujoco_seconds)."
            )
        # Ticks the policy has driven since its latest START: ``policy_seconds``' clock.
        self._policy_ticks = 0
        self._check_state_to_observation_override()
        self._env = env
        self.cfg = env.config
        self.robot_options = self._resolve_robot_options(self.cfg, self.entity_name)

        self.control_dt = self._resolve_control_dt()
        if self.control_freq is None and self.control_dt > 0.0:
            self.control_freq = 1.0 / self.control_dt

        self._resolve_dofs()
        self.num_dofs = len(self.dofs_name)
        # Deployment DOF name -> index. Name-mangled (stored as ``_DeploymentBase__dofs_idx``) so a
        # backend's own map cannot replace it: the SDK backends each bind an Eden -> motor-slot map,
        # and one sharing this attribute would break every lookup by DOF name on real hardware.
        self.__dofs_idx = {name: i for i, name in enumerate(self.dofs_name)}
        self._resolve_disabled_joints()
        # The gains every command sends *now*. They start at the deployed robot's and move only
        # through :meth:`set_gains`, which ramps them to the active policy slot's.
        self.default_dof_kp, self.default_dof_kd = self.wire_gains()
        # The :meth:`set_gains` ramp: its (kp, kd) target, and the control steps left to reach it.
        self._gain_target: tuple[np.ndarray, np.ndarray] | None = None
        self._gain_steps_left = 0
        self._init_recording()
        self._lpf_prev: np.ndarray | None = None
        # Last state read off the wire. Declared here because four call sites guard it with
        # ``self.state is not None`` -- an operator INIT before the first read, a hold payload,
        # the end-of-motion latch -- and without this they raise AttributeError instead of
        # taking the branch they are written for.
        self.state: RobotState | None = None
        # The observation manager arrives holding state the training wrapper's ``env.reset()``
        # computed off the SIMULATED robot at its spawn pose; the first read clears it.
        # See :meth:`state_to_observation`.
        self._observations_stale = True
        self._processed_action = torch.zeros(self.num_dofs, device=self._env.device)
        self._term_action_maps = self._build_term_action_maps()
        self._log_unapplied_action_delays()
        self._direct_target_maps: dict[tuple[str, ...], torch.Tensor] = {}
        # Whether anything has ever reached ``_processed_action`` through
        # ``write_dofs_targets``. Zeros are a legal joint target, so the buffer alone cannot say
        # whether it holds a policy's command or its own initializer; see :meth:`step`.
        self._direct_targets_written = False

        self._rate = None
        if self.sync and self.control_freq is not None:
            self._rate = RateLimiter(self.control_freq, name="[deploy] control loop")

        self._resolve_options()
        # Reject an impossible init_target / on_motion_end here, not at the operator's INIT press
        # or at the end of the clip: by then the backend is connected and the robot is powered,
        # and a config error has no business surfacing mid-operation.
        self._validate_init_target()
        self._validate_on_motion_end()

        self._repointed_refs: list[tuple[Any, str]] = []
        self._repointed_command_refs: list[tuple[Any, str]] = []
        self._swapped_commands: dict[str, Any] = {}
        sim_entity = self._swap_in_state_entity()
        try:
            # ``on_motion_end='stand'`` latch: the pose the stand ramp started from, and how
            # far through it we are. Cleared by ``reset_motion_reference`` (deploy reset, INIT,
            # START, policy switch), so re-arming the clip re-engages the policy.
            self._standing_from: np.ndarray | None = None
            self._stand_count = 0
            self._motion_end_switched = False
            self._pending_policy_switches: list[int | str] = []
            # Operator state machine, which every backend runs. Boot in STOP/hold:
            # nothing moves until the operator presses INIT and then START.
            self._inputs: list[Any] = []
            self._warned_no_operator = False
            self._pending_start_events = 0
            self._frozen_watch: dict[str, Any] | None = None
            self._enter_boot_hold()
            self._handover_from_init = False
            # True while :meth:`send_payload` is sending the state machine's own command (the
            # hold, the INIT ramp, the START tick), not the policy's: see :attr:`sending_hold`.
            self._sending_hold = False
            self._lin_vel_warned = False
            self._lin_vel_estimator = self._make_lin_vel_estimator(sim_entity)

            # Command-term substitution, last: a substitute may open an input device, and
            # everything that can still raise has run by now.
            self._apply_command_overrides()
        except BaseException:
            self._undo_swaps(sim_entity)
            raise

    def _resolve_dofs(self) -> None:
        """Set :attr:`dofs_name`, :attr:`default_dof_pos` and the robot's own unscaled gains.

        All three are read off :attr:`robot_options`, in its declared DOF order -- the layout of
        every :class:`RobotCommand` this deployment sends. A deployment driving more than one robot
        (:class:`~eden.extensions.deployment.composite.CompositeDeployment`) overrides this to lay
        its parts' DOFs end to end.
        """
        self.dofs_name = list(self.robot_options.dofs_name)
        self.default_dof_pos = resolve_dofs_values(self.robot_options.default_dofs_pos, self.dofs_name)
        # The deployed robot's own gains, unscaled: what :meth:`wire_gains` falls back to on every
        # DOF a slot's robot does not name.
        self._robot_dofs_kp = resolve_dofs_values(self.robot_options.default_dofs_kp, self.dofs_name)
        self._robot_dofs_kd = resolve_dofs_values(self.robot_options.default_dofs_kd, self.dofs_name)

    def _swap_in_state_entity(self) -> Any:
        """Replace the sim entity with a :class:`RobotStateEntity` and repoint the terms that cached it.

        From here on observation terms read the real robot's state instead of the simulation's.
        Returns the sim entity that was swapped out, which :meth:`_undo_swaps` puts back. A raise
        while repointing leaves the env as it was built.
        """
        sim_entity = self._env.entities[self.entity_name]
        self._state_entity = RobotStateEntity(sim_entity, device=self._env.device)
        self._env.entities[self.entity_name] = self._state_entity
        try:
            self._repoint_entity_refs(sim_entity, self._state_entity)
        except BaseException:
            self._undo_swaps(sim_entity)
            raise
        return sim_entity

    def _make_lin_vel_estimator(self, sim_entity: Any) -> Any:
        """The onboard base-linear-velocity estimator when ``estimate_lin_vel`` is set, else ``None``.

        A contact-aided linear KF over the sim entity's FK, which ``RobotStateEntity.update`` keeps
        synced with real robot state.
        """
        if not self.estimate_lin_vel:
            return None
        from eden.extensions.deployment.utils.lin_vel_estimator import LinVelKalmanEstimator

        estimator = LinVelKalmanEstimator(
            sim_entity,
            feet_links=self.feet_links,
            dt=self.control_dt,
            **(self.lin_vel_estimator_options or {}),
        )
        en.logger.info(f"[deploy] base_lin_vel estimator active (contact-aided KF, feet: {list(self.feet_links)}).")
        return estimator

    def _resolve_options(self) -> None:
        """Resolve and check everything derivable from the options alone.

        Runs immediately *before* ``__init__`` swaps the sim entity for the
        :class:`RobotStateEntity`, so a misconfiguration raises with ``env`` untouched and
        there is nothing to roll back.

        **A backend's own config checks and index tables belong here, not after
        ``super().__init__()``.** A raise there happens once the swap is already applied and
        outside the rollback that guards it, which leaves ``env.entities[entity_name]``
        holding the wrapper; the next construction attempt on that env then wraps the wrapper
        and fails somewhere else entirely, as ``AttributeError: _entity`` deep in the velocity
        estimator.

        Override to add backend-specific resolution, calling ``super()._resolve_options()``
        first. Read only ``self`` (the options, ``robot_options``, ``dofs_name``, gains) and
        the env's *managers* — the entity swap has not happened, and nothing here may mutate
        the env.
        """
        self._resolve_feet_links()

        # Name-check the command overrides against the env's *managers* only, so a term that
        # does not exist raises here -- before the entity swap, and before a substitute has
        # opened an input device. Building them is left to __init__.
        if self.command_overrides:
            manager = getattr(self._env, "command_manager", None)
            if manager is None:
                raise ValueError(
                    f"command_overrides names {sorted(self.command_overrides)}, but this env has no "
                    "command manager (its config declares no command_options)."
                )
            unknown = sorted(set(self.command_overrides) - set(manager.terms))
            if unknown:
                raise ValueError(
                    f"command_overrides names command term(s) the env does not have: {unknown}. "
                    f"Active command terms: {sorted(manager.terms)}."
                )

    def _resolve_feet_links(self) -> None:
        """Resolve :attr:`feet_links`: the option if given, else the robot's own ``feet_links_name``.

        The robot knows its own feet; the option is the per-run override. Resolving here (like
        control_dt) means every backend gets them, including one selected against the
        robot→backend binding — ``eden deploy --mujoco``.
        """
        override = as_name_tuple(self.feet_links, "feet_links", "it would look for one link per character")
        self.feet_links = override or tuple(self.robot_options.feet_links_name)
        if self.estimate_lin_vel and not self.feet_links:
            raise ValueError(
                f"estimate_lin_vel=True needs foot links, and {type(self.robot_options).__name__} "
                "declares no feet_links_name. Add it to the robot's options (that is where the "
                "feet belong — they are a property of the asset), or name them for this run: "
                "`eden deploy ... --estimate_lin_vel --feet_links <left> <right>`."
            )

    def _undo_swaps(self, sim_entity) -> None:
        """Undo every swap ``__init__`` applied: the entity, and any command terms.

        The entity swap precedes the estimator, so without this a construction that raises after
        it would leave ``env.entities[entity_name]`` holding the ``RobotStateEntity`` — see
        :meth:`_resolve_options` for what that breaks. Rolling back keeps the env
        reusable, so a retry in a notebook or a test starts from the state the first attempt
        did; the command overrides are applied last and unwind the same way.

        Replays the recorded references rather than re-scanning the term trees: a scan can
        itself raise (a ``__slots__`` term), and a rollback that throws would displace the
        failure it is unwinding. Each list is empty until its repoint has fully applied, so
        it names either every reference that was repointed or none — never a mix.
        """
        self._env.entities[self.entity_name] = sim_entity
        for term, attr in self._repointed_refs:
            setattr(term, attr, sim_entity)
        restored = self._restore_swapped_commands()

        swapped = f" and the command term(s) {restored}" if restored else ""
        en.logger.warning(
            f"[deploy] {type(self).__name__} construction failed; rolled back the "
            f"{self.entity_name!r} entity swap{swapped}, so the env still holds what it was built with."
        )

    def _restore_swapped_commands(self) -> list[str]:
        """Close every substituted command term and put the originals back. Idempotent.

        Shared by the construction rollback and :meth:`_teardown_common`, because after a
        deploy a substituted term is not merely stale but *dead*: an
        :class:`~eden.managers.terms.commands.velocity.OperatorVelocityCommand` whose device
        has been stopped reports zeros for ever, silently, to anything that goes on using the
        env in the same process. Unlike the entity swap — where the ``RobotStateEntity`` at
        least still holds the last state it read — there is nothing to be gained by leaving
        it installed, and restoring means a second deployment on the same env finds the
        trained generator again rather than a substitute it cannot recognise.

        Cannot raise — by construction, not by assumption. Closing a substituted term reaches
        a third-party input driver (``pynput``, ``inputs``), which is not code this method can
        vouch for: on the rollback path a throw would displace the very failure ``__init__``
        is unwinding, and on either path it would abandon the terms it had not reached yet,
        leaving a live device attached to a term nothing will close again. Each step is
        therefore guarded and logged, and the loop always runs to the end. Returns the names
        it restored, in sorted order.
        """
        swapped = getattr(self, "_swapped_commands", None)
        if not swapped:
            return []
        manager = getattr(self._env, "command_manager", None)
        for name, old_term in swapped.items():
            if manager is None:
                continue
            try:
                close = getattr(manager.terms.get(name), "close", None)
                if close is not None:
                    close()
            except Exception as exc:
                en.logger.error(f"[deploy] Could not close the substituted command term {name!r}: {exc!r}")
            try:
                manager.restore_term(name, old_term)
            except Exception as exc:
                en.logger.error(f"[deploy] Could not restore the env's own command term {name!r}: {exc!r}")
        # Recorded as (term, attr, old_term) so one list unwinds several swaps.
        for term, attr, old_term in self._repointed_command_refs:
            setattr(term, attr, old_term)

        restored = sorted(swapped)
        self._swapped_commands = {}
        self._repointed_command_refs = []
        return restored

    def _apply_command_overrides(self) -> None:
        """Substitute the configured command terms, repointing anything that cached them.

        The counterpart of the entity swap on the command manager. ``CommandManager.swap_term``
        builds the replacement exactly as ``_prepare_terms`` would and checks its command shape
        against the term it replaces; this method then makes the swap *visible* to terms that
        resolved the old one once in their own ``build()`` (the pattern
        ``self._command = env.command_manager.get_term(...)``), which would otherwise keep
        reading the replaced generator with nothing to show for it.

        Only the term trees that actually run at deploy are repointed — observation terms and
        the other command terms. Reward, termination and curriculum terms are not stepped by
        the deploy loop, so a cached handle there is inert; a caller reusing this swap in a
        *training* env would have to widen the scan.
        """
        if not self.command_overrides:
            return
        manager = self._env.command_manager
        for name, options in self.command_overrides.items():
            old_term = manager.swap_term(name, options)
            self._swapped_commands[name] = old_term
            self._repoint_command_term_refs(old_term, manager.terms[name])
            en.logger.info(
                f"[deploy] Substituted command term {name!r}: "
                f"{type(old_term).__name__} -> {type(manager.terms[name]).__name__}."
            )

    def _repoint_command_term_refs(self, old_term, new_term) -> None:
        """Point every deploy-time term that cached ``old_term`` at ``new_term``."""
        obs_terms = [term for group in self._env.observation_manager._group_obs_terms.values() for term in group]
        command_terms = [term for term in self._env.command_manager.terms.values() if term is not new_term]
        refs = cached_refs(
            [*obs_terms, *command_terms],
            old_term,
            what="the replaced command term",
            remedy=(
                "Read the command lazily instead — `env.command_manager.get_command(name)` in "
                "compute() — or expose the term as a plain attribute so the swap can repoint it."
            ),
        )
        for term, attr in refs:
            setattr(term, attr, new_term)
        # Recorded only once every setattr has landed, so :meth:`_undo_swaps` never sees a
        # partial list.
        self._repointed_command_refs.extend((term, attr, old_term) for term, attr in refs)
        if refs:
            en.logger.info(
                f"[deploy] Repointed {len(refs)} cached reference(s) to the replaced command term: "
                f"{[f'{type(t).__name__}.{a}' for t, a in refs]}"
            )

    def _repoint_entity_refs(self, old_entity, new_entity) -> None:
        """Point every observation and action term that cached the swapped sim entity at ``new_entity``.

        On hardware, observations must read the live :class:`RobotStateEntity`, not the
        simulated entity. Function-based terms re-resolve ``env.entities[name]`` each
        ``compute()`` so the ``env.entities`` swap already reaches them; class-based terms
        instead cache the handle in ``build()`` under inconsistent names (``entity`` /
        ``_entity`` / ``robot`` / ``_robot`` / …). Repointing only an attribute literally named
        ``entity`` would leave the rest **silently reading stale sim state on the real robot** —
        e.g. a projected-gravity term caching ``_robot`` would feed the policy the simulated base
        orientation.

        Action terms are scanned on the same pass and for the same reason. ``action_to_payload``
        runs the action manager for every checkpoint slot and for the env-owning policy slot,
        and a controller whose reference source re-reads the entity each step —
        ``ReferenceSource.DELTA`` reads ``get_dofs_pos`` inside ``compute()`` — would otherwise
        build the real robot's target out of simulated joint positions. The pose-derived
        offsets the other reference sources use are resolved once in ``build()``, so they are
        values by the time the swap happens and need nothing.

        The action modifiers those terms own are scanned too (:func:`with_owned_modifiers`),
        because they cache the entity one level deeper than this scan looks and read it every
        step — see that function for which ones and why it matters. So are the observation terms'
        ``modifiers``, for the same reason.

        We repoint by **object identity**, so every cached handle is caught regardless of
        attribute name, then raise if a term holds the entity in a form we can't safely
        repoint — the forms we detect being a bound method or membership in a top-level
        list/tuple/set/dict. (References nested deeper — inside a helper object, a
        ``functools.partial``, or a closure — are out of reach and would still read stale;
        we have verified none exist in the current term tree.) The point is to turn a
        would-be silent stale read into a loud failure at deploy init.

        We scan each term's ``__dict__``; a term using ``__slots__`` can't be introspected
        this way, so rather than silently skip it (re-creating the exact stale-read bug this
        guards against) we raise.

        The scan (:func:`cached_entity_refs`) completes before a single ``setattr``, so a
        term tree that fails it is left untouched rather than half-repointed — which is what
        lets ``__init__`` roll the swap back cleanly when a later step raises.
        """
        terms = [term for group in self._env.observation_manager._group_obs_terms.values() for term in group]
        # An observation term's modifiers transform frames and have no reason to hold the entity, but one
        # that did would read the simulated robot on hardware, so they are scanned like the action side's.
        terms += [modifier for term in list(terms) for modifier in getattr(term, "_modifiers", ())]
        terms += with_owned_modifiers(self._env.action_manager.terms.values())
        refs = cached_entity_refs(terms, old_entity)
        for term, attr in refs:
            setattr(term, attr, new_entity)
        # Recorded only once every setattr has landed, so :meth:`_undo_swaps` never
        # sees a partial list.
        self._repointed_refs = refs
        repointed = [f"{type(term).__name__}.{attr}" for term, attr in refs]
        en.logger.info(
            f"[deploy] Repointed {len(repointed)} cached term-entity reference(s) to real robot state"
            + (f": {repointed}" if repointed else " (all terms resolve the entity lazily).")
        )

    def _resolve_control_dt(self) -> float:
        if self.control_dt is not None:
            return self.control_dt
        if self.control_freq is not None:
            return 1.0 / self.control_freq
        decimation = self.decimation or self.cfg.env_options.decimation
        return self.cfg.env_options.sim_dt * decimation

    @classmethod
    def _resolve_robot_options(cls, cfg: EdenRLConfig, entity_name: str) -> RobotOptions:
        robot_options = scene_entity_options(cfg.scene_options, entity_name)
        if not isinstance(robot_options, RobotOptions):
            raise TypeError(f"Entity '{entity_name}' is not a RobotOptions instance: {type(robot_options)}")
        return robot_options

    def _resolve_disabled_joints(self) -> None:
        """Resolve :attr:`disable_joints` into :attr:`disabled_dofs`, the DOFs sent kp=kd=0 (free/limp).

        Each token is matched against the deployment DOF names as an exact name or an
        ``fnmatch`` pattern. A token matching nothing is an error (typo guard). The zeroing
        itself is :meth:`wire_gains`', so every slot's gains get it, not only the boot slot's.

        :attr:`disabled_dofs` is empty when nothing was disabled. Anything comparing the deployed
        gains against what a policy expects has to exclude them: their kp/kd are 0 by the
        operator's own request, not by a mismatch, and re-deriving that from the patterns at the
        call site would duplicate the matching done here.
        """
        self.disabled_dofs: tuple[str, ...] = ()
        self._disabled_idx = np.zeros(0, dtype=np.int64)
        tokens = as_name_tuple(
            self.disable_joints, "disable_joints", "each character would be matched as its own pattern"
        )
        if not tokens:
            return
        import fnmatch

        disabled: set[str] = set()
        for token in tokens:
            matched = [name for name in self.dofs_name if fnmatch.fnmatchcase(name, token)]
            if not matched:
                raise ValueError(f"--disable_joints token {token!r} matched no DOF. Robot DOFs: {self.dofs_name}")
            disabled.update(matched)
        self.disabled_dofs = tuple(sorted(disabled))
        self._disabled_idx = np.array([self.__dofs_idx[name] for name in self.disabled_dofs], dtype=np.int64)
        en.logger.warning(
            f"[deploy] DISABLED {len(disabled)} joint(s) — sending kp=kd=0 (free/limp): {sorted(disabled)}. "
            "They will not hold position or damp; never disable load-bearing leg joints on an unsupported robot."
        )

    def wire_gains(self, robot: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """The kp/kd this deployment sends for a policy trained on ``robot``, in deployment DOF order.

        ``robot`` is a policy's pinned :class:`RobotOptions` (``PolicyBase.robot_options()``): its
        ``default_dofs_kp`` / ``default_dofs_kd`` on every DOF it names, the deployed robot's on
        the rest, matched by DOF name. ``None`` -- a policy pinning no robot, a checkpoint trained
        in this env -- or a robot declaring no gains (a composite, or an empty map, as
        ``G1Dofs23``'s) is the deployed robot's throughout. Both are then scaled by
        ``deploy_kp_kd_scale`` and zeroed on :attr:`disabled_dofs`, the operator's choices for the
        whole run.

        Raises ``ValueError`` as :func:`resolve_pinned_gains` does, which resolves the pinned maps.
        """
        kp, kd = self._robot_dofs_kp.copy(), self._robot_dofs_kd.copy()
        for gains, pinned in zip((kp, kd), resolve_pinned_gains(robot, self.dofs_name), strict=True):
            for name, value in pinned.items():
                gains[self.__dofs_idx[name]] = value
        scale = self._gain_scale()
        kp *= scale
        kd *= scale
        kp[self._disabled_idx] = 0.0
        kd[self._disabled_idx] = 0.0
        return kp, kd

    def _gain_scale(self) -> float | np.ndarray:
        """The ``deploy_kp_kd_scale`` :meth:`wire_gains` applies: one number here, one per DOF on a composite rig."""
        return self.deploy_kp_kd_scale

    def set_gains(self, dofs_kp: np.ndarray, dofs_kd: np.ndarray, *, ramp_steps: int | None = None) -> None:
        """Move the kp/kd every command sends to ``dofs_kp`` / ``dofs_kd`` over ``ramp_steps`` control steps.

        The targets are wire gains in deployment DOF order -- :meth:`wire_gains`' output, already
        scaled and with disabled joints zeroed. ``ramp_steps`` defaults to :attr:`gain_ramp_steps`;
        ``0`` jumps, which is only right before :meth:`connect`, while nothing is on the wire.

        The ramp is linear from whatever is being sent now, a ramp still in progress included, and
        advances once per :meth:`step`. Every payload reads the current gains -- the policy's, the
        STOP hold, the INIT ramp, the end-of-clip stand -- so all of them follow it. A policy keeps
        driving through it: holding the pose instead topples a humanoid, which only its policy
        balances (see "Each slot runs at its own PD gains" in ``docs/guides/deployment.md``).

        Raises on a backend that does not send each command's gains (:attr:`sends_command_gains`):
        there the call would change nothing on the robot, which is not what its caller asked for.
        """
        if not self.sends_command_gains:
            raise RuntimeError(
                f"{type(self).__name__}.set_gains would change nothing on the robot: this backend does not send "
                "each command's kp/kd (sends_command_gains=False), it programs its gains once at connect or runs "
                "its own loop. Check sends_command_gains before calling it, and set the backend's own gain "
                "options to change what the robot runs at."
            )
        kp = np.array(dofs_kp, dtype=np.float64).reshape(-1)
        kd = np.array(dofs_kd, dtype=np.float64).reshape(-1)
        if kp.shape != (self.num_dofs,) or kd.shape != (self.num_dofs,):
            raise ValueError(
                f"{type(self).__name__}.set_gains takes one kp and one kd per deployment DOF ({self.num_dofs}, "
                f"in dofs_name order); got {kp.shape[0]} kp and {kd.shape[0]} kd. Build them with wire_gains(robot)."
            )
        steps = self.gain_ramp_steps if ramp_steps is None else ramp_steps
        if steps <= 0:
            self._gain_target = None
            self.default_dof_kp, self.default_dof_kd = kp, kd
        else:
            self._gain_target, self._gain_steps_left = (kp, kd), steps

    def _advance_gain_ramp(self) -> None:
        """Take one control step of the :meth:`set_gains` ramp, if one is running.

        Each step closes ``1 / steps_left`` of the remaining gap, which is a straight line from
        where the ramp started and lands on the target exactly. Rebinds the gain arrays rather than
        writing into them, so a payload already built from them -- the MuJoCo physics thread's held
        command, a recorded step -- keeps the gains it was sent with.
        """
        if self._gain_target is None:
            return
        kp, kd = self._gain_target
        self.default_dof_kp = self.default_dof_kp + (kp - self.default_dof_kp) / self._gain_steps_left
        self.default_dof_kd = self.default_dof_kd + (kd - self.default_dof_kd) / self._gain_steps_left
        self._gain_steps_left -= 1
        if self._gain_steps_left == 0:
            self._gain_target = None
            self.default_dof_kp, self.default_dof_kd = kp, kd  # exact, not to within float rounding

    def programmed_gains(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The kp/kd the robot runs whichever slot is active, on a backend without :attr:`sends_command_gains`.

        In deployment DOF order and in :meth:`wire_gains`' units, so a slot's gains can be compared
        with them; ``None`` when the robot's gains are not in those units (XHand firmware units, a
        controller's own stiffness) or it has none (a position loop), which is the default. A
        backend that sends each command's gains runs no fixed set, and is never asked.
        """
        return None

    def command_gain_dofs(self) -> np.ndarray:
        """Per DOF, whether each command's kp/kd reaches its motor: :attr:`sends_command_gains`, on every DOF.

        A composite rig answers per part, since its body may take each command's gains while its
        hands run their own (:class:`~eden.extensions.deployment.composite.CompositeDeployment`).
        """
        return np.full(self.num_dofs, self.sends_command_gains)

    # ------------------------------------------------------------------
    # Debug recording (commanded vs measured history -> .npz on close)
    # ------------------------------------------------------------------

    _REC_KEYS = ("stamp", "cmd_pos", "cmd_kp", "cmd_kd", "q", "qd", "tau", "base_quat", "base_ang_vel", "base_lin_acc")

    def _init_recording(self) -> None:
        """Allocate the pending-chunk buffer and the streaming writer (no-op when disabled).

        ``_rec`` holds only the steps since the last flush, so memory stays flat however
        long the run is; :class:`StreamingRecorder` owns everything already on disk.
        """
        self._rec: dict[str, list] | None = {k: [] for k in self._REC_KEYS} if self.debug_record else None
        self._recorder: StreamingRecorder | None = None
        if not self.debug_record:
            return
        if self.debug_record_flush_steps < 1:
            raise ValueError(f"debug_record_flush_steps must be >= 1, got {self.debug_record_flush_steps}")
        path = self.debug_record_path or os.path.join(en.log_dir, "deploy_record.npz")
        self._recorder = StreamingRecorder(path, self.dofs_name, flush_steps=self.debug_record_flush_steps)
        en.logger.info(
            f"[deploy] debug_record: streaming to {path} in {self.debug_record_flush_steps}-step chunks "
            "(a killed run leaves recoverable chunks alongside it)."
        )

    def _record_step(self, payload: RobotCommand, state: RobotState) -> None:
        """Append one control step's command + measured proprioception. No-op if disabled.

        Deployment backends that override :meth:`step` should call this after
        :meth:`state_to_observation` (in a ``finally``, as :meth:`step` does), so the recorded
        command matches the step's measurement and ``state.extra`` already holds what the
        observation pass adds to it (the velocity estimator's ``base_lin_vel_estimate``,
        ``contacts`` and ``forces``).
        """
        rec = getattr(self, "_rec", None)
        if rec is None:
            return
        zeros = np.zeros(self.num_dofs)

        def _vec(x, fallback):
            return np.asarray(x if x is not None else fallback, dtype=np.float64).copy()

        rec["stamp"].append(float(state.stamp))
        rec["cmd_pos"].append(_vec(payload.dofs_pos, zeros))
        rec["cmd_kp"].append(_vec(payload.dofs_kp, self.default_dof_kp))
        rec["cmd_kd"].append(_vec(payload.dofs_kd, self.default_dof_kd))
        rec["q"].append(_vec(state.dofs_pos, zeros))
        rec["qd"].append(_vec(state.dofs_vel, zeros))
        rec["tau"].append(_vec(state.dofs_torque, zeros))
        rec["base_quat"].append(_vec(state.base_quat, np.zeros(4)))
        rec["base_ang_vel"].append(_vec(state.base_ang_vel, np.zeros(3)))
        rec["base_lin_acc"].append(_vec(state.base_lin_acc, np.zeros(3)))
        # Backend-specific numeric extras (e.g. Go1 motor temperature / mode / battery)
        # ride along under an ``x_`` prefix so --debug_record captures them too. A backend
        # populates the same keys every step; a channel this step lacks (a tick whose
        # observation pass raised part-way) is padded with NaN so every channel keeps one row
        # per step.
        seen: set[str] = set()
        for key, val in state.extra.items():
            try:
                arr = np.asarray(val, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                continue
            if arr.size:
                rec.setdefault(f"x_{key}", []).append(arr)
                seen.add(f"x_{key}")
        for key, rows in rec.items():
            if key.startswith("x_") and key not in seen and rows:
                rows.append(np.full_like(rows[-1], np.nan))

        # Hand the chunk off once it is full and start a fresh buffer. ``append`` only
        # queues it, so the control thread never waits on the disk. ``getattr`` for the same
        # reason ``_rec`` above uses it: several tests drive ``_record_step`` on a deployment
        # built with ``__new__``, which never ran ``_init_recording``.
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and recorder.append(rec):
            self._rec = {k: [] for k in self._REC_KEYS}

    def _dump_record(self) -> None:
        """Flush the last partial chunk and merge the stream into a single ``.npz``.

        Idempotent: clears the buffer and the recorder so a second ``close`` is a no-op.
        Deployment ``close`` overrides should call this (via ``_teardown_common``).

        Nothing here is load-bearing for *having* a recording -- every full chunk is
        already durable on disk. This is the tidy-up that turns them into the one file
        ``--debug_record_path`` names; if the process dies before it runs,
        :func:`~eden.extensions.deployment.utils.record.load_deploy_record` reads the
        chunks instead.
        """
        recorder = getattr(self, "_recorder", None)
        if recorder is None:
            return
        rec = getattr(self, "_rec", None)
        steps = recorder.close(rec)
        self._rec = {k: [] for k in self._REC_KEYS}
        self._recorder = None
        if steps:
            en.logger.info(f"[deploy] debug_record: wrote {steps} steps to {recorder.path}")

    def _apply_target_lpf(self, dofs_pos: np.ndarray) -> np.ndarray:
        """First-order low-pass the commanded joint targets (no-op when ``target_lpf>=1``).

        Deployment ``send_payload`` should pass ``payload.dofs_pos`` through this
        once per step so it filters every command path (init / hold / policy).
        Warm-started from the first target, so it has no startup transient.
        """
        if self.target_lpf >= 1.0:
            return dofs_pos
        a = float(self.target_lpf)
        if self._lpf_prev is None or self._lpf_prev.shape != dofs_pos.shape:
            self._lpf_prev = np.asarray(dofs_pos, dtype=np.float64).copy()
        else:
            self._lpf_prev = a * dofs_pos + (1.0 - a) * self._lpf_prev
        # Return a copy so the caller (``payload.dofs_pos = _apply_target_lpf(...)``)
        # never aliases the internal filter state.
        return self._lpf_prev.copy()

    def _build_term_action_maps(self) -> list[tuple[ActionTerm, torch.Tensor, torch.Tensor]]:
        """Precompute this deployment's action-term → DOF maps (see :func:`build_term_action_maps`)."""
        return build_term_action_maps(
            self._env.action_manager.terms,
            self.dofs_name,
            self._env.device,
            self.ignore_action_terms,
            entity_name=self.entity_name,
        )

    def _log_unapplied_action_delays(self) -> None:
        """Name the deployed action terms whose ``ActionDelay`` no deployment applies.

        :class:`~eden.managers.modifiers.actions.actuators.ActionDelay` models the simulated
        actuator's latency. It acts inside the env's ``apply_actions()``, once per physics step,
        and a deployment never calls that: every backend sends ``processed_action`` as the policy
        issued it. A real robot then has only its own latency, and the MuJoCo sim-to-sim robot has
        none. That is by design, so this logs at INFO rather than warning, and only when the task
        configures a delay at all. Only terms the payload carries are named, so a delayed term on
        another entity, which is never sent, is left out.
        """
        from eden.managers.modifiers.actions.actuators import ActionDelay

        deployed = {id(term) for term, _src_idx, _dst_idx in self._term_action_maps}
        delayed = [
            name
            for name, term in self._env.action_manager.terms.items()
            if id(term) in deployed and any(isinstance(m, ActionDelay) for m in with_owned_modifiers([term]))
        ]
        if delayed:
            en.logger.info(
                f"[deploy] ActionDelay on action term(s) {delayed} is not applied: a deployment sends each command "
                "as the policy issued it, so a robot has only its own latency and the MuJoCo robot has none."
            )

    def write_dofs_targets(self, targets: torch.Tensor, dofs_name: list[str] | tuple[str, ...]) -> None:
        """Write absolute joint-position targets directly into the processed-action buffer.

        Bypass for an extension-policy slot, which converts its policy's raw action
        with that policy's *own* controller and writes the resulting targets here,
        instead of handing raw actions to the env's action manager (which belongs to
        another policy). ``targets[i]`` is name-mapped onto deployment DOF
        ``dofs_name[i]``; follow with ``step(None)``, which builds the payload from
        this buffer without running the action manager.

        Joint positions only — the payload's ``dofs_vel`` and ``dofs_torque`` fields
        have no writer on this path, which is what keeps a velocity, torque or
        velocity-feedforward controller out of a deploy slot.

        Raises ``KeyError`` if a name is not a deployment DOF.
        """
        key = tuple(dofs_name)
        idx = self._direct_target_maps.get(key)
        if idx is None:
            idx = torch.tensor([self.__dofs_idx[name] for name in key], dtype=torch.long, device=self._env.device)
            self._direct_target_maps[key] = idx
        self._processed_action[idx] = targets.to(device=self._processed_action.device).flatten()
        self._direct_targets_written = True

    def action_to_payload(self, action: Optional[torch.Tensor | dict[str, torch.Tensor]] = None) -> RobotCommand:
        if action is not None:
            self._env.action_manager.compute(action)

            for term, src_idx, dst_idx in self._term_action_maps:
                self._processed_action[dst_idx] = term.processed_action[0, src_idx]
        elif not self._direct_targets_written:
            # :meth:`step` never gets here -- it sends nothing for ``None`` before the first
            # write. A direct caller asking for a payload out of the untouched buffer is a
            # mistake, and an all-zeros command is the wrong thing to hand it.
            raise RuntimeError(
                f"{type(self).__name__}.action_to_payload(None) before anything wrote joint targets. "
                "``action=None`` means 'the active slot already wrote absolute targets via "
                "write_dofs_targets' -- but the buffer is still its all-zeros initializer, and "
                "sending that would drive every joint to 0 rad at the default gains. Have the "
                "policy write its targets first, or pass an action for the action manager to process."
            )

        return RobotCommand(
            dofs_pos=self._processed_action.cpu().numpy(),
            dofs_vel=np.zeros(self.num_dofs),
            dofs_torque=np.zeros(self.num_dofs),
            dofs_kp=self.default_dof_kp,
            dofs_kd=self.default_dof_kd,
        )

    # Backends whose "robot" is a simulator set this: only they can place the base, which is
    # what ``init_target="motion_rsi"`` asks for.
    supports_base_placement: ClassVar[bool] = False

    # Does each command's ``dofs_kp`` / ``dofs_kd`` reach the motors? Off for a robot that runs
    # its own position loop or takes its gains once at connect (the Wuji and XHand hands, the
    # Franka): there :meth:`set_gains` changes nothing on the robot, so a policy slot switch
    # cannot bring that slot's gains along.
    sends_command_gains: ClassVar[bool] = True

    # Heading on the operator cheat-sheet; defaults to the class name.
    operator_guide_title: ClassVar[str] = ""

    # Read the robot and sync it onto the scene *before* the commands publish, so a command that
    # grades the robot's state sees this tick's? Off: the commands publish first and the read
    # follows, one tick of skew the observation side documents (``state_to_observation``). The
    # Wuji hand turns it on, the order its sim-to-sim runs were validated with. It acts on the env,
    # not the wire, so a composite rig follows it only when every part sets it.
    sync_before_commands: ClassVar[bool] = False

    # Advance the env's ``episode_length_buf`` once per policy tick, after the command is sent and
    # the period waited out? The scene is never stepped on the deploy path, so nothing else does; a
    # backend whose action warm-ups and clip clocks read it (the Wuji hand's port) turns it on. The
    # clock is the policy's: it restarts at every START and stands still through STOP and INIT, so
    # a warm-up gated on it (the reorient task's) runs from the handover, not from boot. A composite
    # rig shares one clock, so it advances it only when every part sets this.
    advances_episode_clock: ClassVar[bool] = False

    # May this backend hand itself control with nobody watching? Only a simulator may: with no
    # input source open, ``_auto_advance`` runs INIT and then START on its own, which is right
    # for a headless sim-to-sim rollout and is a robot moving unattended anywhere else. Left
    # False here so a backend has to *say* it is safe rather than inherit it by omission.
    allow_unattended_start: ClassVar[bool] = False

    def _validate_init_target(self) -> None:
        """Reject an unknown ``init_target``, and ``motion_rsi`` on a backend driving hardware."""
        if self.init_target not in ("default", "motion", "motion_rsi"):
            raise ValueError(f"init_target must be 'default', 'motion' or 'motion_rsi'; got {self.init_target!r}.")
        if self.init_target == "motion_rsi" and not self.supports_base_placement:
            raise ValueError(
                f"init_target='motion_rsi' teleports the base to the clip's first frame, which "
                f"{type(self).__name__} cannot do — it drives a real robot. Use 'motion' to ramp "
                "the joints there instead (the base then has to get there on its own), or "
                "'default' for the robot's own stance."
            )

    def _validate_on_motion_end(self) -> None:
        """Reject an unknown ``on_motion_end`` / ``motion_end_slot`` at construction.

        The CLI already constrains ``--on_motion_end`` with ``choices=``; this covers the
        programmatic and ``--config`` paths, where the alternative would be a ``ValueError`` raised
        from ``_motion_end_payload`` on the first step after the operator pressed START — with
        the policy engaged and the robot live.
        """
        if self.on_motion_end not in ("continue", "stand", "switch"):
            raise ValueError(f"on_motion_end must be 'continue', 'stand' or 'switch'; got {self.on_motion_end!r}.")
        if self.on_motion_end == "switch" and self.motion_end_slot < 1:
            raise ValueError(f"motion_end_slot is 1-based, like the operator's 1-9 keys; got {self.motion_end_slot}.")

    def _motion_start_dofs(self) -> np.ndarray | None:
        """Frame-0 joint pose of the tracked reference in wire DOF order, or ``None``.

        ``None`` when nothing in this env tracks ``entity_name`` — no motion manager, or one
        whose terms target other entities — so callers fall back to the default pose.
        """
        mm = self._motion_manager
        if mm is None or mm.get_target_dofs_name(self.entity_name) is None:
            return None
        from eden.utils.motion_reference.lookup import (
            assert_motion_field_present,
            current_motion_frame,
            get_motion_dofs_idx,
        )

        # An unpopulated ``dofs_pos`` reads back as zeros through ``make_empty``, which would
        # ramp the robot to all-zero joints instead of to frame 0 — raise instead.
        assert_motion_field_present(self._env, self.entity_name, "dofs_pos")
        ref = mm.data[self.entity_name]
        # The reference's DOF axis is the motion file's order, which need not match the wire
        # order — resolve by name rather than assuming they line up.
        sel = get_motion_dofs_idx(self._env, self.entity_name, self.dofs_name)
        return current_motion_frame(ref.dofs_pos, sel)[0].detach().cpu().numpy().astype(np.float64)

    def _init_pose_target(self) -> np.ndarray:
        """Joint targets the INIT ramp drives to, per ``init_target``, in wire DOF order."""
        self._validate_init_target()
        if self.init_target == "default":
            return self.default_dof_pos
        dofs = self._motion_start_dofs()
        if dofs is None:
            en.logger.warning(
                f"[deployment] init_target={self.init_target!r} but this env tracks no motion for "
                f"'{self.entity_name}'; INIT ramping to the default pose instead."
            )
            return self.default_dof_pos
        return dofs

    # -- operator state machine -------------------------------------------

    def _setup_inputs(self) -> None:
        """Open the operator input sources: the robot's own remote (:meth:`_remote_inputs`), then the keyboard."""
        from eden.extensions.deployment.utils.operator_input import KeyboardInput

        if self._inputs:
            return
        inputs: list[Any] = list(self._remote_inputs())
        keyboard = KeyboardInput()
        if not self.operator_keyboard:
            en.logger.info(
                "[deploy] Terminal keyboard reader off (operator_keyboard=False): the script relays commands."
            )
        elif keyboard.start():
            inputs.append(keyboard)
            en.logger.info("[deploy] Keyboard control active (s/x/i/k + Enter).")
        else:
            remote = " use the robot's remote." if inputs else " no operator input is open."
            en.logger.info(f"[deploy] No TTY — keyboard disabled;{remote}")
        self._inputs = inputs

    def _remote_inputs(self) -> list[Any]:
        """The robot's own operator inputs (a handheld remote), each with a ``poll() -> list[str]``. None here.

        A backend whose robot carries a remote overrides this rather than :meth:`_setup_inputs`, so
        the one keyboard listener is opened in one place -- a deployment driving several robots
        (:class:`~eden.extensions.deployment.composite.CompositeDeployment`) gathers every part's
        remote and still opens a single keyboard.
        """
        return []

    # -- the operator commands, for a script ----------------------------------

    def request_init(self) -> None:
        """INIT, as the operator's key or remote does: ramp from the measured pose to the ``init_target`` pose."""
        self._warned_no_operator = True  # a script pressing the keys is the operator
        self._handle_init()

    def request_start(self) -> None:
        """START, as the operator's key or remote does: hand control to the policy.

        A script that drives a deployment without an operator calls this explicitly -- nothing a
        backend does on its own starts the policy on a robot (:attr:`allow_unattended_start`).
        """
        self._warned_no_operator = True  # a script pressing the keys is the operator
        self._handle_start()

    def request_stop(self) -> None:
        """STOP, as the operator's key or remote does: disengage the policy and hold the measured pose."""
        self._warned_no_operator = True  # a script pressing the keys is the operator
        self._handle_stop()

    @property
    def sending_hold(self) -> bool:
        """Whether the payload :meth:`send_payload` is sending is the state machine's own, not the policy's.

        True for the STOP/boot hold, the INIT ramp and the START tick, the commands that hold or
        place the robot's joints. A backend whose robot carries a tool the joints do not describe
        leaves it alone on these: the Franka keeps its gripper at the policy's last command, so a
        STOP or INIT does not open a grasp. False for a command sent from outside :meth:`step`.
        """
        return self._sending_hold

    @property
    def init_done(self) -> bool:
        """Whether an INIT ramp has reached its target (and holds it until START or STOP)."""
        return self._get_ready and self._init_count >= self.init_steps

    def _dispatch_command(self, cmd: str) -> None:
        """Apply one operator command. Shared by every input source and backend."""
        from eden.extensions.deployment.utils.operator_input import (
            CMD_GANTRY_DOWN,
            CMD_GANTRY_UP,
            CMD_INIT,
            CMD_KILL,
            CMD_NEXT_POLICY,
            CMD_START,
            CMD_STOP,
            SWITCH_PREFIX,
        )

        if cmd == CMD_START:
            self._handle_start()
        elif cmd == CMD_STOP:
            self._handle_stop()
        elif cmd == CMD_INIT:
            self._handle_init()
        elif cmd == CMD_KILL:
            en.logger.info("[deploy] KILL command received — shutting down.")
            raise KeyboardInterrupt
        elif cmd == CMD_NEXT_POLICY:
            self._pending_policy_switches.append("next")
        elif cmd in (CMD_GANTRY_UP, CMD_GANTRY_DOWN):
            self._adjust_gantry(1.0 if cmd == CMD_GANTRY_UP else -1.0)
        elif cmd.startswith(SWITCH_PREFIX):
            # 1-based slot from the operator -> 0-based index for the loop.
            self._pending_policy_switches.append(int(cmd[len(SWITCH_PREFIX) :]) - 1)

    @property
    def policy_active(self) -> bool:
        """Whether the action passed to :meth:`step` reaches the robot this tick.

        False until START hands control over, and again after STOP or INIT.
        """
        return self._use_policy

    def consume_start_events(self) -> int:
        """Return and clear the number of START presses since the last drain.

        The deploy loop re-arms the active policy slot on START (reset internal state, rewind
        command clocks) because :meth:`_handle_start` re-captures the yaw offset — any policy
        state accumulated while stepping during STOP/hold references the pre-START world frame.
        """
        count = self._pending_start_events
        self._pending_start_events = 0
        return count

    def _watch_frozen_joints(self, payload: RobotCommand, state: RobotState) -> None:
        """Track, per joint, whether its readings ever change while the policy is disengaged.

        A joint whose encoder or driver has stopped reporting returns the same ``q`` / ``qd`` /
        ``tau`` bit for bit, tick after tick, however it is commanded: on one G1 all six wrist
        joints read constant for a whole run while their commands moved 1.5 rad. The
        STOP hold and INIT ramp are the window to catch it, since the INIT ramp moves every
        commanded joint; :meth:`_warn_frozen_joints` reports at START.
        """
        if state.dofs_pos is None or state.dofs_vel is None or state.dofs_torque is None or payload.dofs_pos is None:
            return
        reading = np.concatenate([state.dofs_pos, state.dofs_vel, state.dofs_torque]).reshape(3, -1)
        cmd = np.asarray(payload.dofs_pos, dtype=np.float64)
        watch = getattr(self, "_frozen_watch", None)
        if watch is None:
            self._frozen_watch = {
                "first": reading.copy(),
                "changed": np.zeros(reading.shape[1], dtype=bool),
                "cmd_min": cmd.copy(),
                "cmd_max": cmd.copy(),
                "ticks": 1,
            }
            return
        watch["changed"] |= (reading != watch["first"]).any(axis=0)
        np.minimum(watch["cmd_min"], cmd, out=watch["cmd_min"])
        np.maximum(watch["cmd_max"], cmd, out=watch["cmd_max"])
        watch["ticks"] += 1

    def _warn_frozen_joints(self) -> None:
        """Warn about joints that read bit-constant through STOP/INIT while commanded to move."""
        # ``getattr`` for the reason ``_record_step`` reads ``_rec`` that way: tests drive the
        # state machine on a deployment built with ``__new__``.
        watch, self._frozen_watch = getattr(self, "_frozen_watch", None), None
        if watch is None:
            return
        travel = watch["cmd_max"] - watch["cmd_min"]
        frozen = np.flatnonzero(~watch["changed"] & (travel > 0.01))
        if frozen.size == 0:
            return
        names = [f"{self.dofs_name[i]} (commanded over {travel[i]:.2f} rad)" for i in frozen]
        en.logger.warning(
            f"[deploy] {len(frozen)} joint(s) reported bit-identical q, qd and tau for all {watch['ticks']} "
            f"ticks before START while their command moved: {names}. Their encoders or motor drivers are not "
            "reporting, so the policy is acting on stale readings for them. Check those joints' power and wiring "
            "before trusting this run; --disable_joints does not help, since it only zeroes their gains and the "
            "policy still observes the stale readings."
        )

    def _handle_start(self) -> None:
        self._warn_frozen_joints()
        # The action handed to this tick's ``step`` was computed before START: from the hold's
        # observation, in the pre-START yaw frame, by a slot not yet re-armed. It is not sent;
        # the tick repeats what was being sent before START (see :meth:`_operator_payload`).
        if self._use_policy:
            # START again under the policy (to restart its clip, or after the end-of-clip stand): any
            # latch is from the hold before an earlier START, a pose the robot has since left, so the
            # START tick holds where the robot is now, as STOP would.
            self._release_hold()
        self._handover_pending = True
        self._handover_from_init = self._get_ready
        self._use_policy = True
        self._get_ready = False
        self._pending_start_events += 1
        if self.advances_episode_clock:
            # A new episode for the policy: its warm-ups and phase observations start at the handover.
            self._env.episode_length_buf.zero_()
        # Rewind a motion-imitation reference to frame 0 at the handover so the clip plays from
        # its start each time control is given to the policy.
        self.reset_motion_reference()
        # Capture the yaw the robot is actually facing as control is handed to the policy, so
        # observations align with the trained world frame.
        if self.auto_yaw_align and self.state is not None and self.state.base_quat is not None:
            yaw = self._state_entity.capture_yaw_offset(self.state.base_quat)
            en.logger.info(f"[deploy] START — captured yaw offset {np.degrees(yaw):.2f} deg, policy enabled.")
        else:
            en.logger.info("[deploy] START — policy actions enabled.")
        # Everything the observation windows hold from STOP/INIT is in the pre-START yaw frame and
        # spans a hold and a ramp the policy never trained on as its own past. This step's read
        # starts them afresh, the env-side half of the slot re-arm the deploy loop does at START.
        self._observations_stale = True

    def _handle_stop(self) -> None:
        self._use_policy = False
        self._get_ready = False
        # Re-latch: STOP holds where the robot is *now*, not where a previous hold left off.
        self._release_hold()
        en.logger.info("[deploy] STOP — holding current pose.")

    def _handle_init(self) -> None:
        self._get_ready = True
        self._use_policy = False
        self._init_count = 0
        self._release_hold()
        self._init_start_pos = self.state.dofs_pos.copy() if self.state is not None else None
        # ``init_target="motion"`` ramps to the reference's current frame, and the clip only
        # holds -- it does not rewind -- while the policy is off. Without this a STOP at frame 500
        # followed by INIT ramps the robot to frame 500's pose, not frame 0's. START rewinds again,
        # and the clip holds through INIT, so the handover is unchanged.
        self.reset_motion_reference()
        where = "the default pose" if self.init_target == "default" else f"the {self.init_target} pose"
        en.logger.info(f"[deploy] INIT — interpolating to {where}.")

    def _init_target_payload(self) -> RobotCommand:
        """Interpolate from the pose captured at INIT toward the ``init_target`` pose."""
        target = self._init_pose_target()
        start = self._init_start_pos if self._init_start_pos is not None else target
        alpha = min(self._init_count / max(self.init_steps, 1), 1.0)
        q_target = start + (target - start) * alpha
        if self._init_count == max(self.init_steps, 1):
            en.logger.info("[deploy] INIT reached the target pose and holds it — START hands control to the policy.")
        self._init_count += 1
        return RobotCommand(
            dofs_pos=q_target,
            dofs_vel=np.zeros(self.num_dofs),
            dofs_torque=np.zeros(self.num_dofs),
            dofs_kp=self.default_dof_kp * self.init_dofs_kp_scale,
            dofs_kd=self.default_dof_kd * self.init_dofs_kd_scale,
        )

    # Declared at class level as well as bound in ``__init__``: a test (and a backend doing the
    # same) may build an instance through ``__new__`` and set only the attributes it exercises,
    # and ``_hold_payload`` must not turn that into an ``AttributeError``. Unannotated so the
    # options machinery does not read it as a config field.
    _hold_pos = None

    def _release_hold(self) -> None:
        """Drop the latched hold pose, so the next hold re-latches where the robot then is.

        Called from STOP, INIT and :meth:`reset`, every way into a hold, and from a START that
        arrives while the policy already drives. A latch kept across one of those would command a
        pose the robot has since left — the torque step :meth:`_hold_payload` exists to avoid. A
        START from the hold keeps it: its handover tick repeats that hold.
        """
        self._hold_pos = None

    def _hold_payload(self) -> RobotCommand:
        """Hold the pose latched when this hold began, with the default PD gains.

        Latched once, on the first tick of the hold, and held until :meth:`_release_hold`.
        Re-reading the measured pose every tick instead would leave the command chasing the
        robot: the position error would never exceed one tick of motion, so ``kp`` could only
        act through that lag and the PD would collapse to a damper of ``kd + kp*control_dt`` —
        about 8 N.m.s/rad on a G1 leg against a nominal ``kp`` of 99. A robot that starts
        to sag would be followed to the floor rather than caught, on hardware as much as under
        ``--mujoco``, and STOP is the operator's abort.

        The latch is the pose measured at the previous tick — the same state
        :meth:`_handle_init` captures for its ramp -- so the error starts at zero and the
        restoring torque grows only as the robot deviates. Entering the hold never steps the
        torque, however fast the robot is moving when it is entered.
        """
        if self._hold_pos is None:
            self._hold_pos = self.state.dofs_pos.copy() if self.state is not None else self.default_dof_pos.copy()
        return RobotCommand(
            # A copy per tick: ``send_payload`` rewrites ``dofs_pos`` in place on some
            # backends (the target low-pass, Oli's torque clamp), which must not edit the latch.
            dofs_pos=self._hold_pos.copy(),
            dofs_vel=np.zeros(self.num_dofs),
            dofs_torque=np.zeros(self.num_dofs),
            dofs_kp=self.default_dof_kp,
            dofs_kd=self.default_dof_kd,
        )

    def _auto_advance(self) -> None:
        """Drive INIT then START with no operator, for a run that has no input source.

        Only reached when :meth:`_setup_inputs` opened nothing — no TTY and no remote. A headless
        sim-to-sim rollout would otherwise sit in STOP forever holding its boot pose.

        Gated on :attr:`allow_unattended_start`, which only a simulator sets: anywhere else this
        would INIT and START a powered robot with nobody at the controls, so a hardware backend
        holds in STOP instead and says so once. A script that drives the robot on purpose calls
        :meth:`request_init` / :meth:`request_start` itself, and is left to it.
        """
        if not self.allow_unattended_start:
            # operator_keyboard=False: the script declared it relays the operator's commands itself.
            relayed = not self.operator_keyboard
            if not self._warned_no_operator and not relayed and not (self._use_policy or self._get_ready):
                self._warned_no_operator = True
                en.logger.warning(
                    f"[deploy] {type(self).__name__}: no operator input is open (no remote, no TTY), and it is "
                    "not a simulator that may start itself. Holding in STOP. Run from a terminal, or have the "
                    "script call request_init() / request_start() explicitly."
                )
            return
        if self._get_ready:
            if self._init_count >= self.init_steps:
                self._handle_start()
        elif not self._use_policy and self._init_count == 0:
            en.logger.info("[deploy] No operator input available — running INIT then START automatically.")
            self._handle_init()

    def _operator_payload(self, action) -> RobotCommand | None:
        """The command the operator state machine wants sent this tick, or ``None`` for nothing."""
        for inp in self._inputs:
            for cmd in inp.poll():
                self._dispatch_command(cmd)
        if not self._inputs:
            self._auto_advance()
        if self._get_ready:
            return self._init_target_payload()
        if not self._use_policy:
            return self._hold_payload()
        if self._handover_pending:
            # The START tick keeps sending what it was -- the INIT target, or the hold's latch --
            # and its read is the policy's first observation (the windows cleared, the clock at 0,
            # the clip at frame 0): the deploy loop re-arms the slot and acts on that, so the
            # policy's first command is computed after START, not before it.
            return self._init_target_payload() if self._handover_from_init else self._hold_payload()
        # End-of-clip handover takes precedence over the policy's action.
        payload = self._motion_end_payload()
        if payload is not None:
            return payload
        if action is None and not self._direct_targets_written:
            return None  # nothing to send yet; see :meth:`step`
        return self.action_to_payload(action)

    def _adjust_gantry(self, sign: float) -> None:
        """Crank a simulated bring-up gantry up (+) or down (-). No-op without one.

        Part of the shared operator vocabulary so the sim and the robot take the same keys, but
        only a simulator can rig a hoist — on hardware the keys do nothing and say so once.
        """
        del sign
        if not getattr(self, "_warned_no_gantry", False):
            self._warned_no_gantry = True
            en.logger.info(f"[deploy] {type(self).__name__} has no gantry to raise or lower; ignoring.")

    def _print_control_guide(self) -> None:
        """Print the operator cheat-sheet. Backends with a remote override with its chords."""
        en.logger.info("=" * 60)
        title = self.operator_guide_title or f"{type(self).__name__.upper()} DEPLOY CONTROLS  (keyboard)"
        en.logger.info(title)
        en.logger.info("  s+Enter         START  (hand control to the policy)")
        en.logger.info("  x+Enter         STOP   (hold current pose with PD)")
        en.logger.info("  i+Enter         INIT   (interpolate to the start pose)")
        en.logger.info("  k+Enter         KILL   (soft-damp and exit)")
        en.logger.info("  n+Enter         SWITCH to next policy (if >1 loaded)")
        en.logger.info("  1-9 + Enter     SWITCH to that policy slot")
        if getattr(self, "gantry", False):
            en.logger.info("  + / - + Enter   raise / lower the gantry (SIM RIG, not the robot)")
        en.logger.info("Robot boots in STOP/hold. Press INIT, then START.")
        en.logger.info("=" * 60)

    def motion_finished(self) -> bool:
        """True once every tracked motion reference has played through to its final frame.

        A task drives its reference through the motion *manager*, through a ``MotionCommand``,
        or through both, and this reads whichever are present — the same two paths
        :meth:`_advance_motion_reference` ticks and :meth:`reset_motion_reference` rewinds.
        Reading only the manager would leave ``on_motion_end`` silently inert on every
        command-path deploy (SONIC, ``beyondmimic_moves``), which is most of them.

        The signals are the clocks the observation terms see, so hardware and ``--mujoco``
        agree on when the clip is over. ``False`` when *no* reference is present (a plain RL
        env, or an entity no motion term tracks — a config question, not a reason to park the
        robot mid-run) and when any present reference is still running.

        Costs one device sync per present reference, per control step, until the end fires —
        a host-side branch needs the value on the host. Callers latch: ``on_motion_end``
        stops asking once it has acted (``_standing_from`` / ``_motion_end_switched``), and
        ``on_motion_end="continue"`` never asks at all.
        """
        signals = [sig for sig in (self._manager_motion_finished(), self._command_motion_finished()) if sig is not None]
        return bool(signals) and all(signals)

    def _manager_motion_finished(self) -> bool | None:
        """Manager path: clip phase ``>= 1``, or ``None`` when it tracks nothing here."""
        mm = self._motion_manager
        if mm is None:
            return None
        try:
            phase = mm.get_motion_phase(self.entity_name)
        except (ValueError, AttributeError):
            return None
        return bool(phase.max().item() >= 1.0)

    def _command_motion_finished(self) -> bool | None:
        """Command path: every clip clock held at its last frame, or ``None`` when there is none.

        Duck-typed on :meth:`~eden.managers.terms.commands.mimic.MotionCommand.clip_finished`
        rather than on the term's name or class, because a deploy renames and substitutes
        command terms (``command_overrides``, :meth:`CommandManager.swap_term`) and a task is
        free to carry more than one clip clock. With several, the motion is over when all of
        them are.
        """
        manager = self._env.command_manager
        if manager is None:
            return None
        finished = [term.clip_finished() for term in manager.terms.values() if hasattr(term, "clip_finished")]
        if not finished:
            return None
        return bool(torch.cat([flags.reshape(-1) for flags in finished]).all().item())

    def consume_policy_switches(self) -> list[int | str]:
        """Drain and return pending policy-switch requests (0-based target index or ``"next"``).

        The deploy loop calls this every step and validates the targets against the loaded slot
        count (which the deployment does not know). Requests come from an operator remote /
        keyboard on the hardware backends and from ``on_motion_end="switch"`` here, so both
        reach the loop by the one route — and a MuJoCo sim-to-sim run switches slots exactly as
        the robot does.
        """
        pending = self._pending_policy_switches
        self._pending_policy_switches = []
        return pending

    def request_policy_switch(self, target: int | str) -> None:
        """Queue a slot switch for the deploy loop to apply on its next iteration."""
        self._pending_policy_switches.append(target)

    def _motion_end_payload(self) -> RobotCommand | None:
        """The stand-and-hold command once the clip is done, or ``None`` while the policy drives.

        Latches on the first tick past the end: the ramp interpolates from the pose measured at
        that moment to ``default_dofs_pos`` over ``stand_steps`` ticks and then holds it, which
        is the same interpolation the INIT ramp (:meth:`_init_target_payload`) applies, triggered
        by the clock instead of the operator.
        """
        if self.on_motion_end == "continue":
            return None
        if self.on_motion_end == "switch":
            # Hand over and keep returning None: the newly active slot drives from here, and the
            # request is queued once (re-armed only by reset_motion_reference).
            if not self._motion_end_switched and self.motion_finished():
                self._motion_end_switched = True
                en.logger.info(f"[deployment] Motion finished — handing control to policy slot {self.motion_end_slot}.")
                self.request_policy_switch(self.motion_end_slot - 1)
            return None
        if self._standing_from is None:
            if not self.motion_finished():
                return None
            measured = self.state.dofs_pos if self.state is not None else self.default_dof_pos
            self._standing_from = np.asarray(measured, dtype=np.float64).copy()
            self._stand_count = 0
            en.logger.info(
                f"[deployment] Motion finished — policy disengaged; standing over {self.stand_steps} "
                f"steps ({self.stand_steps * (self.control_dt or 0.0):.2f}s), "
                f"max joint delta {np.degrees(np.abs(self.default_dof_pos - self._standing_from).max()):.1f} deg."
            )
        alpha = min((self._stand_count + 1) / max(self.stand_steps, 1), 1.0)
        self._stand_count += 1
        return RobotCommand(
            dofs_pos=self._standing_from + (self.default_dof_pos - self._standing_from) * alpha,
            dofs_vel=np.zeros(self.num_dofs),
            dofs_torque=np.zeros(self.num_dofs),
            dofs_kp=self.default_dof_kp,
            dofs_kd=self.default_dof_kd,
        )

    def step(self, action: Optional[torch.Tensor | dict[str, torch.Tensor]] = None) -> dict[str, Any]:
        """Send one control step's command, then read and return the resulting observation.

        ``action`` is the policy's raw output, run through the env's action manager by
        :meth:`action_to_payload`. It is ``None`` for an extension-policy slot, which has
        already written its targets with :meth:`write_dofs_targets`: the payload is then
        built from that buffer alone, with the action manager left out of it.

        ``None`` before any such write is the third case — a caller that has nothing to
        send this step — and sends nothing rather than commanding a zero-initialised buffer,
        which on hardware is every joint driven to 0.0 rather than left alone. Hence the
        ``_direct_targets_written`` flag: an all-zero ``_processed_action`` is a legitimate
        command (every joint to its zero position) and cannot be told from an untouched
        buffer by inspecting it.

        Every backend sends what the operator state machine wants: the hold while stopped, the
        INIT ramp, and the policy's command only under START, with the end-of-clip handover
        (``on_motion_end``) taking precedence over the policy. Sending nothing (the third case
        above) is :meth:`_idle_tick`: MuJoCo does not advance and hardware keeps its last command.
        """
        if self.policy_seconds is not None and self._policy_ticks * self.control_dt >= self.policy_seconds - 1e-9:
            en.logger.info(f"[deploy] policy_seconds={self.policy_seconds:g} reached -- ending the run as KILL does.")
            raise KeyboardInterrupt
        self._before_step()
        self._advance_gain_ramp()
        payload = self._operator_payload(action)
        # Only a tick the policy drove advances its clocks: not STOP/hold, not INIT, and not the
        # START tick's hold, so the policy's first observation reads the clock at 0 and the clip
        # at frame 0, as after an env reset.
        driven = self._use_policy and not self._handover_pending
        self._handover_pending = False
        if payload is not None:
            self._sending_hold = not driven
            try:
                self.send_payload(payload)
            finally:
                self._sending_hold = False
        else:
            self._idle_tick()
        if not self._use_policy:
            self._policy_ticks = 0  # a later START runs policy_seconds afresh
        elif driven and payload is not None:
            self._policy_ticks += 1
        if self._rate is not None:
            self._rate.sleep()
        if self.advances_episode_clock and driven:
            self._env.episode_length_buf += 1
        if self.sync_before_commands:
            self.state = self.read_state()
            self._sync_state(self.state)
            self._update_commands(advance_reference=driven)
        else:
            self._update_commands(advance_reference=driven)
            self.state = self.read_state()
        # A control tick is a step, so stateful observation terms advance once here — the deploy
        # loop's phase 7. ``reset`` reads without advancing, as ``EnvBase.reset`` does.
        if not self._use_policy and payload is not None:
            self._watch_frozen_joints(payload, self.state)
        # Recorded after the observation, not straight after ``read_state``: this tick's velocity
        # estimate and foot contacts join ``state.extra`` inside ``state_to_observation``, and a
        # record taken before it would miss them. ``finally`` so a tick whose observation
        # raises -- the one a post-mortem most wants -- is still recorded, with whatever extras it
        # got as far as (``_record_step`` pads the rest).
        try:
            if self.sync_before_commands:  # synced above, once: sync_scene must not run twice a tick
                return self._observe(self.state, post_physics_refresh=True)
            return self.state_to_observation(self.state, post_physics_refresh=True)
        finally:
            if payload is not None:
                self._record_step(payload, self.state)

    @property
    def blocks_for_the_period(self) -> bool:
        """Whether :meth:`send_payload` itself waits out the control period (and so paces the loop). False here.

        A composite rig with such a part lets it pace the rig, as it paces itself deployed alone
        (:class:`~eden.extensions.deployment.composite.CompositeDeployment`).
        """
        return False

    def _before_step(self) -> None:
        """Run at the top of every :meth:`step`, before the action manager. A no-op here; backends override.

        The place for anything that must hold before this tick's command is computed: the Franka
        re-syncs the sim entity its IK reads, the Wuji hand stops the run on stale feedback. A
        backend overrides this rather than :meth:`step`, so the one loop every backend runs stays
        in one place.
        """

    def _idle_tick(self) -> None:
        """A tick with nothing to send (an extension slot that has not written targets yet). A no-op here.

        A backend whose send *is* the control period's wait (the Wuji hand's ``"sync"`` stream)
        overrides this to hold the last target for the period instead, so the tick still lasts one.
        """

    def _prepare_reset(self) -> None:
        """Clear a backend's own per-episode state at the start of :meth:`reset`. A no-op here."""

    def _after_reset_read(self) -> None:
        """Run in :meth:`reset` after its first read and the motion rewind, before the inputs open. A no-op here.

        The place for a backend that moves its robot once the clip sits at frame 0 (the MuJoCo
        backend's ``init_target='motion_rsi'`` placement), re-reads :attr:`state`, or captures
        something from that read. :meth:`reset` returns the observation of :attr:`state` as this
        leaves it.
        """

    def reset(self) -> dict[str, Any]:
        """Boot in STOP/hold and return the first observation: nothing moves until the operator says so.

        Every backend runs the same INIT/START/STOP/KILL state machine (see "The operator state
        machine" in ``docs/guides/deployment.md``). After this, :meth:`step` holds the measured pose
        until INIT ramps the joints to the ``init_target`` pose and START hands control to the
        policy -- from the robot's remote, the keyboard, or a script's :meth:`request_init` /
        :meth:`request_start`. With none of them open, a hardware backend holds in STOP for good.
        """
        # Drop the target low-pass state so the new episode warm-starts from its first command
        # instead of smoothing across the reset from the stale pre-reset target (no-op when
        # target_lpf == 1).
        self._lpf_prev = None
        self._policy_ticks = 0
        # A new episode: its first read starts every observation window afresh, as an env reset does.
        self._observations_stale = True
        self._enter_boot_hold()
        self._prepare_reset()
        self.state = self.read_state()
        # Rewind now so the clip waits at frame 0 through STOP and INIT; START rewinds again.
        self.reset_motion_reference()
        self._after_reset_read()
        self._setup_inputs()
        self._print_control_guide()
        return self.state_to_observation(self.state)

    def _enter_boot_hold(self) -> None:
        """Put the operator state machine in STOP/hold, as at boot: no policy, no INIT ramp, no handover, no latch."""
        self._release_hold()
        self._use_policy = False
        self._handover_pending = False
        self._get_ready = False
        self._init_count = 0
        self._init_start_pos: np.ndarray | None = None

    @property
    def _motion_manager(self):
        """The env's motion manager for motion-imitation tasks, or None for plain RL envs."""
        return getattr(self._env, "motion_manager", None)

    def _update_commands(self, advance_reference: bool) -> None:
        """Advance the motion reference, then publish this step's commands — the deploy loop's phases 3 and 17.

        ``RLEnvBase.step`` runs ``command_manager.pre_physics_advance()`` at phase 3 and
        ``command_manager.compute()`` at phase 17, and the deploy loop needs both in that order: a
        term that derives its command from a time-advancing reference
        (:class:`~eden.managers.terms.commands.mimic.MotionCommand`) must publish against the frame
        the end-of-step observation will read, not the one before it.

        ``advance_reference`` is the caller's policy-active gate. :meth:`step` passes whether the
        policy drove this tick (under START, and not the START tick itself), so the clip holds at
        frame 0 through INIT and hold and starts at the handover — while the commands themselves
        keep publishing, as they do in sim.

        Every deployment ``step`` goes through here, overrides included: a second copy of this
        sequence drifts, and the drift is the frozen clip clock :meth:`_advance_motion_reference`
        describes. ``tests/extensions/deployment/test_base.py`` pins this as the only call site.
        """
        if advance_reference:
            self._advance_motion_reference()
        if self._env.command_manager is not None:
            self._env.command_manager.compute(dt=self.control_dt)
            self._env.command_manager.draw_vis()

    def _advance_motion_reference(self) -> None:
        """Advance a motion-imitation reference clock by one control step (no-op for non-mimic envs).

        A task drives its reference through one of two paths and both are ticked here, because
        ``env.step`` ticks both and the deploy loop calls ``dep.step`` instead:

        * the **manager** path (``MimicRLEnvBase.motion_manager``) — ``motion_manager.compute(dt)``,
          at the top of ``MimicRLEnvBase.step``;
        * the **command** path (``MotionCommand``) —
          :meth:`~eden.managers.command_manager.CommandTerm.advance_reference`, the deployment
          counterpart of the simulation's ``pre_physics_advance`` hook.

        Un-ticked, the reference stays frozen at the frame it started on and a motion-tracking
        policy just holds that pose. The command path's clip clock advances in
        ``pre_physics_advance`` (right in sim: grade the post-physics robot against the advanced
        frame), a hook the deploy loop never reaches, so without this tick every deployed
        ``MotionCommand`` policy, ``beyondmimic_moves`` included, would play a frozen clip.

        Advancing is unconditional here; the policy-active gate lives in :meth:`_update_commands`.
        """
        mm = self._motion_manager
        if mm is not None:
            mm.compute(dt=self.control_dt)
        if self._env.command_manager is not None:
            self._env.command_manager.advance_reference()

    def reset_motion_reference(self) -> None:
        """Restart a motion-imitation reference cleanly at frame 0 (no-op for non-mimic envs).

        Both reference paths are rewound, for the same reason :meth:`_advance_motion_reference`
        advances both: a task drives its reference through the motion *manager*, through a
        ``MotionCommand``, or through both, and the deploy loop is the only thing that reaches
        either one.

        * the **manager** path — :meth:`MotionManager.rewind_to_start` for each env's
          currently-assigned clip, then a refill of the reference buffer;
        * the **command** path — :meth:`CommandTerm.rewind_reference`, the clock-only rewind.

        Motion selection is NOT resampled on either (unlike :meth:`MotionManager.reset`): a deploy
        replays the same clip deterministically instead of the sim's random start phase
        (``motion_start_from_middle_range``), which is correct for a robot that begins from a
        stand. Called at deploy reset (before its INIT loop), at INIT, at START, and on every
        policy switch.

        Rewinding the command path matters most at the handover the operator gate creates. The
        clip holds while the robot is in INIT or hold (:meth:`step` does not advance it), so
        without this a STOP at frame 500 followed by a START hands the policy a reference 500
        frames into the motion while the robot is standing in its INIT stance — a mid-clip pose it
        is nowhere near, on a biped.
        """
        if self._env.command_manager is not None:
            self._env.command_manager.rewind_reference()
        # Rewinding re-arms the clip, so a latched end-of-motion action ends and the policy
        # drives again -- the handover an operator START or a policy switch produces.
        self._standing_from = None
        self._stand_count = 0
        self._motion_end_switched = False
        mm = self._motion_manager
        if mm is None:
            return
        all_idx = torch.arange(self._env.num_envs, device=self._env.device)
        mm.rewind_to_start(all_idx)
        mm.fill_data()

    @property
    def provides_base_lin_vel(self) -> bool:
        """Whether ``base_lin_vel`` observations read a real value on this deploy, or zeros.

        :class:`RobotStateEntity` holds the base linear velocity at zero unless something
        supplies it: the onboard estimator (``estimate_lin_vel``) or a backend that publishes it
        in ``RobotState.extra`` (:meth:`_publishes_base_lin_vel` -- the MuJoCo backend's ground
        truth; no hardware backend does, an IMU measures no velocity). A policy or task whose
        observations read it on a deploy where neither holds is silently-wrong control, which
        is why the CLI refuses that arrangement at startup rather than letting it run.
        """
        return bool(self.estimate_lin_vel) or self._publishes_base_lin_vel()

    def _publishes_base_lin_vel(self) -> bool:
        """Does :meth:`read_state` put ``base_lin_vel`` in ``extra`` every step? Backends override."""
        return False

    @classmethod
    def _check_state_to_observation_override(cls) -> None:
        """Refuse, at construction, an override of :meth:`state_to_observation` that :meth:`step` cannot call.

        ``step`` passes ``post_physics_refresh=True``; a backend that overrides the method with the
        signature ``(self, state)`` would otherwise raise ``TypeError`` on the **first
        control tick**, with the robot already enabled. Checked here so it fails while the operator
        is still at the keyboard, and names the fix.
        """
        import inspect

        method = cls.state_to_observation
        if method is DeploymentBase.state_to_observation:
            return
        parameters = inspect.signature(method).parameters
        accepts = "post_physics_refresh" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
        if not accepts:
            raise TypeError(
                f"{cls.__name__}.state_to_observation overrides DeploymentBase.state_to_observation without the "
                f"keyword-only parameter `post_physics_refresh: bool = False` (or `**kwargs`), which "
                f"DeploymentBase.step passes on every control tick. Add it — and forward it to super() — so "
                f"stateful observation terms keep ticking on this backend."
            )

    def state_to_observation(self, state: RobotState, *, post_physics_refresh: bool = False) -> dict[str, Any]:
        """Sync ``state`` onto the env's robot entity and compute the policy's observation dict.

        ``post_physics_refresh`` is the deploy loop's phase 7. :meth:`step` passes ``True`` so an
        observation term with per-step state (:meth:`ObservationTerm.post_physics_refresh
        <eden.managers.observation_manager.ObservationTerm.post_physics_refresh>`) advances exactly once
        per control tick, on the freshly synced robot state and before the read — the sim step runs it
        at phase 7 and a deploy that skipped it would hand the policy a value frozen at its last
        simulated step, silently (the class of bug ``_advance_motion_reference`` documents). The reset
        paths leave it ``False``: ``EnvBase.reset`` runs no phase 7 either, so a term reads as zero
        there in both places. A command term's ``post_physics_refresh`` still does not run on this path
        (see :meth:`CommandTerm.post_physics_refresh
        <eden.managers.command_manager.CommandTerm.post_physics_refresh>`); the observation hook does
        because the policy's observations *are* the hardware path.

        The first read of an episode -- after construction, :meth:`reset` or START -- also clears the
        observation manager's per-step state and the action manager's last actions, in the order an
        env reset does: this tick's phase 7, then ``reset()``, then ``post_reset()`` (a term that
        rebuilds its state from the current frame, as ``ParkourProprio`` backfills its history, needs
        it), then the read, which backfills every history window from the robot. Without it the
        policy's first frames come from the training wrapper's ``env.reset()`` on the *simulated*
        robot at its spawn pose, or, at START, from the hold and INIT ramp in the yaw frame START just
        replaced. Clearing
        on the read rather than in each ``reset`` covers a backend override that does not chain to
        this class's, and a loop that drives :meth:`step` with no reset at all.

        **Overrides must accept the keyword** (and forward it to ``super()``); the constructor refuses
        one that does not (:meth:`_check_state_to_observation_override`).

        One tick of skew relative to the sim, inherent to this loop's order: :meth:`step` publishes the
        new command (``_update_commands``) *before* ``read_state``, so a term that integrates
        ``command − measured`` here pairs this tick's measurement with the command just published,
        where phase 7 in the sim pairs it with the command that was in effect during the physics.
        """
        self._sync_state(state)
        return self._observe(state, post_physics_refresh=post_physics_refresh)

    def _observe(self, state: RobotState, *, post_physics_refresh: bool) -> dict[str, Any]:
        """The second half of :meth:`state_to_observation`, on a ``state`` already synced: estimate, tick, read."""
        if self._lin_vel_estimator is not None:
            self._update_lin_vel_estimate(state)
        manager = self._env.observation_manager
        if post_physics_refresh:
            manager.post_physics_refresh()
        if self._observations_stale:
            self._observations_stale = False
            self._restart_observed_episode()
        return manager.compute(update_history=True)

    def restart_observations(self) -> dict[str, Any]:
        """Start every observation window afresh from the last reading, as START does; returns the observation.

        The deploy loop calls it when it switches policy slots, so the policy switched in reads what
        an env reset would have handed it, with every history window backfilled from the robot as it
        is now, a fed-back last action of zero, and the motion reference the switch just rewound.
        Without it the switched-in policy reads frames observed while another slot drove, with the
        reference still mid-clip, and its own last action from before it was switched out. The
        reading is not re-synced and the velocity estimator not re-run: the state is the one the last
        :meth:`step` already observed.
        """
        self._observations_stale = False
        self._restart_observed_episode()
        return self._env.observation_manager.compute(update_history=True)

    def _restart_observed_episode(self) -> None:
        """Clear what the policy observes of the episode so far, in an env reset's order (actions, then observations).

        Only the last actions are cleared on the action manager, not its terms: see
        :meth:`ActionManager.clear_action_history <eden.managers.action_manager.ActionManager.clear_action_history>`.
        """
        self._env.action_manager.clear_action_history()
        manager = self._env.observation_manager
        manager.reset()
        manager.post_reset()

    def _sync_state(self, state: RobotState) -> None:
        """Write ``state`` onto the :class:`RobotStateEntity`, and so onto the sim entity's FK.

        Backends extend it (calling ``super()``) to write whatever else they observe into the
        scene -- the Wuji hand's :meth:`sync_scene`. Runs once per read: :meth:`step` calls it
        before the commands publish on a backend that sets :attr:`sync_before_commands`, and
        through :meth:`state_to_observation` otherwise.
        """
        self._state_entity.update(state)

    def _update_lin_vel_estimate(self, state: RobotState) -> None:
        """Run the velocity KF on the freshly synced state and install the result.

        Runs *after* ``RobotStateEntity.update`` so the sim entity's FK
        reflects this step's joint positions and (yaw-aligned) orientation.
        The estimate is therefore already in the policy's yaw-aligned world
        frame and is installed via :meth:`RobotStateEntity.set_vel_estimate`
        (not ``extra["base_lin_vel"]``, which would be yaw-rotated again).
        Passing the wrapper's yaw offset as ``yaw_tag`` makes the filter
        re-initialize whenever the offset is re-captured (START handover).
        """
        if state.base_quat is None or state.base_ang_vel is None or state.base_lin_acc is None:
            if not self._lin_vel_warned:
                en.logger.warning("[deploy] estimate_lin_vel: state has no IMU fields; estimator skipped.")
                self._lin_vel_warned = True
            return
        vel = self._lin_vel_estimator.update(
            self._state_entity.get_quat()[0],
            state.base_ang_vel,
            state.base_lin_acc,
            state.dofs_vel,
            state.dofs_torque,
            yaw_tag=self._state_entity.yaw_offset,
        )
        # Published in ``extra`` so --debug_record captures them (as ``x_base_lin_vel_estimate``,
        # ``x_contacts``, ``x_forces``): the per-foot latch and the solved contact force are what
        # tell a contact dropout apart from IMU drift after the fact.
        state.extra["base_lin_vel_estimate"] = vel
        state.extra["contacts"] = self._lin_vel_estimator.contacts.astype(np.float64)
        state.extra["forces"] = self._lin_vel_estimator.forces.copy()
        self._state_entity.set_vel_estimate(vel)

    def _teardown_common(self) -> None:
        """Release everything ``DeploymentBase`` itself owns. Idempotent.

        Flushes the debug recording, then closes every substituted command term and puts
        the term it replaced back (see :meth:`_restore_swapped_commands`). Every backend's
        :meth:`close` must call this before dropping its transport — the buffers, the input
        device and the command manager are all independent of the wire, and a backend that
        forgets simply loses them (``--debug_record`` then writes nothing at all).

        The entity swap is deliberately *not* undone here: the deploy loop and anything
        mirroring the run still read ``env.entities[entity_name]`` for the last state the
        robot reported, which the wrapper keeps. A stopped input device has no such
        residual value.
        """
        self._dump_record()
        if self._rate is not None and self._rate.ticks:
            # The loop's own verdict on whether it held the control rate. An overrunning loop is
            # otherwise silent on hardware: the robot still gets targets, just late.
            log = en.logger.warning if self._rate.overruns else en.logger.info
            log(self._rate.summary())
        restored = self._restore_swapped_commands()
        if restored:
            en.logger.info(f"[deploy] Restored the env's own command term(s) {restored}.")

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_state(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def send_payload(self, payload: RobotCommand) -> None:
        raise NotImplementedError


class SoftShutdownMixin(ConfigurableMixin, ABC):
    """Settle the robot into a damping state before a backend's ``close`` drops its link.

    Dropping the link outright leaves the robot holding its last stiff PD target until an
    onboard comms watchdog trips, and a standing robot can lurch or collapse. :meth:`_soft_shutdown`
    instead sends ``shutdown_damping_steps`` ticks of kp=0, kd>0: the position servo is off, only
    velocity damping remains, and the joints settle gently.

    With a :class:`~eden.extensions.deployment.utils.republisher.CommandRepublisherMixin` running,
    no watchdog can trip while the process lives: a control thread that stalls leaves the loop
    re-sending its last command at full gains, and the loop reports that rather than acting on it
    (see "A stalled control thread is reported, not acted on" in ``docs/guides/deployment.md``).
    This ramp is then the only orderly way down, which is why such a backend's ``close`` runs it
    before ``_stop_publisher`` -- the damping command goes out at the same packet rate as any other.

    Mixed in ahead of :class:`DeploymentBase` by the backends that run the ramp, which call
    :meth:`_soft_shutdown` from ``close`` and implement :meth:`_shutdown_hold_pos`. The two options
    live here rather than on :class:`DeploymentBase` so a backend without the ramp keeps rejecting
    them instead of accepting and ignoring them.

    shutdown_damping_steps: int
        Control ticks to send the damping command before ``close`` releases the link.
    shutdown_damping_kd_scale: float
        Scale on kd during that soft shutdown.
    """

    shutdown_damping_steps: int = 50
    shutdown_damping_kd_scale: float = 1.0
    #: Names the ramp in its two log lines (``"[deploy] <label>: ramped ..."``).
    _SOFT_SHUTDOWN_LABEL: ClassVar[str] = "Soft shutdown"

    @abstractmethod
    def _shutdown_hold_pos(self) -> np.ndarray | None:
        """The pose the damping command carries, or ``None`` to skip the ramp (the link is down, or nothing to hold).

        Irrelevant under kp=0, but a sane target. Called inside :meth:`_soft_shutdown`'s
        best-effort guard, so a backend that reads it off the wire loses only the ramp when
        that read fails.
        """

    def _soft_shutdown(self) -> None:
        """Ramp the motors to a limp/damping state (kp=0, kd>0) before releasing the link. Never raises."""
        try:
            hold_pos = self._shutdown_hold_pos()
            if hold_pos is None:
                return
            payload = RobotCommand(
                dofs_pos=hold_pos,
                dofs_vel=np.zeros(self.num_dofs),
                dofs_torque=np.zeros(self.num_dofs),
                dofs_kp=np.zeros(self.num_dofs),
                dofs_kd=self.default_dof_kd * self.shutdown_damping_kd_scale,
            )
            for _ in range(self.shutdown_damping_steps):
                self.send_payload(payload)
                if self._rate is not None:
                    self._rate.sleep()
                else:
                    time.sleep(self.control_dt)
            en.logger.info(f"[deploy] {self._SOFT_SHUTDOWN_LABEL}: ramped to damping state.")
        except Exception as exc:  # noqa: BLE001 — best-effort on the way down
            en.logger.warning(f"[deploy] {self._SOFT_SHUTDOWN_LABEL} failed: {exc}")
