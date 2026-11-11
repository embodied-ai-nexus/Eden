"""Domain-randomization event terms (pushes, external wrenches, DOF randomization)."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.typing import Vec2FType

from eden.constants import EventMode, ReferenceSource
from eden.entities.rigid import resolve_link_ref_frame
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.options.entities import GroupedEntityOptions, PrimitiveOptions
from eden.options.managers.events import EventTermOptions
from genesis.utils.geom import inv_transform_by_quat
from eden.utils.misc import as_genesis_index, assign_over_envs, is_env_mask
from eden.utils.sample import sample_uniform
from eden.utils.string import key_selects, term_subject

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.managers.modifiers.actions.actuators import (
        ConstantTorqueKick,
        Deadband,
        GearBacklash,
        MotorStrength,
        TorqueOffset,
    )
    from eden.managers.modifiers.base import ActionModifier
    from eden.managers.terms.actions.joint_actions import (
        ExplicitPDController,
        VelocityFeedforwardPDController,
        _JointPDControllerBase,
    )


# Define proxies for fast lookup
_STARTUP, _RESET, _INTERVAL = EventMode


def _push_rows(envs_idx: slice | torch.Tensor, write_every_row: bool) -> slice | torch.Tensor:
    """The rows a base-velocity push reads, samples and writes for ``envs_idx``.

    Every env's row when ``write_every_row`` and ``envs_idx`` is a bool mask — the interval
    events' case, every step, mostly with no env due: Genesis then applies the mask itself, on the
    device (``RigidEntity.setters_take_every_row``), where selecting the rows would cost a
    GPU->CPU sync per read and per write. Otherwise ``envs_idx``'s rows.
    """
    if write_every_row and is_env_mask(envs_idx):
        return slice(None)
    return envs_idx


def _links_name_for(wanted: list[str], all_names: list[str]) -> list[str]:
    """The shortest ``links_name`` that selects exactly ``wanted`` out of ``all_names``, to paste into a config.

    Tries a ``*<common>*`` glob on the longest substring every wanted name shares, and a regex for "every link
    but the others", and falls back to the explicit list. Each candidate is checked against every name with
    :func:`~eden.utils.string.key_selects`, so only one that resolves to ``wanted`` is ever offered. It feeds
    an error message, so it never raises: anything unexpected degrades to the explicit list.
    """
    wanted_set = set(wanted)
    names = set(all_names)

    def selects_wanted(patterns: list[str]) -> bool:
        return {name for name in all_names if any(key_selects(p, name, names) for p in patterns)} == wanted_set

    try:
        others = [name for name in all_names if name not in wanted_set]
        candidates = [list(wanted)]
        if others:
            excluded = re.escape(others[0]) if len(others) == 1 else "(?:" + "|".join(map(re.escape, others)) + ")"
            candidates.append([f"(?!{excluded}$).*"])
        shortest = min(wanted, key=len)
        for size in range(len(shortest), 2, -1):  # longest first, and no shorter than 3 characters
            subs = {shortest[i : i + size] for i in range(len(shortest) - size + 1)}
            globs = [
                [f"*{sub}*"]
                for sub in sorted(subs)
                if re.fullmatch(r"[\w-]+", sub) and all(sub in name for name in wanted) and selects_wanted([f"*{sub}*"])
            ]
            if globs:
                candidates.append(globs[0])
                break
        return min((c for c in candidates if selects_wanted(c)), key=lambda c: len(repr(c)), default=list(wanted))
    except Exception:  # noqa: BLE001 - a suggestion must not replace the error it is part of
        return list(wanted)


def _reject_world_fixed_links(
    term: EventTerm, entity: RigidEntity, links_name: list[str], names: list[str], ls_idx_local: list[int], what: str
) -> None:
    """Raise when ``links_name`` matched a link fixed to the world, whose ``what`` the solver never reads.

    A link is fixed to the world when every joint from it to the root is fixed (``links[i].is_fixed``): a
    fixed-base arm's base link, a mounted hand's palm, a ``fixed=True`` prop. It never moves, so its mass
    and COM take no part in the dynamics: a randomized value would change nothing while the shift getter
    reported it, and a privileged observation would feed the critic noise.
    """
    fixed = [name for name, i in zip(names, ls_idx_local) if entity.links[i].is_fixed]
    if not fixed:
        return
    getter = "get_mass_shift()" if what == "mass" else "get_COM_shift()"
    all_names = [link.name for link in entity.links]
    moving = [name for name, i in zip(names, ls_idx_local) if not entity.links[i].is_fixed]
    head = (
        f"{term_subject(term)}: links_name={links_name!r} matched {fixed!r}, which are fixed to the "
        f"world, so the solver never reads their {what}: the randomized value would change no dynamics while "
        f"{getter} reported it."
    )
    if moving:
        fix = f"Fix: links_name={_links_name_for(moving, all_names)!r} selects exactly the moving links it matched."
    else:
        entity_moving = [link.name for link in entity.links if not link.is_fixed]
        fix = (
            f"Fix: name moving links instead; links_name={_links_name_for(entity_moving, all_names)!r} selects "
            f"all of them: {', '.join(entity_moving)}."
            if entity_moving
            else f"Fix: remove this term; no link of '{term.entity_name}' moves."
        )
    raise ValueError(
        f"{head}\n{fix}\nPassing the selection as a tensor, or catching this error, is not the fix: either leaves "
        f"a randomization that has no effect."
    )


def _refuse_anchored_inertials(term: EventTerm, quantity: str) -> None:
    """Raise at ``build()`` if Genesis would reject ``term``'s first ``quantity`` write on an aligned body.

    ``term`` must already hold ``entity`` and ``ls_idx_local``; ``quantity`` is ``"mass"`` or ``"COM"``.
    """
    blocked = term.entity.links_with_anchored_inertials(term.ls_idx_local, quantity)
    if not blocked:
        return
    what = {"mass": "mass", "COM": "centre of mass"}[quantity]
    name = term.entity_name
    options = term.entity._options
    if isinstance(options, PrimitiveOptions):
        remedy = (
            f"Fix: scene_options.{name} is a {type(options).__name__}, which Genesis always aligns (primitives have "
            f"no `align` field), so load the shape from a file instead, as `EntityOptions(file=..., align=False)`."
        )
    elif isinstance(options, GroupedEntityOptions):
        remedy = f"Fix: set `align=False` on each file-loaded variant in scene_options.{name}.grouped_entities."
    else:
        remedy = (
            f"Fix: load it unaligned, `align=False` on scene_options.{name} (`{type(options).__name__}(..., "
            f"align=False)`), which keeps the authored frame at the cost of the body's diagonal mass block."
        )
    # Only a body of several links reaches here for a mass write (Genesis' single-link exemption).
    why = " (a mass write is refused only because the body has more than one link)" if quantity == "mass" else ""
    raise ValueError(
        f"{type(term).__name__}(entity_name={name!r}) randomizes the {what} of link(s) {blocked}, but Genesis "
        f"anchored that body's frame on its centre of mass and principal axes at build (`align`, the default for a "
        f"single free rigid body) and does not support writing its {what} afterwards{why}: the write raises inside "
        f"Genesis on CPU and goes through unchecked on GPU, where that guard only runs in debug mode. {remedy} "
        f"Setting `batch_links_info=True` does not help: the refusal is about the frame, not the batching."
    )


def _require_modifier(term, modifier_cls: type, event_cls_name: str, action_term_name: str) -> "ActionModifier":
    """Look up ``modifier_cls`` on ``term._modifier`` and raise a clear error if missing."""
    root = getattr(term, "_modifier", None)
    if root is None:
        raise ValueError(
            f"{event_cls_name} requires action term '{action_term_name}' to have a "
            f"{modifier_cls.__name__} modifier, but the term has no modifier configured. "
            f"Add {modifier_cls.__name__} to its modifier chain."
        )
    modifier = root.get(modifier_cls)
    if modifier is None:
        raise ValueError(
            f"{event_cls_name} requires action term '{action_term_name}' to have a "
            f"{modifier_cls.__name__} modifier. "
            f"Add {modifier_cls.__name__} to its modifier chain."
        )
    return modifier


@EVENT_TERM_REGISTRY.register()
class PushByVelocity(EventTerm):
    """Periodically add a random velocity to a free base.

    ``velocity_range`` maps ``x`` / ``y`` / ``z`` (world-frame linear, m/s) and ``roll`` / ``pitch`` / ``yaw``
    (base-frame angular, rad/s) to ``(low, high)``; an axis left out is not perturbed.
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    entity_name: str = "robot"
    # None auto-detects the free base joint by joint TYPE (works for MJCF and URDF robots
    # alike); name it only to disambiguate a multi-free-joint entity.
    base_joint_name: str | None = None
    velocity_range: dict[str, Vec2FType] = {}

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.entity: Entity | None = None
        self.base_dofs_index: torch.Tensor | None = None

        range_list = [self.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=self._env.device)

    def build(self) -> None:
        self.entity = self.resolve_entity(self.entity_name)
        base_dofs_index = self.entity.find_free_base_dofs_idx_local(self.base_joint_name)
        self.base_dofs_index = torch.as_tensor(base_dofs_index, dtype=gs.tc_int, device=self.device).contiguous()
        self._genesis_base_dofs_idx = as_genesis_index(self.base_dofs_index)
        self._write_every_row = self.entity.setters_take_every_row and isinstance(self._genesis_base_dofs_idx, range)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        rows = _push_rows(envs_idx, self._write_every_row)
        lin_vel_w = self.entity.get_vel(envs_idx=rows)
        ang_vel_w = self.entity.get_ang(envs_idx=rows)
        quat_w = self.entity.get_quat(envs_idx=rows)

        rand_vel = sample_uniform(
            self.ranges[:, 0],
            self.ranges[:, 1],
            (self._env.num_envs, 6),
            device=self._env.device,
        )[rows]
        lin_vel_w += rand_vel[:, :3]
        # Genesis stores a free joint's rotational DOF velocity in the base BODY frame (MuJoCo
        # qvel convention). Convert the current world ang-vel to body, add the (body-frame)
        # perturbation, and write the BODY-frame result straight into DOFs 3-5 — do NOT rotate
        # back to world first. Linear DOFs are world-frame and left as-is.
        ang_vel_b = inv_transform_by_quat(ang_vel_w, quat_w)
        ang_vel_b += rand_vel[:, 3:]
        self.entity.set_dofs_vel(
            torch.cat([lin_vel_w, ang_vel_b], dim=1),
            dofs_idx_local=self._genesis_base_dofs_idx,
            envs_idx=envs_idx,
        )


@EVENT_TERM_REGISTRY.register()
class PushSetPlanarVelocity(EventTerm):
    """Periodically SET the base planar velocity to a random value.

    Unlike :class:`PushByVelocity` (which ADDS a perturbation), this overwrites vx/vy
    with ``U(velocity_xy_range)`` while keeping vz and the angular velocity unchanged —
    the legged_gym/extreme-parkour push semantics. Uses the same free-joint DOF-write
    convention: linear DOFs are world-frame, rotational DOF velocity is body-frame.
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    entity_name: str = "robot"
    # None auto-detects the free base joint by joint TYPE (works for MJCF and URDF robots
    # alike); name it only to disambiguate a multi-free-joint entity.
    base_joint_name: str | None = None
    velocity_xy_range: Vec2FType = (-0.5, 0.5)

    def build(self) -> None:
        entity: RigidEntity = self.resolve_entity(self.entity_name)
        base_dofs_idx = entity.find_free_base_dofs_idx_local(self.base_joint_name)
        self._base_dofs_idx = as_genesis_index(
            torch.as_tensor(base_dofs_idx, dtype=gs.tc_int, device=self.device).contiguous()
        )
        self._write_every_row = entity.setters_take_every_row and isinstance(self._base_dofs_idx, range)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        entity: RigidEntity = self.resolve_entity(self.entity_name)
        rows = _push_rows(envs_idx, self._write_every_row)
        lin_vel_w = entity.get_vel(envs_idx=rows)
        ang_vel_w = entity.get_ang(envs_idx=rows)
        quat_w = entity.get_quat(envs_idx=rows)

        rand_xy = sample_uniform(
            self.velocity_xy_range[0], self.velocity_xy_range[1], (self._env.num_envs, 2), device=self.device
        )[rows]
        lin_vel_w[:, :2] = rand_xy  # SET, not add
        # Rotational free-joint DOF velocity is stored in the base BODY frame (MuJoCo qvel
        # convention) — convert before writing back unchanged.
        ang_vel_b = inv_transform_by_quat(ang_vel_w, quat_w)
        entity.set_dofs_vel(
            torch.cat([lin_vel_w, ang_vel_b], dim=1),
            dofs_idx_local=self._base_dofs_idx,
            envs_idx=envs_idx,
        )


class _ApplyExternalWrenchBase(EventTerm):
    """Shared lifecycle for the randomized external-wrench event terms (force / torque).

    Resolves ``links_name`` to link indices at build time, then on each interval
    fire samples a per-env, per-link 3-axis wrench from the subclass's axis ranges
    (:meth:`_wrench_ranges`) and applies it (:meth:`_apply_wrench`). With ``prob <
    1.0`` a per-env Bernoulli mask zeros the wrench on envs that fail the draw.
    Not registered — instantiate :class:`ApplyExternalForce` or
    :class:`ApplyExternalTorque`.

    Parameters
    ----------
    entity_name: str
        The name of the entity to apply the wrench to.
    links_name: list[str]
        The names of the links to apply the wrench to.
    ref: str
        Reference frame for the wrench. One of ``"link_origin"`` or ``"link_com"``.
    local: bool
        If true, the wrench is interpreted in the link's local frame; otherwise in the world frame.
    prob: float
        Per-env probability of applying the wrench at each interval fire. ``1.0`` (default) applies it to every
        selected env. ``< 1.0`` independently samples a Bernoulli(prob) mask per env per fire and zeros the
        wrench on envs that fail the draw.

    Notes
    -----
    Each term instance samples its own mask. Pairing ``ApplyExternalForce`` with ``ApplyExternalTorque`` at the config
    layer therefore gates them **independently** — the same env may receive force in one fire and not torque (or vice
    versa). This diverges from DexMachina's ``_random_force_torque`` (``dexmachina/envs/randomizations.py``),
    which shares one mask across both signals so the same envs are zeroed for force AND torque atomically. Both terms
    draw from the global ``torch`` RNG, so seeding alone cannot synchronize them — anything that consumes RNG state
    between the two ``compute()`` calls (including the in-call ``sample_uniform`` draws) shifts the mask draws out of
    phase. If synchronized gating is required for
    parity, the only correct fix is a single event term that samples one mask and applies both wrenches in one call;
    no such term exists yet.
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    entity_name: str = "robot"
    links_name: list[str] = []
    ref: str = "link_origin"  # "link_origin" | "link_com"
    local: bool = False
    prob: float = 1.0

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.links_idx: torch.Tensor | None = None
        self.entity: RigidEntity | None = None
        self._ref_frame: gs.link_ref_frame | None = None

    def _wrench_ranges(self) -> tuple[Vec2FType, Vec2FType, Vec2FType]:
        """Return the per-axis (x, y, z) sampling ranges for this wrench."""
        raise NotImplementedError

    def _apply_wrench(self, wrench: torch.Tensor, envs_idx: slice | torch.Tensor) -> None:
        """Apply the sampled wrench to ``links_idx`` via the matching rigid-solver call."""
        raise NotImplementedError

    def build(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"{term_subject(self)}: prob must be in [0.0, 1.0], got {self.prob}.")
        self._ref_frame = resolve_link_ref_frame(self.ref)
        self.entity = self.resolve_entity(self.entity_name)
        resolved_links_name, _ = self.entity.find_named_links_idx_local(self.links_name)
        assert len(resolved_links_name) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.links_idx = torch.tensor(
            [self.entity._entity.get_link(name).idx for name in resolved_links_name],
            dtype=gs.tc_int,
            device=self.device,
        ).contiguous()

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        n_links = len(self.links_idx)
        x_range, y_range, z_range = self._wrench_ranges()
        wrench = torch.stack(
            [
                sample_uniform(axis[0], axis[1], (self._env.num_envs, n_links), device=self._env.device)
                for axis in (x_range, y_range, z_range)
            ],
            dim=-1,
        )[envs_idx]
        if self.prob < 1.0:
            mask = torch.rand((self._env.num_envs,), device=self._env.device) < self.prob
            wrench[~mask[envs_idx]] = 0.0
        self._apply_wrench(wrench, envs_idx)


@EVENT_TERM_REGISTRY.register()
class ApplyExternalForce(_ApplyExternalWrenchBase):
    """
    Apply a randomized external force on specified links.

    ``force_x_range`` / ``force_y_range`` / ``force_z_range`` are the per-axis force ranges (N), sampled uniformly
    per env and link. See :class:`_ApplyExternalWrenchBase` for the shared parameters and the cross-term
    independence caveat.
    """

    force_x_range: Vec2FType = (0.0, 0.0)
    force_y_range: Vec2FType = (0.0, 0.0)
    force_z_range: Vec2FType = (0.0, 0.0)

    def _wrench_ranges(self) -> tuple[Vec2FType, Vec2FType, Vec2FType]:
        return self.force_x_range, self.force_y_range, self.force_z_range

    def _apply_wrench(self, wrench: torch.Tensor, envs_idx: slice | torch.Tensor) -> None:
        self._env.rigid_solver.apply_links_external_wrench(
            force=wrench,
            links_idx=self.links_idx,
            envs_idx=envs_idx,
            ref=self._ref_frame,
            local=self.local,
        )


@EVENT_TERM_REGISTRY.register()
class ApplyExternalTorque(_ApplyExternalWrenchBase):
    """
    Apply a randomized external torque on specified links.

    ``torque_x_range`` / ``torque_y_range`` / ``torque_z_range`` are the per-axis torque ranges (N·m), sampled
    uniformly per env and link. See :class:`_ApplyExternalWrenchBase` for the shared parameters and the cross-term
    independence caveat.
    """

    torque_x_range: Vec2FType = (0.0, 0.0)
    torque_y_range: Vec2FType = (0.0, 0.0)
    torque_z_range: Vec2FType = (0.0, 0.0)

    def _wrench_ranges(self) -> tuple[Vec2FType, Vec2FType, Vec2FType]:
        return self.torque_x_range, self.torque_y_range, self.torque_z_range

    def _apply_wrench(self, wrench: torch.Tensor, envs_idx: slice | torch.Tensor) -> None:
        self._env.rigid_solver.apply_links_external_wrench(
            torque=wrench,
            links_idx=self.links_idx,
            envs_idx=envs_idx,
            ref=self._ref_frame,
            local=self.local,
        )


class _RandomizeDofsPosBase(EventTerm):
    """Shared lifecycle for event terms that write randomized absolute joint positions via ``set_dofs_pos``.

    Samples a per-env, per-DOF position from ``dofs_pos_range`` and writes it with
    ``set_dofs_pos``. The two concrete terms (:class:`SetRandomDofsPos`,
    :class:`RandomizeStartupDofsPos`) are identical in mechanics and differ only
    in their field defaults; they subclass this base and override those defaults.
    Not registered — instantiate a concrete subclass.

    Parameters
    ----------
    entity_name: str
        The name of the entity to randomize.
    dofs_pos_range: Vec2FType
        When ``apply_as_ratio`` is false, absolute joint positions (radians) sampled uniformly per env and DOF.
        When ``apply_as_ratio`` is true, fractions of each DOF's limit range — e.g. ``(0.0, 0.5)`` samples
        positions in ``[lower, lower + 0.5 * (upper - lower)]`` per DOF; ``(0.0, 1.0)`` covers the full range.
        DOFs without finite limits fall back to a ``2 * pi`` range centered at 0.
    apply_as_ratio: bool
        If false, ``dofs_pos_range`` sets absolute positions in radians.
        If true, ``dofs_pos_range`` is interpreted as a fraction of each DOF's configured limit range.
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    # Neutral defaults — each concrete subclass declares its own dofs_pos_range / apply_as_ratio.
    dofs_pos_range: Vec2FType = (0.0, 0.0)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity

    def build(self) -> None:
        self.entity = self.resolve_entity(self.entity_name)
        if self.apply_as_ratio:
            # Per-DOF lower limit and (upper - lower); unbounded joints fall back to a 2*pi range centered at 0
            # so the ratio still produces a sensible position. Both carry the env axis.
            lower, upper = self.entity.get_dofs_limit()
            lower = lower.to(device=self._env.device)
            upper = upper.to(device=self._env.device)
            dof_range = upper - lower
            finite = torch.isfinite(dof_range)
            self._dof_range = torch.where(finite, dof_range, torch.full_like(dof_range, 2.0 * torch.pi))
            self._dof_lower = torch.where(finite, lower, torch.full_like(lower, -torch.pi))

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        sampled = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (self._env.num_envs, self.entity.num_dofs),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            # Slice per-env cached limits to match the subset; (1, n_dofs) broadcasts directly.
            dof_lower = self._dof_lower if self._dof_lower.shape[0] == 1 else self._dof_lower[envs_idx]
            dof_range = self._dof_range if self._dof_range.shape[0] == 1 else self._dof_range[envs_idx]
            dofs_pos = dof_lower + sampled * dof_range
        else:
            dofs_pos = sampled
        self.entity.set_dofs_pos(dofs_pos, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class SetRandomDofsPos(_RandomizeDofsPosBase):
    """
    Set the joint positions of an entity's DOFs to a random position (default mode: reset).

    Defaults to ratio mode, sampling within the first 20% of each DOF's limit range. See
    :class:`_RandomizeDofsPosBase` for the full parameter description.
    """

    dofs_pos_range: Vec2FType = (0.0, 0.2)  # sample within the first 20% of the dofs limits
    apply_as_ratio: bool = True


@EVENT_TERM_REGISTRY.register()
class RandomizeDofsPosOffset(EventTerm):
    """
    Per-env joint-position offset on joint position controllers (encoder bias).

    Each selected controller draws its own offsets, one per env and controlled DOF, and takes them
    through :meth:`~eden.managers.terms.actions.joint_actions._JointPDControllerBase.set_dofs_pos_offset`.
    The draw adds to the controller's position target on top of its configured ``offset`` and its
    ``reference_source`` reference, and replaces neither, so it composes with a static offset and,
    in either order, with :class:`RandomizeStartupDofsPosBias`.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str | list[str] | None
        The joint position controller(s) to randomize: one term name, a list of
        them, or ``None`` (the default) for every joint position controller on ``entity_name`` — see
        :func:`~eden.managers.terms.actions.joint_actions.is_joint_position_controller`. ``None``
        keeps a config that splits the entity's joints across several controllers (a per-limb
        scale or delay) from having to list, and so depend on, each term's name. A named term must
        be a joint position controller on ``entity_name``, and ``None`` raises if the entity has none.
    dofs_pos_range: Vec2FType
        Per-DOF offset range (radians) sampled uniformly per env and DOF.
    """

    mode: EventMode = _RESET
    entity_name: str = ""
    action_term_name: str | list[str] | None = None
    dofs_pos_range: Vec2FType = (
        -0.01745,
        0.01745,
    )  # radians, about +/- 1 degree

    def build(self) -> None:
        # Touched only to fail fast on a bad name; compute() writes to the controllers, not the entity.
        self.resolve_entity(self.entity_name)
        self._targets: list[_JointPDControllerBase] = self._resolve_terms()

    def _resolve_terms(self) -> list[_JointPDControllerBase]:
        """The controllers ``action_term_name`` selects, validated against ``entity_name``."""
        # Local import: eden.managers.terms.actions imports the managers this module is loaded under.
        from eden.managers.terms.actions.joint_actions import is_joint_position_controller

        manager = self._env.action_manager
        if self.action_term_name is None:
            terms = [
                term
                for term in manager.terms.values()
                if is_joint_position_controller(term) and term.entity_name == self.entity_name
            ]
            if not terms:
                available = {name: type(term).__name__ for name, term in manager.terms.items()}
                raise ValueError(
                    f"{term_subject(self)}: action_term_name=None selects every joint position controller on "
                    f"the entity, "
                    f"and the action manager has none. Its terms: {available or 'none'}."
                )
            return terms

        names = [self.action_term_name] if isinstance(self.action_term_name, str) else list(self.action_term_name)
        if not names:
            raise ValueError(
                f"{term_subject(self)}: action_term_name is an empty list. Name at least one controller, or pass None "
                f"for every joint position controller on the entity."
            )
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"{term_subject(self)}: action_term_name lists {duplicates} more than once.")
        terms = []
        for name in names:
            term = manager.get_term(name)  # raises KeyError listing the active terms
            if not is_joint_position_controller(term):
                raise ValueError(
                    f"{term_subject(self)}: action term '{name}' is a {type(term).__name__}, not a joint position "
                    f"controller, so it has no joint-position offset to randomize."
                )
            if term.entity_name != self.entity_name:
                raise ValueError(
                    f"{term_subject(self)}: action term '{name}' controls entity '{term.entity_name}', "
                    f"not '{self.entity_name}'."
                )
            terms.append(term)
        return terms

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        for term in self._targets:
            shape = (self._env.num_envs, len(term.dofs_name))
            offset = sample_uniform(*self.dofs_pos_range, shape, device=self._env.device)[envs_idx]
            term.set_dofs_pos_offset(offset, envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeStartupDofsPos(_RandomizeDofsPosBase):
    """
    Randomize the absolute joint positions of an entity's DOFs (default mode: reset).

    Writes joint positions directly via ``set_dofs_pos`` — useful for spawning each episode with a randomized
    initial pose. Despite the name, defaults to ``mode=_RESET`` so it fires on every episode reset; override to
    ``_STARTUP`` for a one-time draw. Defaults to absolute mode with a small ±1° range. See
    :class:`_RandomizeDofsPosBase` for the full parameter description.
    """

    dofs_pos_range: Vec2FType = (-0.01745, 0.01745)  # radians, about +/- 1 degree
    apply_as_ratio: bool = False


@EVENT_TERM_REGISTRY.register()
class RandomizeStartupDofsPosBias(EventTerm):
    """
    Bake a per-env joint-zero bias into ``entity.default_dofs_pos`` at startup.

    Simulates motor calibration drift: each env gets a small fixed offset on every DOF of the entity, sampled once at
    startup and held constant for the lifetime of the env. The bias persists across resets, unlike the per-reset
    jitter from :class:`RandomizeDofsPosOffset`. The two compose: a calibration drift baked in at startup + small
    per-episode jitter on top.

    When ``action_terms_name`` lists one or more PD controllers, each controller's ``_offset`` cache (folded once at
    build time as ``default_offset + configured_offset``, see ``_JointPDControllerBase._build_scale_and_offset``) is
    updated in lockstep by adding the same per-env bias delta on the DOFs that controller owns. This avoids the
    "stale cache" trap where mutating ``entity.default_dofs_pos`` would diverge from a controller's pre-folded static
    offset.

    **Build-time contract.** ``build()`` performs strict validation so that any stale-cache footgun surfaces loudly at
    startup rather than as silent drift at runtime:

    1. ``action_terms_name`` must contain no duplicates.
    2. Each listed term must be a :class:`_JointPDControllerBase` instance, target the same entity, and have
       ``reference_source == ReferenceSource.DEFAULT``.
    3. Every same-entity action term that caches ``default_dofs_pos`` at build must be either listed (and updated in
       lockstep) or rejected with a clear error. Specifically:

       - ``_JointPDControllerBase`` with ``reference_source=DEFAULT`` not in the listed set: error (its ``_offset``
         would silently go stale).
       - :class:`NullJointAction` caches a copy of ``default_dofs_pos`` at build with no refresh path; if it is present
         on the same entity, ``build()`` errors out. ``OperationalSpaceController`` refreshes its declared nullspace
         reference at reset and therefore does not need this rejection.

    Parameters
    ----------
    entity_name: str
        The entity to bias.
    action_terms_name: list[str]
        Names of PD controller action terms whose ``_offset`` caches should be kept in sync with the bias. Each must
        be a :class:`_JointPDControllerBase` on ``entity_name`` with ``reference_source=DEFAULT``. Must list every
        such controller on the entity — see the build-time contract above. Empty list is fine if and only if no
        DEFAULT-source PD controllers exist on this entity.
    dofs_pos_range: Vec2FType
        Per-DOF bias range, sampled uniformly. Default ±1 degree.
    """

    mode: EventMode = _STARTUP
    entity_name: str = "robot"
    action_terms_name: list[str] = []
    dofs_pos_range: Vec2FType = (-0.01745, 0.01745)  # radians, about ±1 degree

    if TYPE_CHECKING:
        entity: RigidEntity

    def build(self) -> None:
        # Strict three-phase validation so any stale-cache footgun surfaces loudly at
        # startup rather than as silent drift at runtime (see the class docstring).
        self.entity = self.resolve_entity(self.entity_name)
        self._reject_duplicate_action_terms()
        self._targets = self._resolve_listed_controllers()
        self._reject_unhandled_cache_holders()
        self._applied = False

    def _reject_duplicate_action_terms(self) -> None:
        """(1) Reject duplicates — they would double-apply the bias to one controller's
        ``_offset`` while only single-applying it to the entity default.
        """
        duplicates = {name for name in self.action_terms_name if self.action_terms_name.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"{term_subject(self)}: action_terms_name has duplicate entries {sorted(duplicates)}; "
                f"each PD controller must appear at most once or its `_offset` would be biased twice "
                f"while `default_dofs_pos` is biased only once (recreating the double-counting bug "
                f"this term is designed to avoid)."
            )

    def _resolve_listed_controllers(self) -> list[tuple[_JointPDControllerBase, torch.Tensor]]:
        """(2) Resolve each listed controller, validating type / entity / reference_source.

        Failures here surface as a clear ValueError rather than as a downstream
        AttributeError on a non-PD term's missing ``reference_source`` / ``_offset``.
        """
        # Function-local import avoids a load-time cycle through eden.managers.terms
        # (events is imported alongside actions; the runtime isinstance check is safe
        # because action_manager has fully built by the time event_manager builds).
        from eden.managers.terms.actions.joint_actions import _JointPDControllerBase

        targets: list[tuple[_JointPDControllerBase, torch.Tensor]] = []
        for name in self.action_terms_name:
            term = self._env.action_manager.get_term(name)
            if not isinstance(term, _JointPDControllerBase):
                raise ValueError(
                    f"{term_subject(self)}: action_terms_name lists '{name}' which is a "
                    f"{type(term).__name__}, not a `_JointPDControllerBase`. Lockstep "
                    f"`_offset` updates only make sense for PD controllers that fold "
                    f"`default_dofs_pos` into a tensor `_offset` cache."
                )
            if term.entity_name != self.entity_name:
                raise ValueError(
                    f"{term_subject(self)}: action_terms_name lists '{name}' which targets "
                    f"entity '{term.entity_name}', but this term is biasing entity "
                    f"'{self.entity_name}'. Lockstep `_offset` updates only make sense for "
                    f"controllers on the same entity whose `default_dofs_pos` is being mutated."
                )
            if term.reference_source != ReferenceSource.DEFAULT:
                raise ValueError(
                    f"{term_subject(self)}: action_terms_name lists '{name}' but its "
                    f"reference_source={term.reference_source!r} != ReferenceSource.DEFAULT. "
                    f"Only default-source PD controllers fold `default_dofs_pos` into `_offset`; "
                    f"updating non-default controllers' `_offset` here would be incorrect."
                )
            # The DOFs `_offset` spans, which lead `dofs_name`: under `mimic_dofs` the drivers alone, since the
            # followers trail them and take their target from their driver's.
            columns = self.entity.find_named_dofs(term.dofs_name[: term._offset.shape[-1]]).columns
            # Lazy-expand the broadcast (1, n_dofs) offset to per-env (num_envs, n_dofs) so the bias
            # write below has the right shape. This term is the only one that writes `_offset`:
            # RandomizeDofsPosOffset hands its draw to the controller instead, which keeps
            # it in a per-env buffer of its own and adds it on top of whatever is folded in here.
            if term._offset.shape[0] != self._env.num_envs:
                term._offset = term._offset.repeat(self._env.num_envs, 1)
            targets.append((term, columns))
        return targets

    def _reject_unhandled_cache_holders(self) -> None:
        """(3) Scan all same-entity action terms for unhandled ``default_dofs_pos`` cache
        holders. Any DEFAULT-source PD controller not listed, or any action term that
        snapshots defaults at build with no refresh API (NullJointAction), would silently
        drift after the bias mutates the entity defaults. Fail loud rather than ship a
        half-coupled config.
        """
        from eden.managers.terms.actions.joint_actions import (
            NullJointAction,
            _JointPDControllerBase,
        )

        listed_set = set(self.action_terms_name)
        for action_name, action_term in self._env.action_manager._terms.items():
            if getattr(action_term, "entity_name", None) != self.entity_name:
                continue
            if action_name in listed_set:
                continue
            if (
                isinstance(action_term, _JointPDControllerBase)
                and action_term.reference_source == ReferenceSource.DEFAULT
            ):
                raise ValueError(
                    f"{term_subject(self)}: PD controller '{action_name}' has reference_source=DEFAULT (folds "
                    f"`default_dofs_pos` into `_offset` at build), but is not listed in "
                    f"`action_terms_name`. Add it to keep its cache in sync, or change "
                    f"its reference_source so it doesn't snapshot the pre-bias defaults."
                )
            if isinstance(action_term, NullJointAction):
                raise ValueError(
                    f"{term_subject(self)}: action term '{action_name}' "
                    f"({type(action_term).__name__}) "
                    f"caches `default_dofs_pos` at build time and provides no refresh path. "
                    f"Mutating the entity default would leave its cache stale. Either remove "
                    f"the action term or stop biasing this entity."
                )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Idempotence: startup terms are invoked once today, but make the contract explicit.
        if self._applied:
            return
        self._applied = True

        if envs_idx is None:
            envs_idx = slice(None)

        if self.entity.default_dofs_pos.shape[0] == 1 and self._env.num_envs != 1:
            self.entity.default_dofs_pos = self.entity.default_dofs_pos.repeat(self._env.num_envs, 1)
        elif self.entity.default_dofs_pos.shape[0] != self._env.num_envs:
            raise ValueError(
                f"{term_subject(self)} expected default_dofs_pos batch size 1 or {self._env.num_envs}, "
                f"got {self.entity.default_dofs_pos.shape[0]}"
            )

        bias_full = sample_uniform(
            self.dofs_pos_range[0],
            self.dofs_pos_range[1],
            (self._env.num_envs, self.entity.default_dofs_pos.shape[1]),
            device=self._env.device,
        )[envs_idx]

        # 1. Bake the bias into the entity's default DOF positions (additive).
        self.entity.default_dofs_pos[envs_idx] = self.entity.default_dofs_pos[envs_idx] + bias_full

        # 2. Keep each listed PD controller's pre-folded `_offset` cache in lockstep,
        #    on the subset of DOFs that controller owns.
        for term, columns in self._targets:
            term._offset[envs_idx] = term._offset[envs_idx] + bias_full[:, columns]


class _ResetModeActuatorDR(EventTerm):
    """Base for reset-mode DR terms that randomize an action modifier's per-env buffer.

    The randomization write is performed in :meth:`reset`, not :meth:`compute`, because of
    the reset ordering in :meth:`EnvBase._reset_idx` (``EnvBase._manager_order``):

    1. ``event_manager.compute(mode=reset)`` — the reset-mode event phase (calls :meth:`compute`).
    2. ``action_manager.reset()`` — restores each action modifier's per-env buffer to its
       configured baseline row (e.g. :meth:`MotorStrength.reset`).
    3. ``event_manager.reset()`` — calls each term's :meth:`reset`.

    Step 2 before step 3 is the one hard constraint on that order, and
    ``EnvBase._validate_manager_order`` pins it at build.

    A write in :meth:`compute` (step 1) would be overwritten by the baseline restore in
    step 2 within the same reset, leaving the randomization silently inert. Writing in
    :meth:`reset` (step 3) lands after the restore and survives — the same reason
    :class:`ERFI50` performs its per-reset resampling in ``reset()`` (ERFI50 additionally
    keeps a working interval-mode ``compute()``; these terms do not).

    Subclasses implement :meth:`_randomize`; :meth:`compute` is a deliberate no-op.
    Consequently only ``mode="reset"`` with ``min_step_count_between_reset=0`` is
    supported — anything else would be silently inert, so ``__init__`` rejects it.

    NOTE for reset-time readers: every manager that resets before ``event_manager`` in
    ``_manager_order`` (action, command, metric, termination, reward, curriculum) observes
    the PREVIOUS episode's randomization in the modifier buffers — the right frame for
    publishing that episode's statistics; the fresh draw lands only at step 3.
    """

    mode: EventMode = _RESET

    def __init__(self, env, options) -> None:
        super().__init__(env=env, options=options)
        mode = getattr(options, "mode", _RESET)
        throttle = getattr(options, "min_step_count_between_reset", 0)
        if mode != _RESET or throttle != 0:
            raise ValueError(
                f"{type(self).__name__} randomizes in reset() and supports only mode='reset' with "
                f"min_step_count_between_reset=0 (got mode={mode!r}, min_step_count_between_reset="
                f"{throttle}). Its compute() is a no-op, so other modes/throttles would be "
                "silently inert."
            )

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._randomize(envs_idx)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # No-op: reset-mode DR randomizes in reset() so it survives action_manager.reset();
        # a write here (the reset-mode event phase) would be clobbered. See class docstring.
        del envs_idx

    def _randomize(self, envs_idx: slice | torch.Tensor) -> None:
        raise NotImplementedError


@EVENT_TERM_REGISTRY.register()
class RandomizeMotorStrength(_ResetModeActuatorDR):
    """
    Randomize a :class:`MotorStrength` action modifier's scalar (default factor 0.9 to 1.1).

    Requires
    --------
    The action term must have a ``MotorStrength`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``MotorStrength`` modifier.
    motor_strength_range: Vec2FType
        Multiplicative range applied to the modifier's strength, sampled uniformly per env (one scalar per env).
    """

    entity_name: str = "robot"
    action_term_name: str = ""
    motor_strength_range: Vec2FType = (0.9, 1.1)

    if TYPE_CHECKING:
        entity: RigidEntity
        term: ExplicitPDController | VelocityFeedforwardPDController
        modifier: MotorStrength

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import MotorStrength

        self.entity = self.resolve_entity(self.entity_name)
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, MotorStrength, type(self).__name__, self.action_term_name)

    def _randomize(self, envs_idx: slice | torch.Tensor) -> None:
        motor_strength = sample_uniform(
            self.motor_strength_range[0],
            self.motor_strength_range[1],
            (self._env.num_envs, 1),
            device=self._env.device,
        )[envs_idx]
        self.modifier._motor_strength[envs_idx] = motor_strength


@EVENT_TERM_REGISTRY.register()
class RandomizeGearBacklash(_ResetModeActuatorDR):
    """
    Randomize the one-sided target offset of a :class:`GearBacklash` action modifier.

    Requires
    --------
    The action term must have a ``GearBacklash`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``GearBacklash`` modifier.
    backlash_range: Vec2FType
        When ``apply_as_ratio`` is false, the range of one-sided backlash offsets (rad) sampled uniformly per env and
        DOF. When ``apply_as_ratio`` is true, the range of multipliers applied to the modifier's configured per-DOF
        baseline (see ``apply_as_ratio``).
    apply_as_ratio: bool
        If false (default), ``backlash_range`` sets absolute backlash offsets in radians. If true, ``backlash_range``
        is interpreted as ``(low, high)`` multipliers (uniform per env and DOF): the backlash becomes
        ``m * backlash_i``, where ``backlash_i`` is that DOF's configured nominal backlash from
        :class:`~eden.managers.modifiers.actions.actuators.GearBacklash` at build time.
    """

    entity_name: str = "robot"
    action_term_name: str = ""
    backlash_range: Vec2FType = (0.0, 0.01)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase
        modifier: GearBacklash

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import GearBacklash

        self.entity = self.resolve_entity(self.entity_name)
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, GearBacklash, type(self).__name__, self.action_term_name)

    def _randomize(self, envs_idx: slice | torch.Tensor) -> None:
        n_ctrl = len(self.term.dofs_name)
        sampled = sample_uniform(
            self.backlash_range[0],
            self.backlash_range[1],
            (self._env.num_envs, n_ctrl),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            baseline = self.modifier._backlash_row.to(device=sampled.device, dtype=sampled.dtype)
            backlash = sampled * baseline.unsqueeze(0)
        else:
            backlash = sampled
        self.modifier._backlash[envs_idx] = backlash.clamp(min=0.0)


@EVENT_TERM_REGISTRY.register()
class RandomizeConstantTorqueKick(_ResetModeActuatorDR):
    """
    Randomize the torque magnitude of a :class:`ConstantTorqueKick` action modifier.

    Requires
    --------
    The action term must have a ``ConstantTorqueKick`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``ConstantTorqueKick`` modifier.
    torque_kick_range: Vec2FType
        When ``apply_as_ratio`` is false, the range of torque-kick magnitudes (N·m) sampled uniformly per env and
        DOF. When ``apply_as_ratio`` is true, the range of multipliers applied to the modifier's configured per-DOF
        baseline (see ``apply_as_ratio``).
    apply_as_ratio: bool
        If false (default), ``torque_kick_range`` sets absolute torque-kick magnitudes. If true, ``torque_kick_range``
        is interpreted as ``(low, high)`` multipliers (uniform per env and DOF): the kick becomes
        ``m * torque_kick_i``, where ``torque_kick_i`` is that DOF's configured nominal magnitude from
        :class:`~eden.managers.modifiers.actions.actuators.ConstantTorqueKick` at build time.
    """

    entity_name: str = "robot"
    action_term_name: str = ""
    torque_kick_range: Vec2FType = (0.0, 0.01)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase
        modifier: ConstantTorqueKick

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import ConstantTorqueKick

        self.entity = self.resolve_entity(self.entity_name)
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, ConstantTorqueKick, type(self).__name__, self.action_term_name)

    def _randomize(self, envs_idx: slice | torch.Tensor) -> None:
        n_ctrl = len(self.term.dofs_name)
        sampled = sample_uniform(
            self.torque_kick_range[0],
            self.torque_kick_range[1],
            (self._env.num_envs, n_ctrl),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            baseline = self.modifier._torque_kick_row.to(device=sampled.device, dtype=sampled.dtype)
            torque_kick = sampled * baseline.unsqueeze(0)
        else:
            torque_kick = sampled
        self.modifier._torque_kick[envs_idx] = torque_kick.clamp(min=0.0)


@EVENT_TERM_REGISTRY.register()
class RandomizeKpKdGains(EventTerm):
    """
    Scale an entity's joint PD gains by a random factor.

    With ``action_term_name`` set, the draw goes into that controller's own gain caches, over the controller's DOFs
    in its ``dofs_name`` order: each cache becomes the entity's *current* gains times the drawn factor. Writing them
    only in :meth:`compute` is not enough: the controller's ``reset()`` re-reads its caches from the entity, and in
    both ``reset`` and ``startup`` mode that re-read lands after the draw (the ordering :class:`_ResetModeActuatorDR`
    describes), leaving the randomization silently inert. So the term keeps each env's *factors* and applies them
    again in :meth:`reset`, which runs after ``action_manager.reset()``; a draw in any mode then holds until that
    env's next draw. Keeping factors rather than gains is what lets it compose with anything else that changes the
    entity's gains (an entity-path ``RandomizeKpKdGains``, a gain curriculum): each reset scales whatever the entity
    holds then, and an env not yet drawn keeps the entity's gains unchanged (factor 1).

    Requires
    --------
    ``env_options.batch_dofs_info=True`` to support per-env PD gain randomization.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str | None
        If set, the name of an ``ExplicitPDController`` or ``VelocityFeedforwardPDController`` on ``entity_name``,
        whose ``_kp`` / ``_kd`` caches receive the gains; the entity is left untouched. If ``None``, the gains are
        written directly to the entity via ``set_dofs_kp`` / ``set_dofs_kd`` (the path Genesis' built-in PD,
        ``ImplicitPDController``, reads).
    kp_range: Vec2FType
        Multiplicative range, sampled uniformly per env and DOF. On the entity path it multiplies
        ``entity.default_dofs_kp``; on the controller path (``action_term_name`` set) it multiplies the entity's
        current kp, which is ``default_dofs_kp`` unless something else changed it.
    kd_range: Vec2FType
        As ``kp_range``, for kd (``entity.default_dofs_kd`` on the entity path).
    """

    _requires_env_options = ("batch_dofs_info",)

    mode: EventMode = _RESET
    entity_name: str = "robot"
    action_term_name: str | None = None
    kp_range: Vec2FType = (0.9, 1.1)
    kd_range: Vec2FType = (0.9, 1.1)

    if TYPE_CHECKING:
        entity: RigidEntity
        term: ExplicitPDController | VelocityFeedforwardPDController | None

    def build(self) -> None:
        self.entity = self.resolve_entity(self.entity_name)
        self.term = None
        if self.action_term_name is None:
            return

        from eden.managers.terms.actions.joint_actions import ExplicitPDController, VelocityFeedforwardPDController

        term = self._env.action_manager.get_term(self.action_term_name)
        if not isinstance(term, (ExplicitPDController, VelocityFeedforwardPDController)):
            raise ValueError(
                f"{term_subject(self)} (action_term_name={self.action_term_name!r}) writes the joint gain caches of an "
                f"ExplicitPDController or VelocityFeedforwardPDController, but action term "
                f"{self.action_term_name!r} is of type {type(term).__name__}, which has none. For Genesis' built-in PD "
                "(ImplicitPDController) set action_term_name=None: the gains then go to the entity "
                "(set_dofs_kp / set_dofs_kd), which is what that controller reads."
            )
        if term.entity_name != self.entity_name:
            raise ValueError(
                f"{term_subject(self)} (action_term_name={self.action_term_name!r}): "
                f"action term {self.action_term_name!r} controls entity {term.entity_name!r}, so its gains would be "
                f"scaled from the wrong entity's default_dofs_kp / default_dofs_kd. Set "
                f"entity_name={term.entity_name!r}."
            )
        self.term = term
        # Each env's factors in the controller's layout, re-applied to the entity's gains by `reset()`.
        # Identity until the first draw.
        self._kp_factor = torch.ones_like(term._kp)
        self._kd_factor = torch.ones_like(term._kd)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        if self.term is not None:
            self._draw_into_term(envs_idx)
            return
        n_dofs = len(self.entity.dofs_name)

        kp_factor = sample_uniform(
            self.kp_range[0],
            self.kp_range[1],
            (self._env.num_envs, n_dofs),
            device=self._env.device,
        )[envs_idx]
        kd_factor = sample_uniform(
            self.kd_range[0],
            self.kd_range[1],
            (self._env.num_envs, n_dofs),
            device=self._env.device,
        )[envs_idx]

        # NOTE: default_dofs_kp/kd may be (1, n_dofs) even with batch_dofs_info=True
        base_kp = self.entity.default_dofs_kp.expand(self._env.num_envs, -1)[envs_idx]
        base_kd = self.entity.default_dofs_kd.expand(self._env.num_envs, -1)[envs_idx]

        self.entity.set_dofs_kp(
            base_kp * kp_factor,
            dofs_idx_local=self.entity.dofs_idx_local,
            envs_idx=envs_idx,
        )
        self.entity.set_dofs_kd(
            base_kd * kd_factor,
            dofs_idx_local=self.entity.dofs_idx_local,
            envs_idx=envs_idx,
        )

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # Runs after `action_manager.reset()` re-read the controller's caches from the entity (which this term
        # leaves untouched): scale them by this env's factors again. See the class docstring.
        if self.term is None:
            return
        if envs_idx is None:
            envs_idx = slice(None)
        self._scale_term_gains(envs_idx)

    def _draw_into_term(self, envs_idx: slice | torch.Tensor) -> None:
        shape = self._kp_factor.shape
        kp_factor = sample_uniform(self.kp_range[0], self.kp_range[1], shape, device=self._env.device)
        kd_factor = sample_uniform(self.kd_range[0], self.kd_range[1], shape, device=self._env.device)
        assign_over_envs(self._kp_factor, envs_idx, kp_factor)
        assign_over_envs(self._kd_factor, envs_idx, kd_factor)
        self._scale_term_gains(envs_idx)

    def _scale_term_gains(self, envs_idx: slice | torch.Tensor) -> None:
        """The controller's caches for ``envs_idx`` := the entity's current gains times this env's factors.

        Read from the entity, in the controller's own layout (the read its ``build()`` / ``reset()`` makes),
        rather than scaling the cache in place, so a second call cannot compound the factor.
        """
        dofs_idx = self.term.genesis_dofs_idx
        assign_over_envs(self.term._kp, envs_idx, self.entity.get_dofs_kp(dofs_idx_local=dofs_idx) * self._kp_factor)
        assign_over_envs(self.term._kd, envs_idx, self.entity.get_dofs_kd(dofs_idx_local=dofs_idx) * self._kd_factor)


@EVENT_TERM_REGISTRY.register()
class RandomizeDeadbandEpsilon(_ResetModeActuatorDR):
    """
    Randomize the position-error threshold of a :class:`Deadband` action modifier.

    Requires
    --------
    The action term must have a ``Deadband`` modifier (either directly or inside a ``Compose`` chain).

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    action_term_name: str
        The name of the PD controller action term hosting the ``Deadband`` modifier.
    deadband_epsilon_range: Vec2FType
        When ``apply_as_ratio`` is false, the range of the deadband (position error, rad) sampled uniformly per env
        and DOF. When ``apply_as_ratio`` is true, the range of multipliers applied to the modifier's configured
        per-DOF baseline (see ``apply_as_ratio``).
    apply_as_ratio: bool
        If false (default), ``deadband_epsilon_range`` sets absolute deadband values in radians. If true,
        ``deadband_epsilon_range`` is interpreted as ``(low, high)`` multipliers (uniform per env and DOF): the
        deadband becomes ``m * epsilon_i``, where ``epsilon_i`` is that DOF's configured nominal deadband from
        :class:`~eden.managers.modifiers.actions.actuators.Deadband` at build time.
    """

    entity_name: str = "robot"
    action_term_name: str = ""
    deadband_epsilon_range: Vec2FType = (0.0, 0.01)
    apply_as_ratio: bool = False

    if TYPE_CHECKING:
        entity: RigidEntity
        term: _JointPDControllerBase
        modifier: Deadband

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import Deadband

        self.entity = self.resolve_entity(self.entity_name)
        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, Deadband, type(self).__name__, self.action_term_name)

    def _randomize(self, envs_idx: slice | torch.Tensor) -> None:
        n_ctrl = len(self.term.dofs_name)
        sampled = sample_uniform(
            self.deadband_epsilon_range[0],
            self.deadband_epsilon_range[1],
            (self._env.num_envs, n_ctrl),
            device=self._env.device,
        )[envs_idx]
        if self.apply_as_ratio:
            baseline = self.modifier._deadband_epsilon_row.to(device=sampled.device, dtype=sampled.dtype)
            eps = sampled * baseline.unsqueeze(0)
        else:
            eps = sampled
        self.modifier._deadband_epsilon[envs_idx] = eps


@EVENT_TERM_REGISTRY.register()
class RandomizeFrictionRatio(EventTerm):
    """Randomize sliding friction on ``links_name`` to absolute values, not a scale.

    Every ``compute()`` first pins the whole entity's base friction to 1.0 (``set_friction(1.0)``,
    entity-wide — this discards any authored ``default_geoms_friction``, including on links outside
    ``links_name``, which are left at a flat 1.0) and then writes per-link ratios on ``links_name``.
    ``friction_range`` is therefore an absolute coefficient range, regardless of what the asset
    authored; a source config expressed as "scale x authored value" must be converted to the
    resulting absolute range. Genesis multiplies the same ratio into the torsional and rolling
    coefficients where those axes are enabled, so this term randomizes spin resistance too.

    Genesis resolves a contact pair's friction as ``max(a, b)``, so the randomized coefficient only
    takes effect against surfaces whose friction is *below* the sampled value — author the ground
    plane low (e.g. ``PlaneOptions(default_geoms_friction={"*": 0.01})``) so the robot side always
    wins. Overlapping link sets (a whole-robot pass plus a tighter feet range) are ordered by the
    events' ``priority`` field (ascending). ``mode`` defaults to reset, but friction DR is usually a
    per-env-lifetime property — every in-tree task uses ``mode="startup"``. The full porting recipe
    lives in ``docs/guides/entities.md`` ("Porting a friction-randomization setup").
    """

    mode: EventMode = _RESET
    entity_name: str = "robot"
    links_name: list[str] = []
    friction_range: Vec2FType = (0.3, 1.2)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: Entity | None = None

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.links_name, ls_idx_local = entity.find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = entity

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # NOTE: set the friction of the terrain and objects small so that the robot geometry can decide the friction
        # NOTE: the friction between the object A and B is defined by the maximum friction of the two objects
        # NOTE: friction_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
        # NOTE: friction_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_a]
        # NOTE: contact_data.friction[i_c, i_b] = ti.max(ti.max(friction_a, friction_b), 1e-2)
        # NOTE: geoms_info.friction defaults to 1.0 and has to be in range [1e-2, 5.0] for simulation stability.

        friction_ratio = sample_uniform(
            self.friction_range[0],
            self.friction_range[1],
            (
                self._env.num_envs,
                len(self.links_name),
            ),
            device=self._env.device,
        )[envs_idx]

        self.entity.set_friction(1.0)
        self.entity.set_friction_ratio(
            friction_ratio,
            ls_idx_local=self.ls_idx_local,
            envs_idx=envs_idx,
        )


@EVENT_TERM_REGISTRY.register()
class RandomizeComShift(EventTerm):
    """Offset the named links' centres of mass, per env.

    Samples a per-env, per-link xyz offset (meters, link frame) from the three axis ranges and
    writes ``nominal_COM + shift``. Requires ``EnvOptions(batch_links_info=True)`` — Genesis stores
    link inertial info once for the whole batch otherwise, and rejects the per-env write. Rejects
    a link fixed to the world at build, whose COM the solver never reads.
    """

    _requires_env_options = ("batch_links_info",)

    mode: EventMode = _RESET
    entity_name: str = "robot"
    links_name: list[str] = []
    com_shift_x_range: Vec2FType = (-0.01, 0.01)
    com_shift_y_range: Vec2FType = (-0.01, 0.01)
    com_shift_z_range: Vec2FType = (-0.01, 0.01)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: Entity | None = None

    def build(self) -> None:
        resolved_names, ls_idx_local = self.resolve_entity(self.entity_name).find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = self.resolve_entity(self.entity_name)
        _reject_world_fixed_links(self, self.entity, self.links_name, resolved_names, ls_idx_local, "COM")
        _refuse_anchored_inertials(self, "COM")

    def _sample_com_shift_axis(self, axis_range: Vec2FType) -> torch.Tensor:
        return sample_uniform(
            axis_range[0],
            axis_range[1],
            (
                self._env.num_envs,
                len(self.ls_idx_local),
                1,
            ),
            device=self._env.device,
        )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        com_shift = torch.cat(
            [
                self._sample_com_shift_axis(self.com_shift_x_range),
                self._sample_com_shift_axis(self.com_shift_y_range),
                self._sample_com_shift_axis(self.com_shift_z_range),
            ],
            dim=-1,
        )[envs_idx]
        self.entity.set_COM_shift(com_shift, ls_idx_local=self.ls_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeMassShift(EventTerm):
    """Add a per-env offset to the named links' masses.

    Samples a per-env, per-link shift (kg) from ``mass_shift_range`` and writes
    ``nominal_mass + shift``. Requires ``EnvOptions(batch_links_info=True)``, and rejects a
    range whose lower bound could drive a selected link's mass non-positive — the write
    itself is unchecked (Genesis skips the device-side check for speed), so it would
    otherwise silently produce unphysical dynamics. It also rejects a link fixed to the world,
    whose mass the solver never reads.
    """

    _requires_env_options = ("batch_links_info",)

    mode: EventMode = _RESET
    entity_name: str = "robot"
    links_name: list[str] = []
    mass_shift_range: Vec2FType = (-1.0, 3.0)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: Entity | None = None

    def build(self) -> None:
        resolved_names, ls_idx_local = self.resolve_entity(self.entity_name).find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = self.resolve_entity(self.entity_name)
        _reject_world_fixed_links(self, self.entity, self.links_name, resolved_names, ls_idx_local, "mass")
        _refuse_anchored_inertials(self, "mass")
        # The worst-case sampled mass is nominal + range lower bound. A one-time device sync
        # at build is the earliest well-defined point to reject a range that could go
        # non-positive mid-training.
        nominal = self.entity.get_links_inertial_mass(self.ls_idx_local)  # (num_envs, n_selected)
        lightest = nominal.min(dim=0).values
        if float(lightest.min()) + self.mass_shift_range[0] <= 0.0:
            offender = int(lightest.argmin())
            raise ValueError(
                f"{term_subject(self)}: mass_shift_range[0]={self.mass_shift_range[0]} can drive link "
                f"'{resolved_names[offender]}' ({float(lightest[offender]):.3g} kg) non-positive. Narrow the "
                f"range or drop the link from links_name."
            )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        mass_shift = sample_uniform(
            self.mass_shift_range[0],
            self.mass_shift_range[1],
            (
                self._env.num_envs,
                len(self.ls_idx_local),
            ),
            device=self._env.device,
        )[envs_idx]
        self.entity.set_mass_shift(mass_shift, ls_idx_local=self.ls_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeLinkMassScale(EventTerm):
    """
    Multiplicative per-link mass randomization.

    Scales each specified link's mass by a random factor sampled from ``[mass_scale_range[0], mass_scale_range[1]]``.
    Internally uses ``set_mass_shift`` with ``shift = mass * (scale - 1)``. Requires
    ``EnvOptions(batch_links_info=True)``, and rejects a non-positive lower scale bound — the
    write itself is unchecked, so a scale of zero or below would silently produce a massless
    or negative-mass link.

    Parameters
    ----------
    entity_name: str
        The entity to randomize.
    links_name: list[str]
        Link name patterns to randomize. ``["*"]`` selects every link, so it suits only an entity
        with no link fixed to the world. On a fixed-base arm, a mounted hand or a ``fixed=True``
        prop, name the moving links: a fixed link is rejected at build, since its mass is never
        read, and the error gives a ``links_name`` that selects exactly the moving ones.
    mass_scale_range: Vec2FType
        Multiplicative scale range, e.g. ``(0.9, 1.2)`` for 90-120%.
    """

    _requires_env_options = ("batch_links_info",)

    entity_name: str = "robot"
    links_name: list[str] = []
    mass_scale_range: Vec2FType = (0.9, 1.1)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.ls_idx_local: torch.Tensor | None = None
        self.entity: RigidEntity | None = None
        self._base_mass: torch.Tensor | None = None

    def build(self) -> None:
        if self.mass_scale_range[0] <= 0.0:
            raise ValueError(
                f"{term_subject(self)}: mass_scale_range[0]={self.mass_scale_range[0]} must be strictly "
                f"positive — a scale of zero or below writes a massless or negative-mass link."
            )
        resolved_names, ls_idx_local = self.resolve_entity(self.entity_name).find_named_links_idx_local(self.links_name)
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = torch.as_tensor(ls_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self.entity = self.resolve_entity(self.entity_name)
        _reject_world_fixed_links(self, self.entity, self.links_name, resolved_names, ls_idx_local, "mass")
        _refuse_anchored_inertials(self, "mass")
        # Cache the unrandomized per-link mass at build time — the nominal this term's scale
        # factors multiply. ``get_links_inertial_mass`` is the per-link quantity being scaled;
        # ``get_mass()`` is the entity's total (not per-link at all).
        self._base_mass = self.entity.get_links_inertial_mass(self.ls_idx_local)  # (num_envs, n_selected)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Sample at full env count and index — mirrors the rest of this module
        # and avoids shape mismatches when ``envs_idx`` is a slice subset.
        scale = sample_uniform(
            self.mass_scale_range[0],
            self.mass_scale_range[1],
            (self._env.num_envs, len(self.ls_idx_local)),
            device=self._env.device,
        )[envs_idx]
        # shift = base_mass * (scale - 1)
        mass_shift = self._base_mass[envs_idx] * (scale - 1.0)
        self.entity.set_mass_shift(mass_shift, ls_idx_local=self.ls_idx_local, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class RandomizeTorqueNoise(EventTerm):
    """
    Residual Force Injection (RFI) — adds random torque noise to actuated joints.

    At each interval fire, samples ``[-rfi_scale, rfi_scale]`` per DOF and writes it into the ``TorqueOffset``
    modifier's ``_torque_offset`` buffer, which the action term reads when applying control torques. Simulates
    actuator noise and unmodeled dynamics.

    Must be used with ``mode = "interval"`` and an ``interval_range_s`` matching the control timestep; the event
    manager rejects an interval term left at ``interval_range_s=None``.

    Requires
    --------
    The action term must have a ``TorqueOffset`` modifier in its chain.

    Parameters
    ----------
    action_term_name: str
        Name of the action term whose ``TorqueOffset`` modifier should receive noise.
    rfi_scale: float
        Maximum magnitude of the random torque noise (N·m).
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    action_term_name: str = ""
    rfi_scale: float = 0.5

    if TYPE_CHECKING:
        term: ExplicitPDController | VelocityFeedforwardPDController
        modifier: TorqueOffset

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import TorqueOffset

        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, TorqueOffset, type(self).__name__, self.action_term_name)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Always sample for the full env count and index — matches the rest of
        # this module's pattern and avoids subtle shape mismatches when ``envs_idx``
        # is a slice subset (`n_sample == num_envs` but only a slice writes back).
        noise = sample_uniform(
            -self.rfi_scale,
            self.rfi_scale,
            (self._env.num_envs, self.term._n_dofs),
            device=self._env.device,
        )
        self.modifier._torque_offset[envs_idx] = noise[envs_idx]


@EVENT_TERM_REGISTRY.register()
class ERFI50(EventTerm):
    """
    ERFI-50 noise: 50% Random Actuation Offset (RAO) + 50% Random Force Injection (RFI).

    For each environment, a random value *r* in [0, 1] is drawn at reset:

    - **r < erfi_ratio (RAO):** a fixed scalar offset ``r * 3 * upper_bound`` is broadcast to all DOFs and held
      constant for the entire episode.
    - **r >= erfi_ratio (RFI):** independent per-DOF torque noise drawn uniformly from ``[lower_bound, upper_bound]``.

    Requires
    --------
    The action term must have a ``TorqueOffset`` modifier.

    Parameters
    ----------
    interval_range_s: tuple[float, float] | None
        The interval range in seconds. Expected to be set to the control timestep.
    action_term_name: str
        The name of the action term to apply the noise to.
    lower_bound: float
        The lower bound of the torque noise (N·m).
    upper_bound: float
        The upper bound of the torque noise (N·m).
    erfi_ratio: float
        The fraction of envs that use RAO instead of RFI.

    Reference
    ---------
    - Learning and Deploying Robust Locomotion Policies with Minimal Dynamics Randomization
      (https://arxiv.org/pdf/2209.12878)
    """

    mode: EventMode = _INTERVAL
    interval_range_s: tuple[float, float] | None = None
    action_term_name: str = ""
    lower_bound: float = -7.0 / 9.0
    upper_bound: float = 7.0 / 9.0
    erfi_ratio: float = 0.5

    if TYPE_CHECKING:
        term: ExplicitPDController | VelocityFeedforwardPDController
        modifier: TorqueOffset

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import TorqueOffset

        self.term = self._env.action_manager.get_term(self.action_term_name)
        self.modifier = _require_modifier(self.term, TorqueOffset, type(self).__name__, self.action_term_name)
        self._erfi_random = sample_uniform(0.0, 1.0, (self._env.num_envs,), device=self._env.device)
        self._is_rao = (self._erfi_random < self.erfi_ratio).unsqueeze(1)
        self._rao_noise = (self._erfi_random * 3.0 * self.upper_bound).unsqueeze(1).repeat(1, self.term._n_dofs)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Full-batch draws written into the selected rows (`assign_over_envs`): the auto-reset hands a
        # bool mask, and gathering or writing through it as an index is a GPU->CPU sync per op.
        # Resample per-env RAO / RFI selector.
        erfi_random = sample_uniform(0.0, 1.0, (self._env.num_envs,), device=self._env.device)
        assign_over_envs(self._erfi_random, envs_idx, erfi_random)
        assign_over_envs(self._is_rao, envs_idx, (erfi_random < self.erfi_ratio).unsqueeze(1))

        # RAO: fixed scalar noise broadcast to all DOFs.
        rao_noise = (erfi_random * 3.0 * self.upper_bound).unsqueeze(1).expand(-1, self.term._n_dofs)
        assign_over_envs(self._rao_noise, envs_idx, rao_noise)

        # Apply initial offset so first post-reset step is consistent.
        rfi_noise = sample_uniform(
            self.lower_bound, self.upper_bound, (self._env.num_envs, self.term._n_dofs), device=self._env.device
        )
        assign_over_envs(self.modifier._torque_offset, envs_idx, torch.where(self._is_rao, self._rao_noise, rfi_noise))

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)

        # RFI: independent per-DOF noise, drawn for every env and written into the due ones: the interval
        # tick hands a bool mask, which as an index is a GPU->CPU sync per op.
        rfi_noise = sample_uniform(
            self.lower_bound,
            self.upper_bound,
            (self._env.num_envs, self.term._n_dofs),
            device=self._env.device,
        )
        assign_over_envs(self.modifier._torque_offset, envs_idx, torch.where(self._is_rao, self._rao_noise, rfi_noise))
