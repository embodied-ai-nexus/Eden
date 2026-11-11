"""Joint-space PD and velocity action terms (implicit/explicit controllers)."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, ClassVar, Literal

import genesis as gs
import torch

import eden as en
from eden.constants import ReferenceSource
from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.modifiers import ACTION_MODIFIER_REGISTRY, ActionModifier
from eden.utils.misc import assign_over_envs
from eden.utils.string import first_few, key_selects, term_subject

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.modifiers import ActionModifierOptions


class _JointPDControllerBase(ActionTerm):
    """
    Base class for joint PD controllers.

    The processed target is built additively as::

        target = raw_action * scale + offset + reference_offset + dofs_pos_offset

    where ``dofs_pos_offset`` is a per-env offset that domain randomization sets through
    :meth:`set_dofs_pos_offset` (zero until then), and ``reference_offset`` is selected by
    ``reference_source`` (see :class:`eden.constants.ReferenceSource`):

    - ``ReferenceSource.ZERO``: no reference offset.
    - ``ReferenceSource.DEFAULT`` (class default): the entity's
      ``default_dofs_pos`` for the controlled DOFs.
    - ``ReferenceSource.MOTION_REF``: per-step joint position read from the
      motion manager (``env.motion_manager.data[entity_name].dofs_pos[:, 0, :]``).
      Used for residual policies that learn deltas on top of a retargeted
      demonstration. Requires the env to be a :class:`MimicRLEnvBase` whose
      motion manager tracks the controlled entity, and the controller's
      ``dofs_name`` must be a subset of that entity's motion-ref DOFs.
    - ``ReferenceSource.DELTA``: per-step joint position captured from the
      entity at the start of each control step (``entity.get_dofs_pos(...)``);
      the action is then interpreted as a delta on top of the previous step's
      DOF position. The captured offset is constant across decimation substeps.
    - ``ReferenceSource.ACCUMULATE``: the action is integrated onto a *persistent*
      target — ``target_t = clamp(target_{t-1} + raw * scale, soft_pos_limits)`` —
      starting from and reset to ``default_dofs_pos``. Unlike ``DELTA`` (which
      re-reads the measured DOF position each step), the target accumulates across
      steps, so the action manager holds the raw delta (read by obs/reward terms)
      while the controller commands the running target. Requires ``offset == 0``
      (it would otherwise be re-added every step). Position controllers only.

    **Coupled (mimic) joints.** ``mimic_dofs`` derives extra position targets from the action instead of
    reading them off it — an underactuated hand whose follower joints track a driver at a fixed ratio, as
    IsaacLab's ``MimicJointPositionAction`` does for the Inspire hands::

        ImplicitPDController.configure(
            entity_name="left_hand",
            dofs_name=InspireRH56DFQ_L.actuated_dofs_name,  # the 6 drivers: the action columns
            mimic_dofs=InspireRH56DFQ_L.metadata.mimic_dofs,  # follower -> (driver, multiplier, offset)
        )

    Each follower's target is ``driver_target * multiplier + offset``, where ``driver_target`` is the
    driver's full position target (scale, offset and reference applied, before ``dofs_pos_offset`` and the
    modifier). The action stays driver-wide (:attr:`action_dim` counts the drivers), while
    ``processed_action``, ``dofs_name`` and ``dofs_idx_local`` widen to drivers-then-followers after
    :meth:`build`, because the term *commands* every one of them. A follower must be free in the simulator:
    a URDF ``<mimic>`` tag is already a Genesis joint-equality constraint on it, which this would fight, so
    :meth:`build` refuses one.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: str | list[str]
        The names of the DOFs to control.
    reference_source: ReferenceSource
        Source of the reference offset added on top of ``raw_action * scale + offset``.
    scale: float | dict[str, float] | None
        The scale to apply to the actions. ``None`` (the default) applies none, as ``1.0`` does,
        but reads as "not set": a TaskMod that binds a robot's action scale fills a ``None`` and
        keeps an authored ``1.0`` (``VelocityTrackingRobotMod``).
    scale_ratio: float | None
        If set, replace ``scale`` with ``scale_ratio * (upper - lower)`` per
        controlled DOF, using the entity's DOF position limits.
    offset: float | dict[str, float]
        The offset to apply to the actions.
    dict_keys_scope: Literal["term", "entity"]
        What each key of a dict-valued ``scale`` / ``offset`` must match. ``"term"`` (the default):
        at least one of this term's own DOFs, so a key naming a DOF the term does not control
        raises. ``"entity"``: at least one DOF of the entity; keys that match only DOFs outside
        this term are skipped. That lets terms that split one robot between them share its
        whole-body dict (``G1_ACTION_SCALE``) instead of each carrying a hand-filtered copy. A key
        that matches no DOF of the entity still raises under either scope.
    modifier: ActionModifierOptions | None
        Optional composable action modifier (e.g., delay, friction, effort clip).
    mimic_dofs: dict[str, tuple[str, float, float]]
        Coupled DOFs, ``follower -> (driver, multiplier, offset)``: each follower is commanded
        ``driver_target * multiplier + offset``. Drivers must be in ``dofs_name``; followers must not be.
        Position controllers only. Empty (the default) couples nothing.
    """

    reference_source: ReferenceSource = ReferenceSource.DEFAULT
    scale: float | dict[str, float] | None = None
    scale_ratio: float | None = None
    offset: float | dict[str, float] = 0.0
    dict_keys_scope: Literal["term", "entity"] = "term"
    modifier: ActionModifierOptions | None = None
    mimic_dofs: dict[str, tuple[str, float, float]] = {}

    #: Whether the action is a joint *position* target (after ``scale`` / ``offset`` / the
    #: reference). The velocity controllers reuse this base for its scale/offset plumbing but
    #: command a velocity. Leading underscore keeps it out of the configurable parameters. Read it
    #: through :func:`is_joint_position_controller`.
    _targets_position_: ClassVar[bool] = True
    #: Whether ``mimic_dofs`` is honored: False where the action layout is not one column per commanded
    #: position DOF (the velocity controllers, and :class:`VelocityFeedforwardPDController`'s doubled action).
    _supports_mimic_: ClassVar[bool] = True

    if TYPE_CHECKING:
        entity: RigidEntity

    def __init__(
        self,
        env: EnvBase,
        options: ActionTermOptions,
    ):
        super().__init__(env=env, options=options)
        self._scale: float | torch.Tensor | None = None
        self._offset: torch.Tensor | None = None
        self._motion_ref_dofs_idx: torch.Tensor | None = None
        self._delta_reference: torch.Tensor | None = None
        # Per-env (num_envs, n_dofs) position offset from set_dofs_pos_offset; None until first set.
        self._dofs_pos_offset: torch.Tensor | None = None
        # ReferenceSource.ACCUMULATE state: the running target integrated across steps,
        # its default (init/reset) value, and the soft-limit clamp bounds. None otherwise.
        self._accumulated_target: torch.Tensor | None = None
        self._default_target: torch.Tensor | None = None
        self._target_lower: torch.Tensor | None = None
        self._target_upper: torch.Tensor | None = None
        # ``mimic_dofs`` state, set by ``_build_mimic``: the driver count (the action width), the drivers'
        # Genesis index (DELTA captures only those), and per follower its driver's column plus the coupling.
        self._n_action_dofs: int | None = None
        self._driver_genesis_dofs_idx: range | torch.Tensor | None = None
        self._mimic_cols: torch.Tensor | None = None
        self._mimic_mult: torch.Tensor | None = None
        self._mimic_offset: torch.Tensor | None = None
        self._mimic_follower_dofs_name: list[str] = []
        self._modifier: ActionModifier | None = None
        if self.modifier is not None:
            self._modifier = ACTION_MODIFIER_REGISTRY.get(self.modifier.name)(env=env, options=self.modifier)

    @property
    def action_dim(self) -> int:
        """The action columns this term consumes: one per DOF, or one per *driver* under ``mimic_dofs``."""
        return self._n_dofs if self._n_action_dofs is None else self._n_action_dofs

    @property
    def mimic_follower_dofs_name(self) -> list[str]:
        """The DOFs commanded from a driver's target through ``mimic_dofs`` (empty without couplings).

        They are the trailing columns of ``dofs_name`` / ``processed_action`` after :meth:`build`, and take
        no action column. Hardware that couples them itself (a real Inspire hand takes six commands) must not
        be sent them.
        """
        return list(self._mimic_follower_dofs_name)

    @property
    def effective_scale(self) -> float | torch.Tensor:
        """The scale actually applied to the raw action, as resolved by :meth:`build`.

        Not the configured :attr:`scale` field: ``scale_ratio`` replaces it with a
        limit-derived ``(1, num_dofs)`` tensor, and a ``dict`` scale is expanded to
        the same shape; a plain ``float`` scale stays a ``float``, and an unset one is ``1.0``.
        Under ``mimic_dofs`` it is driver-wide, like the action it scales.
        It scales the position target only, which is the whole action except on
        :class:`VelocityFeedforwardPDController`, whose velocity half takes ``velocity_scale``.
        """
        return self._scale

    @property
    def effective_offset(self) -> torch.Tensor:
        """The offset actually added to ``raw_action * scale``, as resolved by :meth:`build`.

        Not the configured :attr:`offset` field: under ``ReferenceSource.DEFAULT`` the
        entity's ``default_dofs_pos`` is folded in, and the per-env offset set through
        :meth:`set_dofs_pos_offset` is added. Shape ``(1, num_dofs)``, or
        ``(num_envs, num_dofs)`` once randomized.

        The per-step ``reference_offset`` for the other ``reference_source`` modes is
        *not* included — it is applied inside ``compute()``. Under ``mimic_dofs`` it is driver-wide, like
        :attr:`effective_scale`: a follower's target is derived from its driver's, not from an offset of its own.
        """
        if self._dofs_pos_offset is None:
            return self._offset
        return self._offset + self._dofs_pos_offset[:, : self._offset.shape[-1]]

    def set_dofs_pos_offset(self, offset: torch.Tensor, envs_idx: slice | torch.Tensor | None = None) -> None:
        """Set the per-env joint-position offset added to this controller's position target.

        The offset adds to everything :meth:`build` resolved (the configured ``offset`` and the
        ``reference_source`` reference) and replaces neither. Under ``ReferenceSource.ACCUMULATE``
        it is added to the running target rather than integrated into it. It persists across
        resets until set again, and :attr:`effective_offset` includes it. This is the hook for
        joint-zero domain randomization such as
        :class:`~eden.managers.terms.events.domain_rand.RandomizeDofsPosOffset`.

        Parameters
        ----------
        offset: torch.Tensor
            ``(num selected envs, num_dofs)`` offsets, columns in this term's ``dofs_name`` order — which
            under ``mimic_dofs`` is drivers then followers, so a follower takes its own offset.
        envs_idx: slice | torch.Tensor | None
            The envs to write; ``None`` for all.

        Raises
        ------
        ValueError
            On a controller that commands velocities, or an ``offset`` whose last dimension is not
            this term's DOF count.
        """
        if not self._targets_position_:
            raise ValueError(
                f"{term_subject(self)} commands joint velocities, so it has no joint-position offset to set."
            )
        if offset.shape[-1] != self._n_dofs:
            raise ValueError(
                f"{term_subject(self)}: set_dofs_pos_offset got an offset for {offset.shape[-1]} DOFs, but the term "
                f"controls {self._n_dofs} ({first_few(self.dofs_name)})."
            )
        if self._dofs_pos_offset is None:
            self._dofs_pos_offset = torch.zeros(self.num_envs, self._n_dofs, device=self.device)
        self._dofs_pos_offset[slice(None) if envs_idx is None else envs_idx] = offset

    def build(self) -> None:
        super().build()
        if self.dict_keys_scope not in ("term", "entity"):
            raise ValueError(
                f"{term_subject(self)}: dict_keys_scope must be 'term' or 'entity', got {self.dict_keys_scope!r}."
            )
        self._build_scale_and_offset()

        if self.reference_source == ReferenceSource.DELTA:
            self._delta_reference = torch.zeros(self.num_envs, self._n_dofs, device=self.device)

        if self.reference_source == ReferenceSource.ACCUMULATE:
            if self._has_nonzero_offset():
                raise ValueError(
                    f"{term_subject(self)}: reference_source='accumulate' does not support a non-zero "
                    f"`offset`: it would be re-added to the running target every step. Got offset={self.offset!r}."
                )
            dofs = self.entity.find_named_dofs(self.dofs_name)
            self._default_target = self.entity.get_default_dofs_pos(dofs)
            self._accumulated_target = self._default_target.clone()
            # Columns rather than `get_soft_dofs_pos_limits`: the clamp in `compute` reads these every step, and
            # the stored `(1 or num_envs, n, 2)` keeps a shared row broadcast instead of one copy per env.
            soft_limits = self.entity.soft_dofs_pos_limits[:, dofs.columns]
            self._target_lower = soft_limits[..., 0].expand(self.num_envs, -1)
            self._target_upper = soft_limits[..., 1].expand(self.num_envs, -1)

        # Everything above is driver-wide; the couplings widen the command view last, so the modifier
        # below sees every DOF the term commands.
        if self.mimic_dofs:
            self._build_mimic()

        # Build action modifier if configured
        if self._modifier is not None:
            self._modifier.build(
                num_envs=self.num_envs,
                device=self.device,
                entity=self.entity,
                dofs_idx_local=self.dofs_idx_local,
            )

    def _build_mimic(self) -> None:
        """Validate ``mimic_dofs`` and widen the command view from the drivers to drivers + followers."""
        if not (self._targets_position_ and self._supports_mimic_):
            raise ValueError(
                f"{term_subject(self)}: `mimic_dofs` is supported on the joint position controllers "
                f"(ImplicitPDController, ExplicitPDController) only. "
                f"Use ImplicitPDController or ExplicitPDController for the coupled DOFs."
            )
        drivers = list(self.dofs_name)
        followers, spec = [], []
        for follower, coupling in self.mimic_dofs.items():
            if not isinstance(coupling, (tuple, list)) or len(coupling) != 3:
                raise ValueError(
                    f"{term_subject(self)}: `mimic_dofs[{follower!r}]` must be (driver, multiplier, offset), "
                    f"got {coupling!r}."
                )
            driver, mult, offset = coupling[0], float(coupling[1]), float(coupling[2])
            if follower not in self.entity.dofs_name:
                raise ValueError(
                    f"{term_subject(self)}: `mimic_dofs` follower {follower!r} is not a DOF of the entity. "
                    f"Its DOFs: {first_few(self.entity.dofs_name, limit=40)}."
                )
            if follower in drivers:
                raise ValueError(
                    f"{term_subject(self)}: {follower!r} is both a `mimic_dofs` follower and in `dofs_name`. A "
                    f"follower's target is derived from its driver, so it takes no action column: remove it "
                    f"from `dofs_name`."
                )
            if driver in self.mimic_dofs:
                raise ValueError(
                    f"{term_subject(self)}: `mimic_dofs` follower {follower!r} is driven by {driver!r}, which is "
                    f"itself a follower. Chains are not resolved: couple {follower!r} to {driver!r}'s own "
                    f"driver {self.mimic_dofs[driver][0]!r} with the multipliers composed."
                )
            if driver not in drivers:
                raise ValueError(
                    f"{term_subject(self)}: `mimic_dofs` follower {follower!r} is driven by {driver!r}, which is "
                    f"not in this term's `dofs_name` {first_few(drivers, limit=40)}. The driver must be an "
                    f"action column of the same term: add {driver!r} to `dofs_name`."
                )
            if not (math.isfinite(mult) and math.isfinite(offset)):
                raise ValueError(
                    f"{term_subject(self)}: `mimic_dofs[{follower!r}]` has a non-finite multiplier/offset "
                    f"({coupling[1]!r}, {coupling[2]!r})."
                )
            followers.append(follower)
            spec.append((drivers.index(driver), mult, offset))
        self._reject_simulator_coupled_followers(followers)

        self._n_action_dofs = len(drivers)
        self._driver_genesis_dofs_idx = self.genesis_dofs_idx
        self._mimic_cols = torch.tensor([col for col, _, _ in spec], dtype=torch.long, device=self.device)
        self._mimic_mult = torch.tensor([[mult for _, mult, _ in spec]], dtype=gs.tc_float, device=self.device)
        self._mimic_offset = torch.tensor([[off for _, _, off in spec]], dtype=gs.tc_float, device=self.device)
        self._mimic_follower_dofs_name = followers

        _, follower_idx = self.entity.find_named_dofs_idx_local(
            followers, name_scope=self.entity.dofs_name, preserve_order=True
        )
        follower_idx = torch.as_tensor(follower_idx, dtype=self.dofs_idx_local.dtype, device=self.device)
        self.dofs_name = drivers + followers
        self.dofs_idx_local = torch.cat([self.dofs_idx_local, follower_idx]).contiguous()
        self._n_dofs = len(self.dofs_name)
        self._processed_action = torch.zeros(self.num_envs, self._n_dofs, device=self.device)

    def _reject_simulator_coupled_followers(self, followers: list[str]) -> None:
        """Refuse a follower Genesis already couples through a joint equality (a URDF ``<mimic>`` tag).

        Parameters
        ----------
        followers: list[str]
            The ``mimic_dofs`` followers to check.
        """
        gs_entity = self.entity._entity
        joint_idx = {gs_entity.get_joint(name).idx: name for name in followers}
        coupled = [
            (joint_idx.get(eq.eq_obj1id) or joint_idx.get(eq.eq_obj2id), eq.name)
            for eq in gs_entity.equalities
            if eq.type == gs.EQUALITY_TYPE.JOINT and (eq.eq_obj1id in joint_idx or eq.eq_obj2id in joint_idx)
        ]
        if coupled:
            raise ValueError(
                f"{term_subject(self)}: `mimic_dofs` followers {[name for name, _ in coupled]} are already coupled in "
                f"the simulator by joint equalities {[eq for _, eq in coupled]} — Genesis turns a URDF <mimic> tag "
                f"into one. Commanding them too would have the PD targets fight the constraint. Either drop "
                f"`mimic_dofs` and let the constraint drive them (with kp=kd=0 on the followers), or load an "
                f"asset with the <mimic> tags stripped (IsaacLab ignores them, so that is the parity setting)."
            )

    def _expand_mimic(self, target: torch.Tensor) -> torch.Tensor:
        """Append the followers' targets, ``driver_target * multiplier + offset``, to a driver-wide target."""
        return torch.cat([target, target.index_select(1, self._mimic_cols) * self._mimic_mult + self._mimic_offset], 1)

    def _has_nonzero_offset(self) -> bool:
        """Return True if the configured ``offset`` parameter is non-zero on any DOF."""
        if isinstance(self.offset, float):
            return self.offset != 0.0
        if isinstance(self.offset, dict):
            return any(v != 0.0 for v in self.offset.values())
        return False

    def _build_scale_and_offset(self) -> None:
        if self.scale_ratio is not None:
            if isinstance(self.scale_ratio, bool) or not isinstance(self.scale_ratio, int | float):
                raise ValueError(f"{term_subject(self)}: unsupported scale_ratio type {type(self.scale_ratio)}.")
            lower, upper = self.entity.get_dofs_limit(dofs_idx_local=self.genesis_dofs_idx)
            dof_range = upper - lower
            revolute_fallback = torch.full_like(dof_range, 2.0 * torch.pi)
            finite_range = torch.isfinite(dof_range)
            nonpositive_finite_range = finite_range & (dof_range <= 0.0)
            if nonpositive_finite_range.any().item():
                warnings.warn(
                    "scale_ratio encountered finite DOF limits with upper <= lower; clamping those DOF ranges to 1e-6.",
                    stacklevel=2,
                )
            dof_range = torch.where(finite_range, dof_range, revolute_fallback).clamp(min=1e-6)
            self._scale = float(self.scale_ratio) * dof_range
        elif self.scale is None:
            self._scale = 1.0
        elif isinstance(self.scale, float):
            self._scale = self.scale
        elif isinstance(self.scale, dict):
            self._scale = self._dofs_dict_to_tensor("scale", self.scale, fill=1.0)
        else:
            raise ValueError(f"{term_subject(self)}: unsupported scale type {type(self.scale)}.")

        # ``_offset`` is consumed by ``_compute_pos_target(raw_pos)`` where
        # ``raw_pos`` has shape (batch, n_dofs) — for sub-controllers like
        # ``VelocityFeedforwardPDController`` (action_dim = 2 * n_dofs)
        # sizing this to ``action_dim`` would break broadcasting on the
        # position half. Match the position-slice width instead.
        if isinstance(self.offset, float):
            self._offset = torch.full((1, self._n_dofs), self.offset, device=self.device)
        elif isinstance(self.offset, dict):
            self._offset = self._dofs_dict_to_tensor("offset", self.offset, fill=0.0)
        else:
            raise ValueError(f"{term_subject(self)}: unsupported offset type {type(self.offset)}.")

        # Fold default DOF positions into the static offset for `reference_source="default"`.
        if self.reference_source == ReferenceSource.DEFAULT:
            if self._has_nonzero_offset():
                en.logger.warning(
                    f"{term_subject(self)}: `offset` is non-zero with "
                    f"reference_source='default'; the static offset stacks additively "
                    f"on top of `default_dofs_pos`."
                )
            # Columns rather than `get_default_dofs_pos`, which would widen `_offset` to every env:
            # it stays the stored `(1 or num_envs, n)`, as `effective_offset` documents.
            default_offset = self.entity.default_dofs_pos[:, self.entity.find_named_dofs(self.dofs_name).columns]
            self._offset = default_offset + self._offset

        if self.reference_source == ReferenceSource.MOTION_REF:
            if self._has_nonzero_offset():
                en.logger.warning(
                    f"{term_subject(self)}: `offset` is non-zero with "
                    f"reference_source='motion_ref'; the static offset stacks additively "
                    f"on top of the per-step motion-reference qpos."
                )
            self._motion_ref_dofs_idx = self._resolve_motion_ref_dofs_idx()

        if self.reference_source == ReferenceSource.DELTA:
            if self._has_nonzero_offset():
                en.logger.warning(
                    f"{term_subject(self)}: `offset` is non-zero with "
                    f"reference_source='delta'; the static offset stacks additively "
                    f"on top of the previous step's DOF positions."
                )

    def _dofs_dict_to_tensor(self, field: str, data: dict[str, float], *, fill: float) -> torch.Tensor:
        """Expand a ``{pattern: value}`` dict over this term's DOFs into a ``(1, num_dofs)`` tensor.

        One column per controlled DOF, the width of the position target both ``scale`` and ``offset``
        apply to -- not ``action_dim``, which :class:`VelocityFeedforwardPDController` doubles for its
        velocity half. DOFs no key matches keep ``fill``. A key must match a DOF of the entity; one
        that matches only DOFs outside this term is skipped under ``dict_keys_scope="entity"`` and
        raises under ``"term"``, with a message that names the scope switch. Keys are applied in dict
        order, so a later key overrides an earlier one on a DOF both match.

        Parameters
        ----------
        field: str
            The option the dict came from, for error messages.
        data: dict[str, float]
            The dict to expand, keyed by glob / regex DOF-name patterns.
        fill: float
            The value for DOFs no key matches.
        """
        out = torch.full((1, self._n_dofs), fill, device=self.device)
        # A key is taken literally when it is exactly one of the *entity's* joints, not only this term's: a key
        # naming another term's ``pin.1``, or a passive one, must not fall back to a pattern and select ``pin_1``.
        literal = {joint.name for joint in self.entity.joints}
        entity_dofs = set(self.entity.dofs_name)
        for pattern, value in data.items():
            # ``dofs_name`` is in column order, so a matching name's position is its column.
            cols = [i for i, name in enumerate(self.dofs_name) if key_selects(pattern, name, literal)]
            if cols:
                out[:, cols] = value
            elif not any(key_selects(pattern, name, literal) for name in entity_dofs):
                raise ValueError(
                    f"{term_subject(self)}: `{field}` key {pattern!r} matches no DOF of the entity. "
                    f"Its DOFs: {first_few(self.entity.dofs_name, limit=40)}."
                )
            elif self.dict_keys_scope == "term":
                raise ValueError(
                    f"{term_subject(self)}: `{field}` key {pattern!r} matches only DOFs this term does not control "
                    f"(it controls {first_few(self.dofs_name, limit=40)}). Drop the key, or set "
                    f"dict_keys_scope='entity' to share one entity-wide dict between the terms that split the "
                    f"entity; keys matching no DOF of the entity still raise under that scope."
                )
            # Otherwise the key names DOFs another term owns, and the "entity" scope skips it.
        return out

    def _resolve_motion_ref_dofs_idx(self) -> torch.Tensor:
        """Map this controller's ``dofs_name`` to indices into the entity's motion-ref DOF axis."""
        # Local import to avoid the eden.envs.base <-> eden.managers cycle at module load.
        from eden.envs.base import MimicRLEnvBase
        from eden.utils.motion_reference.lookup import get_motion_dofs_idx

        if not isinstance(self._env, MimicRLEnvBase):
            raise RuntimeError(
                f"{term_subject(self)}: reference_source='motion_ref' requires a "
                f"MimicRLEnvBase env, got {type(self._env).__name__}."
            )
        return get_motion_dofs_idx(self._env, self.entity_name, self.dofs_name)

    def _motion_ref_offset(self) -> torch.Tensor:
        """Read the current motion-reference qpos for the controlled DOFs.

        Reads from ``motion_manager.data`` (filled once per step by
        ``MotionManager.compute(dt)`` at the start of ``MimicRLEnvBase.step()``)
        rather than ``reference_frame``, which would force an extra recompute pass
        and a GPU sync via the property's ``outdated_mask.any()`` check. Frame ``0``
        of ``data`` is the current-step reference (modulo ``data_start_from``).
        """
        ref = self._env.motion_manager.data[self.entity_name].dofs_pos
        # ref shape: (num_envs, num_frames, num_motion_dofs); frame 0 is the current step.
        return ref[:, 0, :].index_select(1, self._motion_ref_dofs_idx)

    def _compute_pos_target(self, raw_pos: torch.Tensor) -> torch.Tensor:
        """Build a position target from a raw action slice: ``raw * scale + offset (+ ref) (+ dofs_pos_offset)``.

        Under ``mimic_dofs`` the followers' targets are derived from the drivers' before ``dofs_pos_offset``,
        which is per commanded DOF.
        """
        target = raw_pos * self._scale + self._offset
        if self._motion_ref_dofs_idx is not None:
            target = target + self._motion_ref_offset()
        if self._delta_reference is not None:
            target = target + self._delta_reference
        if self._mimic_cols is not None:
            target = self._expand_mimic(target)
        if self._dofs_pos_offset is not None:
            target = target + self._dofs_pos_offset
        return target

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        if self._delta_reference is not None:
            # Capture once per env step; constant across decimation substeps. Drivers only under
            # ``mimic_dofs``: the reference is added to the driver-wide target the followers derive from.
            delta_idx = (
                self.genesis_dofs_idx if self._driver_genesis_dofs_idx is None else self._driver_genesis_dofs_idx
            )
            self._delta_reference[:] = self.entity.get_dofs_pos(dofs_idx_local=delta_idx)
        if self._accumulated_target is None:
            self._processed_action = self._compute_pos_target(self._raw_action)
        else:
            # ReferenceSource.ACCUMULATE: integrate raw * scale (build rejects a non-zero offset,
            # and no reference applies) onto the running target and clamp it to the soft joint
            # limits, in place to keep the hot path allocation-free. The joint-position offset is
            # added to the result, never integrated.
            self._accumulated_target.add_(self._raw_action * self._scale).clamp_(self._target_lower, self._target_upper)
            self._processed_action = self._accumulated_target
            if self._mimic_cols is not None:
                self._processed_action = self._expand_mimic(self._processed_action)
            if self._dofs_pos_offset is not None:
                self._processed_action = self._processed_action + self._dofs_pos_offset

        # Apply the control-step modifiers (e.g., envelope clip, backlash). A delay is not one of
        # them: it acts per physics step, in `_applied_command`.
        # The target is already absolute (delta + dofs_pos), so state-dependent
        # modifiers (e.g. EnvelopeClip) receive correct absolute joint positions.
        if self._modifier is not None:
            self._processed_action = self._modifier.modify_processed_action(self._processed_action)

    def _applied_command(self) -> torch.Tensor:
        """``processed_action`` as the actuator acts on it during this physics step.

        Passes the command through the modifier's per-physics-step hook
        (:meth:`~eden.managers.modifiers.ActionModifier.modify_applied_target` — e.g. ``ActionDelay``).
        Every ``apply_actions()`` must call this exactly once and drive the entity with the result:
        a stateful hook advances on each call, and reading ``_processed_action`` directly skips it.
        """
        if self._modifier is None:
            return self._processed_action
        return self._modifier.modify_applied_target(self._processed_action)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0
        if self._delta_reference is not None:
            # Overwritten in the next compute(); zeroed here for parity with _raw_action.
            self._delta_reference[envs_idx] = 0.0
        if self._accumulated_target is not None:
            # ReferenceSource.ACCUMULATE: restart the running target from the default pose.
            # `torch.where` through the auto-reset's mask: a mask gather + write syncs twice.
            assign_over_envs(self._accumulated_target, envs_idx, self._default_target)

        if self._modifier is not None:
            self._modifier.reset(envs_idx)


@ACTION_TERM_REGISTRY.register()
class ImplicitPDController(_JointPDControllerBase):
    """
    Implicit joint PD controller using Genesis's built-in position control.

    Uses ``control_dofs_pos`` which leverages Genesis's implicit integration scheme:
    the PD target is recomputed at every physics substep and an implicit damping term
    (kd * substep_dt) is added to the mass matrix for numerical stability. This matches
    the behavior of MuJoCo, making it suitable for sim2sim transfer.
    """

    def apply_actions(self) -> None:
        target = self._applied_command()
        self.verify_command_shape(target)
        self.entity.control_dofs_pos(target, dofs_idx_local=self.genesis_dofs_idx)


@ACTION_TERM_REGISTRY.register()
class ExplicitPDController(_JointPDControllerBase):
    """
    Explicit joint PD controller using manual torque computation.

    Computes PD torques explicitly and applies them via ``control_dofs_force``.
    This is more physically accurate as it matches real-world motor control where
    only torque commands are valid. Better suited for sim2real transfer and easier
    to extend with domain randomization (e.g., randomizing kp/kd, adding motor delays).

    An optional ``modifier`` can be provided to compose post-processing steps
    (delay, backlash, DC saturation, friction, effort clipping, motor strength scaling)
    on top of the PD controller::

        ExplicitPDController.configure(
            entity_name="robot",
            dofs_name=["joint1", "joint2"],
            modifier=Compose.configure(
                modifiers=[
                    ActionDelay.configure(delay_range_steps=(1, 3)),
                    FrictionModel.configure(static_friction=2.4, dynamic_friction=0.24),
                    EffortClip.configure(driving_torque_limit=111.0, ...),
                    MotorStrength.configure(motor_strength=0.8),
                ]
            ),
        )

    Note
    ----
    Requires ``sim_substeps=1`` so that PD torques are recomputed at every physics
    substep. Adjust ``decimation`` accordingly to maintain the desired control frequency.

    The ``-kd * qdot`` term is explicit under every ``EnvOptions.integrator``: Genesis' implicit
    passes only damp the DOFs it drives itself (:class:`ImplicitPDController`), and the entity's
    ``kd * substep_dt`` armature fold runs under ``"euler"`` only. A ``kd`` stable through
    ``ImplicitPDController`` at a given substep can therefore diverge here.

    """

    def build(self) -> None:
        super().build()

        # The solver's flag: Genesis batches a heterogeneous entity's DOF info by itself in some scenes.
        self._batch_dofs_info = self.entity.has_batched_dofs_info

        # Cache PD gains, already shaped (num_envs, n_dofs) whatever batch_dofs_info says.
        self._kp = self.entity.get_dofs_kp(dofs_idx_local=self.genesis_dofs_idx)
        self._kd = self.entity.get_dofs_kd(dofs_idx_local=self.genesis_dofs_idx)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        if self._batch_dofs_info:
            # Re-read the per-env gains so gain randomization is picked up. Skipped when DOF
            # info is unbatched purely to save the read: every env shares one row there, so
            # the write would be a no-op — the shapes line up either way.
            self._kp[envs_idx] = self.entity.get_dofs_kp(dofs_idx_local=self.genesis_dofs_idx, envs_idx=envs_idx)
            self._kd[envs_idx] = self.entity.get_dofs_kd(dofs_idx_local=self.genesis_dofs_idx, envs_idx=envs_idx)

    def apply_actions(self) -> None:
        target = self._applied_command()
        dofs_vel = self.entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)
        pos_err = target - self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)

        # PD control
        ctrl_torque = self._kp * pos_err - self._kd * dofs_vel

        # Apply modifiers to the computed torques (e.g., backlash, friction, effort clip)
        if self._modifier is not None:
            ctrl_torque = self._modifier.modify_ctrl_torque(ctrl_torque, dofs_vel, pos_err=pos_err)

        self.verify_command_shape(ctrl_torque)
        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.genesis_dofs_idx)


@ACTION_TERM_REGISTRY.register()
class VelocityFeedforwardPDController(_JointPDControllerBase):
    r"""
    Explicit joint PD controller with velocity feedforward.

    Extends the explicit PD controller by accepting both target joint positions
    and target joint velocities as actions. The torque is computed as:

    .. math::
        \tau = k_p (q_{target} + q_{offset} - q_{current} + q_{motor\_offset})
              + k_d (-\dot{q}_{current} + \dot{q}_{target} \cdot r_{ff})

    where :math:`r_{ff}` is the velocity feedforward ratio. Post-processing
    (motor strength scaling, torque offset injection, effort clipping, etc.)
    is opt-in via the ``modifier`` parameter — see the composable modifiers in
    :mod:`eden.managers.modifiers.actions.actuators`.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: str | list[str]
        The names of the DOFs to control.
    reference_source: ReferenceSource
        Source of the reference offset added on top of the position action.
    scale: float | dict[str, float] | None
        The scale to apply to the position actions; ``None`` (the default) applies none.
    offset: float | dict[str, float]
        The offset to apply to the position actions.
    velocity_scale: float
        The scale to apply to the velocity actions.
    feed_forward_ratio: float
        The ratio for the velocity feedforward term.

    .. note::
        Requires ``sim_substeps=1`` so that PD torques are recomputed at every physics
        substep. Adjust ``decimation`` accordingly to maintain the desired control frequency.
    """

    velocity_scale: float = 1.0
    feed_forward_ratio: float = 1.0
    _supports_mimic_: ClassVar[bool] = False

    @property
    def action_dim(self) -> int:
        return self._n_dofs * 2

    def build(self) -> None:
        if self.reference_source == ReferenceSource.ACCUMULATE:
            # ACCUMULATE is honored only by controllers that use the base ``compute``;
            # this one overrides it, so accumulation would silently never happen.
            raise ValueError(
                f"{term_subject(self)}: reference_source='accumulate' is not supported "
                f"(its overridden compute() does not integrate the action)."
            )
        super().build()

        # Re-allocate _processed_action to hold both position and velocity
        self._processed_action = torch.zeros(self.num_envs, self._n_dofs * 2, device=self.device)

        # The solver's flag: Genesis batches a heterogeneous entity's DOF info by itself in some scenes.
        self._batch_dofs_info = self.entity.has_batched_dofs_info

        # Cache PD gains
        self._kp = self.entity.get_dofs_kp(dofs_idx_local=self.genesis_dofs_idx)
        self._kd = self.entity.get_dofs_kd(dofs_idx_local=self.genesis_dofs_idx)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        if self._batch_dofs_info:
            # Re-read the per-env gains so gain randomization is picked up. Skipped when DOF
            # info is unbatched purely to save the read: every env shares one row there, so
            # the write would be a no-op — the shapes line up either way.
            self._kp[envs_idx] = self.entity.get_dofs_kp(dofs_idx_local=self.genesis_dofs_idx, envs_idx=envs_idx)
            self._kd[envs_idx] = self.entity.get_dofs_kd(dofs_idx_local=self.genesis_dofs_idx, envs_idx=envs_idx)

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        n_dofs = self._n_dofs
        if self._delta_reference is not None:
            # Capture once per env step; constant across decimation substeps.
            self._delta_reference[:] = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        joint_pos = self._compute_pos_target(self._raw_action[:, :n_dofs])
        joint_vel = self._raw_action[:, n_dofs:] * self.velocity_scale

        # Apply the control-step modifiers to the position half (e.g., envelope clip); a delay acts
        # on the whole command per physics step, in `apply_actions`.
        if self._modifier is not None:
            joint_pos = self._modifier.modify_processed_action(joint_pos)

        self._processed_action[:, :n_dofs] = joint_pos
        self._processed_action[:, n_dofs:] = joint_vel

    def apply_actions(self) -> None:
        n_dofs = self._n_dofs

        command = self._applied_command()
        target_pos = command[:, :n_dofs]
        target_vel = command[:, n_dofs:]
        dofs_vel = self.entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)
        pos_err = target_pos - self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)

        ctrl_torque = self._kp * pos_err + self._kd * (target_vel * self.feed_forward_ratio - dofs_vel)

        # Apply modifier to ctrl torque (e.g., backlash, friction, effort clip)
        if self._modifier is not None:
            ctrl_torque = self._modifier.modify_ctrl_torque(ctrl_torque, dofs_vel, pos_err=pos_err)

        self.verify_command_shape(ctrl_torque)
        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.genesis_dofs_idx)


@ACTION_TERM_REGISTRY.register()
class ImplicitVelocityController(_JointPDControllerBase):
    """Joint velocity control through Genesis's built-in velocity mode."""

    # Override the parent's DEFAULT default — velocity controllers have no position reference.
    reference_source: ReferenceSource = ReferenceSource.ZERO
    _targets_position_: ClassVar[bool] = False

    def build(self) -> None:
        if self.reference_source != ReferenceSource.ZERO:
            raise ValueError(
                f"{term_subject(self)}: `reference_source='{self.reference_source}'` is not supported; "
                f"the action is interpreted as a target velocity."
            )
        super().build()

    def apply_actions(self) -> None:
        target = self._applied_command()
        self.verify_command_shape(target)
        self.entity.control_dofs_vel(target, dofs_idx_local=self.genesis_dofs_idx)


@ACTION_TERM_REGISTRY.register()
class ExplicitVelocityController(_JointPDControllerBase):
    r"""
    Explicit joint velocity controller using manual torque computation.

    The policy action is interpreted as a target joint velocity. Torque is computed
    as a proportional law on the velocity error:

    .. math::
        \tau = k_p (\dot{q}_{target} - \dot{q})

    This matches the behavior of MuJoCo's ``velocity`` actuator and the implicit
    counterpart :class:`ImplicitVelocityController`. Use this variant when an
    explicit, auditable torque is required (e.g., sim2real, modifier composition,
    domain randomization of the gain).

    The gain is derived from the DOF force range so it is automatically scaled to
    each joint's effort capacity:

    .. math::
        k_p = (F_{max} - F_{min}) \cdot \text{velocity\_gain}

    Parameters
    ----------
    entity_name : str
        The name of the entity to control.
    dofs_name : str | list[str]
        The names of the DOFs to control. Supports glob patterns.
    velocity_gain : float
        Scales the force range to produce :math:`k_p`.
    modifier : ActionModifierOptions | None
        Optional composable action modifier (e.g., effort clip, friction).

    .. note::
        Requires ``sim_substeps=1`` so that PD torques are recomputed at every physics
        substep. Adjust ``decimation`` accordingly to maintain the desired control frequency.

    .. note::
        ``reference_source`` must remain ``"zero"`` (this is a velocity
        controller). The inherited ``offset`` parameter operates in velocity
        units (rad/s).
    """

    # Override the parent's DEFAULT default — velocity controllers have no position reference.
    reference_source: ReferenceSource = ReferenceSource.ZERO
    velocity_gain: float = 0.25
    _targets_position_: ClassVar[bool] = False

    def _compute_gain(self, envs_idx=None) -> torch.Tensor:
        lower, upper = self.entity.get_dofs_force_range(dofs_idx_local=self.genesis_dofs_idx, envs_idx=envs_idx)
        return (upper - lower) * self.velocity_gain

    def build(self) -> None:
        if self.reference_source != ReferenceSource.ZERO:
            raise ValueError(
                f"{term_subject(self)}: `reference_source='{self.reference_source}'` is not supported; "
                f"the action is interpreted as a target velocity."
            )

        super().build()

        # The solver's flag: Genesis batches a heterogeneous entity's DOF info by itself in some scenes.
        self._batch_dofs_info = self.entity.has_batched_dofs_info

        self._kp = self._compute_gain()

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        if self._batch_dofs_info:
            self._kp[envs_idx] = self._compute_gain(envs_idx=envs_idx)

    def apply_actions(self) -> None:
        dofs_vel = self.entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)
        ctrl_torque = self._kp * (self._applied_command() - dofs_vel)

        if self._modifier is not None:
            ctrl_torque = self._modifier.modify_ctrl_torque(ctrl_torque, dofs_vel)

        self.verify_command_shape(ctrl_torque)
        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.genesis_dofs_idx)


@ACTION_TERM_REGISTRY.register()
class NullJointAction(ActionTerm):
    """
    Virtually fix the DoFs to the default positions.

    Consumes no policy output: the DOFs are held at ``default_dofs_pos`` for the whole episode,
    so there is nothing for the policy to decide and :attr:`action_dim` is 0. The term still
    *commands* its DOFs — ``processed_action`` stays ``(num_envs, num_dofs)`` and reaches
    hardware through the usual deployment map — it just draws none of the action vector.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The names of the DOFs to control.
    """

    def __init__(
        self,
        env: EnvBase,
        options: ActionTermOptions,
    ):
        super().__init__(env=env, options=options)
        self._processed_action: torch.Tensor | None = None
        self._raw_action: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        """No policy output: this term pins its DOFs rather than steering them.

        The base class returns the DOF count, which would have ``ActionManager`` reserve a slice of
        the action vector per pinned DOF — columns the policy would learn to emit and ``compute``
        would discard, since it is a no-op. ``num_dofs`` is left alone: the term commands those DOFs,
        it just takes no input for them. That is also what keeps this term out of
        ``ActionManager._reject_no_op_terms``, which rejects any term whose ``action_dim`` *and*
        ``num_dofs`` are both zero — a `dofs_name`-less term that would command nothing.
        """
        return 0

    def build(self) -> None:
        super().build()

        # Selected by name on the *entity*: the term's DOFs are columns of the entity's `dofs_name`, not its
        # first `num_dofs` — a term controlling any subset but `joint_0` onwards would otherwise hold the wrong
        # joints' defaults. The getter's `(num_envs, n)` is what the command below is written to, whether the
        # stored tensor has one row or one per env.
        #
        # `_raw_action` / `_prev_action` are left as `super().build()` allocated them —
        # `(num_envs, 0)`, matching `action_dim` — so this term's `action` / `prev_action`
        # observations report the empty slice it actually receives rather than a width of
        # permanently-zero columns.
        self._processed_action = self.entity.get_default_dofs_pos(self.dofs_name)

    def compute(self, actions: torch.Tensor) -> None:
        pass

    def apply_actions(self) -> None:
        # The term's DOFs may be a subset of the entity's, or ordered differently: pass its own index.
        self.verify_command_shape(self._processed_action)
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)


def is_joint_position_controller(term: type | ActionTerm) -> bool:
    """Whether ``term`` (a term class or instance) is a joint PD controller that commands positions.

    True for :class:`ImplicitPDController`, :class:`ExplicitPDController`,
    :class:`VelocityFeedforwardPDController` and their subclasses; False for the velocity
    controllers, which share the PD base for its scale/offset plumbing but command a velocity, and
    for every other action term. This is the set a robot's position action scale, and a joint-zero
    offset such as :class:`~eden.managers.terms.events.domain_rand.RandomizeDofsPosOffset`, apply to.

    Parameters
    ----------
    term: type | ActionTerm
        An action term class, or a built term instance.
    """
    cls = term if isinstance(term, type) else type(term)
    return issubclass(cls, _JointPDControllerBase) and cls._targets_position_


# TODO: add JointTorqueController, ActuatorNetMLPController, ActuatorNetLSTMController
