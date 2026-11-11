"""Privileged observation terms exposing per-env domain-randomized dynamics.

These read the *realized* randomized physical parameters relative to their
nominal defaults, so an asymmetric (privileged) critic can condition on the
domain randomization the actor cannot observe — the standard sim-to-real
privileged-critic technique. The actor sees only proprioception; the critic
additionally sees terms like these.

Each term reads back exactly one ``domain_rand`` event:

===========================  =======================================================================
term                         event it reads back
===========================  =======================================================================
:class:`GainScale`           :class:`~eden.managers.terms.events.domain_rand.RandomizeKpKdGains`
:class:`MotorStrengthScale`  :class:`~eden.managers.terms.events.domain_rand.RandomizeMotorStrength`
:class:`AddedMass`           :class:`~eden.managers.terms.events.domain_rand.RandomizeMassShift`
:class:`FootFriction`        :class:`~eden.managers.terms.events.domain_rand.RandomizeFrictionRatio`
:class:`ComOffset`           :class:`~eden.managers.terms.events.domain_rand.RandomizeComShift`
===========================  =======================================================================

Convention: the shift/deviation terms are ``0`` at nominal (``GainScale``,
``MotorStrengthScale``, ``AddedMass``, ``ComOffset``); ``FootFriction`` is a
ratio, ``1.0`` at nominal.

Inspired by TWIST2's ``critic_extras`` (``critic_motor_scales`` /
``critic_added_mass`` / ``critic_foot_friction`` / ``critic_base_com_offset``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import eden as en
from eden.managers import OBSERVATION_TERM_REGISTRY
from eden.managers.observation_manager import ObservationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


@OBSERVATION_TERM_REGISTRY.register()
class GainScale(ObservationTerm):
    """Per-DOF relative deviation of the realized PD gains from their defaults.

    Returns ``concat([kp/kp0 - 1, kd/kd0 - 1])`` over the entity's actuated DOFs
    (in ``dofs_name`` order); ``0`` means nominal, so it is all zeros when
    there is no gain randomization. Pairs with
    :class:`~eden.managers.terms.events.domain_rand.RandomizeKpKdGains`.
    Output shape ``(num_envs, 2 * num_dofs)``.

    Notes
    -----
    Reads the gains held on the *entity* (``get_dofs_kp`` / ``get_dofs_kd``), so
    it only reflects gain randomization written to the entity — i.e.
    ``RandomizeKpKdGains`` with ``action_term_name=None`` (the default, Genesis
    built-in PD path). ``RandomizeKpKdGains`` with ``action_term_name`` set
    writes the PD-controller's ``_kp`` / ``_kd`` caches instead and never touches
    the entity, so this term would read all zeros; :meth:`build` raises if such
    an event targets the same entity. (The controller mirrors the entity gains
    into its caches on every action reset, but that mirror is one-way
    entity→controller, so it does not carry controller-side randomization back
    to the entity.) Per-env gains require ``batch_dofs_info=True`` in
    ``env_options``; otherwise the gains are shared across envs and the term is
    uniformly zero (still valid, just uninformative).

    This is the *gain* deviation, distinct from
    :class:`MotorStrengthScale` (the actuator torque-scale modifier).

    Parameters
    ----------
    entity_name : str
        Name of the articulated entity whose actuator gains to read.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        # Baselines are build-time constants (the nominal per-DOF gains). Cache
        # them plus the derived zero-masks / safe denominators so compute() does
        # no per-step allocation. Clone so the cache never aliases
        # ``default_dofs_kp`` (``as_tensor`` returns the same tensor when it is
        # already float32 on-device).
        self._kp0 = self._ensure_2d(entity.default_dofs_kp).clone()
        self._kd0 = self._ensure_2d(entity.default_dofs_kd).clone()
        self._kp_zero, self._kp_safe = self._zero_mask_and_safe(self._kp0)
        self._kd_zero, self._kd_safe = self._zero_mask_and_safe(self._kd0)
        self._guard_conflicting_gain_events()
        self._warn_zero_baseline_dofs(entity)

    def _ensure_2d(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize ``(ndof,)`` / ``(1, ndof)`` / ``(num_envs, ndof)`` to 2D."""
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        return t if t.ndim == 2 else t.unsqueeze(0)

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        # The getters already reach the full env count, which fixes the output shape; the
        # ``(1, ndof)`` denominators then broadcast against them.
        kp = entity.get_dofs_kp()
        kd = entity.get_dofs_kd()
        dev = torch.cat(
            [self._rel_dev(kp, self._kp_zero, self._kp_safe), self._rel_dev(kd, self._kd_zero, self._kd_safe)],
            dim=-1,
        )
        return dev.float()

    @staticmethod
    def _zero_mask_and_safe(baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-baseline mask and a denominator with those zeros replaced by ``1``.

        A DOF with no nominal gain (e.g. a passive/velocity-controlled joint
        folded into the entity DOFs) has no gain randomization to report, so it
        must read as nominal instead of producing ``inf`` / ``nan``. Precomputed
        once in :meth:`build` since the baseline is a build-time constant.
        """
        zero = baseline == 0
        return zero, baseline.masked_fill(zero, 1.0)

    @staticmethod
    def _rel_dev(value: torch.Tensor, zero_mask: torch.Tensor, safe_baseline: torch.Tensor) -> torch.Tensor:
        """Relative deviation ``value / baseline - 1``, reported as ``0`` where the baseline is zero."""
        return (value / safe_baseline - 1.0).masked_fill(zero_mask, 0.0)

    def _guard_conflicting_gain_events(self) -> None:
        """Raise if a ``RandomizeKpKdGains`` event on this entity writes the controller, not the entity."""
        from eden.managers.terms.events.domain_rand import RandomizeKpKdGains

        # EventManager groups its term instances by mode in ``_mode_terms`` (it is
        # hierarchical and has no flat term API), so flatten across modes.
        event_manager = self._env.event_manager
        all_events = [term for mode_terms in event_manager._mode_terms.values() for term in mode_terms]
        conflicting = [
            type(term).__name__
            for term in all_events
            if isinstance(term, RandomizeKpKdGains)
            and getattr(term, "action_term_name", None) is not None
            and getattr(term, "entity_name", None) == self.entity_name
        ]
        if conflicting:
            raise ValueError(
                f"{term_subject(self)} reads gains from the entity, but "
                f"event term(s) {conflicting} randomize gains with action_term_name set — that path writes the "
                "PD-controller caches, never the entity, so this term would be silently all-zeros. Set "
                "action_term_name=None on those events (Genesis built-in PD path) so the randomization reaches "
                "the entity, or drop this observation term."
            )

    def _warn_zero_baseline_dofs(self, entity: RigidEntity) -> None:
        """Warn about DOFs whose zero nominal gain masks their deviation to zero."""
        zero_dofs = (self._kp_zero | self._kd_zero).any(dim=0)
        if bool(zero_dofs.any()):
            names = [name for name, is_zero in zip(entity.dofs_name, zero_dofs.tolist()) if is_zero]
            en.logger.warning(
                f"{term_subject(self)}: DOFs {names} have a zero nominal kp "
                "or kd baseline; their gain deviation is forced to 0, hiding any real randomization on them "
                "(likely a gains-dict omission)."
            )


@OBSERVATION_TERM_REGISTRY.register()
class MotorStrengthScale(ObservationTerm):
    """Per-DOF deviation of the realized actuator torque-scale from its nominal.

    Returns ``motor_strength - motor_strength_baseline`` over the controller's
    DOFs (``0`` means the configured nominal strength), read from the
    :class:`~eden.managers.modifiers.actions.actuators.MotorStrength` action
    modifier that
    :class:`~eden.managers.terms.events.domain_rand.RandomizeMotorStrength`
    writes. All zeros without motor-strength randomization. Output shape
    ``(num_envs, num_dofs)``.

    The baseline is the modifier's configured strength row (usually ``1.0`` but
    honoring a non-unity ``motor_strength`` config), so the term stays ``0`` at
    the deterministic nominal regardless of that configured value.

    This is the actuator *torque-scale* (a multiplier on the PD output torque),
    distinct from :class:`GainScale` (the kp/kd gain deviation).

    Parameters
    ----------
    action_term_name : str
        Name of the PD-controller action term hosting the ``MotorStrength``
        modifier (directly or inside a ``Compose`` chain).
    """

    action_term_name: str = "dofs_pos_controller"

    def build(self) -> None:
        from eden.managers.modifiers.actions.actuators import MotorStrength
        from eden.managers.terms.events.domain_rand import _require_modifier

        term = self._env.action_manager.get_term(self.action_term_name)
        self._modifier = _require_modifier(term, MotorStrength, type(self).__name__, self.action_term_name)

    def compute(self) -> torch.Tensor:
        return (self._modifier.motor_strength_realized - self._modifier.motor_strength_baseline).float()


class _LinkDrTerm(ObservationTerm):
    """Shared base: resolve ``links_name`` to local link indices at build."""

    entity_name: str = "robot"
    links_name: list[str] = []

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        # The empty-selection check lives in resolve_link_idx, which also lists the entity's links.
        self._ls_idx = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)


@OBSERVATION_TERM_REGISTRY.register()
class AddedMass(_LinkDrTerm):
    """Per-env mass shift (kg) of the named links vs nominal — 0 without mass DR.

    Ports TWIST2's ``critic_added_mass``. Pairs with
    :class:`~eden.managers.terms.events.domain_rand.RandomizeMassShift`
    (or :class:`~eden.managers.terms.events.domain_rand.RandomizeLinkMassScale`).
    Output shape ``(num_envs, len(links_name))``.
    """

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_mass_shift(self._ls_idx).float()


@OBSERVATION_TERM_REGISTRY.register()
class FootFriction(_LinkDrTerm):
    """Mean per-env friction ratio over the named links — 1.0 without friction DR.

    Ports TWIST2's ``critic_foot_friction``. Pairs with
    :class:`~eden.managers.terms.events.domain_rand.RandomizeFrictionRatio`.
    Output shape ``(num_envs, 1)``.
    """

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_friction_ratio(self._ls_idx).mean(dim=1, keepdim=True).float()


@OBSERVATION_TERM_REGISTRY.register()
class ComOffset(_LinkDrTerm):
    """Per-env COM shift (m) of the named links vs nominal — 0 without COM DR.

    Ports TWIST2's ``critic_base_com_offset``. Pairs with
    :class:`~eden.managers.terms.events.domain_rand.RandomizeComShift`.
    Output shape ``(num_envs, 3 * len(links_name))``.
    """

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_COM_shift(self._ls_idx).flatten(start_dim=1).float()
