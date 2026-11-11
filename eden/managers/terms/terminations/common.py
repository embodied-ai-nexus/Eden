"""Common termination terms (timeout, success/failure, illegal contact, limits)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_manager, soft_dof_pos_violation
from eden.utils.sample import apply_probability


if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.entities.base import Entity


@TERMINATION_TERM_REGISTRY.register()
def time_out(env: RLEnvBase) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


@TERMINATION_TERM_REGISTRY.register()
def on_success(env: RLEnvBase) -> torch.Tensor:
    return env.metric_manager.success_buf


@TERMINATION_TERM_REGISTRY.register()
def on_failure(env: RLEnvBase) -> torch.Tensor:
    return ~env.metric_manager.success_buf


@TERMINATION_TERM_REGISTRY.register()
class OnMetricSuccess(TerminationTerm):
    """Terminate the envs in which the named metric term reports success this step.

    :meth:`build` runs the manager's own current-step success check once, so a missing metric
    manager, an unset or unknown ``metric_name`` (the error lists the active metrics), a
    non-INTERVAL metric and a ``contributes_to_success=False`` metric all raise at env
    construction. To end the episode on a metric that defines no success, use :class:`MetricAbove`.
    """

    # Was the function term ``on_metric_success(env, *, metric_name)``; a config saved then carries it in
    # ``params``, which load as fields.
    _function_era_params = frozenset({"metric_name"})

    metric_name: str = ""

    def build(self) -> None:
        metric_manager = resolve_metric_manager(self._env, term="on_metric_success", metric_name=self.metric_name)
        metric_manager.is_current_step_success(self.metric_name)

    def compute(self) -> torch.Tensor:
        return self._env.metric_manager.is_current_step_success(self.metric_name)


@TERMINATION_TERM_REGISTRY.register()
class MetricAbove(TerminationTerm):
    """Terminate the envs in which the named metric's cached value is strictly above ``threshold``.

    Reads ``metric_manager.get_value(metric_name)``: the current-step value, or — for a metric
    declared ``is_cumulative=True`` — the running episode sum (reset per-env on episode reset),
    so on a cumulative metric ``threshold`` bounds the episode total, not one step's value.

    Unlike ``on_metric_success`` this never consults the metric's success check, so it accepts
    a metric with ``contributes_to_success=False`` — e.g. an arm–table collision metric that
    must end the episode without being part of the task's success accounting.

    A missing metric manager or an unset or unknown ``metric_name`` (the error lists the active
    metrics) raises in :meth:`build`, i.e. at env construction.
    """

    metric_name: str = ""
    threshold: float = 0.0

    def build(self) -> None:
        resolve_metric_manager(self._env, term="metric_above", metric_name=self.metric_name)

    def compute(self) -> torch.Tensor:
        return self._env.metric_manager.get_value(self.metric_name) > self.threshold


@TERMINATION_TERM_REGISTRY.register()
def illegal_contact(env: RLEnvBase, *, entity_name: str, illegal_contact_entity_names: list[str]) -> torch.Tensor:
    entity: Entity = env.get_entity(entity_name)
    contacts = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for illegal_contact_entity_name in illegal_contact_entity_names:
        illegal_contact_entity: Entity = env.get_entity(illegal_contact_entity_name)
        contacts |= entity.get_contacts(with_entity=illegal_contact_entity)["valid_mask"].any(dim=-1)
    return contacts


@TERMINATION_TERM_REGISTRY.register()
class IllegalLinkContact(TerminationTerm):
    """Terminate when any of the named links carries a net contact force above ``force_threshold``.

    The link-level sibling of :func:`illegal_contact`, and the reason both exist: that term asks
    *which entities touched*, this one asks *which links of one entity took load, and how hard*. The
    knee-and-torso fall detector every locomotion task wants is this one — the entity-level term
    cannot express it, because a walking robot's feet are in contact with exactly the same entity
    (the ground or the terrain) as its knees. It mirrors Isaac Lab / mjlab's
    ``mdp.illegal_contact(body_names, threshold)``, minus their force *history*: Eden reads the net
    force from the most recent ``scene.step()``, so a contact shorter than one control step can slip
    between two reads.

    Same reading as :class:`~eden.managers.terms.rewards.common.UndesiredContacts` — that term
    counts the offending links so a weight can price them, this one ends the episode on the first —
    and the two resolve ``links_name`` identically, so a config can penalise and terminate on one
    link set without the two drifting apart.

    Parameters
    ----------
    entity_name:
        Entity whose links are checked. Must be a rigid entity (``get_links_net_contact_force``).
    links_name:
        :func:`~eden.utils.string.resolve_matching_names`-compatible patterns selecting the links
        that may not take load — e.g. ``[".*knee.*", "torso.*"]``. Required: an empty list would
        make the term a constant ``False``, so it raises in :meth:`build` listing the entity's links.
    force_threshold:
        Net contact-force magnitude, in newtons, above which a link counts as illegally touching.
        The default of ``1.0`` N (Isaac Lab's) rejects numerical noise while catching any real load;
        raise it on a heavy robot whose links rest against each other.
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    force_threshold: float = 1.0

    def build(self) -> None:
        # Touch the entity for fail-fast but cache only the derived indices, so a deployment
        # hot-swap of env.entities[name] is honoured (see resolve_entity). The indices are
        # name-resolved idx_local, matching the order get_links_net_contact_force returns.
        entity = self.resolve_entity(self.entity_name)
        self._link_idx = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)

    def compute(self) -> torch.Tensor:
        forces = self.resolve_entity(self.entity_name).get_links_net_contact_force()  # (N, n_links, 3)
        magnitudes = torch.norm(forces[:, self._link_idx], dim=-1)  # (N, K)
        return (magnitudes > self.force_threshold).any(dim=-1)


@TERMINATION_TERM_REGISTRY.register()
def dofs_pos_limit_exceeded(
    env: RLEnvBase,
    *,
    entity_name: str,
    probability: float = 1.0,
) -> torch.Tensor:
    """Terminate when any DOF position is outside the entity's soft position limits.

    ``probability`` gates the termination globally per call (one Bernoulli draw),
    leaving probabilistic slack for policies that occasionally bump the limits.
    Set to 1.0 for hard termination.
    """
    entity: Entity = env.get_entity(entity_name)
    soft = entity.soft_dofs_pos_limits
    if soft is None:
        # Non-articulated or unbuilt entity: nothing to violate.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dofs_pos = entity.get_dofs_pos()
    violation = torch.sum(soft_dof_pos_violation(dofs_pos, soft), dim=1) > 0.0
    return apply_probability(violation, probability)


@TERMINATION_TERM_REGISTRY.register()
def dofs_vel_limit_exceeded(
    env: RLEnvBase,
    *,
    entity_name: str,
    max_vel: float,
    probability: float = 1.0,
) -> torch.Tensor:
    """Terminate when any DOF velocity exceeds ``max_vel`` in magnitude.

    ``probability`` gates the termination globally per call. Set to 1.0 for hard
    termination.
    """
    entity: Entity = env.get_entity(entity_name)
    dofs_vel = entity.get_dofs_vel()
    violation = (dofs_vel.abs() > max_vel).any(dim=1)
    return apply_probability(violation, probability)


@TERMINATION_TERM_REGISTRY.register()
def cumulative_reward_below_threshold(
    env: RLEnvBase,
    *,
    reward_term: str,
    per_step_threshold: float,
    grace_steps: int = 0,
    check_interval: int = 1,
) -> torch.Tensor:
    """Terminate envs whose running raw reward for ``reward_term`` falls below a threshold.

    The threshold is ``per_step_threshold * env.dt * completed_steps``.

    Mirrors dexmachina's early-reset curriculum trigger
    (``dexmachina/envs/base_env.py:712-750``). Reads the weight-free,
    dt-scaled accumulation via :meth:`RewardManager.get_episode_sum` so the
    threshold stays stable when reward curricula mutate ``term.weight`` at
    runtime.

    Mental model: average reward per completed step ≈
    ``cum / (dt * (episode_length_buf - 1))``. The term fires when that
    ratio drops below ``per_step_threshold``, optionally rounded down to
    multiples of ``check_interval`` steps.

    Parameters
    ----------
    env : RLEnvBase
        The environment instance.
    reward_term : str
        Name of a term registered with the reward manager. Raises
        ``KeyError`` (with the active term list) if missing.
    per_step_threshold : float
        Floor on raw per-step reward (unweighted ``compute()`` output).
        Compared as ``cum < per_step_threshold * env.dt * completed_steps``.
        Same units as dexmachina's ``--early_reset_threshold`` flag (paper
        default ``0.5``); no manual ``* dt`` rescale at config time.
    grace_steps : int
        Suppress the gate while ``episode_length_buf <= grace_steps``.
        Default ``0`` matches dexmachina (its ``interval=0`` rung is
        implicitly graceful — the natural ``cum == 0`` initial state
        satisfies ``cum < 0`` vacuously).
    check_interval : int
        ``1`` (default) → continuous per-completed-step check.
        ``> 1`` → dexmachina interval-ladder: round elapsed-steps down
        to ``check_interval`` multiples before applying the threshold.
        Pass ``5`` for ``early_reset_interval=5``, ``20`` for the
        auxiliary imitation/contact/BC triggers.
    """
    cum = env.reward_manager.get_episode_sum(reward_term, raw=True)
    # _episode_raw_sums covers steps 1..t-1 when termination evaluates at step t,
    # so the elapsed multiplier is (episode_length_buf - 1) clamped at zero.
    completed = (env.episode_length_buf - 1).clamp(min=0)
    if check_interval > 1:
        # Match dexmachina's strict `>` rung compare: largest rung strictly below N
        # equals ((N - 1) // ci) * ci. Naive (N // ci) * ci would round one rung
        # too high on exact multiples (step 20 vs rung 20 instead of 15).
        completed = (completed // check_interval) * check_interval
    floor = per_step_threshold * env.dt * completed.float()
    return (cum < floor) & (env.episode_length_buf > grace_steps)


@TERMINATION_TERM_REGISTRY.register()
def dofs_torque_limit_exceeded(
    env: RLEnvBase,
    *,
    entity_name: str,
    soft_ratio: float = 1.0,
    probability: float = 1.0,
) -> torch.Tensor:
    """Terminate when any applied DOF torque exceeds ``soft_ratio * default_dofs_force_limits``.

    Uses the entity's per-DOF effort limit (the configured ``default_dofs_force_limits``, else actuator-spec
    ``EFFORT_LIMIT``, else the asset's force range) and triggers when ``|tau| > soft_ratio * limit`` for any
    joint. ``probability`` gates the termination globally per call.
    """
    entity: Entity = env.get_entity(entity_name)
    torques = entity.get_dofs_force()
    limit = entity.default_dofs_force_limits
    if limit is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # default_dofs_force_limits is shaped (num_envs, num_dofs) when batched, else (1, num_dofs).
    threshold = soft_ratio * limit
    violation = (torques.abs() > threshold).any(dim=1)
    return apply_probability(violation, probability)
