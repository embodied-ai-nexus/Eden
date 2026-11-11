"""Common reward/penalty terms (action rate/magnitude, joint limits, torques, contacts, metric and success rewards)."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_manager, soft_dof_pos_violation

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


def _compute_action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(action - prev_action), dim=1)


@REWARD_TERM_REGISTRY.register()
def action_rate_l2(
    env: EnvBase,
    *,
    action_term_name: str | None = None,
) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if action_term_name is not None:
        action_term = env.action_manager.get_term(action_term_name)
        return _compute_action_rate_l2(action_term.action, action_term.prev_action)
    return _compute_action_rate_l2(env.action_manager.action, env.action_manager.prev_action)


def _compute_action_l2(action: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(action), dim=1)


@REWARD_TERM_REGISTRY.register()
def action_l2(
    env: EnvBase,
    *,
    action_term_name: str | None = None,
) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    if action_term_name is not None:
        action_term = env.action_manager.get_term(action_term_name)
        return _compute_action_l2(action_term.action)
    return _compute_action_l2(env.action_manager.action)


def _compute_dofs_pos_limits(dofs_pos: torch.Tensor, soft_limits: torch.Tensor) -> torch.Tensor:
    return torch.sum(soft_dof_pos_violation(dofs_pos, soft_limits), dim=1)


@REWARD_TERM_REGISTRY.register()
def dofs_pos_limits(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    entity: Entity = env.get_entity(entity_name)
    dofs_pos = entity.get_dofs_pos()
    return _compute_dofs_pos_limits(dofs_pos, entity.soft_dofs_pos_limits)


def _compute_dofs_vel_limits(dofs_vel: torch.Tensor, max_vel: float) -> torch.Tensor:
    excess = (dofs_vel.abs() - max_vel).clamp_min(0.0)
    return (excess**2).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
def dofs_vel_limits(
    env: EnvBase,
    *,
    entity_name: str,
    max_vel: float | torch.Tensor,
) -> torch.Tensor:
    """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

    Penalizes only the amount by which |v| exceeds max_vel. Returns the (non-negative) squared L2
    norm of the excess velocities; use a negative weight to penalise.
    """
    entity: Entity = env.get_entity(entity_name)
    dofs_vel = entity.get_dofs_vel()
    return _compute_dofs_vel_limits(dofs_vel, max_vel)


@REWARD_TERM_REGISTRY.register()
def self_collision_cost(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize self collisions."""
    entity: Entity = env.get_entity(entity_name)
    contacts = entity.get_contacts(with_entity=entity, exclude_self_contact=False)
    return contacts["valid_mask"].sum(dim=1)


@REWARD_TERM_REGISTRY.register()
class UndesiredContacts(RewardTerm):
    """Penalty count of links whose net contact-force magnitude exceeds ``threshold``.

    ``links_name`` accepts the same regex / glob patterns as
    :func:`eden.utils.string.resolve_matching_names`. Use a negative reward
    weight to penalise (e.g. -0.1 per offending link).
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    threshold: float = 1.0

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        # Name-resolved link idx_local (matches the internal order get_links_net_contact_force
        # returns), so this stays correct across Genesis link reorders (BFS->DFS in 1.2.0).
        self._link_idx = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        forces = entity.get_links_net_contact_force()  # (N, n_links, 3)
        magnitudes = torch.norm(forces[:, self._link_idx], dim=-1)  # (N, K)
        return (magnitudes > self.threshold).float().sum(dim=-1)


def _compute_action_rate_l2_smooth(
    action: torch.Tensor, prev_action: torch.Tensor, prev_prev_action: torch.Tensor
) -> torch.Tensor:
    c1 = torch.sum(torch.square(action - prev_action), dim=1)
    c2 = torch.sum(torch.square(action - 2 * prev_action + prev_prev_action), dim=1)
    return c1 + c2


@REWARD_TERM_REGISTRY.register()
class ActionRateL2Smooth(RewardTerm):
    """Penalize the 1st and 2nd derivatives of the actions using an L2 squared kernel.

    The 2nd-derivative term needs the action from *two* steps ago. This term owns that buffer
    (the previous step's ``prev_action``), zeroed per env on reset.
    """

    action_term_name: str | None = None

    def build(self) -> None:
        # The action manager is built before the reward manager (``EnvBase._manager_order``), so
        # both the handle and the action width are known here: an unknown ``action_term_name``
        # raises at env construction, and the buffer is sized once against the real shape instead
        # of on the first ``compute()``.
        _, prev_action = self._action_and_prev()
        # ``inference_mode(False)`` states the invariant rather than inheriting it from whoever
        # called ``env.build()``: the buffer must be a *normal* tensor, because ``compute()`` runs
        # under ``RewardManager.compute``'s ``torch.inference_mode`` while ``reset()`` may be
        # called outside it (`eden smoke`'s plain step loop), and mutating an inference tensor
        # from a normal path raises.
        with torch.inference_mode(False):
            self._prev_prev_action: torch.Tensor = torch.zeros_like(prev_action)

    def _action_and_prev(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_term_name is not None:
            term = self._env.action_manager.get_term(self.action_term_name)
            return term.action, term.prev_action
        action_manager = self._env.action_manager
        return action_manager.action, action_manager.prev_action

    def compute(self) -> torch.Tensor:
        action, prev_action = self._action_and_prev()
        out = _compute_action_rate_l2_smooth(action, prev_action, self._prev_prev_action)
        # This step's ``prev_action`` is two-steps-old next step.
        self._prev_prev_action.copy_(prev_action)
        return out

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[envs_idx] = 0.0


def _compute_dofs_torques(torques: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.square(torques), dim=1) + 1e-8) + torch.sum(torch.abs(torques), dim=1)


@REWARD_TERM_REGISTRY.register()
def dofs_torques(
    env: EnvBase,
    *,
    entity_name: str,
) -> torch.Tensor:
    """Penalize joint torques: sqrt(sum(tau^2)) + sum(|tau|)."""
    entity: Entity = env.get_entity(entity_name)
    torques = entity.get_dofs_force()
    return _compute_dofs_torques(torques)


def _compute_dofs_acc_l2(dvel: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(dvel), dim=1)


@REWARD_TERM_REGISTRY.register()
class DofsAccL2(RewardTerm):
    """Penalize joint accelerations approximated as velocity differences.

    Computes ``sum((vel - prev_vel)^2)`` where ``prev_vel`` is the joint velocity
    from the previous step; this term owns that buffer, zeroed per env on reset. The config
    weight should account for the ``1/dt^2`` factor compared to raw simulator accelerations.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        # Entities exist before the reward manager is built, so the buffer is
        # allocated here against the live joint-velocity shape.
        entity: Entity = self.resolve_entity(self.entity_name)
        self._prev_dofs_vel = torch.zeros_like(entity.get_dofs_vel())

    def compute(self) -> torch.Tensor:
        entity: Entity = self.resolve_entity(self.entity_name)
        dofs_vel = entity.get_dofs_vel()
        out = _compute_dofs_acc_l2(dofs_vel - self._prev_dofs_vel)
        self._prev_dofs_vel.copy_(dofs_vel)
        return out

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._prev_dofs_vel.zero_()
        else:
            self._prev_dofs_vel[envs_idx] = 0.0


@REWARD_TERM_REGISTRY.register()
def similar_to_default(env: EnvBase, *, entity_name: str) -> torch.Tensor:
    """Penalize joint poses far away from the entity's default pose."""
    robot = env.get_entity(entity_name)
    return torch.sum(torch.abs(robot.get_dofs_pos() - robot.default_dofs_pos), dim=1)


@REWARD_TERM_REGISTRY.register()
class MetricReward(RewardTerm):
    """Use a :class:`~eden.managers.metric_manager.MetricTerm`'s cached value as the reward.

    Reads the per-env value the metric manager already computed this step
    (``metric_manager`` runs before the reward manager in ``env.step()``), so a quantity
    shared across reward / observation / termination terms is computed **once** as a
    metric and pulled here by name — instead of being recomputed in every term or stashed
    on the ``env`` object. The configured reward ``weight`` scales it as usual (use a
    negative weight for a penalty).

    Rejects a *cumulative* metric by default: its value is the running episode sum, so using
    it directly as a per-step reward re-integrates that sum and balloons over the episode —
    almost never intended. Pass ``allow_cumulative=True`` to opt in if you really want the
    running sum as the reward.

    A missing metric manager, an unset or unknown ``metric_name`` (the error lists the active
    metrics) and a rejected cumulative metric all raise in :meth:`build`, i.e. at env construction.
    """

    # Was the function term ``metric_reward(env, *, metric_name, allow_cumulative=False)``; a config saved
    # then carries these in ``params``, which load as fields.
    _function_era_params = frozenset({"metric_name", "allow_cumulative"})

    metric_name: str = ""
    allow_cumulative: bool = False

    def build(self) -> None:
        metric_manager = resolve_metric_manager(self._env, term="metric_reward", metric_name=self.metric_name)
        if not self.allow_cumulative and metric_manager.is_cumulative(self.metric_name):
            raise ValueError(
                f"metric_reward(metric_name='{self.metric_name}') pulls a cumulative metric, whose value is the "
                f"running episode sum — using it as a per-step reward integrates that sum again and is almost never "
                f"intended. Use a non-cumulative metric, or pass allow_cumulative=True to opt into the running sum as "
                f"the reward."
            )

    def compute(self) -> torch.Tensor:
        return self._env.metric_manager.get_value(self.metric_name)


@REWARD_TERM_REGISTRY.register()
class TaskSuccess(RewardTerm):
    """1 on a step where every success metric holds.

    The bonus for finishing, the reward counterpart of the ``on_success`` termination. Dense progress terms pay
    little for the last few percent of a task (the last cells of a stain, the final centimetres of a placement),
    so without it a policy settles just short of the success threshold. Pays ``metric_manager.success_buf``, and
    raises in :meth:`build` when the task declares no per-step (INTERVAL) metric that contributes to success,
    since that buffer then never turns True.
    """

    def build(self) -> None:
        metric_manager = self._env.metric_manager
        if metric_manager is None or metric_manager.num_step_success_terms == 0:
            raise ValueError(
                "task_success rewards the steps where every success metric holds, but this task declares no "
                "INTERVAL metric that contributes to success (no metric_options, or every metric is RESET-mode "
                "or has contributes_to_success=False). Declare the task's success metrics, or reward a metric's "
                "own value with metric_reward."
            )

    def compute(self) -> torch.Tensor:
        return self._env.metric_manager.success_buf.float()
