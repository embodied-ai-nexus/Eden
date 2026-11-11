"""Reward components of AssistMimic's help-up task, computed once per step and shared by both agents' rewards.

``HumanoidImInterxHelpUp._compute_reward`` gives each humanoid
``imitation + hand (caregiver only) + recipient head height - power`` and then mixes the caregiver's reward
with its partner's (``caregiver = 0.5 * self + 0.5 * recipient``, ``recipient = self``). Here each component
is a metric; the config's two reward groups weight them into those two per-agent rewards.
"""

from __future__ import annotations

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.tasks.assistmimic.custom_terms.commands import ROLES
from eden.tasks.assistmimic.custom_terms.observations import body_state, wrist_hand_idx
from eden.tasks.assistmimic.custom_terms.phc_math import imitation_reward
from eden.utils.string import term_subject

#: Recipient joints the caregiver's wrists are rewarded for approaching (``_compute_hand_distance_rewards``).
HAND_TARGET_BODIES = [
    "Torso", "Spine", "Chest", "Neck", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Knee",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Knee",
]  # fmt: skip


class _ShareOnly(MetricTerm):
    contributes_to_success: bool = False


class _DofsMetric(_ShareOnly):
    """A metric over ``dofs_name`` of ``entity_name``, resolved once in ``build()``."""

    entity_name: str = "caregiver"
    dofs_name: list[str] = []

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if not self.dofs_name:
            raise ValueError(
                f"{term_subject(self)}: dofs_name is empty, so the metric would read 0 "
                "every step. Pass the actuated DOFs, e.g. dofs_name=SMPL_DOF_NAMES "
                f"(eden.options.robots.smpl_humanoid); the entity has {entity.dofs_name[:6]}..."
            )
        _, idx = entity.find_named_dofs_idx_local(self.dofs_name, name_scope=entity.dofs_name, preserve_order=True)
        self._dofs_idx = torch.as_tensor(idx, device=self.device)


@METRIC_TERM_REGISTRY.register()
class PhcImitationReward(_ShareOnly):
    """PHC's imitation reward of one agent against its reference track, in ``[0, 1]``.

    Parameters
    ----------
    entity_name : str
        The agent.
    command_name : str
        The :class:`~eden.tasks.assistmimic.custom_terms.commands.AssistMimicPairMotion` term.
    role : str
        ``"caregiver"`` or ``"recipient"``.
    """

    entity_name: str = "caregiver"
    command_name: str = "motion"
    role: str = "caregiver"

    def build(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"{term_subject(self)}: role '{self.role}' is not one of {ROLES}; it picks the reference track "
                f"(AssistMimicPairMotion.grade_ref) this agent is graded against."
            )
        self.resolve_entity(self.entity_name)
        self._env.command_manager.get_term(self.command_name)

    def compute(self) -> torch.Tensor:
        ref = self._env.command_manager.get_term(self.command_name).grade_ref[self.role]
        return imitation_reward(
            *body_state(self.resolve_entity(self.entity_name)), ref.pos, ref.rot, ref.vel, ref.ang_vel
        )


@METRIC_TERM_REGISTRY.register()
class CaregiverHandReward(_ShareOnly):
    """AssistMimic's hand-distance reward, reduced from fingers to wrists.

    For each caregiver wrist within ``distance_threshold`` of the nearest recipient upper-body joint:
    ``reward_coefficient * exp(-exp_decay_rate * d) * force_multiplier + close_hands_bonus``, where the force
    multiplier sums ``min(exp(|F| - force_threshold), 1)`` over the wrist and hand links (AssistMimic sums it over
    15 finger links).

    Parameters
    ----------
    entity_name : str
        The caregiver.
    partner_entity_name : str
        The recipient.
    distance_threshold, exp_decay_rate, reward_coefficient, close_hands_bonus, force_threshold : float
        AssistMimic ``hand_distance_reward`` / ``finger_force_threshold``.
    force_link_scale : float
        AssistMimic sums the force multiplier over 15 finger links per hand; SMPL has 2 (wrist, hand), so the
        default ``15 / 2`` keeps the original range and hence this reward's weight against the imitation reward.
    """

    entity_name: str = "caregiver"
    partner_entity_name: str = "recipient"
    distance_threshold: float = 0.4
    exp_decay_rate: float = 2.5
    reward_coefficient: float = 0.5
    close_hands_bonus: float = 0.05
    force_threshold: float = 1.0
    force_link_scale: float = 7.5

    def build(self) -> None:
        self._wrist_idx, self._hand_idx = wrist_hand_idx(self.resolve_entity(self.entity_name), self.device)
        self._target_idx = resolve_link_idx(
            self.resolve_entity(self.partner_entity_name), HAND_TARGET_BODIES, self.device
        )

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        wrists = entity.get_links_pos(ls_idx_local=self._wrist_idx)
        targets = self.resolve_entity(self.partner_entity_name).get_links_pos(ls_idx_local=self._target_idx)
        dist = torch.cdist(wrists, targets).min(dim=-1).values  # (N, 2)
        forces = entity.get_links_net_contact_force()
        link_force = torch.stack([forces[:, self._wrist_idx], forces[:, self._hand_idx]], dim=2).norm(dim=-1)
        multiplier = torch.exp(link_force - self.force_threshold).clamp(max=1.0).sum(dim=-1) * self.force_link_scale
        reward = self.reward_coefficient * torch.exp(-self.exp_decay_rate * dist) * multiplier + self.close_hands_bonus
        return torch.where(dist <= self.distance_threshold, reward, torch.zeros_like(reward)).sum(dim=-1)


@METRIC_TERM_REGISTRY.register()
class RecipientHeadHeight(_ShareOnly):
    """AssistMimic's lift-up reward: the recipient's head height, clamped to ``[0, max_height]``.

    Parameters
    ----------
    entity_name : str
        The recipient.
    max_height : float
        ``max_head_height_reward``.
    """

    entity_name: str = "recipient"
    max_height: float = 2.0

    def build(self) -> None:
        self._head_idx = resolve_link_idx(self.resolve_entity(self.entity_name), ["Head"], self.device)

    def compute(self) -> torch.Tensor:
        head = self.resolve_entity(self.entity_name).get_links_pos(ls_idx_local=self._head_idx)[:, 0, 2]
        return head.clamp(0.0, self.max_height)


@METRIC_TERM_REGISTRY.register()
class TorqueReduction(_DofsMetric):
    """``exp(-sum|tau| / scale)`` over an agent's actuated DOFs, in ``(0, 1]``.

    AssistMimic's ``_compute_caregiver_torque_reduction_reward``, which pays the *caregiver* for keeping its
    **recipient's** total joint torque low — assistance that actually takes the recipient's weight. Only the
    HHI-Assist bed task enables it (``caregiver_torque_reduction_reward``); Inter-X help-up leaves it off.
    The coefficient is the reward's weight, not part of this metric.

    Parameters
    ----------
    entity_name : str
        The humanoid whose torque is measured — the recipient, for the caregiver's reward.
    dofs_name : list[str]
        The actuated DOFs.
    scale : float
        ``caregiver_torque_reduction_scale`` [N m]: the total torque at which the metric falls to ``1/e``.
    """

    entity_name: str = "recipient"
    scale: float = 150.0

    def compute(self) -> torch.Tensor:
        torque = self.resolve_entity(self.entity_name).get_dofs_control_force(self._dofs_idx)
        return torch.exp(-torque.abs().sum(dim=-1) / self.scale)


@METRIC_TERM_REGISTRY.register()
class JointPower(_DofsMetric):
    """Mechanical power ``sum |tau * qdot|`` over the actuated DOFs, zero for an episode's first 3 steps (PHC).

    A restored episode (:meth:`~eden.tasks.assistmimic.custom_terms.commands.AssistMimicPairMotion.restore_state`)
    gets those 3 steps too: its episode length restarts at 0.

    Parameters
    ----------
    entity_name : str
        The humanoid.
    dofs_name : list[str]
        The actuated DOFs.
    """

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        power = (entity.get_dofs_control_force(self._dofs_idx) * entity.get_dofs_vel(self._dofs_idx)).abs().sum(-1)
        return torch.where(self._env.episode_length_buf <= 3, torch.zeros_like(power), power)
