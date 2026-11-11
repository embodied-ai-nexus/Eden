"""UWLab's OmniReset rewards (``rl_state_cfg.RewardsCfg``), each with the weight UWLab gives it in the task config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_manager
from eden.utils.geom import transform_by_quat
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase

#: UWLab clamps every squared-norm penalty to this.
_PENALTY_CLAMP = 1e4


@REWARD_TERM_REGISTRY.register()
def action_l2_clamped(env: RLEnvBase) -> torch.Tensor:
    """Squared norm of the raw action, clamped (UWLab ``action_magnitude``, weight -1e-4)."""
    return env.action_manager.action.square().sum(dim=-1).clamp(0.0, _PENALTY_CLAMP)


@REWARD_TERM_REGISTRY.register()
def action_rate_l2_clamped(env: RLEnvBase) -> torch.Tensor:
    """Squared norm of the raw action's change, clamped (UWLab ``action_rate``, weight -1e-3)."""
    change = env.action_manager.action - env.action_manager.prev_action
    return change.square().sum(dim=-1).clamp(0.0, _PENALTY_CLAMP)


@REWARD_TERM_REGISTRY.register()
class JointVelL2Clamped(RewardTerm):
    """Squared norm of ``joints_name``' velocities, clamped (UWLab ``joint_vel`` on the arm, weight -1e-2)."""

    entity_name: str = "robot"
    joints_name: list[str] = []

    def build(self) -> None:
        if not self.joints_name:
            raise ValueError(
                f"{term_subject(self)} needs joints_name: with none it would penalize nothing. Fix: "
                "JointVelL2Clamped.configure(joints_name=[...]), names or patterns of the joints to penalize."
            )
        # Names and patterns resolve here; a pattern that matches no joint raises, naming the joints there are.
        _, self._dofs_idx = self.resolve_entity(self.entity_name).find_named_dofs_idx_local(list(self.joints_name))

    def compute(self) -> torch.Tensor:
        dofs_vel = self.resolve_entity(self.entity_name).get_dofs_vel(dofs_idx_local=self._dofs_idx)
        return dofs_vel.square().sum(dim=-1).clamp(0.0, _PENALTY_CLAMP)


@REWARD_TERM_REGISTRY.register()
class TerminationFired(RewardTerm):
    """1 where termination term ``termination_name`` ended the episode this step (UWLab ``abnormal_robot``, -100)."""

    termination_name: str = ""

    def build(self) -> None:
        self._env.termination_manager.get_term(self.termination_name)  # a typo raises, listing the active terms

    def compute(self) -> torch.Tensor:
        return self._env.termination_manager.get_term_dones(self.termination_name).float()


@REWARD_TERM_REGISTRY.register()
class GripperObjectProximity(RewardTerm):
    """``1 - tanh(d / std)`` for the distance ``d`` from the gripper's grasp point to the object's origin.

    UWLab ``ee_asset_distance`` (weight 0.1, std 1.0): the grasp point is ``link_name`` offset by ``offset`` in its
    frame (the robot metadata's ``gripper_offset``, 13.45 cm along the approach axis of ``robotiq_base_link``).
    """

    entity_name: str = "robot"
    link_name: str = "robotiq_base_link"
    offset: tuple[float, float, float] = (0.1345, 0.0, 0.0)
    object_name: str = "insertive_object"
    std: float = 1.0

    def build(self) -> None:
        self._link = int(resolve_link_idx(self.resolve_entity(self.entity_name), [self.link_name], self.device)[0])
        self.resolve_entity(self.object_name)
        self._offset = torch.tensor(self.offset, device=self.device)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        link_pos = robot.get_links_pos(ls_idx_local=[self._link])[:, 0]
        link_quat = robot.get_links_quat(ls_idx_local=[self._link])[:, 0]
        grasp_point = link_pos + transform_by_quat(self._offset.expand_as(link_pos), link_quat)
        distance = (self.resolve_entity(self.object_name).get_pos(relative=True) - grasp_point).norm(dim=-1)
        return 1.0 - torch.tanh(distance / self.std)


@REWARD_TERM_REGISTRY.register()
class AssemblyReward(RewardTerm):
    """UWLab's assembly rewards, read off the two :class:`AssemblyError` metrics.

    ``kind="dense"`` is ``dense_success_reward`` (weight 0.1): the mean of ``exp(-tilt / std)`` and
    ``exp(-position / std)``. ``kind="success"`` is ``success_reward`` (weight 1.0): 1 while both metrics pass their
    own success check, the one ``Metrics/Success`` grades episodes with.
    """

    position_metric: str = "assembly_position"
    tilt_metric: str = "assembly_tilt"
    kind: str = "dense"
    std: float = 1.0

    def build(self) -> None:
        if self.kind not in ("dense", "success"):
            raise ValueError(f"AssemblyReward: kind must be 'dense' or 'success', got {self.kind!r}.")
        manager = resolve_metric_manager(self._env, term="assembly_reward", metric_name=self.position_metric)
        resolve_metric_manager(self._env, term="assembly_reward", metric_name=self.tilt_metric)
        self._position_term = manager.get_term(self.position_metric)
        self._tilt_term = manager.get_term(self.tilt_metric)

    def compute(self) -> torch.Tensor:
        manager = self._env.metric_manager
        position, tilt = manager.get_value(self.position_metric), manager.get_value(self.tilt_metric)
        if self.kind == "dense":
            return 0.5 * (torch.exp(-tilt / self.std) + torch.exp(-position / self.std))
        return (self._position_term.is_success(position) & self._tilt_term.is_success(tilt)).float()
