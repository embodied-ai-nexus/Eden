from __future__ import annotations
from typing import TYPE_CHECKING, List

import torch

from genesis.utils.geom import inv_transform_by_quat
from eden.managers.terms.commands.mimic import MotionCommand
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.terms.utils import gravity_vec
from eden.options.managers.terminations import TerminationTermOptions

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def bad_anchor_pos(
    env: RLEnvBase,
    *,
    command_name: str = "motion",
    threshold: float = 0.01,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


@TERMINATION_TERM_REGISTRY.register()
def beyondmimic_bad_anchor_pos_z_only(
    env: RLEnvBase,
    *,
    command_name: str = "motion",
    threshold: float = 0.01,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


@TERMINATION_TERM_REGISTRY.register()
def beyondmimic_bad_anchor_ori(
    env: RLEnvBase,
    *,
    command_name: str = "motion",
    threshold: float = 0.01,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Isaac Lab's unit GRAVITY_VEC_W: the solver's gravity is in m/s^2, which would scale the threshold by 9.81.
    gravity_vec_w = gravity_vec(env.num_envs, env.device)
    motion_projected_gravity_b = inv_transform_by_quat(gravity_vec_w, command.anchor_quat_w)
    robot_projected_gravity_b = inv_transform_by_quat(gravity_vec_w, command.robot_anchor_quat_w)
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


@TERMINATION_TERM_REGISTRY.register()
class BadMotionBodyPos(TerminationTerm):
    command_name: str = "motion"
    entity_name: str = "robot"
    links_name: List[str] = []
    threshold: float = 0.01

    def __init__(self, env: RLEnvBase, options: TerminationTermOptions):
        super().__init__(env=env, options=options)
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        ls_idx_local = [
            i for i, name in enumerate(command.links_name) if (self.links_name is None) or (name in self.links_name)
        ]
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = ls_idx_local

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        error = torch.norm(
            command.links_pos_relative_w[:, self.ls_idx_local] - command.robot_links_pos_w[:, self.ls_idx_local],
            dim=-1,
        )
        return torch.any(error > self.threshold, dim=-1)


@TERMINATION_TERM_REGISTRY.register()
class BeyondmimicBadMotionBodyPosZOnly(TerminationTerm):
    command_name: str = "motion"
    entity_name: str = "robot"
    links_name: List[str] = []
    threshold: float = 0.01

    def __init__(self, env: RLEnvBase, options: TerminationTermOptions):
        super().__init__(env=env, options=options)
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        ls_idx_local = [
            i for i, name in enumerate(command.links_name) if (self.links_name is None) or (name in self.links_name)
        ]
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = ls_idx_local

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        error = torch.abs(
            command.links_pos_relative_w[:, self.ls_idx_local, -1] - command.robot_links_pos_w[:, self.ls_idx_local, -1]
        )
        return torch.any(error > self.threshold, dim=-1)
