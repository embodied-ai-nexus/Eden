from __future__ import annotations
from typing import TYPE_CHECKING, List

import torch

from eden.utils.geom import quat_error_magnitude
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.options.managers.rewards import RewardTermOptions

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.terms.commands.mimic import MotionCommand


@REWARD_TERM_REGISTRY.register()
def motion_global_anchor_position_error_exp(
    env: RLEnvBase,
    *,
    command_name: str = "motion",
    std: float = 0.1,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


@REWARD_TERM_REGISTRY.register()
def motion_global_anchor_orientation_error_exp(
    env: RLEnvBase,
    *,
    command_name: str = "motion",
    std: float = 0.1,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


@REWARD_TERM_REGISTRY.register()
class BeyondmimicMotionRelativeBodyPositionErrorExp(RewardTerm):
    command_name: str = "motion"
    entity_name: str = "robot"
    links_name: List[str] = []
    std: float = 0.1

    def __init__(self, env: RLEnvBase, options: RewardTermOptions):
        super().__init__(env=env, options=options)
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        ls_idx_local = [
            i for i, name in enumerate(command.links_name) if (self.links_name is None) or (name in self.links_name)
        ]
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = ls_idx_local

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        error = torch.sum(
            torch.square(
                command.links_pos_relative_w[:, self.ls_idx_local] - command.robot_links_pos_w[:, self.ls_idx_local]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / self.std**2)


@REWARD_TERM_REGISTRY.register()
class BeyondmimicMotionRelativeBodyOrientationErrorExp(RewardTerm):
    command_name: str = "motion"
    entity_name: str = "robot"
    links_name: List[str] = []
    std: float = 0.1

    def __init__(self, env: RLEnvBase, options: RewardTermOptions):
        super().__init__(env=env, options=options)
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        ls_idx_local = [
            i for i, name in enumerate(command.links_name) if (self.links_name is None) or (name in self.links_name)
        ]
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = ls_idx_local

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        error = (
            quat_error_magnitude(
                command.links_quat_relative_w[:, self.ls_idx_local],
                command.robot_links_quat_w[:, self.ls_idx_local],
            )
            ** 2
        )
        return torch.exp(-error.mean(-1) / self.std**2)


@REWARD_TERM_REGISTRY.register()
class BeyondmimicMotionGlobalBodyLinearVelocityErrorExp(RewardTerm):
    command_name: str = "motion"
    entity_name: str = "robot"
    links_name: List[str] = []
    std: float = 0.1

    def __init__(self, env: RLEnvBase, options: RewardTermOptions):
        super().__init__(env=env, options=options)
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        ls_idx_local = [
            i for i, name in enumerate(command.links_name) if (self.links_name is None) or (name in self.links_name)
        ]
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = ls_idx_local

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        error = torch.sum(
            torch.square(
                command.links_lin_vel_w[:, self.ls_idx_local] - command.robot_links_lin_vel_w[:, self.ls_idx_local]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / self.std**2)


@REWARD_TERM_REGISTRY.register()
class BeyondmimicMotionGlobalBodyAngularVelocityErrorExp(RewardTerm):
    command_name: str = "motion"
    entity_name: str = "robot"
    links_name: List[str] = []
    std: float = 0.1

    def __init__(self, env: RLEnvBase, options: RewardTermOptions):
        super().__init__(env=env, options=options)
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        ls_idx_local = [
            i for i, name in enumerate(command.links_name) if (self.links_name is None) or (name in self.links_name)
        ]
        assert len(ls_idx_local) > 0, f"No links found for {self.links_name} in {self.__class__.__name__}"
        self.ls_idx_local = ls_idx_local

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        error = torch.sum(
            torch.square(
                command.links_ang_vel_w[:, self.ls_idx_local] - command.robot_links_ang_vel_w[:, self.ls_idx_local]
            ),
            dim=-1,
        )
        return torch.exp(-error.mean(-1) / self.std**2)


# def feet_contact_time(env: RLEnvBase, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
#     last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
#     return reward
