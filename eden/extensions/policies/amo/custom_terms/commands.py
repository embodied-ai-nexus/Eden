"""AMO motion command term."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from eden.managers.command_manager import CommandTerm, COMMAND_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class AMOCommand(CommandTerm):
    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        # commands: [vx, yaw, vy, height, torso_yaw, torso_pitch, torso_roll, arm_toggle]
        self.commands = torch.zeros(env.num_envs, 8, device=self.device)

        # Arm control state
        self.arm_blend = torch.zeros(env.num_envs, device=self.device)
        self.prev_arm_action = torch.zeros(env.num_envs, 8, device=self.device)
        self.arm_action = torch.zeros(env.num_envs, 8, device=self.device)
        self.toggle_arm = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        # Arm ranges
        self.arm_dof_lower_range = -0.4 * torch.ones(8, device=self.device)
        self.arm_dof_upper_range = 0.4 * torch.ones(8, device=self.device)

        # Default arm pos (from play_amo.py default_dof_pos[15:])
        self.default_arm_pos = torch.tensor(
            [
                0.5,
                0.0,
                0.2,
                0.3,
                0.5,
                0.0,
                -0.2,
                0.3,
            ],
            device=self.device,
        )

        self.prev_arm_action[:] = self.default_arm_pos
        self.arm_action[:] = self.default_arm_pos

        # Gait state
        self.gait_cycle = torch.zeros(env.num_envs, 2, device=self.device)
        self.gait_cycle[:] = 0.25
        self.gait_freq = 1.3
        self._in_place_stand_flag = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)

        # Counter for arm toggle logic
        self.step_counter = 0

    @property
    def command(self) -> torch.Tensor:
        return self.commands

    def _resample_command(self, envs_idx: torch.Tensor):
        self.commands[envs_idx] = 0.0
        self.gait_cycle[envs_idx] = 0.25
        self.arm_blend[envs_idx] = 0.0
        self.prev_arm_action[envs_idx] = self.default_arm_pos
        self.arm_action[envs_idx] = self.default_arm_pos

    def _update_command(self):
        dt = self._env.dt * self._env.env_options.decimation
        self.gait_cycle = (self.gait_cycle + dt * self.gait_freq) % 1.0

        self._in_place_stand_flag = torch.abs(self.commands[:, 0]) < 0.1

        # Standing in place: snap a phase near 0.25 back to 0.25.
        in_place = self._in_place_stand_flag
        near_025_0 = torch.abs(self.gait_cycle[:, 0] - 0.25) < 0.05
        near_025_1 = torch.abs(self.gait_cycle[:, 1] - 0.25) < 0.05
        reset_mask = in_place & (near_025_0 | near_025_1)
        self.gait_cycle[reset_mask] = 0.25

        # Walking with both phases near 0.25: offset the second leg to 0.75.
        set_offset_mask = (~in_place) & near_025_0 & near_025_1
        self.gait_cycle[set_offset_mask, 1] = 0.75

        arm_active = self.commands[:, 7] > 0.5

        self.step_counter += 1
        if self.step_counter % 300 == 0:
            new_target_mask = arm_active
            if new_target_mask.any():
                self.arm_blend[new_target_mask] = 0.0

                # Get current arm dofs
                dofs_pos = self._env.entities["robot"].get_dofs_pos()
                current_arms = dofs_pos[:, 15:]

                self.prev_arm_action[new_target_mask] = current_arms[new_target_mask]

                # Random new action
                rand_act = torch.rand((new_target_mask.sum(), 8), device=self.device)
                lower = self.arm_dof_lower_range
                upper = self.arm_dof_upper_range
                self.arm_action[new_target_mask] = rand_act * (upper - lower) + lower
                self.toggle_arm[new_target_mask] = True

        # Arm toggled off: blend back to the default arm pose.
        not_active = ~arm_active
        reset_mask = not_active & self.toggle_arm
        if reset_mask.any():
            self.toggle_arm[reset_mask] = False
            self.arm_blend[reset_mask] = 0.0
            dofs_pos = self._env.entities["robot"].get_dofs_pos()
            self.prev_arm_action[reset_mask] = dofs_pos[reset_mask, 15:]
            self.arm_action[reset_mask] = self.default_arm_pos

        self.arm_blend = torch.clamp(self.arm_blend + 0.01, 0.0, 1.0)

    def get_arm_target(self):
        """Arm PD target ``(num_envs, 8)``: ``(1 - blend) * prev + blend * target``."""
        blend = self.arm_blend.unsqueeze(-1)
        return (1.0 - blend) * self.prev_arm_action + blend * self.arm_action
