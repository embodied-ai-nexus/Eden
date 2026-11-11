"""AMO observation terms (adapter output, gait, demo, dyaw)."""

from __future__ import annotations
import torch
from hashlib import md5
import os
from genesis.utils.geom import quat_to_xyz

from eden.managers import OBSERVATION_TERM_REGISTRY


class AdapterWrapper:
    def __init__(self, device):
        self.device = device
        self.adapter = None
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def load(self):
        # The adapter ships beside AMO's checkpoint, so it is fetched (and cached) as AMOPolicy's own files.
        # Imported here: the policy module imports this one.
        from eden.extensions.policies.amo.policy import AMOPolicy

        jit_path = AMOPolicy.fetch_asset("amo_adapter_jit.pt")
        stats_path = AMOPolicy.fetch_asset("amo_adapter_norm_stats.pt")
        if md5(open(jit_path, "rb").read()).hexdigest() != "8782f8b362ae3e41940e5811e458b981":
            raise ValueError(f"MD5 hash mismatch for {jit_path}")
        if md5(open(stats_path, "rb").read()).hexdigest() != "bc5267c4650e9b09f05857f422e10f68":
            raise ValueError(f"MD5 hash mismatch for {stats_path}")

        if not os.path.exists(jit_path) or not os.path.exists(stats_path):
            # Return zeros if files missing (e.g. during dev/testing without weights)
            return

        self.adapter = torch.jit.load(jit_path, map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats = torch.load(stats_path, map_location=self.device, weights_only=False)
        self.input_mean = torch.tensor(norm_stats["input_mean"], device=self.device).float()
        self.input_std = torch.tensor(norm_stats["input_std"], device=self.device).float()
        self.output_mean = torch.tensor(norm_stats["output_mean"], device=self.device).float()
        self.output_std = torch.tensor(norm_stats["output_std"], device=self.device).float()

    def __call__(self, adapter_input):
        if self.adapter is None:
            return torch.zeros(adapter_input.shape[0], 15, device=self.device)
        adapter_input = (adapter_input - self.input_mean) / (self.input_std + 1e-8)
        output = self.adapter(adapter_input)
        return output * self.output_std + self.output_mean


AMO_ADAPTER_INSTANCE = None


@OBSERVATION_TERM_REGISTRY.register()
def amo_adapter_output(env, entity_name, command_name="amo_command"):
    global AMO_ADAPTER_INSTANCE
    if AMO_ADAPTER_INSTANCE is None:
        AMO_ADAPTER_INSTANCE = AdapterWrapper(env.device)
        AMO_ADAPTER_INSTANCE.load()

    cmd_term = env.command_manager.get_term(command_name)
    cmds = cmd_term.command

    height = cmds[:, 3] + 0.75
    torso = cmds[:, 4:7]

    dofs_pos = env.entities[entity_name].get_dofs_pos()
    arms = dofs_pos[:, 15:]

    adapter_input = torch.cat([height.unsqueeze(-1), torso, arms], dim=-1)

    return AMO_ADAPTER_INSTANCE(adapter_input)


@OBSERVATION_TERM_REGISTRY.register()
def amo_gait_obs(env, entity_name, command_name="amo_command"):
    cmd_term = env.command_manager.get_term(command_name)
    gait_cycle = cmd_term.gait_cycle
    return torch.sin(gait_cycle * 2 * torch.pi)


@OBSERVATION_TERM_REGISTRY.register()
def amo_demo_obs(env, entity_name, command_name="amo_command"):
    cmd_term = env.command_manager.get_term(command_name)
    cmds = cmd_term.command

    dofs_pos = env.entities[entity_name].get_dofs_pos()

    obs_demo = torch.zeros(env.num_envs, 17, device=env.device)
    obs_demo[:, :8] = dofs_pos[:, 15:]  # arms
    obs_demo[:, 8] = cmds[:, 0]  # vx
    obs_demo[:, 9] = cmds[:, 2]  # vy
    # index 10 is 0
    obs_demo[:, 11] = cmds[:, 4]  # torso yaw
    obs_demo[:, 12] = cmds[:, 5]  # torso pitch
    obs_demo[:, 13] = cmds[:, 6]  # torso roll
    h = 0.75 + cmds[:, 3]
    obs_demo[:, 14:17] = h

    return obs_demo


@OBSERVATION_TERM_REGISTRY.register()
def amo_dyaw_obs(env, entity_name, command_name="amo_command"):
    cmd_term = env.command_manager.get_term(command_name)
    target_yaw = cmd_term.command[:, 1]

    quat = env.entities[entity_name].get_quat()
    # quat_to_xyz returns roll, pitch, yaw
    _, _, yaw = quat_to_xyz(quat, rpy=True).unbind(-1)

    dyaw = yaw - target_yaw
    dyaw = torch.remainder(dyaw + torch.pi, 2 * torch.pi) - torch.pi

    # If in-place stand, dyaw is 0
    in_place = cmd_term._in_place_stand_flag
    dyaw[in_place] = 0.0

    return torch.stack([torch.sin(dyaw), torch.cos(dyaw)], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
def amo_dofs_vel(env, entity_name):
    dofs_vel = env.entities[entity_name].get_dofs_vel()
    # mask [4, 5, 10, 11, 13, 14] -> 0
    # indices:
    # 4: left_ankle_pitch
    # 5: left_ankle_roll
    # 10: right_ankle_pitch
    # 11: right_ankle_roll
    # 13: waist_roll
    # 14: waist_pitch
    mask = torch.tensor([4, 5, 10, 11, 13, 14], device=env.device)
    dofs_vel[:, mask] = 0.0
    return dofs_vel


@OBSERVATION_TERM_REGISTRY.register()
def amo_last_action(env, entity_name, action_name="dofs_pos_controller", command_name="amo_command"):
    action_term = env.action_manager.get_term(action_name)
    last_act = action_term.action.clone()

    dofs_pos = env.entities[entity_name].get_dofs_pos()

    cmd_term = env.command_manager.get_term(command_name)
    default_arm = cmd_term.default_arm_pos

    measured_arm_act = (dofs_pos[:, 15:] - default_arm) / 0.25
    last_act[:, 15:] = measured_arm_act
    return last_act
