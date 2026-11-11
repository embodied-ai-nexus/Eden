"""OmniXtreme motion command term."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.managers.command_manager import CommandTerm, COMMAND_TERM_REGISTRY
from eden.managers.terms.commands.mimic import resolve_body_cols

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class OmniXtremeRefCommand(CommandTerm):
    motion_file: str = "motion.npz"
    anchor_link_name: str = "torso_link"
    debug_vis: bool = False

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        # Written by OmniXtremePolicy.step(); read by the omnixtreme_prev_base_action observation.
        self.prev_base_action = torch.zeros(env.num_envs, 29, device=self.device)

        # Set by OmniXtremePolicy.__init__ (after env.build()); read by omnixtreme_anchor_ori_6d.
        self.fk_session = None

        # Genesis idx_local fallback for the anchor column, used only when the npz carries neither
        # 'body_names' nor a recognizable G1 body set (see resolve_body_cols in _load_motion).
        self._anchor_runtime_idx = env.entities["robot"].find_named_links_idx_local(self.anchor_link_name)[1]

        self._load_motion(options.motion_file)

    def _load_motion(self, motion_file: str):
        data = np.load(motion_file)
        required = ["joint_pos", "body_pos_w", "body_quat_w"]
        for k in required:
            if k not in data:
                raise KeyError(f"Missing key '{k}' in {motion_file}")

        # Resolve the anchor body column with the same name / legacy-BFS / idx_local fallback the
        # stock MotionLoader uses, so it stays correct under Genesis's 1.2.0 depth-first link
        # reorder even for legacy npz that predate 'body_names'.
        self.anchor_link_index = int(resolve_body_cols(data, self._anchor_runtime_idx, [self.anchor_link_name])[0])

        # body_pos_w: (T, num_bodies, 3)
        self._body_pos_w = torch.from_numpy(data["body_pos_w"]).float().to(self.device)
        # body_quat_w: (T, num_bodies, 4) in wxyz format
        self._body_quat_w = torch.from_numpy(data["body_quat_w"]).float().to(self.device)

        # joint_pos / joint_vel: (T, num_dofs) in the robot's URDF joint order, emitted as is by ``command``
        # (OmniXtremePolicy permutes them to BeyondMimic order for the residual policy).
        self._joint_pos = torch.from_numpy(data["joint_pos"]).float().to(self.device)
        if "joint_vel" in data:
            self._joint_vel = torch.from_numpy(data["joint_vel"]).float().to(self.device)
        else:
            self._joint_vel = torch.zeros_like(self._joint_pos)

        self._ref_motion_length = self._joint_pos.shape[0]

        # Frame-0 root xy, z zeroed: subtracting it starts the debug markers at the origin (heights kept).
        self._rootpos_offset = self._body_pos_w[0, 0].clone()
        self._rootpos_offset[2] = 0.0

    @property
    def command(self) -> torch.Tensor:
        """Returns (num_envs, 58): ref_joint_pos(29) + ref_joint_vel(29) in URDF order."""
        frame_idx = self._get_current_frame()
        ref_pos = self._joint_pos[frame_idx]
        ref_vel = self._joint_vel[frame_idx]
        cmd = torch.cat([ref_pos, ref_vel], dim=-1).unsqueeze(0)
        return cmd.expand(self._env.num_envs, -1)

    def _get_current_frame(self) -> int:
        frame_idx = self._env._sim_step_counter // self._env.env_options.decimation
        return min(frame_idx, self._ref_motion_length - 1)

    def get_anchor_ref_data(self):
        """Return the reference pose of the anchor body (``anchor_link_name``) at the current frame.

        Returns
        -------
        ref_pos : torch.Tensor
            ``(1, 3)`` position.
        ref_rot : torch.Tensor
            ``(1, 4)`` quaternion in wxyz format.
        """
        frame_idx = self._get_current_frame()
        ref_pos = self._body_pos_w[frame_idx, self.anchor_link_index].unsqueeze(0)
        ref_rot = self._body_quat_w[frame_idx, self.anchor_link_index].unsqueeze(0)
        return ref_pos, ref_rot

    def _resample_command(self, envs_idx: torch.Tensor):
        pass

    def _update_command(self):
        pass

    def draw_vis(self) -> None:
        frame_idx = self._get_current_frame()
        self._env.clear_debug_objects()

        # Draw reference body position markers (green spheres)
        ref_body_pos = self._body_pos_w[frame_idx] - self._rootpos_offset
        num_bodies = min(ref_body_pos.shape[0], 30)
        self._env.draw_debug_spheres(
            ref_body_pos[:num_bodies].cpu().numpy(),
            radius=0.05,
            color=(0.0, 1.0, 0.0, 0.8),
        )

        # Update ghost robot if available
        if "ghost_robot" in self._env.entities:
            ghost_pos = (self._body_pos_w[frame_idx, 0] - self._rootpos_offset).unsqueeze(0)
            ghost_quat = self._body_quat_w[frame_idx, 0].unsqueeze(0)
            ghost_dofs_pos = self._joint_pos[frame_idx].unsqueeze(0)

            entity = self._env.entities["ghost_robot"]
            entity.set_pos(ghost_pos)
            entity.set_quat(ghost_quat)
            entity.set_dofs_pos(ghost_dofs_pos)
