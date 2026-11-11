"""Oracle pose command terms (uniform T3 / SE3 targets)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
from genesis.utils.geom import xyz_to_quat

from eden.managers import CommandTerm, COMMAND_TERM_REGISTRY
from genesis.utils.geom import transform_by_quat
from eden.utils.sample import sample_uniform_ranges

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class UniformT3Command(CommandTerm):
    """
    Command that samples T(3) translation command.

    Parameters
    ----------
    x_range: tuple[float, float]
        Range of the x-coordinate of the target position.
    y_range: tuple[float, float]
        Range of the y-coordinate of the target position.
    z_range: tuple[float, float]
        Range of the z-coordinate of the target position.
    debug_vis: bool
        When True (and not training), draw a debug sphere at the goal each step.
    marker_frame_entity: str
        Entity whose frame the command is expressed in. When set, the marker is
        placed at ``entity.pos + R(entity.quat) @ command`` (the command is in that
        entity's local frame, e.g. the robot base); empty -> command is world-frame.
    marker_entity_name: str
        Name of a scene entity (e.g. a non-colliding ``SphereOptions``) to move to
        the goal each step. Preferred over ``draw_debug_spheres`` because debug
        draws are viewer-only and do NOT appear in recorded camera videos.
    marker_radius: float
        Fallback debug-sphere radius [m] (used only when ``marker_entity_name`` is
        unset).
    marker_color: tuple[float, float, float, float]
        Fallback debug-sphere RGBA.
    """

    x_range: tuple[float, float] = (0.3, 0.5)
    y_range: tuple[float, float] = (-0.2, 0.2)
    z_range: tuple[float, float] = (0.2, 0.4)
    marker_frame_entity: str = ""
    marker_entity_name: str = ""
    marker_radius: float = 0.03
    marker_color: tuple[float, float, float, float] = (0.1, 0.9, 0.1, 0.85)

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        self.oracle_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.oracle_pos_b

    def _resample_command(self, envs_idx: torch.Tensor):
        # A full-batch draw written through the mask: the mask as an index is a GPU->CPU sync.
        sampled = sample_uniform_ranges((self.x_range, self.y_range, self.z_range), self.num_envs, self.device)
        self.oracle_pos_b.copy_(torch.where(envs_idx.unsqueeze(-1), sampled, self.oracle_pos_b))

    def _update_command(self):
        # No per-step update needed; command stays constant until resample.
        pass

    def draw_vis(self) -> None:
        # Goal marker (eval/inference only; gated on debug_vis by the manager).
        pos = self.oracle_pos_b
        if self.marker_frame_entity:
            ent = self.resolve_entity(self.marker_frame_entity)
            pos = ent.get_pos() + transform_by_quat(pos, ent.get_quat())
        marker = (
            self._env.entities[self.marker_entity_name]
            if self.marker_entity_name and self.marker_entity_name in self._env.entities
            else None
        )
        if marker is not None:
            # Move a real (non-colliding) entity to the goal — renders in cameras,
            # unlike draw_debug_* which only shows in the interactive viewer.
            marker.set_pos(pos.detach())
        else:
            self._env.clear_debug_objects()
            self._env.draw_debug_spheres(pos.detach().cpu().numpy(), radius=self.marker_radius, color=self.marker_color)


@COMMAND_TERM_REGISTRY.register()
class UniformSE3Command(CommandTerm):
    """
    Command that samples SE(3) translation and rotation command.

    Parameters
    ----------
    x_range: tuple[float, float]
        Range of the x-coordinate of the target position.
    y_range: tuple[float, float]
        Range of the y-coordinate of the target position.
    z_range: tuple[float, float]
        Range of the z-coordinate of the target position.
    roll_range: tuple[float, float]
        Range of the roll angle of the target rotation (in radians).
    pitch_range: tuple[float, float]
        Range of the pitch angle of the target rotation (in radians).
    yaw_range: tuple[float, float]
        Range of the yaw angle of the target rotation (in radians).

    Note
    ----
    This term provides an equivalent command to `UniformPoseCommand` in IsaacLab.
    """

    x_range: tuple[float, float] = (0.3, 0.5)
    y_range: tuple[float, float] = (-0.2, 0.2)
    z_range: tuple[float, float] = (0.2, 0.4)
    roll_range: tuple[float, float] = (-np.pi, np.pi)
    pitch_range: tuple[float, float] = (-np.pi, np.pi)
    yaw_range: tuple[float, float] = (-np.pi, np.pi)

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.oracle_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.oracle_pose_b[:, 3] = 1.0  # identity quaternion (wxyz) until the first resample

    @property
    def command(self) -> torch.Tensor:
        return self.oracle_pose_b

    def _resample_command(self, envs_idx: torch.Tensor):
        # Full-batch draws written through the mask: the mask as an index is a GPU->CPU sync.
        # NOTE: sample new position targets
        pos = sample_uniform_ranges((self.x_range, self.y_range, self.z_range), self.num_envs, self.device)
        # NOTE: sample new orientation targets
        rpy = sample_uniform_ranges((self.roll_range, self.pitch_range, self.yaw_range), self.num_envs, self.device)
        pose = torch.cat([pos, xyz_to_quat(rpy, rpy=True)], dim=-1)
        self.oracle_pose_b.copy_(torch.where(envs_idx.unsqueeze(-1), pose, self.oracle_pose_b))

    def _update_command(self):
        pass
