"""Reward terms for the Extreme Parkour teacher task.

Rewards run exactly once per control step (``RewardManager.compute`` in
``RLEnvBase.step``), so the stateful terms here (previous-torque buffer,
feet-contact latch) need no re-entrancy stamp guard — only a ``reset()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import ndimage

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import gravity_vec, projected_gravity, resolve_link_idx
from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.entities.rigid import RigidEntity
    from eden.entities.terrain import Terrain
    from eden.envs.base import RLEnvBase


def _get_parkour_command(env: RLEnvBase, command_name: str, requester: str) -> ParkourCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, ParkourCommand):
        raise TypeError(f"{requester}: command term '{command_name}' is not a ParkourCommand.")
    return command


def _feet_in_contact(entity, feet_idx: torch.Tensor, threshold: float) -> torch.Tensor:
    """Per-foot boolean: net contact-force magnitude on each foot link exceeds ``threshold``.

    Shared by the foot-contact reward terms so the detection convention lives in one place.
    """
    forces = entity.get_links_net_contact_force()[:, feet_idx]
    return torch.norm(forces, dim=-1) > threshold


# ---------------------------------------------------------------------------
# Goal tracking
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def tracking_goal_vel(env: RLEnvBase, *, command_name: str, entity_name: str = "robot") -> torch.Tensor:
    """Reward the WORLD-frame velocity component towards the current goal, capped at the commanded vx."""
    command: ParkourCommand = env.command_manager.get_term(command_name)
    robot: Entity = env.entities[entity_name]
    target_dir = command.target_pos_rel / (torch.norm(command.target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    proj_vel = torch.sum(target_dir * robot.get_vel(frame="world")[:, :2], dim=-1)
    command_vx = command.command[:, 0]
    return torch.minimum(proj_vel, command_vx) / (command_vx + 1e-5)


@REWARD_TERM_REGISTRY.register()
def tracking_yaw(env: RLEnvBase, *, command_name: str, entity_name: str = "robot") -> torch.Tensor:
    """``exp(-|target_yaw - yaw|)`` towards the current goal."""
    command: ParkourCommand = env.command_manager.get_term(command_name)
    yaw = env.entities[entity_name].get_heading()
    return torch.exp(-torch.abs(command.target_yaw - yaw))


# ---------------------------------------------------------------------------
# Effort / smoothness penalties (return positive costs; use negative weights)
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def torques_l2(env: RLEnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Sum of squared applied control torques."""
    return torch.sum(torch.square(env.entities[entity_name].get_dofs_control_force()), dim=1)


@REWARD_TERM_REGISTRY.register()
class DeltaTorquesL2(RewardTerm):
    """Sum of squared torque differences between consecutive control steps (term-owned prev buffer)."""

    entity_name: str = "robot"

    def build(self) -> None:
        entity: Entity = self._env.entities[self.entity_name]
        self._prev_torques = torch.zeros_like(entity.get_dofs_control_force())

    def compute(self) -> torch.Tensor:
        torques = self.resolve_entity(self.entity_name).get_dofs_control_force()
        out = torch.sum(torch.square(torques - self._prev_torques), dim=1)
        self._prev_torques.copy_(torques)
        return out

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._prev_torques.zero_()
        else:
            self._prev_torques[envs_idx] = 0.0


@REWARD_TERM_REGISTRY.register()
def action_rate_norm(env: RLEnvBase) -> torch.Tensor:
    """L2 NORM (not squared) of the action difference — matches the Extreme Parkour source."""
    action_manager = env.action_manager
    return torch.norm(action_manager.action - action_manager.prev_action, dim=1)


@REWARD_TERM_REGISTRY.register()
def ang_vel_xy_l2(env: RLEnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Sum of squared BODY-frame roll/pitch angular velocity."""
    return torch.sum(torch.square(env.entities[entity_name].get_ang(frame="body")[:, :2]), dim=1)


# ---------------------------------------------------------------------------
# Terrain-aware penalties
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
class LinVelZTerrainWeighted(RewardTerm):
    """Squared BODY-frame vertical velocity, halved on non-flat (parkour) cells.

    Vertical motion is unavoidable over obstacles, so the penalty is applied at
    half strength there and at full strength on the flat courses.
    """

    entity_name: str = "robot"
    command_name: str = "parkour_command"

    def build(self) -> None:
        _ = self._env.entities[self.entity_name]
        self._command = _get_parkour_command(self._env, self.command_name, type(self).__name__)

    def compute(self) -> torch.Tensor:
        v_z = self.resolve_entity(self.entity_name).get_vel(frame="body")[:, 2]
        return torch.square(v_z) * torch.where(self._command.env_is_flat, 1.0, 0.5)


@REWARD_TERM_REGISTRY.register()
class FlatOrientationOnFlat(RewardTerm):
    """Projected-gravity-xy (uprightness) penalty, terrain-weighted.

    Full strength on flat (obstacle-suppressed) cells and ``nonflat_scale`` strength
    on obstacle cells. With ``nonflat_scale=0.0`` this is the source's flat-only
    penalty; a small positive value keeps a mild uprightness pressure on the ~80%
    of the grid that is obstacle terrain (which otherwise has *no* pitch penalty at
    all, letting a strongly forward-leaned gait dominate and bleed onto flat cells)
    while still permitting the pitching genuinely needed to clear obstacles. Mirrors
    the ``torch.where(env_is_flat, 1.0, 0.5)`` idiom used by ``LinVelZTerrainWeighted``.
    """

    entity_name: str = "robot"
    command_name: str = "parkour_command"
    nonflat_scale: float = 0.0

    def build(self) -> None:
        _ = self._env.entities[self.entity_name]
        self._command = _get_parkour_command(self._env, self.command_name, type(self).__name__)
        self._grav = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        grav_in_base = projected_gravity(self.resolve_entity(self.entity_name).get_quat(), self._grav)
        weight = torch.where(self._command.env_is_flat, 1.0, self.nonflat_scale)
        return torch.sum(torch.square(grav_in_base[:, :2]), dim=1) * weight


def compute_x_edge_mask(
    height_field_units: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float,
    edge_width_thresh: float,
) -> np.ndarray:
    """Boolean (X, Y) mask of cells at (or dilated around) a steep x-direction height edge.

    A cell is an edge when the height jump to its +x neighbor exceeds
    ``slope_threshold * horizontal_scale`` (in meters; the comparison happens in
    ``vertical_scale`` units). The mask is dilated along x by ``edge_width_thresh``
    meters so contacts *near* the edge count too.
    """
    threshold = slope_threshold * horizontal_scale / vertical_scale
    height_field = np.asarray(height_field_units)
    move_x = np.zeros_like(height_field, dtype=np.int32)
    move_x[:-1] += height_field[1:] - height_field[:-1] > threshold
    move_x[1:] -= height_field[:-1] - height_field[1:] > threshold
    mask = move_x != 0
    half_width_px = int(edge_width_thresh / horizontal_scale)
    structure = np.ones((2 * half_width_px + 1, 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure)


@REWARD_TERM_REGISTRY.register()
class FeetEdgePenalty(RewardTerm):
    """Count of feet in contact on (or near) a steep terrain edge, gated by terrain level.

    The edge mask is computed once at build from the terrain's stitched height
    field; per step each foot's world xy is mapped to a height-field pixel and a
    foot counts when its pixel is masked AND the foot is in (latched) contact.
    The penalty only applies to envs whose terrain level exceeds
    ``min_terrain_level`` (low levels are still learning basic locomotion).
    """

    entity_name: str = "robot"
    terrain_entity_name: str = "terrain"
    command_name: str = "parkour_command"
    feet_link_names: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    slope_threshold: float = 1.5
    edge_width_thresh: float = 0.05
    contact_force_threshold: float = 2.0
    min_terrain_level: int = 3

    def build(self) -> None:
        entity: RigidEntity = self._env.entities[self.entity_name]
        terrain: Terrain = self._env.entities[self.terrain_entity_name]
        self._command = _get_parkour_command(self._env, self.command_name, type(self).__name__)
        # The empty-selection check lives in resolve_link_idx, which also lists the entity's links.
        self._feet_idx = resolve_link_idx(
            entity, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )

        mask = compute_x_edge_mask(
            terrain.height_field_units,
            terrain.horizontal_scale,
            terrain.vertical_scale,
            self.slope_threshold,
            self.edge_width_thresh,
        )
        self._edge_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        x_min, _, y_min, _ = terrain.bounds
        self._terrain_origin_xy = torch.tensor([x_min, y_min], device=self.device)
        self._inv_horizontal_scale = 1.0 / terrain.horizontal_scale
        self._last_contacts = torch.zeros(self.num_envs, self._feet_idx.numel(), dtype=torch.bool, device=self.device)

    def compute(self) -> torch.Tensor:
        entity: RigidEntity = self.resolve_entity(self.entity_name)
        foot_xy = entity.get_links_pos(ls_idx_local=self._feet_idx)[..., :2]  # (N, n_feet, 2)
        pixels = torch.round((foot_xy - self._terrain_origin_xy) * self._inv_horizontal_scale).long()
        px = pixels[..., 0].clamp(0, self._edge_mask.shape[0] - 1)
        py = pixels[..., 1].clamp(0, self._edge_mask.shape[1] - 1)
        at_edge = self._edge_mask[px, py]  # (N, n_feet)

        contact = _feet_in_contact(entity, self._feet_idx, self.contact_force_threshold)
        contact_filt = contact | self._last_contacts
        # copy_ (not rebind): compute may run under torch.inference_mode, and a rebound
        # inference tensor could no longer be mutated by reset() outside it.
        self._last_contacts.copy_(contact)

        feet_at_edge = (at_edge & contact_filt).sum(dim=-1).float()
        return (self._command.terrain_levels > self.min_terrain_level).float() * feet_at_edge

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._last_contacts.zero_()
        else:
            self._last_contacts[envs_idx] = False
