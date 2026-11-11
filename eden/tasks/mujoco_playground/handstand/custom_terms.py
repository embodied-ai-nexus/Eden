"""Custom reward, termination, event, and observation terms for the Go1 handstand task.

Ported from: mujoco_playground/_src/locomotion/go1/handstand.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.options.managers.events import EventTermOptions
from eden.utils.geom import quat_from_euler_xyz, quat_mul
from eden.utils.sample import sample_uniform
from eden.utils.misc import env_indices
from eden.utils.torch import compile_model, device_constant

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@compile_model
def _reward_orientation_fused(
    forward: torch.Tensor,
    desired: torch.Tensor,
) -> torch.Tensor:
    cos_dist = (forward * desired).sum(dim=1)
    normalized = 0.5 * cos_dist + 0.5
    return normalized * normalized


@REWARD_TERM_REGISTRY.register()
def handstand_orientation(
    env: EnvBase,
    *,
    entity_name: str,
    desired_forward: Tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> torch.Tensor:
    """Reward body x-axis aligning with desired direction: (0.5*cos+0.5)^2.

    For handstand desired_forward=(0,0,-1) (body forward points down).
    """
    robot: Entity = env.entities[entity_name]
    quat = robot.get_quat()
    x_axis = device_constant((1.0, 0.0, 0.0), env.device).expand(env.num_envs, -1)
    forward = transform_by_quat(x_axis, quat)
    desired = device_constant(desired_forward, env.device).expand(env.num_envs, -1)
    return _reward_orientation_fused(forward, desired)


@compile_model
def _reward_height_fused(torso_z: torch.Tensor, z_des: float) -> torch.Tensor:
    height = torch.clamp(torso_z, max=z_des)
    error = z_des - height
    return torch.exp(-error)


@REWARD_TERM_REGISTRY.register()
def handstand_height(
    env: EnvBase,
    *,
    entity_name: str,
    z_des: float = 0.55,
) -> torch.Tensor:
    """Reward torso height: exp(-(z_des - min(h, z_des)))."""
    robot: Entity = env.entities[entity_name]
    torso_z = robot.get_pos()[:, 2]
    return _reward_height_fused(torso_z, z_des)


@compile_model
def _compute_rear_feet_contact(feet_z: torch.Tensor, threshold: float) -> torch.Tensor:
    return (feet_z < threshold).any(dim=1).float()


@REWARD_TERM_REGISTRY.register()
class HandstandRearFeetContact(RewardTerm):
    """Penalize rear feet touching the floor (detected via link z-position).

    The link indices are resolved once in :meth:`build`.
    """

    entity_name: str = "robot"
    links_name: Tuple[str, ...] = ("RL_foot", "RR_foot")
    contact_threshold: float = 0.03

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        _, self._idx = self._robot.find_named_links_idx_local(list(self.links_name))

    def compute(self) -> torch.Tensor:
        link_pos = self._robot.get_links_pos(ls_idx_local=self._idx)  # (N, num_links, 3)
        return _compute_rear_feet_contact(link_pos[:, :, 2], self.contact_threshold)


@compile_model
def _compute_pose_cost(dofs_pos: torch.Tensor, default_pos: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(dofs_pos - default_pos), dim=1)


@REWARD_TERM_REGISTRY.register()
def handstand_pose(
    env: EnvBase,
    *,
    entity_name: str,
    dofs_indices: Tuple[int, ...] = (1, 3, 5, 7, 9, 11),
) -> torch.Tensor:
    """Penalize specific joints deviating from default pose.

    Default indices (1,3,5,7,9,11) correspond to rear leg joints in Eden's
    Go1 DOF ordering (RL_hip, RR_hip, RL_thigh, RR_thigh, RL_calf, RR_calf).
    """
    robot: Entity = env.entities[entity_name]
    # Built once per distinct tuple (`device_constant`): a list index uploads on every call.
    idx = device_constant(tuple(dofs_indices), env.device, dtype=torch.long)
    dofs_pos = robot.get_dofs_pos()[:, idx]
    default_pos = robot.default_dofs_pos
    if default_pos.dim() > 1:
        default_pos = default_pos[:, idx]
    else:
        default_pos = default_pos[idx].unsqueeze(0)
    return _compute_pose_cost(dofs_pos, default_pos)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@TERMINATION_TERM_REGISTRY.register()
def handstand_fall(
    env: EnvBase,
    *,
    entity_name: str,
    threshold: float = -0.25,
) -> torch.Tensor:
    """Terminate if body z-axis in world frame has z-component below threshold.

    During a proper handstand, the body is pitched ~90°, so its z-axis
    is roughly horizontal (z-component ≈ 0). Falling on the back causes
    the z-axis to point downward (z-component < 0).
    """
    robot: Entity = env.entities[entity_name]
    quat = robot.get_quat()
    z_axis = device_constant((0.0, 0.0, 1.0), env.device).expand(env.num_envs, -1)
    body_z_world = transform_by_quat(z_axis, quat)
    return body_z_world[:, 2] < threshold


@TERMINATION_TERM_REGISTRY.register()
class HandstandFrontBodyContact(TerminationTerm):
    """Terminate if any front body part (calf, thigh, hip) touches the floor.

    Uses z-position of links as proxy since only feet have collision geometry. Matches the reference's
    contact_termination which checks front calves, thighs, and hips via contact sensors. The link indices
    are resolved once in :meth:`build`.
    """

    entity_name: str = "robot"
    links_name: Tuple[str, ...] = (
        "FL_calf",
        "FR_calf",
        "FL_thigh",
        "FR_thigh",
        "FL_hip",
        "FR_hip",
    )
    contact_threshold: float = 0.05

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        _, self._idx = self._robot.find_named_links_idx_local(list(self.links_name))

    def compute(self) -> torch.Tensor:
        link_pos = self._robot.get_links_pos(ls_idx_local=self._idx)  # (N, num_links, 3)
        return (link_pos[:, :, 2] < self.contact_threshold).any(dim=1)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@OBSERVATION_TERM_REGISTRY.register()
def handstand_last_action(env: EnvBase) -> torch.Tensor:
    """Return the raw policy delta from the previous control step.

    The ACCUMULATE controller stores the raw delta in the action manager, so this is
    just ``prev_action`` — i.e. the obs sees the delta applied *before* the current
    step (matching the reference, where ``last_act`` updates after obs). On the first
    observation(s) of a new episode this reads 0, because the action manager zeroes
    ``prev_action`` on reset (the pre-reset delta does not leak into the new episode).
    """
    return env.action_manager.prev_action


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@EVENT_TERM_REGISTRY.register()
class RandomStandingReset(EventTerm):
    """Reset to default standing pose with randomized xy position, yaw, and root velocity.

    Ported from the handstand reference reset logic.
    """

    entity_name: str = "robot"
    xy_range: Tuple[float, float] = (-0.5, 0.5)
    yaw_range: Tuple[float, float] = (-3.14, 3.14)
    root_vel_range: Tuple[float, float] = (-0.5, 0.5)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.entity: Entity | None = None

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]

    def compute(self, envs_idx: Optional[slice | torch.Tensor] = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Int ids, once: every write below takes the selected rows, a sync per gather and per Genesis
        # setter through the auto-reset's bool mask and none through ids.
        envs_idx = env_indices(envs_idx)

        n = self._env.num_envs
        device = self._env.device

        # Default standing position with random xy offset
        default_pos = device_constant(self.entity._options.default_root_pos, device)
        pos = default_pos.unsqueeze(0).expand(n, -1).clone()
        dxy = sample_uniform(self.xy_range[0], self.xy_range[1], (n, 2), device=device)
        pos[:, :2] += dxy
        self.entity.set_pos(pos[envs_idx], envs_idx=envs_idx)

        # Default orientation with random yaw
        default_quat = self.entity.default_root_quat.expand(n, -1).clone()
        yaw = sample_uniform(self.yaw_range[0], self.yaw_range[1], (n, 1), device=device).squeeze(1)
        zero = torch.zeros_like(yaw)
        new_quat = quat_mul(default_quat, quat_from_euler_xyz(zero, zero, yaw))
        self.entity.set_quat(new_quat[envs_idx], envs_idx=envs_idx)

        # Default joint positions
        default_dofs = self.entity.default_dofs_pos.expand(n, -1)
        self.entity.set_dofs_pos(default_dofs[envs_idx], envs_idx=envs_idx)

        # Random root velocity
        vel = sample_uniform(
            self.root_vel_range[0],
            self.root_vel_range[1],
            (n, 6),
            device=device,
        )
        # The list, not a tensor built per reset: the entity resolves it to a cached `range`.
        base_dofs_index = self.entity.find_free_base_dofs_idx_local()
        self.entity.set_dofs_vel(vel[envs_idx], dofs_idx_local=base_dofs_index, envs_idx=envs_idx)
