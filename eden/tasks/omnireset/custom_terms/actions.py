"""UWLab's relative Cartesian operational-space controller for the UR5e (``RelCartesianOSCAction``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import quat_to_R

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.geom import (
    quat_box_minus,
    quat_from_axis_angle,
    quat_mul,
    subtract_frame_transforms,
)
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


@ACTION_TERM_REGISTRY.register()
class RelCartesianOSC(ActionTerm):
    """Track a pose delta of ``ee_link_name`` in the robot base frame with ``tau = J^T (Kp e - Kd J qdot)``.

    Once per policy step the 6-D action, scaled per axis, offsets the *measured* end-effector pose: translation
    added, axis-angle rotation composed on the left. Every physics step the arm torques are recomputed from the
    pose error to that held target and the end-effector velocity ``J qdot``, then clipped to ``torque_limit``.
    No inertial decoupling, no gravity term: UWLab's robot floats (gravity compensation here).

    Why not :class:`~eden.managers.terms.actions.OperationalSpaceController`: it refuses an entity with DOFs
    outside ``dofs_name`` (it would need them in its mass matrix), and the 2F-85's passive linkage joints are
    exactly that. With no inertial decoupling none of them enter the control law. UWLab uses a calibrated
    analytic Jacobian; the converted robot's kinematics match it to 1e-7 m, so Genesis' Jacobian is the same one.

    Parameters
    ----------
    ee_link_name:
        Link whose origin is controlled (UWLab: ``wrist_3_link``).
    base_link_name:
        Link whose frame the deltas, Jacobian and error live in (the robot's root, ``base_link``).
    scale:
        Per-axis action scale (m, m, m, rad, rad, rad).
    kp, damping_ratio:
        Per-axis task-space stiffness and damping ratio; ``kd = 2 * damping_ratio * sqrt(kp)``.
    torque_limit:
        Symmetric per-joint torque clip, N·m, in ``dofs_name`` order.
    """

    ee_link_name: str = "wrist_3_link"
    base_link_name: str = "base_link"
    scale: tuple[float, float, float, float, float, float] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.2)
    kp: tuple[float, float, float, float, float, float] = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
    damping_ratio: tuple[float, float, float, float, float, float] = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
    torque_limit: tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)

    def __init__(self, env: EnvBase, options: ActionTermOptions) -> None:
        super().__init__(env=env, options=options)
        self._goal_pos: torch.Tensor | None = None
        self._goal_quat: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return 6

    def build(self) -> None:
        super().build()
        if len(self.torque_limit) != self.num_dofs:
            raise ValueError(
                f"{term_subject(self)}: torque_limit has {len(self.torque_limit)} values for {self.num_dofs} DOFs "
                f"{list(self.dofs_name)}; give one per DOF."
            )
        entity = self.entity
        self._ee_link = int(resolve_link_idx(entity, [self.ee_link_name], self.device)[0])
        self._base_link = int(resolve_link_idx(entity, [self.base_link_name], self.device)[0])
        # Jacobian columns come in the entity's dofs_name order; keep this term's DOFs.
        self._jacobian_columns = torch.tensor(
            [entity.dofs_name.index(name) for name in self.dofs_name], dtype=torch.long, device=self.device
        )
        as_row = dict(dtype=torch.float32, device=self.device)
        self._scale = torch.tensor(self.scale, **as_row)
        self._kp = torch.tensor(self.kp, **as_row)
        self._kd = 2.0 * torch.tensor(self.damping_ratio, **as_row) * torch.sqrt(self._kp)
        self._torque_limit = torch.tensor(self.torque_limit, **as_row)
        self._goal_pos, self._goal_quat = self._ee_pose_in_base()

    def _ee_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        entity = self.resolve_entity(self.entity_name)
        links = [self._base_link, self._ee_link]
        pos, quat = entity.get_links_pos(ls_idx_local=links), entity.get_links_quat(ls_idx_local=links)
        return subtract_frame_transforms(pos[:, 0], quat[:, 0], pos[:, 1], quat[:, 1])

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        scaled = actions * self._scale
        ee_pos, ee_quat = self._ee_pose_in_base()
        self._goal_pos = ee_pos + scaled[:, :3]
        self._goal_quat = quat_mul(quat_from_axis_angle(scaled[:, 3:]), ee_quat)

    def apply_actions(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        ee_pos, ee_quat = self._ee_pose_in_base()
        base_quat = entity.get_links_quat(ls_idx_local=[self._base_link])[:, 0]
        jacobian_w = entity.get_jacobian(link=entity.links[self._ee_link]).index_select(-1, self._jacobian_columns)
        # World -> base frame: rotate the linear and the angular rows alike.
        base_rot_t = quat_to_R(base_quat).transpose(-1, -2)
        jacobian = torch.cat([base_rot_t @ jacobian_w[:, :3], base_rot_t @ jacobian_w[:, 3:]], dim=1)
        dofs_vel = entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)
        ee_vel = torch.bmm(jacobian, dofs_vel.unsqueeze(-1)).squeeze(-1)
        error = torch.cat([self._goal_pos - ee_pos, quat_box_minus(self._goal_quat, ee_quat)], dim=-1)
        wrench = self._kp * error - self._kd * ee_vel
        torque = torch.bmm(jacobian.transpose(1, 2), wrench.unsqueeze(-1)).squeeze(-1)
        self._processed_action = torch.clamp(torque, -self._torque_limit, self._torque_limit)
        self.verify_command_shape(self._processed_action)
        entity.control_dofs_force(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._raw_action[envs_idx] = 0.0
        self._prev_action[envs_idx] = 0.0
        ee_pos, ee_quat = self._ee_pose_in_base()
        self._goal_pos[envs_idx] = ee_pos[envs_idx]
        self._goal_quat[envs_idx] = ee_quat[envs_idx]
