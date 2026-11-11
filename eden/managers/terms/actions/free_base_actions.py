"""Pose control of a floating base: a hand or gripper that flies, with no arm under it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, quat_to_xyz, transform_by_quat

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.utils.geom import quat_from_axis_angle, quat_mul
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


@ACTION_TERM_REGISTRY.register()
class FreeBasePoseController(ActionTerm):
    """Drive a floating base to a pose near its default root pose (Genesis free-joint PD).

    For an entity that stands on nothing (a dexterous hand or a gripper with no arm: a "flying" end effector), this
    term is the arm. Its action is six numbers per environment: a translation along the world axes, at
    ``position_scale`` metres per unit, then a rotation vector about the world axes, at ``orientation_scale``
    radians per unit. Both are applied to the entity's default root pose, about its authored origin (a hand's
    wrist), so the target is absolute: a zero action holds the default pose.

    The target orientation is composed as a quaternion and only then written as the free joint's intrinsic xyz
    Euler angles, which Genesis turns straight back into a quaternion to take a rotation-vector error. Nothing is
    added in Euler space, so a default orientation at the Euler singularity (pitch at 90 degrees, which a palm-down
    hand is) costs nothing.

    Genesis position control places the solver frame, which for a re-origined body is not the authored frame that
    ``default_root_pos`` / ``default_root_quat`` describe (see ``docs/guides/entities.md``). The offset between them
    is measured once at build and applied to every target.

    The gains below are written to the six free-joint DOFs at build. Nothing else sets them: Eden applies
    ``default_dofs_kp`` / ``default_dofs_kd`` to the joints in ``dofs_name`` only, and Genesis leaves a free joint
    at zero gain with an unbounded force range. Gravity is not in the loop. Give the entity
    ``RigidMaterialOptions(gravity_compensation=1.0)``, as a real arm's controller would, or it hangs
    ``m * g / position_kp`` below its target and the orientation loop carries the weight's moment about the origin.

    Parameters
    ----------
    entity_name : str
        The floating-base entity. It must have exactly one free joint and must not be attached to another entity.
    position_scale : float | tuple[float, float, float]
        Metres per unit of translation action, scalar or per world axis.
    orientation_scale : float | tuple[float, float, float]
        Radians per unit of rotation-vector action, scalar or per world axis.
    position_kp : float
        Translational stiffness, N/m.
    position_kd : float
        Translational damping, N·s/m.
    orientation_kp : float
        Rotational stiffness, N·m/rad.
    orientation_kd : float
        Rotational damping, N·m·s/rad.
    force_limit : float
        Bound on each translational DOF's control force, N.
    torque_limit : float
        Bound on each rotational DOF's control torque, N·m.
    """

    position_scale: float | tuple[float, float, float] = 0.05
    orientation_scale: float | tuple[float, float, float] = 0.2
    position_kp: float = 1000.0
    position_kd: float = 60.0
    orientation_kp: float = 40.0
    orientation_kd: float = 2.0
    force_limit: float = 100.0
    torque_limit: float = 10.0

    def build(self) -> None:
        if self.dofs_name:
            raise ValueError(
                f"{term_subject(self)} drives the entity's free joint, so it takes no "
                f"`dofs_name`; got {self.dofs_name!r}. Drive the other joints with a joint controller of their own."
            )
        gains = {
            "position_kp": self.position_kp,
            "position_kd": self.position_kd,
            "orientation_kp": self.orientation_kp,
            "orientation_kd": self.orientation_kd,
            "force_limit": self.force_limit,
            "torque_limit": self.torque_limit,
        }
        negative = {name: value for name, value in gains.items() if value < 0.0}
        if negative:
            raise ValueError(f"{term_subject(self)}: gains must be >= 0, got {negative}.")

        self._entity = self.resolve_entity(self.entity_name)
        entity: RigidEntity = self._entity
        # Raises, naming the joints, unless there is exactly one free joint: a fixed or attached base has none.
        base_dofs = entity.find_free_base_dofs_idx_local()
        self.dofs_idx_local = torch.as_tensor(base_dofs, dtype=gs.tc_int, device=self.device)
        self._n_dofs = len(base_dofs)

        self._position_scale = self._per_axis(self.position_scale, "position_scale")
        self._orientation_scale = self._per_axis(self.orientation_scale, "orientation_scale")

        kp = torch.tensor([self.position_kp] * 3 + [self.orientation_kp] * 3, dtype=gs.tc_float, device=self.device)
        kd = torch.tensor([self.position_kd] * 3 + [self.orientation_kd] * 3, dtype=gs.tc_float, device=self.device)
        limit = torch.tensor([self.force_limit] * 3 + [self.torque_limit] * 3, dtype=gs.tc_float, device=self.device)
        if entity.has_batched_dofs_info:
            kp, kd, limit = (value.expand(self.num_envs, -1) for value in (kp, kd, limit))
        entity.set_dofs_kp(kp, dofs_idx_local=self.genesis_dofs_idx)
        entity.set_dofs_kd(kd, dofs_idx_local=self.genesis_dofs_idx)
        entity.set_dofs_force_range(-limit, limit, dofs_idx_local=self.genesis_dofs_idx)
        if entity._needs_explicit_damping_armature:
            # The same first-order implicit damping Eden folds into an actuated joint's armature (see RigidEntity).
            armature = entity.get_dofs_armature(dofs_idx_local=self.genesis_dofs_idx)
            substep_dt = self._env.scene.sim._substep_dt
            entity.set_dofs_armature(
                (armature if entity.has_batched_dofs_info else armature[0]) + kd * substep_dt,
                dofs_idx_local=self.genesis_dofs_idx,
            )

        # Solver frame relative to the authored frame, in the authored frame: a rigid offset of the body, so reading
        # it wherever the entity happens to be is exact.
        authored_pos, authored_quat = entity.get_pos(relative=True), entity.get_quat(relative=True)
        self._offset_pos = inv_transform_by_quat(entity.get_pos() - authored_pos, authored_quat)
        self._offset_quat = quat_mul(inv_quat(authored_quat), entity.get_quat())
        self._default_pos, self._default_quat = entity.get_default_root_pose()

        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_action = torch.zeros(self.num_envs, self._n_dofs, device=self.device)

    def _per_axis(self, value: float | tuple[float, float, float], name: str) -> torch.Tensor:
        values = (value,) * 3 if isinstance(value, float | int) else tuple(value)
        if len(values) != 3:
            raise ValueError(
                f"{term_subject(self)}: `{name}` takes one value or one per world axis (x, y, z); got {value!r}."
            )
        return torch.tensor(values, dtype=gs.tc_float, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    def target_pose(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The authored-frame pose ``actions`` asks for: the default root pose, moved and then turned.

        Parameters
        ----------
        actions : torch.Tensor
            ``(num_envs, 6)``: translation, then rotation vector, in action units.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Position ``(num_envs, 3)`` and wxyz quaternion ``(num_envs, 4)``.
        """
        pos = self._default_pos + actions[:, :3] * self._position_scale
        quat = quat_mul(quat_from_axis_angle(actions[:, 3:] * self._orientation_scale), self._default_quat)
        return pos, quat

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        pos, quat = self.target_pose(self._raw_action)
        solver_pos = pos + transform_by_quat(self._offset_pos, quat)
        solver_quat = quat_mul(quat, self._offset_quat)
        # A free joint's DOF positions are its position and its intrinsic xyz Euler angles.
        self._processed_action = torch.cat((solver_pos, quat_to_xyz(solver_quat, rpy=False)), dim=-1)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.verify_command_shape(self._processed_action)
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)
