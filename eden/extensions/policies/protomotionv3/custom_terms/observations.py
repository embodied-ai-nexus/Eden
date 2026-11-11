"""Observation terms for the ProtoMotions v3 actor, ported from the obs pipeline its ONNX exporter bakes in.

``build_reduced_coords_obs`` reads gravity in the anchor (torso) frame, ``build_reduced_coords_target_poses``
reads four future reference frames relative to the current anchor, and
``historical_previous_processed_actions`` feeds back the joint targets the actor last commanded.

The rotation scalars (projected gravity, tangent/normal of the relative anchor rotation) are all
``R(q) @ v`` for fixed canonical vectors, which does not depend on whether ``q`` is stored wxyz or
xyzw, so these terms work in Genesis's wxyz throughout; ``eden.utils.geom.quat_to_tan_norm``
reproduces ``protomotions.utils.rotations.quat_to_tan_norm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat, transform_quat_by_quat

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import gravity_vec, projected_gravity
from eden.utils.geom import quat_to_tan_norm, yaw_quat
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.managers.terms.commands.mimic import MotionCommand


@OBSERVATION_TERM_REGISTRY.register()
class ProtomotionAnchorGravity(ObservationTerm):
    """Gravity direction in the anchor link's frame, ``(num_envs, 3)``: ``R(anchor)^T @ [0, 0, -1]``."""

    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"

    def build(self) -> None:
        robot = self.resolve_entity(self.entity_name)
        _, idx = robot.find_named_links_idx_local([self.anchor_link_name], preserve_order=True)
        self._anchor_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        self._gravity = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        anchor_quat = self.resolve_entity(self.entity_name).get_links_quat(ls_idx_local=self._anchor_idx)[:, 0, :]
        return projected_gravity(anchor_quat, self._gravity)


@OBSERVATION_TERM_REGISTRY.register()
class ProtomotionTargetPoses(ObservationTerm):
    """Future reference frames relative to the current anchor, ``(num_envs, K * (6 + 2 * num_dofs))``.

    Per future offset ``k`` (frames ahead of the command's current one, clamped to the clip's last
    frame), in ``build_reduced_coords_target_poses`` order: the tangent/normal of the reference
    anchor's rotation relative to the robot's anchor, then the reference joint velocities, then the
    reference joint positions.

    The reference rotations are first turned by a yaw offset that aligns the clip's heading with the
    robot's. It is latched in :meth:`post_reset`, from the final post-reset state, and held for the
    episode; a deployment re-latches through the same hook whenever it restarts the observations
    (reset, START and a policy switch: ``DeploymentBase.restart_observations``).
    """

    command_name: str = "motion"
    entity_name: str = "robot"
    anchor_link_name: str = "torso_link"
    future_offsets: tuple[int, ...] = (1, 2, 4, 8)

    def build(self) -> None:
        robot = self.resolve_entity(self.entity_name)
        _, idx = robot.find_named_links_idx_local([self.anchor_link_name], preserve_order=True)
        self._anchor_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        command = self._command()
        if self.anchor_link_name not in command.links_name:
            raise ValueError(
                f"{term_subject(self)}: the '{self.command_name}' command tracks links {command.links_name}, "
                f"not the anchor '{self.anchor_link_name}'. Add it to the command's links_name."
            )
        self._anchor_in_motion = command.links_name.index(self.anchor_link_name)
        self._offsets = torch.as_tensor(self.future_offsets, dtype=torch.long, device=self.device)
        self._heading_offset = torch.zeros(self.num_envs, 4, device=self.device)
        self._heading_offset[:, 0] = 1.0  # identity until the first reset latches it

    def _command(self) -> MotionCommand:
        return self._env.command_manager.get_term(self.command_name)

    def _anchor_quat(self) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_links_quat(ls_idx_local=self._anchor_idx)[:, 0, :]

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        """Latch the yaw offset from the clip's anchor at its current frame to the robot's anchor."""
        robot_yaw = yaw_quat(self._anchor_quat()[envs_idx])
        motion_yaw = yaw_quat(self._command().links_quat_w[envs_idx, self._anchor_in_motion])
        self._heading_offset[envs_idx] = transform_quat_by_quat(inv_quat(motion_yaw), robot_yaw)

    def compute(self) -> torch.Tensor:
        command = self._command()
        max_idx = command.frame_total_per_env() - 1  # (N,) per-env clip length - 1
        future_t = (command.time_steps[:, None] + self._offsets[None, :]).clamp_max(max_idx[:, None])  # (N, K)
        future_dof_pos = command.gather_motion_frames("dofs_pos", future_t)  # (N, K, n_dofs)
        future_dof_vel = command.gather_motion_frames("dofs_vel", future_t)  # (N, K, n_dofs)
        future_anchor = command.gather_motion_frames("links_quat_w", future_t)[..., self._anchor_in_motion, :]
        future_anchor = transform_quat_by_quat(future_anchor, self._heading_offset[:, None, :].expand_as(future_anchor))

        n, k = future_anchor.shape[:2]
        current = self._anchor_quat().unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
        # rel = inv(current) * future; transform_quat_by_quat(v, u) returns u * v.
        relative = quat_to_tan_norm(transform_quat_by_quat(future_anchor.reshape(-1, 4), inv_quat(current)))
        n_dofs = future_dof_pos.shape[-1]
        per_step = torch.cat([relative, future_dof_vel.reshape(-1, n_dofs), future_dof_pos.reshape(-1, n_dofs)], dim=-1)
        return per_step.view(n, -1)


@OBSERVATION_TERM_REGISTRY.register()
class ProtomotionPrevTargets(ObservationTerm):
    """The joint targets the action term applied on the last step, ``(num_envs, num_dofs)``; zero after a reset.

    The *processed* action (``offset + scale * raw``), which is what ProtoMotions feeds back as
    ``historical_previous_processed_actions``, where the core ``last_action`` term reads the raw one.
    It reads zero from a reset until the env's next action, as the source's buffer does.
    """

    action_name: str = "dofs_pos_controller"

    def build(self) -> None:
        self._env.action_manager.get_term(self.action_name)  # a bad name fails at construction
        self._since_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        self._since_reset[slice(None) if envs_idx is None else envs_idx] = True

    def post_physics_refresh(self) -> None:
        """An action was applied this step, so every env now has targets to report."""
        self._since_reset.fill_(False)

    def compute(self) -> torch.Tensor:
        targets = self._env.action_manager.get_term(self.action_name).processed_action
        return torch.where(self._since_reset[:, None], torch.zeros_like(targets), targets)
