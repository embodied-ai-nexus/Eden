"""The world model's state: generalized coordinates of a floating-base robot, its goals and its analytic reward.

Full state: ``[root_pos(3), root_quat(4), q(J), root_lin_vel_w(3), root_ang_vel_w(3), qd(J)]``.
Model state: ``[z(1), g_b(3), v_b(3), w_b(3), q(J), qd(J)]``, the root height, gravity and twist in the root frame.
Auxiliary state: the root pose ``[root_pos(3), root_quat(4)]``, integrated outside the network.
Prediction: ``[dv_b(3), dw_b(3), dqd(J)]``, velocity changes in the current root frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from eden.utils.geom import (
    R_error_magnitude,
    inv_quat,
    inv_transform_by_quat,
    quat_integrate,
    quat_mul,
    quat_to_R,
    quat_to_rot6d_cols,
    subtract_frame_transforms,
    transform_by_quat,
    yaw_quat,
)

if TYPE_CHECKING:
    from eden.utils.motion_reference.library import MotionLibrary


@dataclass(frozen=True)
class GoalWindow:
    """The reference frames a goal holds: ``frames`` of them, ``stride`` control steps apart, from the current one."""

    frames: int
    stride: int

    def dims(self, num_joints: int) -> tuple[int, int]:
        """``(world-model goal width, actor goal width)``; the actor's goal omits the relative root position."""
        content = 2 * num_joints * self.frames
        return content + 9 * self.frames, content + 6 * self.frames

    def indices(self, library: MotionLibrary, frames: torch.Tensor) -> torch.Tensor:
        """``(E, frames)`` library frames ``f, f + stride, ...``, held at the last frame of each env's clip."""
        offsets = torch.arange(self.frames, device=frames.device) * self.stride
        return torch.minimum(frames[:, None] + offsets, library.last_frame(frames)[:, None])


@dataclass(frozen=True)
class RewardSpec:
    """The weights and kernel widths of the task's reward, which :meth:`GenCoordRepresentation.reward` computes on a
    predicted state. The runner reads them off the task's reward terms
    (:func:`~eden.learning.wrappers.rsl_rl.supermimic.task_contract.task_reward_spec`).

    Attributes
    ----------
    tracking : tuple[tuple[float, float], ...]
        ``(weight, Gaussian kernel std)`` of the six tracking terms: anchor position, anchor orientation, relative
        body position, relative body orientation, body linear velocity, body angular velocity.
    action_rate, joint_limit : float
        Weights of the action-rate and joint-limit penalties.
    """

    tracking: tuple[tuple[float, float], ...]
    action_rate: float
    joint_limit: float

    def __post_init__(self) -> None:
        if len(self.tracking) != 6:
            raise ValueError(f"RewardSpec.tracking holds the six tracking terms, got {len(self.tracking)}.")

    @property
    def max_step_rate(self) -> float:
        """The largest reward rate: every positive weight at a kernel value of 1."""
        weights = [w for w, _ in self.tracking] + [self.action_rate, self.joint_limit]
        return sum(w for w in weights if w > 0)


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for small trailing matrices, without TF32 rounding and fusable by ``torch.compile``."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def anchor_window(
    root_pos: torch.Tensor, root_quat: torch.Tensor, target_pos: torch.Tensor, target_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference root poses ``(E, N, ·)`` in the robot's root frame ``(E, ·)``: positions ``(E, 3N)`` and rot6d
    ``(E, 6N)``.
    """
    num_envs, num_frames = target_pos.shape[:2]
    rel_pos, rel_quat = subtract_frame_transforms(
        root_pos[:, None].expand(num_envs, num_frames, 3).reshape(-1, 3),
        root_quat[:, None].expand(num_envs, num_frames, 4).reshape(-1, 4),
        target_pos.reshape(-1, 3),
        target_quat.reshape(-1, 4),
    )
    return rel_pos.reshape(num_envs, -1), quat_to_rot6d_cols(rel_quat).reshape(num_envs, -1)


def motion_goal(
    library: MotionLibrary, frames: torch.Tensor, aux_state: torch.Tensor, window: GoalWindow
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(world-model goal, actor goal)`` at ``frames`` for robots at root poses ``aux_state``.

    The world-model goal is ``[q, qd per frame, relative root position, relative root rot6d]``; the actor's drops the
    relative root position, which a deployed robot does not know.
    """
    idx = window.indices(library, frames)
    content = torch.cat([library.joint_pos[idx], library.joint_vel[idx]], dim=-1).flatten(1)
    rel_pos, rel_ori = anchor_window(
        aux_state[..., :3], aux_state[..., 3:7], library.root_pos_w[idx], library.root_quat_w[idx]
    )
    return torch.cat([content, rel_pos, rel_ori], dim=-1), torch.cat([content, rel_ori], dim=-1)


#: The terms of the actor's proprioception, in the order :meth:`GenCoordRepresentation.proprio` builds them and the
#: task's actor observation group lists them: projected gravity, body-frame angular velocity, ``q - q0``, ``qd`` and
#: the last action.
PROPRIO_TERMS = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action")


class GenCoordRepresentation(torch.nn.Module):
    """Generalized-coordinate state of the world model (module docstring), with its observation and reward.

    Parameters
    ----------
    fk : torch.nn.Module
        Forward kinematics ``(root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_w, qd) -> (pos, quat, lin, ang)``
        of the tracked bodies, the root first.
    default_joint_pos : torch.Tensor
        ``(J,)`` default pose, subtracted from ``q`` in the observation.
    soft_joint_pos_limits : torch.Tensor
        ``(J, 2)`` soft joint limits of the joint-limit penalty.
    dt : float
        Control period.
    decimation : int
        Physics steps per control step; the pose integrates ``(decimation + 1) / (2 decimation)`` of the velocity
        change, as the simulator's substeps compose under constant acceleration.
    num_actions : int
        Width of the action, the last part of the observation.
    reward : RewardSpec
        The task's reward weights and kernel widths.
    proprio_noise : Sequence[float]
        Uniform noise half-width of each proprioception term ``[projected_gravity, base_ang_vel, q - q0, qd,
        last_action]``.
    """

    def __init__(
        self,
        fk: torch.nn.Module,
        default_joint_pos: torch.Tensor,
        soft_joint_pos_limits: torch.Tensor,
        dt: float,
        decimation: int,
        num_actions: int,
        reward: RewardSpec,
        proprio_noise: Sequence[float],
    ):
        super().__init__()
        self.fk = fk
        self.reward_spec = reward
        self.num_joints = int(default_joint_pos.shape[0])
        self.num_actions = num_actions
        self.full_state_dim = 13 + 2 * self.num_joints
        self.rep_state_dim = 10 + 2 * self.num_joints
        self.prediction_dim = 6 + self.num_joints
        self.dt = dt
        self.substep_coeff = (decimation + 1) / (2 * decimation)
        self.register_buffer("default_joint_pos", default_joint_pos.clone(), persistent=False)
        self.register_buffer("soft_joint_pos_limits", soft_joint_pos_limits.clone(), persistent=False)
        widths = self.proprio_term_widths()
        self.register_buffer(
            "proprio_noise",
            torch.cat([torch.full((w,), h) for w, h in zip(widths, proprio_noise, strict=True)]).to(
                default_joint_pos.device
            ),
            persistent=False,
        )
        #: The channel groups of the state loss, named as :meth:`loss_blocks` returns them.
        self.loss_groups = ("root_pos", "root_ori", "joint_pos", "joint_vel")

    def split_full(self, full_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_w, qd)`` of a full state."""
        j = self.num_joints
        return full_state.split((3, 4, j, 3, 3, j), dim=-1)

    def split_rep(self, rep_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(z, g_b, v_b, w_b, q, qd)`` of a model state."""
        j = self.num_joints
        return rep_state.split((1, 3, 3, 3, j, j), dim=-1)

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        root_pos, root_quat, q, lin, ang, qd = self.split_full(full_state)
        down = torch.zeros_like(lin)
        down[..., 2] = -1.0
        g_b = inv_transform_by_quat(down, root_quat)
        return torch.cat(
            [
                root_pos[..., 2:3],
                g_b,
                inv_transform_by_quat(lin, root_quat),
                inv_transform_by_quat(ang, root_quat),
                q,
                qd,
            ],
            dim=-1,
        )

    @staticmethod
    def aux_state(full_state: torch.Tensor) -> torch.Tensor:
        return full_state[..., :7]

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        """``[q, qd]``."""
        return rep_state[..., 10:]

    def apply_prediction(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the predicted velocity changes semi-implicitly: ``(next aux state, next model state)``.

        Velocities advance by the full change; the joints and the root pose integrate the velocity at
        ``substep_coeff`` of the change, and the root height and gravity follow from the integrated pose.
        """
        c, dt = self.substep_coeff, self.dt
        _, _, v_b, w_b, q, qd = self.split_rep(rep_state)
        dv_b, dw_b, dqd = prediction.split((3, 3, self.num_joints), dim=-1)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        v_w = transform_by_quat(v_b + dv_b, root_quat)
        w_w = transform_by_quat(w_b + dw_b, root_quat)
        next_pos = root_pos + transform_by_quat(v_b + c * dv_b, root_quat) * dt
        next_quat = quat_integrate(root_quat, transform_by_quat(w_b + c * dw_b, root_quat), dt)
        down = torch.zeros_like(v_w)
        down[..., 2] = -1.0
        next_rep = torch.cat(
            [
                next_pos[..., 2:3],
                inv_transform_by_quat(down, next_quat),
                inv_transform_by_quat(v_w, next_quat),
                inv_transform_by_quat(w_w, next_quat),
                q + (qd + c * dqd) * dt,
                qd + dqd,
            ],
            dim=-1,
        )
        return torch.cat([next_pos, next_quat], dim=-1), next_rep

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Channel groups of the state loss. The root rotation is scaled by sqrt(1/2), so its squared difference is
        ``2 (1 - cos theta) ~ theta^2``, in radians like its nominal error.
        """
        rot = quat_to_R(aux_state[..., 3:7]).flatten(-2, -1) * math.sqrt(0.5)
        _, _, _, _, q, qd = self.split_rep(rep_state)
        return {"root_pos": aux_state[..., :3], "root_ori": rot, "joint_pos": q, "joint_vel": qd}

    def proprio_term_widths(self) -> list[int]:
        """Widths of the proprioception terms, in :data:`PROPRIO_TERMS` order."""
        return [3, 3, self.num_joints, self.num_joints, self.num_actions]

    @property
    def proprio_dim(self) -> int:
        return sum(self.proprio_term_widths())

    def add_proprio_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + ((torch.rand_like(obs) * 2 - 1) * self.proprio_noise).detach()

    def proprio(
        self, rep_state: torch.Tensor, last_action: torch.Tensor, encoder_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The proprioception a robot in ``rep_state`` reads, noise-free.

        ``encoder_bias`` is each joint's encoder bias, which shifts the joint reading as the task's biased joint
        observation does; ``None`` reads the joints without one.
        """
        _, g_b, _, w_b, q, qd = self.split_rep(rep_state)
        q_rel = q - self.default_joint_pos
        if encoder_bias is not None:
            q_rel = q_rel + encoder_bias
        return torch.cat([g_b, w_b, q_rel, qd, last_action], dim=-1)

    def bodies(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """World-frame ``(pos, rot_mat, lin_vel, ang_vel)`` of the tracked bodies."""
        _, _, v_b, w_b, q, qd = self.split_rep(rep_state)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        pos, quat, lin, ang = self.fk(
            root_pos, root_quat, q, transform_by_quat(v_b, root_quat), transform_by_quat(w_b, root_quat), qd
        )
        return pos, quat_to_R(quat), lin, ang

    def reward(
        self,
        aux_state: torch.Tensor,
        rep_state: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        target: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """The tracking reward of the task, scaled by the control period.

        ``target`` is the reference ``(pos, quat, lin_vel, ang_vel)`` of the tracked bodies, the root first. Body
        poses are compared with the reference re-anchored to the robot's root xy and heading, as the motion tracking
        terms do.
        """
        pos, rot, lin, ang = self.bodies(aux_state, rep_state)
        t_pos, t_quat, t_lin, t_ang = target
        t_rot = quat_to_R(t_quat)
        delta_pos = torch.cat([aux_state[..., :2], t_pos[..., 0, 2:3]], dim=-1)
        delta_rot = quat_to_R(yaw_quat(quat_mul(aux_state[..., 3:7], inv_quat(t_quat[..., 0, :])))).unsqueeze(-3)
        rel_pos = t_pos - t_pos[..., 0:1, :]
        anchored_pos = delta_pos.unsqueeze(-2) + small_matmul(delta_rot, rel_pos.unsqueeze(-1))[..., 0]
        anchored_rot = small_matmul(delta_rot, t_rot)
        errors = (
            ((pos[..., 0, :] - t_pos[..., 0, :]) ** 2).sum(-1),
            R_error_magnitude(rot[..., 0, :, :], t_rot[..., 0, :, :]) ** 2,
            ((pos - anchored_pos) ** 2).sum(-1).mean(-1),
            (R_error_magnitude(rot, anchored_rot) ** 2).mean(-1),
            ((lin - t_lin) ** 2).sum(-1).mean(-1),
            ((ang - t_ang) ** 2).sum(-1).mean(-1),
        )
        spec = self.reward_spec
        reward = sum(w * torch.exp(-e / std**2) for e, (w, std) in zip(errors, spec.tracking, strict=True))
        action_rate = ((action - last_action) ** 2).sum(-1)
        q = rep_state[..., 10 : 10 + self.num_joints]
        lo, hi = self.soft_joint_pos_limits.unbind(-1)
        joint_limit = ((lo - q).clamp(min=0) + (q - hi).clamp(min=0)).sum(-1)
        penalties = spec.action_rate * action_rate + spec.joint_limit * joint_limit
        return self.dt * (reward + penalties)
