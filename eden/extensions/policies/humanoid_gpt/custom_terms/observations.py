"""Observation terms for the Humanoid-GPT actor: its joint pose about the nominal stance, and the reference block.

The reference block reads the motion through a cache the term builds once from the
:class:`~eden.managers.terms.commands.mimic.MotionCommand` library, EMA-smoothed with velocities
recomputed, as the source's ``apply_ema_qpos`` + ``recompute_qvel`` do before ``get_nn_state``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

from eden.extensions.policies.humanoid_gpt.constants import HGPT_DEFAULT_DOFS_POS, PELVIS_COM_OFFSET
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import gravity_vec, projected_gravity
from eden.utils.geom import quat_box_minus, wrap_to_pi, yaw_from_quat, yaw_quat
from eden.utils.string import term_subject
from eden.utils.torch import device_constant

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.terms.commands.mimic import MotionCommand

# ``_ema`` / ``_ema_quat`` / ``_recompute_root_vel`` faithfully port the reference
# ``apply_ema_qpos`` + ``recompute_qvel``: alpha on the running value, backward
# finite differences. They feed the 6-dim ``ref_cvel`` obs (MuJoCo ``cvel`` on the
# smoothed qpos + recomputed qvel).


def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Causal EMA along dim 0 (``apply_ema_qpos``): ``out[t] = alpha*out[t-1] +
    (1-alpha)*x[t]`` — alpha weights history (0.8 => 80% history).
    """
    out = x.clone()
    for t in range(1, x.shape[0]):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * x[t]
    return out


def _ema_quat(q_wxyz: torch.Tensor, alpha: float) -> torch.Tensor:
    """Causal quaternion EMA (wxyz, time = dim 0): hemisphere-aligned lerp +
    renormalize, alpha on the running value. Batches over trailing dims.
    """
    out = q_wxyz.clone()
    for t in range(1, q_wxyz.shape[0]):
        q_t = q_wxyz[t]
        # flip the new sample into the running mean's hemisphere before the lerp
        dot = (q_t * out[t - 1]).sum(dim=-1, keepdim=True)
        q_t = torch.where(dot < 0.0, -q_t, q_t)
        blended = alpha * out[t - 1] + (1.0 - alpha) * q_t
        out[t] = blended / blended.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return out


def _recompute_root_vel(pos: torch.Tensor, quat_wxyz: torch.Tensor, fps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Root lin/ang velocity (world frame) by backward finite difference
    (``recompute_qvel``, frame="world"): velocity at frame ``t`` is the t-1 -> t
    delta; frame 0 repeats frame 1. ``ang`` is the world-frame delta axis-angle.
    """
    lin = torch.zeros_like(pos)
    ang = torch.zeros_like(pos)
    if pos.shape[0] > 1:
        lin[1:] = (pos[1:] - pos[:-1]) * fps
        lin[0] = lin[1]
        ang[1:] = quat_box_minus(quat_wxyz[1:], quat_wxyz[:-1]) * fps
        ang[0] = ang[1]
    return lin, ang


@OBSERVATION_TERM_REGISTRY.register()
def hgpt_dofs_pos_rel(env: RLEnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Joint positions about Humanoid-GPT's nominal stance, ``(num_envs, 29)`` in ``dofs_name`` order.

    The nominal stance is :data:`HGPT_DEFAULT_DOFS_POS`, not the robot's ``default_dofs_pos``: the
    policy runs on the stock ``G1Dofs29``, whose default pose differs from it on six joints.
    """
    nominal = device_constant(HGPT_DEFAULT_DOFS_POS, env.device, dtype=torch.float32)
    return env.get_entity(entity_name).get_dofs_pos() - nominal


@OBSERVATION_TERM_REGISTRY.register()
class HgptReference(ObservationTerm):
    """The 43-dim reference block of Humanoid-GPT's observation, ``(num_envs, 43)``.

    With ``t`` the command's current frame and ``t_next = min(t + 1, last)``::

        ref_q_next - nominal (29)   reference joint pos at t_next, about the nominal stance
        ref_height (1)              reference pelvis z at t_next
        ref_gvec (3)                gravity in the reference pelvis frame at t_next
        ref_cvel_gv (6)             [angular, linear] reference pelvis twist at t_next, in its yaw frame
        yaw_cmd (2)                 [cos, sin] of wrap(yaw(ref@t) - yaw(robot))
        xy_cmd (2)                  reference xy at t minus the robot's, in the robot's yaw frame

    Parameters
    ----------
    command_name : str
        The :class:`~eden.managers.terms.commands.mimic.MotionCommand` to read; its ``links_name``
        must include ``root_link_name``.
    root_link_name : str
        The reference link the block describes: the pelvis, the robot's root.
    entity_name : str
        The robot the yaw and xy commands are relative to.
    ema_alpha : float | None
        Weight on the running value of the reference EMA (0.8 keeps 80% history). ``None`` reads the
        raw clip, and its loader's velocities.
    cvel_at_com : bool
        Report the reference pelvis velocity at its centre of mass, as MuJoCo's ``cvel`` does,
        rather than at the link origin.
    """

    command_name: str = "motion"
    root_link_name: str = "pelvis"
    entity_name: str = "robot"
    ema_alpha: float | None = 0.8
    cvel_at_com: bool = True

    def build(self) -> None:
        self.resolve_entity(self.entity_name)  # touched now, so a missing robot fails at construction
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        if self.root_link_name not in command.links_name:
            raise ValueError(
                f"{term_subject(self)}: the '{self.command_name}' command tracks links {command.links_name}, "
                f"not the root '{self.root_link_name}'. Add it to the command's links_name."
            )
        root = command.links_name.index(self.root_link_name)
        motion = command.motion
        # Library tracks are (M, T_max, ...), indexed per env by (motion_id, frame).
        pos = motion.links_pos_w[..., root, :].clone()  # (M, T, 3)
        quat = motion.links_quat_w[..., root, :].clone()  # (M, T, 4) wxyz
        dofs = motion.dofs_pos.clone()  # (M, T, 29)
        lin = motion.links_lin_vel_w[..., root, :].clone()  # (M, T, 3)
        ang = motion.links_ang_vel_w[..., root, :].clone()  # (M, T, 3)
        if self.ema_alpha is not None:
            alpha = float(self.ema_alpha)
            env_options = self._env.env_options
            fps = 1.0 / (env_options.sim_dt * env_options.decimation)
            # time -> dim 0, run the batch-aware helpers once over all clips, restore.
            pos, dofs, quat = (x.transpose(0, 1).contiguous() for x in (pos, dofs, quat))
            pos, dofs, quat = _ema(pos, alpha), _ema(dofs, alpha), _ema_quat(quat, alpha)
            lin, ang = _recompute_root_vel(pos, quat, fps)
            pos, dofs, quat, lin, ang = (x.transpose(0, 1).contiguous() for x in (pos, dofs, quat, lin, ang))
        self._ref_pos, self._ref_quat, self._ref_dofs = pos, quat, dofs
        self._ref_lin, self._ref_ang = lin, ang
        self._nominal = torch.tensor(HGPT_DEFAULT_DOFS_POS, device=self.device)
        self._gravity = gravity_vec(self.num_envs, self.device)
        self._pelvis_com_offset = torch.tensor(PELVIS_COM_OFFSET, device=self.device).expand(self.num_envs, 3)

    def compute(self) -> torch.Tensor:
        command: MotionCommand = self._env.command_manager.get_term(self.command_name)
        robot = self.resolve_entity(self.entity_name)
        t, mid = command.time_steps, command.motion_id
        t_next = torch.minimum(t + 1, command.frame_total - 1)  # clamped to each env's own clip

        ref_q_next = self._ref_dofs[mid, t_next]
        ref_pelvis_pos_next = self._ref_pos[mid, t_next]
        ref_pelvis_quat_next = self._ref_quat[mid, t_next]
        ref_lin_next = self._ref_lin[mid, t_next]
        ref_ang_next = self._ref_ang[mid, t_next]
        ref_pelvis_pos_curr = self._ref_pos[mid, t]
        ref_pelvis_quat_curr = self._ref_quat[mid, t]

        ref_height = ref_pelvis_pos_next[..., 2:3]
        ref_gvec = projected_gravity(ref_pelvis_quat_next, self._gravity)
        # Root twist in the reference's yaw-only frame, [angular, linear] (MuJoCo cvel order).
        if self.cvel_at_com:
            # cvel linear is at the body CoM: v_com = v_origin + w x r_world.
            r_world = transform_by_quat(self._pelvis_com_offset, ref_pelvis_quat_next)
            ref_lin_next = ref_lin_next + torch.cross(ref_ang_next, r_world, dim=-1)
        ref_yaw = yaw_quat(ref_pelvis_quat_next)
        ref_cvel_gv = torch.cat(
            [inv_transform_by_quat(ref_ang_next, ref_yaw), inv_transform_by_quat(ref_lin_next, ref_yaw)], dim=-1
        )

        # Heading command: yaw(ref@t) - yaw(robot). ``yaw_from_quat`` == Humanoid-GPT's quat2yaw.
        pelvis_quat = robot.get_quat()
        yaw_d = wrap_to_pi(yaw_from_quat(ref_pelvis_quat_curr) - yaw_from_quat(pelvis_quat))
        yaw_cmd = torch.stack([torch.cos(yaw_d), torch.sin(yaw_d)], dim=-1)
        # XY position command rotated into the robot's heading frame: R(-yaw_robot) @ dxy.
        dxy = torch.zeros_like(pelvis_quat[..., :3])
        dxy[..., :2] = ref_pelvis_pos_curr[..., :2] - robot.get_pos()[..., :2]
        xy_cmd = inv_transform_by_quat(dxy, yaw_quat(pelvis_quat))[..., :2]

        return torch.cat([ref_q_next - self._nominal, ref_height, ref_gvec, ref_cvel_gv, yaw_cmd, xy_cmd], dim=-1)
