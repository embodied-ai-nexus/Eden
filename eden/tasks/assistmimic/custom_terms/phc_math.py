"""PHC's SMPL observation and reward kernels, in Eden's wxyz convention.

Ports of ``compute_humanoid_observations_smpl_max`` (self observation, 358 dims for 24 bodies),
``compute_imitation_observations_v6`` (task observation, 576 dims) and ``compute_imitation_reward`` from
PHC / AssistMimic, for an upright-start humanoid (so no base-rotation removal). The pretrained ``phc_3``
weights only reproduce PHC's behaviour when these match bit-for-bit in layout, so the order of every
concatenation is PHC's.
"""

from __future__ import annotations

import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from eden.utils.geom import quat_error_magnitude, quat_mul, quat_to_tan_norm, yaw_quat


def heading_quat_inv(root_rot: torch.Tensor) -> torch.Tensor:
    """PHC ``calc_heading_quat_inv``: rotation by minus the heading of ``R(q)·x̂`` about world z."""
    return inv_quat(yaw_quat(root_rot))


def heading_quat(root_rot: torch.Tensor) -> torch.Tensor:
    """PHC ``calc_heading_quat``."""
    return yaw_quat(root_rot)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (..., 3) by ``q`` (..., 4) broadcast over the body axis."""
    q = q.expand(*v.shape[:-1], 4)
    return transform_by_quat(v.reshape(-1, 3), q.reshape(-1, 4)).reshape(v.shape)


def self_obs(body_pos, body_rot, body_vel, body_ang_vel) -> torch.Tensor:
    """``compute_humanoid_observations_smpl_max`` (local root obs, root height obs, no shape params)."""
    root_pos, root_rot = body_pos[:, 0], body_rot[:, 0]
    n, j = body_pos.shape[:2]
    h_inv = heading_quat_inv(root_rot)[:, None]
    local_pos = rotate(h_inv, body_pos - root_pos[:, None])[:, 1:].reshape(n, -1)
    local_rot = quat_mul(h_inv.expand(n, j, 4).reshape(-1, 4), body_rot.reshape(-1, 4))
    local_rot = quat_to_tan_norm(local_rot).reshape(n, -1)
    local_vel = rotate(h_inv, body_vel).reshape(n, -1)
    local_ang_vel = rotate(h_inv, body_ang_vel).reshape(n, -1)
    return torch.cat([root_pos[:, 2:3], local_pos, local_rot, local_vel, local_ang_vel], dim=-1)


def imitation_obs_v6(body_pos, body_rot, body_vel, body_ang_vel, ref_pos, ref_rot, ref_vel, ref_ang_vel):
    """``compute_imitation_observations_v6`` with a single future step."""
    root_pos, root_rot = body_pos[:, 0], body_rot[:, 0]
    n, j = body_pos.shape[:2]
    h_inv = heading_quat_inv(root_rot)[:, None].expand(n, j, 4)
    h = heading_quat(root_rot)[:, None].expand(n, j, 4)

    def flat(x):
        return x.reshape(-1, x.shape[-1])

    diff_pos = rotate(h_inv, ref_pos - body_pos)
    diff_rot = quat_mul(flat(ref_rot), inv_quat(flat(body_rot)))
    diff_rot = quat_mul(quat_mul(flat(h_inv), diff_rot), flat(h))
    diff_vel = rotate(h_inv, ref_vel - body_vel)
    diff_ang_vel = rotate(h_inv, ref_ang_vel - body_ang_vel)
    local_ref_pos = rotate(h_inv, ref_pos - root_pos[:, None])
    local_ref_rot = quat_to_tan_norm(quat_mul(flat(h_inv), flat(ref_rot)))
    return torch.cat(
        [
            diff_pos.reshape(n, -1),
            quat_to_tan_norm(diff_rot).reshape(n, -1),
            diff_vel.reshape(n, -1),
            diff_ang_vel.reshape(n, -1),
            local_ref_pos.reshape(n, -1),
            local_ref_rot.reshape(n, -1),
        ],
        dim=-1,
    )


#: PHC's default ``reward_specs``.
REWARD_SPECS = {
    "k_pos": 100.0,
    "k_rot": 10.0,
    "k_vel": 0.1,
    "k_ang_vel": 0.1,
    "w_pos": 0.5,
    "w_rot": 0.3,
    "w_vel": 0.1,
    "w_ang_vel": 0.1,
}


def imitation_reward(body_pos, body_rot, body_vel, body_ang_vel, ref_pos, ref_rot, ref_vel, ref_ang_vel):
    """``compute_imitation_reward`` over all bodies (``full_body_reward``)."""
    s = REWARD_SPECS
    r_pos = torch.exp(-s["k_pos"] * ((ref_pos - body_pos) ** 2).mean(-1).mean(-1))
    # angle of the relative rotation, in [0, pi] (PHC quat_to_angle_axis)
    angle = quat_error_magnitude(ref_rot.reshape(-1, 4), body_rot.reshape(-1, 4)).reshape(body_rot.shape[:2])
    r_rot = torch.exp(-s["k_rot"] * (angle**2).mean(-1))
    r_vel = torch.exp(-s["k_vel"] * ((ref_vel - body_vel) ** 2).mean(-1).mean(-1))
    r_ang_vel = torch.exp(-s["k_ang_vel"] * ((ref_ang_vel - body_ang_vel) ** 2).mean(-1).mean(-1))
    return s["w_pos"] * r_pos + s["w_rot"] * r_rot + s["w_vel"] * r_vel + s["w_ang_vel"] * r_ang_vel
