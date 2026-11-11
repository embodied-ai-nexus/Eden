"""Custom observation term that replicates HumEnv's ``compute_humanoid_self_obs_v2``.

Produces a heading-invariant, egocentric proprioceptive observation from
the Genesis SMPL humanoid link state, matching the format expected by
MetaMotivo policies (FBcpr).

Observation layout (concatenated)::

    root_h           (1,)       pelvis height
    local_body_pos   ((N_B-1)*3,) body positions relative to root, heading-rotated (root excluded)
    local_body_rot   (N_B*6,)   body rotations in heading frame as tan/norm 6D repr (all bodies)
    local_body_vel   (N_B*3,)   linear velocities in heading frame (all bodies)
    local_body_ang   (N_B*3,)   angular velocities in heading frame (all bodies)

where N_B = number of rigid bodies (24 for SMPL humanoid).

.. note::
    This term requires the Genesis SMPL humanoid to be loaded from the
    **original HumEnv Y-up MuJoCo XML** with ``up=(0,1,0)`` so that
    body-local frames match HumEnv exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from genesis.utils.geom import inv_quat, transform_by_quat

from eden.utils.geom import quat_mul, quat_to_tan_norm, yaw_quat
from eden.utils.torch import device_constant

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


# ---------------------------------------------------------------------------
# HumEnv-specific quaternion helpers
# ---------------------------------------------------------------------------


def _calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse heading (yaw-only) quaternion from a root orientation (wxyz).

    HumEnv extracts the heading by projecting the local x-axis onto the ground
    plane, ``atan2((R·x̂)_y, (R·x̂)_x)`` — which is exactly the yaw
    :func:`eden.utils.geom.yaw_quat` builds its pure-z quaternion from, so this
    is ``inv(yaw_quat(q))`` (pinned by
    ``tests/utils/test_geom.py::test_inv_yaw_quat_matches_humenv_heading_oracle``).
    """
    return inv_quat(yaw_quat(q))


def _remove_base_rot_smpl(quat: torch.Tensor) -> torch.Tensor:
    """Remove the SMPL base rotation from root quaternion.

    SMPL models have a 90-deg X rotation built into the skeleton.
    This must be removed before extracting the heading angle.
    """
    # A device constant: building the tensor from a host list each step is a blocking copy.
    base_rot_inv = device_constant((0.5, -0.5, -0.5, -0.5), quat.device, dtype=quat.dtype).expand_as(quat)
    return quat_mul(quat, base_rot_inv)


# ---------------------------------------------------------------------------
# Body reindexing (Genesis link order -> MuJoCo body order)
# ---------------------------------------------------------------------------

# MuJoCo body order (depth-first traversal of the SMPL kinematic tree).
# fmt: off
_MUJOCO_BODY_ORDER = [
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]
# fmt: on
_reindex_cache: dict[int, torch.Tensor] = {}


def _get_body_reindex(entity, device: torch.device) -> torch.Tensor:
    """Build a mapping from MuJoCo body order to Genesis link order.

    Returns a LongTensor of shape ``(24,)`` such that
    ``genesis_data[:, reindex]`` gives data in MuJoCo body order.
    """
    key = id(entity)
    if key in _reindex_cache:
        return _reindex_cache[key]

    genesis_name_to_idx = {link.name: link.idx_local for link in entity.links}

    reindex = []
    for name in _MUJOCO_BODY_ORDER:
        if name not in genesis_name_to_idx:
            raise ValueError(f"Link '{name}' not found in Genesis entity. Available: {list(genesis_name_to_idx)}")
        reindex.append(genesis_name_to_idx[name])

    t = torch.tensor(reindex, dtype=torch.long, device=device)
    _reindex_cache[key] = t
    return t


# ---------------------------------------------------------------------------
# Observation term
# ---------------------------------------------------------------------------


@OBSERVATION_TERM_REGISTRY.register()
def humenv_proprio(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute HumEnv-compatible proprioceptive observation from Genesis state.

    This replicates ``humenv.env.compute_humanoid_self_obs_v2`` using Genesis
    link-level accessors, producing a heading-invariant egocentric observation
    that MetaMotivo policies expect.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        The SMPL humanoid entity name in the scene.
    envs_idx : slice | torch.Tensor | None, optional
        Environment indices to compute for; ``None`` computes every env.

    Returns
    -------
    torch.Tensor
        Flattened proprioceptive observation, shape ``(num_envs, obs_dim)``.
    """
    entity = env.entities[entity_name]

    # Retrieve all link states in world frame.
    body_pos = entity.get_links_pos(envs_idx=envs_idx)  # (B, L, 3)
    body_quat = entity.get_links_quat(envs_idx=envs_idx)  # (B, L, 4) wxyz
    body_vel = entity.get_links_vel(envs_idx=envs_idx)  # (B, L, 3)
    body_ang = entity.get_links_ang(envs_idx=envs_idx)  # (B, L, 3)

    # Reindex from Genesis link order to MuJoCo body order (depth-first).
    reindex = _get_body_reindex(entity, body_pos.device)
    body_pos = body_pos[:, reindex]
    body_quat = body_quat[:, reindex]
    body_vel = body_vel[:, reindex]
    body_ang = body_ang[:, reindex]

    num_envs, num_links, _ = body_pos.shape

    # --- Root state ---
    root_pos = body_pos[:, 0, :]  # (B, 3)
    root_quat = body_quat[:, 0, :]  # (B, 4) wxyz

    # Remove SMPL base rotation (90-deg X) for correct heading extraction.
    root_quat_corrected = _remove_base_rot_smpl(root_quat)

    # Heading-inverse quaternion
    heading_rot_inv = _calc_heading_quat_inv(root_quat_corrected)  # (B, 4)

    # --- Root height (Z-up after Genesis conversion) ---
    root_h = root_pos[:, 2:3]  # (B, 1)

    # --- Local body positions (heading-rotated, root-relative) ---
    local_body_pos = body_pos - root_pos.unsqueeze(1)
    heading_inv_expanded = heading_rot_inv.unsqueeze(1).expand_as(body_quat)
    flat_heading_inv = heading_inv_expanded.reshape(-1, 4)
    flat_local_pos = local_body_pos.reshape(-1, 3)
    flat_local_pos = transform_by_quat(flat_local_pos, flat_heading_inv)
    local_body_pos = flat_local_pos.reshape(num_envs, num_links * 3)
    local_body_pos = local_body_pos[:, 3:]  # exclude root (first 3 components)

    # --- Local body rotations (heading-rotated, tan/norm 6D) ---
    flat_body_quat = body_quat.reshape(-1, 4)
    flat_local_rot = quat_mul(flat_heading_inv, flat_body_quat)
    flat_local_rot_6d = quat_to_tan_norm(flat_local_rot)  # (B*L, 6)
    local_body_rot = flat_local_rot_6d.reshape(num_envs, num_links * 6)

    # --- Local body velocities (heading-rotated) ---
    flat_body_vel = body_vel.reshape(-1, 3)
    flat_local_vel = transform_by_quat(flat_body_vel, flat_heading_inv)
    local_body_vel = flat_local_vel.reshape(num_envs, num_links * 3)

    # --- Local body angular velocities (heading-rotated) ---
    flat_body_ang = body_ang.reshape(-1, 3)
    flat_local_ang = transform_by_quat(flat_body_ang, flat_heading_inv)
    local_body_ang = flat_local_ang.reshape(num_envs, num_links * 3)

    # --- Concatenate all ---
    return torch.cat(
        [
            root_h,
            local_body_pos,
            local_body_rot,
            local_body_vel,
            local_body_ang,
        ],
        dim=-1,
    )
