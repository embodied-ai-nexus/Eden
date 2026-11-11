"""OmniXtreme observation terms (previous base action, anchor orientation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.managers import OBSERVATION_TERM_REGISTRY
from genesis.utils.geom import inv_quat, quat_to_R, transform_quat_by_quat
from eden.extensions.policies.omnixtreme.onnx_wrapper import run_fk_session

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

# Waist DOF indices in URDF order: waist_yaw, waist_roll, waist_pitch
WAIST_DOF_INDICES = [12, 13, 14]

# Approximate standing pelvis height (meters), used as FK base position input
FK_BASE_HEIGHT = 0.79


@OBSERVATION_TERM_REGISTRY.register()
def omnixtreme_prev_base_action(
    env: EnvBase,
    *,
    command_name: str = "omnixtreme_motion",
    envs_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the previous base policy action (num_envs, 29), updated by policy step()."""
    cmd_term = env.command_manager.get_term(command_name)
    if envs_idx is not None:
        return cmd_term.prev_base_action[envs_idx]
    return cmd_term.prev_base_action


@OBSERVATION_TERM_REGISTRY.register()
def omnixtreme_anchor_ori_6d(
    env: EnvBase,
    *,
    entity_name: str = "robot",
    command_name: str = "omnixtreme_motion",
    envs_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute 6D anchor orientation between current robot torso and reference motion body.

    The current torso orientation comes from the released FK ONNX graph (waist angles + base
    quaternion at a fixed pelvis height), matching the original deployment code exactly. Reading
    ``entity.get_links_quat(ls_idx_local=[torso_link_idx])`` instead would drop the FK dependency.
    """
    cmd_term = env.command_manager.get_term(command_name)
    entity = env.entities[entity_name]

    B = env.num_envs if envs_idx is None else envs_idx.shape[0]

    # The FK session is set by OmniXtremePolicy.__init__ after env.build(); until then (observation
    # dim inference in ObservationManager._prepare_terms) it is None.
    if cmd_term.fk_session is None:
        return torch.zeros(B, 6, device=env.device)

    dof_pos = entity.get_dofs_pos(envs_idx=envs_idx)
    waist_angles = dof_pos[:, WAIST_DOF_INDICES].detach().cpu().numpy().astype(np.float32)
    B = waist_angles.shape[0]

    base_pos = np.zeros((B, 3), dtype=np.float32)
    base_pos[:, 2] = FK_BASE_HEIGHT
    base_quat = entity.get_quat(envs_idx=envs_idx).detach().cpu().numpy().astype(np.float32)

    _, rot_np = run_fk_session(
        cmd_term,
        {"joint_angles": waist_angles, "base_pos": base_pos, "base_quat": base_quat},
    )

    # rot_np is wxyz from FK model
    cur_anchor_quat = torch.from_numpy(rot_np).to(env.device)

    # Compute relative rotation: q_rel = inv(q_current) * q_reference
    ref_pos, ref_rot = cmd_term.get_anchor_ref_data()  # (1, 3), (1, 4) wxyz
    q_rel = transform_quat_by_quat(ref_rot.expand(B, -1), inv_quat(cur_anchor_quat))

    # Convert to rotation matrix, take first 2 columns -> 6D representation
    rot_matrix = quat_to_R(q_rel)
    return rot_matrix[..., :2].reshape(-1, 6)
