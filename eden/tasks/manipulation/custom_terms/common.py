"""Shared geometry / frame / state helpers for the manipulation MDP terms.

The end-effector used by every term is the *grasp center* of the gripper: the
midpoint of the two finger-pad **collision geoms** (the pad *link* origins sit
~13 cm proximal of the actual silicone pad surfaces, so link positions point the
grasp 13 cm off the real contact). Object pose and the goal command are expressed
in the robot base frame to match mjlab's ``*_in_robot_base_frame``.

These helpers are imported by the per-manager term modules in this package
(``observations``/``rewards``/``terminations``/``curricula``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

from eden.utils.misc import as_genesis_index

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


# Cache of pad collision-geom global indices per (env, gripper, links).
_PAD_GEOM_IDX: dict = {}


def _grasp_point_w(
    env: "EnvBase",
    gripper_name: str,
    grasp_links: list[str],
    approach_offset: float = 0.0,
    base_link: str = "base",
) -> torch.Tensor:
    """World-frame grasp center: midpoint of the finger-pad **collision geoms**. (N, 3).

    Not the pad link origins, which sit ~13 cm proximal of the pad surfaces (see the module
    docstring). A nonzero ``approach_offset`` (m) nudges the point further along the approach axis,
    from ``base_link`` through the grasp center.
    """
    gripper: Entity = env.entities[gripper_name]
    key = (id(env), gripper_name, tuple(grasp_links))
    geom_idx = _PAD_GEOM_IDX.get(key)
    if geom_idx is None:
        keep = set(grasp_links)
        # A device tensor, built once: indexing with a Python list uploads it on every call, which is a
        # blocking host->device copy.
        geom_idx = torch.tensor([g.idx for g in gripper.geoms if g.link.name in keep], device=env.device)
        _PAD_GEOM_IDX[key] = geom_idx
    # dyn_state.geoms.pos is (n_geoms, n_envs, 3) in world frame.
    gpos = gripper.solver.dyn_state.geoms.pos.to_torch(device=env.device)
    pt = gpos[geom_idx].mean(dim=0)  # (n_envs, 3)
    if approach_offset:
        base_key = (id(env), gripper_name, base_link)
        base_idx = _PAD_GEOM_IDX.get(base_key)
        if base_idx is None:
            # A `range`, not the one-element link tensor Genesis would read back with `.item()` per call.
            _, bidx = gripper.find_named_links_idx_local([base_link], preserve_order=True)
            base_idx = _PAD_GEOM_IDX[base_key] = as_genesis_index(bidx)
        base = gripper.get_links_pos(ls_idx_local=base_idx).squeeze(1)
        direction = pt - base
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        pt = pt + approach_offset * direction
    return pt


def _object_pos_w(env: "EnvBase", object_name: str) -> torch.Tensor:
    return env.entities[object_name].get_pos()


def _to_base_frame(env: "EnvBase", robot_name: str, pos_w: torch.Tensor) -> torch.Tensor:
    """World positions ``(N, 3)`` or ``(N, K, 3)`` (K points per env) in the robot's base frame."""
    robot: Entity = env.entities[robot_name]
    # View the per-env base pose as (N, 1, ..., 1, D) so it broadcasts over the points of each env.
    points_dims = (1,) * (pos_w.dim() - 2)
    base_pos = robot.get_pos().view(-1, *points_dims, 3)
    base_quat = robot.get_quat().view(-1, *points_dims, 4).expand(*pos_w.shape[:-1], 4)
    return inv_transform_by_quat(pos_w - base_pos, base_quat)


def _base_to_world(env: "EnvBase", robot_name: str, pos_b: torch.Tensor) -> torch.Tensor:
    robot: Entity = env.entities[robot_name]
    return robot.get_pos() + transform_by_quat(pos_b, robot.get_quat())


def _gripper_closure(env: "EnvBase", gripper_name: str, gripper_dofs: list[str], close_target: float) -> torch.Tensor:
    """Return normalized gripper closure in [0, 1] (0 = open, 1 = at close target), shape (N,)."""
    gripper: Entity = env.entities[gripper_name]
    _, idx = gripper.find_named_dofs_idx_local(gripper_dofs, preserve_order=True)
    pos = gripper.get_dofs_pos(dofs_idx_local=idx)  # (N, k)
    return torch.clamp(pos.mean(dim=-1) / close_target, min=0.0, max=1.0)
