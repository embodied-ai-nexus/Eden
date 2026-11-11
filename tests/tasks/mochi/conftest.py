"""Builders shared by the Mochi task tests: a task env at two envs, and moving a YAM's tool centre."""

from __future__ import annotations

import math

import genesis as gs
import torch
from genesis.utils.geom import transform_by_quat

from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from eden.tasks.mochi.common import YAM_ARM_DOFS, YAM_HAND_DOWN, YAM_TCP_OFFSET, link_point
from eden.utils.geom import quat_mul


def build_task(name: str, num_envs: int = 2, *, recording: bool = False) -> RLEnvBase:
    """Build and reset the registered Mochi task ``name`` with ``num_envs`` envs.

    ``recording`` adds the ``rec`` camera as ``eden inference --rec`` does.
    """
    config = TASK_REGISTRY.build(name)
    config.env_options.num_envs = num_envs
    if recording:
        config.add_recording_camera()
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    return env


def tool_centre(env: RLEnvBase, robot: str) -> torch.Tensor:
    """World position of a YAM's tool centre, ``(num_envs, 3)``."""
    offset = torch.tensor(YAM_TCP_OFFSET, device=gs.device)
    return link_point(env.entities[robot].get_link("link_6"), offset)


def place_tcp(env: RLEnvBase, robot: str, tcp: torch.Tensor, yaw: float = 0.0) -> None:
    """Put a YAM's tool centre at ``tcp`` (``(num_envs, 3)``), pointing down with its jaws at ``yaw``, and hold it.

    ``yaw`` turns the jaws' closing axis from world x about the vertical. The arm is solved by IK and written
    outright, so the scene is not stepped to get there.
    """
    yam = env.entities[robot]
    link = yam.get_link("link_6")
    turn = torch.tensor([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], device=tcp.device)
    down = torch.tensor(YAM_HAND_DOWN, device=tcp.device)
    # Pointing down, then turned about world z.
    quat = quat_mul(turn[None], down[None]).expand(len(tcp), -1).contiguous()
    offset = torch.tensor(YAM_TCP_OFFSET, device=tcp.device).expand(len(tcp), -1)
    link_pos = tcp - transform_by_quat(offset, quat)
    current = yam.entity.get_qpos()
    solved = yam.entity.inverse_kinematics(link=link, pos=link_pos, quat=quat, init_qpos=current, max_solver_iters=200)
    # Only the arm moves: the jaws keep their opening.
    _, arm = yam.find_named_dofs_idx_local(YAM_ARM_DOFS)
    qpos = current.clone()
    qpos[:, arm] = solved[:, arm]
    yam.set_qpos(qpos)
    yam.control_dofs_pos(qpos[:, arm], dofs_idx_local=arm)


def action(env: RLEnvBase, *, close: dict[str, bool] | None = None, move: dict[str, torch.Tensor] | None = None):
    """An action that holds every hand still, closes the named grippers and moves the named hands.

    ``close`` maps a gripper action term to whether it closes; ``move`` maps a hand (differential-IK) action term
    to a raw ``(num_envs, 3)`` position action.
    """
    manager = env.action_manager
    out = torch.zeros(env.num_envs, manager.total_action_dim, device=gs.device)
    start = 0
    for name in manager.active_terms:
        dim = manager.get_term(name).action_dim
        if close is not None and name in close:
            out[:, start] = 1.0 if close[name] else -1.0
        elif dim == 1:
            out[:, start] = -1.0
        if move is not None and name in move:
            out[:, start : start + 3] = move[name]
        start += dim
    return out
