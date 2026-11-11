"""Build helpers for the robot tests: one scene per robot module, and link-frame reads off it."""

from __future__ import annotations

import numpy as np
import torch

from eden.options import EnvOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


def build_scene(*, sim_dt: float = 0.005, sim_substeps: int = 4, **entities):
    """A one-env scene holding ``entities`` (composites included), at the robots' usual 200 Hz / 4 substeps."""
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=sim_dt, sim_substeps=sim_substeps),
            scene_options=SceneOptions(**entities),
        )
    )


def links_pos(entity, names: list[str]) -> np.ndarray:
    """World positions ``(len(names), 3)`` of ``names``, in the order given."""
    _, idx = entity.find_named_links_idx_local(names, preserve_order=True)
    return entity.get_links_pos(ls_idx_local=idx)[0].cpu().numpy()


def link_pose(entity, name: str) -> tuple[np.ndarray, np.ndarray]:
    """World ``(pos, wxyz quat)`` of one link."""
    _, idx = entity.find_named_links_idx_local([name], preserve_order=True)
    return entity.get_links_pos(ls_idx_local=idx)[0, 0].cpu().numpy(), entity.get_links_quat(ls_idx_local=idx)[
        0, 0
    ].cpu().numpy()


def set_dofs(entity, values: dict[str, float], base: torch.Tensor | None = None) -> None:
    """Teleport the named DOFs (every other DOF to ``base``, default zero), with zero velocity."""
    q = torch.zeros((1, entity.num_dofs), device=entity.device) if base is None else base.clone().reshape(1, -1)
    order = {name: i for i, name in enumerate(entity.dofs_name)}
    for name, value in values.items():
        q[0, order[name]] = value
    entity.set_dofs_pos(q)


def dofs_in_declared_order(entity, declared: list[str]) -> list[str]:
    names, idx = entity.find_named_dofs_idx_local(declared, preserve_order=True)
    assert len(idx) == len(declared)
    return list(names)
