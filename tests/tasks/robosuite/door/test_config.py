"""Door: robosuite's per-reset placement of a fixed-base door, and the hinge / handle predicates its reward reads."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.utils.geom import inv_transform_by_quat, yaw_from_quat
from tests.support.envs import build_env

#: robosuite's ``UniformRandomSampler`` for the door, about ``table_offset`` (-0.2, -0.35).
_TABLE_XY = (-0.2, -0.35)
_X_RANGE, _Y_RANGE = (0.07, 0.09), (-0.01, 0.01)
_YAW_RANGE = (-math.pi / 2 - 0.25, -math.pi / 2)


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_door")
    config.env_options.num_envs = 4
    return build_env(config, reset=True)


def _set_dof(env, name: str, values: list[float]) -> None:
    door = env.entities["door"]
    idx = door.find_named_dofs([name]).dofs_idx_local
    door.set_dofs_pos(torch.tensor(values, device=env.device)[:, None], dofs_idx_local=idx)


@pytest.mark.requires_genesis
def test_a_reset_places_the_door_inside_robosuites_sampler_range(env):
    env.reset(seed=0)
    door = env.entities["door"]
    pos, quat = door.get_pos(relative=True), door.get_quat(relative=True)
    x, y, yaw = pos[:, 0] - _TABLE_XY[0], pos[:, 1] - _TABLE_XY[1], yaw_from_quat(quat)
    assert ((x >= _X_RANGE[0] - 1e-5) & (x <= _X_RANGE[1] + 1e-5)).all(), x
    assert ((y >= _Y_RANGE[0] - 1e-5) & (y <= _Y_RANGE[1] + 1e-5)).all(), y
    assert ((yaw >= _YAW_RANGE[0] - 1e-5) & (yaw <= _YAW_RANGE[1] + 1e-5)).all(), yaw
    assert len({tuple(row) for row in torch.cat([pos[:, :2], yaw[:, None]], dim=-1).tolist()}) == env.num_envs


@pytest.mark.requires_genesis
def test_the_panel_and_handle_move_with_the_re_posed_base_and_stay_there(env):
    """Every link sits at the same place *in the door's frame* in every env, before and after stepping.

    A base write that moved only the root link (or that physics undid on the next step) would leave the
    hinge, the panel and the latch where the door was built, and the reach reward aiming at nothing.
    Across envs, not against the first read: the sprung latch settles a few millimetres once stepped.
    """
    env.reset(seed=1)
    door = env.entities["door"]

    def links_in_door_frame():
        base_pos, base_quat = door.get_pos(relative=True), door.get_quat(relative=True)
        links = door.get_links_pos(relative=True)  # (N, L, 3)
        rel = links - base_pos[:, None]
        return inv_transform_by_quat(rel, base_quat[:, None].expand(-1, rel.shape[1], -1))

    placed_pos = door.get_pos(relative=True).clone()
    local = links_in_door_frame()
    torch.testing.assert_close(local, local[:1].expand_as(local), atol=1e-4, rtol=0)
    for _ in range(5):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    torch.testing.assert_close(door.get_pos(relative=True), placed_pos, atol=1e-5, rtol=0)
    settled = links_in_door_frame()
    torch.testing.assert_close(settled, settled[:1].expand_as(settled), atol=1e-3, rtol=0)


@pytest.mark.requires_genesis
def test_the_door_counts_open_past_robosuites_0_3_rad(env):
    env.reset(seed=2)
    _set_dof(env, "hinge", [0.0, 0.29, 0.31, 1.0])
    opened = env.metric_manager.get_term("door_open").compute()
    assert opened.tolist() == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.requires_genesis
def test_handle_rotation_pays_robosuites_clipped_fraction_of_a_quarter_turn(env):
    """``clip(0.25 |handle_qpos| / (pi / 2), -0.25, 0.25)`` — the term returns the fraction, the weight is 0.25."""
    env.reset(seed=3)
    _set_dof(env, "latch_joint", [0.0, math.pi / 4, -math.pi / 4, math.pi])
    fraction = env.reward_manager.get_term("handle_rotation").compute()
    torch.testing.assert_close(fraction, torch.tensor([0.0, 0.5, 0.5, 1.0], device=env.device), atol=1e-4, rtol=0)
    assert env.reward_manager.get_term("handle_rotation").weight == 0.25
