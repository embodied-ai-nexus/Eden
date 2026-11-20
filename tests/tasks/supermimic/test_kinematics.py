"""``ForwardKinematics`` against the simulator's own link states."""

import pytest
import torch

from eden.entities.rigid import StagedStateWrite
from eden.tasks.supermimic.config import TRACKED_LINKS
from eden.tasks.supermimic.kinematics import ForwardKinematics
from eden.utils.assets import get_asset_path
from eden.utils.geom import inv_transform_by_quat, normalize

pytestmark = [pytest.mark.requires_genesis]


def test_links_match_genesis_at_random_states(env):
    robot = env.entities["robot"]
    fk = ForwardKinematics(get_asset_path(env.scene_options.robot.file), robot.dofs_name, TRACKED_LINKS)
    n, j = env.num_envs, len(robot.dofs_name)
    gen = torch.Generator().manual_seed(0)
    root_pos = torch.rand(n, 3, generator=gen) + torch.tensor([0.0, 0.0, 1.0])
    root_quat = normalize(torch.randn(n, 4, generator=gen))
    q = 0.5 * torch.randn(n, j, generator=gen)
    lin, ang, qd = (torch.randn(n, k, generator=gen) for k in (3, 3, j))
    write = StagedStateWrite()
    base = robot.find_free_base_dofs_idx_local()
    write.set_dofs_vel(
        robot, torch.cat([lin, inv_transform_by_quat(ang, root_quat), qd], 1), [*base, *robot.dofs_idx_local.tolist()]
    )
    write.set_qpos(robot, torch.cat([root_pos, root_quat, q], 1), robot.qs_idx_local.tolist())
    write.apply(torch.arange(n))
    try:
        _, links = robot.find_named_links_idx_local(TRACKED_LINKS)
        pos, quat, lin_vel, ang_vel = fk(root_pos, root_quat, q, lin, ang, qd)
        torch.testing.assert_close(pos, robot.get_links_pos(ls_idx_local=links), atol=1e-5, rtol=0)
        # q and -q are one rotation.
        torch.testing.assert_close(
            (quat * robot.get_links_quat(ls_idx_local=links)).sum(-1).abs(),
            torch.ones(n, len(links)),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(lin_vel, robot.get_links_vel(ls_idx_local=links), atol=1e-4, rtol=0)
        torch.testing.assert_close(ang_vel, robot.get_links_ang(ls_idx_local=links), atol=1e-4, rtol=0)
    finally:
        env.reset()


def test_a_link_missing_from_the_model_is_named():
    from eden.options.robots.unitree_g1 import G1Dofs29

    robot = G1Dofs29()
    with pytest.raises(ValueError, match=r"no link\(s\) \['head'\]"):
        ForwardKinematics(get_asset_path(robot.file), robot.dofs_name, ["pelvis", "head"])
