"""Reorient's event terms."""

from __future__ import annotations

import pytest
import torch

from eden.options import EntityOptions, EnvOptions, EventManagerOptions, SceneOptions
from eden.tasks.reorient.custom_terms.events import ResetDofsByPatternOffset
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# Fixed base: a three-DOF ball joint, then a hinge.
_BALL_MJCF = """<mujoco model="ball"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="0 0 0.4"><geom type="box" size="0.04 0.04 0.02" mass="1"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" mass="0.1"/>
      <body name="spindle" pos="0.08 0 0"><joint name="spin_joint" axis="0 0 1" range="-0.4 0.4"/>
        <geom type="cylinder" size="0.02 0.01" mass="0.1"/></body></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    path = tmp_path_factory.mktemp("reorient_events") / "ball.xml"
    path.write_text(_BALL_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                ball=EntityOptions(
                    file=str(path),
                    is_fixed_base=True,
                    is_articulated=True,
                    dofs_name=["ball_joint", "spin_joint"],
                ),
            ),
            event_options=EventManagerOptions(
                reset_ball=ResetDofsByPatternOffset.configure(
                    entity_name="ball", dofs_pos_range={"ball_*": (0.0, 0.0), "spin_*": (0.1, 0.1)}
                ),
            ),
        )
    )


def test_a_ball_joint_resets_through_one_column_per_dof(env):
    term = env.event_manager.get_term("reset_ball")
    ball = env.entities["ball"]
    assert term._dofs.columns.tolist() == [ball.dofs_idx_map[int(i)] for i in term._dofs.dofs_idx_local]
    assert len(term._dofs) == 4

    term.compute()

    spin = ball.find_named_dofs_idx_local(["spin_joint"])[1]
    torch.testing.assert_close(
        ball.get_dofs_pos(dofs_idx_local=spin), torch.full((2, 1), 0.1, device=ball.default_dofs_pos.device)
    )
