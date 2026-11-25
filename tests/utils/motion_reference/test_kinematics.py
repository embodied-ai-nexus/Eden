"""``ForwardKinematics`` against MuJoCo's own kinematics of the same model."""

import mujoco
import numpy as np
import pytest
import torch

from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.assets import get_asset_path
from eden.utils.geom import quat_mul, transform_by_quat
from eden.utils.motion_reference.kinematics import ForwardKinematics, _mul, _rotate

LINKS = ["pelvis", "left_ankle_roll_link", "torso_link", "right_wrist_yaw_link"]


@pytest.fixture(scope="module")
def robot():
    return G1Dofs29()


#: A tree the G1 does not cover: tilted body frames, hinge axes off the frame axes, anchors off the body origin, a
#: body with two hinges, a fixed body and a non-zero reference angle.
CHAIN_MJCF = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom size="0.1"/>
      <body name="upper" pos="0.1 0.2 -0.3" quat="0.9 0.1 0.3 0.3">
        <joint name="j0" type="hinge" axis="0.3 0.5 0.8" pos="0.05 0 0.02"/>
        <joint name="j1" type="hinge" axis="1 0 0" pos="0 0.03 0" ref="0.4"/>
        <geom size="0.05"/>
        <body name="bracket" pos="0 0.1 0.1" quat="0.7 0.7 0 0.1">
          <geom size="0.02"/>
          <body name="lower" pos="0.2 0 0">
            <joint name="j2" type="hinge" axis="0 1 1" pos="-0.04 0.01 0"/>
            <geom size="0.05"/>
          </body>
        </body>
      </body>
      <body name="side" pos="-0.2 0 0" quat="0.5 0.5 0.5 0.5">
        <joint name="j3" type="hinge" axis="0 0 1"/>
        <geom size="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def chain_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("chain") / "chain.xml"
    path.write_text(CHAIN_MJCF)
    return str(path)


def _assert_matches_mujoco(path: str, links: list[str]) -> None:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    joints = [model.joint(j).name for j in range(1, model.njnt)]
    # The module holds its constants in float32, which bounds the agreement.
    fk = ForwardKinematics(path, joints, links).double()
    rng = np.random.default_rng(0)
    for _ in range(4):
        data.qpos[:3] = rng.uniform(-1, 1, 3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[7:] = model.qpos0[7:] + 0.5 * rng.normal(size=model.nq - 7)
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        root_rot = data.xmat[1].reshape(3, 3)
        # MuJoCo's free joint holds the linear velocity in the world frame and the angular one in the body frame.
        args = (data.qpos[:3], data.qpos[3:7], data.qpos[7:], data.qvel[:3], root_rot @ data.qvel[3:6], data.qvel[6:])
        pos, quat, lin, ang = fk(*(torch.as_tensor(a).unsqueeze(0) for a in args))
        for row, name in enumerate(links):
            body = model.body(name).id
            np.testing.assert_allclose(pos[0, row], data.xpos[body], atol=1e-6)
            assert abs(float(quat[0, row] @ torch.as_tensor(data.xquat[body]))) == pytest.approx(1.0, abs=1e-6)
            twist = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, twist, 0)
            np.testing.assert_allclose(ang[0, row], twist[:3], atol=1e-6)
            np.testing.assert_allclose(lin[0, row], twist[3:], atol=1e-6)


def test_links_match_mujoco_at_random_states(robot):
    _assert_matches_mujoco(get_asset_path(robot.file), LINKS)


def test_links_match_mujoco_on_a_tree_with_tilted_frames_and_offset_hinges(chain_path):
    _assert_matches_mujoco(chain_path, ["base", "upper", "bracket", "lower", "side"])


def test_a_joint_that_is_not_a_hinge_is_named(tmp_path):
    path = tmp_path / "slide.xml"
    path.write_text(CHAIN_MJCF.replace('name="j3" type="hinge"', 'name="j3" type="slide"'))
    with pytest.raises(ValueError, match="body side has a joint that is not a hinge"):
        ForwardKinematics(str(path), ["j0", "j1", "j2", "j3"], ["side"])


def test_a_link_missing_from_the_model_is_named(robot):
    with pytest.raises(ValueError, match=r"no link\(s\) \['head'\]"):
        ForwardKinematics(get_asset_path(robot.file), robot.dofs_name, ["pelvis", "head"])


def test_the_component_products_are_the_geom_helpers():
    gen = torch.Generator().manual_seed(0)
    a, b = (torch.nn.functional.normalize(torch.randn(8, 4, generator=gen), dim=-1) for _ in range(2))
    v = torch.randn(8, 3, generator=gen)
    torch.testing.assert_close(torch.stack(_mul(a.unbind(-1), b.unbind(-1)), dim=-1), quat_mul(a, b))
    torch.testing.assert_close(torch.stack(_rotate(a.unbind(-1), v.unbind(-1)), dim=-1), transform_by_quat(v, a))
