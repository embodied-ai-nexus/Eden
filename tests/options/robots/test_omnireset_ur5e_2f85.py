"""``OmniResetUR5e2f85`` built: UWLab's calibrated kinematics, implicit-actuator dynamics and a closing linkage.

The oracle for the arm is UWLab's own calibrated analytic model (``uwlab_assets.robots.ur5e_robotiq_gripper.
kinematics``, constants from the robot's ``metadata.yaml``), independent of the USD joint frames the
converter reads. The gripper's oracle is the sign pattern Isaac Lab recorded in OmniReset's reset states.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eden.options.robots import OmniResetUR5e2f85
from eden.options.robots.omnireset_ur5e_2f85 import ARM_DOFS, GRIPPER_DRIVER
from tests.options.robots.conftest import build_scene, link_pose

pytestmark = pytest.mark.requires_genesis

# UWLab metadata.yaml ``calibrated_joints``: fixed transform (xyz, rpy) before each arm joint's z-rotation.
_CALIBRATED_XYZ = np.array(
    [
        [0.0, 0.0, 0.1625269213099744],
        [0.0001655127603476787, 0.0, 0.0],
        [-0.42543123286729523, 0.0, 0.0],
        [-0.3926255442838572, -0.00018118202218157404, 0.13378567426143276],
        [7.719320843298656e-05, -0.0997722260056821, 6.2471135291889e-05],
        [9.607876543233622e-05, 0.09944214059801317, 0.00012685702594143576],
    ]
)
_CALIBRATED_RPY = np.array(
    [
        [0.0, 0.0, -2.9298064682709875e-08],
        [1.569956524571092, 0.0, -6.769377576441778e-10],
        [3.1401294729209073, 3.1401513871885234, 3.1415926282859967],
        [0.001354269897852176, -0.0010953222406931517, 2.9544587825348556e-06],
        [1.570170189345338, 0.0, -3.311599503007332e-08],
        [1.5720720129010473, 3.141592653589793, 3.141592595496277],
    ]
)
# Isaac Lab's joint order for the gripper (PhysX breadth-first) and the sign each reads relative to
# finger_joint, as recorded in OmniReset's reset states.
_LINKAGE_SIGNS = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": -1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_finger_knuckle_joint": -1.0,
    "left_inner_finger_knuckle_joint": -1.0,
}


def _calibrated_wrist_pose(q: np.ndarray) -> np.ndarray:
    """wrist_3_link in base_link (REP-103): UWLab's chain from base_link_inertia, then 180 deg about z."""
    transform = np.diag([-1.0, -1.0, 1.0, 1.0])
    for xyz, rpy, angle in zip(_CALIBRATED_XYZ, _CALIBRATED_RPY, q):
        fixed = np.eye(4)
        fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()  # UWLab's rpy_to_matrix: Rz @ Ry @ Rx
        fixed[:3, 3] = xyz
        joint = np.eye(4)
        joint[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
        transform = transform @ fixed @ joint
    return transform


@pytest.fixture(scope="module")
def env():
    return build_scene(sim_dt=1 / 120, sim_substeps=1, robot=OmniResetUR5e2f85())


def test_implicit_actuator_dynamics_reach_the_solver(env):
    robot = env.entities["robot"]
    _, every_dof = robot.find_named_dofs_idx_local(list(ARM_DOFS) + list(_LINKAGE_SIGNS))
    assert robot.num_dofs == len(ARM_DOFS) + 1  # the passive linkage joints are not controlled DOFs
    torch.testing.assert_close(robot.get_dofs_armature(dofs_idx_local=every_dof)[0], torch.zeros(12))
    torch.testing.assert_close(robot.get_dofs_kp()[0], torch.tensor([0.0] * 6 + [17.0]))
    torch.testing.assert_close(robot.get_dofs_force_range()[1][0], torch.tensor([150.0] * 3 + [28.0] * 3 + [60.0]))


def test_arm_kinematics_match_uwlab_calibration(env):
    robot = env.entities["robot"]
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(-np.pi, np.pi, 6)
        robot.set_dofs_pos(torch.tensor([[*q, 0.0]], dtype=torch.float32))
        base_pos, base_quat = link_pose(robot, "base_link")
        wrist_pos, wrist_quat = link_pose(robot, "wrist_3_link")
        base_rot = Rotation.from_quat(base_quat, scalar_first=True)
        expected = _calibrated_wrist_pose(q)
        np.testing.assert_allclose(base_rot.inv().apply(wrist_pos - base_pos), expected[:3, 3], atol=2e-5)
        relative = base_rot.inv() * Rotation.from_quat(wrist_quat, scalar_first=True)
        assert (Rotation.from_matrix(expected[:3, :3]).inv() * relative).magnitude() < 1e-4


def test_gripper_closes_with_the_linkage_following_finger_joint(env):
    robot = env.entities["robot"]
    home = torch.tensor([[0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708, 0.0]])
    robot.set_dofs_pos(home)
    _, linkage = robot.find_named_dofs_idx_local(list(_LINKAGE_SIGNS))
    _, driver = robot.find_named_dofs_idx_local([GRIPPER_DRIVER])
    robot.control_dofs_pos(torch.tensor([[0.785398]]), dofs_idx_local=driver)
    for _ in range(120):  # 1 s at the policy's 120 Hz physics step
        env.scene.step()
    q = robot.get_dofs_pos(dofs_idx_local=linkage)[0].cpu().numpy()
    assert q[0] > 0.7, f"finger_joint reached only {q[0]:.3f} rad of 0.785 in 1 s"
    np.testing.assert_allclose(q, q[0] * np.array(list(_LINKAGE_SIGNS.values())), atol=5e-3)
    # The two inner fingers mirror each other across the gripper's x-z plane.
    base_pos, base_quat = link_pose(robot, "robotiq_base_link")
    base_rot = Rotation.from_quat(base_quat, scalar_first=True)
    left = base_rot.inv().apply(link_pose(robot, "left_inner_finger")[0] - base_pos)
    right = base_rot.inv().apply(link_pose(robot, "right_inner_finger")[0] - base_pos)
    np.testing.assert_allclose(left * [1, -1, 1], right, atol=1e-3)
