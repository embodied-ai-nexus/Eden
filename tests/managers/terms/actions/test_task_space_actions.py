"""The task-space controllers in ``eden/managers/terms/actions/task_space_actions.py``.

Built on a real Panda. The operational-space expectations are the controller's definition written
out against the entity's own Jacobian and mass matrix, ``tau = J^T Lambda a`` with
``Lambda = (J M^-1 J^T)^-1``, solved here with a plain inverse where the controller uses a damped
block pseudoinverse. The comparisons that invert the operational inertia set that damping
negligibly small, since at the home pose it measurably damps the weakest task direction.

``IKPoseController`` is checked kinematically: the arm is teleported to the joint targets it solves
and the hand is read back in the base frame.
"""

from __future__ import annotations

import math

import pytest
import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.managers.terms.actions.task_space_actions import (
    DifferentialIKController,
    IKPoseController,
    OperationalSpaceController,
    _damped_pseudoinverse,
    _solve_mass_matrix_jacobian_transpose,
)
from eden.options.managers.actions import ActionTermOptions
from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.robosuite import RethinkMount
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from eden.options.robots.ufactory_xarm7 import Xarm7
from eden.options.robots.universal_ur5e import UR5e
from eden.utils.configs import EdenConfig
from eden.utils.geom import (
    quat_box_minus,
    quat_from_axis_angle,
    quat_mul,
    subtract_frame_transforms,
    transform_by_quat,
)
from tests.managers.terms.conftest import PANDA_ARM, panda_options, wheel_options
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

HAND = "hand"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A Panda (``robot``) with the two-hinge wheel welded to its hand (``wheel``), and a free ``cube``."""
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                robot=panda_options(),
                # Welding drops the wheel's root hinge, so only the flap is left to name.
                wheel=wheel_options(
                    tmp_path_factory.mktemp("wheel") / "wheel.xml",
                    default_root_pos=(0.0, 0.0, 0.1),
                    dofs_name=["flap"],
                    default_dofs_pos={"flap": 0.0},
                ),
                cube=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(1.0, 1.0, 1.0)),
                attachments_dict={"wheel": ("robot", HAND)},
            ),
        )
    )


@pytest.fixture
def robot(env, restore_state):
    return restore_state(env.entities["robot"])


def _osc(env, **params):
    params = {
        "entity_name": "robot",
        "dofs_name": PANDA_ARM,
        "ee_link_name": HAND,
        "kp": 1.0,
        "damping_ratio": 0.0,
        "inertial_decoupling": "none",
        "nullspace_kp": None,
        **params,
    }
    term = OperationalSpaceController(env, OperationalSpaceController.configure(**params))
    term.build()
    return term


def _jacobian(robot, rows=6, local_point=None, dofs=PANDA_ARM):
    """The hand Jacobian, first ``rows`` task rows, columns in ``dofs`` order."""
    full = robot.get_jacobian(link=robot.get_link(HAND), local_point=local_point)
    return full[:, :rows, [robot.dofs_name.index(name) for name in dofs]]


def _mass(robot, dofs=PANDA_ARM):
    idx = [robot.dofs_name.index(name) for name in dofs]
    return robot.get_mass_mat()[:, idx][:, :, idx]


def _apply(term):
    term.apply_actions()
    return term.processed_action.clone()


def _move_arm(robot, delta):
    pos = robot.get_dofs_pos().clone()
    pos[:, :7] += torch.tensor(delta)
    robot.set_dofs_pos(pos)


def _hand_pose(robot):
    link = robot.find_named_links_idx_local([HAND])[1]
    return robot.get_links_pos(ls_idx_local=link)[:, 0], robot.get_links_quat(ls_idx_local=link)[:, 0]


def _bmv(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# DifferentialIKController
# ---------------------------------------------------------------------------


def test_differential_ik_moves_the_hand_by_the_commanded_displacement(env, robot):
    """Joint targets one IK step away land the hand on ``scale * tanh(action)`` to first order."""
    term = DifferentialIKController(
        env,
        DifferentialIKController.configure(entity_name="robot", dofs_name=PANDA_ARM, ee_link_name=HAND, damping=1e-4),
    )
    term.build()
    start, _ = _hand_pose(robot)
    term.compute(torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.4, 0.0, 0.0, 0.0]]))
    targets = robot.get_dofs_pos().clone()
    targets[:, :7] = term.processed_action
    robot.set_dofs_pos(targets)
    moved, _ = _hand_pose(robot)
    expected = torch.tensor([[0.05 * math.tanh(0.2), 0.0, 0.0], [0.0, 0.0, 0.05 * math.tanh(-0.4)]])
    torch.testing.assert_close(moved - start, expected, atol=1e-3, rtol=0.0)  # 1 and 2 cm moves, to 1 mm


def test_differential_ik_turns_the_hand_about_its_control_point(env, robot):
    """A pure rotation command leaves ``ee_local_pos`` in place, where the link origin swings about it."""
    local = torch.tensor([0.0, 0.0, 0.1])  # 10 cm along the hand's z, toward the fingertips

    def control_point():
        pos, quat = _hand_pose(robot)
        return pos + transform_by_quat(local.expand_as(pos), quat)

    term = DifferentialIKController(
        env,
        DifferentialIKController.configure(
            entity_name="robot",
            dofs_name=PANDA_ARM,
            ee_link_name=HAND,
            ee_local_pos=tuple(local.tolist()),
            damping=1e-4,
        ),
    )
    term.build()
    start_point = control_point()
    start_hand, _ = _hand_pose(robot)
    term.compute(torch.tensor([[0.0, 0.0, 0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.2, 0.0]]))
    targets = robot.get_dofs_pos().clone()
    targets[:, :7] = term.processed_action
    robot.set_dofs_pos(targets)
    hand, _ = _hand_pose(robot)
    # A ~0.1 rad turn swings the hand origin ~1 cm about the point, which stays put to second order.
    assert bool(torch.all((hand - start_hand).norm(dim=-1) > 5e-3))
    torch.testing.assert_close(control_point(), start_point, atol=1e-3, rtol=0.0)


def test_differential_ik_rejects_a_control_point_that_is_not_three_finite_values(env):
    term = DifferentialIKController(
        env,
        DifferentialIKController.configure(
            entity_name="robot", dofs_name=PANDA_ARM, ee_link_name=HAND, ee_local_pos=(0.0, math.nan, 0.1)
        ),
    )
    with pytest.raises(ValueError, match=r"ee_local_pos must contain three finite values, got \(0\.0, nan, 0\.1\)"):
        term.build()


def test_differential_ik_scales_position_and_orientation_independently(env):
    term = DifferentialIKController(
        env,
        DifferentialIKController.configure(
            entity_name="robot",
            dofs_name=PANDA_ARM,
            ee_link_name=HAND,
            position_scale=[0.1, 0.2, 0.3],
            orientation_scale=[0.4, 0.5, 0.6],
        ),
    )
    term.build()
    torch.testing.assert_close(term.effective_scale, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))


@pytest.mark.parametrize("controller_cls", [DifferentialIKController, OperationalSpaceController])
def test_task_space_controllers_reject_a_combined_scale(controller_cls):
    with pytest.raises(ValueError, match=r"invalid: \['scale'\]"):
        controller_cls.configure(entity_name="robot", scale=0.5)


@pytest.mark.parametrize(
    ("field", "bad_scale", "error", "match"),
    [
        ("position_scale", None, TypeError, "got NoneType"),
        # A str of len 3 would otherwise reach torch.tensor() as a "valid" vector.
        ("orientation_scale", "abc", TypeError, "got str"),
        # A dict of the wrong length would otherwise be misreported as a wrong-length vector.
        ("position_scale", {"a": 1.0}, TypeError, "got dict"),
        ("orientation_scale", True, TypeError, "got bool"),
        ("position_scale", [0.1, 0.2], ValueError, r"contain 3 values; got 2"),
        ("orientation_scale", [0.1, 0.2], ValueError, r"contain 3 values; got 2"),
    ],
)
def test_differential_ik_rejects_a_bad_axis_scale_at_build(env, field, bad_scale, error, match):
    class BadScaleIK(DifferentialIKController):
        ee_link_name = HAND

    setattr(BadScaleIK, field, bad_scale)
    term = BadScaleIK(env, ActionTermOptions(entity_name="robot", dofs_name=PANDA_ARM))  # keeps the class value
    with pytest.raises(error, match=rf"Axis parameter.*{match}"):
        term.build()


# ---------------------------------------------------------------------------
# OperationalSpaceController: the commanded torque
# ---------------------------------------------------------------------------


def test_direct_impedance_is_the_jacobian_transpose_of_the_scaled_pose_delta(env, robot):
    """Position and orientation scale separately, and the goal is a world-frame delta on the current pose."""
    term = _osc(env, control_mode="pose", position_scale=(0.05, 0.1, 0.15), orientation_scale=(0.2, 0.25, 0.3))
    torch.testing.assert_close(term.effective_scale, torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))
    action = torch.tensor([[1.0, -1.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.5]])
    term.compute(action)
    wrench = action * term.effective_scale
    torch.testing.assert_close(_apply(term), _bmv(_jacobian(robot).transpose(1, 2), wrench), atol=1e-5, rtol=1e-4)


def test_the_goal_holds_while_feedback_is_recomputed_every_physics_step(env, robot):
    """One policy action fixes the goal; moving the arm changes the error, and so the torque."""
    term = _osc(env, control_mode="position", position_scale=1.0, kp=10.0)
    start, _ = _hand_pose(robot)
    term.compute(torch.tensor([[0.02, 0.0, 0.0]] * 2))
    goal = start + torch.tensor([0.02, 0.0, 0.0])
    _move_arm(robot, [0.1, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    now, _ = _hand_pose(robot)
    expected = _bmv(_jacobian(robot, rows=3).transpose(1, 2), 10.0 * (goal - now))
    torch.testing.assert_close(_apply(term), expected, atol=1e-5, rtol=1e-4)


def test_damping_acts_on_the_task_velocity_of_every_dof(env, robot):
    """joint7 is not controlled, yet its velocity moves the hand, and the damping opposes that motion."""
    dofs = PANDA_ARM[:6]
    term = _osc(env, dofs_name=dofs, control_mode="position", kp=1.0, damping_ratio=1.0)
    vel = torch.zeros(2, 9)
    vel[:, 6] = 0.5
    robot.set_dofs_vel(vel)
    task_velocity = _bmv(_jacobian(robot, rows=3), vel[:, :7])  # every arm DOF's contribution
    expected = _bmv(_jacobian(robot, rows=3, dofs=dofs).transpose(1, 2), -2.0 * task_velocity)  # kd = 2 sqrt(kp)
    torch.testing.assert_close(_apply(term), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("control_mode", ["position", "pose"])
def test_full_decoupling_scales_the_acceleration_by_the_operational_inertia(env, robot, control_mode):
    rows = 3 if control_mode == "position" else 6
    term = _osc(
        env,
        control_mode=control_mode,
        inertial_decoupling="full",
        position_scale=1.0,
        orientation_scale=1.0,
        operational_inertia_damping=1e-10,
    )
    action = torch.tensor([[0.01, -0.02, 0.015, 0.02, 0.0, -0.01][:rows]] * 2)
    term.compute(action)
    jac, mass = _jacobian(robot, rows=rows), _mass(robot)
    inertia = torch.linalg.inv(jac @ torch.linalg.solve(mass, jac.transpose(1, 2)))
    torch.testing.assert_close(_apply(term), _bmv(jac.transpose(1, 2) @ inertia, action), atol=1e-4, rtol=1e-3)


def test_position_orientation_decoupling_inverts_each_block_on_its_own(env, robot):
    term = _osc(
        env,
        control_mode="pose",
        inertial_decoupling="position_orientation",
        position_scale=1.0,
        orientation_scale=1.0,
        operational_inertia_damping=1e-10,
    )
    action = torch.tensor([[0.01, -0.02, 0.015, -0.015, 0.025, -0.005]] * 2)
    term.compute(action)
    jac, mass = _jacobian(robot), _mass(robot)
    inverse_inertia = jac @ torch.linalg.solve(mass, jac.transpose(1, 2))
    wrench = torch.cat(
        [
            _bmv(torch.linalg.inv(inverse_inertia[:, :3, :3]), action[:, :3]),
            _bmv(torch.linalg.inv(inverse_inertia[:, 3:, 3:]), action[:, 3:]),
        ],
        dim=-1,
    )
    torch.testing.assert_close(_apply(term), _bmv(jac.transpose(1, 2), wrench), atol=1e-4, rtol=1e-3)
    full = _bmv(jac.transpose(1, 2) @ torch.linalg.inv(inverse_inertia), action)
    assert not torch.allclose(term.processed_action, full, atol=1e-4)  # the coupled inertia would differ


def test_the_torque_comes_back_in_the_configured_dof_order(env, robot):
    """``dofs_name`` in any order is one controller space: the same torques, permuted with it."""
    permuted = ["joint3", "joint1", "joint2", "joint6", "joint4", "joint5", "joint7"]
    action = torch.tensor([[0.01, -0.02, 0.015]] * 2)
    natural = _osc(env, control_mode="position", inertial_decoupling="full", position_scale=1.0)
    shuffled = _osc(env, dofs_name=permuted, control_mode="position", inertial_decoupling="full", position_scale=1.0)
    natural.compute(action)
    shuffled.compute(action)
    torch.testing.assert_close(_apply(shuffled), _apply(natural)[:, [PANDA_ARM.index(n) for n in permuted]])


def test_the_nullspace_posture_torque_does_not_accelerate_the_task(env, robot):
    """At zero task error the posture pull toward the default pose is all there is, and ``J M^-1`` kills it."""
    _move_arm(robot, [0.2, -0.1, 0.1, 0.2, -0.1, 0.1, 0.0])
    term = _osc(
        env, control_mode="position", nullspace_kp=5.0, nullspace_damping_ratio=0.0, operational_inertia_damping=1e-10
    )
    term.compute(torch.zeros(2, 3))
    torque = _apply(term)
    assert (torque.abs() > 1e-3).any()
    jac, mass = _jacobian(robot, rows=3), _mass(robot)
    torch.testing.assert_close(_bmv(jac, torch.linalg.solve(mass, torque)), torch.zeros(2, 3), atol=1e-4, rtol=0.0)


def test_the_commanded_torque_is_left_for_the_entity_to_saturate(env, robot):
    """The controller's own command is not clipped to the force range; the entity's actuators clip it."""
    term = _osc(env, control_mode="position", position_scale=1.0, kp=1e5)
    term.compute(torch.tensor([[0.5, 0.0, 0.0]] * 2))
    torque = _apply(term)
    _, upper = robot.get_dofs_force_range()
    assert (torque.abs() > upper[:, :7]).any()
    assert (robot.get_dofs_control_force()[:, :7].abs() <= upper[:, :7] + 1e-4).all()


def test_a_singular_configuration_yields_a_finite_torque(env, robot):
    """The arm stretched straight up loses a task direction; the damped inverse keeps the command finite."""
    robot.set_dofs_pos(torch.zeros(2, 9))
    term = _osc(env, control_mode="position", inertial_decoupling="full", position_scale=1.0, kp=100.0)
    term.compute(torch.tensor([[0.1, -0.2, 0.3]] * 2))
    torque = _apply(term)
    assert torch.isfinite(torque).all() and (torque.abs() > 0).any()


def test_invalid_dynamics_are_contained_per_env():
    """A singular mass matrix or a non-finite Jacobian zeroes that env's command, never its neighbours'."""
    jac = torch.eye(3, 4).expand(3, 3, 4).clone()
    mass = torch.eye(4).expand(3, 4, 4).clone()
    mass[1] = 0.0
    jac[2, 0, 0] = torch.nan
    solved, invalid = _solve_mass_matrix_jacobian_transpose(mass, jac)
    assert invalid.tolist() == [False, True, True]
    torch.testing.assert_close(solved[0], jac[0].T)
    torch.testing.assert_close(solved[1:], torch.zeros(2, 4, 3))

    rank_deficient = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[torch.nan, 0.0], [0.0, 1.0]]])
    pinv, invalid = _damped_pseudoinverse(rank_deficient, torch.eye(2).unsqueeze(0), 1e-6)
    assert invalid.tolist() == [False, True]
    torch.testing.assert_close(pinv[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]]), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(pinv[1], torch.zeros(2, 2))


# ---------------------------------------------------------------------------
# OperationalSpaceController: the control point and the goal
# ---------------------------------------------------------------------------


def test_the_control_point_is_carried_in_the_hand_frame(env, robot):
    """A control point at the fingertip midpoint keeps tracking that midpoint after the hand turns."""

    def midpoint():
        _, fingers = robot.find_named_links_idx_local(["left_finger", "right_finger"])
        return robot.get_links_pos(ls_idx_local=fingers).mean(dim=1)

    hand_pos, hand_quat = _hand_pose(robot)
    local = inv_transform_by_quat(midpoint() - hand_pos, hand_quat)[0]
    term = _osc(env, control_mode="position", ee_local_pos=tuple(local.tolist()))
    term.compute(torch.zeros(2, 3))
    start = midpoint()
    _move_arm(robot, [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5])  # wrist joints: the hand turns about itself
    expected = _bmv(_jacobian(robot, rows=3, local_point=local).transpose(1, 2), start - midpoint())
    torch.testing.assert_close(_apply(term), expected, atol=1e-5, rtol=1e-4)


class _HalfOrientationErrorOsc(OperationalSpaceController):
    def _orientation_error(self, goal_quat, current_quat):
        return 0.5 * super()._orientation_error(goal_quat, current_quat)


class _LiberoGoalOsc(OperationalSpaceController):
    """The LIBERO benchmark's goal lifecycle: keep the orientation goal unless rotated; reset to the build one."""

    def build(self):
        super().build()
        self._construction_goal_quat = self._goal_quat.clone()

    def _update_goal_from_action(self, current_pos, current_quat, scaled_action):
        previous_goal_quat = self._goal_quat.clone()
        super()._update_goal_from_action(current_pos, current_quat, scaled_action)
        has_rotation_delta = torch.any(scaled_action[:, 3:] != 0.0, dim=1, keepdim=True)
        torch.where(has_rotation_delta, self._goal_quat, previous_goal_quat, out=self._goal_quat)

    def _reset_goal(self, envs_idx):
        current_pos, _ = self._control_point_pose()
        self._goal_pos[envs_idx] = current_pos[envs_idx]
        self._goal_quat[envs_idx] = self._construction_goal_quat[envs_idx]


def _rotation_torque(robot, rotation):
    return _bmv(_jacobian(robot).transpose(1, 2), torch.nn.functional.pad(rotation, (3, 0)))


def test_the_orientation_error_hook_shapes_the_torque(env, robot):
    params = {"control_mode": "pose", "position_scale": 1.0, "orientation_scale": 1.0}
    rotation = torch.tensor([[0.4, 0.0, 0.0]] * 2)  # about world x, whatever the hand's own orientation
    for cls, gain in ((OperationalSpaceController, 1.0), (_HalfOrientationErrorOsc, 0.5)):
        term = cls(
            env,
            cls.configure(
                entity_name="robot", dofs_name=PANDA_ARM, ee_link_name=HAND, kp=1.0, damping_ratio=0.0, **params
            ),
        )
        term.build()
        term.compute(torch.nn.functional.pad(rotation, (3, 0)))
        torch.testing.assert_close(_apply(term), gain * _rotation_torque(robot, rotation), atol=1e-5, rtol=1e-4)


def test_the_goal_hooks_carry_a_benchmarks_orientation_lifecycle(env, robot):
    term = _LiberoGoalOsc(
        env,
        _LiberoGoalOsc.configure(
            entity_name="robot",
            dofs_name=PANDA_ARM,
            ee_link_name=HAND,
            control_mode="pose",
            position_scale=1.0,
            orientation_scale=1.0,
            kp=1.0,
            damping_ratio=0.0,
        ),
    )
    term.build()
    _, built_quat = _hand_pose(robot)
    term.compute(torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0, 0.0]] * 2))
    term.compute(torch.zeros(2, 6))  # no rotation delta: the rotated goal is kept
    torch.testing.assert_close(
        _apply(term), _rotation_torque(robot, torch.tensor([[0.5, 0.0, 0.0]] * 2)), atol=1e-5, rtol=1e-4
    )

    _move_arm(robot, [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3])
    term.reset()  # the goal returns to the build-time orientation, at the current position
    _, now_quat = _hand_pose(robot)
    torch.testing.assert_close(
        _apply(term), _rotation_torque(robot, quat_box_minus(built_quat, now_quat)), atol=1e-5, rtol=1e-4
    )


def test_a_partial_reset_reanchors_only_those_envs(env, robot):
    """Env 1 is moved and reset: it then commands what a controller built at its new pose would."""
    term = _osc(env, control_mode="position", position_scale=1.0, nullspace_kp=10.0, nullspace_damping_ratio=0.0)
    term.compute(torch.tensor([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]))
    before = _apply(term)
    pos = robot.get_dofs_pos().clone()
    pos[1, :7] += 0.1
    robot.set_dofs_pos(pos)
    term.reset(torch.tensor([1]))
    after = _apply(term)
    fresh = _osc(env, control_mode="position", position_scale=1.0, nullspace_kp=10.0, nullspace_damping_ratio=0.0)
    fresh.compute(torch.zeros(2, 3))
    torch.testing.assert_close(after[1], _apply(fresh)[1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(after[0], before[0])


# ---------------------------------------------------------------------------
# OperationalSpaceController: what build() refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"dofs_name": PANDA_ARM[:6], "inertial_decoupling": "full"}, r"dofs_name omits .*joint7"),
        (
            {"dofs_name": PANDA_ARM[:6], "control_mode": "pose", "nullspace_kp": 10.0},
            "more selected DOFs than controlled task axes",
        ),
        ({"ee_link_name": "left_.*"}, "must resolve exactly one link"),
        ({"position_scale": (0.1, 0.2)}, "Axis parameter.*contain 3 values; got 2"),
        ({"control_mode": "pose", "orientation_scale": (0.1, 0.2)}, "Axis parameter.*contain 3 values; got 2"),
        ({"operational_inertia_damping": 0.0}, "operational_inertia_damping must be positive"),
    ],
    ids=["ancestor-omitted", "no-redundancy", "ambiguous-link", "position-scale", "orientation-scale", "damping"],
)
def test_osc_rejects_a_bad_configuration_at_build(env, params, match):
    with pytest.raises(ValueError, match=match):
        _osc(env, **params)


@pytest.mark.parametrize(
    ("entity", "dofs_name", "match"),
    [("cube", [], "no root DOFs"), ("wheel", ["flap"], "welded to a moving parent")],
    ids=["floating-base", "moving-parent"],
)
def test_osc_rejects_an_unsupported_topology(env, entity, dofs_name, match):
    with pytest.raises(ValueError, match=match):
        _osc(env, entity_name=entity, dofs_name=dofs_name, ee_link_name="flap_link")


def test_osc_rejects_an_action_of_the_wrong_width(env):
    term = _osc(env, control_mode="position")
    assert term.action_dim == 3 and _osc(env, control_mode="pose").action_dim == 6
    with pytest.raises(ValueError, match="shape"):
        term.compute(torch.zeros(2, 6))


# ---------------------------------------------------------------------------
# IKPoseController: absolute base-frame pose targets, solved from the measured joints
# ---------------------------------------------------------------------------


def _ik(env, **params):
    options = ActionTermOptions(entity_name="robot", dofs_name=PANDA_ARM, ee_link_name=HAND, **params)
    term = IKPoseController(env, options)
    term.build()
    return term


def _hand_in_base(robot, tool_offset=(0.0, 0.0, 0.0)):
    """Base-frame pose of the frame ``tool_offset`` from the hand origin (in the hand frame)."""
    index = [robot.get_link(HAND).idx_local]
    pos_w, quat_w = robot.get_links_pos(ls_idx_local=index)[:, 0], robot.get_links_quat(ls_idx_local=index)[:, 0]
    offset = torch.tensor(tool_offset, dtype=pos_w.dtype, device=pos_w.device).expand_as(pos_w)
    return subtract_frame_transforms(
        robot.get_pos(), robot.get_quat(), pos_w + transform_by_quat(offset, quat_w), quat_w
    )


def _arm(robot, term):
    return robot.get_dofs_pos(dofs_idx_local=term.genesis_dofs_idx).clone()


@pytest.mark.parametrize("tool_offset", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1)], ids=["hand", "tool-point"])
def test_ik_pose_controller_places_the_tool_frame_at_the_base_frame_target(env, robot, tool_offset):
    term = _ik(env, ee_link_offset_pos=tool_offset)
    pos_b, quat_b = _hand_in_base(robot, tool_offset)
    target_pos_b = pos_b + torch.tensor([0.03, -0.02, 0.02])
    before = _arm(robot, term)

    term.compute(torch.cat([target_pos_b, quat_b], dim=-1))
    robot.set_dofs_pos(term.processed_action, dofs_idx_local=term.genesis_dofs_idx)

    reached_pos, reached_quat = _hand_in_base(robot, tool_offset)
    torch.testing.assert_close(reached_pos, target_pos_b, atol=2e-3, rtol=0)
    assert (torch.linalg.vector_norm(quat_box_minus(reached_quat, quat_b), dim=-1) < 1e-2).all()
    # Seeded from the measured joints: a 4 cm move stays on the current elbow branch.
    assert (term.processed_action - before).abs().max() < 0.3


def test_ik_pose_controller_bounds_each_joints_step(env, robot):
    term = _ik(env, max_joint_delta=0.05)
    pos_b, quat_b = _hand_in_base(robot)
    before = _arm(robot, term)

    term.compute(torch.cat([pos_b + torch.tensor([0.25, 0.2, -0.2]), quat_b], dim=-1))

    step = (term.processed_action - before).abs()
    assert step.max() == pytest.approx(0.05, abs=1e-6)
    assert (step <= 0.05 + 1e-6).all()


def test_ik_pose_controller_holds_the_measured_arm_when_the_residual_is_too_large(env, robot):
    term = _ik(env, max_ik_position_error=0.01)
    pos_b, quat_b = _hand_in_base(robot)
    before = _arm(robot, term)

    term.compute(torch.cat([pos_b + torch.tensor([3.0, 0.0, 0.0]), quat_b], dim=-1))  # out of reach

    torch.testing.assert_close(term.processed_action, before)


def test_ik_pose_controller_rejects_a_floating_base(env):
    """The IK seed is a full qpos indexed by DOF; a free joint's quaternion breaks that."""
    term = IKPoseController(env, ActionTermOptions(entity_name="cube", dofs_name=[], ee_link_name=HAND))
    with pytest.raises(ValueError, match="fixed-base"):
        term.build()


# ---------------------------------------------------------------------------
# OperationalSpaceController: closed-loop physics
# ---------------------------------------------------------------------------

POSITION_SCALE = 0.05
ORIENTATION_SCALE = 0.5
#: entity name -> (end-effector link, controlled DOFs, nullspace_kp)
ARMS = {
    "xarm7": ("link_eef", [f"joint{index}" for index in range(1, 8)], 10.0),
    "ur5e": ("ee_link", ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"], None),
    "panda": ("hand", [f"joint{index}" for index in range(1, 8)], 10.0),
}
# Env 0 moves to a new pose; env 1 holds its current one.
POSITION_DELTA = [[-0.015, 0.012, 0.018], [0.0, 0.0, 0.0]]
ORIENTATION_DELTA = [[-0.06, 0.05, 0.05], [0.0, 0.0, 0.0]]


def _gravity_compensated():
    return RigidMaterialOptions(gravity_compensation=1.0)


@pytest.fixture(scope="module")
def tracked():
    """Closed loop on three arm topologies in one scene: an xArm7 and a UR5e on fixed bases, a Panda welded to a mount.

    One policy step toward each arm's target; per arm, ``(target_pos, target_quat, final_pos, final_quat, force)``.
    """
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=0.002, decimation=400),
            scene_options=SceneOptions(
                xarm7=Xarm7(material=_gravity_compensated()),
                ur5e=UR5e(default_root_pos=(1.5, 0.0, 0.0), material=_gravity_compensated()),
                # Genesis applies gravity compensation at the root of a welded kinematic tree.
                mount=RethinkMount(default_root_pos=(-1.5, 0.0, 0.922), material=_gravity_compensated()),
                panda=FrankaEmikaPanda(material=_gravity_compensated()),
                attachments_dict={"panda": ("mount", "base")},
            ),
            action_options=ActionManagerOptions(
                **{
                    name: OperationalSpaceController.configure(
                        entity_name=name,
                        dofs_name=dofs,
                        ee_link_name=ee_link,
                        control_mode="pose",
                        position_scale=POSITION_SCALE,
                        orientation_scale=ORIENTATION_SCALE,
                        kp=(150.0, 150.0, 150.0, 100.0, 100.0, 100.0),
                        damping_ratio=1.0,
                        inertial_decoupling="position_orientation",
                        nullspace_kp=nullspace_kp,
                    )
                    for name, (ee_link, dofs, nullspace_kp) in ARMS.items()
                }
            ),
        ),
        reset=True,
    )
    try:
        assert env.entities["panda"].is_attached
        position_delta = torch.tensor(POSITION_DELTA, device=env.device)
        orientation_delta = torch.tensor(ORIENTATION_DELTA, device=env.device)
        command = torch.cat([position_delta / POSITION_SCALE, orientation_delta / ORIENTATION_SCALE], dim=-1)

        def ee_pose(name):
            robot = env.entities[name]
            _, (index,) = robot.find_named_links_idx_local([ARMS[name][0]])
            return robot.get_links_pos(ls_idx_local=index).squeeze(1), robot.get_links_quat(ls_idx_local=index).squeeze(
                1
            )

        targets = {}
        for name in ARMS:
            pos, quat = ee_pose(name)
            targets[name] = (pos + position_delta, quat_mul(quat_from_axis_angle(orientation_delta), quat))

        # One policy action stays the fixed goal while torque is recomputed every physics substep.
        env.step({name: command for name in ARMS})

        yield {
            name: (*targets[name], *ee_pose(name), env.action_manager.processed_actions[name].clone()) for name in ARMS
        }
    finally:
        env.destroy()


@pytest.mark.slow
@pytest.mark.parametrize("arm", list(ARMS))
def test_operational_space_control_tracks_a_pose_target_and_holds_the_current_pose(tracked, arm):
    target_pos, target_quat, final_pos, final_quat, applied_force = tracked[arm]

    position_error = torch.linalg.vector_norm(target_pos - final_pos, dim=-1)
    orientation_error = torch.linalg.vector_norm(quat_box_minus(target_quat, final_quat), dim=-1)

    assert torch.isfinite(applied_force).all()
    assert (position_error < 1e-3).all(), f"Position errors: {position_error}"
    assert (orientation_error < 1e-2).all(), f"Orientation errors: {orientation_error}"
