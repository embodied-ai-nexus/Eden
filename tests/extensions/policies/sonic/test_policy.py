"""``SonicPolicy``: its pinned robot, its encoder/decoder layout, and ``step()`` driven through tiny real ONNX graphs.

The released bundle is not available offline, so ``step()`` runs against two stand-in graphs whose
outputs expose what the policy fed them: an encoder that passes its leading input columns through,
and a decoder that either does the same or emits a one-hot action.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eden.extensions.deployment.base import resolve_dofs_values
from eden.extensions.policies.holosoma.policy import holosoma_g1_robot
from eden.extensions.policies.sonic.joint_maps import ACTION_SCALE_MJ, DEFAULT_ANGLES_MJ
from eden.extensions.policies.sonic.policy import (
    _DECODER_INPUT_DIM,
    _DECODER_OFFSETS,
    _DECODER_SLOTS,
    _ENCODER_INPUT_DIM,
    _ENCODER_OFFSETS,
    _ENCODER_SLOTS,
    SonicPolicy,
    _quat_to_6d,
)
from eden.options.robots.unitree_g1 import G1Dofs29

MJ_NAMES = G1Dofs29().dofs_name
#: Every joint of the reference clip sits at this angle, so a raw (not default-subtracted) feed is visible.
MOTION_JOINT_POS = 0.1


# ---------------------------------------------------------------------------
# the pinned robot
# ---------------------------------------------------------------------------


def test_robot_runs_the_released_hip_pitch_gains():
    """The stock G1 would deploy the hips at 0.41x the checkpoint's stiffness (40.18 vs 99.10)."""
    robot = SonicPolicy.robot_options()
    kp = resolve_dofs_values(robot.default_dofs_kp, robot.dofs_name)
    kd = resolve_dofs_values(robot.default_dofs_kd, robot.dofs_name)
    for side in ("left", "right"):
        i = robot.dofs_name.index(f"{side}_hip_pitch_joint")
        assert (kp[i], kd[i]) == (pytest.approx(99.098428, rel=1e-6), pytest.approx(6.308802, rel=1e-6))
    # Control gains only: armature and effort limit describe the actuator and keep the stock spec.
    assert robot.default_dofs_armature["*_hip_pitch_joint"] == pytest.approx(0.010177520, rel=1e-6)
    assert robot.default_dofs_force_limits["*_hip_pitch_joint"] == pytest.approx(88.0)


def test_robot_stands_in_the_pose_the_action_transform_is_centred_on():
    """A stock G1 handed the policy a robot 0.37 rad off at the knees on the step it took over."""
    robot = SonicPolicy.robot_options()
    assert resolve_dofs_values(robot.default_dofs_pos, robot.dofs_name) == pytest.approx(DEFAULT_ANGLES_MJ, abs=1e-12)
    # Holosoma pins the same stance, so a two-slot deploy built from either robot costs only the gains.
    assert robot.default_dofs_pos == pytest.approx(holosoma_g1_robot().default_dofs_pos)


# ---------------------------------------------------------------------------
# encoder / decoder buffer layout
# ---------------------------------------------------------------------------


def test_input_dims_match_the_release():
    assert (_ENCODER_INPUT_DIM, _DECODER_INPUT_DIM) == (1762, 994)


@pytest.mark.parametrize(
    "slots, offsets, total",
    [(_ENCODER_SLOTS, _ENCODER_OFFSETS, _ENCODER_INPUT_DIM), (_DECODER_SLOTS, _DECODER_OFFSETS, _DECODER_INPUT_DIM)],
)
def test_slot_offsets_tile_the_buffer_in_order(slots, offsets, total):
    cursor = 0
    for name, (start, end) in offsets.items():
        assert (start, end) == (cursor, cursor + slots[name]), name
        cursor = end
    assert cursor == total


@pytest.mark.parametrize(
    "yaw, expected",
    [(0.0, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), (math.pi / 2, [0.0, -1.0, 1.0, 0.0, 0.0, 0.0])],
)
def test_quat_to_6d_flattens_the_first_two_columns_row_major(yaw, expected):
    """``[R00, R01, R10, R11, R20, R21]``: the column-stacked layout would feed the encoder other scalars."""
    q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]).expand(3, 1, 4)
    torch.testing.assert_close(_quat_to_6d(q), torch.tensor(expected).expand(3, 1, 6), atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# step(), through the task-less env ``eden inference --policy sonic`` builds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from eden.utils.cli.tasks import _build_extension_env, _extension_command_options

    frames = 60
    quat = np.zeros((frames, 1, 4), np.float32)
    quat[..., 0] = 1.0
    pelvis_pos = np.zeros((frames, 1, 3), np.float32)
    pelvis_pos[..., 2] = 0.793
    motion_file = tmp_path_factory.mktemp("sonic") / "motion.npz"
    np.savez(
        motion_file,
        joint_pos=np.full((frames, 29), MOTION_JOINT_POS, np.float32),
        joint_vel=np.zeros((frames, 29), np.float32),
        body_pos_w=pelvis_pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((frames, 1, 3), np.float32),
        body_ang_vel_w=np.zeros((frames, 1, 3), np.float32),
        body_names=np.array(["pelvis"]),
    )
    command = _extension_command_options(SonicPolicy, "robot", str(motion_file))
    env = _build_extension_env(SonicPolicy, "G1Dofs29", "robot", False, command)
    env.reset()
    return env


@pytest.fixture
def step_targets(env, local_sonic_policy, monkeypatch):
    """Return ``run(decoder_nodes=None, decoder_initializers=None, robot=None) -> (29,)`` targets from one ``step()``.

    The default decoder is the pass-through (see ``local_sonic_policy``); ``robot``, if given,
    maps the simulated robot to the entity ``step()`` reads instead.
    """

    def run(decoder_nodes=None, decoder_initializers=None, robot=None):
        policy_cls = local_sonic_policy(SonicPolicy, decoder_nodes, decoder_initializers)
        policy = policy_cls(env=env, options=policy_cls.configure())
        policy.reset()
        if robot is not None:  # what `eden deploy` does to the built env before the first step
            monkeypatch.setitem(env.entities, "robot", robot(env.entities["robot"]))
        return policy.step(obs=None)["dofs_pos_controller"][0].cpu().numpy()

    return run


def test_encoder_input_opens_with_a_zero_g1_mode_then_the_raw_reference_joints(step_targets, isaaclab_joint_names):
    """``encoder_mode_4`` is ``[mode_id, 0, 0, 0]`` = zeros for g1, not a one-hot (that one means teleop).

    The action is the encoder's first 29 input columns: the four mode slots, then the current reference
    frame's joint positions in IsaacLab order, fed raw rather than relative to the default pose.
    """
    targets = step_targets()  # pass-through decoder
    il_slot = [isaaclab_joint_names.index(name) for name in MJ_NAMES]
    action_il = np.array([0.0 if slot < 4 else MOTION_JOINT_POS for slot in il_slot])
    np.testing.assert_allclose(targets, np.array(DEFAULT_ANGLES_MJ) + np.array(ACTION_SCALE_MJ) * action_il, atol=1e-6)


@pytest.mark.parametrize("joint", ["right_hip_pitch_joint", "waist_yaw_joint", "right_wrist_yaw_joint"])
def test_an_isaaclab_order_action_lands_on_its_joint_with_that_joints_scale(step_targets, isaaclab_joint_names, joint):
    """Gathering with ``MJ_TO_IL`` instead would move another joint, by another joint's scale."""
    bias = np.zeros(29, np.float32)
    bias[isaaclab_joint_names.index(joint)] = 1.0
    targets = step_targets(
        [("MatMul", ["obs_dict", "zero"], ["h"], {}), ("Add", ["h", "bias"], ["action"], {})],
        {"zero": np.zeros((_DECODER_INPUT_DIM, 29), np.float32), "bias": bias},
    )
    expected = np.array(DEFAULT_ANGLES_MJ)
    expected[MJ_NAMES.index(joint)] += ACTION_SCALE_MJ[MJ_NAMES.index(joint)]
    np.testing.assert_allclose(targets, expected, atol=1e-6)


def test_step_reads_the_deployment_robot_entity(step_targets, env):
    """``eden deploy`` swaps the robot for a ``RobotStateEntity``, whose pose getters refuse ``relative=True``.

    A ``get_quat(relative=True)`` in the state read raised ``NotImplementedError`` on the first policy tick
    of every deployment backend.
    """
    from eden.extensions.deployment.utils.state import RobotStateEntity

    sim_targets = step_targets()
    deploy_targets = step_targets(robot=lambda sim: RobotStateEntity(sim, device=env.device))
    np.testing.assert_allclose(deploy_targets, sim_targets, atol=1e-6)


def test_base_policy_keeps_the_validated_contact_settings():
    """The teleop contact tuning (multi-contact, a global constraint timeconst) must not reach deploy / inference."""
    env_options = SonicPolicy.env_options()
    assert env_options["enable_multi_contact"] is False
    assert "constraint_timeconst" not in env_options
