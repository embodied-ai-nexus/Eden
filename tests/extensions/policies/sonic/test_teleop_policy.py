"""``SonicTeleopPolicy``: the built teleop robot against the reference plant, and its runtime contract.

The robot is built through the same task-less env path ``eden deploy/inference --policy`` uses, so
these pin what Genesis actually received, not the option dicts. The reference values are the
GEAR-SONIC MJCF as MuJoCo resolves it (the reference sim's plant).

The encoder-slot and history tests drive a real ``SonicTeleopPolicy.step`` on that env, fed by a
real ``MotionScheduler``, and read what the networks were fed through tiny real ONNX graphs: an
encoder that gathers chosen input columns into the token (the pass-through decoder hands token
columns 0..28 back as the action), an encoder that sums ``|x|`` per slot to show which slots were
filled, or a decoder that gathers chosen history columns.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from eden.extensions.policies.sonic.joint_maps import (
    ACTION_SCALE_MJ,
    DEFAULT_ANGLES_MJ,
    LOWER_BODY_IL_INDICES,
    MJ_TO_IL,
    MODE_G1,
    MODE_SMPL,
    MODE_TELEOP,
)
from eden.extensions.policies.sonic.motion import MotionFrames, MotionScheduler
from eden.extensions.policies.sonic.policy import (
    _DECODER_OFFSETS,
    _ENCODER_INPUT_DIM,
    _ENCODER_OFFSETS,
    RobotStateSnapshot,
)
from eden.extensions.policies.sonic.teleop_policy import (
    UPPER_BODY_IL_INDICES,
    VR_3POINT_DEFAULT_ORIENTATION,
    VR_3POINT_DEFAULT_POSITION,
    WRIST_IL_INDICES,
    SonicTeleopPolicy,
)
from eden.options.robots.unitree_g1 import (
    G1_DEX3_LEFT_DOFS_NAME,
    G1_DEX3_RIGHT_DOFS_NAME,
    G1_DOFS29_NAME,
    G1Dofs29Dex3,
)
from eden.utils.assets import get_asset_path
from tests.extensions.policies.sonic.conftest import yawq
from tests.support.sonic import IDENTITY, resolve_pattern


@pytest.fixture(scope="module")
def env():
    from eden.utils.cli.tasks import _build_extension_env

    env = _build_extension_env(SonicTeleopPolicy, "G1Dofs29Dex3", "robot", False, {})
    env.reset()
    return env


@pytest.fixture(scope="module")
def mjcf():
    """The teleop MJCF loaded by MuJoCo: per-DOF values after default-class resolution."""
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_path(get_asset_path(G1Dofs29Dex3().file))


def _mjcf_dof_values(model, names, attr):
    return np.array([getattr(model, attr)[model.joint(name).dofadr[0]] for name in names])


def test_built_armature_is_the_reference_mjcfs(env, mjcf):
    """An empty ``default_dofs_armature`` built the body at the actuator-spec inertias (knee 0.0251), not 0.01."""
    robot = env.entities["robot"]
    names = list(robot.dofs_name)
    built = robot.get_dofs_armature()[0].cpu().numpy()
    np.testing.assert_allclose(built, _mjcf_dof_values(mjcf, names, "dof_armature"), rtol=1e-6)


def test_built_gains_and_torque_clamps(env):
    robot = env.entities["robot"]
    names = list(robot.dofs_name)
    kp = robot.get_dofs_kp()[0].cpu().numpy()
    _, upper = robot.get_dofs_force_range()
    upper = upper[0].cpu().numpy()
    for side in ("left", "right"):
        # The checkpoint's 7520-22 hip pitch, not the stock spec's 7520-14 (40.18).
        assert kp[names.index(f"{side}_hip_pitch_joint")] == pytest.approx(99.098428, rel=1e-5)
        # Reference-sim clamps where tighter than the spec / MJCF.
        assert upper[names.index(f"{side}_hip_roll_joint")] == pytest.approx(88.0)
        assert upper[names.index(f"{side}_hand_thumb_0_joint")] == pytest.approx(2.45)
        assert upper[names.index(f"{side}_hand_index_1_joint")] == pytest.approx(0.7)


def test_pelvis_solver_and_authored_frames_share_their_rotation(env):
    """The runtime pairs ``get_quat(relative=True)`` with ``get_ang(frame="body")``, which rotates by ``get_quat()``.

    The two only describe the same body frame while the asset's pelvis carries no rotational
    re-origin; a model swap that adds one would silently mix frames in the policy's history.
    """
    from genesis.utils.geom import xyz_to_quat

    robot = env.entities["robot"]
    tilted = torch.tensor(xyz_to_quat(np.array([0.3, -0.2, 0.7]), rpy=True), dtype=torch.float32).reshape(1, 4)
    robot.set_quat(tilted.to(env.device), relative=True)
    try:
        torch.testing.assert_close(robot.get_quat(), robot.get_quat(relative=True), atol=1e-6, rtol=0.0)
    finally:
        env.reset()


@pytest.fixture
def local_policy_cls(local_sonic_policy):
    return local_sonic_policy(SonicTeleopPolicy)


def test_reset_without_the_teleop_runtime_fails_at_setup(env, local_policy_cls):
    """``eden deploy/inference --policy`` cannot feed it a reference; that used to surface on the first tick."""
    policy = local_policy_cls(env=env, options=local_policy_cls.configure())
    with pytest.raises(RuntimeError, match="eden teleop sonic robot"):
        policy.reset()


# ---------------------------------------------------------------------------
# A real teleop step on tiny graphs
# ---------------------------------------------------------------------------

_DEFAULT_MJ = torch.tensor(DEFAULT_ANGLES_MJ, dtype=torch.float32)
_SCALE_MJ = torch.tensor(ACTION_SCALE_MJ, dtype=torch.float32)
_LOWER = np.asarray(LOWER_BODY_IL_INDICES)
_UPPER = np.asarray(UPPER_BODY_IL_INDICES)
_WRIST = np.asarray(WRIST_IL_INDICES)


def _robot_state(env, pelvis_quat=IDENTITY, *, q_offset=0.0, base_ang=(0.0, 0.0, 0.0)) -> RobotStateSnapshot:
    """A runtime-style per-tick snapshot: the stand plus ``q_offset``, at rest, pelvis at ``pelvis_quat``."""
    device = env.device
    return RobotStateSnapshot(
        q_mj=(_DEFAULT_MJ + q_offset).reshape(1, -1).to(device),
        qdot_mj=torch.zeros(1, len(DEFAULT_ANGLES_MJ), device=device),
        base_ang_local=torch.tensor([base_ang], dtype=torch.float32, device=device),
        pelvis_quat_wxyz=torch.tensor(np.asarray(pelvis_quat), dtype=torch.float32, device=device).reshape(1, 4),
    )


class _TeleopSteps:
    """Build real ``SonicTeleopPolicy`` instances on tiny graphs and run their ``step``.

    The pass-through decoder returns its first 29 input columns, the encoder's token columns
    0..28, as the IsaacLab-order action; ``act`` undoes the public action transform
    (``default + scale * action``, MJCF order) to hand that action back.
    """

    def __init__(self, env, make_policy_cls, write_onnx, tmp_path):
        self._env = env
        self._make_policy_cls = make_policy_cls
        self._write_onnx = write_onnx
        self._tmp_path = tmp_path
        self._ids = itertools.count()

    def policy(self, scheduler, *, encoder=None, decoder_columns=None) -> SonicTeleopPolicy:
        """``encoder``: ``(nodes, initializers)`` producing ``encoded_tokens``; ``decoder_columns``: 29 to gather."""
        decoder = {}
        if decoder_columns is not None:
            decoder = {
                "decoder_nodes": [("Gather", ["obs_dict", "idx"], ["action"], {"axis": 1})],
                "decoder_initializers": {"idx": np.asarray(decoder_columns, dtype=np.int64)},
            }
        cls = self._make_policy_cls(SonicTeleopPolicy, **decoder)
        if encoder is not None:
            nodes, initializers = encoder
            path = self._write_onnx(
                self._tmp_path / f"probe_encoder_{next(self._ids)}.onnx",
                nodes,
                [("obs_dict", [1, _ENCODER_INPUT_DIM])],
                [("encoded_tokens", [1, 64])],
                initializers,
            )
            cls.use_local_checkpoint(path)
        policy = cls(env=self._env, options=cls.configure())
        policy.attach_runtime(scheduler)
        policy.reset()
        return policy

    def act(self, policy, robot_state=None) -> np.ndarray:
        state = _robot_state(self._env) if robot_state is None else robot_state
        targets = policy.step(obs=None, robot_state=state)["dofs_pos_controller"][0].cpu()
        return ((targets - _DEFAULT_MJ) / _SCALE_MJ)[list(MJ_TO_IL)].double().numpy()

    def encoder_columns(self, scheduler, columns, robot_state=None) -> np.ndarray:
        """The encoder-input values at ``columns``, 29 per step (one Gather encoder per chunk)."""
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        values = []
        for start in range(0, len(columns), 29):
            chunk = columns[start : start + 29]
            idx = np.zeros(64, dtype=np.int64)
            idx[: len(chunk)] = chunk
            gather = ([("Gather", ["obs_dict", "idx"], ["encoded_tokens"], {"axis": 1})], {"idx": idx})
            values.append(self.act(self.policy(scheduler, encoder=gather), robot_state)[: len(chunk)])
        return np.concatenate(values)

    def slot(self, scheduler, name, shape=None, index=(), robot_state=None) -> np.ndarray:
        """Slot ``name`` of the encoder input, reshaped to ``shape`` and indexed by ``index``."""
        start, end = _ENCODER_OFFSETS[name]
        flat = np.arange(start, end).reshape(shape or (-1,))[index]
        return self.encoder_columns(scheduler, flat, robot_state).reshape(np.shape(flat))

    def filled_slots(self, scheduler, robot_state=None) -> set[str]:
        """Names of the encoder slots holding any nonzero value (an Abs + MatMul graph sums ``|x|`` per slot)."""
        names = list(_ENCODER_OFFSETS)
        weights = np.zeros((_ENCODER_INPUT_DIM, 64), dtype=np.float32)
        for k, (start, end) in enumerate(_ENCODER_OFFSETS.values()):
            weights[start:end, k] = 1.0
        norm = ([("Abs", ["obs_dict"], ["abs"], {}), ("MatMul", ["abs", "w"], ["encoded_tokens"], {})], {"w": weights})
        sums = self.act(self.policy(scheduler, encoder=norm), robot_state)
        return {name for name, total in zip(names, sums) if total > 0}


@pytest.fixture
def steps(env, local_sonic_policy, write_onnx, tmp_path):
    return _TeleopSteps(env, local_sonic_policy, write_onnx, tmp_path)


def _motion(timesteps=60, *, with_smpl=True, joint_width=29, encode_mode=MODE_G1, name="clip"):
    """Motion whose joint_pos[t, il] = t + il/100 and joint_vel = 0.1 * that (frame and joint identifiable)."""
    t = np.arange(timesteps, dtype=np.float64)[:, None]
    joint_pos = t + np.arange(joint_width) / 100.0
    motion = MotionFrames(
        name=name,
        joint_pos=joint_pos,
        joint_vel=joint_pos * 0.1,
        body_quat=np.tile(IDENTITY, (timesteps, 1, 1)),
        smpl_joints=(t[:, :, None] + np.arange(24)[None, :, None] / 100.0) * np.ones(3) if with_smpl else None,
        smpl_pose=np.zeros((timesteps, 21, 3)) if with_smpl else None,
    )
    motion.encode_mode = encode_mode
    return motion


def _scheduler(motion, frame=0, play=True) -> MotionScheduler:
    scheduler = MotionScheduler()
    scheduler.set_motion(motion, frame=frame, play=play)
    return scheduler


def _yaw_6d(theta):
    return np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta), 0.0, 0.0])


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# State history: the runtime's snapshot, and the pre-engage hold
# ---------------------------------------------------------------------------


def _decoder_columns(name, width, frames):
    """Decoder-input columns of history slot ``name`` (``width`` per frame, newest last) at ``frames``."""
    start, _ = _DECODER_OFFSETS[name]
    return (start + width * np.asarray(frames)[:, None] + np.arange(width)).reshape(-1)


def test_step_reads_the_runtimes_snapshot_not_the_entity(env, steps):
    """The reference serves every observation from one logged ``LowState`` per tick, so the step must not re-read.

    The snapshot is far from the built robot's real (standing, upright, at rest) state: every
    joint 0.25 off the stand, spinning, pelvis rolled 90 deg about x.
    """
    columns = np.concatenate([
        _decoder_columns("his_body_joint_positions_10frame_step1", 29, [9])[:23],
        _decoder_columns("his_base_angular_velocity_10frame_step1", 3, [9]),
        _decoder_columns("his_gravity_dir_10frame_step1", 3, [9]),
    ])  # fmt: skip
    policy = steps.policy(_scheduler(_motion()), decoder_columns=columns)
    rolled = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    fed = steps.act(policy, _robot_state(env, rolled, q_offset=0.25, base_ang=(0.1, 0.2, 0.3)))
    _assert_close(fed[:23], 0.25)  # q - default, as the decoder history stores it
    _assert_close(fed[23:26], [0.1, 0.2, 0.3])
    _assert_close(fed[26:], [0.0, -1.0, 0.0])  # world down, seen from a pelvis rolled +90 deg about x


def test_record_state_rolls_the_history_during_the_pre_engage_hold(env, steps):
    """Only ``step`` rolled it, so the first engaged tick saw nine zero frames (zero gravity included)."""
    every_frame = np.arange(10)
    gravity_z = _decoder_columns("his_gravity_dir_10frame_step1", 3, every_frame)[2::3]
    first_action = _decoder_columns("his_last_actions_10frame_step1", 29, every_frame)[::29]
    columns = np.concatenate([gravity_z, first_action, np.full(9, gravity_z[0])])
    policy = steps.policy(_scheduler(_motion()), decoder_columns=columns)
    hold = _robot_state(env)  # upright pelvis
    for _ in range(9):
        policy.record_state(hold)
    fed = steps.act(policy, hold)  # the first engaged tick rolls the tenth frame itself
    _assert_close(fed[:10], -1.0)  # every history frame carries gravity
    _assert_close(fed[10:20], 0.0)  # no decoder output exists for a hold tick


# ---------------------------------------------------------------------------
# Encoder modes: which slots each fills, and the fallback walk
# ---------------------------------------------------------------------------

_G1_SLOTS = {
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_anchor_orientation_10frame_step5",
}  # encoder_mode_4 is filled too, with g1's id 0
_TELEOP_SLOTS = {
    "encoder_mode_4",
    "motion_joint_positions_lowerbody_10frame_step5",
    "motion_joint_velocities_lowerbody_10frame_step5",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
    "motion_anchor_orientation",
}
_SMPL_SLOTS = {
    "encoder_mode_4",
    "smpl_joints_10frame_step1",
    "smpl_anchor_orientation_10frame_step1",
    "motion_joint_positions_wrists_10frame_step1",
}


@pytest.mark.parametrize(
    ("motion_kwargs", "accepted", "filled"),
    [
        ({"encode_mode": MODE_G1}, MODE_G1, _G1_SLOTS),
        ({"encode_mode": MODE_TELEOP}, MODE_TELEOP, _TELEOP_SLOTS),
        ({"encode_mode": MODE_SMPL}, MODE_SMPL, _SMPL_SLOTS),
        # An id outside the configured modes (the all-active id 3) requires every slot, like the
        # reference's use_mode_filter=false branch, rather than raising a lookup error.
        ({"encode_mode": 3}, 3, _G1_SLOTS | _TELEOP_SLOTS | _SMPL_SLOTS),
        # Missing data falls back to the first workable mode in config order, never a crash.
        ({"encode_mode": MODE_SMPL, "with_smpl": False}, MODE_G1, _G1_SLOTS),
        ({"encode_mode": 3, "with_smpl": False}, MODE_G1, _G1_SLOTS),
        # A narrow stream fails every subset gather past its width: the wrist slot needs index 28.
        ({"encode_mode": MODE_SMPL, "joint_width": 20}, MODE_G1, _G1_SLOTS),
    ],
    ids=["g1", "teleop", "smpl", "all-active", "smpl-without-smpl", "all-active-without-smpl", "narrow-smpl"],
)
def test_encoder_mode_fills_exactly_its_slots(steps, motion_kwargs, accepted, filled):
    motion = _motion(**motion_kwargs)
    scheduler = _scheduler(motion, frame=2)
    assert steps.filled_slots(scheduler) == filled  # every other slot is zero
    # The scalar mode id + three zeros, not a one-hot.
    _assert_close(steps.slot(scheduler, "encoder_mode_4"), [accepted, 0.0, 0.0, 0.0])
    assert motion.encode_mode == accepted  # the fallback mutates the motion, like the reference


@pytest.mark.parametrize("joint_width", [None, 0], ids=["no-motion", "jointless-smpl-stream"])
def test_no_workable_encoder_mode_stops_control(steps, joint_width):
    """A v2 stream (SMPL, no joints) satisfies no mode: g1/teleop need joints, smpl's wrist slot reads them too."""
    motion = None if joint_width is None else _motion(joint_width=joint_width, encode_mode=MODE_SMPL)
    policy = steps.policy(_scheduler(motion))
    with pytest.raises(RuntimeError, match="no encoder mode"):
        steps.act(policy)


# ---------------------------------------------------------------------------
# Slot contents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "name", "shape", "index", "expected"),
    [
        # g1: 10 future frames, step 5, raw joint values in IsaacLab order.
        (MODE_G1, "motion_joint_positions_10frame_step5", (10, 29), 0, 3 + np.arange(29) / 100.0),
        (MODE_G1, "motion_joint_velocities_10frame_step5", (10, 29), (slice(None), 0), 0.1 * np.arange(3, 53, 5)),
        # teleop: the lower-body columns, in LOWER_BODY_IL_INDICES order.
        (MODE_TELEOP, "motion_joint_positions_lowerbody_10frame_step5", (10, 12), 0, 3 + _LOWER / 100.0),
        (MODE_TELEOP, "motion_joint_velocities_lowerbody_10frame_step5", (10, 12), 1, 0.1 * (8 + _LOWER / 100.0)),
        # smpl: 10 consecutive frames (step 1) of SMPL joints and of the IL wrist columns.
        (MODE_SMPL, "smpl_joints_10frame_step1", (10, 24, 3), (slice(None), 0, 0), np.arange(3, 13)),
        (MODE_SMPL, "smpl_joints_10frame_step1", (10, 24, 3), (3, slice(None), 0), 6 + np.arange(24) / 100.0),
        (MODE_SMPL, "motion_joint_positions_wrists_10frame_step1", (10, 6), 1, 4 + _WRIST / 100.0),
    ],
    ids=[
        "g1-positions",
        "g1-velocities",
        "teleop-positions",
        "teleop-velocities",
        "smpl-frames",
        "smpl-joints",
        "smpl-wrists",
    ],
)
def test_slot_gathers_the_motions_future_frames(steps, mode, name, shape, index, expected):
    scheduler = _scheduler(_motion(encode_mode=mode), frame=3)
    _assert_close(steps.slot(scheduler, name, shape, index), expected)


@pytest.mark.parametrize(
    ("timesteps", "frame", "play", "expected"),
    [
        (60, 3, True, np.arange(3, 53, 5)),
        (20, 10, True, [10, 15, 19, 19, 19, 19, 19, 19, 19, 19]),  # clamped at the motion end
        (60, 7, False, np.full(10, 7)),  # paused playback holds the current frame
    ],
    ids=["playing", "clamped-at-end", "paused"],
)
def test_future_frame_indices(steps, timesteps, frame, play, expected):
    scheduler = _scheduler(_motion(timesteps), frame=frame, play=play)
    frames = steps.slot(scheduler, "motion_joint_positions_10frame_step5", (10, 29), (slice(None), 0))
    _assert_close(frames, expected)


def test_paused_playback_zeroes_the_velocity_slot(steps):
    filled = steps.filled_slots(_scheduler(_motion(), frame=7, play=False))
    assert filled == _G1_SLOTS - {"motion_joint_velocities_10frame_step5"}


def test_narrow_joint_stream_fills_the_leading_columns(steps):
    """Fewer than 29 joints fill the leading slot columns (the reference's memcpy); the rest stay zero."""
    scheduler = _scheduler(_motion(joint_width=20, encode_mode=MODE_SMPL), frame=2)
    q = steps.slot(scheduler, "motion_joint_positions_10frame_step5", (10, 29), [0, 9])
    _assert_close(q[:, :20], np.array([[2.0], [47.0]]) + np.arange(20) / 100.0)
    np.testing.assert_array_equal(q[:, 20:], 0.0)


@pytest.mark.parametrize(
    "streamed_velocities", [None, np.linspace(-0.8, 0.8, 17)], ids=["position-only-stream", "streamed-velocities"]
)
def test_frozen_upper_overrides_the_upper_body_in_every_frame(steps, streamed_velocities):
    scheduler = _scheduler(_motion())
    targets = np.linspace(1000, 1016, 17)
    scheduler.frozen_upper_targets = targets
    scheduler.frozen_upper_velocities = streamed_velocities
    frames = np.array([[0.0], [45.0]])  # future frames 0 and 9 sit at motion frames 0 and 45
    q = steps.slot(scheduler, "motion_joint_positions_10frame_step5", (10, 29), [0, 9])
    qd = steps.slot(scheduler, "motion_joint_velocities_10frame_step5", (10, 29), [0, 9])
    upper_velocities = 0.0 if streamed_velocities is None else streamed_velocities  # T3 streams positions only
    _assert_close(q[:, _UPPER], np.broadcast_to(targets, (2, 17)))
    _assert_close(qd[:, _UPPER], np.broadcast_to(upper_velocities, (2, 17)))
    # The lower body keeps the motion's positions and velocities.
    _assert_close(q[:, _LOWER], frames + _LOWER / 100.0)
    _assert_close(qd[:, _LOWER], 0.1 * (frames + _LOWER / 100.0))


@pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "reference-defaults"])
def test_vr_slots_use_the_stream_or_the_reference_defaults(steps, streamed):
    scheduler = _scheduler(_motion(encode_mode=MODE_TELEOP))
    position, orientation = VR_3POINT_DEFAULT_POSITION, VR_3POINT_DEFAULT_ORIENTATION  # before any VR data arrives
    if streamed:
        position, orientation = np.arange(9.0) + 0.5, np.arange(12.0) - 0.5
        scheduler.vr_position, scheduler.vr_orientation = position, orientation
    _assert_close(steps.slot(scheduler, "vr_3point_local_target"), position)
    _assert_close(steps.slot(scheduler, "vr_3point_local_orn_target"), orientation)


@pytest.mark.parametrize(
    ("mode", "name", "num_frames", "step"),
    [
        (MODE_G1, "motion_anchor_orientation_10frame_step5", 10, 5),
        (MODE_TELEOP, "motion_anchor_orientation", 1, 1),
        (MODE_SMPL, "smpl_anchor_orientation_10frame_step1", 10, 1),
    ],
    ids=["g1", "teleop", "smpl"],
)
def test_anchor_diffs_the_heading_corrected_reference_against_the_base(env, steps, mode, name, num_frames, step):
    """6D (first two columns of R, row-major) of R(conj(base) * apply_delta_heading * ref_root), per future frame."""
    motion = _motion(encode_mode=mode)
    motion.body_quat[:, 0] = [yawq(0.3 + 0.01 * t) for t in range(motion.timesteps)]  # reference root yaw
    scheduler = _scheduler(motion)
    scheduler.begin_control_tick(IDENTITY)  # heading captured with the robot at yaw 0 and the reference at 0.3
    scheduler.add_heading_increment(0.5)  # so apply_delta_heading = yaw(0 - 0.3 + 0.5)
    scheduler.set_frame(4)
    rows = [0, num_frames - 1]
    anchors = steps.slot(scheduler, name, (num_frames, 6), rows, _robot_state(env, yawq(0.1)))
    # yaw: -0.1 (base) + 0.2 (correction) + 0.3 + 0.01 * frame (reference), at frame 4 + step * row.
    _assert_close(anchors, [_yaw_6d(0.4 + 0.01 * (4 + step * row)) for row in rows])


@pytest.mark.parametrize(("mode", "window"), [(MODE_G1, 46), (MODE_TELEOP, 46), (MODE_SMPL, 10)])
def test_streamed_playback_keeps_the_observation_window_inside_the_buffer(steps, mode, window):
    """A step's gathers note their future reach: (10 - 1) * 5 + 1 = 46 frames for step-5 slots, 10 for step-1."""
    timesteps = 60
    scheduler = _scheduler(_motion(timesteps, encode_mode=mode, name="streamed"))
    steps.slot(scheduler, "encoder_mode_4")  # one real step
    scheduler.set_frame(timesteps - window - 2)
    scheduler.advance_frame()
    assert scheduler.frame == timesteps - window - 1
    scheduler.advance_frame()  # one more frame would push the window past the buffer end
    assert scheduler.frame == timesteps - window - 1


# ---------------------------------------------------------------------------
# Env / action / command specs
# ---------------------------------------------------------------------------

_HIP_PITCH_KP_REF = 99.0984278
_HIP_PITCH_KD_REF = 6.3088019


def test_teleop_robot_options_reference_parity():
    robot = SonicTeleopPolicy.robot_options()
    for side in ("left", "right"):
        assert resolve_pattern(robot.default_dofs_kp, f"{side}_hip_pitch_joint") == pytest.approx(_HIP_PITCH_KP_REF)
        assert resolve_pattern(robot.default_dofs_kd, f"{side}_hip_pitch_joint") == pytest.approx(_HIP_PITCH_KD_REF)
        # Reference-sim torque clamps where tighter than the actuator spec:
        # hip roll 88 (motor_effort_limit_list), non-thumb-base fingers 0.7.
        assert resolve_pattern(robot.default_dofs_force_limits, f"{side}_hip_roll_joint") == 88.0
        assert resolve_pattern(robot.default_dofs_force_limits, f"{side}_hand_thumb_0_joint") == 2.45
        for name in (f"{side}_hand_thumb_1_joint", f"{side}_hand_middle_0_joint", f"{side}_hand_index_1_joint"):
            assert resolve_pattern(robot.default_dofs_force_limits, name) == 0.7
    # Armature is the MJCF's uniform 0.01 (the reference sim's plant), not the
    # actuator-spec reflected inertias; the built values are pinned in
    # test_built_armature_is_the_reference_mjcfs.
    assert robot.default_dofs_armature == {"*": 0.01}
    # Default pose == SONIC standing pose, so the pre-engage hold is a no-op.
    for name, angle in zip(G1_DOFS29_NAME, DEFAULT_ANGLES_MJ):
        assert resolve_pattern(robot.default_dofs_pos, name) == pytest.approx(angle)
    for name in G1_DEX3_LEFT_DOFS_NAME + G1_DEX3_RIGHT_DOFS_NAME:
        assert resolve_pattern(robot.default_dofs_pos, name) == 0.0
    # The merged pattern dicts must stay disjoint per DOF.
    for name in robot.dofs_name:
        for patterns in (robot.default_dofs_pos, robot.default_dofs_kp, robot.default_dofs_kd,
                         robot.default_dofs_armature, robot.default_dofs_force_limits):  # fmt: skip
            resolve_pattern(patterns, name)


def test_contact_parity_settings_are_teleop_only():
    """Multi-contact + MuJoCo's default timeconst are the teleop plant's; the base policy keeps its own."""
    from eden.extensions.policies.sonic.policy import SonicPolicy

    teleop, base = SonicTeleopPolicy.env_options(), SonicPolicy.env_options()
    assert teleop["enable_multi_contact"] is True and teleop["constraint_timeconst"] == 0.02
    assert base["enable_multi_contact"] is False and "constraint_timeconst" not in base
    # Everything else is shared: the 50 Hz cadence the checkpoint was exported at.
    assert {k: v for k, v in teleop.items() if k not in ("enable_multi_contact", "constraint_timeconst")} == {
        k: v for k, v in base.items() if k != "enable_multi_contact"
    }


def test_upper_and_wrist_index_tables_match_reference():
    # policy_parameters.hpp: upper_body_joint_isaaclab_order_in_isaaclab_index,
    # wrist_joint_isaaclab_order_in_isaaclab_index.
    assert UPPER_BODY_IL_INDICES == (2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
    assert WRIST_IL_INDICES == (23, 24, 25, 26, 27, 28)
    # lower_body_joint_mujoco_order_in_isaaclab_index (the lowerbody slot's column order).
    assert LOWER_BODY_IL_INDICES == (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18)


def test_action_options_split_body_and_hands():
    opts = SonicTeleopPolicy.action_options()
    assert set(opts) == {"dofs_pos_controller", "hands_pos_controller"}
    assert len(opts["dofs_pos_controller"].dofs_name) == 29
    assert len(opts["hands_pos_controller"].dofs_name) == 14
    assert all("hand" in n for n in opts["hands_pos_controller"].dofs_name)


def test_command_options_empty():
    assert SonicTeleopPolicy.command_options() == {}
