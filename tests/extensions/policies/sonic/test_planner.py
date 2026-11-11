"""``eden.extensions.policies.sonic.planner``: the locomotion planner thread, its resampling, blending and context."""

from __future__ import annotations

import time

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.extensions.policies.sonic.joint_maps import (
    IL_TO_MJ,
    LOWER_BODY_IL_INDICES,
    MJ_TO_IL,
)
from eden.extensions.policies.sonic.motion import (
    MotionFrames,
    MotionScheduler,
)
from eden.extensions.policies.sonic.planner import (
    LocomotionMode,
    MovementState,
    PlannerConfig,
    PlannerOnnx,
    SonicPlanner,
    blend_planner_motion,
    get_motion_set,
    initial_context,
    is_squat_mode,
    is_static_mode,
    replan_interval_for,
    resample_30hz_to_50hz,
    sample_context_from_motion,
    should_replan,
    valid_mode_count,
)
from tests.extensions.policies.sonic.conftest import (
    PLANNER_INPUTS,
    linear_qpos30,
    make_motion,
    planner_command,
    yawq,
)


def _yaw_of(quat_wxyz: np.ndarray) -> float:
    forward = gu.transform_by_quat(np.array([1.0, 0.0, 0.0]), np.asarray(quat_wxyz, dtype=np.float64))
    return float(np.arctan2(forward[1], forward[0]))


def _linear_motion(timesteps=100):
    """50 Hz motion whose x-position equals the frame index and joints the IsaacLab index."""
    motion = MotionFrames(
        name="planner_motion",
        joint_pos=np.tile(np.arange(29.0) * 0.01, (timesteps, 1)),
        joint_vel=np.zeros((timesteps, 29)),
        body_quat=np.tile(np.array([1.0, 0, 0, 0]), (timesteps, 1, 1)),
        body_pos=np.zeros((timesteps, 1, 3)),
    )
    motion.body_pos[:, 0, 0] = np.arange(timesteps)
    return motion


def _const_motion(value, timesteps, yaw=0.0):
    motion = MotionFrames(
        name="planner_motion",
        joint_pos=np.full((timesteps, 29), float(value)),
        joint_vel=np.full((timesteps, 29), float(value)),
        body_quat=np.tile(yawq(yaw), (timesteps, 1, 1)),
        body_pos=np.full((timesteps, 1, 3), float(value)),
    )
    return motion


def _ms(mode=LocomotionMode.WALK, movement=(1, 0, 0), facing=(1, 0, 0), speed=1.0, height=-1.0):
    return MovementState(
        locomotion_mode=int(mode),
        movement_direction=np.array(movement, dtype=float),
        facing_direction=np.array(facing, dtype=float),
        movement_speed=speed,
        height=height,
    )


def _make_planner(onnx, config=None):
    """A SonicPlanner on the ``planner_onnx`` graph; ``state["movement"]`` is the operator's command."""
    scheduler = MotionScheduler()
    state = {"movement": None}
    planner = SonicPlanner(
        onnx=onnx,
        scheduler=scheduler,
        robot_state_fn=lambda: (np.array([1.0, 0, 0, 0]), np.zeros(29)),
        movement_state_fn=lambda: state["movement"],
        config=config or PlannerConfig(),
    )
    return planner, scheduler, state


def test_unsupported_model_version_fails_at_construction(planner_onnx):
    """It raised from _generate on every replan, inside the planner thread."""
    with pytest.raises(ValueError, match=r"unsupported planner model version 7; supported: \[0, 1, 2\]"):
        SonicPlanner(
            onnx=planner_onnx,
            scheduler=MotionScheduler(),
            robot_state_fn=lambda: None,
            movement_state_fn=lambda: None,
            config=PlannerConfig(version=7),
        )


def test_a_planner_thread_failure_is_raised_on_the_control_thread(planner_onnx):
    """A failed planner disabled itself and left the robot holding its stance, with one print as the only sign."""
    # The graph echoes the context's root heights into its output, so a NaN default height comes
    # back as a NaN trajectory, which the resampler rejects on the planner thread.
    planner, scheduler, _ = _make_planner(planner_onnx, config=PlannerConfig(default_height=float("nan")))
    streamed = make_motion("streamed", 10)
    scheduler.set_motion(streamed, play=True)
    planner.state.enabled = True
    planner.config.planner_dt = 0.001
    planner.start()
    deadline = time.monotonic() + 5.0
    while planner.error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    planner.stop()
    assert planner.state.enabled is False
    assert scheduler.motion is streamed  # the planner thread no longer touches the scheduler
    with pytest.raises(RuntimeError, match="planner thread failed") as excinfo:
        planner.raise_if_failed()
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "planner qpos output contains NaNs" in str(excinfo.value)


def test_mode_values_match_reference_enum():
    assert LocomotionMode.IDLE == 0
    assert LocomotionMode.RUN == 3
    assert LocomotionMode.CRAWLING == 8
    assert LocomotionMode.INJURED_WALK == 19
    assert LocomotionMode.SCARE_WALK == 26


def test_mode_predicates():
    assert is_static_mode(LocomotionMode.IDLE)
    assert is_static_mode(LocomotionMode.IDLE_BOXING)
    assert not is_static_mode(LocomotionMode.WALK_BOXING)
    assert not is_static_mode(LocomotionMode.CRAWLING)  # crawling moves
    assert is_squat_mode(LocomotionMode.CRAWLING)
    assert not is_squat_mode(LocomotionMode.RUN)


def test_motion_sets():
    assert get_motion_set(0)[:3] == [LocomotionMode.SLOW_WALK, LocomotionMode.WALK, LocomotionMode.RUN]
    assert LocomotionMode.IDLE_LYING_FACE_DOWN not in get_motion_set(1)  # excluded like the reference
    assert len(get_motion_set(2)) == 7
    assert len(get_motion_set(3)) == 7
    assert get_motion_set(99) == get_motion_set(0)  # invalid index falls back to standing


def test_valid_mode_count():
    assert valid_mode_count(0) == 4
    assert valid_mode_count(1) == 20
    assert valid_mode_count(2) == 27
    with pytest.raises(ValueError):
        valid_mode_count(3)


def test_resample_lengths_and_linear_interpolation():
    motion = resample_30hz_to_50hz(linear_qpos30(n=30, vx=0.1))
    assert motion.timesteps == 50  # floor(30/30*50)
    # x is linear in 30 Hz frame index: x(f50) = 0.1 * (f50 * 30/50).
    for f in (0, 7, 23, 45):
        assert motion.body_pos[f, 0, 0] == pytest.approx(0.1 * f * 30 / 50, abs=1e-9)
    assert motion.body_pos[0, 0, 2] == pytest.approx(0.78)


def test_resample_reorders_joints_to_isaaclab():
    motion = resample_30hz_to_50hz(linear_qpos30())
    # Source column 7+mj holds 0.01*mj, so IsaacLab slot il must hold 0.01*MJ_TO_IL[il].
    np.testing.assert_allclose(motion.joint_pos[0], 0.01 * np.asarray(MJ_TO_IL), atol=1e-12)


def test_resample_slerps_root_quaternion():
    motion = resample_30hz_to_50hz(linear_qpos30(dyaw=0.02))
    for f in (1, 25, 40):
        expected_yaw = (f * 30 / 50) * 0.02
        # 1e-6 abs: nearby quaternions take the reference's nlerp fast path,
        # which deviates from exact angular linearity by O(angle^3).
        assert _yaw_of(motion.body_quat[f, 0]) == pytest.approx(expected_yaw, abs=1e-6)


def test_resample_finite_difference_velocities():
    qpos = linear_qpos30()
    qpos[:, 7] = np.arange(30) * 0.03  # MuJoCo joint 0 moves linearly
    motion = resample_30hz_to_50hz(qpos)
    il_slot = MJ_TO_IL.index(0)
    # dq = d(pos)/dt with pos = 0.03 * f30 = 0.03 * t * 30 -> 0.9 rad/s.
    np.testing.assert_allclose(motion.joint_vel[:-2, il_slot], 0.9, atol=1e-6)
    assert motion.joint_vel[-1, il_slot] == motion.joint_vel[-2, il_slot]  # last frame repeats


def test_resample_rejects_bad_output():
    with pytest.raises(ValueError, match="> 64"):
        resample_30hz_to_50hz(np.zeros((65, 36)))
    bad = linear_qpos30()
    bad[3, 5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        resample_30hz_to_50hz(bad)


def test_context_samples_four_frames_at_30hz():
    context = sample_context_from_motion(_linear_motion(), gen_frame=12)
    assert context.shape == (4, 36)
    # Frame n samples 50 Hz index 12 + n*50/30 (linear x == index).
    for n in range(4):
        assert context[n, 0] == pytest.approx(12 + n * 50.0 / 30.0, abs=1e-9)


def test_context_clamps_at_motion_end():
    context = sample_context_from_motion(_linear_motion(timesteps=15), gen_frame=13)
    assert context[0, 0] == pytest.approx(13.0)
    assert context[3, 0] == pytest.approx(14.0)  # clamped to last frame


def test_context_reorders_joints_to_mujoco():
    context = sample_context_from_motion(_linear_motion(), gen_frame=0)
    # IsaacLab slot il holds 0.01*il; MuJoCo column mj must hold 0.01*IL_TO_MJ[mj].
    np.testing.assert_allclose(context[0, 7:], 0.01 * np.asarray(IL_TO_MJ), atol=1e-12)


def test_context_requires_frames():
    with pytest.raises(ValueError, match="no frames"):
        sample_context_from_motion(_linear_motion(timesteps=0), gen_frame=0)


def test_initial_context():
    q = np.linspace(-0.5, 0.5, 29)
    context = initial_context(q, default_height=0.7887)
    assert context.shape == (4, 36)
    np.testing.assert_allclose(context[:, :3], [[0, 0, 0.7887]] * 4)
    np.testing.assert_allclose(context[:, 3:7], [[1, 0, 0, 0]] * 4)
    np.testing.assert_allclose(context[:, 7:], np.tile(q, (4, 1)))


def test_blend_fades_linearly_over_eight_frames():
    old = _const_motion(0.0, 20)
    new = _const_motion(1.0, 36)
    blended = blend_planner_motion(old, new, current_frame=0, gen_frame=10)
    assert blended.timesteps == 10 + 36
    np.testing.assert_allclose(blended.joint_pos[:10], 0.0)  # before the generation starts
    for f in range(10, 18):
        expected = (f - 10) / 8.0
        np.testing.assert_allclose(blended.joint_pos[f], expected, atol=1e-12)
        np.testing.assert_allclose(blended.body_pos[f, 0], expected, atol=1e-12)
    np.testing.assert_allclose(blended.joint_pos[18:], 1.0)


def test_blend_starts_immediately_when_generation_is_late():
    old = _const_motion(0.0, 30)
    new = _const_motion(1.0, 36)
    # Playback already passed gen_frame: fade must start at output frame 0.
    blended = blend_planner_motion(old, new, current_frame=8, gen_frame=4)
    assert blended.timesteps == 4 - 8 + 36
    np.testing.assert_allclose(blended.joint_pos[0], 0.0)
    np.testing.assert_allclose(blended.joint_pos[4], 0.5, atol=1e-12)
    np.testing.assert_allclose(blended.joint_pos[8:], 1.0)


def test_blend_slerps_root_quaternion():
    old = _const_motion(0.0, 20, yaw=0.0)
    new = _const_motion(1.0, 36, yaw=1.0)
    blended = blend_planner_motion(old, new, current_frame=0, gen_frame=10)
    assert _yaw_of(blended.body_quat[14, 0]) == pytest.approx(0.5, abs=1e-9)


def test_blend_preserves_encode_mode_and_rejects_late_arrival():
    old = _const_motion(0.0, 20)
    old.encode_mode = 1
    blended = blend_planner_motion(old, _const_motion(1.0, 36), current_frame=0, gen_frame=10)
    assert blended.encode_mode == 1
    with pytest.raises(ValueError, match="too late"):
        blend_planner_motion(old, _const_motion(1.0, 2), current_frame=10, gen_frame=2)


def test_should_replan_truth_table():
    base = _ms()
    assert not should_replan(base, _ms(), timer_expired=False)
    # Mode / facing / height changes always replan, static modes included.
    assert should_replan(base, _ms(mode=LocomotionMode.RUN), timer_expired=False)
    assert should_replan(_ms(mode=LocomotionMode.IDLE), _ms(mode=LocomotionMode.IDLE, facing=(0, 1, 0)), False)
    assert should_replan(base, _ms(height=0.5), timer_expired=False)
    # Speed / direction changes replan only outside static modes.
    assert should_replan(base, _ms(speed=2.0), timer_expired=False)
    assert should_replan(base, _ms(movement=(0, 1, 0)), timer_expired=False)
    idle = _ms(mode=LocomotionMode.IDLE)
    assert not should_replan(idle, _ms(mode=LocomotionMode.IDLE, speed=2.0), timer_expired=False)
    # Timer expiry replans while moving, never when the stick is centered.
    assert should_replan(base, _ms(), timer_expired=True)
    assert not should_replan(_ms(speed=0.0), _ms(speed=0.0), timer_expired=True)
    assert not should_replan(idle, idle, timer_expired=True)


def test_replan_intervals_per_mode_family():
    config = PlannerConfig()
    assert replan_interval_for(int(LocomotionMode.RUN), config) == 0.1
    assert replan_interval_for(int(LocomotionMode.CRAWLING), config) == 0.2
    assert replan_interval_for(int(LocomotionMode.LEFT_HOOK), config) == 1.0
    assert replan_interval_for(int(LocomotionMode.WALK_BOXING), config) == 1.0  # not in the punch family
    assert replan_interval_for(int(LocomotionMode.WALK), config) == 1.0


def test_planner_onnx_feeds_the_declared_signature_and_keeps_num_pred_frames_rows(tmp_path, write_planner_onnx):
    """``generate`` feeds all 11 inputs with the dtypes and static shapes of ``PLANNER_INPUTS``.

    onnxruntime rejects a missing input, a wrong dtype or a wrong shape, so the run itself pins the
    signature. The graph echoes every input (flattened, in declaration order) as its qpos rows, plus
    two junk rows past ``num_pred_frames`` that ``generate`` must drop.
    """
    sizes = [int(np.prod(shape)) for _, _, shape in PLANNER_INPUTS]
    rows = -(-sum(sizes) // 36)
    nodes, flat = [], []
    for name, dtype, _ in PLANNER_INPUTS:
        src = name
        if dtype == "INT64":
            nodes.append(("Cast", [name], [f"{name}_f"], {"to": 1}))  # 1 = FLOAT
            src = f"{name}_f"
        nodes.append(("Reshape", [src, "flat"], [f"{name}_flat"], {}))
        flat.append(f"{name}_flat")
    nodes += [
        ("Concat", flat, ["echo"], {"axis": 1}),
        ("Pad", ["echo", "pads"], ["echo_padded"], {}),
        ("Reshape", ["echo_padded", "frames"], ["echo_rows"], {}),
        ("Concat", ["echo_rows", "junk"], ["mujoco_qpos"], {"axis": 1}),
        ("Identity", ["num_frames"], ["num_pred_frames"], {}),
    ]
    path = write_planner_onnx(
        tmp_path / "echo_planner.onnx",
        nodes,
        {
            "flat": np.array([1, -1], np.int64),
            "pads": np.array([0, 0, 0, rows * 36 - sum(sizes)], np.int64),
            "frames": np.array([1, rows, 36], np.int64),
            "junk": np.full((1, 2, 36), -7.0, np.float32),
            "num_frames": np.array([rows], np.int64),
        },
    )
    context = np.arange(4 * 36, dtype=np.float64).reshape(4, 36) / 100.0

    qpos = PlannerOnnx(path, device="cpu").generate(
        context_qpos=context,
        mode=2,
        target_vel=1.5,
        target_height=-1.0,
        movement_direction=np.array([1.0, 0, 0]),
        facing_direction=np.array([0.0, 1, 0]),
        random_seed=1234,
    )

    assert qpos.shape == (rows, 36)  # the junk rows past num_pred_frames are dropped
    names = [name for name, _, _ in PLANNER_INPUTS]
    fed = dict(zip(names, np.split(qpos.reshape(-1)[: sum(sizes)], np.cumsum(sizes)[:-1]), strict=True))
    expected = {
        "context_mujoco_qpos": context.reshape(-1),
        "mode": [2],
        "target_vel": [1.5],
        "height": [-1.0],
        "movement_direction": [1, 0, 0],
        "facing_direction": [0, 1, 0],
        "random_seed": [1234],
        "has_specific_target": [0],
        "specific_target_positions": np.zeros(12),
        "specific_target_headings": np.zeros(4),
        "allowed_pred_num_tokens": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # 9-11 tokens, like the reference
    }
    for name in names:
        np.testing.assert_allclose(fed[name], expected[name], rtol=1e-6, err_msg=name)


def test_planner_disabled_does_nothing(planner_onnx):
    planner, scheduler, _ = _make_planner(planner_onnx)
    planner.step()
    assert not planner.state.initialized
    planner.state.enabled = planner.state.initialized = True  # let advance() adopt whatever step() produced
    planner.advance(measured_q_mj=None)
    assert scheduler.motion is None  # nothing was generated


def test_planner_initializes_with_idle_command(planner_onnx):
    planner, scheduler, _ = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.step()
    assert planner.state.initialized

    # The generation is adopted by the control thread, resetting the cursor.
    assert scheduler.motion is None
    planner.advance(measured_q_mj=None)
    assert scheduler.motion is planner.planner_motion
    assert planner.planner_motion.timesteps == 60  # floor(36/30*50)
    assert scheduler.frame == 0  # play is False, no advance yet

    # It was generated from an IDLE command at mode-default speed and height, standing at the
    # default height in the identity frame.
    fed = planner_command(planner.planner_motion)
    assert fed["mode"] == int(LocomotionMode.IDLE)
    assert fed["target_vel"] == pytest.approx(-1.0)
    assert fed["height"] == pytest.approx(-1.0)
    assert fed["random_seed"] == PlannerConfig().initial_random_seed
    np.testing.assert_allclose(fed["context_root_z"], 0.788740, rtol=1e-6)


def test_planner_advances_and_replans_on_command_change(planner_onnx):
    planner, scheduler, state = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.step()  # initialize
    planner.advance(measured_q_mj=None)  # adopt
    scheduler.play = True
    planner.advance(measured_q_mj=None)
    assert scheduler.frame == 1
    adopted = planner.planner_motion

    # An IDLE command identical to the initialization defaults: no replan, so the next tick
    # just advances playback on the same motion.
    state["movement"] = _ms(mode=LocomotionMode.IDLE, speed=-1.0, movement=(0, 0, 0))
    planner.step()
    planner.step()
    planner.advance(measured_q_mj=None)
    assert planner.planner_motion is adopted
    assert scheduler.frame == 2

    # A walk command replans; the fresh generation is blended in on advance().
    state["movement"] = _ms(mode=LocomotionMode.WALK, speed=1.0)
    planner.step()
    planner.advance(measured_q_mj=None)
    assert planner.planner_motion is not adopted
    assert scheduler.motion is planner.planner_motion
    assert scheduler.frame == 1  # rebased to 0 by the blend, then advanced
    fed = planner_command(planner.planner_motion)  # the last frame is all fresh generation
    assert fed["mode"] == int(LocomotionMode.WALK)
    assert fed["target_vel"] == pytest.approx(1.0)


def test_planner_invalid_mode_falls_back_to_idle(planner_onnx):
    planner, _scheduler, state = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.step()
    planner.advance(measured_q_mj=None)
    state["movement"] = _ms(mode=99, speed=1.0)
    planner.step()
    planner.advance(measured_q_mj=None)
    fed = planner_command(planner.planner_motion)
    assert fed["target_vel"] == pytest.approx(1.0)  # the replan, not the initial IDLE generation (-1)
    assert fed["mode"] == int(LocomotionMode.IDLE)


def test_idle_readapt_state_machine(planner_onnx):
    planner, scheduler, state = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    planner.planner_motion = _const_motion(0.0, 5)
    scheduler.set_motion(planner.planner_motion, frame=4, play=True, reinitialize_heading=False)
    state["movement"] = MovementState(locomotion_mode=int(LocomotionMode.IDLE))

    lower = list(LOWER_BODY_IL_INDICES)
    # Large error on the lower body triggers ADAPTING: targets creep toward measured.
    measured_il = np.zeros(29)
    measured_il[lower] = 0.5
    measured_mj = measured_il[np.asarray(IL_TO_MJ)]  # convert to hardware order for the call
    planner.advance(measured_q_mj=measured_mj)
    assert planner._readapt_state == "adapting"
    target = planner.planner_motion.joint_pos[4]
    np.testing.assert_allclose(target[lower], 0.02 * 0.5, atol=1e-12)

    # A small error first settles back to IDLE, then (still small) starts RECOVERING.
    measured_il[lower] = target[lower] + 0.01
    measured_mj = measured_il[np.asarray(IL_TO_MJ)]
    planner.advance(measured_q_mj=measured_mj)
    assert planner._readapt_state == "idle"
    np.testing.assert_allclose(planner.planner_motion.joint_pos[4][lower], 0.02 * 0.5, atol=1e-12)
    planner.advance(measured_q_mj=measured_mj)
    assert planner._readapt_state == "recovering"
    np.testing.assert_allclose(planner.planner_motion.joint_pos[4][lower], 0.98 * 0.02 * 0.5, atol=1e-12)


def test_late_generation_still_advances_playback(planner_onnx):
    # A generation that arrived too late to blend is discarded, but the tick
    # still advances the playback cursor (reference falls through on failure).
    planner, scheduler, _ = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    planner.planner_motion = _const_motion(0.0, 40)
    scheduler.set_motion(planner.planner_motion, frame=20, play=True, reinitialize_heading=False)
    with planner._motion_lock:
        planner._generated = _const_motion(1.0, 5)  # 20 - 30 + 5 < 0: too late
        planner._generated_at_frame = -30
        planner._motion_available = True
    planner.advance(measured_q_mj=None)
    assert scheduler.frame == 21  # no adoption, but playback moved on
    np.testing.assert_allclose(scheduler.motion.joint_pos, 0.0)  # old motion kept


def test_readapt_only_runs_at_the_clamped_last_frame(planner_onnx):
    planner, scheduler, state = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    planner.planner_motion = _const_motion(0.0, 10)
    scheduler.set_motion(planner.planner_motion, frame=2, play=True, reinitialize_heading=False)
    state["movement"] = MovementState(locomotion_mode=int(LocomotionMode.IDLE))
    measured = np.full(29, 0.5)
    planner.advance(measured_q_mj=measured)
    assert scheduler.frame == 3
    np.testing.assert_allclose(planner.planner_motion.joint_pos, 0.0)  # untouched mid-motion


def test_idle_readapt_skips_when_motion_replaced(planner_onnx):
    # If planner_motion is swapped to an empty motion (e.g. the planner error
    # handler resetting it) after the last-frame decision but before the
    # readapt, the readapt is a no-op rather than an IndexError. Both sides run
    # under _motion_lock, so this only happens if the object is replaced; the
    # bounds guard keeps it safe.
    planner, scheduler, state = _make_planner(planner_onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    planner.planner_motion = _const_motion(0.0, 5)
    scheduler.set_motion(planner.planner_motion, frame=4, play=True, reinitialize_heading=False)
    state["movement"] = MovementState(locomotion_mode=int(LocomotionMode.IDLE))
    planner.planner_motion = planner._empty_planner_motion()  # replaced; frame 4 now out of range
    planner._idle_readapt(4, np.full(29, 0.5))  # must not raise
    assert planner.planner_motion.timesteps == 0
