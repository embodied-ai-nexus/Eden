"""``eden.extensions.policies.sonic.zmq_runtime``: the robot's operator server, Dex3 hand controller, and the operator->robot loop over loopback."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from eden.extensions.policies.sonic import wire
from eden.extensions.policies.sonic.joint_maps import MODE_SMPL
from eden.extensions.policies.sonic.motion import MotionScheduler
from eden.extensions.policies.sonic.planner import (
    LocomotionMode,
    PlannerConfig,
    SonicPlanner,
)
from eden.extensions.policies.sonic.teleop_policy import VR_3POINT_DEFAULT_ORIENTATION
from eden.extensions.policies.sonic.wire import build_command_message
from eden.extensions.policies.sonic.zmq_runtime import Dex3HandController, ManagedMode, SonicZmqServer
from eden.extensions.teleop.sonic.feedback import FeedbackReader
from eden.extensions.teleop.sonic.planner_streamer import PlannerStreamer
from eden.extensions.teleop.sonic.pose_streamer import PoseStreamer
from eden.extensions.teleop.sonic.state_machine import (
    ButtonSnapshot,
    Effect,
    OperatorStateMachine,
    StreamMode,
)
from eden.extensions.teleop.sonic.three_point import ThreePointPose
from tests.extensions.policies.sonic.conftest import make_motion
from tests.support.sonic import (
    IDENTITY,
    bind_debug_publisher,
    make_body_sample,
    make_controllers,
    smpl_fk,
    unpaced,
)

# The operator<->robot loopback runs over the `sonic` extra (pyzmq + msgpack).
zmq = pytest.importorskip("zmq")
pytest.importorskip("msgpack")


def make_server(onnx, host="localhost", port=wire.DEFAULT_PACKED_PORT):
    """Robot-side SonicZmqServer + planner (on ``onnx``, the ``planner_onnx`` graph) + hands.

    No socket is bound unless the server is started. Most tests feed it with
    ``server._dispatch(msg)``, the receive thread's per-message entry: it is the only
    synchronous form of the receive path (the thread itself just ``recv``s and calls it), so
    a test can assert on the very next ``apply()`` without racing a PUB/SUB join, and a
    rejected payload raises where the test can see it instead of being logged and dropped.
    The socket path itself is covered by ``test_full_mode_flow_over_loopback``.
    """
    scheduler = MotionScheduler()
    planner = SonicPlanner(
        onnx=onnx,
        scheduler=scheduler,
        robot_state_fn=lambda: (IDENTITY, np.zeros(29)),
        movement_state_fn=lambda: server.movement_state,
        config=PlannerConfig(),
    )
    hands = Dex3HandController()
    server = SonicZmqServer(scheduler, planner, hands, host=host, port=port)
    return server, scheduler, planner, hands


@pytest.fixture
def loopback(planner_onnx, g1_key_frame_fk):
    # Both FKs resolve (or skip) before anything is started: a skip must not leak the server thread.
    body_fk = smpl_fk()
    # The operator PUB binds its own random port, so only g1_debug needs the probe.
    pub = zmq.Context.instance().socket(zmq.PUB)
    cmd_port = pub.bind_to_random_port("tcp://127.0.0.1")
    debug_pub, dbg_port = bind_debug_publisher()

    # Robot side.
    server, scheduler, planner, _hands = make_server(planner_onnx, host="127.0.0.1", port=cmd_port)
    server.start()

    # Operator side.
    three_point = ThreePointPose(g1_key_frame_fk)
    feedback = FeedbackReader(host="127.0.0.1", port=dbg_port)
    pose_streamer = unpaced(PoseStreamer(send=pub.send, three_point=three_point, fk=body_fk))
    planner_streamer = unpaced(PlannerStreamer(send=pub.send, three_point=three_point, feedback=feedback))
    sm = OperatorStateMachine()

    yield {
        "server": server, "scheduler": scheduler, "planner": planner, "pub": pub, "sm": sm,
        "pose_streamer": pose_streamer, "planner_streamer": planner_streamer, "three_point": three_point,
        "feedback": feedback, "debug_pub": debug_pub,
    }  # fmt: skip

    server.stop()
    planner.stop()
    pub.close()
    feedback.close()
    debug_pub.close()


def _wait_for(predicate, server, *, sender=None, timeout=5.0, message="condition"):
    """Pump ``server.apply()`` (after ``sender()``, if given) until ``predicate()`` holds (PUB/SUB joins are slow)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sender is not None:
            sender()
        server.apply()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {message}")


def _pose_chunk_msg(start, n=5, version=3, **extra):
    data = {
        "smpl_pose": np.zeros((n, 21, 3), dtype=np.float32),
        "smpl_joints": np.zeros((n, 24, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (n, 1)),
        "joint_pos": np.zeros((n, 29)),
        "joint_vel": np.zeros((n, 29)),
        "frame_index": np.arange(start, start + n, dtype=np.int64),
    }
    data.update(extra)
    return wire.pack_pose_message(data, version=version)


def _ready_planner_server(onnx):
    server, scheduler, planner, hands = make_server(onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    server.operator_start = True
    scheduler.set_motion(make_motion("planner_motion", 30, body_pos=True, encode_mode=0), play=True)
    return server, scheduler, planner, hands


def _streamed_server(onnx):
    server, scheduler, planner, hands = make_server(onnx)
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=False))
    server.apply()
    assert server.use_zmq_stream
    return server, scheduler, planner, hands


def test_full_mode_flow_over_loopback(loopback):
    env = loopback
    server, scheduler, planner, pub, sm = env["server"], env["scheduler"], env["planner"], env["pub"], env["sm"]

    # --- engage: A+B+X+Y -> PLANNER, robot starts and enables the planner ---
    _wait_for(
        lambda: server.operator_start,
        env["server"],
        sender=lambda: pub.send(build_command_message(start=True, stop=False, planner=True)),
        timeout=10.0,
        message="operator start",
    )
    sm.step(ButtonSnapshot(a=True, b=True, x=True, y=True))
    assert sm.mode == StreamMode.PLANNER
    env["three_point"].calibrate_now(make_body_sample(1).body_poses)
    assert server.active_mode == ManagedMode.PLANNER
    assert planner.state.enabled

    # Planner thread work, done synchronously here: initialize + adopt.
    planner.step()
    assert planner.state.initialized
    planner.advance(measured_q_mj=np.zeros(29))
    _wait_for(lambda: scheduler.play, server, message="planner playback")
    assert scheduler.motion.name == "planner_motion"
    assert scheduler.motion.encode_mode == 0  # g1 encoder

    # Stick commands flow into the movement state.
    _wait_for(
        lambda: np.linalg.norm(server.movement_state.movement_direction[:2]) > 0.9,
        env["server"],
        sender=lambda: env["planner_streamer"].run_once(
            StreamMode.PLANNER,
            make_body_sample(2),
            make_controllers(left_axis=(0.0, 1.0)),
            timestamp_ns=time.monotonic_ns(),
        ),
        timeout=10.0,
        message="movement command",
    )

    # --- A+X -> POSE: streamed motion takes over with the smpl encoder ---
    sm.step(ButtonSnapshot())  # release the engage combo first
    t = sm.step(ButtonSnapshot(a=True, x=True))
    assert t.mode == StreamMode.POSE
    pub.send(build_command_message(start=True, stop=False, planner=False))
    _wait_for(lambda: server.active_mode == ManagedMode.STREAMED_MOTION, server, message="streamed mode")

    pose_ticks = itertools.count(1)

    def _pump_pose():
        for _ in range(3):
            stamp_ns = 1_000_000_000 + next(pose_ticks) * 11_000_000
            env["pose_streamer"].run_once(make_body_sample(stamp_ns), make_controllers())

    _wait_for(
        lambda: scheduler.motion is not None and scheduler.motion.name == "streamed",
        env["server"],
        sender=_pump_pose,
        timeout=10.0,
        message="pose stream",
    )
    assert scheduler.motion.encode_mode == MODE_SMPL
    assert scheduler.play is True

    # --- B+Y -> FROZEN: operator grabs the measured upper body from g1_debug ---
    sm.step(ButtonSnapshot())
    t = sm.step(ButtonSnapshot(b=True, y=True))
    assert t.mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
    body_q = np.linspace(-0.5, 0.5, 29)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        env["debug_pub"].publish({"body_q_measured": body_q, "left_hand_q_measured": np.zeros(7)})
        time.sleep(0.02)
        env["feedback"].poll_feedback()
        if env["feedback"].upper_body_position_target is not None:
            break
    assert env["feedback"].upper_body_position_target is not None, "feedback never arrived"
    pub.send(build_command_message(start=True, stop=False, planner=True))
    _wait_for(lambda: server.active_mode == ManagedMode.PLANNER, server, message="planner mode")
    # The streamed-mode safety reset disabled the planner; the (synchronous
    # here) planner thread re-initializes and the control side re-adopts.
    _wait_for(lambda: planner.state.enabled, server, message="planner re-enable")
    planner.step()
    planner.advance(measured_q_mj=np.zeros(29))
    _wait_for(lambda: scheduler.play, server, message="planner playback resumed")

    _wait_for(
        lambda: scheduler.frozen_upper_targets is not None,
        env["server"],
        sender=lambda: env["planner_streamer"].run_once(
            StreamMode.PLANNER_FROZEN_UPPER_BODY,
            make_body_sample(3),
            make_controllers(),
            timestamp_ns=time.monotonic_ns(),
        ),
        timeout=10.0,
        message="frozen upper targets",
    )
    np.testing.assert_allclose(scheduler.frozen_upper_targets, body_q[[12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]], rtol=1e-6)  # fmt: skip

    # --- L-stick -> VR_3PT: teleop encoder with the streamed 3-point targets ---
    t = sm.step(ButtonSnapshot(left_axis_click=True))
    assert t.mode == StreamMode.PLANNER_VR_3PT
    env["planner_streamer"].recalibrate_for_vr3pt()
    scheduler.set_motion(planner.planner_motion, play=True, reinitialize_heading=False)

    def _vr_tick():
        # The robot publishes g1_debug every control tick; the wrist calibration waits for it.
        env["debug_pub"].publish({"body_q_measured": body_q, "left_hand_q_measured": np.zeros(7)})
        env["planner_streamer"].run_once(
            StreamMode.PLANNER_VR_3PT, make_body_sample(4), make_controllers(), timestamp_ns=time.monotonic_ns()
        )

    _wait_for(
        lambda: scheduler.vr_position is not None and scheduler.motion.encode_mode == 1,
        env["server"],
        sender=_vr_tick,
        timeout=10.0,
        message="vr targets + teleop mode",
    )

    # --- A+B+X+Y -> disengage: the planner's IDLE, g1 encoder again, session still running ---
    sm.step(ButtonSnapshot())
    t = sm.step(ButtonSnapshot(a=True, b=True, x=True, y=True))
    assert t.mode == StreamMode.OFF and Effect.RELEASE_TO_PLANNER in t.effects

    def _release():  # what the operator manager sends for that effect
        pub.send(build_command_message(start=False, stop=False, planner=True))
        env["planner_streamer"].send_idle()

    _wait_for(
        lambda: scheduler.motion.encode_mode == 0,
        env["server"],
        sender=_release,
        timeout=10.0,
        message="VR control dropped",
    )
    assert server.operator_start and not server.stop_requested

    # --- a reference streamer's stop still ends the session ---
    _wait_for(
        lambda: server.stop_requested,
        env["server"],
        sender=lambda: pub.send(build_command_message(start=False, stop=True, planner=True)),
        timeout=10.0,
        message="stop",
    )


def test_hand_controller_defaults_to_open():
    hands = Dex3HandController()
    targets = hands.compute(np.zeros(7), np.zeros(7))
    np.testing.assert_allclose(targets, 0.0)


def test_hand_controller_delta_clamp():
    hands = Dex3HandController()
    hands.set_targets(np.full(7, -1.5), None)
    # Left middle_0 (index 3) has range [-1.57, 0]: full close requested, but
    # only MAX_DELTA_Q per tick from the measured 0.
    targets = hands.compute(np.zeros(7), np.zeros(7))
    assert targets[3] == pytest.approx(-0.25)
    # Next tick from the moved state continues by another delta.
    targets = hands.compute(targets[:7], np.zeros(7))
    assert targets[3] == pytest.approx(-0.5)


def test_hand_controller_close_ratio_clip():
    hands = Dex3HandController(max_close_ratio=0.5)
    hands.set_targets(np.array([0, 0, 0, -1.5, 0, 0, 0]), np.array([0, 0, 0, 1.5, 0, 0, 0]))
    left_measured = np.full(7, -0.7)
    right_measured = np.full(7, 0.7)
    targets = hands.compute(left_measured, right_measured)
    # Left middle_0 limit -1.57 * 0.5 = -0.785; delta from -0.7 is small, allowed.
    assert targets[3] == pytest.approx(-1.57 * 0.5)
    # Right middle_0 limit 1.57 * 0.5 = 0.785.
    assert targets[10] == pytest.approx(1.57 * 0.5)


def test_hand_controller_ratio_bounds():
    hands = Dex3HandController()
    hands.set_max_close_ratio(0.05)
    assert hands.max_close_ratio == 0.2
    hands.set_max_close_ratio(2.0)
    assert hands.max_close_ratio == 1.0


def test_stop_command_latches_and_disables_planner(planner_onnx):
    server, _, planner, _ = make_server(planner_onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    server._dispatch(wire.build_command_message(start=False, stop=True, planner=True))
    server.apply()
    assert server.stop_requested is True
    assert planner.state.enabled is False


def test_start_command_enables_planner_and_waits_then_plays(planner_onnx):
    server, scheduler, planner, _ = make_server(planner_onnx)
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=True))
    server.apply()
    assert server.operator_start is True
    assert planner.state.enabled is True
    assert scheduler.play is False  # still waiting for the first planner motion

    planner.state.initialized = True
    scheduler.set_motion(make_motion("planner_motion", 30, body_pos=True, encode_mode=0), play=False)
    server.apply()
    assert scheduler.play is True  # planner motion arrived -> playback starts


def test_planner_init_timeout_stops_control(planner_onnx, clock):
    server, _, _, _ = make_server(planner_onnx)
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=True))
    server.apply()
    clock(server.PLANNER_INIT_TIMEOUT_S + 0.1)  # the planner never produced a motion
    server.apply()
    assert server.stop_requested is True


def test_mode_switch_to_streamed_enables_zmq_streaming(planner_onnx):
    server, scheduler, planner, _ = make_server(planner_onnx)
    planner.state.enabled = True
    planner.state.initialized = True
    scheduler.set_motion(make_motion("planner_motion", 30, body_pos=True, encode_mode=0), play=True)
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=False))
    server.apply()
    assert server.active_mode == ManagedMode.STREAMED_MOTION
    assert server.use_zmq_stream is True
    # Safety reset froze the motion under a clip-like name and disabled the planner.
    assert scheduler.motion.name == "temporary_motion"
    assert scheduler.play is False
    assert planner.state.enabled is False


def test_planner_message_updates_movement_state(planner_onnx):
    server, _, _, _ = _ready_planner_server(planner_onnx)
    server._dispatch(
        wire.build_planner_message(mode=int(LocomotionMode.WALK), movement=[3, 4, 0], facing=[0, 2, 0], speed=1.5)
    )
    server.apply()
    state = server.movement_state
    assert state.locomotion_mode == LocomotionMode.WALK
    np.testing.assert_allclose(state.movement_direction, [0.6, 0.8, 0])  # normalized
    np.testing.assert_allclose(state.facing_direction, [0, 1, 0])
    assert state.movement_speed == pytest.approx(1.5)


def test_planner_message_clamps_squat_height_and_static_speed(planner_onnx):
    server, _, _, _ = _ready_planner_server(planner_onnx)
    server._dispatch(
        wire.build_planner_message(
            mode=int(LocomotionMode.IDLE_SQUAT), movement=[1, 0, 0], facing=[1, 0, 0], height=0.05
        )
    )
    server.apply()
    assert server.movement_state.height == pytest.approx(0.2)  # squat floor
    assert server.movement_state.movement_speed == pytest.approx(-1.0)  # static mode


def test_vr_presence_toggles_encoder_mode(planner_onnx):
    server, scheduler, _, _ = _ready_planner_server(planner_onnx)
    vr_msg = wire.build_planner_message(
        mode=int(LocomotionMode.IDLE),
        movement=[0, 0, 0],
        facing=[1, 0, 0],
        vr_3pt_position=np.arange(9.0),
        vr_3pt_orientation=np.arange(12.0),
    )
    server._dispatch(vr_msg)
    server.apply()
    assert scheduler.motion.encode_mode == 1  # rising edge -> teleop
    np.testing.assert_allclose(scheduler.vr_position, np.arange(9.0))
    np.testing.assert_allclose(scheduler.vr_orientation, np.arange(12.0))

    server._dispatch(wire.build_planner_message(mode=int(LocomotionMode.IDLE), movement=[0, 0, 0], facing=[1, 0, 0]))
    server.apply()
    assert scheduler.motion.encode_mode == 0  # falling edge -> g1
    assert scheduler.vr_position is not None  # buffers retain the last value


def test_frozen_upper_targets_follow_message_presence(planner_onnx):
    server, scheduler, _, hands = _ready_planner_server(planner_onnx)
    upper = np.linspace(-1, 1, 17)
    server._dispatch(
        wire.build_planner_message(
            mode=int(LocomotionMode.IDLE),
            movement=[0, 0, 0],
            facing=[1, 0, 0],
            upper_body_position=upper,
            left_hand_position=np.full(7, -0.1),
            right_hand_position=np.full(7, 0.1),
        )
    )
    server.apply()
    np.testing.assert_allclose(scheduler.frozen_upper_targets, upper, rtol=1e-6)
    assert scheduler.frozen_upper_velocities is None  # zeros are applied by the policy
    targets = hands.compute(np.full(7, -0.1), np.full(7, 0.1))
    expected_left = np.clip(-0.1, hands.MIN_LIMITS_LEFT, hands.MAX_LIMITS_LEFT)  # thumb_2 clips to its 0 floor
    np.testing.assert_allclose(targets[:7], expected_left, atol=1e-6)

    server._dispatch(wire.build_planner_message(mode=int(LocomotionMode.IDLE), movement=[0, 0, 0], facing=[1, 0, 0]))
    server.apply()
    assert scheduler.frozen_upper_targets is None


def test_planner_silence_keeps_vr_3point_control(planner_onnx, clock):
    # The reference timeout clears only the upper-body/hand controls; VR
    # 3-point control persists, so no encoder-mode edge fires mid-dropout.
    server, scheduler, _, _ = _ready_planner_server(planner_onnx)
    server._dispatch(
        wire.build_planner_message(
            mode=int(LocomotionMode.IDLE),
            movement=[0, 0, 0],
            facing=[1, 0, 0],
            vr_3pt_position=np.arange(9.0),
            vr_3pt_orientation=np.arange(12.0),
        )
    )
    server.apply()
    assert scheduler.motion.encode_mode == 1
    clock(server.PLANNER_TIMEOUT_S + 0.1)
    server.apply()
    assert scheduler.motion.encode_mode == 1  # teleop mode survives the dropout
    assert server._has_vr_3point_control is True


def test_vr_edge_lands_on_the_planner_motion_adopted_after_the_wait(planner_onnx):
    """VR present at engage: the edge must fire on the planner motion, not on the pre-adoption gap.

    The planner flags itself initialized a tick before its first generation is adopted, so the first
    VR message is applied while ``scheduler.motion`` is still None. Latching the edge there left the
    adopted motion in g1 mode for the whole VR_3PT session.
    """
    server, scheduler, planner, _ = make_server(planner_onnx)
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=True))
    server.apply()
    planner.state.initialized = True  # planner thread: generation stored, not yet adopted
    server._dispatch(
        wire.build_planner_message(
            mode=int(LocomotionMode.IDLE),
            movement=[0, 0, 0],
            facing=[1, 0, 0],
            vr_3pt_position=np.arange(9.0),
            vr_3pt_orientation=np.arange(12.0),
        )
    )
    server.apply()
    assert scheduler.motion is None
    scheduler.set_motion(make_motion("planner_motion", 30, body_pos=True, encode_mode=0), play=False)  # adoption
    server.apply()
    assert scheduler.motion.encode_mode == 1


@pytest.mark.parametrize(
    "kwarg, wire_name, size",
    [
        ("left_hand_position", "left_hand_joints", 6),
        ("right_hand_position", "right_hand_joints", 14),
        ("upper_body_position", "upper_body_position", 16),
        ("upper_body_velocity", "upper_body_velocity", 29),
        ("vr_3pt_position", "vr_position", 3),
        ("vr_3pt_orientation", "vr_orientation", 4),
    ],
)
def test_wrong_size_planner_field_is_rejected_on_the_receive_thread(planner_onnx, kwarg, wire_name, size):
    """Unchecked, a 6-element hand target raised in ``reshape(7)`` inside ``apply()``, on the control loop."""
    server, scheduler, _, hands = _ready_planner_server(planner_onnx)
    bad = wire.build_planner_message(
        mode=int(LocomotionMode.IDLE), movement=[0, 0, 0], facing=[1, 0, 0], **{kwarg: np.zeros(size)}
    )
    with pytest.raises(ValueError, match=f"'{wire_name}' has {size} elements"):
        server._dispatch(bad)  # the receive loop logs and drops it
    server.apply()  # the control tick never sees it
    assert scheduler.frozen_upper_targets is None and scheduler.vr_position is None
    np.testing.assert_allclose(hands.compute(np.zeros(7), np.zeros(7)), 0.0)


@pytest.mark.parametrize("field", ["heading_increment", "left_trigger", "catch_up"])
def test_non_finite_pose_scalar_is_rejected(planner_onnx, field):
    """A NaN heading_increment was summed into the robot's delta heading, poisoning every anchor slot."""
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    with pytest.raises(ValueError, match=f"'{field}' is nan"):
        server._dispatch(_pose_chunk_msg(0, **{field: np.array([np.nan], np.float32)}))
    server.apply()
    assert scheduler.delta_heading == 0.0


def test_non_finite_command_flag_is_rejected(planner_onnx):
    """bool(nan) is True: a NaN 'stop' would have ended the session."""
    msg = wire.pack_message(
        "command",
        {
            "start": np.array([0.0], np.float32),
            "stop": np.array([np.nan], np.float32),
            "planner": np.array([1.0], np.float32),
        },
        version=1,
    )
    server, _, _, _ = make_server(planner_onnx)
    with pytest.raises(ValueError, match="'stop' is nan"):
        server._dispatch(msg)
    server.apply()
    assert server.stop_requested is False


def test_the_operators_idle_hand_off_releases_vr_and_the_frozen_upper_body(planner_onnx, g1_key_frame_fk):
    """The mode command alone left VR_3PT tracking the last wrist targets: the planner timeout keeps VR."""
    server, scheduler, _, _ = _ready_planner_server(planner_onnx)
    server._dispatch(
        wire.build_planner_message(
            mode=int(LocomotionMode.WALK),
            movement=[1, 0, 0],
            facing=[1, 0, 0],
            upper_body_position=np.zeros(17),
            vr_3pt_position=np.arange(9.0),
            vr_3pt_orientation=np.arange(12.0),
        )
    )
    server.apply()
    assert scheduler.motion.encode_mode == 1 and scheduler.frozen_upper_targets is not None

    sent = []
    publisher, port = bind_debug_publisher()
    feedback = FeedbackReader(host="127.0.0.1", port=port)
    try:
        streamer = PlannerStreamer(send=sent.append, three_point=ThreePointPose(g1_key_frame_fk), feedback=feedback)
        streamer.send_idle()
    finally:
        feedback.close()
        publisher.close()
    server._dispatch(sent[-1])
    server.apply()
    assert scheduler.motion.encode_mode == 0  # g1 encoder again
    assert scheduler.frozen_upper_targets is None
    assert server.movement_state.locomotion_mode == LocomotionMode.IDLE


def test_vr_edge_fires_again_after_a_streamed_round_trip(planner_onnx):
    """VR_3PT -> POSE -> PLANNER -> VR_3PT: the old session's latch swallowed the new rising edge."""
    vr_msg = wire.build_planner_message(
        mode=int(LocomotionMode.IDLE),
        movement=[0, 0, 0],
        facing=[1, 0, 0],
        vr_3pt_position=np.arange(9.0),
        vr_3pt_orientation=np.arange(12.0),
    )
    server, scheduler, planner, _ = _ready_planner_server(planner_onnx)
    server._dispatch(vr_msg)
    server.apply()
    assert scheduler.motion.encode_mode == 1  # first VR session

    server._dispatch(wire.build_command_message(start=True, stop=False, planner=False))  # -> POSE
    server.apply()
    assert server.active_mode == ManagedMode.STREAMED_MOTION and planner.state.enabled is False
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=True))  # -> PLANNER
    server.apply()
    assert planner.state.enabled is True  # waiting for a fresh planner motion
    planner.state.initialized = True
    server._dispatch(vr_msg)  # VR_3PT again, before the new motion is adopted
    server.apply()
    scheduler.set_motion(make_motion("planner_motion", 30, body_pos=True, encode_mode=0), play=False)
    server.apply()
    assert scheduler.motion.encode_mode == 1


def test_non_finite_planner_movement_is_rejected(planner_onnx):
    """A NaN stick survives ``max(nan, 1e-12)`` normalization and disables the planner on its next generation."""
    server, _, _, _ = _ready_planner_server(planner_onnx)
    bad = wire.build_planner_message(mode=int(LocomotionMode.WALK), movement=[np.nan, 0, 0], facing=[1, 0, 0])
    with pytest.raises(ValueError, match="'movement' contains NaN"):
        server._dispatch(bad)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("joint_pos", np.zeros(29), r"'joint_pos' has shape \(29,\); expected \(5, 29\)"),
        ("joint_vel", np.zeros((5, 23)), r"'joint_vel' has shape \(5, 23\); expected \(5, 29\)"),
        ("smpl_joints", np.zeros((5, 22, 3), np.float32), r"'smpl_joints' has shape \(5, 22, 3\)"),
        ("left_hand_joints", np.zeros(6, np.float32), "'left_hand_joints' has 6 elements"),
        ("vr_position", np.zeros(3, np.float32), "'vr_position' has 3 elements"),
    ],
)
def test_wrong_shape_pose_field_is_rejected_on_the_receive_thread(planner_onnx, field, value, match):
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    with pytest.raises(ValueError, match=match):
        server._dispatch(_pose_chunk_msg(0, **{field: value}))
    server.apply()
    assert scheduler.motion is None
    server._dispatch(_pose_chunk_msg(0))
    server.apply()
    assert scheduler.motion is not None  # healthy chunks still merge


def test_malformed_pose_chunk_is_dropped_not_fatal(planner_onnx):
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    # frame_index length disagrees with the data rows: rejected on the
    # receive thread (whose loop logs and drops it); the session keeps running.
    bad = wire.pack_pose_message(
        {
            "smpl_pose": np.zeros((5, 21, 3), dtype=np.float32),
            "smpl_joints": np.zeros((5, 24, 3), dtype=np.float32),
            "body_quat_w": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (5, 1)),
            "joint_pos": np.zeros((5, 29)),
            "joint_vel": np.zeros((5, 29)),
            "frame_index": np.arange(3, dtype=np.int64),
        },
        version=3,
    )
    with pytest.raises(ValueError, match="the frame count comes from 'frame_index'"):
        server._dispatch(bad)
    server.apply()  # must not raise
    assert scheduler.motion is None
    server._dispatch(_pose_chunk_msg(0))
    server.apply()
    assert scheduler.motion is not None  # healthy chunks still merge


@pytest.mark.parametrize("catch_up", [True, False])
def test_pose_catch_up_flag_bounds_the_stream_lag(planner_onnx, catch_up):
    """A chunk landing more than MAX_GAP + HISTORY frames past the paused playhead resets the window only with catch-up on."""
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    server._dispatch(_pose_chunk_msg(0, n=210))  # the stream's first chunk; playback stays at frame 0
    server.apply()
    # Contiguous with the window, but 210 frames ahead of the playhead (> 200 + 5).
    server._dispatch(_pose_chunk_msg(210, catch_up=np.array([catch_up])))
    server.apply()
    if catch_up:
        assert server.midstream_catchups == 1
        assert scheduler.motion.timesteps == 5  # the window restarted at the new chunk
    else:
        assert server.midstream_catchups == 0
        assert scheduler.motion.timesteps == 215  # unbounded lag: the chunk was appended


def test_pose_vr_without_orientation_resets_to_defaults(planner_onnx):
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    scheduler.vr_orientation = np.full(12, 9.9)  # stale value
    server._dispatch(_pose_chunk_msg(0, vr_position=np.arange(9, dtype=np.float32)))
    server.apply()
    np.testing.assert_allclose(scheduler.vr_position, np.arange(9.0))
    np.testing.assert_allclose(scheduler.vr_orientation, VR_3POINT_DEFAULT_ORIENTATION)


def test_planner_silence_resets_to_idle_keeping_facing(planner_onnx, clock):
    server, scheduler, _, _ = _ready_planner_server(planner_onnx)
    server._dispatch(
        wire.build_planner_message(mode=int(LocomotionMode.WALK), movement=[1, 0, 0], facing=[0, 1, 0], speed=1.0)
    )
    server.apply()
    scheduler.frozen_upper_targets = np.zeros(17)
    clock(server.PLANNER_TIMEOUT_S + 0.1)  # silence exceeds the timeout
    server.apply()
    state = server.movement_state
    assert state.locomotion_mode == LocomotionMode.IDLE
    np.testing.assert_allclose(state.facing_direction, [0, 1, 0])  # facing preserved
    assert state.movement_speed == pytest.approx(-1.0)
    assert scheduler.frozen_upper_targets is None
    # A second silent tick stays quiet (buffer cleared, no repeated resets).
    server.apply()
    assert server.movement_state.locomotion_mode == LocomotionMode.IDLE


def test_pose_chunks_merge_into_streamed_motion(planner_onnx):
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    server._dispatch(_pose_chunk_msg(100, extra_field=np.zeros(1, dtype=np.float32)))
    server.apply()
    assert scheduler.motion.name == "streamed"
    assert scheduler.motion.encode_mode == 2  # v3 -> smpl
    assert scheduler.play is True
    assert scheduler.frame == 0

    server._dispatch(_pose_chunk_msg(101))
    server.apply()
    assert scheduler.motion.timesteps == 6  # overlapping window merged


def test_pose_v1_sets_g1_mode(planner_onnx):
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    msg = wire.pack_pose_message(
        {
            "body_quat_w": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (5, 1)),
            "joint_pos": np.zeros((5, 29)),
            "joint_vel": np.zeros((5, 29)),
            "frame_index": np.arange(5, dtype=np.int64),
        },
        version=1,
    )
    server._dispatch(msg)
    server.apply()
    assert scheduler.motion.encode_mode == 0


def test_pose_heading_increment_and_hands(planner_onnx):
    server, scheduler, _, hands = _streamed_server(planner_onnx)
    server._dispatch(
        _pose_chunk_msg(
            0,
            heading_increment=np.array([0.05], dtype=np.float32),
            left_hand_joints=np.full(7, -0.2, dtype=np.float32),
            right_hand_joints=np.full(7, 0.2, dtype=np.float32),
        )
    )
    server.apply()
    assert scheduler.delta_heading == pytest.approx(0.05, abs=1e-7)
    targets = hands.compute(np.full(7, -0.2), np.full(7, 0.2))
    expected_left = np.clip(-0.2, hands.MIN_LIMITS_LEFT, hands.MAX_LIMITS_LEFT)
    np.testing.assert_allclose(targets[:7], expected_left, atol=1e-6)


def test_pose_protocol_version_latch(planner_onnx):
    server, scheduler, _, _ = _streamed_server(planner_onnx)
    server._dispatch(_pose_chunk_msg(0, version=3))
    server.apply()
    assert scheduler.play is True
    # A v1 chunk mid-session kills streaming.
    msg_v1 = wire.pack_pose_message(
        {
            "body_quat_w": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (5, 1)),
            "joint_pos": np.zeros((5, 29)),
            "joint_vel": np.zeros((5, 29)),
            "frame_index": np.arange(10, 15, dtype=np.int64),
        },
        version=1,
    )
    server._dispatch(msg_v1)
    server.apply()
    assert server.use_zmq_stream is False
    assert scheduler.play is False


def test_pose_v4_rejected(planner_onnx):
    server, _, _, _ = _streamed_server(planner_onnx)
    server._dispatch(wire.pack_pose_message({"token_state": np.zeros(64, dtype=np.float32)}, version=4))
    server.apply()
    assert server.use_zmq_stream is False


def test_emergency_stop(planner_onnx):
    server, _, planner, _ = _ready_planner_server(planner_onnx)
    server.emergency_stop()
    assert server.stop_requested is True
    assert planner.state.enabled is False


def test_reset_session_returns_to_launch_state(planner_onnx):
    server, scheduler, planner, hands = make_server(planner_onnx)
    # Engage, then dirty everything: streamed mode with a pose chunk, closed
    # hands, VR fields.
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=True))
    server.apply()
    planner.state.initialized = True
    scheduler.set_motion(make_motion("planner_motion", 30, body_pos=True, encode_mode=0), play=True)
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=False))
    server._dispatch(_pose_chunk_msg(0))
    server.apply()
    hands.set_targets(np.full(7, -1.0), None)
    scheduler.vr_position = np.zeros(9)
    scheduler.vr_orientation = np.zeros(12)
    scheduler.left_hand_joints = np.zeros(7)
    assert server.use_zmq_stream is True and scheduler.motion is not None

    server.reset_session()

    assert server.stop_requested is False  # unlike emergency_stop: keep running
    assert server.operator_start is False
    assert server.active_mode == ManagedMode.PLANNER
    assert server.use_zmq_stream is False
    assert server.movement_state.locomotion_mode == int(LocomotionMode.IDLE)
    assert planner.state.enabled is False and planner.state.initialized is False
    assert scheduler.motion is None and scheduler.play is False
    assert scheduler.vr_position is None and scheduler.vr_orientation is None
    assert scheduler.left_hand_joints is None and scheduler.frozen_upper_targets is None
    assert server._active_protocol_version == -1  # the next stream re-latches its version
    np.testing.assert_allclose(hands.compute(np.zeros(7), np.zeros(7)), 0.0)  # hands ramp back open

    # Re-engage exactly like a fresh launch: start -> the planner wait begins.
    server._dispatch(wire.build_command_message(start=True, stop=False, planner=True))
    server.apply()
    assert server.operator_start is True
    assert planner.state.enabled is True
    assert scheduler.play is False  # waiting for the first planner motion
