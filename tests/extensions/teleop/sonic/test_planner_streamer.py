"""``eden.extensions.teleop.sonic.planner_streamer``: joystick -> planner commands, frozen-upper and VR 3-point attachments.

The streamer runs over its real collaborators: a ``ThreePointPose`` on the G1 key-frame FK
and a ``FeedbackReader`` subscribed to a loopback robot ``G1DebugPublisher``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from eden.extensions.policies.sonic import wire
from eden.extensions.policies.sonic.planner import LocomotionMode
from eden.extensions.teleop.sonic.feedback import UPPER_BODY_MJ_GATHER
from eden.extensions.teleop.sonic.pose_streamer import IDLE_POLL_S
from eden.extensions.teleop.sonic.state_machine import StreamMode
from tests.extensions.teleop.sonic.conftest import build_planner_streamer, publish_until, run_planner_streamer
from tests.support.sonic import make_body_sample, make_controllers

pytestmark = pytest.mark.requires_genesis  # the ThreePointPose runs on the Genesis G1 FK


@pytest.fixture
def planner(three_point, debug_link):
    """(streamer, sent messages, robot-side g1_debug publisher); the robot publishes nothing yet."""
    publisher, reader = debug_link
    sent = []
    return build_planner_streamer(sent, three_point, reader), sent, publisher


def _parse(msg: bytes) -> wire.PlannerMessage:
    return wire.parse_planner(wire.unpack_message(msg, wire.PLANNER_TOPIC))


def test_planner_idle_stick_sends_idle(planner):
    streamer, sent, _ = planner
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(), ts=1)
    msg = _parse(sent[0])
    assert msg.mode == LocomotionMode.IDLE
    assert msg.speed == pytest.approx(-1.0)
    np.testing.assert_allclose(msg.movement, 0.0)
    np.testing.assert_allclose(msg.facing, [1, 0, 0])
    assert msg.upper_body_position is None
    assert msg.vr_position is None


def test_planner_stick_maps_movement_and_speed(planner):
    streamer, sent, _ = planner
    streamer.mode = LocomotionMode.RUN
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(left_axis=(0.0, 1.0)), ts=1)
    msg = _parse(sent[0])
    assert msg.mode == LocomotionMode.RUN
    assert msg.speed == pytest.approx(1.5 + 3.0 * 1.0)  # full deflection
    np.testing.assert_allclose(msg.movement, [1, 0, 0], atol=1e-6)  # forward along facing


def test_planner_mode_cycling_edges_and_clamps(planner):
    streamer, _, _ = planner
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(a=True, b=True), ts=1)
    assert streamer.mode == LocomotionMode.SLOW_WALK
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(a=True, b=True), ts=2)
    assert streamer.mode == LocomotionMode.SLOW_WALK  # held, no retrigger
    streamer.mode = LocomotionMode.INJURED_WALK
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(), ts=3)
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(a=True, b=True), ts=4)
    assert streamer.mode == LocomotionMode.INJURED_WALK  # clamped at the top
    streamer.mode = LocomotionMode.IDLE
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(), ts=5)
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(x=True, y=True), ts=6)
    assert streamer.mode == LocomotionMode.IDLE  # clamped at the bottom


def test_planner_stalled_headset_sends_nothing_and_waits(planner):
    """Between headset frames the manager loop re-ran the streamer at kHz with no sleep."""
    streamer, sent, _ = planner
    sleeps = []
    streamer._sleep = sleeps.append  # the sleep seam (clock boundary)
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(), ts=7)
    sleeps.clear()
    run_planner_streamer(streamer, StreamMode.PLANNER, make_controllers(), ts=7)  # timestamp unchanged
    assert len(sent) == 1
    assert sleeps == [IDLE_POLL_S]


def test_frozen_mode_attaches_upper_body_targets(planner):
    streamer, sent, publisher = planner
    body_q = np.linspace(-1, 1, 29)
    robot_state = {
        "body_q_measured": body_q,
        "left_hand_q_measured": np.full(7, 0.2),
        "right_hand_q_measured": np.full(7, -0.2),
    }

    def frozen():
        streamer.save_upper_body_position_target()  # the SAVE_FROZEN_TARGETS effect
        return streamer.feedback.upper_body_position_target is not None

    publish_until(publisher, robot_state, frozen)
    run_planner_streamer(streamer, StreamMode.PLANNER_FROZEN_UPPER_BODY, make_controllers(), ts=1)
    msg = _parse(sent[0])
    np.testing.assert_allclose(msg.upper_body_position, body_q[list(UPPER_BODY_MJ_GATHER)], rtol=1e-6)
    np.testing.assert_allclose(msg.left_hand_joints, 0.2, atol=1e-6)
    np.testing.assert_allclose(msg.right_hand_joints, -0.2, atol=1e-6)
    assert msg.vr_position is None


def test_vr3pt_mode_attaches_calibrated_vr_pose_and_hands(planner, three_point, g1_key_frame_fk):
    streamer, sent, _ = planner
    three_point.calibrate_now(make_body_sample(0).body_poses)  # the engage CALIBRATE_FULL effect
    run_planner_streamer(streamer, StreamMode.PLANNER_VR_3PT, make_controllers(right_trigger=0.9), ts=1)
    msg = _parse(sent[0])
    # The same body frame as the calibration: the streamed wrists land on FK of q = 0,
    # the neck on the synthesized chain straight up (0.05 + 0.35 m).
    targets = g1_key_frame_fk.key_frame_poses(np.zeros(29))
    np.testing.assert_allclose(msg.vr_position[0:3], targets["left_wrist"]["position"], atol=1e-5)
    np.testing.assert_allclose(msg.vr_position[3:6], targets["right_wrist"]["position"], atol=1e-5)
    np.testing.assert_allclose(msg.vr_position[6:9], [0.0, 0.0, 0.4], atol=1e-5)
    assert msg.vr_orientation.shape == (12,)
    assert msg.upper_body_position is None  # VR 3-point never sends the frozen upper body
    assert np.abs(msg.right_hand_joints).sum() > 0  # trigger closed the right hand
    np.testing.assert_allclose(msg.left_hand_joints, 0.0, atol=1e-6)


def test_vr3pt_recalibration_waits_for_robot_feedback(planner, three_point, g1_key_frame_fk):
    """No feedback at VR_3PT entry: hold the 3-point targets back, retry, never calibrate against zeros."""
    streamer, sent, publisher = planner
    three_point.calibrate_now(make_body_sample(0).body_poses)
    measured_q = np.linspace(-0.3, 0.3, 29)
    measured = g1_key_frame_fk.key_frame_poses(measured_q)["left_wrist"]["position"]
    zero = g1_key_frame_fk.key_frame_poses(np.zeros(29))["left_wrist"]["position"]
    assert np.linalg.norm(measured - zero) > 0.05  # the two calibrations are distinguishable

    streamer.recalibrate_for_vr3pt()  # the RECALIBRATE_WRISTS effect, with the robot silent
    run_planner_streamer(streamer, StreamMode.PLANNER_VR_3PT, make_controllers(), ts=1)
    held = _parse(sent[-1])
    assert held.vr_position is None and held.vr_orientation is None  # the old fallback calibrated against zeros

    ticks = itertools.count(2)

    def tick_streams_vr():
        run_planner_streamer(streamer, StreamMode.PLANNER_VR_3PT, make_controllers(), ts=next(ticks))
        return _parse(sent[-1]).vr_position is not None

    # Every tick before the robot state arrives is held back (the loop's condition); the first
    # one after recalibrates against the measured joints and streams on that same frame.
    publish_until(publisher, {"body_q_measured": measured_q}, tick_streams_vr)
    msg = _parse(sent[-1])
    np.testing.assert_allclose(msg.vr_position[0:3], measured, atol=1e-5)
    targets = g1_key_frame_fk.key_frame_poses(measured_q)
    np.testing.assert_allclose(msg.vr_position[3:6], targets["right_wrist"]["position"], atol=1e-5)


def test_frozen_targets_are_retried_until_robot_feedback_arrives(planner):
    streamer, sent, publisher = planner
    streamer.save_upper_body_position_target()  # the SAVE_FROZEN_TARGETS effect, with the robot silent
    run_planner_streamer(streamer, StreamMode.PLANNER_FROZEN_UPPER_BODY, make_controllers(), ts=1)
    assert _parse(sent[-1]).upper_body_position is None  # stays on the planner, no zero-pose freeze

    ticks = itertools.count(2)

    def tick_freezes():
        run_planner_streamer(streamer, StreamMode.PLANNER_FROZEN_UPPER_BODY, make_controllers(), ts=next(ticks))
        return _parse(sent[-1]).upper_body_position is not None

    body_q = np.linspace(-0.5, 0.5, 29)
    publish_until(publisher, {"body_q_measured": body_q}, tick_freezes)
    msg = _parse(sent[-1])
    np.testing.assert_allclose(msg.upper_body_position, body_q[list(UPPER_BODY_MJ_GATHER)], rtol=1e-6)
