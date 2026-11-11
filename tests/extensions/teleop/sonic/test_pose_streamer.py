"""``eden.extensions.teleop.sonic.pose_streamer``: the 50 Hz pose resampler and its stream window.

The streamer runs over the real SMPL-X FK and a real ``ThreePointPose`` on the G1 key-frame FK.
"""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.policies.sonic import wire
from eden.extensions.teleop.sonic.pose_streamer import IDLE_POLL_S, PoseStreamer
from tests.support.sonic import make_body_sample, make_controllers, smpl_fk, unpaced

pytestmark = pytest.mark.requires_genesis  # the ThreePointPose runs on the Genesis G1 FK


@pytest.fixture
def pose(three_point):
    """(streamer, sent messages)."""
    sent = []
    streamer = unpaced(PoseStreamer(send=sent.append, three_point=three_point, fk=smpl_fk()))
    return streamer, sent


def _pump(streamer, start_i, n_calls, controllers=None):
    # ~90 Hz device frames (11 ms apart) resampled onto the 50 Hz virtual clock:
    # a virtual tick completes only when a device frame brackets it, so five
    # window entries need nine device frames.
    for i in range(start_i, start_i + n_calls):
        streamer.run_once(make_body_sample(1_000_000_000 + i * 11_000_000), controllers or make_controllers())


def _parse(msg: bytes) -> wire.PoseChunk:
    return wire.parse_pose(wire.unpack_message(msg, wire.POSE_TOPIC))


def test_pose_streamer_gates_on_full_window(pose):
    streamer, sent = pose
    _pump(streamer, 0, 9)
    assert len(sent) == 1  # published exactly once the window filled
    chunk = _parse(sent[0])
    np.testing.assert_array_equal(chunk.frame_index, [0, 1, 2, 3, 4])
    assert chunk.smpl_pose.shape == (5, 21, 3)
    assert chunk.joint_pos.shape == (5, 29)
    assert chunk.joint_pos.dtype == np.float64

    # Subsequent virtual ticks slide the window by one.
    _pump(streamer, 9, 2)
    np.testing.assert_array_equal(_parse(sent[-1]).frame_index, [1, 2, 3, 4, 5])


def test_pose_streamer_mode_exit_requires_fresh_window(pose):
    streamer, sent = pose
    _pump(streamer, 0, 11)
    assert sent
    sent.clear()
    streamer.on_mode_exit()
    _pump(streamer, 11, 7)
    assert sent == []  # window not yet refilled (needs 5 fresh virtual ticks)
    _pump(streamer, 18, 2)
    assert len(sent) >= 1
    np.testing.assert_array_equal(_parse(sent[0]).frame_index, [0, 1, 2, 3, 4])  # counter restarted


def test_pose_streamer_skips_duplicate_device_frames(pose):
    streamer, _ = pose
    streamer.run_once(make_body_sample(1_000_000_000), make_controllers())
    step_before = streamer.step
    streamer.run_once(make_body_sample(1_000_000_000), make_controllers())  # same timestamp
    assert streamer.step == step_before


def test_pose_streamer_waits_instead_of_spinning_on_a_repeated_frame(pose):
    """Between device frames the manager loop re-ran the SMPL FK and published at kHz with no sleep."""
    streamer, _ = pose
    sleeps = []
    streamer._sleep = sleeps.append  # the sleep seam (clock boundary)
    sample = make_body_sample(1_000_000_000)
    streamer.run_once(sample, make_controllers())  # primes the resampler
    sleeps.clear()
    for _ in range(3):
        streamer.run_once(sample, make_controllers())
    assert sleeps == [IDLE_POLL_S] * 3


def test_pose_streamer_streams_hand_targets_and_triggers(pose):
    streamer, sent = pose
    _pump(streamer, 0, 9, make_controllers(left_trigger=0.9))
    chunk = _parse(sent[-1])
    assert chunk.left_trigger == pytest.approx(0.9)
    assert np.abs(chunk.left_hand_joints).sum() > 0  # middle-gesture close
    np.testing.assert_allclose(chunk.right_hand_joints, 0.0)


def test_pose_streamer_streams_the_calibrated_vr_3pt_pose(pose, three_point, g1_key_frame_fk):
    streamer, sent = pose
    three_point.calibrate_now(make_body_sample(0).body_poses)  # the engage CALIBRATE_FULL effect
    _pump(streamer, 0, 9)  # every pumped frame is the calibration pose
    chunk = _parse(sent[-1])
    targets = g1_key_frame_fk.key_frame_poses(np.zeros(29))
    np.testing.assert_allclose(chunk.vr_position[0:3], targets["left_wrist"]["position"], atol=1e-5)
    np.testing.assert_allclose(chunk.vr_position[3:6], targets["right_wrist"]["position"], atol=1e-5)
    np.testing.assert_allclose(chunk.vr_position[6:9], [0.0, 0.0, 0.4], atol=1e-5)  # neck chain straight up
