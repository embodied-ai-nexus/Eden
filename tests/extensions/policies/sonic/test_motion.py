"""``eden.extensions.policies.sonic.motion``: the streamed-motion merger and the playback scheduler."""

from __future__ import annotations

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.extensions.policies.sonic.motion import (
    MergeResult,
    MotionScheduler,
    StreamedMotionMerger,
    heading_quat,
)
from tests.extensions.policies.sonic.conftest import make_motion, yawq
from tests.support.sonic import IDENTITY


def _chunk(start: int, n: int = 5, step: int = 1, mark: float | None = None):
    """Build a v3 chunk whose joint_pos rows encode their global frame index."""
    idx = np.arange(start, start + n * step, step, dtype=np.int64)
    value = idx.astype(np.float64) if mark is None else np.full(n, mark)
    return {
        "frame_indices": idx,
        "body_quat": np.tile(np.array([1.0, 0, 0, 0]), (n, 1, 1)),
        "joint_pos": np.tile(value[:, None], (1, 29)),
        "joint_vel": np.zeros((n, 29)),
        "smpl_joints": np.tile(value[:, None, None], (1, 24, 3)),
        "smpl_pose": np.zeros((n, 21, 3)),
        "protocol_version": 3,
    }


def test_stream_lag_frames():
    scheduler = MotionScheduler()
    assert scheduler.stream_lag_frames() is None
    scheduler.set_motion(make_motion(name="planner_motion", timesteps=20), frame=3)
    assert scheduler.stream_lag_frames() is None  # only streamed motions report lag
    scheduler.set_motion(make_motion(name="streamed", timesteps=20), frame=3)
    assert scheduler.stream_lag_frames() == 16


def test_merger_counts_only_midstream_catchups():
    merger = StreamedMotionMerger()

    def chunk(start: int, n: int = 5) -> dict:
        return {
            "frame_indices": np.arange(start, start + n),
            "body_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1, 1)),
            "joint_pos": np.zeros((n, 29)),
            "joint_vel": np.zeros((n, 29)),
            "protocol_version": 1,
        }

    first = merger.merge(**chunk(0), current_playback_frame=0)
    assert first.did_catchup_reset  # a stream's first chunk always catches up
    assert merger.midstream_catchup_count == 0

    merger.merge(**chunk(5), current_playback_frame=2)
    assert merger.midstream_catchup_count == 0

    stale = merger.merge(**chunk(0), current_playback_frame=2)  # already-covered frames
    assert stale.did_catchup_reset
    assert merger.midstream_catchup_count == 1

    gap = merger.merge(**chunk(500), current_playback_frame=2)  # hole in the timeline
    assert gap.did_catchup_reset
    assert merger.midstream_catchup_count == 2


def test_first_chunk_is_a_catchup_reset():
    merger = StreamedMotionMerger()
    result = merger.merge(**_chunk(100), current_playback_frame=0)
    assert result.did_catchup_reset is True
    assert result.window_start == 100
    assert result.frame_offset_adjustment == 0
    assert result.frame_step == 1
    assert result.motion.timesteps == 5
    np.testing.assert_allclose(result.motion.joint_pos[:, 0], [100, 101, 102, 103, 104])


def test_overlapping_chunk_merges_with_history_prefix():
    merger = StreamedMotionMerger()
    merger.merge(**_chunk(100), current_playback_frame=0)
    # Window [100..104], playback at 2: incoming [101..105] keeps window start
    # (playback is within HISTORY_FRAMES) and copies frame 100 as prefix.
    result = merger.merge(**_chunk(101), current_playback_frame=2)
    assert result.did_catchup_reset is False
    assert result.window_start == 100
    assert result.frame_offset_adjustment == 0
    assert result.motion.timesteps == 6
    np.testing.assert_allclose(result.motion.joint_pos[:, 0], [100, 101, 102, 103, 104, 105])
    np.testing.assert_allclose(result.motion.smpl_joints[:, 0, 0], [100, 101, 102, 103, 104, 105])


def test_window_shifts_forward_with_playback():
    merger = StreamedMotionMerger()
    merger.merge(**_chunk(100, n=10), current_playback_frame=0)
    # Playback at 9: window trails it by HISTORY_FRAMES, so the new window
    # starts at global frame 104 and the cursor must shift by 4.
    result = merger.merge(**_chunk(110), current_playback_frame=9)
    assert result.did_catchup_reset is False
    assert result.window_start == 104
    assert result.frame_offset_adjustment == 4
    assert result.motion.timesteps == 11
    np.testing.assert_allclose(result.motion.joint_pos[:, 0], np.arange(104, 115))


def test_large_gap_triggers_catchup():
    merger = StreamedMotionMerger()
    merger.merge(**_chunk(100), current_playback_frame=0)
    result = merger.merge(**_chunk(400), current_playback_frame=0)
    assert result.did_catchup_reset is True
    assert result.window_start == 400
    assert result.frame_offset_adjustment == 0
    assert result.motion.timesteps == 5
    np.testing.assert_allclose(result.motion.joint_pos[:, 0], np.arange(400, 405))


def test_stale_or_covered_chunks_force_catchup():
    merger = StreamedMotionMerger()
    merger.merge(**_chunk(100), current_playback_frame=0)
    # Starts at/before the current window start.
    result = merger.merge(**_chunk(98), current_playback_frame=0)
    assert result.did_catchup_reset is True
    assert result.window_start == 98
    # Ends at/before the current window end (no new frames).
    merger.reset()
    merger.merge(**_chunk(100), current_playback_frame=0)
    result = merger.merge(**_chunk(101, n=3), current_playback_frame=0)  # [101..103] <= end 104
    assert result.did_catchup_reset is True
    assert result.window_start == 101


def test_decimated_stream_detects_frame_step():
    merger = StreamedMotionMerger()
    result = merger.merge(**_chunk(100, n=5, step=2), current_playback_frame=0)
    assert result.frame_step == 2
    # Incoming [110..118] step 2: window keeps start 100 (playback 3 < history),
    # prefix copies all 5 old frames.
    result = merger.merge(**_chunk(110, n=5, step=2), current_playback_frame=3)
    assert result.did_catchup_reset is False
    assert result.window_start == 100
    assert result.frame_offset_adjustment == 0
    assert result.motion.timesteps == 10
    np.testing.assert_allclose(result.motion.joint_pos[:, 0], np.arange(100, 120, 2))


def test_v2_chunk_has_zero_width_joint_data():
    merger = StreamedMotionMerger()
    chunk = _chunk(0)
    chunk.pop("joint_pos")
    chunk.pop("joint_vel")
    chunk["protocol_version"] = 2
    result = merger.merge(**chunk, current_playback_frame=0)
    # Zero joint width mirrors the reference num_joints=0, so a g1-mode gather
    # fails and the encoder falls back instead of seeing silent zeros.
    assert result.motion.joint_pos.shape == (5, 0)
    assert result.motion.smpl_joints.shape == (5, 24, 3)


def test_v1_chunk_has_no_smpl_data():
    merger = StreamedMotionMerger()
    chunk = _chunk(0)
    chunk.pop("smpl_joints")
    chunk.pop("smpl_pose")
    chunk["protocol_version"] = 1
    result = merger.merge(**chunk, current_playback_frame=0)
    assert result.motion.smpl_joints is None
    assert result.motion.joint_pos.shape == (5, 29)


@pytest.mark.parametrize(
    "version, drop",
    [(3, ("joint_pos", "joint_vel")), (3, ("smpl_joints", "smpl_pose")), (2, ("smpl_joints", "smpl_pose")),
     (1, ("joint_pos", "joint_vel")), (7, ())],
)  # fmt: skip
def test_merge_validates_per_version_requirements(version, drop):
    merger = StreamedMotionMerger()
    chunk = _chunk(0)
    for name in drop:
        chunk.pop(name)
    chunk["protocol_version"] = version
    with pytest.raises(ValueError):
        merger.merge(**chunk, current_playback_frame=0)


def test_catchup_disabled_allows_unbounded_lag():
    merger = StreamedMotionMerger()
    merger.merge(**_chunk(0, n=5), current_playback_frame=0)
    # A 300-frame jump would trip MAX_GAP_FRAMES... but it is also a gap from
    # the old window end, which forces catch-up regardless. Use a contiguous
    # stream with a stalled playback cursor instead: merge many chunks while
    # playback stays at 0 so the copied prefix keeps growing past MAX_GAP.
    for start in range(5, 5 + 250 * 5, 5):
        result = merger.merge(**_chunk(start), current_playback_frame=0, catch_up_enabled=False)
    assert result.did_catchup_reset is False
    assert result.motion.timesteps == 5 + 249 * 5 + 5


def test_catchup_enabled_bounds_the_window():
    merger = StreamedMotionMerger()
    merger.merge(**_chunk(0, n=5), current_playback_frame=0)
    result = None
    for start in range(5, 5 + 250 * 5, 5):
        result = merger.merge(**_chunk(start), current_playback_frame=0)
        if result.did_catchup_reset:
            break
    assert result.did_catchup_reset is True


def test_clip_playback_stops_and_rewinds_at_end():
    sched = MotionScheduler()
    sched.set_motion(make_motion(timesteps=3), play=True)
    sched.advance_frame()
    sched.advance_frame()
    assert sched.frame == 2
    assert sched.play is True
    sched.advance_frame()  # steps past the end
    assert sched.frame == 0
    assert sched.play is False


def test_streamed_playback_holds_before_observation_window():
    sched = MotionScheduler()
    sched.set_motion(make_motion(name="streamed", timesteps=12), play=True)
    sched.begin_control_tick(IDENTITY)
    sched.note_observation_window(num_frames=10, step_size=1)  # hold = 10
    sched.advance_frame()
    assert sched.frame == 1
    sched.advance_frame()  # 2 >= 12 - 10 -> held back to 1
    assert sched.frame == 1
    assert sched.play is True  # streamed motions never stop, they wait


def test_observation_window_hold_resets_each_control_tick():
    sched = MotionScheduler()
    sched.set_motion(make_motion(name="streamed", timesteps=12), play=True)
    sched.begin_control_tick(IDENTITY)
    sched.note_observation_window(10, 5)  # (10-1)*5+1 = 46
    sched.note_observation_window(10, 1)  # max keeps 46
    sched.advance_frame()
    assert sched.frame == 0  # 1 >= 12 - 46 -> held
    sched.begin_control_tick(IDENTITY)  # new tick: hold resets to 0
    sched.advance_frame()
    assert sched.frame == 1


def test_planner_motion_clamps_at_last_frame():
    sched = MotionScheduler()
    sched.set_motion(make_motion(name="planner_motion", timesteps=3), play=True)
    for _ in range(5):
        sched.advance_frame(is_planner_motion=True)
    assert sched.frame == 2
    assert sched.play is True


def test_paused_playback_does_not_advance():
    sched = MotionScheduler()
    sched.set_motion(make_motion(), play=False)
    sched.advance_frame()
    assert sched.frame == 0


def test_motion_and_frame_returns_a_consistent_pair():
    sched = MotionScheduler()
    m = make_motion(timesteps=5)
    sched.set_motion(m, frame=3, play=True, reinitialize_heading=False)
    motion, frame = sched.motion_and_frame()
    assert motion is m
    assert frame == 3
    # None motion + reset frame come back together (never a frame from a
    # since-swapped motion).
    sched.set_motion(None)
    assert sched.motion_and_frame() == (None, 0)


def test_apply_merge_catchup_restarts_playback():
    sched = MotionScheduler()
    sched.set_motion(make_motion(name="streamed"), frame=3, play=False, reinitialize_heading=False)
    sched.apply_merge(MergeResult(motion=make_motion(name="streamed", timesteps=7), did_catchup_reset=True))
    assert sched.frame == 0
    assert sched.play is True
    assert sched.motion.timesteps == 7


def test_apply_merge_shifts_cursor_with_window():
    sched = MotionScheduler()
    sched.set_motion(make_motion(name="streamed", timesteps=11), frame=9, play=True)
    sched.apply_merge(MergeResult(motion=make_motion(name="streamed", timesteps=11), frame_offset_adjustment=4))
    assert sched.frame == 5  # 9 - 4

    # Shift past the cursor clamps to 0; a short window clamps to the last frame.
    sched.apply_merge(MergeResult(motion=make_motion(name="streamed", timesteps=11), frame_offset_adjustment=50))
    assert sched.frame == 0
    sched.set_motion(make_motion(name="streamed", timesteps=11), frame=9, play=True, reinitialize_heading=False)
    sched.apply_merge(MergeResult(motion=make_motion(name="streamed", timesteps=3), frame_offset_adjustment=0))
    assert sched.frame == 2


def test_apply_merge_without_motion_is_a_no_op():
    sched = MotionScheduler()
    sched.set_motion(make_motion(), frame=2, play=False, reinitialize_heading=False)
    sched.apply_merge(MergeResult(motion=None))
    assert sched.frame == 2
    assert sched.play is False


def test_heading_captured_on_reinitialize():
    sched = MotionScheduler()
    sched.set_motion(make_motion(root_yaw=0.0), play=True)  # set_motion requests re-capture
    base = yawq(np.pi / 2)
    sched.begin_control_tick(base)
    # apply_delta_heading = heading(base) ⊗ heading_inv(ref_root=identity) = yaw(pi/2)
    np.testing.assert_allclose(sched.snapshot().apply_delta_heading, yawq(np.pi / 2), atol=1e-12)
    # Without a new re-capture request, later ticks keep the captured heading.
    sched.begin_control_tick(yawq(-np.pi / 2))
    np.testing.assert_allclose(sched.snapshot().apply_delta_heading, yawq(np.pi / 2), atol=1e-12)


def test_heading_subtracts_reference_root_yaw():
    sched = MotionScheduler()
    sched.set_motion(make_motion(root_yaw=np.pi / 4), play=True)
    sched.begin_control_tick(yawq(np.pi / 2))
    expected = gu.transform_quat_by_quat(yawq(-np.pi / 4), yawq(np.pi / 2))
    np.testing.assert_allclose(sched.snapshot().apply_delta_heading, expected, atol=1e-12)


def test_delta_heading_accumulates_and_premultiplies():
    sched = MotionScheduler()
    sched.set_motion(make_motion(), play=True)
    sched.begin_control_tick(IDENTITY)
    sched.add_heading_increment(0.1)
    sched.add_heading_increment(0.15)
    assert sched.delta_heading == pytest.approx(0.25)
    np.testing.assert_allclose(sched.snapshot().apply_delta_heading, yawq(0.25), atol=1e-12)


def test_reference_root_recaptured_at_frame_zero():
    motion = make_motion(timesteps=4, root_yaw=np.pi / 3)
    motion.body_quat[0, 0] = yawq(np.pi / 6)  # frame 0 differs from the rest
    sched = MotionScheduler()
    sched.set_motion(motion, frame=2, play=True)
    sched.begin_control_tick(IDENTITY)  # captures ref root from frame 2 (yaw pi/3)
    np.testing.assert_allclose(sched.snapshot().apply_delta_heading, yawq(-np.pi / 3), atol=1e-12)
    sched.set_motion(motion, frame=0, play=True, reinitialize_heading=False)
    sched.begin_control_tick(IDENTITY)  # frame 0 -> ref root re-captured from frame 0
    np.testing.assert_allclose(sched.snapshot().apply_delta_heading, yawq(-np.pi / 6), atol=1e-12)


def test_snapshot_copies_zmq_extras():
    sched = MotionScheduler()
    motion = make_motion(name="planner_motion")
    motion.encode_mode = 1
    sched.set_motion(motion, play=True)
    sched.begin_control_tick(IDENTITY)
    sched.frozen_upper_targets = np.arange(17.0)
    sched.vr_position = np.arange(9.0)
    sched.vr_orientation = np.zeros(12)
    sched.left_hand_joints = np.full(7, 0.5)

    snap = sched.snapshot()
    assert snap.encode_mode == 1
    assert snap.play is True
    snap.frozen_upper_targets[0] = 99.0  # snapshot owns a copy
    assert sched.frozen_upper_targets[0] == 0.0
    np.testing.assert_allclose(snap.vr_position, np.arange(9.0))
    assert snap.right_hand_joints is None


def test_snapshot_without_motion():
    sched = MotionScheduler()
    snap = sched.snapshot()
    assert snap.motion is None
    assert snap.encode_mode == -2
    assert snap.play is False


def test_transform_quat_by_quat_left_multiplies():
    # transform_quat_by_quat(v, u) = u \u2297 v \u2014 the composition order the
    # scheduler and slot gathers rely on.
    a = yawq(0.3)
    b = yawq(0.5)
    np.testing.assert_allclose(gu.transform_quat_by_quat(b, a), yawq(0.8), atol=1e-12)


def test_heading_quat_ignores_tilt():
    # Compose yaw(0.7) with a roll: the heading of the x-axis stays 0.7.
    roll = gu.axis_angle_to_quat(np.array(0.4), np.array([1.0, 0.0, 0.0]))
    q = gu.transform_quat_by_quat(roll, yawq(0.7))
    np.testing.assert_allclose(heading_quat(q), yawq(0.7), atol=1e-12)
    np.testing.assert_allclose(heading_quat(q, inverse=True), yawq(-0.7), atol=1e-12)
