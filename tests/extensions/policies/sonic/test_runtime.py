"""``eden teleop sonic robot``: argument checks (before the Genesis build and the ONNX loads) and loop helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.extensions.policies.sonic import runtime
from eden.extensions.policies.sonic.joint_maps import DEFAULT_ANGLES_MJ
from eden.extensions.policies.sonic.runtime import (
    FALL_TRIGGER_TICKS,
    FallDetector,
    HoldRamp,
    _check_args,
    build_arg_parser,
    joint_over_velocity_limit,
)
from tests.extensions.policies.sonic.conftest import make_motion
from tests.support.sonic import IDENTITY


@pytest.mark.parametrize("flag", ["--encoder-path", "--decoder-path", "--planner-path"])
def test_a_missing_local_checkpoint_is_rejected_up_front(tmp_path, capsys, flag):
    parser = build_arg_parser()
    missing = str(tmp_path / "nope.onnx")
    with pytest.raises(SystemExit):
        _check_args(parser, parser.parse_args([flag, missing]))
    assert f"{flag} {missing!r} is not a file" in capsys.readouterr().err


@pytest.mark.parametrize("ratio", ["0.1", "1.5"])
def test_an_out_of_range_close_ratio_is_rejected_not_clipped(capsys, ratio):
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _check_args(parser, parser.parse_args(["--max-close-ratio", ratio]))
    assert "--max-close-ratio must be in [0.2, 1.0]" in capsys.readouterr().err


def test_valid_flags_pass(tmp_path):
    ckpt = tmp_path / "encoder.onnx"
    ckpt.write_bytes(b"")
    parser = build_arg_parser()
    _check_args(parser, parser.parse_args(["--encoder-path", str(ckpt), "--max-close-ratio", "0.5"]))


def _rejected(capsys, *argv: str) -> str:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _check_args(parser, parser.parse_args(list(argv)))
    return capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--deploy_kp_kd_scale", "0.5"],
        ["--target_lpf", "0.5"],
        ["--disable_joints", "left_wrist_*"],
        ["--debug_record"],
        ["--debug_record_path", "run.npz"],
    ],
)
def test_a_deployment_flag_without_a_deployment_body_is_rejected_not_ignored(capsys, argv):
    assert f"{argv[0]}: these configure a deployment body" in _rejected(capsys, *argv)
    parser = build_arg_parser()
    _check_args(parser, parser.parse_args(["--mujoco", *argv]))


def test_flags_the_selected_body_cannot_honor_are_rejected(capsys):
    assert "--target_lpf must be in (0, 1]" in _rejected(capsys, "--mujoco", "--target_lpf", "0")
    assert "pass --real" in _rejected(capsys, "--mujoco", "--joint-vel-limit", "20")
    assert "drives no hands" in _rejected(capsys, "--mujoco", "--max-close-ratio", "0.5")
    assert "not allowed with argument" in _rejected(capsys, "--mujoco", "--real")


@pytest.mark.parametrize("argv", [["--auto-reset"], ["--no-band"], ["--drop-band-after", "3"]])
def test_the_real_robot_has_no_band_and_no_reset(capsys, argv):
    assert f"{argv[0]}: the real robot has no elastic band" in _rejected(capsys, "--real", *argv)


def test_the_real_robot_needs_the_unitree_sdk_before_anything_is_built(capsys, monkeypatch):
    def missing() -> None:
        raise RuntimeError("Unitree SDK (unitree_sdk2py) failed to import.")

    monkeypatch.setattr("eden.extensions.deployment.utils.unitree_sdk2.ensure_unitree_sdk", missing)
    assert "--real: Unitree SDK (unitree_sdk2py) failed to import." in _rejected(capsys, "--real")


STAND = np.asarray(DEFAULT_ANGLES_MJ, dtype=np.float64)


def test_the_hold_at_the_stand_is_the_exact_reference_pose():
    hold = HoldRamp(control_dt=0.02, device="cpu")
    expected = torch.tensor(DEFAULT_ANGLES_MJ, dtype=torch.float32).reshape(1, -1)
    assert torch.equal(hold.step(), expected)  # unseeded: already there
    hold.seed(STAND + 1e-4)  # a simulator's spawn, off by float noise
    assert torch.equal(hold.step(), expected)


def test_the_hold_approaches_the_stand_at_a_bounded_joint_speed():
    hold = HoldRamp(control_dt=0.02, device="cpu", rate_rad_s=0.5)
    start = STAND.copy()
    start[3] += 0.035  # 3.5 ticks away at 0.01 rad per tick
    hold.seed(start)

    offsets = [float(hold.step()[0, 3]) - STAND[3] for _ in range(5)]

    np.testing.assert_allclose(offsets, [0.025, 0.015, 0.005, 0.0, 0.0], atol=1e-6)
    with pytest.raises(ValueError, match="expects 29 body joints, got 3"):
        hold.seed(np.zeros(3))


def test_the_velocity_trip_names_the_first_runaway_joint():
    qd = np.zeros(29)
    assert joint_over_velocity_limit(qd, 35.0) is None
    qd[[4, 9]] = (-36.0, 50.0)
    assert joint_over_velocity_limit(qd, 35.0) == 4
    assert joint_over_velocity_limit(qd, 0.0) is None  # 0 disables


HALF = np.sqrt(0.5)


ROLL_90 = np.array([HALF, HALF, 0.0, 0.0])


PITCH_90 = np.array([HALF, 0.0, HALF, 0.0])


YAW_90 = np.array([HALF, 0.0, 0.0, HALF])


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _patch_clock(monkeypatch, clock: _FakeClock) -> None:
    monkeypatch.setattr(runtime.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(runtime.time, "sleep", clock.sleep)


def test_disarmed_or_referenceless_never_triggers():
    det = FallDetector()
    for _ in range(5 * FALL_TRIGGER_TICKS):
        assert det.update(armed=False, base_quat_wxyz=ROLL_90, ref_root_quat_wxyz=IDENTITY) is False
        assert det.update(armed=True, base_quat_wxyz=ROLL_90, ref_root_quat_wxyz=None) is False


def test_tracking_the_reference_never_triggers():
    det = FallDetector()
    for _ in range(5 * FALL_TRIGGER_TICKS):
        assert det.update(armed=True, base_quat_wxyz=IDENTITY, ref_root_quat_wxyz=IDENTITY) is False
        # Yaw is free: heading offsets between robot and reference are normal.
        assert det.update(armed=True, base_quat_wxyz=YAW_90, ref_root_quat_wxyz=IDENTITY) is False


def test_commanded_crawl_does_not_trigger():
    det = FallDetector()
    for _ in range(5 * FALL_TRIGGER_TICKS):
        # Base pitched 90° while the reference commands the same posture.
        assert det.update(armed=True, base_quat_wxyz=PITCH_90, ref_root_quat_wxyz=PITCH_90) is False


def test_transient_deviation_does_not_trigger():
    det = FallDetector()
    for _ in range(FALL_TRIGGER_TICKS - 1):
        assert det.update(armed=True, base_quat_wxyz=ROLL_90, ref_root_quat_wxyz=IDENTITY) is False
    # Recovery resets the counter; the next sustained run starts from zero.
    assert det.update(armed=True, base_quat_wxyz=IDENTITY, ref_root_quat_wxyz=IDENTITY) is False
    for _ in range(FALL_TRIGGER_TICKS - 1):
        assert det.update(armed=True, base_quat_wxyz=ROLL_90, ref_root_quat_wxyz=IDENTITY) is False


def test_sustained_fall_triggers_once_then_rearms():
    det = FallDetector()
    fired = [
        det.update(armed=True, base_quat_wxyz=ROLL_90, ref_root_quat_wxyz=IDENTITY) for _ in range(FALL_TRIGGER_TICKS)
    ]
    assert fired == [False] * (FALL_TRIGGER_TICKS - 1) + [True]
    # The counter reset on trigger: another full sustained run is needed.
    fired = [
        det.update(armed=True, base_quat_wxyz=ROLL_90, ref_root_quat_wxyz=IDENTITY) for _ in range(FALL_TRIGGER_TICKS)
    ]
    assert fired == [False] * (FALL_TRIGGER_TICKS - 1) + [True]


def test_falling_backward_from_a_lying_reference_triggers():
    det = FallDetector()
    lying_face_down = PITCH_90
    fallen_on_back = np.array([HALF, 0.0, -HALF, 0.0])  # pitched 90° the other way
    fired = [
        det.update(armed=True, base_quat_wxyz=fallen_on_back, ref_root_quat_wxyz=lying_face_down)
        for _ in range(FALL_TRIGGER_TICKS)
    ]
    assert fired[-1] is True


def test_reference_is_upright_when_disengaged():
    from eden.extensions.policies.sonic.runtime import fall_reference_quat

    # A robot dropped before (or without) an engage falls against an upright
    # reference — the reference sim resets that fall too.
    np.testing.assert_array_equal(fall_reference_quat(False, None, 0), IDENTITY)
    np.testing.assert_array_equal(
        fall_reference_quat(False, make_motion("m", 10, quat=PITCH_90, body_pos=True), 0), IDENTITY
    )


def test_reference_is_the_commanded_root_when_engaged():
    from eden.extensions.policies.sonic.runtime import fall_reference_quat

    motion = make_motion("m", 10, quat=PITCH_90, body_pos=True)
    np.testing.assert_array_equal(fall_reference_quat(True, motion, 3), PITCH_90)
    # Frame past the end clamps to the last one instead of raising.
    np.testing.assert_array_equal(fall_reference_quat(True, motion, 99), PITCH_90)
    # Engaged but referenceless (shouldn't happen, but never crash): upright.
    np.testing.assert_array_equal(fall_reference_quat(True, None, 0), IDENTITY)


def test_pace_sleeps_to_the_absolute_deadline(monkeypatch):
    clock = _FakeClock(start=100.0)
    _patch_clock(monkeypatch, clock)
    stats = runtime._LoopStats()
    new_deadline = runtime._pace(deadline=100.015, control_dt=0.02, stats=stats)
    assert clock.sleeps == pytest.approx([0.015])
    assert new_deadline == pytest.approx(100.035)
    assert stats.deadline_resyncs == 0


def test_pace_amortizes_a_small_overrun_without_resync(monkeypatch):
    clock = _FakeClock(start=100.1)  # 100 ms past the deadline
    _patch_clock(monkeypatch, clock)
    stats = runtime._LoopStats()
    new_deadline = runtime._pace(deadline=100.0, control_dt=0.02, stats=stats)
    # No sleep: the next ticks run back-to-back until the schedule is caught up.
    assert clock.sleeps == []
    assert new_deadline == 100.02
    assert stats.deadline_resyncs == 0


def test_pace_resyncs_after_a_long_stall(monkeypatch):
    clock = _FakeClock(start=101.0)  # 1 s past the deadline
    _patch_clock(monkeypatch, clock)
    stats = runtime._LoopStats()
    new_deadline = runtime._pace(deadline=100.0, control_dt=0.02, stats=stats)
    assert clock.sleeps == []
    assert new_deadline == 101.02  # lost ticks abandoned, schedule restarts from now
    assert stats.deadline_resyncs == 1


def test_loop_stats_report_resets_window():
    stats = runtime._LoopStats()
    stats.note(0.004)
    stats.note(0.008)
    line = stats.report(stream_lag=12, midstream_catchups=1)
    assert "stream lag 12 frames" in line
    assert "catchups 1" in line
    assert "8.00 ms max" in line
    line2 = stats.report(stream_lag=None, midstream_catchups=1)
    assert "stream lag - frames" in line2
    assert "0.00 ms max" in line2
