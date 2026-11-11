"""``eden.extensions.teleop.sonic.state_machine``: the operator's mode transitions, their effects and commands."""

from __future__ import annotations

from eden.extensions.teleop.sonic.state_machine import (
    ButtonSnapshot,
    Effect,
    OperatorStateMachine,
    StreamMode,
)
from tests.extensions.teleop.sonic.conftest import engage, press


def test_engage_from_off_calibrates_and_starts_planner():
    sm = OperatorStateMachine()
    t = sm.step(ButtonSnapshot(a=True, b=True, x=True, y=True))
    assert (t.previous, t.mode, t.changed) == (StreamMode.OFF, StreamMode.PLANNER, True)
    assert t.effects == [Effect.CALIBRATE_FULL, Effect.RESET_PLANNER_YAW]
    assert t.command == ("start", True)
    # Holding the combo does not retrigger (rising edge).
    t2 = sm.step(ButtonSnapshot(a=True, b=True, x=True, y=True))
    assert not t2.changed


def test_engage_again_disengages_to_the_planner_without_stopping_the_robot():
    sm = OperatorStateMachine()
    engage(sm)
    t = press(sm, a=True, b=True, x=True, y=True)
    assert t.mode == StreamMode.OFF
    assert t.effects == [Effect.RELEASE_TO_PLANNER]
    assert t.command is None  # no stop: on hardware that is a damping-only command and a fall


def test_planner_pose_toggle():
    sm = OperatorStateMachine()
    engage(sm)
    t = press(sm, a=True, x=True)
    assert t.mode == StreamMode.POSE
    assert t.effects == [Effect.RESET_POSE_YAW]
    assert t.command == ("start", False)
    t = press(sm, a=True, x=True)
    assert t.mode == StreamMode.PLANNER
    assert t.effects == [Effect.POSE_EXIT, Effect.RESET_PLANNER_YAW]
    assert t.command == ("start", True)


def test_pose_frozen_toggle_regrabs_targets():
    sm = OperatorStateMachine()
    engage(sm)
    press(sm, a=True, x=True)  # -> POSE
    t = press(sm, b=True, y=True)
    assert t.mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
    assert t.effects == [Effect.POSE_EXIT, Effect.RESET_PLANNER_YAW, Effect.SAVE_FROZEN_TARGETS]
    t = press(sm, b=True, y=True)
    assert t.mode == StreamMode.POSE
    assert t.effects == [Effect.RESET_POSE_YAW]


def test_menu_pause_is_level_triggered():
    sm = OperatorStateMachine()
    engage(sm)
    press(sm, a=True, x=True)  # -> POSE
    t = sm.step(ButtonSnapshot(left_menu=True))
    assert t.mode == StreamMode.POSE_PAUSE
    assert t.effects == [Effect.POSE_EXIT]
    assert t.command is None  # the robot keeps replaying its last window
    t = sm.step(ButtonSnapshot(left_menu=True))
    assert not t.changed  # still held
    t = sm.step(ButtonSnapshot(left_menu=False))
    assert t.mode == StreamMode.POSE
    assert t.effects == [Effect.RESET_POSE_YAW]
    assert t.command == ("start", False)


def test_vr3pt_remembers_planner_parent():
    sm = OperatorStateMachine()
    engage(sm)
    t = press(sm, left_axis_click=True)
    assert t.mode == StreamMode.PLANNER_VR_3PT
    assert t.effects == [Effect.RECALIBRATE_WRISTS]
    t = press(sm, left_axis_click=True)
    assert t.mode == StreamMode.PLANNER
    assert t.effects == []  # returning keeps the accumulated planner yaw


def test_vr3pt_remembers_frozen_parent_and_regrabs():
    sm = OperatorStateMachine()
    engage(sm)
    press(sm, a=True, x=True)  # POSE
    press(sm, b=True, y=True)  # FROZEN
    t = press(sm, left_axis_click=True)
    assert t.mode == StreamMode.PLANNER_VR_3PT
    t = press(sm, left_axis_click=True)
    assert t.mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
    # Targets are stale after VR_3PT moved the arms: re-grab, but keep yaw.
    assert t.effects == [Effect.SAVE_FROZEN_TARGETS]


def test_vr3pt_exits_to_pose_on_either_chain_combo():
    for combo in ({"a": True, "x": True}, {"b": True, "y": True}):
        sm = OperatorStateMachine()
        engage(sm)
        press(sm, left_axis_click=True)
        t = press(sm, **combo)
        assert t.mode == StreamMode.POSE
        assert t.command == ("start", False)


def test_disengage_works_from_every_live_mode():
    for path in ([], [{"a": True, "x": True}], [{"left_axis_click": True}]):
        sm = OperatorStateMachine()
        engage(sm)
        for combo in path:
            press(sm, **combo)
        t = press(sm, a=True, b=True, x=True, y=True)
        assert t.mode == StreamMode.OFF
        assert Effect.RELEASE_TO_PLANNER in t.effects and t.command is None


def test_force_off_from_pose_clears_stream_without_command():
    sm = OperatorStateMachine()
    engage(sm)
    press(sm, a=True, x=True)
    assert sm.mode == StreamMode.POSE
    t = sm.force_off()
    assert t.previous == StreamMode.POSE and t.mode == StreamMode.OFF and t.changed
    assert t.command is None  # no stop: the robot kept running
    assert Effect.POSE_EXIT in t.effects


def test_force_off_from_planner_has_no_effects():
    sm = OperatorStateMachine()
    engage(sm)
    t = sm.force_off()
    assert t.changed and t.effects == [] and t.command is None


def test_force_off_when_already_off_is_a_no_op():
    sm = OperatorStateMachine()
    t = sm.force_off()
    assert t.changed is False and t.command is None


def test_reengage_after_force_off_recalibrates_fully():
    sm = OperatorStateMachine()
    engage(sm)
    sm.force_off()
    t = press(sm, a=True, b=True, x=True, y=True)
    assert t.mode == StreamMode.PLANNER
    assert Effect.CALIBRATE_FULL in t.effects
    assert t.command == ("start", True)
