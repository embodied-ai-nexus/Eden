"""Routine demonstrators on the real robosuite ``Lift`` scene: motions, their outcomes, the trace, and the privilege line.

One ``robosuite_lift`` build (two envs) serves the privileged tests, each resetting it; the camera
tests build a second one that renders every env.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import numpy as np
import pytest
import torch

from eden.extensions.scripted_demos.generate import GenerationReport, _keep_failure_trace
from eden.extensions.scripted_demos.primitives import effector_rot
from eden.extensions.scripted_demos.routines import (
    ROTATION_TOLERANCE,
    RoutineDemonstrator,
    _angle_between,
    _reachable_heading,
)
from eden.extensions.scripted_demos.tools import BLOCKED, CONTACT, NO_CONTACT, REACHED
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.common import TABLE_TOP_Z
from eden.utils.geom import yaw_from_quat
from tests.support.envs import build_env

WORKSPACE = ((-0.35, -0.35, TABLE_TOP_Z + 0.005), (0.35, 0.35, TABLE_TOP_Z + 0.45))


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_lift")
    config.env_options.num_envs = 2
    # _run does not re-start a routine the env auto-resets; the task's 10 s RL episode would reset the
    # scene under the longer routines here (a pick, a lift and a 6 s guarded descent end past 10 s).
    config.env_options.episode_length_s = 60.0
    return build_env(config, reset=True)


@pytest.fixture(scope="module")
def camera_env():
    config = TASK_REGISTRY.build("robosuite_lift")
    config.env_options.num_envs = 2
    config.env_options.env_separate_rigid = True
    return build_env(config, reset=True)


def _run(env, routine, *, max_steps=600, **kwargs) -> RoutineDemonstrator:
    """Reset ``env`` and run ``routine`` in every env until all have returned (or ``max_steps``)."""
    env.reset()
    demo = RoutineDemonstrator(env, routine, workspace=WORKSPACE, **kwargs)
    demo.reset()
    for _ in range(max_steps):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())
    return demo


def pick_and_lift(tools, look):
    cube = look(tools)
    yaw = cube.yaw + math.pi / 2 - tools.finger_axis_yaw
    yield tools.open_gripper()
    yield tools.move_to(cube.top_center + (0.0, 0.0, 0.1), yaw)
    yield tools.move_to((*cube.top_center[:2], cube.top_z - 0.02), yaw)
    yield tools.close_gripper()
    yield tools.move_by((0.0, 0.0, 0.15))
    tools.note(f"holding={tools.holding()}" + (f" grasped={tools.grasped('cube')}" if tools.privileged else ""))


# --- the rig ----------------------------------------------------------------------------------------


def test_the_panda_closes_its_fingers_across_the_effector_heading(env):
    demo = RoutineDemonstrator(env, pick_and_lift, workspace=WORKSPACE)
    # robosuite's Panda hand: heading 0 has the fingers' line along world y.
    assert demo.driver.finger_axis_yaw == pytest.approx(math.pi / 2, abs=0.05)
    assert demo.driver.tip_reach == pytest.approx(0.0103, abs=0.002)


# --- motions and outcomes ---------------------------------------------------------------------------


def test_a_privileged_pick_and_lift_lifts_the_cube_in_every_env(env):
    demo = _run(env, lambda tools: pick_and_lift(tools, lambda t: t.object_footprint("cube")))

    assert demo.done.all()
    assert env.metric_manager.success_buf.all()
    heights = env.entities["cube"].get_pos()[:, 2] - TABLE_TOP_Z
    assert (heights > 0.1).all(), heights
    for env_id in range(2):
        trace = demo.trace(env_id)
        assert trace[0].endswith("open_gripper() -> done in 0.50 s")
        assert any("move_by(0.000, 0.000, 0.150) -> reached" in line for line in trace)
        assert trace[-1].endswith("# holding=True grasped=True")
    assert demo.subgoal_counts == {
        "open_gripper:done": 2,
        "move_to:reached": 4,
        "close_gripper:done": 2,
        "move_by:reached": 2,
    }


def test_a_guarded_descent_stops_when_the_closed_fingers_meet_the_cube(env):
    outcomes = {}

    def press(tools):
        cube = tools.object_footprint("cube")
        yield tools.close_gripper()
        yield tools.move_to(cube.top_center + (0.0, 0.0, 0.06))
        outcomes[tools.env_id] = (yield tools.move_until_contact(max_distance=0.1, speed=0.03)), cube.top_z

    _run(env, press)
    for outcome, top_z in outcomes.values():
        assert outcome.status == CONTACT
        # Stopped on the cube's top, short of the 10 cm asked for (6 cm of air, then the cube).
        assert outcome.elapsed_s < 0.1 / 0.03
        assert outcome.error_m > 0.03


def test_a_guarded_move_into_free_space_ends_without_contact(env):
    outcomes = {}

    def rise(tools):
        outcomes[tools.env_id] = yield tools.move_until_contact((0.0, 0.0, 1.0), max_distance=0.03, speed=0.05)

    _run(env, rise)
    assert {outcome.status for outcome in outcomes.values()} == {NO_CONTACT}


def test_a_guarded_move_held_back_by_the_workspace_floor_ends_blocked(env):
    outcomes = {}

    def dig(tools):
        # Over bare table, so no finger touches anything: the workspace floor stops the hand.
        yield tools.move_to((0.2, 0.2, TABLE_TOP_Z + 0.05))
        outcomes[tools.env_id] = yield tools.move_until_contact(max_distance=0.2, speed=0.05)

    _run(env, dig)
    assert {outcome.status for outcome in outcomes.values()} == {BLOCKED}


def test_a_target_outside_the_workspace_is_clipped_and_the_trace_says_so(env):
    outcomes = {}

    def overreach(tools):
        outcomes[tools.env_id] = yield tools.move_to((0.0, 0.0, TABLE_TOP_Z + 1.0))

    demo = _run(env, overreach)
    assert {outcome.status for outcome in outcomes.values()} == {REACHED}
    assert "clipped to the workspace: (0.000, 0.000, 1.250)" in demo.trace(0)[0]


def test_fingers_closed_on_nothing_are_not_holding(env):
    seen = {}

    def grab_air(tools):
        yield tools.close_gripper()
        seen[tools.env_id] = (tools.holding(), tools.grasped("cube"), tools.gripper_closure())

    _run(env, grab_air)
    for holding, grasped, closure in seen.values():
        assert not holding
        assert not grasped
        assert closure > 0.95


def test_lowering_a_held_cube_is_not_stopped_by_the_touch_it_started_with(env):
    outcomes = {}

    def set_down(tools):
        yield from pick_and_lift(tools, lambda t: t.object_footprint("cube"))
        # The fingers touch the cube throughout: only its landing stops the hand — held back by it
        # (blocked), or, if the cube slides up between the pads, by the fingertips meeting the table.
        outcome = yield tools.move_until_contact(max_distance=0.3, speed=0.05)
        outcomes[tools.env_id] = (outcome, tools.object_footprint("cube").bottom_z)

    _run(env, set_down)
    for outcome, cube_bottom in outcomes.values():
        assert outcome.status in (BLOCKED, CONTACT)
        assert outcome.elapsed_s > 2.0  # ~13 cm of descent at 5 cm/s, not stopped on its first step
        assert cube_bottom == pytest.approx(TABLE_TOP_Z, abs=0.005)


def test_a_wait_after_a_move_keeps_closing_on_the_moves_target(env):
    errors = {}
    target = np.array([0.05, 0.05, TABLE_TOP_Z + 0.1])

    def settle(tools):
        # A loose tolerance ends the move with the hand still short of its target ...
        yield tools.move_to(target, tolerance=0.03)
        ended = np.linalg.norm(tools.tool_pos() - target)
        # ... and the wait after it holds the target, not the spot the hand had got to.
        yield tools.wait(1.5)
        errors[tools.env_id] = ended, np.linalg.norm(tools.tool_pos() - target)

    _run(env, settle)
    for ended, settled in errors.values():
        assert settled < 0.004
        assert settled < ended


def test_a_move_to_a_tilted_rotation_reaches_it(env):
    """``move_to(rot=...)``: a hand leaning 30 degrees off vertical, judged on the full rotation, not the heading."""
    reached = {}

    def lean(tools):
        # Leaning sideways: leaning back toward the arm's base at this height is out of its reach.
        lean_angle = math.radians(30.0)
        rot = tools.aim((0.0, math.sin(lean_angle), -math.cos(lean_angle)), (1.0, 0.0, 0.0))
        outcome = yield tools.move_to((0.0, 0.0, TABLE_TOP_Z + 0.15), rot=rot, timeout_s=3.0)
        reached[tools.env_id] = outcome.status, rot

    demo = _run(env, lean)
    _, tool_rot = demo.driver.tool_pose()
    for env_id, (status, rot) in reached.items():
        assert status == REACHED
        angle = _angle_between(tool_rot[env_id : env_id + 1], torch.tensor([rot], device=tool_rot.device))
        assert float(angle) <= ROTATION_TOLERANCE


def test_tool_rotation_reads_the_whole_rotation_a_tilted_move_left_the_hand_in(env):
    """What a routine measures a held object against: the full effector rotation, not only its heading."""
    readings = {}

    def lean(tools):
        rot = tools.aim((0.0, math.sin(math.radians(30.0)), -math.cos(math.radians(30.0))), (1.0, 0.0, 0.0))
        yield tools.move_to((0.0, 0.0, TABLE_TOP_Z + 0.15), rot=rot, timeout_s=3.0)
        readings[tools.env_id] = tools.tool_rotation(), rot

    _run(env, lean)
    for reading, rot in readings.values():
        reading = torch.tensor([reading])
        assert float(_angle_between(reading, torch.tensor([rot], dtype=torch.float32))) <= ROTATION_TOLERANCE
        # The lean is in it, not only the heading: 30 degrees off the straight-down pose at that heading.
        straight_down = effector_rot(yaw_from_quat(reading))
        assert float(_angle_between(reading, straight_down)) == pytest.approx(math.radians(30.0), abs=0.1)


def test_link_pose_reads_an_entity_where_the_simulator_has_it_in_its_authored_frame(env):
    poses = {}

    def look(tools):
        poses[tools.env_id] = tools.link_pose("cube")
        return
        yield  # a generator that ends at its first resume: the read and the check see the same step

    _run(env, look)
    cube = env.entities["cube"]
    for env_id, (pos, quat) in poses.items():
        np.testing.assert_allclose(pos, cube.get_pos(relative=True)[env_id].cpu().numpy(), atol=1e-6)
        np.testing.assert_allclose(quat, cube.get_quat(relative=True)[env_id].cpu().numpy(), atol=1e-6)


def test_a_heading_is_turned_the_short_way_inside_the_wrists_range():
    window = torch.tensor([[-2.5, 2.6], [-2.5, 2.6]])
    # Asked for -1.5 (or its half-turn twin 1.64) from 2.0: the twin is 0.36 rad away, -1.5 is 3.5.
    heading = _reachable_heading(torch.tensor([-1.5, -1.5]), torch.tensor([2.0, -1.0]), window)
    torch.testing.assert_close(heading, torch.tensor([-1.5 + math.pi, -1.5]))
    # A twin outside the window loses to a farther one inside it.
    heading = _reachable_heading(torch.tensor([2.9]), torch.tensor([2.5]), window[:1])
    torch.testing.assert_close(heading, torch.tensor([2.9 - math.pi]))


# --- what a routine may get wrong -------------------------------------------------------------------


def test_a_routine_naming_an_entity_the_scene_lacks_is_refused_before_it_runs(env):
    def look(tools):
        yield tools.wait(0.1)

    look.entity_names = ("cube", "bowl")
    with pytest.raises(KeyError, match=r"entities \['bowl'\], which are not in the scene; its entities: \[.*'cube'"):
        RoutineDemonstrator(env, look, workspace=WORKSPACE)


def test_a_routine_that_yields_something_else_is_refused_naming_it(env):
    def bad(tools):
        yield (0.1, 0.2, 0.3)

    with pytest.raises(TypeError, match=r"yielded \(0.1, 0.2, 0.3\) in env 0.*tools.move_to"):
        _run(env, bad)


def test_an_error_inside_a_routine_is_raised_with_its_env_and_trace(env):
    def clumsy(tools):
        yield tools.wait(0.2)
        raise ValueError("lost the cube")

    with pytest.raises(RuntimeError, match=r"raised in env 0 at t=0.2. s: ValueError: lost the cube.*wait\(0.20\)"):
        _run(env, clumsy)


def test_the_first_failed_episode_trace_is_kept_for_the_report(env):
    demo = _run(env, lambda tools: pick_and_lift(tools, lambda t: t.object_footprint("cube")))
    report = GenerationReport(dataset_path="unused")
    _keep_failure_trace(report, demo, torch.tensor([False, True]))
    _keep_failure_trace(report, demo, torch.tensor([True, False]))  # one is enough
    assert report.failure_trace[0] == "env 1:"
    assert report.failure_trace[1:] == demo.trace(1)


def test_the_heading_window_brackets_the_hand_and_spans_the_wrists_range(env):
    windows = {}

    def look(tools):
        windows[tools.env_id] = (tools.heading_window(), tools.tool_yaw())
        yield tools.wait(0.1)

    _run(env, look)
    for (low, high), heading in windows.values():
        # The window may run past +-pi; the heading is read wrapped.
        assert any(low < heading + k * 2 * math.pi < high for k in (-1, 0, 1))
        # The Panda's last joint turns +-2.9 rad, less the planner's margin at either stop.
        assert high - low == pytest.approx(2 * (2.8973 - 0.35), abs=0.3)


# --- the privilege line and the cameras -------------------------------------------------------------


@pytest.mark.parametrize(
    ("reading", "alternative"),
    [("object_footprint", r"locate\('cube'"), ("grasped", r"holding\(\)"), ("link_pose", r"locate\('cube'")],
)
def test_a_camera_only_routine_is_refused_simulator_state_naming_what_to_measure_instead(env, reading, alternative):
    def peek(tools):
        getattr(tools, reading)("cube")
        yield tools.wait(0.1)

    with pytest.raises(
        RuntimeError, match=rf"PrivilegedToolError: {reading}\(\) reads the simulator's state.*{alternative}"
    ):
        _run(env, peek, privileged=False)


def test_a_camera_the_scene_lacks_is_refused_naming_its_cameras(env):
    with pytest.raises(KeyError, match=r"Cameras \['wrist'\] are not in the scene; its cameras: \['agentview'\]"):
        RoutineDemonstrator(env, pick_and_lift, workspace=WORKSPACE, cameras=("wrist",))


def test_a_camera_rendering_one_view_for_every_env_is_refused_with_the_fix(env):
    with pytest.raises(ValueError, match=r"renders 1 view\(s\) for 2 envs.*env_separate_rigid: true"):
        RoutineDemonstrator(env, pick_and_lift, workspace=WORKSPACE, cameras=("agentview",))


def test_locate_sees_the_cube_where_the_simulator_has_it(camera_env):
    measured = {}

    def compare(tools):
        yield tools.wait(0.1)
        measured[tools.env_id] = (tools.locate("cube"), tools._executor.object_footprint("cube", tools.env_id))

    _run(camera_env, compare, privileged=False, cameras=("agentview",))
    for seen, truth in measured.values():
        assert seen is not None
        # The camera sees the top and the sides facing it: the centre in xy and the top agree to a few mm.
        np.testing.assert_allclose(seen.center[:2], truth.center[:2], atol=0.005)
        assert seen.top_z == pytest.approx(truth.top_z, abs=0.003)


def test_a_camera_only_pick_and_lift_lifts_the_cube_in_every_env(camera_env):
    demo = _run(
        camera_env,
        lambda tools: pick_and_lift(tools, lambda t: t.locate("cube")),
        privileged=False,
        cameras=("agentview",),
    )
    assert camera_env.metric_manager.success_buf.all()
    assert all(demo.trace(env_id)[-1].endswith("# holding=True") for env_id in range(2))
