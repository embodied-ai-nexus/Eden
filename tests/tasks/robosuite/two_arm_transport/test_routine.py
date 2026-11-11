"""The TwoArmTransport routines: their geometry helpers, and a real two-arm run on the robosuite scene."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import pytest
import torch

from eden.envs.base import env_class_for_config
from eden.extensions.scripted_demos.multi_arm import MultiArmDemonstrator
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.two_arm_transport.routine import _nearest_equivalent, turn_within_window
from tests.support.envs import build_env


def test_an_angle_is_moved_by_whole_periods_to_its_reference():
    assert _nearest_equivalent(0.1, 3.0, math.pi) == pytest.approx(0.1 + math.pi)
    assert _nearest_equivalent(0.1, -1.4, math.pi / 2) == pytest.approx(0.1 - math.pi / 2)


def test_the_hammer_turn_is_cut_to_the_wrists_room_and_never_reversed():
    assert turn_within_window(1.2, 0.0, -1.0, 1.5) == pytest.approx(1.2)  # room to spare
    assert turn_within_window(1.2, 0.5, -1.0, 1.5) == pytest.approx(1.0)  # cut at the window's end
    assert turn_within_window(-1.2, 0.0, -1.0, 1.5) == pytest.approx(-1.0)
    # A wrist already past the end the turn goes toward has no room: no turn, not one the other way.
    assert turn_within_window(1.2, 1.7, -1.0, 1.5) == 0.0
    assert turn_within_window(-1.2, -1.3, -1.0, 1.5) == 0.0


def test_the_demonstrator_refuses_the_parallel_arrangement_it_is_not_written_for():
    config = TASK_REGISTRY.build("robosuite_two_arm_transport", arrangement="parallel")
    unbuilt = env_class_for_config(config).from_config(config)  # the check reads the scene options only
    factory = SCRIPTED_DEMO_REGISTRY.get("robosuite_two_arm_transport")
    with pytest.raises(ValueError, match=r"written for the opposed arrangement.*robot0 stands at \(-0\.56, -0\.6"):
        factory(unbuilt)


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_two_arm_transport")
    config.env_options.num_envs = 2
    config.env_options.episode_length_s = 150.0  # the routines run past the RL episode's 10 s
    return build_env(config)


@pytest.mark.slow
def test_the_two_arms_hand_the_hammer_over_and_finish_the_task(env):
    """Seeded: lid off, trash moved, the hammer handed across the gap and into the target bin, in both envs."""
    env.reset(seed=0)
    demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_two_arm_transport")(env, seed=0)
    assert isinstance(demo, MultiArmDemonstrator)
    demo.reset()
    for _ in range(int(140.0 / env.dt)):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())

    assert demo.done.all()
    assert env.metric_manager.get_value("transported").bool().all()
    for env_id in range(env.num_envs):
        trace = demo.trace(env_id)
        taken = _time_of(trace, "[robot1]", "took the hammer from robot0")
        held_out = _time_of(trace, "[robot0]", "hammer held out")
        released = min(t for t in _times(trace, "[robot0]", "open_gripper()") if t > held_out)
        # robot1 had closed on the hammer before robot0 opened its fingers.
        assert taken <= released


@pytest.mark.slow
def test_robot1_missing_the_trash_ends_both_routines_before_the_hand_off():
    """With the trash out of robot1's reach the episode cannot succeed, so neither arm goes on to the hammer hand-off."""
    config = TASK_REGISTRY.build("robosuite_two_arm_transport")
    config.env_options.num_envs = 2
    config.env_options.episode_length_s = 150.0
    # The trash is put on the floor, which would otherwise end the episode and put it back in its bin.
    config.termination_options.__pydantic_extra__.pop("trash_fallen")
    env = build_env(config)
    env.reset(seed=0)
    out_of_reach = torch.tensor([[0.0, 2.0, 0.05]] * env.num_envs, device=env.device)  # on the floor beyond table1
    env.entities["trash"].set_pos(out_of_reach)
    demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_two_arm_transport")(env, seed=0)
    demo.reset()
    for _ in range(int(140.0 / env.dt)):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())

    assert demo.done.all()
    for env_id in range(env.num_envs):
        trace = "\n".join(demo.trace(env_id))
        assert "[robot1]" in trace and "trash: grasp 3 missed" in trace
        assert "took the hammer from robot0" not in trace
        assert "the other arm gave up" in trace  # robot0 stopped waiting for the hand-off


def _times(trace: list[str], arm: str, text: str) -> list[float]:
    return [float(line.split()[1]) for line in trace if line.startswith(arm) and text in line]


def _time_of(trace: list[str], arm: str, text: str) -> float:
    (time,) = _times(trace, arm, text)
    return time
