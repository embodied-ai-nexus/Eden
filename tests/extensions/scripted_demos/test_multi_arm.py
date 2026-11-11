"""Multi-arm demonstrators on the real two-arm robosuite ``TwoArmTransport`` scene: composition, the shared blackboard, the merged trace."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import re

import pytest

from eden.extensions.scripted_demos.multi_arm import MultiArmDemonstrator
from eden.extensions.scripted_demos.routines import RoutineDemonstrator
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.common import TABLE_TOP_Z
from tests.support.envs import build_env

WORKSPACES = {
    "robot0": ((-0.45, -0.75, TABLE_TOP_Z + 0.005), (0.45, 0.2, 1.3)),
    "robot1": ((-0.45, -0.2, TABLE_TOP_Z + 0.005), (0.45, 0.75, 1.3)),
}


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_two_arm_transport")
    config.env_options.num_envs = 2
    return build_env(config, reset=True)


def _run(demo, env, *, max_steps=400):
    demo.reset()
    for _ in range(max_steps):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())


def signal_then_rise(tools):
    """robot0: rise, then tell robot1 how high it got."""
    yield tools.move_by((0.0, 0.0, 0.05))
    tools.shared["robot0_z"] = float(tools.tool_pos()[2])
    tools.note("signalled")


def wait_then_rise(tools):
    """robot1: wait for robot0's signal, then rise as well."""
    while "robot0_z" not in tools.shared:
        yield tools.wait(0.1)
    tools.note(f"saw robot0 at {tools.shared['robot0_z']:.3f}")
    yield tools.move_by((0.0, 0.0, 0.05))


def test_a_single_arm_demonstrator_on_a_two_arm_task_is_refused_naming_the_other_arms_terms(env):
    with pytest.raises(TypeError, match=r"Action terms \['robot1_arm', 'robot1_gripper'\] drive something other"):
        RoutineDemonstrator(env, signal_then_rise, rig="robot0", workspace=WORKSPACES["robot0"])


def test_the_arms_of_an_env_share_one_blackboard_and_hand_off_through_it(env):
    env.reset()
    demo = MultiArmDemonstrator(env, {"robot0": signal_then_rise, "robot1": wait_then_rise}, workspace=WORKSPACES)
    _run(demo, env)

    assert demo.done.all()
    for env_id in range(env.num_envs):
        trace = demo.trace(env_id)
        signalled = next(i for i, line in enumerate(trace) if "signalled" in line)
        seen = next(i for i, line in enumerate(trace) if "saw robot0" in line)
        # robot1 waited for robot0's key: it saw it after robot0 wrote it, and every line names its arm.
        assert signalled < seen
        assert all(re.match(r"\[robot[01]\] ", line) for line in trace)
        times = [float(re.match(r"\[robot[01]\]\s+(\d+\.\d+) s", line).group(1)) for line in trace]
        assert times == sorted(times)
    assert demo.subgoal_counts["robot0/move_by:reached"] == env.num_envs
    assert demo.subgoal_counts["robot1/move_by:reached"] == env.num_envs
    assert demo.subgoal_counts["robot1/wait:done"] >= env.num_envs


def test_a_reset_gives_each_env_a_fresh_blackboard_both_arms_see(env):
    seen = {}

    def write(tools):
        tools.shared["episode"] = tools.shared.get("episode", 0) + 1
        yield tools.wait(0.1)

    def read(tools):
        yield tools.wait(0.2)
        seen.setdefault(tools.env_id, []).append(tools.shared["episode"])

    env.reset()
    demo = MultiArmDemonstrator(env, {"robot0": write, "robot1": read}, workspace=WORKSPACES)
    _run(demo, env)
    _run(demo, env)
    # Each episode's board starts empty (robot0 counts 1, not 2) and is the one robot1 reads.
    assert seen == {0: [1, 1], 1: [1, 1]}


def test_the_action_covers_every_arms_ik_and_gripper_term(env):
    env.reset()
    demo = MultiArmDemonstrator(env, {"robot0": signal_then_rise, "robot1": wait_then_rise}, workspace=WORKSPACES)
    demo.reset()
    action = demo.act()
    assert set(action) == set(env.action_manager.active_terms)
    assert {name: tuple(value.shape) for name, value in action.items()} == {
        "robot0_arm": (2, 6),
        "robot0_gripper": (2, 1),
        "robot1_arm": (2, 6),
        "robot1_gripper": (2, 1),
    }


def test_one_arm_is_a_routine_demonstrator_not_a_multi_arm_one(env):
    with pytest.raises(ValueError, match=r"drives two or more arms.*RoutineDemonstrator\(env, routine, rig='<slot>'\)"):
        MultiArmDemonstrator(env, {"robot0": signal_then_rise}, workspace=WORKSPACES["robot0"])


def test_a_per_arm_option_must_name_every_arm(env):
    with pytest.raises(
        ValueError, match=r"Per-arm option 'workspace' is given for \['robot0'\].*\['robot0', 'robot1'\]"
    ):
        MultiArmDemonstrator(
            env,
            {"robot0": signal_then_rise, "robot1": wait_then_rise},
            workspace={"robot0": WORKSPACES["robot0"]},
        )
