"""The scripted drawer finishes each whiteboard task through the task's own action, and is its datagen demonstrator."""

import h5py
import pytest
import torch

from eden.extensions.scripted_demos.generate import generate_demos
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.whiteboard.scripted_demos import ScriptedDrawer, plan_path
from eden.tasks.whiteboard.strokes import DRAWINGS
from tests.tasks.whiteboard.conftest import build_task


def test_the_plan_lifts_between_strokes_and_presses_along_them():
    path, corners = plan_path(
        DRAWINGS["table"], hover=0.012, press_depth=0.001, draw_speed=0.05, travel_speed=0.1, dt=0.05
    )
    pressing = path[:, 2] < 0
    # Seven strokes: seven touch-downs, each followed by a lift.
    assert int((pressing[1:] & ~pressing[:-1]).sum()) == 7 and int((~pressing[1:] & pressing[:-1]).sum()) == 7
    assert bool(corners[0]) and bool(corners[-1])
    steps = path[1:] - path[:-1]
    assert float(abs(steps[pressing[1:] & pressing[:-1], :2]).max()) <= 0.05 * 0.05 + 1e-9  # drawing speed


@pytest.mark.requires_genesis
@pytest.mark.slow
@pytest.mark.parametrize("task", ["whiteboard_write_text", "whiteboard_write_in_colour", "whiteboard_draw_table"])
def test_the_drawer_succeeds_within_the_episode(task):
    env = build_task(task, num_envs=1)
    drawer = ScriptedDrawer(env)
    drawing = env.metric_manager.get_term("drawing")
    for _ in range(int(env.max_episode_length)):
        score = drawing.score.clone()  # the auto-reset wipes the ink of the step that ends the episode
        _, _, terminated, truncated, _ = env.step(drawer.act())
        if bool(terminated[0] | truncated[0]):
            break
    # The termination flags outlive the auto-reset (the success buffer does not).
    assert bool(env.termination_manager.get_term_dones("drawn")[0]), (
        f"{task}: ended without success at {float(score[0]):.3f}"
    )
    assert not bool(env.termination_manager.get_term_dones("marker_dropped")[0])
    assert torch.isfinite(score).all()


TASKS = ("whiteboard_write_text", "whiteboard_write_in_colour", "whiteboard_draw_table")


@pytest.mark.parametrize("task", TASKS)
def test_each_task_registers_its_drawer_with_its_own_episode_length(task):
    factory = SCRIPTED_DEMO_REGISTRY.get(task)
    assert factory.episode_length_s == TASK_REGISTRY.build(task).env_options.episode_length_s


def test_a_mistake_rate_is_refused_naming_the_task():
    with pytest.raises(ValueError, match=r"'whiteboard_draw_table' makes no deliberate mistakes, so p_mistake=0.1"):
        SCRIPTED_DEMO_REGISTRY.get("whiteboard_draw_table")(None, seed=0, p_mistake=0.1)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_eden_datagen_records_a_successful_drawing(tmp_path):
    config = TASK_REGISTRY.build("whiteboard_write_text")
    config.env_options.num_envs = 1
    report = generate_demos(config, "whiteboard_write_text", num_episodes=1, output_dir=str(tmp_path))

    assert report.episodes == 1 and report.successes == 1 and report.cut_short == 0
    # The episode ends on success: the F1 passes during the second L, with H, E and the first L (6 strokes) done.
    assert report.subgoals["stroke"] >= 6
    with h5py.File(report.dataset_path, "r") as f:
        demo = f["data"][next(iter(f["data"]))]
        assert bool(demo.attrs["success"])
        steps = demo["action"].shape[0]
        assert demo["action"].shape[1] == 6  # the IK delta at the nib: the task has no gripper action
        assert demo["obs/policy"].shape[0] == steps and demo["state/qpos"].shape[0] == steps
