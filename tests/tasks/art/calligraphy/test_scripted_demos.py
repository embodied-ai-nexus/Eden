"""The scripted writer copies each work through the task's own action, and is its datagen demonstrator."""

import h5py
import numpy as np
import pytest
import torch

from eden.extensions.scripted_demos.generate import generate_demos
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.art.calligraphy.scripted_demos import ScriptedWriter
from eden.tasks.art.calligraphy.works import WORKS
from eden.tasks.art.plans import TRAVEL_HEIGHT
from eden.tasks.art.scripted import plan_timeline
from tests.tasks.art.calligraphy.conftest import build_task

TASKS = ("art_calligraphy_dai", "art_calligraphy_san", "art_calligraphy_ei", "art_calligraphy_lettering_minimum")


def test_the_timeline_writes_slowly_travels_high_and_ends_at_home():
    path, stroke_of = WORKS["san"].plan.path()
    goal, writing, _ = plan_timeline(
        path, stroke_of, draw_speed=0.02, travel_speed=0.08, dt=0.05, home=(0.0, 0.0, -0.03), turn_radius=0.005
    )
    assert np.allclose(goal[0], (0.0, 0.0, -0.03)) and np.allclose(goal[-1], (0.0, 0.0, -0.03))
    steps = np.linalg.norm(np.diff(goal, axis=0), axis=1)
    assert steps[writing[1:] & writing[:-1]].max() <= 0.02 * 0.05 + 1e-9  # at most the drawing speed
    assert steps.max() <= 0.08 * 0.05 + 1e-9
    assert goal[~writing, 2].max() < 0  # off a stroke, the tip is in the air
    assert goal[:, 2].min() <= -TRAVEL_HEIGHT + 1e-9
    # Three strokes: three runs of writing steps.
    assert int((writing[1:] & ~writing[:-1]).sum()) == 3


@pytest.mark.parametrize("task", TASKS)
def test_each_task_registers_its_writer_with_its_own_episode_length(task):
    factory = SCRIPTED_DEMO_REGISTRY.get(task)
    assert factory.episode_length_s == TASK_REGISTRY.build(task).env_options.episode_length_s


def test_a_mistake_rate_is_refused_naming_the_task():
    with pytest.raises(ValueError, match=r"'art_calligraphy_ei' makes no deliberate mistakes, so p_mistake=0.1"):
        SCRIPTED_DEMO_REGISTRY.get("art_calligraphy_ei")(None, seed=0, p_mistake=0.1)


@pytest.mark.requires_genesis
@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_the_writer_succeeds_within_the_episode(task):
    env = build_task(task, num_envs=1)
    writer = ScriptedWriter(env)
    writing = env.metric_manager.get_term("writing")
    for _ in range(int(env.max_episode_length)):
        score = writing.score.clone()  # the auto-reset wipes the ink of the step that ends the episode
        _, _, terminated, truncated, _ = env.step(writer.act())
        if bool(terminated[0] | truncated[0]):
            break
    # The termination flags outlive the auto-reset (the success buffer does not).
    assert bool(env.termination_manager.get_term_dones("written")[0]), f"{task}: ended without success at {score}"
    assert torch.isfinite(score).all()


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_eden_datagen_records_a_successful_copy(tmp_path):
    config = TASK_REGISTRY.build("art_calligraphy_dai")
    config.env_options.num_envs = 1
    report = generate_demos(config, "art_calligraphy_dai", num_episodes=1, output_dir=str(tmp_path))

    assert report.episodes == 1 and report.successes == 1 and report.cut_short == 0
    assert report.subgoals["stroke"] >= 2  # success can come during the last stroke
    with h5py.File(report.dataset_path, "r") as f:
        demo = f["data"][next(iter(f["data"]))]
        assert bool(demo.attrs["success"])
        steps = demo["action"].shape[0]
        assert demo["action"].shape[1] == 6  # the IK delta at the tip: the task has no gripper action
        assert demo["obs/policy"].shape[0] == steps and demo["state/qpos"].shape[0] == steps
