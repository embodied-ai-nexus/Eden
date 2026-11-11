"""The scripted painter copies each painting through the task's own action, and is its datagen demonstrator."""

import h5py
import numpy as np
import pytest

from eden.extensions.scripted_demos.generate import generate_demos
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.art.painting.scripted_demos import ScriptedPainter
from tests.tasks.art.painting.conftest import build_task

TASKS = ("art_painting_sunset", "art_painting_fuji_study", "art_painting_fuji")


@pytest.mark.parametrize("task", TASKS)
def test_each_task_registers_its_painter_with_its_own_episode_length(task):
    factory = SCRIPTED_DEMO_REGISTRY.get(task)
    assert factory.episode_length_s == TASK_REGISTRY.build(task).env_options.episode_length_s


def test_a_mistake_rate_is_refused_naming_the_task():
    with pytest.raises(ValueError, match=r"'art_painting_sunset' makes no deliberate mistakes, so p_mistake=0.1"):
        SCRIPTED_DEMO_REGISTRY.get("art_painting_sunset")(None, seed=0, p_mistake=0.1)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_painter_succeeds_within_the_episode():
    env = build_task("art_painting_sunset", num_envs=1)
    painter = ScriptedPainter(env)
    painting = env.metric_manager.get_term("painting")
    for _ in range(int(env.max_episode_length)):
        score = painting.score.clone()  # the auto-reset wipes the paint of the step that ends the episode
        _, _, terminated, truncated, _ = env.step(painter.act())
        if bool(terminated[0] | truncated[0]):
            break
    # The termination flags outlive the auto-reset (the success buffer does not).
    assert bool(env.termination_manager.get_term_dones("painted")[0]), f"ended without success at {score}"
    assert painter.subgoal_counts["dip"] >= 3, "a dip for each of the three paints"


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_eden_datagen_records_a_successful_copy(tmp_path):
    config = TASK_REGISTRY.build("art_painting_sunset")
    config.env_options.num_envs = 1
    report = generate_demos(config, "art_painting_sunset", num_episodes=1, output_dir=str(tmp_path))

    assert report.episodes == 1 and report.successes == 1 and report.cut_short == 0
    assert report.subgoals["stroke"] >= 6 and report.subgoals["dip"] >= 3
    with h5py.File(report.dataset_path, "r") as f:
        demo = f["data"][next(iter(f["data"]))]
        assert bool(demo.attrs["success"])
        steps = demo["action"].shape[0]
        assert demo["action"].shape[1] == 6  # the IK delta at the tip: the task has no gripper action
        assert demo["obs/policy"].shape[0] == steps and demo["state/qpos"].shape[0] == steps
        assert np.isfinite(np.asarray(demo["action"])).all()
