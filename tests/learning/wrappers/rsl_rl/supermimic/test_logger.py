"""The run logger's TensorBoard output."""

import contextlib
import io

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from eden.learning.wrappers.rsl_rl.supermimic.logger import RunLogger
from eden.learning.wrappers.rsl_rl.supermimic.options import SuperMimicRunnerOptions


def test_every_scalar_of_an_iteration_reaches_tensorboard_at_its_step(tmp_path):
    options = SuperMimicRunnerOptions(experiment_name="test", logger="tensorboard")
    logger = RunLogger(str(tmp_path), options, num_envs=4, device=torch.device("cpu"))
    logger.open()
    with contextlib.redirect_stdout(io.StringIO()):
        for it in range(3):
            logger.process_env_step(
                torch.ones(4), torch.tensor([0, 1, 0, 0]), {"log": {"Episode_Reward/a": torch.tensor(0.5 + it)}}
            )
            logger.log(it, 3, 40, 0.5, 0.25, {"policy/reward": torch.tensor(1.5 * it)}, normalized_return=0.75)
    logger.close()
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    read = {tag: [(e.step, e.value) for e in events.Scalars(tag)] for tag in events.Tags()["scalars"]}
    assert read == {
        "Loss/policy/reward": [(0, 0.0), (1, 1.5), (2, 3.0)],
        "Episode_Reward/a": [(0, 0.5), (1, 1.5), (2, 2.5)],
        "Train/mean_reward": [(0, 1.0), (1, 1.0), (2, 1.0)],
        "Train/mean_episode_length": [(0, 1.0), (1, 1.0), (2, 1.0)],
        "Train/mean_normalized_return": [(0, 0.75), (1, 0.75), (2, 0.75)],
        "Perf/total_fps": [(0, 53.0), (1, 53.0), (2, 53.0)],
        "Perf/collection_time": [(0, 0.5), (1, 0.5), (2, 0.5)],
        "Perf/learning_time": [(0, 0.25), (1, 0.25), (2, 0.25)],
    }
