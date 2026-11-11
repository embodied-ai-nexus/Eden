"""``ObservationRecorder``: each step's row holds the observations the step's action was chosen from."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden import observations
from eden.managers.recorder_manager import RecorderManager
from eden.managers.terms.recorders import ActionRecorder, ObservationRecorder
from eden.options import BoxOptions, EnvOptions, PlaneOptions, RecorderManagerOptions, SceneOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions
from eden.terms import ObsTerm
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def env():
    """A box dropped from 0.5 m: its height changes every step, so a stale or shifted row shows."""
    height = ObsTerm.configure(func=observations.base_pos, params={"entity_name": "box"})
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                plane=PlaneOptions(), box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))
            ),
            observation_options=ObservationManagerOptions(
                policy=ObservationGroupOptions(pos=height), critic=ObservationGroupOptions(pos=height)
            ),
        ),
        reset=True,
    )


def _recorder(env, tmp_path, **terms) -> RecorderManager:
    return RecorderManager(
        env=env,
        options=RecorderManagerOptions(dataset_export_dir_path=str(tmp_path), dataset_filename="obs", **terms),
    )


def test_each_row_is_the_observation_before_that_step(env, tmp_path):
    built_with = env.recorder_manager
    env.recorder_manager = recorder = _recorder(env, tmp_path, obs=ObservationRecorder.configure(groups=["policy"]))
    try:
        env.reset()
        seen = []
        recorder.start_recording()
        for _ in range(3):
            seen.append(env.obs_buf["policy"][0].clone())
            env.step()
        recorder.end_recording()
        episode = recorder.load_episode("demo_0", device=env.device)
        assert episode.size("obs/policy") == 3
        rows = torch.stack([episode.get("obs/policy", index=t) for t in range(3)])
        torch.testing.assert_close(rows, torch.stack(seen))
        assert episode.get("obs/critic") is None  # only the groups asked for
    finally:
        recorder.close()
        env.recorder_manager = built_with


def test_an_unknown_group_fails_at_build_naming_the_available_ones(env, tmp_path):
    with pytest.raises(KeyError, match=r"\['teacher'\].*available groups: \['policy', 'critic'\]"):
        _recorder(
            env, tmp_path, obs=ObservationRecorder.configure(groups=["teacher"]), actions=ActionRecorder.configure()
        )
