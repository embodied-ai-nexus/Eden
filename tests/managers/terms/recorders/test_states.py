"""``StateRecorder``: one entity's joint and root state per step, or several entities' concatenated."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.managers.recorder_manager import RecorderManager
from eden.managers.terms.recorders import StateRecorder
from eden.options import BoxOptions, EnvOptions, PlaneOptions, RecorderManagerOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def env():
    """Two free boxes at different heights: a concatenation in the wrong order shows."""
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                low=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.3)),
                high=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.5, 0.0, 0.7)),
            ),
        ),
        reset=True,
    )


def _state(env, tmp_path, entity_name) -> dict[str, torch.Tensor]:
    recorder = RecorderManager(
        env=env,
        options=RecorderManagerOptions(
            dataset_export_dir_path=str(tmp_path),
            dataset_filename="state",
            state=StateRecorder.configure(entity_name=entity_name),
        ),
    )
    return recorder.get_term("state").record_pre_step()["state"]


def test_one_entity_records_its_own_state(env, tmp_path):
    state = _state(env, tmp_path, "low")
    torch.testing.assert_close(state["pos"], env.entities["low"].get_pos())
    assert state["quat"].shape == (2, 4)


def test_several_entities_are_concatenated_in_the_order_named(env, tmp_path):
    state = _state(env, tmp_path, ["high", "low"])
    low, high = env.entities["low"], env.entities["high"]
    torch.testing.assert_close(state["pos"], torch.cat([high.get_pos(), low.get_pos()], dim=-1))
    torch.testing.assert_close(state["quat"], torch.cat([high.get_quat(), low.get_quat()], dim=-1))
    torch.testing.assert_close(state["qpos"], torch.cat([high.get_qpos(), low.get_qpos()], dim=-1))


def test_an_entity_the_scene_lacks_is_refused_when_the_recorder_is_built(env, tmp_path):
    with pytest.raises(KeyError, match="nowhere"):
        _state(env, tmp_path, ["low", "nowhere"])


def test_an_empty_entity_list_is_refused(env, tmp_path):
    with pytest.raises(ValueError, match="lists no entity"):
        _state(env, tmp_path, [])
