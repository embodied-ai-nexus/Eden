"""Tests for ``RecorderManager``: start/end/close cycles, dataset creation and resumption, the export
modes, and the per-step and post-step recording hooks.

One built ``EnvBase`` drives every test. Each test installs its own ``RecorderManager`` (a real one,
over its own ``tmp_path`` dataset) as the env's recorder, so ``env.step()`` / ``env.reset()`` drive it
exactly as they drive the one the env built.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eden.constants import DatasetExportMode
from eden.managers.recorder_manager import RECORDER_TERM_REGISTRY, RecorderManager, RecorderTerm
from eden.options import BoxOptions, EnvOptions, PlaneOptions, RecorderManagerOptions, SceneOptions
from eden.utils.configs import EdenConfig
from eden.utils.file_handler.base import FILE_HANDLER_REGISTRY
from tests.support.envs import build_env


@RECORDER_TERM_REGISTRY.register()
class MarkerRecorder(RecorderTerm):
    """Records a marker at every post-reset and pre-step collection; counts pre-steps per env."""

    def build(self) -> None:
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, envs_idx=None) -> None:
        self.steps[slice(None) if envs_idx is None else envs_idx] = 0

    def record_post_reset(self, envs_idx=None) -> dict[str, torch.Tensor]:
        n = self.num_envs if envs_idx is None else int(envs_idx.numel())
        return {"reset_marker": torch.ones((n, 1), device=self.device)}

    def record_pre_step(self) -> dict[str, torch.Tensor]:
        self.steps += 1
        return {"step_marker": torch.ones((self.num_envs, 1), device=self.device)}


@RECORDER_TERM_REGISTRY.register()
class PostStepMarker(RecorderTerm):
    def record_post_step(self) -> dict[str, torch.Tensor]:
        return {"post_step_marker": torch.full((self.num_envs, 1), 7.0, device=self.device)}


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                plane=PlaneOptions(), box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.2))
            ),
        ),
        reset=True,
    )


@pytest.fixture
def install(env, tmp_path):
    """Return ``install(**options)`` -> a ``RecorderManager`` over ``tmp_path/dataset``, installed as ``env``'s recorder.

    Every installed manager is closed at teardown and the env's own recorder restored.
    """
    built_with = env.recorder_manager
    installed: list[RecorderManager] = []

    def install_recorder(**options) -> RecorderManager:
        options.setdefault("recorder", MarkerRecorder.configure())
        manager = RecorderManager(
            env=env,
            options=RecorderManagerOptions(
                dataset_export_dir_path=str(tmp_path), dataset_filename="dataset", **options
            ),
        )
        env.recorder_manager = manager
        installed.append(manager)
        env.reset()
        return manager

    yield install_recorder

    try:
        for manager in installed:
            manager.close()
    finally:
        env.recorder_manager = built_with


def _dataset_path(tmp_path: Path) -> Path:
    """Where ``install`` puts the dataset, as the default file handler resolves it."""
    handler = FILE_HANDLER_REGISTRY.get(RecorderManagerOptions().file_handler_options.name)()
    return Path(handler.resolve_path(str(tmp_path / "dataset")))


def _record_one_step(env, recorder: RecorderManager, *, success: bool | None = None) -> None:
    recorder.start_recording()
    env.step()
    if success is not None:
        recorder.set_episode_success(None, torch.full((env.num_envs,), success, dtype=torch.bool))
    recorder.end_recording()


def _assert_every_demo_loads(env, recorder: RecorderManager) -> None:
    for index in range(recorder.demo_count):
        episode = recorder.load_episode(f"demo_{index}", device=env.device)
        assert episode is not None and not episode.is_empty(), f"demo_{index}"


# ---------------------------------------------------------------------------
# Start, end, close
# ---------------------------------------------------------------------------


def test_repeated_start_end_cycles_append_one_demo_each(env, install):
    recorder = install()

    for _ in range(3):
        _record_one_step(env, recorder)

    assert recorder.demo_count == 3
    _assert_every_demo_loads(env, recorder)
    assert recorder.load_episode("demo_2", device=env.device).size("step_marker") == 1


def test_a_second_start_is_ignored(env, install):
    recorder = install()

    recorder.start_recording()
    recorder.start_recording()
    assert recorder.is_recording
    env.step()
    recorder.end_recording()

    assert not recorder.is_recording
    assert recorder.demo_count == 1
    _assert_every_demo_loads(env, recorder)


def test_ending_without_exporting_unfinished_drops_the_episode_in_progress(env, install):
    """A finished episode is kept; the one cut short by ``end_recording(export_unfinished=False)`` is not."""
    recorder = install()

    recorder.start_recording()
    env.step()
    env.reset()  # closes and exports the first episode
    env.step()
    recorder.end_recording(export_unfinished=False)

    assert recorder.demo_count == 1
    assert recorder.load_episode("demo_0", device=env.device).size("step_marker") == 1
    # The dropped episode leaves the recorder clean for the next session.
    _record_one_step(env, recorder)
    assert recorder.demo_count == 2


def test_the_dataset_is_created_only_once_recording_starts(env, install, tmp_path):
    """``end_recording`` without a start, or ``load_episode`` before any, writes nothing."""
    recorder = install()

    recorder.end_recording()
    recorder.end_recording()
    with pytest.raises(RuntimeError, match="not been initialized"):
        recorder.load_episode("demo_0", device=env.device)
    assert (recorder.demo_count, recorder.is_recording) == (0, False)
    assert not _dataset_path(tmp_path).exists()

    _record_one_step(env, recorder)

    assert _dataset_path(tmp_path).exists()
    assert recorder.demo_count == 1


def test_a_reset_while_recording_keeps_the_recorder_usable(env, install):
    recorder = install()

    recorder.start_recording()
    env.step()
    env.reset()
    env.step()
    assert recorder.is_recording
    recorder.end_recording()

    assert recorder.demo_count >= 1
    _assert_every_demo_loads(env, recorder)


def test_close_while_recording_flushes_the_current_episode_and_a_second_close_is_safe(env, install, tmp_path):
    recorder = install()

    recorder.start_recording()
    env.step()
    recorder.close()
    recorder.close()

    assert not recorder.is_recording
    assert _dataset_path(tmp_path).exists()


def test_a_dataset_is_resumed_by_default_and_replaced_with_override(env, install):
    first = install()
    _record_one_step(env, first)
    first.close()

    resumed = install()
    _record_one_step(env, resumed)
    assert resumed.demo_count == 2
    _assert_every_demo_loads(env, resumed)
    resumed.close()

    overridden = install(override=True)
    _record_one_step(env, overridden)
    assert overridden.demo_count == 1
    assert overridden.load_episode("demo_1", device=env.device) is None
    _assert_every_demo_loads(env, overridden)


def test_reset_reaches_every_recorder_term_on_the_given_envs(env, install):
    """Recorder terms get the two-phase lifecycle of every other term: ``reset()`` before ``post_reset()``."""
    recorder = install()
    recorder.start_recording()
    env.step()
    term = recorder.get_term("recorder")
    assert term.steps.tolist() == [1]

    recorder.reset(torch.tensor([0]))

    assert term.steps.tolist() == [0]
    recorder.end_recording()


# ---------------------------------------------------------------------------
# Export modes
# ---------------------------------------------------------------------------


def test_export_none_never_creates_a_dataset(env, install, tmp_path):
    recorder = install(dataset_export_mode=DatasetExportMode.EXPORT_NONE)

    _record_one_step(env, recorder)

    assert recorder.dataset_file_handler is None
    assert recorder.demo_count == 0
    assert not _dataset_path(tmp_path).exists()


def test_export_succeeded_only_drops_failed_episodes(env, install):
    recorder = install(dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY)

    _record_one_step(env, recorder, success=False)
    assert recorder.demo_count == 0
    _record_one_step(env, recorder, success=True)

    assert recorder.demo_count == 1
    assert (recorder.exported_successful_episode_count, recorder.exported_failed_episode_count) == (1, 0)
    _assert_every_demo_loads(env, recorder)


def test_export_to_separate_files_routes_failures_to_their_own_dataset(env, install):
    """Successes go to ``<dataset_filename>``, failures to ``<dataset_filename>_failed``."""
    recorder = install(dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES)

    _record_one_step(env, recorder, success=True)
    _record_one_step(env, recorder, success=False)

    assert (recorder.exported_successful_episode_count, recorder.exported_failed_episode_count) == (1, 1)
    assert recorder.dataset_file_handler.demo_count == 1
    assert recorder.failed_episode_dataset_file_handler.demo_count == 1


def test_success_values_must_align_one_to_one_with_the_envs(env, install):
    recorder = install()
    recorder.start_recording()
    env.step()

    with pytest.raises(ValueError, match="must align 1:1"):
        recorder.set_episode_success(None, torch.tensor([True, False]))
    recorder.end_recording()


# ---------------------------------------------------------------------------
# Post-step recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_post_step_term", [True, False], ids=["opted-in", "absent"])
def test_post_step_keys_come_only_from_an_opted_in_post_step_term(env, install, with_post_step_term):
    """The manager emits no reward/done/timestamp of its own."""
    extra = dict(post_step=PostStepMarker.configure()) if with_post_step_term else {}
    recorder = install(**extra)

    recorder.start_recording()
    env.step()
    env.step()
    recorder.end_recording()

    demo = recorder.load_episode("demo_0", device=env.device)
    assert demo.size("post_step_marker") == (2 if with_post_step_term else None)
    assert demo.size("step_marker") == 2
    assert (demo.size("reward"), demo.size("done"), demo.size("timestamp")) == (None, None, None)
