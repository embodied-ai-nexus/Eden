"""Shared helpers for the audio sensor tests: a scripted source, the scene's registry, and build-time log capture."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import genesis as gs
import pytest
import torch

from eden.engine.sensors import AudioContext, AudioSource


class ProbeSource(AudioSource):
    """An :class:`AudioSource` that emits a scripted value per simulation step, so a microphone's read names the step.

    ``value_per_step[i]`` fills every sample of the ``i``-th block; past the script it emits zeros. With
    ``link_idx=-1`` the source is static at world ``pos``; otherwise ``pos`` is in that link's frame.
    ``step_motion`` is a world displacement added on every emit, i.e. once per simulation step.
    """

    def __init__(self, env, value_per_step, pos=(0.0, 0.0, 0.0), audio_substeps=20, step_motion=None, link_idx=-1):
        self._step_motion = step_motion
        self._values = list(value_per_step)
        self._k = audio_substeps
        self.emit_count = 0
        B = env.num_envs
        self._block_t = torch.zeros((B, 1, audio_substeps), dtype=gs.tc_float, device=env.device)
        self._links_t = torch.full((1,), link_idx, dtype=gs.tc_int, device=env.device)
        self._offset_t = torch.tensor([[list(pos)]], dtype=gs.tc_float, device=env.device).expand(B, 1, 3).contiguous()

    @property
    def block(self):
        return self._block_t

    @property
    def emit_links(self):
        return self._links_t

    @property
    def emit_offset(self):
        return self._offset_t

    @property
    def audio_substeps(self) -> int:
        return self._k

    def reset(self, envs_idx):
        self._block_t[slice(None) if envs_idx is None else envs_idx] = 0.0

    def emit(self):
        self._block_t[:] = self._values[self.emit_count] if self.emit_count < len(self._values) else 0.0
        if self._step_motion is not None:
            self._offset_t += torch.tensor(self._step_motion, dtype=gs.tc_float, device=self._offset_t.device)
        self.emit_count += 1


def _audio_context(env) -> AudioContext:
    # The registry has no public handle: SensorManager holds the one instance per scene.
    return env.scene.sim._sensor_manager._shared_contexts[AudioContext]


class _MessageList(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _eden_messages():
    """Collect ``en.logger`` messages; usable where the function-scoped ``eden_caplog`` is not (module fixtures)."""
    logger = logging.getLogger("eden")
    handler, level = _MessageList(), logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


@pytest.fixture(scope="session")
def probe_source():
    """The :class:`ProbeSource` class."""
    return ProbeSource


@pytest.fixture(scope="session")
def audio_context():
    """``audio_context(env)``: the scene's :class:`AudioContext`."""
    return _audio_context


@pytest.fixture(scope="session")
def eden_messages():
    """``with eden_messages() as messages:`` collects ``en.logger`` output, e.g. around a module-scoped build."""
    return _eden_messages
