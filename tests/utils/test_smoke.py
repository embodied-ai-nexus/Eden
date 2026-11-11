"""Tests for ``eden.utils.smoke``: ``is_smoke`` and the ``build_and_step`` harness (run on the in-repo reacher)."""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY
from eden.utils.smoke import build_and_step, is_smoke


@pytest.mark.parametrize(
    "eden_smoke, pytest_version, expected",
    [
        ("1", None, True),
        ("true", None, True),
        ("on", None, True),  # any non-empty, non-falsey value
        ("0", None, False),  # a real off-switch, not bool("0")
        ("false", None, False),
        ("No", None, False),  # case-insensitive
        (None, "8.3.0", True),  # unset falls through to PYTEST_VERSION
        (None, None, False),
        ("", "8.3.0", True),  # empty is unset
        ("0", "8.3.0", False),  # an explicit value wins over PYTEST_VERSION
    ],
)
def test_is_smoke_parses_eden_smoke_as_boolean(monkeypatch, eden_smoke, pytest_version, expected):
    for name, value in (("EDEN_SMOKE", eden_smoke), ("PYTEST_VERSION", pytest_version)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert is_smoke() is expected


@pytest.mark.requires_genesis
def test_build_and_step_honours_num_envs_eval_mode_and_modifier_kwargs():
    """Mod kwargs go through ``TASK_REGISTRY.build``: ``difficulty="hard"`` is a 6 s episode."""
    env = build_and_step("reacher", num_envs=2, num_steps=2, eval_mode=True, modifier_kwargs={"difficulty": "hard"})
    assert env.num_envs == 2
    assert env.training is False
    assert env.max_episode_length_s == 6.0


def test_build_and_step_rejects_an_unknown_modifier_kwarg_before_building():
    with pytest.raises(ValueError, match="Unknown modifier kwargs"):
        build_and_step("reacher", modifier_kwargs={"dificulty": "hard"})


def test_build_and_step_rejects_modifier_kwargs_on_a_built_config():
    """A config object is past the point where mods run, so the kwargs must not be dropped silently."""
    with pytest.raises(TypeError, match="modifier_kwargs"):
        build_and_step(TASK_REGISTRY.build("reacher"), modifier_kwargs={"difficulty": "hard"})
