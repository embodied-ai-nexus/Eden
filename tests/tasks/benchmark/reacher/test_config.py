"""Tests for ``eden.tasks.benchmark.reacher.config``: the ``--difficulty`` presets."""

from __future__ import annotations

import pytest

from eden.tasks import TASK_REGISTRY


@pytest.mark.parametrize(
    ("difficulty", "episode_length_s", "reaching_std", "x_range"),
    [
        ("easy", 12.0, 0.30, (0.25, 0.55)),
        ("medium", 9.0, 0.20, (0.30, 0.50)),
        ("hard", 6.0, 0.10, (0.35, 0.45)),
    ],
)
def test_difficulty_presets(difficulty, episode_length_s, reaching_std, x_range):
    cfg = TASK_REGISTRY.build("reacher", difficulty=difficulty)
    assert cfg.env_options.episode_length_s == episode_length_s
    assert cfg.reward_options.reaching_task.params["reaching_std"] == reaching_std
    assert cfg.command_options.reaching_target.x_range == pytest.approx(x_range, abs=1e-9)


def test_easy_widens_the_lateral_target_range():
    assert TASK_REGISTRY.build("reacher", difficulty="easy").command_options.reaching_target.y_range == pytest.approx(
        (-0.15, 0.15), abs=1e-9
    )


def test_default_build_is_the_medium_preset_and_matches_the_bare_class():
    """``get("reacher")()`` runs no mod, so the class defaults must describe the medium preset too."""
    built = TASK_REGISTRY.build("reacher")
    medium = TASK_REGISTRY.build("reacher", difficulty="medium")
    bare = TASK_REGISTRY.get("reacher")()
    for cfg in (built, bare):
        assert cfg.env_options.episode_length_s == medium.env_options.episode_length_s
        assert cfg.reward_options.reaching_task.params["reaching_std"] == 0.20
        for axis in ("x_range", "y_range", "z_range"):
            assert getattr(cfg.command_options.reaching_target, axis) == pytest.approx(
                getattr(medium.command_options.reaching_target, axis), abs=1e-9
            ), axis


def test_unknown_difficulty_raises():
    with pytest.raises(ValueError, match="difficulty"):
        TASK_REGISTRY.build("reacher", difficulty="impossible")
