"""Tests for ``eden.extensions.scripted_demos.registry``: every demonstrator is keyed by a task that exists."""

from __future__ import annotations

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY


def test_every_demonstrator_is_registered_under_a_task_name():
    """``eden datagen --task X`` looks the demonstrator up by X, so a key no task has is a demonstrator nothing runs."""
    keys = sorted(name for name, _factory in SCRIPTED_DEMO_REGISTRY)
    assert keys, "SCRIPTED_DEMO_REGISTRY is empty: its populated_by modules registered nothing."
    orphans = [name for name in keys if name not in TASK_REGISTRY]
    assert not orphans, (
        f"scripted demonstrators {orphans} are registered under names no task has; register each under its task's "
        f"name exactly as `eden list` prints it (renaming a task means renaming its demonstrator's key too)."
    )
