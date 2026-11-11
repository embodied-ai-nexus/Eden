"""Builders shared by the calligraphy task tests."""

from __future__ import annotations

from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY


def build_task(name: str, num_envs: int = 2, *, recording: bool = False) -> RLEnvBase:
    """Build and reset the registered calligraphy task ``name`` with ``num_envs`` envs.

    ``recording`` adds the ``rec`` camera as ``eden inference --rec`` does.
    """
    config = TASK_REGISTRY.build(name)
    config.env_options.num_envs = num_envs
    if recording:
        config.add_recording_camera()
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    return env
