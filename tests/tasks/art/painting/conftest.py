"""Builders shared by the painting task tests."""

from __future__ import annotations

import numpy as np
from PIL import Image

from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY


def build_task(name: str, num_envs: int = 2, *, recording: bool = False) -> RLEnvBase:
    """Build and reset the registered painting task ``name`` with ``num_envs`` envs.

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


def picture_file(path) -> str:
    """A 300 x 200 px picture: a red square (60-180, 20-140 px) on blue."""
    pixels = np.zeros((200, 300, 3), dtype=np.uint8)
    pixels[...] = (20, 40, 200)
    pixels[20:140, 60:180] = (220, 30, 30)
    Image.fromarray(pixels).save(path)
    return str(path)
