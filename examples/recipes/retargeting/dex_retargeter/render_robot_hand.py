r"""Replay a retargeted qpos pickle in the Genesis viewer.

---
tier: recipe
capabilities: [retarget.hand, robot.xhand]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/dex_retargeter/render_robot_hand.py --pickle-path data/qpos.pkl
---

Eden port of ``dex-retargeting``'s ``example/vector_retargeting/render_robot_hand.py``.
Reads a pickle produced by ``detect_from_video.py`` and replays it on the
matching ``RigidEntity``.

Run::

    uv run python examples/recipes/retargeting/dex_retargeter/render_robot_hand.py \\
        --pickle-path data/qpos.pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import torch
import tyro

import eden as en
import genesis as gs
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions

from hand_presets import HAND_PRESETS


def main(
    pickle_path: str,
    output_video_path: str | None = None,
    fps: float = 30.0,
    headless: bool = False,
) -> None:
    """Replay a retargeted qpos sequence.

    Args:
        pickle_path: ``.pkl`` from ``detect_from_video.py``.
        output_video_path: optional ``.mp4`` to record (requires camera setup).
        fps: playback frame rate.
        headless: skip the live viewer (useful when only recording).
    """
    data = pickle.loads(Path(pickle_path).read_bytes())
    qpos_list = np.asarray(data["data"], dtype=np.float32)
    meta = data["meta_data"]
    preset = meta["preset"]

    if preset not in HAND_PRESETS:
        raise SystemExit(f"Unknown preset {preset!r}. Available: {list(HAND_PRESETS)}")

    en.init(backend=gs.cpu)
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=HAND_PRESETS[preset].robot_options_factory(),
            floor=PlaneOptions(),
        ),
        show_viewer=not headless,
    )
    env.build()
    env.reset()

    robot = env.entities["robot"]
    frame_period = 1.0 / fps

    en.logger.info(f"Replaying {len(qpos_list)} frames at {fps:.1f} fps")
    for q in qpos_list:
        q_t = torch.as_tensor(q, dtype=torch.float32, device=robot.device).unsqueeze(0)
        robot.set_qpos(q_t)
        env.scene.visualizer.update()
        time.sleep(frame_period)

    if output_video_path is not None:
        en.logger.warning("Video recording not yet wired up; use a CameraOptions with `start_recording` instead.")


if __name__ == "__main__":
    tyro.cli(main)
