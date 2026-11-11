r"""Offline vector retargeting: MP4 -> robot qpos trajectory.

---
tier: recipe
capabilities: [retarget.hand, robot.xhand]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/dex_retargeter/detect_from_video.py --video-path /path/to/clip.mp4 --output-path data/qpos.pkl --hand-type right
---

Eden port of ``dex-retargeting``'s ``example/vector_retargeting/detect_from_video.py``.
MediaPipe detects 21 hand keypoints per frame; the Eden ``DexRetargeter`` then
maps the keypoint vectors onto an Eden ``RigidEntity``'s joint angles.

Run (after ``uv pip install -e '.[dex-retargeter-example]'``)::

    uv run python examples/recipes/retargeting/dex_retargeter/detect_from_video.py \\
        --video-path /path/to/clip.mp4 \\
        --output-path data/qpos.pkl \\
        --hand-type right

The output pickle has ``{"data": [qpos, ...], "meta_data": {...}}``. Replay
with ``examples/recipes/retargeting/dex_retargeter/render_robot_hand.py``.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import tqdm
import tyro

import eden as en
import genesis as gs
from eden.envs.base import EnvBase
from eden.extensions.retargeting.dex_retargeter import DexRetargeter
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions

from hand_presets import HAND_PRESETS
from single_hand_detector import SingleHandDetector


def retarget_video(
    retargeter: DexRetargeter,
    video_path: str,
    output_path: str,
    hand_type: str,
    preset: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")

    detector = SingleHandDetector(hand_type=hand_type.capitalize(), selfie=False)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_link_human_indices = retargeter.optimizer.target_link_human_indices
    retargeting_type = retargeter.retargeting_type

    data: list[np.ndarray] = []
    tic = time.perf_counter()
    with tqdm.tqdm(total=length) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = frame[..., ::-1]
            num_box, joint_pos, _, _ = detector.detect(rgb)
            if num_box == 0:
                pbar.update(1)
                continue

            ref_value = DexRetargeter.extract_ref_value(joint_pos, target_link_human_indices, retargeting_type)
            qpos = retargeter.retarget(ref_value)
            data.append(qpos)
            pbar.update(1)

    cap.release()
    elapsed = time.perf_counter() - tic
    en.logger.info(f"Retargeted {len(data)} frames in {elapsed:.2f}s")

    meta = dict(
        preset=preset,
        hand_type=hand_type,
        dof=len(retargeter.joint_names),
        joint_names=retargeter.joint_names,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta), f)
    en.logger.info(f"Wrote {output} ({len(data)} frames)")


def main(
    video_path: str,
    output_path: str,
    preset: str = "xhand1_right_dexpilot",
    hand_type: str = "right",
) -> None:
    """Offline vector retargeting from an MP4.

    Args:
        video_path: input .mp4 file.
        output_path: output pickle path.
        preset: which preset in :mod:`examples.dex_retargeter.hand_presets`.
        hand_type: ``"left"`` or ``"right"``.
    """
    if preset not in HAND_PRESETS:
        raise SystemExit(f"Unknown preset {preset!r}. Available: {list(HAND_PRESETS)}")

    en.init(backend=gs.cpu)
    spec = HAND_PRESETS[preset]

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=spec.robot_options_factory(),
            floor=PlaneOptions(),
        ),
        show_viewer=False,
    )
    env.build()
    env.reset()

    retargeter = DexRetargeter(env=env, options=spec.retargeter_options())

    retarget_video(retargeter, video_path, output_path, hand_type, preset)


if __name__ == "__main__":
    tyro.cli(main)
