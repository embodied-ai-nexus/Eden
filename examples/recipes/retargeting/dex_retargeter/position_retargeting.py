r"""Offline position retargeting from a saved MANO keypoint trajectory.

---
tier: recipe
capabilities: [retarget.hand, robot.xhand]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/retargeting/dex_retargeter/position_retargeting.py --keypoints-npz data/mano_traj.npz --output-path data/qpos_position.pkl
---

Eden port of ``dex-retargeting``'s ``example/position_retargeting``. The
upstream demo consumes the DexYCB dataset; here we keep it dataset-agnostic
and load a generic NPZ with two arrays:

- ``keypoints`` : ``(T, 21, 3)`` float, MANO-ordered keypoints (right hand).
- ``fps``       : scalar float, playback rate.

Generate such a file from any source you like (DexYCB, ARCTIC, your own
collection); the retargeting math doesn't care.

Run::

    uv run python examples/recipes/retargeting/dex_retargeter/position_retargeting.py \\
        --keypoints-npz data/mano_traj.npz \\
        --output-path data/qpos_position.pkl

Note: upstream's ``position`` config injects a 6-DOF dummy free joint so the
hand can fly through space. Eden's port targets a **fixed-base** entity for
now; absolute keypoint positions are matched relative to the wrist mounting
pose. Floating-base position retargeting is a planned follow-up.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

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


def main(
    keypoints_npz: str,
    output_path: str,
    preset: str = "xhand1_right_position",
) -> None:
    """Retarget a MANO keypoint trajectory to robot qpos.

    Args:
        keypoints_npz: path to ``.npz`` with ``keypoints`` (T, 21, 3) and ``fps``.
        output_path: output pickle path.
        preset: which preset in :mod:`examples.dex_retargeter.hand_presets`
            (must be a ``type='position'`` preset).
    """
    if preset not in HAND_PRESETS:
        raise SystemExit(f"Unknown preset {preset!r}. Available: {list(HAND_PRESETS)}")

    npz = np.load(keypoints_npz)
    keypoints = np.asarray(npz["keypoints"], dtype=np.float32)
    fps = float(npz["fps"]) if "fps" in npz.files else 30.0
    if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 3):
        raise SystemExit(f"Expected keypoints shape (T, 21, 3); got {keypoints.shape}.")

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
    if retargeter.retargeting_type != "POSITION":
        raise SystemExit(f"Preset {preset!r} is type {retargeter.retargeting_type!r}; expected POSITION.")

    indices = retargeter.optimizer.target_link_human_indices
    data: list[np.ndarray] = []
    tic = time.perf_counter()
    for t in tqdm.tqdm(range(keypoints.shape[0]), desc="position retargeting"):
        ref_value = keypoints[t, indices, :]
        data.append(retargeter.retarget(ref_value))
    elapsed = time.perf_counter() - tic
    en.logger.info(f"Retargeted {len(data)} frames in {elapsed:.2f}s")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(
            dict(
                data=data,
                meta_data=dict(
                    preset=preset,
                    dof=len(retargeter.joint_names),
                    joint_names=retargeter.joint_names,
                    fps=fps,
                ),
            ),
            f,
        )
    en.logger.info(f"Wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)
