"""Recover a world-grounded SMPL-X motion from a monocular video with GVHMR.

---
tier: recipe
capabilities: [pose.gvhmr]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/pose_estimation/gvhmr/video_to_motion.py --video clip.mp4 --output motions/clip.npz
---

GVHMR is an optional extra installed alongside Eden (``pip install 'eden-world[pose]'``) and runs
in-process. Setup: ``docs/guides/pose-estimation.md``.

The written ``.npz`` is an AMASS-style, **Z-up** SMPL-X clip — exactly what ``eden retarget
soma --body-model smplx`` reads, the command this script prints at the end. GVHMR's world
translation is only defined up to a vertical offset, so retarget with ``--snap-to-ground``.
No Genesis scene is involved; the robot only enters at retargeting.

Example
-------
    python examples/recipes/pose_estimation/gvhmr/video_to_motion.py \
        --video ~/research/eden_gvhmr/docs/example_video/tennis.mp4 \
        --output   motions/tennis.npz \
        --checkpoint-root ~/research/eden_gvhmr/inputs/checkpoints \
        --static-cam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import eden as en
from eden.extensions.pose_estimation.gvhmr import GvhmrEstimator, save_amass_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("---")[0].strip())
    parser.add_argument("--video", type=Path, required=True, help="Input monocular video.")
    parser.add_argument("-o", "--output", type=Path, default=Path("motions/gvhmr_clip.npz"), help="Output .npz clip.")
    parser.add_argument("--checkpoint-root", default="", help="GVHMR checkpoint directory.")
    parser.add_argument("--static-cam", action="store_true", help="Assume a static camera (skips visual odometry).")
    args = parser.parse_args()

    estimator = GvhmrEstimator(
        options=GvhmrEstimator.configure(
            checkpoint_root=args.checkpoint_root,
            static_cam=args.static_cam,
        )
    )
    try:
        clip = estimator.estimate_clip(args.video)
    finally:
        estimator.close()

    path = save_amass_npz(clip, args.output)

    pelvis = clip["gvhmr_pelvis_j0"] + clip["trans"]  # SMPL-X anchors the root at J0
    travelled = np.linalg.norm(np.diff(pelvis[:, :2], axis=0), axis=-1).sum()
    en.logger.info(f"-> {path}")
    en.logger.info(f"   frames {clip['poses'].shape[0]} @ {float(clip['mocap_frame_rate']):g} fps")
    en.logger.info(f"   pelvis height (z) {pelvis[:, 2].min():.3f} .. {pelvis[:, 2].max():.3f} m")
    en.logger.info(f"   horizontal path {travelled:.2f} m")
    en.logger.info(
        "Retarget it with:\n"
        f"  eden retarget soma --src {path} --out qpos/ --body-model smplx "
        "--body-model-path <SMPLX_NEUTRAL.npz> --robot unitree_g1 --snap-to-ground"
    )


if __name__ == "__main__":
    main()
