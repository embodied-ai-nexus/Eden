"""Pose-estimation commands: ``eden pose gvhmr``.

Batch driver over :mod:`eden.extensions.pose_estimation`: monocular video in, AMASS-style SMPL
``.npz`` out — the input format of ``eden retarget soma``. Arguments are parsed before the
backend is touched, so ``--help`` works without a GVHMR environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eden.utils.cli.paths import iter_src_files, resolve_out
from eden.utils.distribution import install_hint


def _gvhmr_parser() -> argparse.ArgumentParser:
    """Argument surface of ``eden pose gvhmr`` (built without touching the backend)."""
    parser = argparse.ArgumentParser(
        prog="eden pose gvhmr",
        description="Recover world-grounded SMPL-X motion from monocular video (GVHMR) as AMASS-style .npz.",
        epilog=f"Needs GVHMR installed ({install_hint('pose')}) and its checkpoints downloaded "
        "(`gvhmr-download-checkpoints`). Retarget the result with `eden retarget soma --body-model smplx`.",
    )
    parser.add_argument("--src", type=Path, required=True, help="A single video, or a directory searched recursively.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (source tree mirrored); may be a .npz file for a single-file --src.",
    )
    parser.add_argument("--pattern", default="*.mp4", help="Glob for directory --src. Default: *.mp4.")
    parser.add_argument(
        "--checkpoint-root",
        default="",
        help="GVHMR checkpoint directory (gvhmr/, hmr2/, vitpose/, yolo/, body_models/). "
        "Default: GVHMR's own resolution ($GVHMR_CHECKPOINT_ROOT, else ./inputs/checkpoints).",
    )
    parser.add_argument(
        "--ckpt", default="", help="GVHMR checkpoint file. Default: <root>/gvhmr/gvhmr_siga24_release.ckpt."
    )
    parser.add_argument(
        "--static-cam", action="store_true", help="Assume a static camera (skips visual odometry; much faster)."
    )
    parser.add_argument("--use-dpvo", action="store_true", help="Use DPVO instead of SimpleVO for camera motion.")
    parser.add_argument("--f-mm", type=float, default=None, help="Full-frame focal length in mm. Default: estimated.")
    parser.add_argument(
        "--output-root",
        default="",
        help="Where GVHMR writes (and reuses) its per-video intermediates. Default: its own outputs/demo.",
    )
    # No --fps: GVHMR always expresses its output at 30 fps (it relabels every input to that rate
    # and integrates root motion at it), so the only valid stamp is the default. A flag that can
    # only take one value is a trap — `--fps 60` on a 60 fps source looks like the fix and is not.
    parser.add_argument("--skip-existing", action="store_true", help="Skip a video when its output NPZ already exists.")
    parser.add_argument("--verbose", action="store_true", help="Save GVHMR's intermediate preprocess overlays.")
    return parser


def _run_gvhmr(extra_args: list[str]) -> None:
    """``eden pose gvhmr`` — batch monocular video → world-grounded SMPL-X ``.npz``.

    Writes one AMASS-style, Z-up clip per input video, mirroring the source tree under
    ``--out``; one model load covers the batch. Setup: ``docs/guides/pose-estimation.md``.
    """
    args = _gvhmr_parser().parse_args(extra_args)

    import eden as en
    from eden.extensions.pose_estimation.gvhmr import GvhmrEstimator, save_amass_npz

    single, src_root, inputs = iter_src_files(args.src, args.pattern)
    out_paths = [resolve_out(args.out, src_root, s, ".npz", single=single) for s in inputs]
    en.logger.info(f"Found {len(inputs)} video(s) under {src_root}")

    # No import guard: the estimator raises an actionable ImportError naming the extra if
    # `hmr4d` is missing, and does so before any video is read.
    estimator = GvhmrEstimator(
        options=GvhmrEstimator.configure(
            checkpoint_root=args.checkpoint_root,
            ckpt_path=args.ckpt,
            static_cam=args.static_cam,
            use_dpvo=args.use_dpvo,
            f_mm=args.f_mm,
            output_root=args.output_root,
            verbose=args.verbose,
        )
    )
    try:
        for i, (src, out_path) in enumerate(zip(inputs, out_paths)):
            if args.skip_existing and out_path.exists():
                en.logger.info(f"[{i + 1}/{len(inputs)}] skip (exists): {out_path}")
                continue
            en.logger.info(f"[{i + 1}/{len(inputs)}] {src.name}")
            payload = estimator.estimate_clip(src)
            written = save_amass_npz(payload, out_path)
            en.logger.info(
                f"      -> {written}  poses {payload['poses'].shape} @ {float(payload['mocap_frame_rate']):g} fps"
            )
    finally:
        estimator.close()
