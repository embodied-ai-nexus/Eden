"""Batched directory retargeting to robot qpos with SomaRetargeter.

---
tier: recipe
capabilities: [retarget.soma, robot.g1]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/retargeting/soma_retargeter/retarget_robot_batched.py --body-model smplx --body-model-path /path/to/body_models/smplx/SMPLX_NEUTRAL.npz --input-dir motions/smplx --batch-size 8
---

Same inputs and outputs as ``retarget_robot.py``, but retargets ``--batch-size``
clips per Newton IK solve (one env per clip) for higher GPU throughput. Batching
clips of similar length is most efficient — the IK cost of a batch scales with
its longest clip. Use this for large corpora; use ``retarget_robot.py`` for the
simple one-clip-at-a-time loop.

Example
-------
    python examples/recipes/retargeting/soma_retargeter/retarget_robot_batched.py \
        --body-model smplx \
        --body-model-path /path/to/body_models/smplx/SMPLX_NEUTRAL.npz \
        --input-dir motions/smplx --output-dir results/smplx --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
import numpy as np

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.retargeting.soma_retargeter import SomaRetargeter
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions

# Reuse the single-clip loader / field mapping from the sibling script.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from retarget_robot import _ROBOTS, _SMPL_FAMILY, _load_clip  # noqa: E402


def _sorted_inputs(input_dir: Path, body_model: str) -> list[Path]:
    if body_model == "soma":
        files = list(input_dir.rglob("*.bvh")) + list(input_dir.rglob("*.npz"))
    else:
        files = list(input_dir.rglob("*.npz"))
    # Group similar lengths together for efficient batches (largest file ~ longest).
    return sorted(files, key=lambda p: p.stat().st_size)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    en.init(backend=gs.gpu)

    inputs = _sorted_inputs(args.input_dir, args.body_model)
    if not inputs:
        raise SystemExit(f"No input files found under {args.input_dir}")
    en.logger.info(f"Found {len(inputs)} clips; batch size {args.batch_size}")

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=_ROBOTS[args.robot](), floor=PlaneOptions()),
        show_viewer=False,
    )
    env.build()
    env.reset()
    robot = env.entities["robot"]

    retargeter = SomaRetargeter(
        env=env,
        options=SomaRetargeter.configure(
            entity_name="robot",
            body_model=args.body_model,
            body_model_path=args.body_model_path or "",
            gender=args.gender,
            target_fps=args.target_fps,
            snap_to_ground=args.snap_to_ground,
            device=args.device,
            robot=args.robot,
        ),
    )

    done = 0
    written: set[Path] = set()
    for start in range(0, len(inputs), args.batch_size):
        batch_paths = inputs[start : start + args.batch_size]
        clips, kept = [], []
        for src in batch_paths:
            try:
                clip = _load_clip(src, args.body_model, args.source_fps)
                if args.num_frames is not None:
                    for key in ("poses", "trans", "transl"):
                        if key in clip:
                            clip[key] = clip[key][: args.num_frames]
                clips.append(clip)
                kept.append(src)
            except Exception as exc:  # noqa: BLE001
                en.logger.error(f"skipping {src.name}: {type(exc).__name__}: {exc}")

        if not clips:
            continue
        try:
            results = list(zip(kept, retargeter.retarget_motions(clips)))
        except Exception as exc:  # noqa: BLE001 - retry per-clip so one bad clip doesn't drop the batch
            en.logger.error(f"batch failed ({type(exc).__name__}: {exc}); retrying per-clip")
            results = []
            for src, clip in zip(kept, clips):
                try:
                    results.append((src, retargeter.retarget_motions([clip])[0]))
                except Exception as exc2:  # noqa: BLE001
                    en.logger.error(f"skipping {src.name}: {type(exc2).__name__}: {exc2}")

        for src, qpos in results:
            out_path = (args.output_dir / src.relative_to(args.input_dir)).with_suffix(".npz")
            if out_path in written:
                en.logger.error(f"skipping {src.name}: output path already written: {out_path}")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                qpos=qpos,
                dof_names=np.array(list(robot.dofs_name)),
                fps=np.float32(args.target_fps),
                source_path=str(src.relative_to(args.input_dir)),
            )
            written.add(out_path)
            done += 1
            en.logger.info(
                f"[{done}/{len(inputs)}] {src.name} -> {out_path.relative_to(args.output_dir)}  qpos {qpos.shape}"
            )

    retargeter.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batched directory retargeting to robot qpos.")
    parser.add_argument(
        "--body-model",
        required=True,
        choices=["smpl", "smplh", "smplx", "soma"],
        help="SMPL-family body model the source motion uses",
    )
    parser.add_argument("--robot", choices=sorted(_ROBOTS), default="unitree_g1", help="Target robot.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of source motions to retarget")
    parser.add_argument(
        "--body-model-path", type=str, default=None, help="Directory holding the SMPL-family model files"
    )
    parser.add_argument(
        "--gender", choices=["neutral", "male", "female"], default="neutral", help="Body-model gender to instantiate"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("results/soma_retargeter"), help="Directory to write results to"
    )
    parser.add_argument(
        "--target-fps", type=float, default=30.0, help="Resample the retargeted motion to this frame rate"
    )
    parser.add_argument(
        "--source-fps", type=float, default=None, help="Override source FPS when npz metadata is missing/unreliable."
    )
    parser.add_argument(
        "--num-frames", type=int, default=None, help="Stop after this many frames (default: the whole clip)"
    )
    parser.add_argument(
        "--snap-to-ground",
        action="store_true",
        help="Shift each clip vertically so its lowest collision point touches the ground.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Clips per Newton IK solve.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to run the retargeting solve on")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.body_model in _SMPL_FAMILY and not args.body_model_path:
        raise SystemExit(f"--body-model-path is required for --body-model {args.body_model}")
    main(args)
