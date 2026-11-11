"""Retarget a directory of human-motion clips to robot qpos with SomaRetargeter.

---
tier: recipe
capabilities: [retarget.soma, robot.g1]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/retargeting/soma_retargeter/retarget_robot.py --body-model smplx --body-model-path /path/to/body_models/smplx/SMPLX_NEUTRAL.npz --input-dir motions/smplx --robot unitree_g1
---

Supports SMPL / SMPL-H / SMPL-X (AMASS-style ``.npz``) and native SOMA
(``.bvh`` from BONES-SEED, or ``save_soma_npz`` ``.npz``) as sources, and any
robot the retargeting library ships a config for (``--robot``: Unitree G1 or
LimX Oli). One ``SomaRetargeter`` is built at ``--target-fps``; each clip is
retargeted and written as a Genesis-compatible qpos ``.npz`` mirroring the
input layout under ``--output-dir``. With ``--record`` an mp4 of every clip is
rendered alongside its npz.

Example
-------
    python examples/recipes/retargeting/soma_retargeter/retarget_robot.py \
        --body-model smplx \
        --body-model-path /path/to/body_models/smplx/SMPLX_NEUTRAL.npz \
        --input-dir  motions/smplx \
        --output-dir   results/soma_retargeter/smplx --record \
        --robot limx_oli
"""

from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
import numpy as np

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.retargeting.soma_retargeter import SomaRetargeter
from eden.options import CameraOptions, CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions

# FPS npz key per body model (SMPL-X AMASS uses the underscored spelling).
_SMPL_FAMILY = {
    "smpl": ("mocap_framerate", "mocap_frame_rate"),
    "smplh": ("mocap_framerate", "mocap_frame_rate"),
    "smplx": ("mocap_frame_rate", "mocap_framerate"),
}

# Library robot id -> Eden robot options for the scene entity.
_ROBOTS = {
    "unitree_g1": lambda: en.robots.G1Dofs29(),
    "limx_oli": lambda: en.robots.Oli(),
}


def _gender_str(value) -> str:
    v = value.item() if hasattr(value, "item") else value
    return v.decode() if isinstance(v, bytes) else str(v)


def _fps_from(npz, keys, override) -> float:
    if override is not None:
        return float(override)
    for k in keys:
        if k in npz.files:
            return float(npz[k])
    raise KeyError(f"no FPS field {keys} in npz; pass --source-fps")


def _load_clip(path: Path, body_model: str, source_fps: float | None) -> dict:
    """Return the ``retarget_motion`` kwargs for one clip (or a BVH marker)."""
    if body_model == "soma" and path.suffix == ".bvh":
        return {"bvh_path": path}

    with np.load(path, allow_pickle=True) as d:
        if body_model in _SMPL_FAMILY:
            fields = {
                "poses": np.asarray(d["poses"], np.float32),
                "trans": np.asarray(d["trans"], np.float32),
                "betas": np.asarray(d["betas"], np.float32),
                "source_fps": _fps_from(d, _SMPL_FAMILY[body_model], source_fps),
            }
            if "gender" in d.files:
                fields["gender"] = _gender_str(d["gender"])
            return fields
        # native SOMA arrays (save_soma_npz layout); --source-fps overrides the file's 'fps'
        if source_fps is not None:
            fps = float(source_fps)
        elif "fps" in d.files:
            fps = float(d["fps"])
        else:
            raise KeyError("SOMA npz has no 'fps'; pass --source-fps")
        return {
            "poses": np.asarray(d["poses"], np.float32),
            "transl": np.asarray(d["transl"], np.float32),
            "source_fps": fps,
        }


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    en.init(backend=gs.gpu)

    if args.body_model == "soma":
        inputs = sorted(list(args.input_dir.rglob("*.bvh")) + list(args.input_dir.rglob("*.npz")))
    else:
        inputs = sorted(args.input_dir.rglob("*.npz"))
    if not inputs:
        raise SystemExit(f"No input files found under {args.input_dir}")
    en.logger.info(f"Found {len(inputs)} clips under {args.input_dir}")

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=_ROBOTS[args.robot](), floor=PlaneOptions()),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(2.0, 0.0, 1.5),
                cam_lookat=(0.0, 0.0, 1.0),
                cam_fov=60.0,
                follow_entity_name="robot",
                debug=True,
            )
        )
        if args.record
        else None,
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

    written: set[Path] = set()
    for i, src in enumerate(inputs):
        try:
            out_path = (args.output_dir / src.relative_to(args.input_dir)).with_suffix(".npz")
            if out_path in written:
                # e.g. a soma dir with both foo.bvh and foo.npz collapsing to foo.npz
                raise FileExistsError(f"output path already written by another input: {out_path}")

            clip = _load_clip(src, args.body_model, args.source_fps)
            if "bvh_path" in clip:
                qpos = retargeter.retarget_bvh(clip["bvh_path"])
            else:
                if args.num_frames is not None:
                    for key in ("poses", "trans", "transl"):
                        if key in clip:
                            clip[key] = clip[key][: args.num_frames]
                qpos = retargeter.retarget_motion(**clip)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Stamp dof_names so downstream loads columns by name (order-independent).
            np.savez(
                out_path,
                qpos=qpos,
                dof_names=np.array(list(robot.dofs_name)),
                fps=np.float32(args.target_fps),
                source_path=str(src.relative_to(args.input_dir)),
            )
            written.add(out_path)
            en.logger.info(
                f"[{i + 1}/{len(inputs)}] {src.name} -> {out_path.relative_to(args.output_dir)}  qpos {qpos.shape}"
            )

            if args.record:
                _record(env, robot, qpos, out_path.with_suffix(".mp4"), args.target_fps)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not abort the run
            en.logger.error(f"[{i + 1}/{len(inputs)}] skipping {src.name}: {type(exc).__name__}: {exc}")

    retargeter.close()


def _record(env, robot, qpos, video_path: Path, fps: float) -> None:
    import torch

    camera = env.cameras["camera"]
    camera.start_recording(video_path=str(video_path), fps=fps)
    for q in qpos:
        robot.set_qpos(torch.as_tensor(q, device=robot.device))
        env.scene.visualizer.update()
        camera.render_rgb()
    camera.stop_recording()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retarget a directory of human motion to robot qpos.")
    parser.add_argument(
        "--body-model",
        required=True,
        choices=["smpl", "smplh", "smplx", "soma"],
        help="SMPL-family body model the source motion uses",
    )
    parser.add_argument("--robot", choices=sorted(_ROBOTS), default="unitree_g1", help="Target robot.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of clips (searched recursively).")
    parser.add_argument(
        "--body-model-path",
        type=str,
        default=None,
        help="Body-model file or per-gender directory (required for smpl/smplh/smplx).",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Default gender; a per-clip npz 'gender' field overrides it.",
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
    parser.add_argument("--num-frames", type=int, default=None, help="Truncate each clip to N frames (quick tests).")
    parser.add_argument(
        "--snap-to-ground",
        action="store_true",
        help="Shift each clip vertically so its lowest collision point touches the ground.",
    )
    parser.add_argument("-r", "--record", action="store_true", help="Render an mp4 of every retargeted clip.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to run the retargeting solve on")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.body_model in _SMPL_FAMILY and not args.body_model_path:
        raise SystemExit(f"--body-model-path is required for --body-model {args.body_model}")
    main(args)
