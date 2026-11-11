"""Visual side-by-side: LAFAN1 clips before and after the smooth-start modifier.

---
tier: recipe
capabilities: [dataset.lafan1, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/lafan1_smooth_start.py --lafan1-dir /path/to/LAFAN1_Retargeting_Dataset/g1
---

Picks a few representative LAFAN1 clips (one locomotion, one dance, one
fall-and-getup), renders the **raw** start of each, then re-runs through the
``SmoothStartTransition`` modifier and renders the result. The two videos
let you see the "teleport" at the raw clip's frame 0 vs the synthesized
stand-up → step → merge prefix.

Usage::

    uv run examples/recipes/datasets/lafan1_smooth_start.py \
        --lafan1-dir /path/to/LAFAN1_Retargeting_Dataset/g1 \
        --output-dir /tmp/smooth_start_demo

Each clip produces two MP4s:
    ``<output_dir>/<clip>_before.mp4`` — raw clip frame 0 onwards (truncated
        to a few seconds of playback).
    ``<output_dir>/<clip>_after.mp4`` — modifier output: default-pose hold,
        IK-synthesized step, SLERP merge, then the same raw clip frames.

The Genesis env (and its kernel compile) is built once and reused across
every clip. The modifier's internal scene build is separate; expect ~30 s
of one-time setup on a cold cache.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import genesis as gs
import torch

import eden as en
from eden.datasets.lafan1_dataset import LAFAN1RetargetedG1Dataset
from eden.datasets.modifiers import DATASET_MODIFIER_REGISTRY, SmoothStartTransition, TrimFrames


# Three clips spanning the difficulty spectrum: a walk (low difficulty,
# minimal stance gap), a dance (medium, stance and pose drift), and a
# fall-and-getup (extreme, starting in a non-upright pose).
DEFAULT_CLIPS = (
    "walk2_subject1",
    "dance1_subject1",
    "fallAndGetUp1_subject1",
)


def _trim_after_n_frames(dataset: LAFAN1RetargetedG1Dataset, n_keep: int) -> None:
    """In-place: drop everything past frame ``n_keep`` in each sequence.

    The "before" render only needs the first second or two of the raw clip
    — the smooth-start prefix vs the raw start is the comparison we care
    about. Trimming keeps each render under ~5 s of wall time.
    """
    for key in dataset.motion_sequence_keys:
        seq = dataset.motion_sequences[key]
        full_len = int(seq.buffer_length[0].item())
        if n_keep >= full_len:
            return
        trim_last = full_len - n_keep
        opts = TrimFrames.configure(trim_first=0, trim_last=trim_last)
        mod = DATASET_MODIFIER_REGISTRY.get(opts.name)(options=opts)
        dataset.motion_sequences = mod.apply(dataset.motion_sequences)


def _build_render_env(show_viewer: bool):
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=0.02, decimation=1, show_world_frame=True),
        scene_options=SceneOptions(
            robot=en.robots.G1Dofs29(),
            plane=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(2.5, 2.5, 1.2),
                cam_lookat=(0.0, 0.0, 0.6),
                cam_fov=45,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()
    return env


def _render_clip(env, seq, out_path: Path, label: str) -> None:
    """Replay one motion sequence through the env and record to ``out_path``.

    The sequence is whichever the dataset's first motion key holds — we
    assume single-clip datasets here (each example call loads one CSV).
    """
    robot = env.entities["robot"]
    cam = env.cameras["camera"]

    T = int(seq.buffer_length[0].item())
    print(f"  rendering {label}: {T} frames → {out_path.name}")

    # Anchor the camera to the clip's starting XY so a long horizontal walk
    # doesn't drift the robot out of frame mid-render.
    start_xy = seq.base_pos_w[0, 0].cpu().numpy()[:2]
    cam.set_pose(pos=(start_xy[0] + 2.5, start_xy[1] + 2.5, 1.2), lookat=(start_xy[0], start_xy[1], 0.6))

    cam.start_recording(video_path=str(out_path), fps=50)
    for i in range(T):
        robot.set_pos(seq.base_pos_w[0, i])
        robot.set_quat(seq.base_quat_w[0, i])
        robot.set_dofs_pos(seq.dofs_pos[0, i])
        env.scene.visualizer.update()
        cam.render_rgb()
    cam.stop_recording()


def _load_single_clip(csv_path: str, device: torch.device) -> LAFAN1RetargetedG1Dataset:
    """Load one LAFAN1 CSV as a single-sequence dataset.

    The dataset's HF auto-download path is bypassed by passing
    ``raw_data_path=<csv>`` directly.
    """
    return LAFAN1RetargetedG1Dataset(
        options=LAFAN1RetargetedG1Dataset.configure(
            raw_data_path=csv_path,
            target_framerate=50.0,
        ),
        device=device,
    )


def _apply_smooth_start(dataset: LAFAN1RetargetedG1Dataset) -> None:
    """Run ``SmoothStartTransition`` in place on the dataset's motion_sequences.

    Uses the modifier's defaults (``default_pose_sec=1.0``, ``step_sec=0.3``,
    ``transition_sec=0.3``), which give a ~80-frame prefix at 50 Hz with a
    visibly relaxed transition. Override on the ``SmoothStartTransition``
    options if you want a faster or slower preprocessing pace.
    """
    from eden.options.robots.unitree_g1 import G1_IK_PROFILE

    opts = SmoothStartTransition.configure(
        ik_profile=G1_IK_PROFILE,
        motion_links_name_per_key=dict(LAFAN1RetargetedG1Dataset.motion_links_name),
    )
    mod = DATASET_MODIFIER_REGISTRY.get(opts.name)(options=opts)
    dataset.motion_sequences = mod.apply(dataset.motion_sequences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lafan1-dir",
        type=str,
        required=True,
        help="Directory of LAFAN1 retargeted-G1 CSVs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="/tmp/smooth_start_demo",
        help="Where to write the before/after MP4s.",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=list(DEFAULT_CLIPS),
        help="Clip basenames (without .csv) to render. Defaults to one walk, one dance, one fall.",
    )
    parser.add_argument(
        "--n-keep-frames",
        type=int,
        default=100,
        help="Trim each raw clip to keep only the first N frames (at target_framerate). "
        "Keeps per-clip render time under ~5 s. The smooth-start prefix runs ~70 frames, "
        "so the 'after' video is roughly N + 70 frames long.",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(debug=False, backend=gs.cpu if args.cpu else gs.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One Genesis render env shared across every clip. The smooth-start
    # modifier builds its *own* env internally — that pays one kernel compile
    # per clip-set, not per clip.
    env = _build_render_env(args.vis)

    for clip_name in args.clips:
        csv_path = os.path.join(args.lafan1_dir, f"{clip_name}.csv")
        if not os.path.exists(csv_path):
            print(f"[skip] {clip_name}: {csv_path} not found")
            continue
        print(f"\n=== {clip_name} ===")

        # 1. Load raw clip → render BEFORE.
        ds_before = _load_single_clip(csv_path, gs.device)
        _trim_after_n_frames(ds_before, args.n_keep_frames)
        _render_clip(env, ds_before.motion_sequences["g1"], out_dir / f"{clip_name}_before.mp4", label="raw clip")

        # 2. Load raw clip → trim → run smooth-start modifier → render AFTER.
        # Reloading from CSV (instead of mutating ``ds_before``) keeps the two
        # renders independent: any bug that double-applies or stale-caches a
        # modifier shows up as a visible delta between the two videos.
        ds_after = _load_single_clip(csv_path, gs.device)
        _trim_after_n_frames(ds_after, args.n_keep_frames)
        _apply_smooth_start(ds_after)
        _render_clip(env, ds_after.motion_sequences["g1"], out_dir / f"{clip_name}_after.mp4", label="smooth-start")

    print(f"\nDone. Videos saved under {out_dir}")


if __name__ == "__main__":
    main()
