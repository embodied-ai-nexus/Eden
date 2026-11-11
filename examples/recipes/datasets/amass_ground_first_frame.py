"""Visual demo: AMASS-soma-G1 clips with the GroundFirstFrame modifier.

---
tier: recipe
capabilities: [dataset.amass, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/datasets/amass_ground_first_frame.py --raw-data-path /path/to/AMASS/retargeted_g1_soma
---

Loads a small subset of AMASS-retargeted-G1 in the soma / ProtoMotions
format, renders the **raw** start of each in env 0..N-1, then re-runs
through ``GroundFirstFrame`` and renders the result. The two videos let
you see any clip whose frame-0 footprint sat below the floor lifted to
``z=0``.

Usage::

    uv run examples/recipes/datasets/amass_ground_first_frame.py \
        --raw-data-path /data/group_data/katefgroup/datasets/AMASS/retargeted_g1_soma \
        --output-dir /tmp/ground_first_frame_demo

The ProtoMotions-format reader is inlined here as a small adaptation of the
AMASS dataset reader (support for the ProtoMotions key set). The
``GroundFirstFrame`` modifier itself is dataset-agnostic — running it
against the canonical IsaacLab-AMP AMASS files via the unmodified
``AMASSRetargetedG1Dataset`` works the same way.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np

import eden as en
from eden.datasets.amass_dataset import AMASSRetargetedG1Dataset
from eden.datasets.modifiers import GroundFirstFrame
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.motion_reference.motion_data import BatchMotionSequence


# Soma subset — clips drawn from the ``amass_soma_training.yaml`` curriculum.
# Filter is ``re.search`` over the file basename (no directory prefix).
DEFAULT_INCLUDE = (
    r"^45_01_stageii\.npz$",
    r"^138_03_stageii\.npz$",
    r"^13_18_stageii\.npz$",
    r"^13_38_stageii\.npz$",
)


class _AMASSSomaG1Dataset(AMASSRetargetedG1Dataset):
    """Subclass that reads ProtoMotions-style files (``root_pos`` / ``root_rot`` /
    ``dof_pos`` / ``fps``) and reconstructs link world poses via Genesis FK.

    The long-term home for this support is per-file dispatch in
    ``AMASSRetargetedG1Dataset`` itself; this subclass exists so the present
    demo doesn't depend on that landing.
    """

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        # ProtoMotions files ship only root pose + dof_pos; fill link buffers
        # via FK so downstream consumers (the modifier, the renderer, etc.)
        # see a fully-populated BatchMotionSequence.
        return self._prepare_single_key(
            self._read_single_sequence,
            desc="AMASS-soma: reading raw .npz",
            robot=G1Dofs29(dofs_name=self.motion_dofs_name["g1"]),
        )

    def _read_single_sequence(self, file_path: str):
        raw = np.load(file_path, allow_pickle=True)
        data = raw.item() if isinstance(raw, np.ndarray) and raw.ndim == 0 else dict(raw)

        window = self._frame_window()
        return self._pack_motion_sequence(
            framerate=float(np.asarray(data["fps"]).item()),
            base_pos_w=np.asarray(data["root_pos"])[window],
            base_quat_w=np.asarray(data["root_rot"])[window],  # wxyz
            dofs_pos=np.asarray(data["dof_pos"])[window],
        )


def _build_render_env(num_envs: int, show_viewer: bool):
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, PlaneOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=num_envs,
            sim_dt=0.02,
            decimation=1,
            show_world_frame=True,
            env_spacing=(2.5, 2.5),
        ),
        scene_options=SceneOptions(
            robot=G1Dofs29(),
            plane=PlaneOptions(),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                # Low, close camera — the per-clip lift is on the order of
                # a few centimetres, so a ground-level viewpoint is what
                # actually shows the floor penetration / clearance.
                cam_pos=(4.0, 4.0, 0.4),
                cam_lookat=(0.0, 0.0, 0.1),
                cam_fov=45,
            )
        ),
        show_viewer=show_viewer,
    )
    env.build()
    env.reset()
    return env


def _render(env, seq, out_path: Path, label: str, max_frames: int) -> None:
    """Replay every clip in ``seq`` (one per env) and record to ``out_path``."""
    robot = env.entities["robot"]
    cam = env.cameras["camera"]

    T = min(int(seq.buffer_length.max().item()), max_frames)
    print(f"  rendering {label}: {T} frames → {out_path.name}")

    cam.start_recording(video_path=str(out_path), fps=50)
    for i in range(T):
        base_pos, valid_mask = seq.base_pos_w.with_mask[:, i]
        if not valid_mask.any():
            break
        robot.set_pos(base_pos[valid_mask], envs_idx=valid_mask)
        robot.set_quat(seq.base_quat_w[:, i][valid_mask], envs_idx=valid_mask)
        robot.set_dofs_pos(seq.dofs_pos[:, i][valid_mask], envs_idx=valid_mask)
        env.scene.visualizer.update()
        cam.render_rgb()
    cam.stop_recording()


def _load_dataset(raw_data_path, include_pattern, modifiers, device):
    return _AMASSSomaG1Dataset(
        options=_AMASSSomaG1Dataset.configure(
            raw_data_path=raw_data_path,
            target_framerate=50.0,
            include_pattern=include_pattern,
            modifiers=modifiers,
        ),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="/data/group_data/katefgroup/datasets/AMASS/retargeted_g1_soma",
        help="Local AMASS-G1-soma root.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="/tmp/ground_first_frame_demo", help="Directory to write results to"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=list(DEFAULT_INCLUDE),
        help="Per-file regex patterns (re.search on basename). Defaults to a soma subset.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Cap the per-clip frame count for a quick demo render.",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(debug=False, backend=gs.cpu if args.cpu else gs.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ``re.search`` ORs the entries by joining with ``|``.
    include_pattern = "|".join(args.include)
    print(f"Loading AMASS-soma-G1 from {args.raw_data_path} (include_pattern={include_pattern!r})...")

    # Two independent loads — keeps the renders trivially comparable (any
    # modifier bug shows as a visible delta between the two videos).
    ds_before = _load_dataset(args.raw_data_path, include_pattern, modifiers=(), device=gs.device)
    ds_after = _load_dataset(
        args.raw_data_path,
        include_pattern,
        modifiers=(GroundFirstFrame.configure(robot_options=G1Dofs29()),),
        device=gs.device,
    )

    num_envs = len(ds_before)
    print(f"Loaded {num_envs} motion sequence(s); per-env spacing 2.5 m, side-by-side render")

    # Quick text diagnostic: per-clip frame-0 z before vs after.
    z0_before = ds_before.motion_sequences["g1"].base_pos_w[:, 0][:, 2].tolist()
    z0_after = ds_after.motion_sequences["g1"].base_pos_w[:, 0][:, 2].tolist()
    for i, (zb, za) in enumerate(zip(z0_before, z0_after)):
        delta = za - zb
        marker = "↑" if delta > 1e-4 else "·"
        print(f"  env {i}: base_z[0] {zb:+.4f} → {za:+.4f}  ({marker} dz={delta:+.4f} m)")

    env = _build_render_env(num_envs, args.vis)
    _render(env, ds_before.motion_sequences["g1"], out_dir / "before.mp4", "raw clips", args.max_frames)
    _render(env, ds_after.motion_sequences["g1"], out_dir / "after.mp4", "GroundFirstFrame", args.max_frames)

    print(f"\nDone. Videos saved under {out_dir}/")


if __name__ == "__main__":
    main()
