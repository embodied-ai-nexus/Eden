r"""Replay a (TCRS-adapted) motion clip on its terrain and record an mp4.

---
tier: recipe
capabilities: [retarget.tcrs, robot.g1, terrain.procedural, render.offscreen]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/motion_optimization/tcrs/render_on_terrain.py --demo
---

Companion to ``adapt_clip.py`` for visual confirmation: builds a Genesis scene
with a G1 and the terrain the clip was adapted to, kinematically replays the
clip (``set_qpos`` per frame, no physics stepping), and records an offscreen
camera to ``--output``.

The terrain comes from the same height-field ``.npz``
(:mod:`eden.utils.terrains.io`) that was passed to ``adapt_clip.py`` — the
grid is loaded into a ``gs.morphs.Terrain`` so the rendered surface is exactly
the surface the optimizer planned against. Clips that carry
``dof_names``/``body_names`` are resolved by name; name-less files are assumed
to be in Eden's canonical G1 orderings.

Usage:
    # adapt, then render on the same terrain
    uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py \\
        --motion my_clip.npz --terrain stairs.npz --output-dir results/tcrs
    uv run python examples/recipes/motion_optimization/tcrs/render_on_terrain.py \\
        --motion results/tcrs/optimized.npz --terrain stairs.npz \\
        --output results/tcrs/optimized.mp4

    # self-contained demo (synthetic clip on procedural stairs)
    uv run python examples/recipes/motion_optimization/tcrs/render_on_terrain.py --demo
"""

import argparse
import os
import tempfile

import numpy as np
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import SceneOptions, TerrainOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME, G1Dofs29
from eden.utils.string import columns_by_name
from eden.utils.terrains.heightfield_terrains import StairsTerrain
from eden.utils.terrains.io import (
    export_height_field,
    height_field_from_terrain_options,
    terrain_options_from_height_field_file,
)


def _demo_inputs(out_dir: str) -> tuple[str, str]:
    """Adapt the built-in demo walking clip on procedural stairs; return (clip npz, terrain npz)."""
    from eden.extensions.motion_optimization.tcrs.demo import make_demo_clip
    from eden.extensions.motion_optimization.tcrs.pipeline import ClipArrays, TCRSMotionAdapter
    from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain
    from eden.utils.assets import get_asset_path

    options = TerrainOptions(
        terrain_generators=[[StairsTerrain.configure(step_width=0.3, step_height=0.1)]],
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.02,
        vertical_scale=0.005,
        default_root_pos=(-1.0, -2.0, 0.0),
    )
    data = height_field_from_terrain_options(options)
    terrain_path = os.path.join(out_dir, "demo_terrain.npz")
    export_height_field(terrain_path, data.height_field_m, data.horizontal_scale, origin=data.origin)

    num_frames = 40 if en.is_smoke() else 100
    fps, dofs_pos, links_pos, links_quat = make_demo_clip(num_frames=num_frames)
    adapter = TCRSMotionAdapter(model_path=get_asset_path("humanoids/unitree_g1/g1_29dof_rev_1_0.xml"))
    result = adapter.adapt(
        ClipArrays(fps=fps, dofs_pos=dofs_pos, links_pos_w=links_pos, links_quat_w=links_quat),
        HeightFieldTerrain.from_file(terrain_path),
    )
    clip_path = os.path.join(out_dir, "demo_optimized.npz")
    # On-disk keys follow the BeyondMimic format (joint_*/body_*); Python identifiers
    # follow Eden's dofs_*/links_* convention.
    np.savez(
        clip_path,
        fps=np.array([fps]),
        joint_pos=result.optimized.dofs_pos.astype(np.float32),
        body_pos_w=result.optimized.links_pos_w.astype(np.float32),
        body_quat_w=result.optimized.links_quat_w.astype(np.float32),
    )
    return clip_path, terrain_path


def main(args: argparse.Namespace) -> None:
    en.init()
    if en.is_smoke():
        args.output = os.path.join(tempfile.mkdtemp(prefix="tcrs_render_smoke_"), "clip.mp4")
    if args.motion is None and not args.demo:
        en.logger.info("No --motion given; running the built-in --demo.")
        args.demo = True
    if args.demo:
        demo_dir = os.path.dirname(args.output) or "."
        os.makedirs(demo_dir, exist_ok=True)
        args.motion, args.terrain = _demo_inputs(demo_dir)
    if args.terrain is None:
        raise SystemExit("--terrain (height-field .npz used for the adaptation) is required outside --demo.")

    # On-disk keys follow the BeyondMimic format (joint_*/body_*/dof_names); Python
    # identifiers follow Eden's dofs_*/links_* convention.
    raw = np.load(args.motion, allow_pickle=False)
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    if fps <= 0:
        raise ValueError(f"'{args.motion}' carries fps={fps}; the clip framerate must be > 0 to replay it.")
    dofs_name = [str(n) for n in raw["dof_names"]] if "dof_names" in raw.files else list(G1_DOFS29_NAME)
    links_name = (
        [str(n) for n in raw["body_names"]] if "body_names" in raw.files else list(G1_DOFS29_TRACKED_LINKS_NAME)
    )
    pelvis = links_name.index("pelvis")

    terrain_options = terrain_options_from_height_field_file(args.terrain)

    lookat = raw["body_pos_w"][:, pelvis].mean(axis=0)
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / fps),
        scene_options=SceneOptions(robot=G1Dofs29(), terrain=terrain_options),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_resolution=tuple(args.resolution),
                cam_pos=tuple(np.asarray(lookat) + np.asarray(args.camera_offset)),
                cam_lookat=tuple(lookat),
            ),
        ),
        show_viewer=False,
    )
    env.build()
    robot = env.entities["robot"]

    # Clip dofs are in the clip's own order; set_qpos wants dofs_name order.
    perm = columns_by_name(dofs_name, robot.dofs_name, what="motion npz 'dof_names'")

    dofs_pos = torch.from_numpy(np.asarray(raw["joint_pos"])[:, perm]).float().to(env.device)
    base_pos = torch.from_numpy(np.asarray(raw["body_pos_w"])[:, pelvis]).float().to(env.device)
    base_quat = torch.from_numpy(np.asarray(raw["body_quat_w"])[:, pelvis]).float().to(env.device)

    cam = env.cameras["camera"]
    cam.start_recording(args.output, fps=fps)
    num_frames = dofs_pos.shape[0]
    for frame in range(num_frames):
        robot.set_qpos(torch.cat([base_pos[frame], base_quat[frame], dofs_pos[frame]]).unsqueeze(0))
        env.scene.visualizer.update()
        cam.render_rgb()
        if frame % 200 == 0:
            en.logger.info(f"rendered {frame}/{num_frames}")
    cam.stop_recording()
    # Frame count included so an empty clip reads as one: with no frame there is no file,
    # and stop_recording() says so rather than leaving a claim of success behind.
    en.logger.info(f"wrote {args.output} ({num_frames} frames)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("---")[0], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--motion", default=None, help="clip npz to replay (e.g. adapt_clip.py's optimized.npz)")
    parser.add_argument("--terrain", default=None, help="height-field .npz the clip was adapted to")
    parser.add_argument("--demo", action="store_true", help="self-contained demo: adapt + render a synthetic clip")
    parser.add_argument("-o", "--output", default="results/tcrs/render.mp4", help="output mp4 path")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(1280, 720),
        metavar=("W", "H"),
        help="Width and height of the rendered video",
    )
    parser.add_argument(
        "--camera-offset",
        type=float,
        nargs=3,
        default=(2.2, -2.0, 1.4),
        metavar=("X", "Y", "Z"),
        help="camera position relative to the clip's mean pelvis position",
    )
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
