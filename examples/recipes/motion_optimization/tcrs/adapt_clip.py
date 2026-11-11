r"""Adapt a flat-ground G1 motion clip to a terrain with the TCRS optimizer.

---
tier: recipe
capabilities: [retarget.tcrs, robot.g1, terrain.procedural]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo
---

Reads a BeyondMimic-format clip npz (``fps``, ``joint_pos`` (T, 29),
``body_pos_w`` (T, 30, 3), ``body_quat_w`` (T, 30, 4) wxyz), runs
``TCRSMotionAdapter.adapt`` against a terrain, and writes ``raw.npz`` /
``ghost.npz`` / ``optimized.npz`` in the same schema to ``--output-dir``. Clips that
carry ``dof_names``/``body_names`` are resolved by name (any ordering);
``--joint-order pmt`` only applies to name-less files.

Terrain: by default a procedural ``StairsTerrain`` ``TerrainOptions``;
``--terrain`` accepts a height-field ``.npz`` (``eden.utils.terrains.io``) or
a mesh file (rasterized by downward raycasts) — dispatch is by extension.

Velocity keys: an input ``body_lin_vel_w`` is forwarded into the pipeline (so
contact detection uses the recorded velocities, matching the dataset-modifier
path); on save, whichever of ``joint_vel`` / ``body_lin_vel_w`` /
``body_ang_vel_w`` the input carried are re-estimated from the adapted
positions via ``eden.utils.motion_reference.utils.estimate_motion_velocities``.
Inputs without velocity keys get none written.

Usage:
    # synthetic walking clip, procedural stairs
    uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo --output-dir /tmp/tcrs_demo

    # your clip on an exported height field, shifted 0.5 m forward
    uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py \
        --motion walk.npz --terrain stairs_heightfield.npz --dx 0.5 --output-dir results/tcrs

    # upstream-style 1->N random SE(2) placement variants
    uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo --output-dir /tmp/tcrs_rounds \
        --n-rounds 4 --dx-range 0.0 0.6 --dy-range -0.2 0.2 --dyaw-range -0.3 0.3 --seed 7
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch

import eden as en
from eden.extensions.motion_optimization.tcrs import HeightFieldTerrain, TCRSPlannerOptions, TCRSRobotOptions
from eden.extensions.motion_optimization.tcrs.demo import make_demo_clip
from eden.extensions.motion_optimization.tcrs.pipeline import ClipArrays, TCRSMotionAdapter
from eden.options.entities import EntityOptions, TerrainOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from eden.utils.assets import get_asset_path
from eden.utils.motion_reference.utils import estimate_motion_velocities
from eden.utils.terrains.heightfield_terrains import StairsTerrain
from eden.utils.terrains.io import height_field_from_terrain_options

# Upstream PMT clips store joints in an interleaved "policy" order; Eden clips
# use the G1 chain order. Both cover the same 29 joints — reorder by name.
PMT_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
_MESH_EXTENSIONS = (".obj", ".stl", ".ply", ".glb", ".gltf", ".dae")


def load_clip(path: str, joint_order: str):
    """Load a BeyondMimic-format clip; returns (raw npz dict, ClipArrays, TCRSRobotOptions).

    Files that carry their own ``dof_names``/``body_names`` (BeyondMimic clips
    do) get a TCRSRobotOptions built from those lists — the pipeline resolves every
    body/joint by name, so any ordering works and ``--joint-order`` is
    ignored. Name-less files fall back to the canonical Eden orderings, with
    ``--joint-order pmt`` reordering interleaved PMT joint columns.

    A file-carried ``body_lin_vel_w`` is forwarded into the ClipArrays: contact
    detection then uses the recorded velocities instead of finite differences,
    matching the dataset-modifier path.
    """
    # On-disk keys follow the BeyondMimic format (joint_*/body_*); Python identifiers
    # follow Eden's dofs_*/links_* convention (same split as eden/managers/terms/commands/mimic.py).
    raw = dict(np.load(path, allow_pickle=False))
    dofs_pos = np.asarray(raw["joint_pos"], dtype=np.float64)
    if "dof_names" in raw or "body_names" in raw:
        if "dof_names" not in raw or "body_names" not in raw:
            raise ValueError(f"'{path}' carries only one of dof_names/body_names; need both (or neither).")
        spec = TCRSRobotOptions(
            dofs_name=[str(n) for n in raw["dof_names"]],
            links_name=[str(n) for n in raw["body_names"]],
        )
    else:
        if joint_order == "pmt":
            pmt_to_eden = [PMT_JOINT_ORDER.index(name) for name in G1_DOFS29_NAME]
            dofs_pos = dofs_pos[:, pmt_to_eden]
        spec = TCRSRobotOptions()
    clip = ClipArrays(
        fps=float(np.asarray(raw["fps"]).reshape(-1)[0]),
        dofs_pos=dofs_pos,
        links_pos_w=np.asarray(raw["body_pos_w"], dtype=np.float64),
        links_quat_w=np.asarray(raw["body_quat_w"], dtype=np.float64),
        links_lin_vel_w=np.asarray(raw["body_lin_vel_w"], dtype=np.float64) if "body_lin_vel_w" in raw else None,
    )
    return raw, clip, spec


def build_terrain(path: str | None, mesh_resolution: float) -> HeightFieldTerrain:
    """Terrain from --terrain (npz height field / mesh file) or default procedural stairs."""
    if path is None:
        options = TerrainOptions(
            terrain_generators=[[StairsTerrain.configure(step_width=0.3, step_height=0.1)]],
            subterrain_size=(4.0, 4.0),
            horizontal_scale=0.02,
            vertical_scale=0.005,
            default_root_pos=(0.4, -2.0, 0.0),  # first tread 0.4 m ahead of the clip origin
        )
        data = height_field_from_terrain_options(options)
        # Prepend a flat 1.4 m apron so the clip starts on level ground.
        apron = np.zeros((70, data.height_field_m.shape[1]), dtype=np.float32)
        grid = np.concatenate([apron, data.height_field_m], axis=0)
        return HeightFieldTerrain(grid, data.horizontal_scale, origin=(-1.0, -2.0, 0.0))
    if path.endswith(".npz"):
        return HeightFieldTerrain.from_file(path)
    if path.endswith(_MESH_EXTENSIONS):
        return HeightFieldTerrain.from_entity_options(EntityOptions(file=path), resolution=mesh_resolution)
    raise ValueError(f"--terrain must be a height-field .npz or a mesh file {_MESH_EXTENSIONS}; got '{path}'.")


def save_motion_set(path: str, motion_set, fps: float, template: dict, joint_order: str) -> None:
    """Write one variant in the input schema (velocity keys only if the input had them)."""
    dofs_pos = motion_set.dofs_pos
    if joint_order == "pmt":
        eden_to_pmt = [list(G1_DOFS29_NAME).index(name) for name in PMT_JOINT_ORDER]
        dofs_pos = dofs_pos[:, eden_to_pmt]
    # On-disk keys follow the BeyondMimic format (joint_*/body_*); Python identifiers
    # follow Eden's dofs_*/links_* convention.
    out = {
        "fps": np.asarray(template["fps"]),
        "joint_pos": dofs_pos.astype(np.float32),
        "body_pos_w": motion_set.links_pos_w.astype(np.float32),
        "body_quat_w": motion_set.links_quat_w.astype(np.float32),
    }
    if "body_names" in template:
        out["body_names"] = template["body_names"]
    if "dof_names" in template:
        out["dof_names"] = template["dof_names"]
    if any(key in template for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")):
        # Re-estimate every velocity track from the adapted positions with the
        # shared framework helper (the base outputs are unused: this schema has
        # no base fields, the pelvis is body 0 of the link tracks).
        _, _, dofs_vel, links_lin_vel, links_ang_vel = estimate_motion_velocities(
            root_trans=torch.from_numpy(np.ascontiguousarray(motion_set.links_pos_w[:, 0])),
            root_quat=torch.from_numpy(np.ascontiguousarray(motion_set.links_quat_w[:, 0])),
            dofs_pos=torch.from_numpy(np.ascontiguousarray(dofs_pos)),
            links_pos_w=torch.from_numpy(np.ascontiguousarray(motion_set.links_pos_w)),
            links_quat_w=torch.from_numpy(np.ascontiguousarray(motion_set.links_quat_w)),
            dt=1.0 / fps,
        )
        if "joint_vel" in template:
            out["joint_vel"] = dofs_vel.numpy().astype(np.float32)
        if "body_lin_vel_w" in template:
            out["body_lin_vel_w"] = links_lin_vel.numpy().astype(np.float32)
        if "body_ang_vel_w" in template:
            out["body_ang_vel_w"] = links_ang_vel.numpy().astype(np.float32)
    np.savez(path, **out)
    en.logger.info(f"  wrote {path}")


def main(args: argparse.Namespace) -> None:
    en.init()
    if en.is_smoke():
        args.output_dir = tempfile.mkdtemp(prefix="tcrs_smoke_")  # keep smoke runs out of the repo tree
    if args.motion is None and not args.demo:
        en.logger.info("No --motion given; running the built-in --demo walking clip.")
        args.demo = True
    if args.demo:
        num_frames = 40 if en.is_smoke() else 100
        fps, dofs_pos, links_pos, links_quat = make_demo_clip(num_frames=num_frames)
        template = {"fps": np.float64(fps), "joint_pos": dofs_pos}  # "joint_pos" is the on-disk BeyondMimic key
        clip = ClipArrays(fps=fps, dofs_pos=dofs_pos, links_pos_w=links_pos, links_quat_w=links_quat)
        args.joint_order = "eden"
        robot_spec = TCRSRobotOptions()
    else:
        template, clip, robot_spec = load_clip(args.motion, args.joint_order)
        if "dof_names" in template:
            args.joint_order = "eden"  # names drive everything; skip the PMT reorder on save

    terrain = build_terrain(args.terrain, args.mesh_resolution)
    terrain.print_info()
    mjcf = get_asset_path("humanoids/unitree_g1/g1_29dof_rev_1_0.xml")
    adapter = TCRSMotionAdapter(model_path=mjcf, robot_spec=robot_spec, params=TCRSPlannerOptions())

    os.makedirs(args.output_dir, exist_ok=True)
    if args.n_rounds > 0:  # upstream-style 1->N random SE(2) placement variants
        rng = np.random.default_rng(args.seed)
        placements = [
            (rng.uniform(*args.dx_range), rng.uniform(*args.dy_range), rng.uniform(*args.dyaw_range))
            for _ in range(args.n_rounds)
        ]
        prefixes = [f"round_{i:04d}_" for i in range(args.n_rounds)]
    else:
        placements = [(args.dx, args.dy, args.dyaw)]
        prefixes = [""]

    for prefix, (dx, dy, dyaw) in zip(prefixes, placements):
        en.logger.info(f"Adapting at dx={dx:+.2f} dy={dy:+.2f} dyaw={dyaw:+.3f}")
        result = adapter.adapt(clip, terrain, dx=dx, dy=dy, dyaw=dyaw)
        en.logger.info(
            f"  IK bad frames: {result.ik_bad_frame_count} (p95 {np.percentile(result.ik_pos_errors, 95):.0f} mm)"
        )
        for name, motion_set in (("raw", result.raw), ("ghost", result.ghost), ("optimized", result.optimized)):
            save_motion_set(
                os.path.join(args.output_dir, f"{prefix}{name}.npz"), motion_set, clip.fps, template, args.joint_order
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCRS terrain-adaptive motion optimization for one G1 clip.")
    parser.add_argument(
        "--motion", default=None, help="BeyondMimic-format clip npz (fps/joint_pos/body_pos_w/body_quat_w)"
    )
    parser.add_argument("--demo", action="store_true", help="use a built-in synthetic walking clip instead of --motion")
    parser.add_argument("--joint-order", choices=("eden", "pmt"), default="eden", help="joint column order of --motion")
    parser.add_argument("--terrain", default=None, help="height-field .npz or mesh file; default: procedural stairs")
    parser.add_argument("--mesh-resolution", type=float, default=0.02, help="raster cell size for mesh terrains (m)")
    parser.add_argument("--dx", type=float, default=0.0, help="clip x translation (m)")
    parser.add_argument("--dy", type=float, default=0.0, help="clip y translation (m)")
    parser.add_argument("--dyaw", type=float, default=0.0, help="clip yaw about the frame-0 pelvis (rad)")
    parser.add_argument("-o", "--output-dir", default="results/tcrs", help="output directory")
    parser.add_argument("--n-rounds", type=int, default=0, help="generate N random SE(2) placement variants instead")
    parser.add_argument(
        "--dx-range", type=float, nargs=2, default=(0.0, 0.5), metavar=("LO", "HI"), help="x range to sample (m)"
    )
    parser.add_argument(
        "--dy-range", type=float, nargs=2, default=(-0.2, 0.2), metavar=("LO", "HI"), help="y range to sample (m)"
    )
    parser.add_argument(
        "--dyaw-range", type=float, nargs=2, default=(-0.3, 0.3), metavar=("LO", "HI"), help="yaw range to sample (rad)"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for the random placements")
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
