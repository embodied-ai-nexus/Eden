"""Wire the TCRSTerrainAdapt dataset modifier: adapt a motion dataset to stairs.

---
tier: recipe
capabilities: [retarget.tcrs, robot.g1, terrain.procedural]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/motion_optimization/tcrs/adapt_dataset.py
---

Demonstrates the dataset-facing entry point of the TCRS optimizer: the
``tcrs_terrain_adapt`` dataset modifier. On a real dataset you declare it in
the dataset options — every clip is then terrain-adapted (and disk-cached) at
load time::

    dataset = AMASSRetargetedG1Dataset(
        options=AMASSRetargetedG1Dataset.configure(
            raw_data_path="/path/to/amass_g1",
            modifiers=(
                TCRSTerrainAdapt.configure(
                    terrain=stairs_options,            # TerrainOptions | EntityOptions | .npz path
                    robot_options=G1Dofs29(),          # resolves the MJCF the IK solves against
                    terrain_offset=(0.5, 0.0, 0.0),    # SE(2) clip placement (dx, dy, dyaw)
                    planner={"mppi_n_samples": 128},   # TCRSPlannerOptions overrides (or pass TCRSPlannerOptions(...))
                ),
            ),
        ),
    )

To stay download-free this demo builds a synthetic one-clip
``BatchMotionSequence`` (the exact structure datasets hand to modifiers) and
calls the modifier directly, then prints before/after pelvis-z statistics —
the adapted pelvis climbs the stairs.
"""

from __future__ import annotations

import argparse

import torch

import eden as en
from eden.datasets.modifiers import TCRSTerrainAdapt
from eden.extensions.motion_optimization.tcrs.demo import make_demo_clip
from eden.options.entities import TerrainOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME, G1Dofs29
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.terrains.heightfield_terrains import StairsTerrain


def build_demo_sequence(num_frames: int) -> BatchMotionSequence:
    """A one-clip BatchMotionSequence shaped like what Eden mimic datasets produce."""
    fps, dofs_pos, links_pos, links_quat = make_demo_clip(num_frames=num_frames)
    seq = BatchMotionSequence.make_empty(
        buffer_lengths=[num_frames], num_dofs=len(G1_DOFS29_NAME), num_links=len(G1_DOFS29_TRACKED_LINKS_NAME)
    )
    seq.framerate[:] = fps
    seq.dofs_pos.concatenated_tensor.copy_(torch.from_numpy(dofs_pos).float())
    seq.links_pos_w.concatenated_tensor.copy_(torch.from_numpy(links_pos).float())
    seq.links_quat_w.concatenated_tensor.copy_(torch.from_numpy(links_quat).float())
    seq.base_pos_w.concatenated_tensor.copy_(torch.from_numpy(links_pos[:, 0]).float())  # pelvis is link 0
    seq.base_quat_w.concatenated_tensor.copy_(torch.from_numpy(links_quat[:, 0]).float())
    seq.contact_points_w = None  # flat-ground contact points would be stale after adaptation
    seq.contact_points_valid = None
    return seq


def main(args: argparse.Namespace) -> None:
    en.init()
    num_frames = 40 if en.is_smoke() else 100

    # Procedural stairs 0.4 m ahead of the clip start (see the TerrainOptions
    # guide); an exported height-field .npz or a mesh EntityOptions works too.
    stairs = TerrainOptions(
        terrain_generators=[[StairsTerrain.configure(step_width=0.3, step_height=0.1)]],
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.02,
        vertical_scale=0.005,
        default_root_pos=(0.4, -2.0, 0.0),
    )
    modifier = TCRSTerrainAdapt(
        options=TCRSTerrainAdapt.configure(
            terrain=stairs,
            robot_options=G1Dofs29(),
            cache_dir=args.cache_dir,  # None disables the on-disk result cache
        )
    )

    seq = build_demo_sequence(num_frames)
    pelvis_before = seq.base_pos_w.concatenated_tensor[:, 2].clone()
    adapted = modifier.apply({"g1": seq})["g1"]
    pelvis_after = adapted.base_pos_w.concatenated_tensor[:, 2]

    en.logger.info(f"pelvis z before: start {pelvis_before[0]:.3f} m, end {pelvis_before[-1]:.3f} m")
    en.logger.info(f"pelvis z after:  start {pelvis_after[0]:.3f} m, end {pelvis_after[-1]:.3f} m")
    en.logger.info(f"pelvis climbed {pelvis_after[-1] - pelvis_before[-1]:+.3f} m over the stairs")
    en.logger.info(f"dofs_vel were recomputed: max |v| = {adapted.dofs_vel.concatenated_tensor.abs().max():.1f} rad/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCRSTerrainAdapt dataset-modifier wiring demo.")
    parser.add_argument("--cache-dir", default="~/.cache/eden/tcrs", help="result cache directory (None-like: '')")
    args = parser.parse_args()
    if args.cache_dir == "":
        args.cache_dir = None
    main(args)
