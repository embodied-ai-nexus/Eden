# TCRS — Terrain-Adaptive Motion Optimization

Transform flat-ground G1 motion clips into terrain-adapted versions with Eden's
port of the Mondo-Robotics/PMT **TCRS** optimizer
(`eden.extensions.motion_optimization.tcrs`). Per clip it runs:

1. **Contact / gait-phase detection** — per-foot contact masks and an optional
   phase clock derived from the raw motion.
2. **MPPI foot planning** — footsteps snap onto safe terrain support surfaces;
   every swing trajectory is re-optimized against the height field (clearance,
   step edges, side walls, leg/foot collision capsules).
3. **Pelvis-z retargeting** — a support-aware, reach-bounded filter re-derives
   the pelvis height from the planned footholds.
4. **Whole-body Jacobian IK** — per-frame damped-least-squares IK on a MuJoCo
   model of the robot tracks the new foot/pelvis targets while holding the
   upper body to the reference, with a lower-body stance stabilizer post-pass
   and a zero-phase lowpass export filter.

The result is a drop-in replacement clip: same joints, same bodies, same
frame count — but walking *on* the terrain instead of through it.

## Terrain inputs

All three reduce to one bilinear height field (`HeightFieldTerrain`):

| Input | Built via | Use for |
|-------|-----------|---------|
| `TerrainOptions` | `HeightFieldTerrain.from_terrain_options(...)` | Eden's procedural terrains (stairs, slopes, ...) |
| height-field `.npz` | `HeightFieldTerrain.from_file(...)` | Terrain exported once via `eden.utils.terrains.io.export_height_field` (or `Terrain.export_height_field` from a live scene) |
| mesh `EntityOptions` | `HeightFieldTerrain.from_entity_options(...)` | Any 3D terrain mesh file, rasterized by downward raycasts |

## Quickstart

```bash
# Synthetic walking clip over procedural stairs -> raw / ghost / optimized npz
uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo --output-dir /tmp/tcrs_demo

# Your BeyondMimic-format clip on an exported height field, shifted 0.5 m into the stairs
uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py \
    --motion walk.npz --terrain stairs_heightfield.npz --dx 0.5 --output-dir results/tcrs

# Upstream-style 1->N SE(2) placement variants (random dx/dy/dyaw per round)
uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo --output-dir /tmp/tcrs_rounds \
    --n-rounds 8 --dx-range 0.0 0.6 --dy-range -0.2 0.2 --dyaw-range -0.3 0.3 --seed 7

# Dataset-modifier wiring demo (synthetic 1-clip BatchMotionSequence)
uv run python examples/recipes/motion_optimization/tcrs/adapt_dataset.py

# Render an adapted clip on its terrain for visual confirmation (mp4)
uv run python examples/recipes/motion_optimization/tcrs/render_on_terrain.py     --motion /tmp/tcrs/optimized.npz --terrain stairs.npz --output /tmp/tcrs/optimized.mp4
```

`adapt_clip.py` resolves clips that carry their own `dof_names`/`body_names`
(BeyondMimic files) by name, so any ordering works. Name-less clips are
assumed to be in Eden's G1 chain order (the default) or upstream PMT "policy"
order (`--joint-order pmt`); columns are reordered by joint name in both
directions, so the output npz keeps the input convention. It writes three variants per placement: `raw` (SE(2)-placed input),
`ghost` (raw joints + terrain-adjusted pelvis z, FK only) and `optimized`
(the full pipeline). Requires `mujoco` (installed with Eden's deps).

## Dataset use: the `tcrs_terrain_adapt` modifier

For training datasets, wire `TCRSTerrainAdapt` into the dataset options —
every clip is adapted at load time and all velocity / body-frame fields are
recomputed from the adapted poses:

```python
from eden.datasets.modifiers import TCRSTerrainAdapt

modifiers = (
    TCRSTerrainAdapt.configure(
        terrain=stairs_options,          # TerrainOptions | EntityOptions | .npz path
        robot_options=G1Dofs29(),        # resolves the MJCF the IK solves against
        terrain_offset=(0.5, 0.0, 0.0),  # SE(2) placement (dx, dy, dyaw)
        planner={"mppi_n_samples": 128}, # TCRSPlannerOptions overrides (unknown keys raise)
    ),
)
```

The optimization costs minutes per clip on real data, so results are cached in
`~/.cache/eden/tcrs` keyed by a content hash of the clip, terrain, parameters
and robot MJCF — the second load of an unchanged dataset is near-free. The
modifier adapts clips 1→1 at a single placement; for 1→N augmentation
generate variants offline with `adapt_clip.py --n-rounds` (above) and feed the
generated files to your dataset.

See `docs/guides/tcrs-terrain-adaptation.md` for the full guide (heightfield
npz schema, `TCRSRobotOptions` overrides for non-G1 bipeds, cache invalidation).

## Citation

TCRS is part of Mondo-Robotics **PMT**; the port covers the offline
`stair_mppi` pipeline (viewer, window/trajectory IK and curobo/drake IK
backends were not ported).
