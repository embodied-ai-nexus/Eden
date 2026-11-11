# TCRS Terrain Adaptation

Turn flat-ground biped motion clips into versions that walk on a terrain. Eden ships a port of the **TCRS**
optimizer from Mondo-Robotics/PMT as `eden.extensions.motion_optimization.tcrs`: footsteps are re-planned onto a
terrain height field with MPPI, the pelvis height is re-derived from the planned support feet, and whole-body Jacobian
IK on a MuJoCo model rebuilds a consistent pose for every frame. The output has the same joints, bodies and frame
count as the input. Use it to make terrain-aware motion-tracking data from flat-ground datasets such as AMASS or
LAFAN1 retargeted to the G1.

```text
flat-ground clip                                                     adapted clip
(fps, dofs_pos,       ┌──────────────────────────────────────────┐   raw / ghost /
 links_pos_w,  ─────▶ │ 1 contact + gait-phase detection         │─▶ optimized
 links_quat_w)        │ 2 SE(2) placement (dx, dy, dyaw)         │   motion sets +
                      │ 3 footstep resolution on the terrain     │   contact masks +
      terrain ──────▶ │ 4 MPPI swing planning + collision fixes  │   IK error metrics
 (height field)       │ 5 support-aware pelvis-z retargeting     │
                      │ 6 whole-body Jacobian IK + stabilizer    │
   robot MJCF ──────▶ │ 7 zero-phase low-pass export filter      │
                      └──────────────────────────────────────────┘
```

Every run returns three variants:

- **raw** — the input, SE(2)-placed on the terrain and otherwise unchanged;
- **ghost** — the raw joints with the terrain-adjusted pelvis height, rebuilt by forward kinematics only, which
  isolates what the IK changed;
- **optimized** — the full pipeline.

The engine is plain numpy, scipy and MuJoCo (`mujoco` is a core dependency of Eden), so adapting a clip needs no
Genesis scene. Only building a height field from procedural `TerrainOptions` needs Genesis initialized (see
[Terrain inputs](#terrain-inputs)).

## Try it

The example recipe adapts a built-in synthetic walking clip onto procedural stairs and writes the three variants:

```bash
uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo --output-dir /tmp/tcrs_demo
```

The same script takes your own clip and terrain, and generates several random placements of one clip:

```bash
# A BeyondMimic-format clip on an exported height field, shifted 0.5 m into the stairs.
uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py \
    --motion walk.npz --terrain stairs_heightfield.npz --dx 0.5 --output-dir results/tcrs

# Eight random SE(2) placements of one clip.
uv run python examples/recipes/motion_optimization/tcrs/adapt_clip.py --demo --output-dir /tmp/tcrs_rounds \
    --n-rounds 8 --dx-range 0.0 0.6 --dy-range -0.2 0.2 --dyaw-range -0.3 0.3 --seed 7

# Render an adapted clip on the height field it was adapted to, as an mp4
# (or pass --demo to adapt and render the synthetic clip in one go).
uv run python examples/recipes/motion_optimization/tcrs/render_on_terrain.py \
    --motion results/tcrs/optimized.npz --terrain stairs_heightfield.npz --output results/tcrs/optimized.mp4
```

`adapt_clip.py` reads a clip that carries its own joint and body names (BeyondMimic files) by name, so any ordering
works. A clip without names is assumed to be in Eden's G1 order (the default) or upstream PMT's policy order
(`--joint-order pmt`). `examples/recipes/motion_optimization/tcrs/adapt_dataset.py` demonstrates the dataset
modifier on a synthetic one-clip dataset.

## Adapt one clip in Python

```python
from eden.extensions.motion_optimization.tcrs import ClipArrays, HeightFieldTerrain, TCRSMotionAdapter
from eden.options.robots.unitree_g1 import G1Dofs29

adapter = TCRSMotionAdapter(model_path=G1Dofs29().fetch_asset())  # the G1's MJCF
terrain = HeightFieldTerrain.from_file("stairs_heightfield.npz")

clip = ClipArrays(fps=50.0, dofs_pos=dofs_pos, links_pos_w=links_pos_w, links_quat_w=links_quat_w)
result = adapter.adapt(clip, terrain, dx=0.5, dy=0.0, dyaw=0.0)

result.optimized.dofs_pos   # (T, 29), same joint order as the input
result.left_contact         # (T,) detected left-foot contacts
result.ik_pos_errors        # (T,) per-frame maximum IK position error, in mm
result.ik_bad_frame_count   # frames above params.reject_ik_mm (200 mm by default): inspect the clip if > 0
```

The `ClipArrays` fields are `dofs_pos` `(T, J)` in `TCRSRobotOptions.dofs_name` order, `links_pos_w` `(T, K, 3)` and
`links_quat_w` `(T, K, 4)` (**wxyz**) in `TCRSRobotOptions.links_name` order, and an optional `links_lin_vel_w`
`(T, K, 3)`, finite-differenced when omitted.

`dx`, `dy` and `dyaw` place the clip on the terrain: the clip is rotated by `dyaw` about the frame-0 pelvis, then
translated by `(dx, dy)`. The adapter loads the MuJoCo model once, reuses the IK solver across `adapt()` calls and
allocates fresh MuJoCo data per call, so one adapter serves many clips. It refuses a model whose first joint is not a
free joint, and a `TCRSRobotOptions.dofs_name` whose length differs from the model's non-root joint count.

## Terrain inputs

All three terrain descriptions reduce to one bilinear height field, `HeightFieldTerrain`:

```python
from eden.extensions.motion_optimization.tcrs import HeightFieldTerrain

terrain = HeightFieldTerrain.from_terrain_options(stairs_options)                # procedural TerrainOptions
terrain = HeightFieldTerrain.from_file("terrain.npz")                            # exported height field
terrain = HeightFieldTerrain.from_entity_options(mesh_options, resolution=0.02)  # 3D mesh, raycast-rasterized
```

- `from_terrain_options` runs the terrain generators off-scene. It needs Genesis initialized (`en.init()` first),
  and raises `RuntimeError` saying so otherwise.
- `from_entity_options` applies the entity's `scale`, `up`/`front` alignment and default root pose, then casts one
  downward ray per cell. Cells the rays miss get `miss_height` (default `0.0`) and the miss fraction is logged. Very
  fine grids (over 4 million rays) warn, because rasterizing is slow without an accelerated trimesh ray backend.

Bilinear interpolation matches the simulated surface: Genesis triangulates height fields into meshes, so the terrain
*is* piecewise-linear between cells. A query outside the grid returns `oob_height` (default `0.0`) and logs a
one-time warning with the out-of-bounds fraction. If you see it, check the clip's placement against the terrain
origin.

### The height-field `.npz` format

Export a height field from a `TerrainOptions` with `eden.utils.terrains.io.height_field_from_terrain_options` plus
`export_height_field`, or straight from a built scene with `env.entities["terrain"].export_height_field(path)` (it
raises `RuntimeError` before `env.build()`).

| Key | Type | Meaning |
|-----|------|---------|
| `height_field` | `float32 [H, W]` | Heights in metres (`vertical_scale` already applied). Axis 0 is world +x, axis 1 is world +y. |
| `horizontal_scale` | `float` | Cell size in metres: `x = origin[0] + i * horizontal_scale`. |
| `origin` | `float [3]` | World position of grid corner `[0, 0]`; `origin[2]` offsets every height. |
| `_format_version` | `int` | Format tag, currently `1`. Loaders reject any other version. |

`export_height_field` requires the path to end in `.npz`. Both ends validate: a non-finite cell (for example the
`-inf` fill of a generator grid that does not cover the whole terrain), a missing key or an unknown version raises
with a message naming the problem.

## Use it on a training dataset: `TCRSTerrainAdapt`

For training data, add the `TCRSTerrainAdapt` dataset modifier (registry key `tcrs_terrain_adapt`) to a motion
dataset's `modifiers`. Every clip is adapted when the dataset loads, and every velocity and body-frame field is
recomputed from the adapted poses, so no stale velocity reaches a tracking reward.

```python
from eden.datasets import AMASSRetargetedG1Dataset
from eden.datasets.modifiers import TCRSTerrainAdapt
from eden.options.entities import TerrainOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.terrains.heightfield_terrains import StairsTerrain

dataset_options = AMASSRetargetedG1Dataset.configure(
    raw_data_path="/path/to/amass_g1",
    modifiers=(
        TCRSTerrainAdapt.configure(
            terrain=TerrainOptions(
                terrain_generators=[[StairsTerrain.configure(step_width=0.3, step_height=0.1)]],
                subterrain_size=(4.0, 4.0),
                horizontal_scale=0.02,
                vertical_scale=0.005,
            ),
            robot_options=G1Dofs29(),         # resolves the MJCF the IK solves against
            terrain_offset=(0.5, 0.0, 0.0),   # SE(2) placement (dx, dy, dyaw)
            planner={"mppi_n_samples": 128},  # TCRSPlannerOptions overrides
        ),
    ),
)
```

| Field | Default | Meaning |
|---|---|---|
| `terrain` | required | `TerrainOptions` (procedural), a mesh-file `EntityOptions`, or the path of a height-field `.npz`. |
| `robot_options` | required | The robot's options. Its `file` must resolve to a MuJoCo MJCF (`.xml`). |
| `terrain_offset` | `(0.0, 0.0, 0.0)` | `(dx, dy, dyaw)`: every clip is yaw-rotated about its frame-0 pelvis, then translated. |
| `mesh_resolution` | `0.02` | Raster cell size (m) for a mesh terrain. |
| `robot_spec` | `None` (the G1 spec) | `TCRSRobotOptions`, or a partial dict of its fields. |
| `planner` | `None` (defaults) | `TCRSPlannerOptions`, or a partial dict of its fields. |
| `cache_dir` | `"~/.cache/eden/tcrs"` | Result cache directory; `None` disables caching. |

What the modifier checks, before any optimization runs:

- `terrain` and `robot_options` are set, `robot_options.file` is not empty and resolves to an `.xml` file.
- A `TerrainOptions` terrain has `randomize=False`: a randomized terrain would not be the surface the clips were
  adapted to, and the cache could never hit.
- Every clip has `dofs_pos`, `base_pos_w`, `base_quat_w`, `links_pos_w` and `links_quat_w` populated (Eden's mimic
  datasets fill the links by forward kinematics) and a free-floating base.
- A clip that carries `contact_points_w` / `contact_points_valid` gets a warning: those were recorded on the flat
  motion and are not re-derived, so they are stale after adaptation.

After the run it logs the cache hits and the median and maximum IK error, and warns how many clips have frames above
the rejection threshold. Inspect those clips before training on them.

### Caching

The optimization takes minutes per clip on real mocap data, and dataset modifiers re-run on every dataset load (the
dataset cache is saved before modifiers apply). So `TCRSTerrainAdapt` caches each adapted clip in `cache_dir`, keyed by
a SHA-256 over:

- the clip arrays (`fps`, `dofs_pos`, `links_pos_w`, `links_quat_w`, and `links_lin_vel_w` when present, since
  velocities feed contact detection);
- the terrain grid, scale, origin and `oob_height`;
- every `TCRSPlannerOptions` and `TCRSRobotOptions` field;
- the `terrain_offset`;
- the bytes of the robot's MJCF file (its meshes are not hashed: the pipeline never reads them);
- a cache-layout version that is bumped when the pipeline's behaviour changes.

Change any of these and the key changes, so a stale result is never served; the second load of an unchanged dataset
is close to free. Each entry is also stamped with the Eden version that wrote it, and entries from another version are
used with a warning (delete the directory to force recomputation). An unreadable or malformed entry is discarded and
recomputed.

Writes are atomic (`os.replace` from a temporary file), so concurrent writers never expose a torn file. Under DDP with
a cache, rank 0 adapts first and the other ranks wait, so their pass is all cache hits.

## Robot and planner options

Everything robot-specific lives in `TCRSRobotOptions` (defined in `eden.options.extensions.motion_optimization` and
re-exported from `eden.extensions.motion_optimization.tcrs`): the joint and link orderings of the clip arrays, the
pelvis, foot, hip and knee link names, the leg and waist joint groups, toe and heel contact offsets, leg reach, and the
per-link IK weights (`ik_config`). The no-argument `TCRSRobotOptions()` reproduces upstream's Unitree G1 values, with
Eden's canonical `G1_DOFS29_NAME` and `G1_DOFS29_TRACKED_LINKS_NAME` as the expected clip orderings. That is exactly
what Eden's G1 datasets produce, so the modifier needs no `robot_spec` in the common case.

Pipeline knobs live in `TCRSPlannerOptions`: contact detection thresholds, MPPI settings (`mppi_n_samples=128`,
`mppi_n_iterations=10`, ...), leg-collision repair, stance latching, the IK lower-body stabilizer, the export
low-pass filter (`lowpass_cutoff_hz=5.0`), the IK rejection threshold (`reject_ik_mm=200.0`) and the random `seed`.

Both are pydantic options: unspecified fields keep their defaults and unknown keys raise at construction. `robot_spec`
and `planner` accept the typed options or a plain dict:

```python
from eden.extensions.motion_optimization.tcrs import TCRSPlannerOptions

TCRSTerrainAdapt.configure(
    ...,
    robot_spec={"dofs_name": my_dof_order},              # the clip stores a different joint order
    planner=TCRSPlannerOptions(lowpass_cutoff_hz=8.0),   # or planner={"lowpass_cutoff_hz": 8.0}
)
```

`TCRSRobotOptions` validates its names at construction: no duplicate joint or link names, every role link
(`pelvis_link`, `left_foot_link`, ...) in `links_name`, and every leg and waist joint in `dofs_name`.

### `ik_config` merges, it does not replace

A partial `ik_config` deep-merges onto `DEFAULT_IK_CONFIG` at construction:

- `solver` keys merge one by one; an unknown solver key raises, listing the valid ones.
- `link_weights` entries merge per link name, and per field within a link, so `{"pelvis": {"pos_weight": 2}}` keeps
  the default `rot_weight`. A link name absent from the defaults is added as a new constraint.
- Any other section raises (and the old section name `body_weights` raises with a pointer to `link_weights`).

So `robot_spec={"ik_config": {"solver": {"tol": 1e-4}}}` changes exactly that one knob and keeps every other solver
setting and link weight. The merged values must be JSON-serializable, because they enter the cache key.

### Another biped

Columns are always resolved by name, never by index, so a reordered clip only needs the right `dofs_name` /
`links_name`. Targeting a different biped means a full `TCRSRobotOptions` (names, foot geometry, reach) plus its MJCF.
The merge above applies there too: the default G1 `link_weights` entries always merge in, so zero out (by name) the
G1 entries you do not want rather than expecting a custom `ik_config` to replace them.

## Why the IK runs on MuJoCo

A Genesis-native rebuild of the solver was evaluated and not adopted. The built-in
`RigidEntity.inverse_kinematics_multilink` covers the geometric structure (multi-link targets, repeated links with
`local_points` for the toe and heel contact points, DOF subsets, warm starts) but cannot express what makes TCRS work
on stair edges: per-target weights, sign-dependent penetration weighting, posture regularization toward the clip,
soft joint limits, custom penalty rows and deterministic multi-seed restarts. A custom Genesis damped-least-squares
loop could express all of it, but the solver issues on the order of a thousand FK/Jacobian calls per frame. When
this was evaluated (July 2026, against the Genesis release of that time), Genesis's per-call primitives measured
10–45× slower than MuJoCo's for that pattern, which projected a 1,618-frame clip from 26.7 s to about 124 s.
Revisit it if Genesis gains batched or cheaper Jacobian primitives. `tests/data/tcrs_stair_golden.npz` anchors the
numerics for any future backend swap.

## Limitations

- **One placement per clip in the modifier.** `TCRSTerrainAdapt` adapts each clip in place at a single
  `terrain_offset`. For several SE(2) placements of each clip, generate them offline with `adapt_clip.py --n-rounds`
  and point the dataset at the generated files.
- **G1-tuned defaults.** The IK weights, foot offsets and planner settings mirror upstream's G1 tuning; another biped
  needs its own `TCRSRobotOptions` and likely some retuning.
- **The first load is slow.** Expect minutes per clip on real mocap data (MPPI plus per-frame IK). Budget for the
  first dataset load and let the cache absorb the rest.
- **Bipeds on height fields only.** A height field cannot represent overhangs, tunnels or multi-storey structures,
  and quadrupeds and hand-terrain contact are out of scope.
- **Not ported from upstream PMT:** the viewer, the window and trajectory IK modes, and the cuRobo and Drake IK
  backends. The port covers the offline `stair_mppi` pipeline.

## See also

- [Motion datasets](motion-datasets.md) — the dataset loaders and dataset modifiers this plugs into.
- [Motion tracking](motion-tracking.md) — how a dataset becomes the motion reference a task tracks.
- [Scenes and terrains](scenes-and-terrains.md) — `TerrainOptions` and the terrain generators.
- [Locomotion tasks](../tasks/locomotion.md) — the terrain locomotion tasks.
