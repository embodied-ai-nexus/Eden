# SomaRetargeter

Retarget human motion — **SMPL / SMPL-H / SMPL-X** or native **SOMA** — to a
humanoid robot (`--robot`: **Unitree G1** 29-DOF, the default, or **LimX Oli**
31-DOF), producing Genesis-compatible `qpos`. Backed by the
[`eden-soma-retargeter`](https://github.com/NVIDIA/soma-retargeter) library
(py-soma-x fit → Newton IK).

`retarget_robot.py` builds one `SomaRetargeter` and loops over a directory of
clips, writing one qpos `.npz` per input (mirroring the input layout under
`--output-dir`) plus an optional `.mp4`. `retarget_robot_batched.py` is the same
but retargets `--batch-size` clips per Newton solve for higher throughput on
large corpora.

The scripts are robot-agnostic: `--robot` selects the scene entity and the
library robot id together (the `_ROBOTS` map in `retarget_robot.py`), and the
outputs are decoded **by DOF name**, so nothing downstream depends on the
robot or its joint count. For robots whose model the library does not bundle
(LimX Oli), `SomaRetargeter` automatically forwards the scene entity's own
resolved model file (`HU_D04_01.urdf`) to the library, so both sides are
guaranteed to use the same asset. Supporting another robot = a
`configs/<robot>/` bundle in the library + one `_ROBOTS` entry here.

> `SomaRetargeter` keeps the fit out of the Genesis process, in a hidden worker
> subprocess — you just call `retarget_motion` in a loop; the isolation is handled
> for you.

> The smpl-family fit targets the **public 78-joint SOMA rig**: py-soma-x's
> expanded twist-joint rig needs `SOMA_procedural_transforms.json` /
> `SOMA_template_rig.usda`, which the public `nvidia/soma-x` assets do not ship,
> so `SomaRetargeter` passes `enable_procedural_transforms=False`.
> Both modes emit the same 78-joint interface (the fitted rotations differ —
> procedural mode refines the skin the fit is matched against); pass
> `enable_procedural_transforms=True` only if you have those assets.

## Prerequisites

1. Install the optional `soma-retargeter` extra (`eden-soma-retargeter>=0.4.0`,
   `py-soma-x[smpl]>=0.2,<0.3`; no chumpy: the retargeter's worker reads the
   SMPL / SMPL-H `.pkl` files through `eden.utils.chumpy_pickle`). From the Eden repo root:

   ```bash
   uv sync --inexact --extra soma-retargeter
   ```

   The pins are load-bearing: `eden-soma-retargeter` 0.4.0 is the release that
   accepts `enable_procedural_transforms` (older ones build py-soma-x's
   twist-joint rig unconditionally and cannot fit against the public assets at
   all), and `py-soma-x` 0.1.0 fits near-zero rotations while reporting them as a
   GPU fault. `SomaRetargeter` refuses an out-of-range install up front rather
   than retargeting garbage.

2. A body model for the smpl family:
   - **smplx** — `SMPLX_{NEUTRAL,MALE,FEMALE}.npz`
   - **smplh** — `SMPLH_{MALE,FEMALE}.pkl` (no neutral)
   - **smpl** — `SMPL_{NEUTRAL,MALE,FEMALE}.pkl`

   Pass the file directly, or a directory holding the per-gender files (the
   per-clip `gender` field then selects the right one). Not needed for `soma`.
3. A directory of input clips (searched recursively).

## Input fields per body model

| `--body-model` | files | pose | translation | betas | gender | source FPS |
|---|---|---|---|---|---|---|
| `smpl`  | `*.npz` | `poses` (T,72)  | `trans` | `betas` | `gender` | `mocap_framerate` |
| `smplh` | AMASS `*_poses.npz`   | `poses` (T,156) | `trans` | `betas` | `gender` | `mocap_framerate` |
| `smplx` | AMASS `*_stageii.npz` | `poses` (T,165) | `trans` | `betas` | `gender` | `mocap_frame_rate` |
| `soma`  | `*.bvh` (BONES-SEED)  | — (BVH channels) | — | — | — | BVH `Frame Time` |
| `soma`  | `*.npz` (`save_soma_npz`) | `poses` | `transl` | — | — | `fps` field |

`--source-fps` overrides the metadata when it is missing or unreliable (AMASS
SMPL-X `mocap_frame_rate` sometimes is). FPS does not affect the retargeting,
only the resample rate.

## Usage

```bash
# SMPL-X
python examples/recipes/retargeting/soma_retargeter/retarget_robot.py \
    --body-model smplx \
    --body-model-path /path/to/body_models/smplx/SMPLX_NEUTRAL.npz \
    --input-dir  /path/to/amass_smplx \
    --output-dir   results/soma_retargeter/smplx \
    --target-fps 30 --record

# SMPL-H (mixed male/female directory; per-clip gender is read from each npz)
python examples/recipes/retargeting/soma_retargeter/retarget_robot.py \
    --body-model smplh \
    --body-model-path /path/to/body_models/smplh \
    --input-dir /path/to/amass_smplh --output-dir results/soma_retargeter/smplh

# SOMA (BONES-SEED BVH — no body model needed)
python examples/recipes/retargeting/soma_retargeter/retarget_robot.py \
    --body-model soma \
    --input-dir /path/to/seed_bvh --output-dir results/soma_retargeter/soma --record

# LimX Oli instead of the default G1 (works with every body model)
python examples/recipes/retargeting/soma_retargeter/retarget_robot.py \
    --body-model soma --robot limx_oli \
    --input-dir /path/to/seed_bvh --output-dir results/soma_retargeter/oli --record

# Batched (throughput)
python examples/recipes/retargeting/soma_retargeter/retarget_robot_batched.py \
    --body-model smplx --body-model-path .../SMPLX_NEUTRAL.npz \
    --input-dir .../amass_smplx --output-dir results/smplx --batch-size 8
```

Flags: `--robot {unitree_g1,limx_oli}` (default `unitree_g1`),
`--gender {neutral,male,female}` (default gender; per-clip npz gender
wins), `--num-frames N` (truncate for quick tests), `--snap-to-ground` (shift
each clip vertically so its lowest collision point touches the ground — fixes
clips whose feet sink below the floor), `--device`, and for the batched script
`--batch-size`.

## Output

Each output `.npz` contains:

| key | shape | meaning |
|---|---|---|
| `qpos` | `(T, 7 + N)` float32 | `[x, y, z, qw, qx, qy, qz, joints…]`, Z-up, wxyz quat, radians, in `robot.dofs_name` order (N = 29 for G1, 31 for Oli) |
| `dof_names` | `(N,)` str | DOF labels for `qpos[:, 7:]` — decode columns **by name** (`eden.utils.string.columns_by_name`) so the file is independent of the robot and of Genesis's internal DOF ordering |
| `fps` | scalar | target FPS |
| `source_path` | str | input path relative to `--input-dir` |

Replay: `robot.set_qpos(qpos[t])` frame by frame (columns already in
`dofs_name` order).

## Test data

`test/` (gitignored) holds a small verification corpus built by
`test/make_test_set.py` — 10 clips each of smplh / smplx (from AMASS) and soma
(BONES-SEED BVH, stream-extracted). Retargeted results land in
`test/retargeted/{body_model}/` (G1) and `test/oli_retarget/{body_model}/`
(LimX Oli, plus `test/oli_retarget/oli_batch_retarget/` from the batched
script). See `test/PROGRESS.md`.
