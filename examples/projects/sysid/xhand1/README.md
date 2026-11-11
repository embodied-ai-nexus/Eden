# XHand1 Real-to-Sim Calibration

End-to-end system-identification workflow for the RoboTera XHand1 dex-hand
using `eden.extensions.sysid`. Drives the real hand with a known
excitation, then fits per-joint `damping`, `armature`, `frictionloss`,
and the PD gains `kp` / `kd` on the Eden sim twin so that the simulated
response matches the real robot under the same action trace.

## Files

| File                  | Purpose                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| `sim_twin_config.py`  | Minimal `EdenRLConfig` — single fixed-base XHand1 + `ExplicitPDController` + `SysIDRecorder`. `sim_dt * decimation` is tuned to the real-loop cadence (≈22.65 ms / 44 Hz). |
| `collect_data.py`     | Connects to the real XHand1 and records a chirp / PRBS trajectory to `.npz`.   |
| `identify.py`         | Loads one or more trajectories and fits `damping` / `armature` / `frictionloss` / `kp` / `kd`. Supports `--resume` to warm-start from a previous YAML. |
| `verify.py`           | Replays a held-out trajectory with identified params; reports RMSE improvement. |

The sim twin uses `ExplicitPDController` rather than the implicit variant:
the implicit controller adds `kd · substep_dt` to armature for numerical
stability, which silently injects a large amount of "phantom inertia" (≈100×
a real XHand joint) and pollutes identification. Explicit PD means the
fitted `armature` value *is* the rotor inertia.

**`kp` / `kd` must match across all three invocations.** They seed the
sim's PD controller and are read by `make_parameter_from_default` as the
initial guess — passing different values to `identify.py` vs `verify.py`
fits under one controller and replays under another.

## Prerequisites

- Eden installed (`uv pip install -e .`)
- `cma` package (`uv pip install cma`) — only required if you pass
  `--optimizer cmaes` (the default scipy backend has no extra deps)
- RoboTera XHand1 SDK installed and the hand connected via RS485 or
  EtherCAT. Confirm with `ls /dev/ttyUSB*`.

## Step 1 — collect an excitation trajectory on the real hand

The collection script centers each joint at the midpoint of its URDF
range, then excites symmetrically around that center. `--dofs-range-ratio`
is the fraction of the full joint range covered by the trajectory: `1.0`
spans lower-to-upper, `0.8` spans the middle 80 %, and `0.4` spans the
middle 40 %. This avoids the old failure mode where one-sided joints
started near their lower stop and clipped for half of every cycle.

The recorder checks this at startup and emits a per-DOF warning if the
commanded centered range would exceed URDF limits — watch the log for
`DOF '…' excitation range […] exceeds URDF limits` and reduce
`--dofs-range-ratio` if any joint is flagged.

```bash
# Identification trace (chirp sweep over all 12 joints, 30 s)
uv run examples/projects/sysid/xhand1/collect_data.py \
    --side left \
    --excitation chirp \
    --seconds 30 \
    --dofs-range-ratio 0.40 \
    --safety-limit 0.30 \
    --serial-port /dev/ttyUSB0 \
    --output data/xhand_chirp.npz
```

The excitation is added on top of the URDF midpoint with per-joint
peak offsets equal to `0.5 * dofs-range-ratio * URDF position range`,
then clamped to `±safety-limit` radians per joint. Chirp sweeps 0.2 → 3.0 Hz — broadband content
exposes both damping (high-frequency attenuation) and friction
(low-frequency stick-slip). All joints are phase-staggered by π so
simultaneous excitation doesn't produce a rank-1 input.

Record a second, independent trajectory for **holdout verification** —
PRBS is a good choice because its spectrum is orthogonal to chirp and
surfaces stiction clearly:

```bash
uv run examples/projects/sysid/xhand1/collect_data.py \
    --side left --excitation prbs --seconds 20 \
    --output data/xhand_prbs_verify.npz
```

Safety notes:
- The script refuses to run if `--dofs-range-ratio` is outside `[0, 1]` or if any per-side amplitude exceeds `--safety-limit`.
- `--kp-scale` / `--kd-scale` multiply the hand's firmware PD gains
  (`RoboTeraXHandDeployment.dofs_kp` / `dofs_kd`, rounded to integers)
  before `connect()` programs them. Lower `--kp-scale` (e.g. 0.5) is
  safer for first runs.
- The XHand SDK **does not honor the per-tick `kp` / `kd` fields** on
  the `RobotCommand` — gains are programmed once at `connect()`, so the
  scales set the PD stiffness for the whole recording, not per tick. For
  the same reason `eden deploy` refuses `--deploy_kp_kd_scale` and
  `--disable_joints` on the XHand.

### Timing and the sim twin

The real control loop on RS485 runs at whatever rate `read_state()`
tolerates — ≈44 Hz in practice (the SDK blocks on the serial round-trip).
`sim_twin_config.py` is tuned to match (`sim_dt = 0.005663`,
`decimation = 4` → `env.dt = 22.65 ms`). If you switch to a faster bus
(EtherCAT) or a slower poll rate, pass the same `--sim-dt` and
`--decimation` to `collect_data.py`, `identify.py`, and `verify.py` so
the sim twin replay cadence matches the recorded trajectory. `identify.py`
emits a warning when `trajectory.dt` and `sim_dt * decimation` differ by
more than 2 %.

## Step 2 — identify parameters

```bash
# Default (serial scipy, well-conditioned, fast on CPU for 60 params):
uv run examples/projects/sysid/xhand1/identify.py \
    --trajectory data/xhand_chirp.npz \
    --kp 50 --kd 5 \
    --output results/xhand_sysid2/
```

Fits five per-joint solver fields by default (`damping`, `armature`,
`frictionloss`, `kp`, `kd`) for 12 joints → **60 parameters**. `kp` / `kd`
are in the fit because the XHand SDK's internal PD gains are in
firmware-specific units that don't match Eden's conventional N·m/rad;
letting the optimiser find the effective sim-side gains is the cleanest
way to match the real closed-loop bandwidth.

The residual uses `dofs_pos` only, the one signal the XHand SDK measures
directly. `dofs_vel` is a finite difference of consecutive position reads in
`read_state()` — a control-period average, noisier and a period behind the
sim's instantaneous velocity — and trajectories recorded before that estimate
existed hold zeros there. `dofs_torque` is the firmware's own unsigned reading
in firmware units, not N·m. Including either would bias the normalised
residual.

### Parallelism — picking an optimiser

| Backend                         | When to use                                                        | `--num-envs`                  |
| ------------------------------- | ------------------------------------------------------------------ | ----------------------------- |
| `--optimizer scipy` (default)   | Serial TRF least-squares — well-conditioned, converges in <~50 iters when the model is right. Works on any hardware. | `1`  |
| `--optimizer scipy_parallel_fd` | Same TRF but the Jacobian columns are scored in one batched rollout. Good when one rollout is expensive and you have GPU headroom. | `n_params + 1` (= 61 here) |
| `--optimizer cmaes`             | Noisy / non-smooth residuals, rank-deficient Jacobians, or when scipy plateaus at a local minimum. Slower but globally robust. | population size (16–4096) |

```bash
# Parallel-FD scipy:
uv run examples/projects/sysid/xhand1/identify.py \
    --trajectory data/xhand_chirp.npz --kp 50 --kd 5 \
    --optimizer scipy_parallel_fd --num-envs 61 \
    --output results/xhand_sysid/

# CMA-ES with a wide population for global search:
uv run examples/projects/sysid/xhand1/identify.py \
    --trajectory data/xhand_chirp.npz --kp 50 --kd 5 \
    --optimizer cmaes --num-envs 1024 \
    --output results/xhand_sysid_cmaes/
```

CMA-ES prints per-generation progress with `best / mean / sigma / elapsed`
so you can see it's not stuck. On `gs.cpu` backend, each generation
takes ≈ population × rollout time — drop to `--num-envs 64` if CPU-bound;
bump to `--num-envs 1024+` on GPU.

### Useful flags

- `--include damping armature frictionloss kp kd stiffness` — add passive
  `stiffness` if residuals remain large after a first fit
- `--no-per-joint` — fit one shared scalar per property (5 params instead
  of 60) for a sanity check on short traces
- `--signals dofs_pos` — default; omit `dofs_vel` / `dofs_torque` because
  the SDK measures neither (velocity is a finite-difference estimate, torque
  is in firmware units)
- `--diff-step 1e-2` — relative FD step in unit-hypercube parameter
  space. 1 % of each bound range. Bump to `2e-2` to escape shallow minima,
  drop to `5e-3` for tight convergence near the optimum.
- `--max-iters 200` — increase if the optimiser hasn't plateaued.

### Fit multiple traces jointly

```bash
uv run examples/projects/sysid/xhand1/identify.py \
    --trajectory data/xhand_chirp.npz data/xhand_prbs_train.npz \
    --kp 50 --kd 5 --output results/xhand_sysid/
```

Chirp and PRBS have orthogonal spectra; fitting both jointly improves
conditioning and decorrelates damping from friction.

### Resume from a previous fit

```bash
uv run examples/projects/sysid/xhand1/identify.py \
    --trajectory data/xhand_chirp.npz --kp 50 --kd 5 \
    --resume results/xhand_sysid/params_identified.yaml \
    --output results/xhand_sysid_v2/
```

`--resume` warm-starts the optimiser from a saved `params_identified.yaml`
instead of the URDF defaults. Useful for:

- Adding a property (`--include … stiffness`) without throwing away the
  damping / armature / kp you already fit — new params start from
  nominal, existing ones start from the saved fit.
- Widening bounds (e.g. bumping `_BOUNDS["kp"][1]` in `identify.py`
  because the optimiser saturated against it last time) and continuing.
- Switching optimisers mid-flow — e.g. CMA-ES for global exploration,
  then scipy TRF for fast local refinement.
- Resuming a long run that was interrupted.

Behaviour on mismatch:

- **Bounds always come from the current `_BOUNDS`**, not the YAML. Loaded
  values outside the new bounds are clipped with a warning.
- **Missing params** (e.g. previous run didn't fit `kp`, this run does)
  start from the fresh URDF nominal, with a warning.
- **Shape mismatches** (switching `--per-joint` → `--no-per-joint`) skip
  the warm-start for that parameter and fall back to nominal.

Each edge case is logged so a silent partial warm-start can't slip
through; the final log line reports `warm-started N, skipped M`.

#### CMA-ES on warm-start: tighten `sigma0`

CMA-ES samples its first population from `N(x0, sigma0²)` in the
unit-hypercube parameter space, so a wide `sigma0` will disperse the
1024 candidates across the whole search domain and throw away your
warm-start. The default is automatically tightened when `--resume` is
set:

| Mode              | Default `sigma0` | Effective 1st-gen best-candidate distance from warm-start (60 params, pop 1024) |
| ----------------- | ---------------- | ------------------------------------------------------------------------------- |
| Cold start        | `0.25`           | ~19 % of the search diagonal — wide exploration                                 |
| `--resume`        | `0.05`           | ~4 % of the search diagonal — local refinement                                   |

Override with `--sigma0 <float>` if you want something in between (e.g.
`0.1` after widening a bound to re-explore the new region around the
previous fit). If best/mean in the first CMA-ES generation look like a
cold start even with `--resume`, check the `CMA-ES sigma0=... (warm-start)`
log line to confirm it's actually tight.

### Outputs under `--output`

- `params_identified.yaml` — fitted values
- `params_initial.yaml` — URDF defaults (useful for diff)
- `summary.txt` — per-parameter initial / identified / relative Δ table
- `plots/*.png` — measured vs **initial** vs **identified** predictions
  overlaid on trajectory 0 (three curves per DOF subplot)

## Step 3 — verify on the held-out trajectory

```bash
uv run examples/projects/sysid/xhand1/verify.py \
    --trajectory data/xhand_prbs_verify.npz \
    --params results/xhand_sysid/params_identified.yaml \
    --output results/xhand_sysid/verify/
```

Replays the trajectory twice under the sim twin — once with URDF
defaults, once with the identified YAML — and reports per-signal RMSE:

```
signal             baseline RMSE   identified RMSE   improvement
dofs_pos                0.031412          0.008771        72.1%
```

Exits non-zero if identification failed to improve any signal by more
than 1 %. `plots/*.png` overlays measured + baseline + identified for a
visual check.

## How it works

- `sim_twin_config.py` wraps a single fixed-base XHand1 with an
  `ExplicitPDController` acting on all 12 joints. `num_envs=1` for
  serial fits; set it equal to the CMA-ES population or `n_params + 1`
  for batched backends.
- `collect_data.py` calls `DeploymentRecorder.run()`, which writes
  `URDF_midpoint + excitation(t)` to the hand each control tick, reads
  the state, and finally serialises a `Trajectory` npz. Centering on
  the URDF midpoint keeps one-sided joints away from their stops.
- `identify.py` constructs a `ParameterSet` from the entity's current
  per-DOF values (URDF defaults) and then calls the selected backend's
  `fit()`. The underlying rollout bypasses the reward / termination /
  reset / command / event / recorder managers so that a long trajectory
  isn't corrupted by auto-reset and domain-randomisation events don't
  mutate the very parameters under optimisation.
- `verify.py` uses `ParameterSet.load_yaml` + `single_candidate_rollout`
  to apply the identified values and replay — it does **not** re-optimise.

## Troubleshooting

- **`RoboTeraXHandDeployment is not registered`** — the XHand SDK isn't
  importable in the current env. The extension skips registration
  silently when its deps are missing; install the SDK and re-run.
- **A joint's sim trace is flat across multiple runs** — the commanded
  trajectory is pushing below the joint's URDF lower limit. The real
  hand clips at the stop; the optimiser responds by locking the sim
  joint with high `frictionloss` and low `kp`. Watch the collection log
  for `excitation range [...] exceeds URDF limits` warnings, and
  reduce `--dofs-range-ratio`.
- **Improvement < 1 % on verify** — `kp` / `kd` at collection time don't
  match what you pass to `identify.py`, the trajectory is too short /
  narrowband (try a longer chirp + a PRBS trace fitted jointly), or
  `trajectory.dt` and `sim.dt` differ by more than 2 % (check the
  warning at the top of `identify.py`).
- **Optimiser terminates at iteration 0** — FD step is below the sim's
  numerical floor. Raise `--diff-step` (e.g. `2e-2`).
- **Optimiser plateaus at a local minimum with odd parameter patterns**
  (e.g., `stiffness` pushed high, `frictionloss` near 1 N·m on some
  joints, `armature` stuck at a bound) — the residual surface has a
  saddle. Switch to `--optimizer cmaes --num-envs 256`, which escapes
  saddles that trust-region gets stuck in.
- **`optimality` hasn't dropped but cost has** — normal near a degenerate
  manifold (e.g., kp/armature products). Accept and move on, or add a
  second trajectory for identifiability.
