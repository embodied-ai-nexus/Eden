# Eden - Embodied-AI Development Environments Nexus

Robotics RL simulation framework built on Genesis. Vectorized environments for robot learning (locomotion, manipulation, dexterous tasks, motion imitation). Declarative config via Pydantic, pluggable terms composed through registries.

**How to use this file.** It carries the rules that apply to every change, plus one-line reminders of the failure modes that are expensive to rediscover. The guides under `docs/guides/` are the authoritative prose for everything else — read the linked guide rather than re-deriving it from source.

## Development

- Use `uv run` for all Python execution (scripts, tests, linting).
- Always use absolute imports in non-`__init__.py` modules (`from eden.managers.reward_manager import ...`, not relative). Relative imports are fine in `__init__.py` for package re-exports.
- Circular imports are broken with `TYPE_CHECKING` guards, and managers import lazily inside `_load_managers()` — keep both patterns when adding a manager or a heavy import.
- **Testing**: tests mirror `eden/` (`tests/<path>/test_<module>.py`) and are Detroit style — real options, managers, entities and built envs, test doubles only at hardware/network/clock boundaries → `docs/guides/testing.md`. Run per-file: `uv run pytest tests/<path>/test_X.py`.
- **Geometry / quaternion / frame math**: import the helpers, never re-implement them at the call site. Prefer a `genesis.utils.geom` function with an exact torch equivalent; `eden.utils.geom` re-exports that quaternion/pose-transform family as the same objects and adds the rest (`quat_mul`, `subtract_frame_transforms`, …). If an op is missing from both, add it to `eden.utils.geom` and pin it in `tests/utils/test_geom.py` — do not inline a one-off. `genesis.utils.geom`'s numba/numpy-only converters are not torch drop-ins.

## Design Principles

- **Fail fast, fail loud.** Surface a misconfiguration at the earliest well-defined point — Pydantic validation, term `configure()`, or term `build()` — never deep inside a hot `compute()` step or mid-training. Resolve and validate handles (`entity_name`, link/DOF names, `func=`, `command_name`) in `build()` so a bad name raises at env-construction time with a message that names the offending value and the available options — not a `KeyError`/`AttributeError` on the first step, and never silently-wrong output.
- **Never mask a caller mistake.** Don't silently ignore an unknown param, swallow an exception, or fall back to a default that hides the error. Fix the root cause at the source, not the call site.
- **Write the message for whoever reads the traceback — increasingly an agent with no running process.** Quote the offending value, enumerate what *is* available, give the fix as something typeable, and name the plausible *wrong* fix (a `try`/`except`, a default) when one exists. Never put a newline in a `KeyError`, never let message-building code raise, and render the message from a script and read it before shipping. → `docs/guides/error-messages.md`
- **Lazy resolution still fails fast.** A term that must re-resolve an entity each step in `compute()` (deployment hot-swaps `env.entities[name]` — see `ManagerTermBase.resolve_entity`) should still *touch* that entity once in `build()` so a missing entity/link is caught at setup. Cache only derived data (resolved link/DOF indices), not the entity handle.

## DDP and SLURM

**Eden must be imported before genesis and torch** — `eden/__init__.py` isolates GPUs per rank before CUDA init. `import eden` itself imports neither (both load in `eden.init()`), so a module needing `genesis` must import it itself rather than relying on Eden having pulled it in. → `docs/guides/ddp.md`

On a SLURM cluster, write a job script from this template, fill the site-specific lines, and `sbatch` it from the repo root. One script covers one or many GPUs and nodes: `srun` starts one `torchrun` per node, `torchrun` starts one rank per GPU, and a world size of 1 is a plain run (a single-GPU-only algorithm refuses only when `WORLD_SIZE > 1`). `num_envs` is **per rank**.

```bash
#!/bin/bash
#SBATCH --job-name=eden
#SBATCH --nodes=1                  # >1 for multi-node; nothing else changes
#SBATCH --ntasks-per-node=1        # one torchrun per node — it spawns the per-GPU ranks
#SBATCH --gres=gpu:A6000:4         # site-specific GPU type; the count must equal NGPU below
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=general        # site-specific
#SBATCH --output=slurm/%x_%j.out   # mkdir -p slurm before sbatch: SLURM won't create it
#SBATCH --error=slurm/%x_%j.err
set -euo pipefail
NGPU=4

cd "$SLURM_SUBMIT_DIR"
module load cuda-12.9              # site-specific; drop it if the cluster has no modules
# `module load` puts the system CUDA ahead of torch's bundled libs, and the mixed cuBLAS/cuBLASLt pair fails
# batched-gemm IK with CUBLAS_STATUS_INVALID_VALUE: put torch's own NVIDIA libs first.
export LD_LIBRARY_PATH="$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib | paste -sd:):${LD_LIBRARY_PATH:-}"
# Assets download on first use into ~/.cache/eden/assets, which on a cluster is often a small home quota or a
# node-local disk (every node then fetches its own copy). Site-specific: a shared filesystem with tens of GB free.
export EDEN_ASSETS_DIR=/shared/$USER/eden_assets

# Several nodes: node 0 hosts the rendezvous store, on a port below the ephemeral range derived from the job id.
export RDZV_ENDPOINT="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1):$((20000 + SLURM_JOB_ID % 10000))"

# The bash -c body runs once per node. The rendezvous must be this job's alone: when torchrun has to guess
# whether it hosts the store and finds the port taken, it does not fail. It silently joins whatever store holds
# the port (another job's) and dies mid-run when that job ends. So one node uses --standalone (the OS picks a free
# port), and several name the host explicitly (is_host), which turns a taken port into an EADDRINUSE at launch.
# --kill-on-bad-exit then stops the other nodes instead of leaving them waiting out the join timeout.
# --cpus-per-task again: Slurm 22.05-23.02 do not pass sbatch's CPU count on to srun, so the step (and every
# rank in it) would be confined to one CPU. --no-sync: the ranks must not race to rewrite one venv.
srun --kill-on-bad-exit=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none bash -c '
    if [ "$SLURM_NNODES" -eq 1 ]; then
        rdzv=(--standalone)
    else
        rdzv=(--nnodes="$SLURM_NNODES" --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d
              --rdzv_endpoint="$RDZV_ENDPOINT" --rdzv_conf=is_host=$((SLURM_NODEID == 0)))
    fi
    exec uv run --no-sync torchrun "${rdzv[@]}" "$@"
' torchrun --nproc_per_node="$NGPU" \
    -m eden train --task <Task> --num_envs 4096   # or a script: drop -m and give its path
```

Track it with `squeue -j <id>` / `sacct -j <id>` and read `slurm/eden_<id>.out`; the run's logs and checkpoints land under `logs/` as usual.

## Where the Details Live

| Topic | Canonical source |
|-------|------------------|
| Task configs: hierarchy, task registry, every option group, strict fields, serialization | `docs/guides/task-config.md` |
| Task variants and their CLI flags | `docs/guides/task-mods.md` |
| Terms: function-vs-class, the `build()`/`compute()` lifecycle, which per-step hook state belongs in, caching, validation, registries | `docs/guides/terms.md` |
| What an error message must say, and the traps that make one unreadable | `docs/guides/error-messages.md` |
| Writing tests: layout, real collaborators vs. boundary doubles, what to assert, build cost | `docs/guides/testing.md` |
| Every registered term with its accepted params, types and defaults | `eden terms --grep <text>` (names, params **and** summaries); full page `docs/api/terms/catalog.md` |
| Entities: pose frames, DOF ordering, armature, contacts, closed-loop grippers, deformables | `docs/guides/entities.md` |
| Environment construction (`from_config`, `env_class_for_config`), the step loop, events | `docs/guides/environments.md` |
| Training runs, checkpoints, custom RL algorithms and where they live (`eden/learning/wrappers/<library>/<algorithm>`, never `examples/`), console log theme | `docs/guides/training-runs.md` |
| Training with another RL library — `gym.make_vec("Eden/<task>")`, the SB3 / skrl wrappers (one shared `VecEnvMixin`), CleanRL | `docs/guides/rl-libraries.md` |
| Porting a frozen policy from another stack | `docs/guides/policy-porting.md` |
| Exporting a trained policy for a runtime outside Eden (`eden export`: ONNX / TorchScript + `policy_spec.json`) | `docs/guides/policy-export.md` |
| Genesis sensors upstream doesn't ship (`eden/engine/`), and where a non-upstream solver goes instead | `docs/guides/engine-addons.md` |
| Scripted, demo-like demonstration datasets (`eden datagen`, OCBench port) | `docs/guides/scripted-demos.md` |
| Multi-GPU training / deployment (hardware and MuJoCo sim-to-sim) | `docs/guides/ddp.md`, `docs/guides/deployment.md` (per-robot setup: `docs/guides/deployment-hardware.md`) |
| One manager each: actions and controllers, observation groups / noise / history, rewards and terminations, commands, events and domain randomization, curricula, metrics and recorders | `docs/guides/{actions,observations,rewards-and-terminations,commands,events,curricula,metrics-and-recorders}.md` |
| Robot presets, composite rigs, actuators, adding a robot; scene presets, object libraries, terrains, splat scenes | `docs/guides/robots.md`, `docs/guides/scenes-and-terrains.md` |
| `EnvOptions` physics (integrators, solvers, precision, Mochi); cameras, renderers, sensors, video recording | `docs/guides/physics.md`, `docs/guides/sensors-and-rendering.md` |
| Motion tracking (`EdenMimicConfig`, motion reference, `eden preprocess`), dataset loaders, retargeting, pose estimation | `docs/guides/{motion-tracking,motion-datasets,retargeting,dex-retargeting,pose-estimation}.md` |
| Teleoperation devices, recording demonstrations, LeRobot conversion, planning and system identification | `docs/guides/{teleoperation,datasets-and-recording,planning-and-sysid}.md` |
| Shipped frozen policies (`eden inference --policy`), server-backed policy clients, adding a policy | `docs/guides/{policy-zoo,remote-policies,extending-policies}.md` |
| Every registered task by family, and the ported benchmark suites | `docs/tasks/index.md` |
| Every `eden` command and flag; every environment variable Eden reads | `docs/reference/cli.md`, `docs/reference/environment-variables.md` |
| The `eden` CLI command table and how to add a subcommand | module docstrings of `eden/__main__.py` and `eden/utils/cli/registry.py` |
| Camera recording contract | `Camera.start_recording` in `eden/entities/camera.py` |
| Painting task state the scene doesn't render (ink, marks) onto a camera's frames | module docstring of `eden/entities/camera_overlays.py` |
| Writing and running examples | `examples/AGENTS.md` |

Three facts worth having before you open any of them:

- **Config ↔ env pairing.** `EdenConfig`→`EnvBase`, `EdenRLConfig`→`RLEnvBase`, `EdenMimicConfig`→`MimicRLEnvBase`, and the leaf must match — a mismatch raises `TypeError` (e.g. `RLEnvBase.from_config(mimic_cfg)` would otherwise build with no motion reference). Construct with `from_config` (not `from_cfg`); use `env_class_for_config(cfg).from_config(cfg)` when the class isn't statically known. The env classes **construct only** — call `build()` and `reset()` yourself; the training wrappers (`RslRlVecEnvWrapper`, `RlGamesVecEnvWrapper`) build and reset inside `from_config`. → `docs/guides/environments.md`
- **Task configs** follow `eden/tasks/benchmark/reacher/config.py`. Tasks self-register with `@TASK_REGISTRY.register()` at the definition site; there is no central index to update.
- **Terms are resolved by registry key, never by object identity.** A class term that was never `@REGISTRY.register()`-ed configures fine, then fails to build. The key is always `to_snake_case(__name__)`, so a term registry's `register()` takes no `name=` — rename the term to change its key. (Asset/robot/task registries are looked up only by key, so `name=` there is a normal alias.)

## Critical Invariants

One line each — follow the link before acting on any of them.

- **Quaternions are wxyz** everywhere, config fields (`default_root_quat`) included.
- **`get_pos()` / `get_quat()` are not the URDF root** — they default to the CoM-re-origined solver frame (gap 2.9–14.8 cm on DexToolBench tools); `relative=True` gives the authored frame that `default_root_pos`/`default_root_quat` use and a ported policy expects. Setter defaults are asymmetric (`set_pos` `relative=False`, `set_quat` `relative=True`), and a fixed child link of a free DOF-less body carries an offset `relative` cannot strip. → `docs/guides/entities.md`
- **Two DOF index spaces, don't cross them.** *Column* = position in the entity's resolved `dofs_name` (the configured order, not the asset's): `default_dofs_*`, `soft_dofs_pos_limits`, `get_dofs_limit` and every full-width `get_dofs_*()` read; `get_jacobian` too, but behind six base columns unless `entity.has_immobile_base` (joint column `c` is Jacobian column `c + 6` on a floating base). *Local* = Genesis tree order, shifted by a free base: only ever a `dofs_idx_local=` argument. Resolve once, in `build()`, with `find_named_dofs(names)`, and hand its `.columns` to tensors and its `.dofs_idx_local` to Genesis calls; `get_default_dofs_*` reads a property by name, in `build()` (each call copies). → `docs/guides/entities.md`
- **Asset loading defaults `armature=0.1`** on every single-hinge/slide DOF whose asset armature is zero (every URDF joint), filled at `scene.build()` — often ~30× too high for low-stiffness fingers. Set `default_dofs_armature` from the source asset, or `default_armature=0.0`. → `docs/guides/entities.md`
- **Implicit damping mutates armature — under an explicit integrator only** (`post_build()` adds `kd * substep_dt` under `EnvOptions.integrator="euler"` only; `approximate_implicitfast`, the default, and `implicitfast` integrate implicitly only the `kd` of DOFs Genesis drives in position/velocity mode). A torque-mode term (`control_dofs_force`: `ExplicitPDController`, `OperationalSpaceController`, …) computes `-kd·q̇` itself, so its damping is explicit under every integrator. Audit the built `get_dofs_armature`. → `docs/guides/entities.md`
- **Contact friction combines as `max`**, not min/average/product. Torsional/rolling friction are opt-in per scene (`EnvOptions.enable_*_friction`) and parse to **0** for MJCF geoms without `condim ≥ 4` — set `friction_torsional`. → `docs/guides/entities.md`
- **Closed-loop grippers (Robotiq 2F85) need an effective substep dt ≤ 0.001.** Stiffen graspable objects per-object via `default_geoms_sol_params` — **never** a global `EnvOptions.constraint_timeconst`, which ejects mesh objects on contact. → `docs/guides/entities.md`
- **Gravity compensation is a material field**: `RigidMaterialOptions(gravity_compensation=1.0)` — not a morph, `RigidOptions`, or `add_entity` argument.
- **`ParticleEntity.set_pos()` / `set_quat()` raise** (MPM/SPH/PBD), as does `FEMEntity.set_quat()` — bake the pose into the morph at creation time.
- **Genesis' URDF parser crashes on `<inertial><density>`** (no explicit `<mass>`/`<inertia>`). Use an asset with explicit mass and inertia.
- **Fetch an entity's or policy's file through its owner — `options.fetch_asset(file)`, `MyPolicy.fetch_asset(file)` — never `get_asset_path(..., local_dir=options.local_dir)`** — an unset `local_dir` means the family's `assets_subdir` under `get_assets_dir()`, resolved at load time, so the raw `None` silently fetches into the asset root instead. A new asset family sets `assets_subdir`, never a `local_dir` default; a policy's `registry`/`dataset`/`local_dir` are class constants, not options. Two source scans in `tests/options/test_entities.py` refuse both mistakes. → `docs/guides/entities.md`
- **Auto-reset runs before observation.** In `RLEnvBase.step()` resets happen after reward/termination, so the returned observations for done envs reflect the fresh post-reset state. → `docs/guides/environments.md`
- **Grading state goes in `post_physics_refresh`, not `_update_command`** — the reward reads at phase 9 and `_update_command` publishes at 17, so state written there is a step late and state cleared there is never read; a `reward_*` shadow buffer is the smell. Deployment never runs `post_physics_refresh`. → `docs/guides/terms.md`
- **Observation `compute()` is a read, not a tick** — it runs at build, reset, several step phases and from `get_observations()`, so a counter advanced inside it is silently wrong. Per-step state advances in `ObservationTerm.post_physics_refresh()` (phase 7); function terms are stateless by construction. → `docs/guides/terms.md`
- **A Mochi scene is all-Mochi.** The MochiSolver (Eden-hosted on stock Genesis: `eden/engine/solvers/mochi`, installed per scene on its first Mochi entity) assembles every body into one Newton system, so every entity — ground plane included — takes a `Mochi*MaterialOptions`; `SceneOptions` rejects the mix. It needs `en.init(precision="64")` at the default 1e9 Pa/m contact stiffness, refuses `requires_grad`, and rejects `default_dofs_frictionloss` / `default_dofs_force_limits` / sol params / support surfaces at validation. Per-env DOF info is `MochiOptions.batch_dofs_info`. → `docs/guides/entities.md`
- **Camera recording is Eden-owned.** Use `Camera.start_recording(video_path, fps)` / `stop_recording()`; never delegate to `gs.Camera.start_recording` (records nothing for kinematic replay), and don't inline `Camera._render()` into `render()` (double capture). → `eden/entities/camera.py`
- **`extra="forbid"` is the default** for `ConfigurableOptions`; only `DynamicConfigurableOptions` subclasses allow extras. → `docs/guides/task-config.md`
- **`load_from_file`'s subclass rule is not task identity** (`ManipulationLiftBoxRGBConfig` is a `ManipulationLiftBoxConfig`) — compare by exact class when a saved config has to *be* a named task's. → `docs/guides/task-mods.md`
- **`import eden` imports no term and no options namespace** — they re-export lazily, and each registry names the modules that fill it (`Registry(populated_by=...)`). A new term registry must list **every** module its terms are defined in (`TERMINATION_TERM_REGISTRY` also names `terms.metrics`), or a lookup reads as "no such term". `eden.utils.*` ↔ `eden.options.*` is an import cycle in both directions: a utils module may import `eden.options` at module scope only if nothing in `eden.options` imports *it* at module scope; defer the options→utils edge (see the `default_factory` in `eden/options/managers/recorders.py`).
- **`eden/entities/support_mixin.py` cannot use `from __future__ import annotations`** — its `@qd.kernel` annotations must stay real objects.
- **A checkpoint's `.pt` layout belongs to `rsl-rl-lib`, not to Eden** — 3.x and 4.x/5.x differ and the pin admits both, so a checkpoint may not replay under the same Eden. → `docs/guides/training-runs.md`
- **Bit-exact obs/action parity does not mean a ported policy is correct** — mismatched dynamics (armature, gains, friction, control rate) diverge the trajectory; audit the *built* robot. → `docs/guides/policy-porting.md`
