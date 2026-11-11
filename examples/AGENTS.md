# Working in `examples/`

This directory is Eden's runnable, verifiable capability catalog — the ground truth
that `docs/` narrates. It exists to serve two readers:

1. **A newcomer or coding agent building a task** — fork the closest template, pull
   recipes for the novel parts, follow Eden conventions, verify.
2. **A lookup dictionary** — "how do I use *X* in Eden" answered by one small script.

## Three tiers

| Tier | Home | What lives here |
|------|------|-----------------|
| **projects** | `examples/projects/<area>/` | End-to-end, trainable, forkable task templates (`train` / `play`). |
| **policies** | `examples/policies/<name>/` | Run a pretrained checkpoint in sim; most support `--deploy` to hardware. |
| **recipes** | `examples/recipes/<capability>/` | Minimal, single-capability scripts. One concept each. |

## To build a new task (the loop)

1. **Fork** the closest `projects/<area>/` template — it wires config → env → runner correctly.
2. **Pull recipes** for the novel parts (a sensor, a solver, a teleop device, a term).
3. **Follow the conventions** below; write terms per the `writing-eden-terms` skill.
4. **Verify** with a smoke run: `eden smoke --task <name>` for a task, `eden examples smoke <key>`
   for an example (see [Smoke-testing](#smoke-testing) below).

Task-config, term and entity conventions live in the root `AGENTS.md` and the guides it links
(`task-config.md`, `terms.md`, `entities.md`) — follow them there; this file covers only what is
specific to `examples/`.

## Command-line convention

Every example parses its arguments the same way, so learning one script's CLI teaches all
of them. Ported from [genesis-world#3104](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3104),
with Eden's GPU-first default.

**Short flags are a closed, reserved set.** Each letter means exactly one thing across the
whole tree. A flag that is not in this table gets no short form.

| Short | Long | Type | Default |
|-------|------|------|---------|
| `-v` | `--vis` | `store_true` | off |
| `-c` | `--cpu` | `store_true` | off (GPU) |
| `-b` | `--num-envs` | `int` | per script |
| `-s` | `--steps` | `int` | per script |
| `-t` | `--seconds` | `float` | per script |
| `-e` | `--exp-name` | `str` | per script |
| `-r` | `--record` | `store_true` | off |
| `-o` | `--output-dir` / `--output` | `str` \| `Path` | per script |
| `-d` | `--debug-vis` | `store_true` | off |

Rules:

1. Short flag first, then long: `parser.add_argument("-v", "--vis", ...)`.
2. Long names use dashes between words (`--num-envs`); argparse exposes them as the
   underscored `args.num_envs`, so no `dest=` is needed.
3. Single-dash flags are one character.
4. Booleans are `store_true` and default off, which makes `default=False` redundant. A
   viewer-first script — one whose whole point is the interactive window — instead takes a
   single `--no-vis` (`store_false`, `dest="vis"`) and never offers `-v`. Such a script
   must pass `show_viewer=args.vis and not en.is_smoke()`: the smoke harness runs it with
   no arguments, so an ungated viewer dies on `No display detected` wherever there is no
   display. `tests/utils/test_example_flags.py` enforces this.
5. Use `-s/--steps` when the script's natural unit is iterations, `-t/--seconds` when it
   reasons in simulated time.
6. `-o` means "where this run writes its output": `--output-dir` for a directory,
   `--output` for a single file. A script exposes at most one of the two.
7. Every flag carries a `help=`. Keyword order is `action=` / `type=`, then `default=`,
   then `help=`.
8. The parser is named `parser` and is built as the first statements of `main()`.
   Subparsers keep a descriptive name (`train_parser`, `inference_parser`).
9. A registered task is constructed with `TASK_REGISTRY.build(name)` — never
   `TASK_REGISTRY.get(name)()`, and never by importing the config class and calling it
   (`ReacherConfig()`). Both give the bare class, and instantiating that runs none of the
   task's TaskMods: a mod-bound task builds a config that dies in `env.build()` on
   an empty link name, and a mod added later is silently skipped. This holds for a
   fixed task as much as for `--task <name>`. `tests/utils/test_example_flags.py` enforces this.

**Backend.** Eden is GPU-first: `gs.init()` prefers a GPU and the RL envs are built to
batch, so an example runs on GPU unless `-c/--cpu` is passed. A script that genuinely
cannot run on one hardcodes the backend and says why in a comment rather than exposing a
flag with a single usable value.

`-e/--exp-name` is reserved but not yet used by any example; training scripts should adopt
that spelling rather than inventing another.

**Chaining one subcommand into another.** A training script that renders a video of the
policy it just trained hands its own `inference` a set of arguments. Build them by
re-invoking the parser — `inference(parser.parse_args(["inference", "--record", ...]))` —
never by hand-building an `argparse.Namespace`. A hand-built one has to restate argparse's
dest names, and drifts from them silently: `Namespace(rec=True, out=None)` (the short-flag spellings)
handed to an `inference` reading `args.record` / `args.output` kills the promised video on
`AttributeError` *after* the whole training run has been paid for.
Round-tripping the CLI cannot drift, and picks up new flags' defaults for free.

**Validate.** The convention is machine-checked, including that every short flag quoted in
a docstring or README is one the referenced script really accepts — a renamed flag with a
stale usage line is otherwise invisible, since examples are smoke-run with no arguments.

```bash
uv run pytest tests/utils/test_example_flags.py   # reserved shorts, dash-case, help=, live docs, headless-safe,
                                                 # bundled path defaults, chained args, TASK_REGISTRY.build
```

## Where the training algorithm lives (env vs. algorithm)

A task ships its env in `eden/tasks/`; how it is solved is a separate choice that lives in
`eden/learning/` (one subpackage per RL library under `eden/learning/wrappers/`). Every registered
task that trains should train with `eden train --task <name>` — keep algorithm code out of
`examples/`: a saved `config.json` names its algorithm/model/runner by import path, and only a path
inside the package resolves for `eden train --resume` / `eden inference`.

- **Stock algorithm** (rsl_rl PPO, rl_games): set `runner_options` on the task config.
- **Non-stock model or algorithm** (a custom actor, an extra loss, a discriminator): add it under
  `eden/learning/wrappers/<library>/<name>/` with an options class whose `class_name` defaults to it,
  and select it in the task's `runner_options` (`actor=` / `critic=` / `algorithm=`). Keep it
  task-agnostic: what ties it to one task (an observation layout, a mirror map, an asset path) comes
  in through its options — e.g. `RslRlRmaPpoAlgorithmOptions(mirror_class_name=...)` names the
  parkour task's mirror in `eden/tasks/extreme_parkour/symmetry.py`.
- **Non-stock loop** (a distillation that backpropagates through the rollout): ship a runner there
  too and name it in `runner_options.class_name` by import path (see "Custom runners" in
  `docs/guides/training-runs.md`).
- **External backend** (a fork worth its own repo, e.g. `rsl_rl_contrack`): Eden keeps the seam in
  `eden/learning/wrappers/<fork>/` — options, VecEnv wrapper, runner — and the runner module raises
  with the install command when the backend is missing. The task stays `eden smoke`-runnable
  without it.

An example then only adds what `eden train` / `eden inference` do not: playback with a viewer,
custom terrain, recordings, start-phase probes.

## Example header (required on every runnable example)

Every entry-point script (`main.py`, single-file recipe/project) carries a metadata block
in its **module docstring**: a one-line summary, then a `---`-fenced YAML block, then the
usual prose/usage. The block is the machine-readable source for the examples catalog
(`eden examples list`) and the smoke gate — it is AST-parsed, never imported.

```python
"""
LiDAR / DepthCamera sensor on a Go2 robot surrounded by obstacle rings.

---
tier: recipe
capabilities: [sensor.lidar, sensor.depth_camera, robot.go2]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/sensors/sensors_lidar_go2.py -v
---

Usage:
    uv run python examples/recipes/sensors/sensors_lidar_go2.py -v --pattern grid
    uv run python examples/recipes/sensors/sensors_lidar_go2.py -c -s 50   # CPU, 50 steps
"""
```

**Rules.**
- The **first non-empty docstring line** is the example's one-line purpose (what, not how).
- Exactly one `---`…`---` YAML block, immediately after the summary. `yaml.safe_load`-able.
- `run` must be a copy-pasteable command with the **correct current path**.
- `hardware` describes the **canonical `run` command**. When a flag or `--task` raises the
  requirement (a CUDA-only vision task in an otherwise-CPU driver), keep the tag at the canonical
  run's level and reject the unsupported invocation up front — see the `--student --cpu` check in
  `projects/locomotion/extreme_parkour/main.py` and `_require_gpu_for_vision_task` in
  `projects/manipulation/main.py`. A `--cpu` flag whose only possible outcome is a crash deep
  inside the build must not exist.
- Library/helper modules that aren't run directly (e.g. `hand_presets.py`,
  `single_hand_detector.py`, `sim_twin_config.py`) do **not** carry a block.

### Fields

| Field | Required | Value |
|-------|----------|-------|
| `tier` | yes | `project` \| `policy` \| `recipe` |
| `capabilities` | yes | list of `namespace.leaf` tags (closed namespaces, snake_case leaves) |
| `hardware` | yes | `cpu` \| `gpu` \| `gpu-fp64` \| `real-robot` |
| `optional_deps` | no (default `[]`) | list from the optional-dep set below |
| `run` | yes | one canonical copy-paste command |

### Capability namespaces (closed set; leaves are snake_case)

`solver` (rigid, mpm, sph, pbd, fem, ipc, sap) · `sensor` (lidar, depth_camera, imu, contact,
camera) · `robot` (go1, go2, g1, franka, xhand, ur5e, wx250s, aloha, …) · `teleop` (keyboard,
gello, spacemouse, ps4, metaquest, pico, wuji_glove, gui) · `dataset` (amass, arctic, dex_ycb, lafan1, omomo,
phuma, seed, hhi_assist, bridge, droid) · `pose` (gvhmr) · `retarget` (hand, human, soma) · `term` (reward,
observation, termination, command, event, curriculum, taskmod, profiling) · `render` (nyx,
viser, green_screen, splat, camera) · `planning` (ompl, curobo, graspgen, graspgenx) · `task` (locomotion,
manipulation, mimic, reaching, sysid, dexterous) · `scene` (libero, placement,
heterogeneous) · `deploy` (real_robot) · `collision` (mask, contype) · `terrain` (procedural) · `rl`
(gymnasium, sb3, skrl, cleanrl — training with an RL library other than `eden train`'s).

A physics coupling demo carries every solver it uses, e.g. `[solver.mpm, solver.rigid]`.
Add a leaf freely; add a **namespace** only by updating this list and `CAP_NAMESPACES` in
`eden/utils/examples_catalog.py` (the validator's source of truth) together.

### Optional-dep set

`libuipc` · `madrona` · `curobo` · `ompl` · `graspgen` · `graspgenx` · `metaquest` · `pico` · `gello` ·
`wuji` · `spacemouse` · `viser` · `nyx` · `onnx` · `metamotivo` · `sb3` · `skrl`.

The set is `OPTIONAL_DEPS` in `eden/utils/examples_catalog.py`, which binds each tag to the
module the smoke gate imports to prove it is installed **and** to a one-line install hint the
skip message shows (`optional dependency 'x' not installed (pip install ...)`) — add all three
together. The hint is the documented home for deps that cannot be a pip extra (source builds,
git URLs, wheels bundled with another package). Declare a dep the example reaches
**transitively**: the ONNX-backed policy examples import a policy class, not `onnxruntime`,
and an undeclared transitive dep is reported as a hard failure rather than a skip.

### Validate

```bash
uv run pytest tests/utils/test_examples_catalog.py   # every runnable example parses + closed vocab
```

## Smoke-testing

An example is *smoke-runnable* when it makes its step horizon tiny in smoke mode:

```python
horizon = 5 if en.is_smoke() else 1000     # en.is_smoke() honors EDEN_SMOKE / PYTEST_VERSION
```

Instrument every new headless recipe this way. Then:

```bash
uv run eden examples smoke recipes/physics/mpm_tutorial      # run one headless, a few steps
uv run eden smoke --task reacher                             # build a registered task and step it
uv run pytest tests/utils/test_example_smoke.py -k recipe_smoke  # the whole recipe suite (GPU-gated)
```

`eden smoke` takes the task by flag (`--task <name>`), not positionally — it shares the task
argparser with `train` / `inference` (`--cpu`, `--config`, `--override`, and the task's own
TaskMod flags), and adds `--num_envs` / `--steps` / `--eval_mode` / `--show_viewer` on top.
`eden examples smoke`, by contrast, takes the example key positionally.

`test_recipe_smoke` in `tests/utils/test_example_smoke.py` is manifest-driven: it uses each example's `hardware` /
`optional_deps` / capability tags to decide what it can run here, and **skips with a
visible reason** anything that needs real hardware, a teleop device, a missing optional
dep, an absent GPU, or isn't smoke-instrumented — so coverage gaps are never silent. It
is meant to run locally where the GPU/deps exist, not on the free CI plan. Training
projects/policies (checkpoints/data/hardware) are out of scope for auto-smoke.
