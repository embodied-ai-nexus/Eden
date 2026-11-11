# Exporting a Policy for Your Own Runtime

`eden deploy` runs a checkpoint on a robot, but it needs Eden installed on the robot's computer. To run a trained
policy on your own stack instead (ROS 2, a Jetson, a C++ controller), export it with `eden export`. The export is a
graph that any ONNX or TorchScript runtime can load, plus a `policy_spec.json` that describes the observation the
graph expects, how its output becomes joint targets, and the control rate. This page explains the command, the
graph's inputs and outputs, and every field of the spec.

```bash
eden export --task <task> --checkpoint logs/<task>/<run>/model_1500.pt
```

```text
logs/<task>/<run>/exported/model_1500/
├── policy.onnx         # ONNX graph
├── policy.pt           # TorchScript module
└── policy_spec.json    # the contract around the graph
```

## The command

| Flag | Default | What it does |
|------|---------|--------------|
| `--task TASK` | required | The task the checkpoint was trained on. |
| `--checkpoint PATH` | required | An rsl_rl `model_<iteration>.pt`. |
| `--format FORMATS` | `onnx,jit` | Comma-separated: `onnx` writes `policy.onnx`, `jit` writes `policy.pt` (TorchScript). |
| `--out DIR` | `exported/<checkpoint name>/` beside the checkpoint | Where the files go. An earlier export there is replaced, including a graph of a format not exported this time. |
| `--no_verify` | off | Skip the parity check (below). |
| `--config`, `--override`, `--cpu` | | As for every task command. |

The ONNX graph needs the `onnx` package, and its parity check needs `onnxruntime`. Install both with the `onnx` extra:
`pip install 'eden-world[onnx]'`, or `uv sync --inexact --extra onnx --reinstall-package onnxruntime` in a checkout
(`eden export --help` prints the full command for your install, including the step that repairs an `onnxruntime`
wheel the GPU extra overwrote). A missing package is reported before anything is built; `--format jit` needs neither.

How the env is built:

- From the checkpoint's own `config.json`, with `--config` / `--override` on top, exactly as
  [`eden inference`](training-runs.md#replay-a-checkpoint) builds it. One env, in eval mode.
- Event terms that bias the default joint pose at startup (`RandomizeStartupDofsPosBias`) are disabled first, because
  the spec records that pose. An env built with one is refused by the export function itself.

What exports:

- Only **rsl_rl** checkpoints. An rl_games checkpoint is refused; export it with rl_games' own tooling.
- Only under **rsl-rl-lib 4 or newer**, whose models have export hooks. A 3.x checkpoint loads into a 4.x / 5.x
  runner, so upgrade rather than export under 3.x.

## The graph

| | `policy.onnx` | `policy.pt` |
|---|---|---|
| input | `obs`, shape `(batch, obs_dim)` | `obs`, shape `(batch, obs_dim)` |
| output | `actions`, shape `(batch, action_dim)` | the actions |
| recurrent state | explicit: `h_in` → `h_out` (and `c_in` → `c_out` for an LSTM), shaped `(num_layers, batch, hidden)`; start from zeros and feed each output back as the next input | inside the module, batch 1; call `reset()` at the start of each episode |

- **`obs` is raw.** It is the actor's observation groups concatenated in the order the spec lists them. The
  observation normalizer is baked into the graph, so do not normalize the input yourself.
- **`actions` is the raw policy output**, before any action term scales it. The deterministic mean is exported, never
  a sample.
- **The ONNX batch axis is dynamic.** TorchScript's is dynamic only for a non-recurrent policy.

### Parity check

Before writing the spec, the export steps the env for a few control steps with the policy `eden inference` runs, then
replays those observations through every exported graph, threading the recurrent state, and fails if any action
disagrees (tolerance `1e-4`, since the graph runs on the CPU and the reference on the env's device). This catches a
learner whose export hook exports something other than what its model runs. The files are written to a staging
directory and moved into `--out` only after the check passes, so a failed export leaves `--out` as it was. Keep the
check on unless you have no `onnxruntime`.

### Learners with their own graph

The graph comes from the policy model's own `as_onnx()` and `as_jit()` hooks; rsl-rl-lib's `MLPModel` and `RNNModel`
define them. (Before rsl-rl-lib 5.5.2 the ONNX hook was `as_onnx(verbose)`; the export passes `verbose=False` only to a
hook that declares it, so a model written against either form exports.) A learner whose deployed policy is more than
an MLP over the observation defines both hooks on its model, each returning a module that runs the deployed
composition on the flat observation, and sets `obs_groups` (the groups it concatenates). For an example see
`RmaActorModel` in `eden/learning/wrappers/rsl_rl/rma/model.py`, which injects its estimator's prediction. A model
with neither hook (HIMLoco's `HIMActorCritic`) is refused, and the error names what to define.

## `policy_spec.json`

Every per-DOF list in the spec is in the entity's **column order**: its configured `dofs_name`, given under
`entities`, not the order of the asset file. The top-level keys are `schema_version`, `model`, `observations`,
`actions`, `entities`, `rate` and `provenance`. `schema_version` changes whenever a consumer would have to change with
it.

### `model`

The exported files, with the ONNX graph's tensors as it declares them (a dynamic axis appears as `"batch"`):

```json
{"onnx": {"file": "policy.onnx",
          "inputs": [{"name": "obs", "shape": ["batch", 25]}],
          "outputs": [{"name": "actions", "shape": ["batch", 6]}]},
 "jit": {"file": "policy.pt"}}
```

### `observations`

```json
{"dim": 25, "groups": [{"name": "policy", "slice": [0, 25], "history_length": 0, "backfill": null,
  "nan_policy": "disabled", "terms": [
  {"name": "dofs_pos", "term": "dofs_pos", "slice": [0, 8], "frame_dim": 8, "history_length": 0,
   "backfill": null, "clip": null, "scale": 1.0, "modifiers": [],
   "config": {"entity_name": "robot", "...": "..."}}]}]}
```

To build `obs`, compute each term and lay it into its `slice` of `obs`. Groups follow each other in `groups` order,
and each term's slice is relative to its group. Process a term's frame in this order:

1. **Compute** the term. `term` is its registry key; `eden terms --grep <term>` documents what it measures, and
   `config` holds the parameters it was configured with (entity, link and DOF names, frames).
2. **Apply `modifiers`.** These also run on hardware, so a delay or a filter listed here must be reproduced.
3. **`clip`**, then multiply by **`scale`**. A list scale is per component.
4. **Apply the group's `nan_policy`.** Under `sanitize` and `warn`, NaN and ±inf become 0. Under `error`, Eden raises
   on them; under `disabled` they pass through.
5. **Stack the history.** With a term `history_length` H > 0, the term's slice holds its last H frames, oldest first,
   each `frame_dim` wide. A group with its own `history_length` stacks whole group frames instead: the term slices
   then index one frame, and the group's block is H frames, oldest first. At an episode start the window is filled
   with copies of the first frame when `backfill` is true, and with zeros when it is false. `backfill` is `null`
   where there is no history.

Training noise is never part of the contract: the export reads the env in eval mode.

### `actions`

```json
{"dim": 6, "clip": null, "terms": [
  {"name": "joint_pos_controller", "term": "implicit_pd_controller", "class": "ImplicitPDController",
   "slice": [0, 6], "entity_name": "robot", "target": "position", "reference_source": "default",
   "dofs_name": ["joint1", "..."], "scale": 0.25, "offset": [0.0, "..."], "mimic_dofs": {},
   "config": {"...": "..."}}]}
```

1. If `clip` is `[lo, hi]`, clamp the raw `actions` to it (the runner's `clip_actions`).
2. Each term reads its `slice`. A joint controller turns its first `len(dofs_name)` columns into a `target` (a
   position or a velocity) for `dofs_name`: `target = action * scale + offset + reference`, where `reference` follows
   `reference_source`:

    | `reference_source` | `reference` |
    |---|---|
    | `zero` | 0 |
    | `default` | 0: the default pose is already folded into `offset` |
    | `delta` | the joint positions measured at the start of the control step |
    | `accumulate` | the previous step's target, clamped to the soft joint limits (`offset` is 0) |
    | `motion_ref` | the reference motion's joint positions, which only an Eden motion manager provides |

3. A follower in `mimic_dofs` (`follower: [driver, multiplier, offset]`) is commanded
   `driver_target * multiplier + offset`.
4. A term with `velocity_feedforward` (`VelocityFeedforwardPDController`) has a second block of columns at
   `velocity_feedforward.slice`, a target velocity for the same DOFs:
   `target_velocity = action * velocity_feedforward.scale`. The PD law is then
   `kp * (target - q) + kd * (feed_forward_ratio * target_velocity - q̇)`.

`offset` is the nominal one. A per-env joint-zero offset drawn by domain randomization (`RandomizeDofsPosOffset`) is
never part of it.

Other action terms (task-space controllers, binary grippers) carry only their `class` and `config`, as does a joint
controller whose columns do not map one-to-one onto DOFs as above. A runtime implements those itself.

### `entities`, `rate`, `provenance`

- **`entities.<name>`**: `dofs_name` in column order, with `default_dofs_pos`, `kp` and `kd` as built for training.
  Hold the DOFs that no action term drives at their default pose with these gains.
- **`rate`**: `control_dt` (run the policy once every `control_dt` seconds), with the `decimation` and `sim_dt` it was
  built from.
- **`provenance`**: the task, its config class, the checkpoint path and SHA-256, the rsl-rl-lib version
  (`rl_library`) and the torch version. `eden` and `genesis` each record a `version` and a `commit`, `null` when
  unknown: Eden's commit is the `HEAD` of the clone it ran from (`null` for an installed package), Genesis's the one a
  `git+` or editable install records. `eden.dirty` says whether that clone had uncommitted changes.

## Loading an export back into Eden

`OnnxModule` from `eden.extensions.policies.onnx` runs a non-recurrent ONNX export on a batch of observations, which
is a quick way to check an export inside Eden:

```python
import json

from eden.extensions.policies.onnx import OnnxModule

spec = json.load(open("exported/model_1500/policy_spec.json"))
policy = OnnxModule(
    "exported/model_1500/policy.onnx", "obs", "actions",
    expected_input_dim=spec["observations"]["dim"],
    expected_output_dim=spec["actions"]["dim"],
)
actions = policy(obs)   # (batch, obs_dim) -> (batch, action_dim)
```

## See also

- [eden.learning.export](../api/learning.md): `export_policy` and `build_policy_spec`
- [Training Runs](training-runs.md): checkpoints and their `config.json`
- [Deployment](deployment.md): running a checkpoint on hardware with Eden itself
- [Observations](observations.md) and [Actions](actions.md): the terms the spec describes
