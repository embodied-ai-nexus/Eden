# Quick start

This page trains a policy, watches it, and then takes it further: export, a MuJoCo sim-to-sim check, and a
published policy you can run without training. It uses `reacher`, the smallest built-in task: a 6-DOF Yam arm
learns to move its gripper to target points sampled in front of it.

You need Eden installed ([Installation](installation.md)). The commands default to the GPU; add `--cpu` to any of
them on a machine without one.

## 1. Find a task

```bash
eden list
```

This prints every registered task with its source (`eden` for a built-in task) and the task mods that give it
variant flags. `reacher` has one mod, `ReacherDifficultyMod`, which adds `--difficulty`. To see every flag a task
accepts, its variant flags included:

```bash
eden train --task reacher --help
```

[Task suites](../tasks/index.md) describes the built-in tasks by family.

## 2. Train it

```bash
eden train --task reacher
```

`reacher` builds 4096 parallel environments and runs 100 PPO iterations with rsl_rl (its `runner_options` set
`max_iterations=100`). The first run compiles Genesis' kernels before training starts and caches them for later
runs, and it downloads the Yam arm's model files ([Assets](assets.md)). While training, rsl_rl prints a summary
per iteration; the mean episode reward should rise as the arm learns to reach.

<!-- TODO(material): wall-clock time of the first compile and of the 100 iterations on a named GPU. -->

The run writes to a new directory, `logs/reacher/<timestamp>-<id>/`, relative to the directory you ran the command
from:

| File | What it is |
|---|---|
| `model_<iteration>.pt` | The policy checkpoint. Only the newest is kept (`keep_last_checkpoints=1` by default). |
| `config.json` | The full config the run trained with. `eden inference`, `eden export` and `eden deploy` rebuild the task from it. |
| `run.json` | The seed, the command line and the Eden and Genesis versions, for reproducing the run. |
| `progress.json` | A small summary of the run's progress, rewritten every iteration. |
| `events.out.tfevents.*` | TensorBoard logs: `tensorboard --logdir logs/reacher` |

Useful flags (all of them are in `eden train --task reacher --help`):

```bash
eden train --task reacher --num_envs 1024                    # fewer environments, e.g. for a smaller GPU
eden train --task reacher --difficulty hard                  # a variant flag from the task's mod
eden train --task reacher --override 'runner_options: {max_iterations: 300}'
eden train --task reacher --resume latest                    # continue the newest run, in a new run directory
eden train --task reacher --logger wandb                     # log to Weights & Biases instead of TensorBoard
eden train --task reacher --run_name first_try               # logs/reacher/<timestamp>-first_try-<id>/
eden train --task reacher --cpu --num_envs 64                # no GPU
```

`--override` takes a YAML mapping that is merged into the task's config, and can be repeated. `--config <file>`
does the same from a `.json` or `.yaml` file.

## 3. Watch the trained policy

Point `eden inference` at a checkpoint. This picks the newest one:

```bash
eden inference --task reacher --checkpoint "$(ls -t logs/reacher/*/model_*.pt | head -n1)" --vis
```

`--vis` opens the Genesis viewer and runs one episode in one environment. `eden inference` loads the
`config.json` next to the checkpoint, so it rebuilds the task exactly as it was trained, in evaluation mode. For
the same reason a variant flag such as `--difficulty` cannot be combined with a checkpoint that has a
`config.json`; `--override` still applies on top.

On a machine without a display, record a video instead of opening the viewer:

```bash
eden inference --task reacher --checkpoint <path/to/model_100.pt> --rec
```

`--rec` adds a recording camera and writes an `.mp4` into the new run directory that `eden inference` creates under
`logs/reacher/`. `--steps N`, `--episodes N` and `--num_envs N` change how long and how wide the rollout is.

## 4. Export it

```bash
eden export --task reacher --checkpoint <path/to/model_100.pt>
```

This writes the policy as ONNX and TorchScript, plus a `policy_spec.json` that describes its observations, actions
and control rate, into `exported/<checkpoint name>/` next to the checkpoint. Each exported graph is checked
against the policy on the environment's observations; the ONNX check needs the `onnx` extra, and `--no_verify`
skips the checks. [Exporting a policy](../guides/policy-export.md) describes the files.

## 5. Check it against MuJoCo, then hardware

`eden deploy` runs the policy through the same loop it uses on a real robot. With `--mujoco`, the robot is a MuJoCo
model instead, which checks the policy against a second physics engine before you touch hardware:

```bash
eden deploy --task <task> --checkpoint model.pt --mujoco --vis     # sim-to-sim
eden deploy --task <task> --checkpoint model.pt                    # real hardware
```

The robot and its hardware backend come from the task's scene. Deployment has many safety-relevant flags (gain
scaling, joint ramps, a simulated gantry); read [Deployment](../guides/deployment.md) before running a policy on a
robot.

## 6. Run a published policy

Eden ships ports of published controllers that run without any training:

```bash
eden inference --policy homie --vis                         # HOMIE walk/stand controller on a Unitree G1
eden inference --policy homie --vis --base_command 0.5 0 0  # a constant velocity command (vx, vy, yaw rate)
```

`--policy` accepts a package name from `eden/extensions/policies/` (for example `holosoma` or `homie`). Without
`--task`, Eden builds a scene from the policy's own robot, observations and actions. The checkpoint downloads on
first use. Some policies need more: motion-tracking policies such as `sonic` need `--motion_file`, and the ONNX
ones need the `onnx` or `onnx-gpu` extra. The [policy zoo](../guides/policy-zoo.md) lists every policy and what it
needs.

## Where to go next

- [Key concepts](concepts.md): what a task config is made of, and how one control step runs.
- [Your first task](first-task.md): write a task of your own and train it with `eden train`.
- [Training runs](../guides/training-runs.md): checkpoints, resuming, loggers, other runners.
- [Multi-GPU training](../guides/ddp.md): `torchrun` and SLURM.
- `eden terms --grep friction`: find a built-in term and its parameters ([command reference](../reference/cli.md)).

## See also

- [Troubleshooting](troubleshooting.md)
- [Assets and the download cache](assets.md)
- [Examples](../examples/index.md)
