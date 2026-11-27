---
name: eden-rl-task
description: Author, smoke-test, train, diagnose and hand off an RL task in Eden (locomotion, manipulation, dexterous, vision). Covers surveying a reference task, checking rewards before training, launching one training run and polling progress.json, reading Episode_Reward / Episode_Termination curves, and replaying and exporting the checkpoint. Use when creating or tuning an Eden RL task or diagnosing a training run. Don't use for porting an mjlab task (use mjlab-to-eden), for writing a single term (use writing-eden-terms), or for non-RL controllers.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
---

<!--
The workflow and the reward-shaping lessons are adapted from google-deepmind/robotics-antigravity-plugin,
plugins/robotics/skills/mujoco-playground-rl/SKILL.md (Copyright 2026 Google LLC, Apache License 2.0). Modified
for Eden: every API, command and log key is Eden's, and per-domain weights were replaced by pointers to Eden's
reference tasks.
-->

# Authoring and training an RL task in Eden

The guides are the contract. Read the ones a step names before doing the step; this skill only orders them.

## 1. Survey once, then write

- `eden list --tag <locomotion|manipulation|dexterous|...>` and pick the closest task. Read its `config.py` whole
  before writing anything. Good starting points:

  | Domain | Reference task |
  |---|---|
  | Velocity-tracking locomotion | `velocity_tracking_flat`, then `velocity_tracking_terrain` |
  | Reaching | `reacher` (`eden/tasks/benchmark/reacher/config.py`, the canonical config layout) |
  | Lifting with a parallel gripper | `manipulation_lift_box`; from pixels, `manipulation_lift_box_rgb` |
  | Dexterous hands | `reorient`, `manipulation_dex_lift` |

- Find existing terms before writing one: `eden terms --grep '<what you need>'` searches names, parameters and
  summaries. A new term follows the `writing-eden-terms` skill.
- Read `docs/guides/task-config.md` (layout, registration, tags) and, for each manager you touch,
  `docs/guides/{observations,rewards-and-terminations,commands,events,curricula,actions}.md`.
- A robot without a preset: `docs/guides/robots.md`, "Adding a robot", then `eden audit <Robot>` (see the
  `eden-asset-audit` skill) before any training.
- Register with `@TASK_REGISTRY.register(tags=...)` and add a row to `docs/tasks/index.md`;
  `tests/tasks/test_init.py` checks both.

## 2. Check it before training

1. `eden smoke --task <name>` builds two envs and steps them. Fix every error it raises; they name the bad value.
2. Check that the rewards move. Write a short script that builds the task with a few envs, steps random actions for
   a second of sim time, and prints `env.reward_manager.get_term_value(name)` for each `name, _ in
   env.reward_manager.iter_terms()` (`docs/guides/rewards-and-terminations.md`, "Reading rewards"). A progress or
   stage reward that reads exactly zero everywhere is miswired: a wrong height threshold, a gate that can never
   open, an entity or link name that resolved to something else.
3. `eden smoke --task <name> --profile` writes `term_cost.json` when steps are slow. Hot paths stay sync-free
   (`docs/guides/terms.md`, "Stay sync-free on the hot path").

## 3. Train once and poll

```bash
eden train --task <name> --num_envs 4096 --run_name <what-changed> \
    --override 'runner_options: {max_iterations: 400}'
```

- Run it in the background and poll `logs/<name>/<run>/progress.json` (`status`, `iteration`, `mean_reward`,
  `updated_at`) instead of tailing the log (`docs/guides/training-runs.md`, "progress.json"). A `"running"` status
  with a stale `updated_at` is a dead run.
- Launch one run per hypothesis. Do not start a second copy of the same config while the first is still learning.
- On a cluster, use the SLURM template in `AGENTS.md`; `num_envs` there is per rank.

## 4. Read the curves

TensorBoard (`tensorboard --logdir logs/<name>`) carries `Episode_Reward/<term>` (weighted, per episode-second),
`Episode_Termination/<term>` (how many resets each term caused) and the runner's loss and episode-length scalars.

| What you see | Usual cause |
|---|---|
| Episodes collapse to one step in the first iterations | Expected with per-step penalties and cheap terminations; PPO recovers. Wait a few hundred iterations (`docs/guides/training-runs.md`, "Training dynamics that look like bugs"). |
| One stage reward climbs, the next stays flat (reach rises, lift never does) | The next stage's threshold or gate is miscalibrated, often a height measured from the floor instead of the table top. |
| The robot learns to stand still | A constant survival reward outweighs the tracking terms, or a gait term is negative for small steps. Keep survival small and gait rewards non-negative. |
| One termination dominates `Episode_Termination` | That condition is too strict, or the reset puts the robot in it. |
| `Episode_Termination/solver_error` above zero | The physics diverged in those envs; check the robot with `eden audit` and the contacts. |

Reward-shaping lessons that carried over from other stacks: make the task reward dominate the regularizers;
gate a stage reward on the previous stage (grasp reward only near the object); reward progress toward a goal
rather than only arrival; for a sparse success bonus, remember the step reward is scaled by `dt`.

## 5. Replay, export, hand off

- `eden inference --task <name> --checkpoint logs/<name>/<run>/model_<it>.pt --rec` records a video into the run
  directory; `--live_rewards` opens per-term reward graphs in the browser with editable weights.
- `eden export --task <name> --checkpoint ...` writes ONNX / TorchScript plus `policy_spec.json`
  (`docs/guides/policy-export.md`).
- Before any hardware, follow the `eden-sim2sim` skill.
