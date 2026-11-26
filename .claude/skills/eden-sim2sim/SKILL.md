---
name: eden-sim2sim
description: Gate a Genesis-trained Eden policy through MuJoCo sim-to-sim before hardware. Covers the order of checks (eden inference, inference --sim2sim, eden deploy --mujoco in lockstep then --async, --menagerie), robustness sweeps with Eden's events, comparing two checkpoints fairly, and the review to pass before a hardware run. Use when validating a checkpoint for deployment, triaging a policy that falls over in MuJoCo, or choosing between two candidates. Don't use for training (use eden-rl-task) or for robot SDK setup (use eden-hardware-bringup).
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
---

<!--
The candidate-comparison discipline and the pre-hardware review are adapted from
google-deepmind/robotics-antigravity-plugin, plugins/robotics/skills/sim2real-iteration/SKILL.md and
references/critic_rubric.md (Copyright 2026 Google LLC, Apache License 2.0). Modified for Eden: the gauntlet
library is replaced by Eden's deploy loop, MuJoCo backend and event terms.
-->

# Sim-to-sim validation before hardware

MuJoCo is the independent physics engine here: Genesis trains, MuJoCo checks. Read `docs/guides/deployment.md`,
"Sim-to-sim validation with MuJoCo", before the first run; it lists what the backend rebuilds and why.

## The gate, in order

Each step must pass before the next. Use the checkpoint's run directory, so every command rebuilds the env from
its `config.json`.

1. **Genesis, evaluation mode.** `eden inference --task <task> --checkpoint <ckpt> --episodes 20` and
   `--rec` for a video. Read success from the task's metric or termination terms, not from the video alone.
2. **Genesis, deployment timing.** Add `--sim2sim`: the policy runs on its own thread against wall-clock physics,
   and the last action repeats when it is late. Read the timing report it prints.
3. **MuJoCo, lockstep.** `eden deploy --task <task> --checkpoint <ckpt> --mujoco --base_policy none`, with
   `--menagerie` when the robot has a Menagerie mapping (G1 29-DOF, Go2, Go1, Oli today). Lockstep is
   deterministic, so it is the mode for comparing runs. A topple on the training asset alone is not yet a policy
   bug; re-run with `--menagerie` first.
4. **MuJoCo, real time.** Add `--async`: physics runs on its own thread at wall-clock rate, as on the robot. A
   `fell behind wall clock` warning means the run was not real time and its timing conclusions do not hold.
5. **MuJoCo, sensor error.** `--imu_noise ACC GYRO` and `--imu_bias ...` for a legged robot, at the levels the
   robot's IMU datasheet gives.

Every deploy run boots in STOP; headless MuJoCo presses INIT and START itself. `--debug_record` writes commanded
targets and measured state to an `.npz` for comparison against a Genesis rollout.

## When MuJoCo and Genesis disagree

- Audit the built robot before the policy: `eden audit <Robot>` and `docs/guides/policy-porting.md` ("Bit-exact
  obs/action parity does not mean a ported policy is correct"). Armature, gains, friction and control rate are
  the usual differences.
- Compare the deploy config against the training config: `deploy_config.json` in the deploy run directory.
  Training randomization is stripped at deploy.
- If the dynamics really differ from hardware, identify them: `docs/guides/planning-and-sysid.md`, "System
  identification".

## Robustness sweeps in Genesis

Widen the randomization the policy will meet, in eval mode, with Eden's event terms
(`docs/guides/events.md`, "Physical properties", "Actuator and controller randomization", "Pushes and external
wrenches") passed through `--override` or a small config file. Sweep one factor at a time (mass, friction, gains,
latency, pushes) so a failure names its cause.

## Comparing two candidates

- Same task config, same seeds, same number of episodes, same event settings: only the checkpoint differs. Pair
  each episode of candidate B with the episode of candidate A on the same seed.
- One seed proves nothing. Count the pairs where exactly one candidate succeeded; with fewer than about six such
  pairs, the candidates are indistinguishable and the simpler one stays.
- A safety regression (falls, joint-limit hits, self-collision, torque saturation) vetoes a task-score gain.

## Review before a hardware run

Pass every item, then hand over to the `eden-hardware-bringup` skill and the hardware safety rule
(`.claude/rules/hardware-safety.md`).

- The policy reads only what the robot provides: no privileged observation survived into the actor
  (`policy_spec.json` lists the inputs).
- It succeeds across the randomized sweeps above, not only at nominal.
- MuJoCo lockstep and `--async` both pass, on the Menagerie model when one exists.
- No joint sits at its limit or its torque cap for long stretches in `--debug_record`.
- After the episode ends, the robot comes to rest: no oscillation at the end pose.
