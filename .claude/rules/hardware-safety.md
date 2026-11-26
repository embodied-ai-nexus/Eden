---
paths:
  - "eden/extensions/deployment/**"
  - "examples/recipes/deployment/**"
  - "examples/policies/**"
  - "docs/guides/deployment*.md"
---

<!--
Adapted from google-deepmind/robotics-antigravity-plugin, plugins/robotics/rules/physical_robot_safety.md
(Copyright 2026 Google LLC, Apache License 2.0). Modified for Eden: the eden deploy safety mechanisms, the
MuJoCo sim-to-sim gate and the state-check recipes were added; the generic rules were kept.
-->

# Physical robot safety

These rules apply whenever you run, write or change code that commands a real robot, gripper, hand or motor bus.
They do not apply to `eden deploy --mujoco` or to simulation-only work.

## Before anything moves

- **Ask first.** Before you start a command that moves hardware, tell the user what will move and how, and wait
  for them to confirm that someone is next to the robot, the workspace is clear and the e-stop is in reach. Never
  start hardware motion from a background or unattended process.
- **Rehearse in MuJoCo.** Run the same command with `--mujoco` (and `--menagerie` when the robot has a mapping)
  before the hardware run. See the `eden-sim2sim` skill.
- **Check the wiring.** On a robot new to this machine, run its state-check recipe under
  `examples/recipes/deployment/` (`real_g1_states.py`, `real_franka_states.py`, ...) and read the mismatches
  before running a policy.

## Prefer `eden deploy` over a hand-written loop

`eden deploy` already boots in STOP, moves only on the operator's INIT and START, ramps to the start pose, clamps
per-robot limits, and settles the robot in `close()` on KILL, Ctrl-C or an error (`docs/guides/deployment.md`,
"Safety"). Commission a new policy with `--deploy_kp_kd_scale 0.2` and raise it toward 1.0 over runs.

On the G1, Go2 and Oli a stalled control thread is reported but not acted on: the re-send loop keeps sending the
last command and the robot holds it. Stop the process or hit the e-stop.

## Hand-written scripts that command hardware

- **Speed.** Use the robot's documented default and ceiling. For an unknown or uncalibrated robot, default to
  0.5 rad/s per joint and never exceed 1.0 rad/s unless the user asks. Interpolate every move with zero velocity
  at both ends (cosine or minimum-jerk); never send one large jump.
- **Park before letting go.** Never disable torque, exit, or call the SDK's close while the arm is raised. Return
  to a rest pose in `finally:` and in `SIGINT`/`SIGTERM` handlers. Read the *measured* joint positions there and
  interpolate from them at 0.5 rad/s or slower, so a step or velocity guard cannot reject the move. Holding an
  object, or inside a bin, retreat vertically first. Never leave a raised arm in a zero-gravity or
  gravity-compensation-only mode.
- **Keep the heartbeat.** Short computation (OpenCV, NumPy, IK under about 50 ms) in the control process is fine.
  Multi-second stalls are not: video encoding, network or model calls, `input()`. Run those in another process or
  after parking. Keep one long-lived process per bus, never two on the same CAN, serial or DDS interface, and never
  exit expecting another script to take over a raised arm.
- **Heat.** Holding a grasp for seconds of transit is expected. Clamping a gripper or motor against a hard stop
  for tens of seconds can trip a thermal fault that drops every joint on the bus; warn the user and confirm
  before doing it on request.
- **Bench debugging.** `--disable_joints` sends kp = kd = 0. Never use it on a load-bearing joint.
