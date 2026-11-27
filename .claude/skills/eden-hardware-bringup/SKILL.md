---
name: eden-hardware-bringup
description: Bring up a real robot that Eden deploys to (Unitree G1, Go2, Go1, LimX Oli, Franka Panda / FR3, RoboTera XHand, Wuji Hand) and run a first policy on it. Covers the SDK install, connecting, the state-check recipe, the MuJoCo rehearsal, commissioning gains and the operator sequence. Use when setting up, checking, or first running a physical robot with eden deploy. Don't use for simulation-only work or for robots Eden has no deployment backend for.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Hardware bring-up with `eden deploy`

Follow `.claude/rules/hardware-safety.md` throughout. Ask the user before every command that moves hardware, and
never run one in the background.

## 0. Is the robot supported?

`docs/guides/deployment-hardware.md` lists every backend with its SDK and operator inputs. A robot not in that
table has no `eden deploy` backend: stop and tell the user. Composite rigs (an arm with a hand) deploy each part
through its own backend (`--entity_name`, `--part_option PART.OPTION=VALUE`).

## 1. Install and connect

Read the robot's section of `docs/guides/deployment-hardware.md` and do exactly what it says: the SDK, the
network interface or address, the remote. The SDKs live outside the lockfile, so a plain `uv sync` removes them;
sync with `--inexact`. ONNX policies (the G1's Holosoma safety slot) need the `onnx` or `onnx-gpu` extra, never both.

## 2. Check the state mapping

Run the robot's recipe under `examples/recipes/deployment/` (`real_g1_states.py`, `real_oli_states.py`,
`real_franka_states.py`, `real_xhand.py`, `real_wuji_hand2.py`). It mirrors the real robot in sim and prints every
observed quantity side by side, flagging mismatches: joint order, sign, offset, IMU frame. Resolve every flag
before a policy runs. The G1 and Franka recipes need the `LD_PRELOAD` their headers show.

## 3. Rehearse in MuJoCo

Run the exact deploy command with `--mujoco` (and `--menagerie` where mapped) and pass the `eden-sim2sim` gate.

## 4. First hardware run

```bash
eden deploy --task <task> --checkpoint <ckpt> --deploy_kp_kd_scale 0.8 --debug_record
```

- The robot boots in STOP. INIT (`i`, Enter) ramps to the start pose over `--init_steps`; START (`s`) hands over to
  the policy; STOP (`x`) latches the current pose; KILL (`k`) exits and settles. The robot's remote has the same
  commands (`docs/guides/deployment.md`, "The operator state machine").
- Raise `--deploy_kp_kd_scale` toward 1.0 over successive runs while the motion stays calm.
- A legged robot's first runs belong on a gantry or harness.
- Read `deploy_record.npz` after each run: targets against measured joints, time at limits and torque caps.

## 5. When something is off

- Joint tracking error grows over time: gains, armature or the control rate differ from training. Compare the
  built training robot (`eden audit <Robot>`, `get_dofs_kp/kd/armature`) with the hardware's, and see
  `docs/guides/planning-and-sysid.md`, "System identification".
- Control-loop overrun warnings: `docs/guides/deployment.md`, "Control-loop timing".
- An action term the wire cannot carry stops the deploy at startup; drive it yourself and pass
  `--ignore_action_terms`, or remove it from the task.
