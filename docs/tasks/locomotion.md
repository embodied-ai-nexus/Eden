# Locomotion Tasks

Eden's locomotion tasks train legged robots to walk, recover, balance and ride. This page gives, for each task, the
robot, what the policy learns, its variant flags, what it needs installed, and the commands to train and replay it.
The commands shared by every task (`eden smoke`, `--resume`, `eden export`) are on the [task index](index.md).

| Task | Robot | Flags | Default `num_envs` |
|---|---|---|---|
| [`velocity_tracking_flat`](#velocity-tracking) | Unitree G1 29-DOF or LimX Oli | `--robot`, `--obs_preset` | 4096 |
| [`velocity_tracking_terrain`](#velocity-tracking) | Unitree G1 29-DOF or LimX Oli | `--robot`, `--obs_preset` | 4096 |
| [`extreme_parkour_teacher`](#extreme-parkour) | Unitree Go2 | — | 4096 |
| [`extreme_parkour_student`](#extreme-parkour) | Unitree Go2 | — | 192 |
| [`mujoco_playground_getup`](#mujoco-playground-getup-and-handstand) | Unitree Go1 | — | 4096 |
| [`mujoco_playground_handstand`](#mujoco-playground-getup-and-handstand) | Unitree Go1 | — | 4096 |
| [`homie`](#homie) | Unitree G1 29-DOF | — | 4096 |
| [`husky`](#husky-skateboarding) | Unitree G1, 23-DOF skater model | — | 4096 |
| [`move_forward`](#move-forward) | Unitree Go2 or Go1 | `--robot`, `--controller`, `--runner` | 4096 |

`num_envs` is per GPU rank. Override it with `--num_envs N`.

## Velocity tracking

A bipedal humanoid follows a random planar velocity command (forward, lateral, yaw rate) while staying upright and
moving smoothly. `velocity_tracking_flat` runs on a ground plane; `velocity_tracking_terrain` runs on a procedurally
generated grid of stairs, inverted stairs, slopes, random bumps and waves, and adds a 17 × 11 height scan to both
observation groups and an out-of-terrain-bounds termination. Both share one config base,
`eden/tasks/velocity_tracking/base_config.py`.

```bash
eden train --task velocity_tracking_flat                          # Unitree G1 29-DOF
eden train --task velocity_tracking_terrain --robot oli           # LimX Oli 31-DOF
eden train --task velocity_tracking_flat --obs_preset privileged  # the policy also reads base_lin_vel
eden inference --task velocity_tracking_flat \
    --checkpoint logs/velocity_tracking_flat/<run>/model_<iter>.pt --rec
```

| Flag | Choices (default first) | Effect |
|---|---|---|
| `--robot` | `g1`, `oli` | Which humanoid. The mod binds the robot's joints, action scale, torso and feet links, and prefixes the experiment name (`g1_velocity_tracking_flat`). |
| `--obs_preset` | `proprioceptive`, `privileged` | `proprioceptive`: sensor-only policy input, 5 frames of history. `privileged`: the policy also reads the simulator's `base_lin_vel`, single frame, unscaled joint velocities. |

The command range widens over training (`VelocityCommandCurriculum`): forward speed starts at ±1 m/s and reaches
−2 to 3 m/s. Training uses rsl_rl PPO, 30 000 iterations at 50 Hz control.

!!! warning "`privileged` needs a velocity estimator on the robot"
    A humanoid has no sensor for base linear velocity. A policy trained with `--obs_preset privileged` must run with
    `eden deploy --estimate_lin_vel` (a contact-aided Kalman filter) on hardware. The default `proprioceptive`
    policy needs no estimator. See [Deployment](../guides/deployment.md).

To drive a trained policy with the keyboard in the viewer, use the example script:

```bash
uv run python examples/projects/locomotion/velocity_tracking/main.py play --model <path/to/model_<iter>.pt>
```

The reasons behind the robot binding, the observation split, the posture tolerances and the smoothness penalties are
on [Velocity tracking: design notes](../guides/velocity-tracking.md).

## Extreme Parkour

A port of *Extreme Parkour with Legged Robots* (Cheng et al. 2023) on the Unitree Go2, in two phases.

- `extreme_parkour_teacher` — the Go2 follows waypoint goals across a grid of parkour courses (gaps, hurdles, flat
  lanes, steps, stepping stones), with a walked-distance terrain-level curriculum. The policy reads one 753-wide
  `policy` group: proprioception, terrain scan dots, privileged state and a 10-frame history. It trains with the
  RMA stack in `eden.learning.wrappers.rsl_rl.rma` (scan, privileged and history encoders, a base-velocity
  estimator, periodic DAgger updates of the history encoder, a left-right mirror loss). The config selects that
  stack itself; stock PPO with a plain MLP does not learn this task.
- `extreme_parkour_student` — the same environment plus a forward depth camera (106 × 60, mounted on the `base`
  link and pitched 20° down), a one-control-step action delay and camera-mount pitch randomization. The policy is
  distilled from a teacher checkpoint (DAgger), so `--checkpoint` names the **teacher**.

```bash
eden train --task extreme_parkour_teacher --logger wandb
eden train --task extreme_parkour_student \
    --checkpoint logs/extreme_parkour_teacher/<run>/model_<iter>.pt
eden inference --task extreme_parkour_student \
    --checkpoint logs/extreme_parkour_student/<run>/model_<iter>.pt
```

Both tasks take one flag:

| Flag | Choices (default first) | Effect |
|---|---|---|
| `--terrain_sampler` | `level`, `uniform`, `sgs` | Which terrain cell each episode starts on. `level`: the walked-distance curriculum moves the env up or down a difficulty row and keeps its course column. `uniform`: every cell of the grid, uniformly. `sgs`: Success Guided Sampling over every cell, graded by `all_goals_reached` (see [Curricula](../guides/curricula.md#reset-configuration-samplers)). |

All three modes log `Commands/parkour_command/terrain_level` (the row each finished episode ran on) and
`Metrics/parkour_goal_progress`, whose all-goals episodes make up `Metrics/Success`; compare the modes on those.
The `feet_edge` penalty applies only above its `min_terrain_level`, and `uniform` and `sgs` place envs on those rows
from the first episode, so the reward differs between the modes as well as the sampling.

The student renders depth with the gs-madrona batch renderer, which is not an Eden extra: install `gs-madrona`
(Linux x86-64 only) before training it. The teacher needs nothing beyond the core install.

`examples/projects/locomotion/extreme_parkour/` has a viewer playback script and a full description of the
observation slot table and the network architecture.

## MuJoCo Playground: getup and handstand

Two Go1 tasks reimplemented from MuJoCo Playground's `locomotion/go1` (no MuJoCo involved):

- `mujoco_playground_getup` — fall recovery. At reset the Go1 is dropped from 0.5 m with a random orientation and
  random joint angles (with probability 0.6; otherwise it starts standing), and it must stand up and hold its
  default pose at a torso height of 0.275 m. 42-wide observation; the action is 12 joint-position targets offset from the default
  pose.
- `mujoco_playground_handstand` — the body's x-axis points down, the torso is held at 0.55 m, and the rear feet are
  penalized for touching the floor. 45-wide observation; the action is a joint-position delta accumulated onto the
  previous target (`ReferenceSource.ACCUMULATE`, scale 0.3, clamped to the soft joint limits).

Both run at 50 Hz control (`sim_dt` 0.004, `decimation` 5) and train with rsl_rl PPO for 3000 iterations.

```bash
eden train --task mujoco_playground_getup
eden inference --task mujoco_playground_getup \
    --checkpoint logs/mujoco_playground_getup/<run>/model_<iter>.pt --rec
```

The handstand config also declares the upstream action-rate, torque, joint-acceleration, stay-still, energy and
termination rewards at weight 0, for fine-tuning. `examples/projects/locomotion/mujoco_playground/` has
train-then-play scripts for both tasks.

## HOMIE

A port of OpenHomie's G1 lower-body controller. The policy tracks a 4-D command
`[lin_vel_x, lin_vel_y, ang_vel_z, base_height]` with the 12 leg joints, while the 3 waist and 14 arm joints are
driven to random poses by an upper-body pose curriculum, so the legs stay robust to whatever the upper body does.
It trains with OpenHomie's HIMLoco learner (`eden.learning.wrappers.rsl_rl.him`), which the config selects.

```bash
eden train --task homie
eden inference --task homie --checkpoint logs/homie/<run>/model_<iter>.pt
```

The differences from OpenHomie are listed in the module docstring of `eden/tasks/homie/config.py`. The main ones:
the full 29-DOF G1 (OpenHomie welds the waist to one DOF), actions delayed by less than one control step, an
upper-body curriculum on a fixed step schedule, and friction ranges remapped because Genesis combines contact
friction as `max`. The frozen OpenHomie release checkpoints are a separate, inference-only port:
`eden inference --policy homie` (see the [policy zoo](../guides/policy-zoo.md)).

!!! note "Licence"
    OpenHomie is released under CC BY-NC-SA 4.0 (non-commercial), and this task inherits that licence.

## HUSKY skateboarding

A training port of HUSKY (humanoid skateboarding). A G1 rides a free-floating skateboard whose trucks steer when
the deck tilts. A 6 s phase clock cycles push (the left leg strokes the ground), transition, steer (both feet on the
deck) and transition; the command is `[board speed, relative heading]`. The robot is `HuskyG1`, the 23-DOF skater
model (wrists welded) used by the frozen HUSKY policy.

The config trains PPO with an adversarial motion prior (`RslRlAmpPpoAlgorithmOptions`) over retargeted human pushing
clips, which is the full HUSKY recipe. The clips are an asset file fetched on first use.

```bash
eden train --task husky --logger wandb
eden inference --task husky --checkpoint logs/husky/<run>/model_<iter>.pt --rec
```

To train the same MDP without the style reward, replace `runner_options.algorithm` with a plain
`RslRlPpoAlgorithmOptions(...)` (setting `class_name` alone is not enough). The `SkatePushFlag` observation warns at
the start of such a run, because the result is not comparable to a HUSKY result. The frozen published policy runs
with `eden inference --policy husky`; `examples/projects/locomotion/husky_skate/main.py` adds a play script with
rough terrain.

!!! note "Licence"
    HUSKY is released under CC BY-NC 4.0 (non-commercial).

## Move forward

A small Go2 task that rewards walking forward, used as a benchmark and as the worked example of composing several
TaskMods.

```bash
eden train --task move_forward
eden train --task move_forward --robot go1 --controller explicit --runner rl_games
```

| Flag | Choices (default first) | Effect |
|---|---|---|
| `--robot` | `go2`, `go1` | Which quadruped. |
| `--controller` | `implicit`, `explicit` | Genesis's built-in PD, or user-space torques (halves `sim_dt` and sets `decimation=2`). |
| `--runner` | `rsl_rl`, `rl_games` | The RL library. `rl_games` replaces the runner options with the rl_games schema and needs the `rl-games` extra. |

=== "pip"

    ```bash
    pip install "eden-world[rl-games]"
    ```

=== "uv (checkout)"

    ```bash
    uv sync --inexact --extra rl-games
    ```

## See also

- [Velocity tracking: design notes](../guides/velocity-tracking.md)
- [Terrain adaptation of motion clips (TCRS)](../guides/tcrs-terrain-adaptation.md)
- [Scenes and terrains](../guides/scenes-and-terrains.md) — the terrain generators these tasks use.
- [Training runs](../guides/training-runs.md) and [Multi-GPU training](../guides/ddp.md)
- [Deployment](../guides/deployment.md) — run a locomotion checkpoint on hardware or in MuJoCo.
