# Task Suites

Eden ships 66 registered tasks: its own locomotion, motion-tracking and manipulation tasks, plus ports of
established benchmarks. This page lists every one of them by family, shows the commands every task shares, and links
to the page that documents each family in depth. Read it to find a task to train, or to see what a task name on the
command line refers to.

## Find a task

`eden list` prints every registered task with its variant modifiers:

```bash
eden list
```

```text
Available Tasks in Eden: Task Name | Source | Modifiers
  ...
  velocity_tracking_flat | eden | VelocityTrackingRobotMod, VelocityTrackingObsMod
  ...
```

The first column is the name you pass to `--task`. The last column names the task's
[TaskMods](../guides/task-mods.md): typed presets that turn into command-line flags (`--robot`, `--difficulty`,
`--song`, ...). A task's `--help` lists its flags with their choices and defaults:

```bash
eden train --task velocity_tracking_flat --help
```

Every task registers itself with `@TASK_REGISTRY.register(...)` in a `config.py` under `eden/tasks/<package>/`;
there is no central index. The task name defaults to the config class name in snake case with the trailing
`_config` dropped (`ManipulationLiftBoxConfig` → `manipulation_lift_box`), and the convention is to name the class
`<Package><Task>Config` so the task reads `<package>_<task>`. A registration may pass an explicit `name=` instead
(`beyondmimic`, `homie`, `omnireset`, ...). See [Task configs](../guides/task-config.md) for the registry and
the package layout.

## The commands every task shares

The same five commands work for any task. Substitute the task name and its flags.

```bash
# Build the task with 2 envs and step it twice: a quick check that it builds on this machine.
eden smoke --task reacher

# Train. num_envs is per GPU rank; each task sets its own default.
eden train --task reacher --difficulty hard --num_envs 2048 --logger wandb

# Resume the most recent run of this task (or a run id, or a run_name substring).
eden train --task reacher --resume latest

# Roll a checkpoint out. --rec records an mp4 into the run's log directory.
eden inference --task reacher --checkpoint logs/reacher/<run>/model_<iter>.pt --rec

# Export the policy for a runtime outside Eden (ONNX / TorchScript + policy_spec.json).
eden export --task reacher --checkpoint logs/reacher/<run>/model_<iter>.pt
```

A training run writes into `logs/<task>/<timestamp>-<run id>/`: the checkpoints `model_<iter>.pt`, the exact
`config.json` the run trained with, and `progress.json`. See [Training runs](../guides/training-runs.md) for the
runners, resuming, loggers and the run directory layout, and [Policy export](../guides/policy-export.md) for
`eden export`.

!!! warning "`eden inference` replays the checkpoint's own config"
    When the checkpoint has a `config.json` beside it, `eden inference` (and `eden deploy`) loads that file, not the
    task's current defaults, so the rollout runs the environment the policy was trained in. Task-mod flags such as
    `--robot` cannot apply to a saved config, so passing one is an error. Change a field with `--override` instead,
    for example `--override 'env_options: {episode_length_s: 30.0}'`. A `--task` that disagrees with the saved config
    only warns, and the run is logged under the saved config's task.

`eden inference` needs a `--checkpoint`. To roll out a published controller instead of your own checkpoint, use
`eden inference --policy <name>`; see the [policy zoo](../guides/policy-zoo.md).

To train on several GPUs, launch the same `eden train` command under `torchrun`; see
[Multi-GPU training](../guides/ddp.md).

## Locomotion

[Locomotion tasks](locomotion.md) — velocity tracking, parkour, recovery and skateboarding.

| Task | Robot | What the policy learns |
|---|---|---|
| `velocity_tracking_flat` | Unitree G1 29-DOF (`--robot oli`: LimX Oli) | Follow a planar velocity command on flat ground |
| `velocity_tracking_terrain` | Unitree G1 29-DOF (`--robot oli`: LimX Oli) | The same on stairs, slopes, bumps and waves, with a height scan |
| `extreme_parkour_teacher` | Unitree Go2 | Cross a grid of parkour courses from privileged scan dots |
| `extreme_parkour_student` | Unitree Go2 | Distil the teacher onto a forward depth camera |
| `mujoco_playground_getup` | Unitree Go1 | Stand up after a fall |
| `mujoco_playground_handstand` | Unitree Go1 | Balance on the front legs with the rear legs raised |
| `homie` | Unitree G1 29-DOF | Track velocity and base height with the legs while the upper body moves |
| `husky` | Unitree G1 (23-DOF skater model) | Ride a simulated skateboard: push, steer and balance |
| `move_forward` | Unitree Go2 (`--robot go1`) | Walk forward; a small benchmark task |

## Motion tracking

[Motion-tracking tasks](motion-tracking.md) — whole-body imitation of reference clips, with and without objects.

| Task | Robot | What the policy learns |
|---|---|---|
| `beyondmimic` | Unitree G1 29-DOF | Track one motion clip (BeyondMimic recipe) |
| `general_mimic` | Unitree G1 29-DOF | Track a library of clips (LAFAN1 or AMASS) with an adaptive per-clip curriculum |
| `weave_hoi` | G1 + two Inspire RH56DFQ hands | Track OMOMO human-object clips with the object simulated |
| `contrack` | xArm7 + XHand | Track a hand-object clip with contact annotations |
| `assistmimic` | Two SMPL humanoids | A caregiver helps a weakened recipient up off the floor |
| `assistmimic_bed` | Two SMPL humanoids | A caregiver helps a recipient out of bed |

## Manipulation and dexterous hands

[Manipulation tasks](manipulation.md) — arm-and-gripper lifting and reaching, in-hand reorientation, piano,
assembly and hand-object tracking.

| Task | Robot | What the policy learns |
|---|---|---|
| `manipulation_reach_droid` | Franka Panda + Robotiq 2F-85 | Move the grasp point to a sampled 3D target |
| `manipulation_lift_box` | Franka Panda + Robotiq 2F-85 (`--manipulator`: UR5e, UR10e) | Reach, grasp and lift a box to a 3D goal |
| `manipulation_lift_box_rgb` | Franka Panda + Robotiq 2F-85 | The same, finding the box in a wrist RGB camera |
| `manipulation_lift_ycb_rgb` | Franka Panda + Robotiq 2F-85 | Lift a different YCB object in each env, from a wrist RGB camera |
| `manipulation_dex_lift` | Franka + XHand (left) | Grasp and lift a cube with a 12-DOF dexterous hand |
| `reorient` | Wuji Hand 1 | Rotate a cube in the hand to a stream of target orientations |
| `piano` | Two Wuji Hand 2 | Play a song on an 88-key keyboard |
| `omnireset` | UR5e + Robotiq 2F-85 | Assemble an object pair, starting from recorded reset states |
| `flashdexretarget` | XHand or Sharpa Wave hands | Make retargeted human hand-object demos physically valid |
| `reacher` | I2RT YAM arm | Reach a sampled 3D target; the reference task-config template |

## Ported benchmarks

These reimplement existing benchmark suites in Eden, keeping their scenes, success checks and robot setups. Each has
its own page.

| Family | Tasks | Page |
|---|---|---|
| robosuite | `robosuite_door`, `robosuite_lift`, `robosuite_nut_assembly`, `robosuite_pick_place`, `robosuite_stack`, `robosuite_tool_hang`, `robosuite_wipe`, `robosuite_two_arm_handover`, `robosuite_two_arm_lift`, `robosuite_two_arm_peg_in_hole`, `robosuite_two_arm_transport` | [robosuite](robosuite.md) |
| LIBERO | `libero` (one task; `--task_name` picks the LIBERO task) | [LIBERO](libero.md) |
| SIMPLER (Bridge) | `simpler_put_carrot_on_plate`, `simpler_put_eggplant_in_basket`, `simpler_put_spoon_on_table_cloth`, `simpler_stack_green_cube_on_yellow_cube` | [SIMPLER](simpler.md) |
| PolaRiS | `polaris_block_stack_kitchen`, `polaris_food_bussing`, `polaris_move_latte_cup`, `polaris_organize_tools`, `polaris_pan_clean`, `polaris_tape_into_container`, and a `_batched` variant of each | [PolaRiS](polaris.md) |
| Mochi (deformables) | `mochi_fold_the_towel`, `mochi_route_the_cable`, `mochi_wipe_the_bar` | [Mochi tasks](mochi.md) |
| Whiteboard | `whiteboard_draw_table`, `whiteboard_write_in_colour`, `whiteboard_write_text` | [Whiteboard](whiteboard.md) |
| Art | `art_calligraphy_dai`, `art_calligraphy_san`, `art_calligraphy_ei`, `art_calligraphy_lettering_minimum`, `art_painting_fuji`, `art_painting_fuji_study`, `art_painting_sunset` | [Calligraphy and painting](art.md) |

Eden also ships a CALVIN scene and its objects (`eden.options.scenes.calvin`), which have no registered task; see
[Scenes and terrains](../guides/scenes-and-terrains.md).

## See also

- [Your first task](../getting-started/first-task.md) — write a task end to end, starting from `reacher`.
- [TaskMods](../guides/task-mods.md) — how variant flags are declared and composed.
- [Training runs](../guides/training-runs.md) — `eden train` and `eden inference` in detail.
- [Deployment](../guides/deployment.md) — `eden deploy`, on hardware or in MuJoCo.
