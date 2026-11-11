# Command reference

This page lists every `eden` command and subcommand with its flags, as `--help` prints them. Use it to look up a
flag; the linked guides explain the workflows behind each command.

Every command prints its own help with `--help`. For the commands that take a task, add the task so that its
variant flags are listed too:

```bash
eden --help
eden train --task reacher --help        # the shared flags plus reacher's --difficulty
```

Without an activated virtual environment, run `uv run eden ...` in a clone, or `python -m eden ...` anywhere
Eden is installed.

## Overview

| Command | What it does |
|---|---|
| [`eden list`](#eden-list) | List every registered task, its source and its task mods |
| [`eden terms`](#eden-terms) | List registered terms and the parameters they accept |
| [`eden train`](#eden-train) | Train a registered task |
| [`eden inference`](#eden-inference) | Roll out a trained checkpoint, or a published policy, in simulation |
| [`eden export`](#eden-export) | Export a checkpoint's policy as ONNX / TorchScript plus `policy_spec.json` |
| [`eden deploy`](#eden-deploy) | Run a policy on hardware, or against a MuJoCo model (`--mujoco`) |
| [`eden smoke`](#eden-smoke) | Build a task and step it a few times, as a sanity check |
| [`eden preprocess`](#eden-preprocess) | Build a motion task's dataset cache ahead of training |
| [`eden datagen`](#eden-datagen) | Record scripted demonstrations of a task into HDF5 |
| [`eden examples list`](#eden-examples-list) / [`smoke`](#eden-examples-smoke) | Browse and smoke-run the runnable examples |
| [`eden lerobot convert`](#eden-lerobot-convert) | Convert an Eden HDF5 recording into a LeRobot v3 dataset |
| [`eden pose gvhmr`](#eden-pose-gvhmr) | Estimate SMPL-X motion from monocular video |
| [`eden retarget ...`](#eden-retarget) | Retarget human motion to robots (OmniRetarget, OTRetarget, SOMA, DexRetargeter, FlashDexRetarget) |
| [`eden gello ...`](#eden-gello-calibrate) | Calibrate and stream a GELLO leader arm |
| [`eden metaquest bridge`](#eden-metaquest-bridge) | Meta Quest WebXR teleoperation bridge |
| [`eden pico bridge`](#eden-pico-bridge) | PICO (XRoboToolkit) teleoperation bridge |
| [`eden teleop sonic ...`](#eden-teleop-sonic-robot) | SONIC whole-body teleoperation (robot process and operator station) |
| [`eden wuji ...`](#eden-wuji-scan) | Find and monitor Wuji Gloves |
| [`eden view`](#eden-view) / [`arrange`](#eden-arrange) / [`snap`](#eden-snap) | Look at an entity, arrange a scene, take a snapshot |
| [`eden gif`](#eden-gif) | Convert an `.mp4` to a `.gif` |
| [`eden clean`](#eden-clean) | Remove empty log directories |

## Global options

These go **before** the subcommand, because the task commands hand everything after their own flags to the task's
parser:

| Flag | Meaning |
|---|---|
| `--log-theme {auto,human,agent}` | How the command renders console output. `agent` (the default): no ANSI colors, no greeting box. `human`: color. `auto`: color on a terminal, honoring `NO_COLOR` and `FORCE_COLOR`. It sets `EDEN_LOG_THEME` for this process and everything it launches. |

```bash
eden --log-theme human train --task reacher
```

Putting `--log-theme` after the subcommand is an error that says where it goes. To pick a theme for a whole shell,
set `EDEN_LOG_THEME` instead ([Environment variables](environment-variables.md)).

## Flags shared by the task commands

`train`, `inference`, `export`, `deploy`, `smoke`, `preprocess` and `datagen` share one argument parser. Every one of
them takes:

| Flag | Meaning |
|---|---|
| `--task TASK` | The registered task name (`eden list`). Adds the task's variant flags. |
| `--config CONFIG` | A `.json`, `.yaml` or `.yml` file. A config Eden saved (it carries its class metadata) replaces the task's config; any other file is merged into it as a partial override. |
| `--override YAML` | An inline YAML mapping merged into the config, e.g. `--override 'env_options: {num_envs: 64}'`. Repeatable; later ones win; applied after `--config`. |
| `--checkpoint CHECKPOINT` | Path to a checkpoint file. |
| `--cpu` | Use the CPU backend instead of the GPU. |
| `--run_name RUN_NAME` | A name folded into the run directory, `<timestamp>-<run_name>-<id>`, and so into the W&B run name. |
| `--<variant>` | One flag per argument of each of the task's mods, e.g. `--difficulty {easy,medium,hard}` for `reacher`. |

When `--checkpoint` points into a run directory that holds a `config.json`, `inference`, `export` and `deploy`
load that config instead of rebuilding the task from its current defaults. `--config` (as a partial) and
`--override` still apply on top, but a variant flag set away from its default is refused, because the saved
config already decides it.

## Tasks

### `eden list`

```bash
eden list
```

Prints every registered task with its source (`eden` for a built-in task, otherwise the distribution or module it
comes from) and the task mods that give it variant flags. Takes no flags, and imports no task module, so it is fast
and needs no network.

### `eden terms`

```bash
eden terms                         # every registered term
eden terms --kind reward           # one kind
eden terms --grep 'height|altitude'
eden terms --json                  # for tooling
```

| Flag | Meaning |
|---|---|
| `-k`, `--kind KIND` | Only this kind of term, e.g. `reward`, `observation`, `event`. |
| `-g`, `--grep PATTERN` | Only terms whose name, parameters or summary match `PATTERN`, a case-insensitive regular expression. Exits 1 with a message when nothing matches. |
| `--json` | Emit the catalog as JSON. |
| `--markdown` | Emit the catalog as the full docs page, with parameters ([Term catalog](../api/terms/catalog.md) is generated this way). |

`--json` and `--markdown` are mutually exclusive.

### `eden train`

```bash
eden train --task <task> [variant flags] [flags]
```

Trains a registered task with the runner its `runner_options` select: rsl_rl by default, rl_games when the
options are an rl_games runner's. Each run writes to `logs/<task>/<timestamp>[-<run_name>]-<id>/` under the current
directory: checkpoints, `config.json`, `run.json`, and the logger's files. A task whose `runner_options` is `None`
is not trained through `eden train`. See [Training runs](../guides/training-runs.md).

Besides the [shared flags](#flags-shared-by-the-task-commands):

| Flag | Meaning |
|---|---|
| `--num_envs N` | Override the task's number of parallel environments (per rank in a multi-GPU run). |
| `--logger {tensorboard,neptune,wandb}` | rsl_rl's logger. Default `tensorboard`. Ignored for rl_games. |
| `--resume REF` | Continue a previous run of this task, found under `logs/<task>/`: `latest` (the most recently modified), a run id, or a substring of a run name. Loads that run's latest checkpoint into a new run directory, which records its parent. Cannot be combined with `--checkpoint`. |
| `--wandb_group GROUP` | The W&B group for this run. Default: the task name. Ignored unless `--logger wandb`. |

`--checkpoint` starts training from that checkpoint's weights.

### `eden inference`

```bash
eden inference --task <task> --checkpoint <path/to/model.pt> [flags]
eden inference --policy <name> [--task <task>] [flags]
```

Rolls a policy out in simulation, in evaluation mode, under `logs/<task>/` (or `logs/inference/` without a task).
It needs either `--checkpoint` (with `--task`) or `--policy`, which runs a frozen published policy instead; the two
cannot be combined. See [Training runs](../guides/training-runs.md) and the [policy zoo](../guides/policy-zoo.md).

| Flag | Meaning |
|---|---|
| `--vis`, `--show_viewer` | Open the Genesis viewer. |
| `--rec` | Record an `.mp4` through a `rec` camera, saved in the run's log directory. |
| `--num_envs N` | Parallel evaluation environments. Default 1. |
| `--steps N` | Stop after `N` steps. Default: one episode for a checkpoint, 1000 steps for `--policy`. |
| `--episodes N` | Run until each environment has finished `ceil(N / num_envs)` episodes. Not with `--policy`. |
| `--async`, `--sim2sim` | Run deployment-style: the policy runs in a background thread, the simulation is paced to wall-clock time, and the last action repeats when the policy is late. A simulation-side rehearsal of `eden deploy`. |
| `--policy NAME`, `--extension_policy NAME` | A published policy: a package name under `eden/extensions/policies/` (e.g. `holosoma`, `homie`), its `POLICY_REGISTRY` key, or the key without `_policy`. Makes `--task` and `--checkpoint` optional. |
| `--robot ROBOT` | The robot options class from `eden.robots` for a `--policy` run without a task. Default `G1Dofs29`. Ignored with `--task`. |
| `--entity_name NAME` | The scene entity the policy controls. Default `robot`. |
| `--base_command VX VY WZ` | A constant velocity command for policies driven by one. Default: stand still. |
| `--motion_file FILE` | The reference motion `.npz` for motion-tracking policies (e.g. `sonic`); required by those. |
| `--clip_name CLIP` | The clip to track, for policies with per-clip checkpoints (e.g. `beyondmimic_moves`). |

### `eden export`

```bash
eden export --task <task> --checkpoint <path/to/model.pt> [flags]
```

Exports a trained checkpoint's policy for a runtime outside Eden, with a `policy_spec.json` describing its
observations, actions and control rate. The environment is built from the checkpoint's `config.json`, with one
environment. See [Exporting a policy](../guides/policy-export.md).

| Flag | Meaning |
|---|---|
| `--format FORMAT` | Comma-separated formats from `onnx` and `jit`. Default: all of them. |
| `--out DIR` | The output directory. Default: `exported/<checkpoint name>/` beside the checkpoint. |
| `--no_verify` | Skip checking each exported graph against the policy on the environment's observations. The ONNX check needs the `onnx` extra. |

### `eden deploy`

```bash
eden deploy --task <task> --checkpoint model.pt --mujoco [--vis]     # sim-to-sim
eden deploy --task <task> --checkpoint model.pt                      # hardware
eden deploy --policy <name> [--mujoco]                               # a published policy, no task
```

Runs a policy through the deployment loop: on the real robot that the task's scene names, or, with `--mujoco`,
against a MuJoCo model of it. Training-time randomization is stripped, and the deploy config is saved as
`deploy_config.json` in the run directory. Read [Deployment](../guides/deployment.md) before using it on hardware:
it explains every safety flag below.

**Policies and slots.** Several policies can be loaded at once and switched live (the G1 remote's Select button,
or the keys `n` and `1` to `9`; nine at most).

| Flag | Meaning |
|---|---|
| `--checkpoint PATH` | The trained policy. |
| `--policy NAME`, `--extension_policy NAME` | A published policy to load as an extra slot (repeatable). Makes `--task` and `--checkpoint` optional. |
| `--extra_checkpoint PATH` | Another checkpoint to preload as a slot (repeatable). |
| `--base_policy {auto,holosoma,none}` | A known-good policy loaded as slot 1 and active on boot. `auto` (default) loads the Holosoma walking policy on a G1 29-DOF and nothing elsewhere. |
| `--policy_checkpoint NAME=PATH` | Point a published policy at a local checkpoint (repeatable). |
| `--robot ROBOT` | The robot for a task-less `--policy` deploy. Default `G1Dofs29`. |
| `--entity_name NAME` | The scene entity to deploy. Default `robot`. For a composite rig (an arm with a hand), every part deploys through its own backend; name one part to deploy it alone. |
| `--motion_file FILE` | The reference motion for motion-tracking policies; the motion restarts from frame 0 at each switch. |
| `--on_motion_end {continue,stand,switch}` | What a motion-imitation deploy does when its clip ends. Default `continue`. |
| `--motion_end_slot N` | The slot `--on_motion_end switch` hands control to. Default 1. |
| `--stand_steps N` | Control cycles of the `--on_motion_end stand` ramp. Default 100. |

**Gains and commands.**

| Flag | Meaning |
|---|---|
| `--deploy_kp_kd_scale S` | Scale on the default Kp/Kd. Ramp up from 0.2 to 1.0 when commissioning. |
| `--gain_ramp_steps N` | Control steps over which a slot switch ramps to the new slot's gains. Backend default 50; 0 jumps. |
| `--target_lpf A` | First-order low-pass on the commanded joint targets, `a` in (0, 1]; 1.0 is off. |
| `--init_steps N` | Control cycles of the ramp to the start pose before the policy takes over. Default 250; 0 disables it. |
| `--init_target {default,motion,motion_rsi}` | Where that ramp drives the joints: the default pose, or frame 0 of the tracked clip. `motion_rsi` also teleports the base and works only with `--mujoco`. |
| `--velocity_command {constant,keyboard,gamepad,task}` | What drives the task's velocity command. Default `constant` (holds `--base_command`). `task` keeps the training-time random command generator. |
| `--base_command VX VY WZ` | The constant velocity command. Default: stand still. |
| `--velocity_command_scale VX VY WZ` | A per-axis limit on the substituted velocity command. |
| `--gamepad_layout {ps4,xbox}` | Axis scaling for `--velocity_command gamepad`. Default `ps4`. |
| `--disable_joints NAME ...` | Joint names or patterns sent with kp = kd = 0, for bench debugging. Never on load-bearing joints of an unsupported robot. |
| `--ignore_action_terms TERM ...` | Action terms whose commands the wire format cannot carry and that you drive yourself; without it, deployment refuses to start. |
| `--part_option PART.OPTION=VALUE` | Composite rigs: set one option of one part's backend (repeatable), e.g. `--part_option robot.robot_ip=172.16.0.2`. |

**State estimation and diagnostics.**

| Flag | Meaning |
|---|---|
| `--estimate_lin_vel` | Estimate the base linear velocity onboard with a contact-aided Kalman filter. |
| `--feet_links LINK ...` | The foot links that filter uses. Default: the robot's declared feet. |
| `--debug_record` | Record commanded targets and measured state, written as an `.npz`. |
| `--debug_record_path PATH` | Where. Default `<log_dir>/deploy_record.npz`. |
| `--debug_record_flush_steps N` | Steps per streamed chunk of the recording. Backend default 100. |

**Backend-specific.**

| Flag | Meaning |
|---|---|
| `--publish_rate HZ` | LimX Oli, Unitree G1 and Go2: the rate at which the held command is re-sent between control ticks. Backend defaults: 1000 Hz (Oli), 500 Hz (G1, Go2). |
| `--power_protect_factor N` | Unitree Go1 only: the SDK's power-protection factor, 1 to 10 (10%–100% of rated power). Backend default 10. |

**MuJoCo sim-to-sim** (with `--mujoco`).

| Flag | Meaning |
|---|---|
| `--mujoco` | Deploy against a MuJoCo model of the robot instead of hardware. Works for any task or robot. Add `--vis` for MuJoCo's viewer. |
| `--menagerie` | Use the robot's MuJoCo Menagerie model instead of its own training asset. Errors if the robot has none. |
| `--async` | Run the physics on its own thread in wall-clock time, as a robot's controller does, instead of in lockstep. |
| `--mujoco_props [ENTITY ...]` | Composite rigs: the scene entities to simulate besides the rig. Default: every colliding one but the ground. |
| `--imu_noise ACC GYRO` | White noise on the simulated IMU (per-axis standard deviation). |
| `--imu_bias AX AY AZ GX GY GZ` | A constant bias on the simulated IMU. |
| `--imu_noise_seed SEED` | Seed for `--imu_noise`. |
| `--gantry` | Hang the robot from a simulated overhead hoist. Simulation only. |
| `--gantry_body LINK ...` | The bodies the ropes are tied to. Default: the robot's declared mount links, else its base. |
| `--gantry_attach X Y Z` | The attachment point in each rope's body frame, in metres. |
| `--gantry_height Z` | The world height of the anchor. Default: 1 m above the attachment point. |
| `--gantry_step M` | Metres the `+` / `-` keys move the anchor. Default 0.05. |
| `--gantry_slack M` | Slack in the rope. Default 0 (taut). |
| `--gantry_support S` | Scale on the rope force, 0 to 1. Default 1. |
| `--gantry_no_draw` | Do not draw the ropes and their force arrows in the viewer. |
| `--gantry_force_scale S` | Metres of force arrow per newton. Default 1e-3. |

`--vis` (`--show_viewer`) opens the Genesis viewer mirroring the hardware state, or MuJoCo's viewer with `--mujoco`.

### `eden smoke`

```bash
eden smoke --task <task> [flags]
```

Builds a task and steps it a few times with no action, under `logs/smoke/<task>/`. It runs every term's `build()`,
so a wrong name in the config fails here in seconds rather than in a training run. Prints
`✅ smoke OK: built '<task>' with N env(s) and ran M step(s).` on success.

| Flag | Meaning |
|---|---|
| `--num_envs N` | Environments to build. Default 2. |
| `--steps N` | Steps to run. Default 2. |
| `--eval_mode` | Build in evaluation mode (training and randomization off). |
| `--show_viewer` | Open the Genesis viewer. |
| `--profile` | Time each term's `compute()` instead, and write `term_cost.json`. |
| `--profile_warmup N` | Warm-up steps before measuring, with `--profile`. Default 10. |
| `--profile_steps N` | Measured steps, with `--profile`. Default 50. |

### `eden preprocess`

```bash
eden preprocess --task <task> [--num_batches N --batch_idx I]
```

Builds the dataset cache of a motion task ahead of training. With `--num_batches N` greater than 1 the clip list is
split into `N` contiguous batches and this process handles one, so you can run one process per GPU.

| Flag | Meaning |
|---|---|
| `--num_batches N` | Split the clip list into this many batches. |
| `--batch_idx I` | The batch this process owns, in `[0, N)`. Default `0`: pass it to every process, or they all build batch 0. |

### `eden datagen`

```bash
eden datagen --task <task> [flags]
```

Rolls a task's scripted demonstrator out in parallel environments and records the episodes to HDF5. Only tasks
with a scripted demonstrator support it. See [Scripted demonstrations](../guides/scripted-demos.md).

| Flag | Meaning |
|---|---|
| `--num_episodes N` | Episodes to record. Default 100. |
| `--num_envs N` | Parallel environments. Default 16. |
| `--output_dir DIR` | The dataset's directory. Default `data/demos`. |
| `--filename NAME` | The file name without extension. Default: the task name. |
| `--export {all,succeeded,separate}` | Keep every episode (tagged with its success), only the successful ones, or the failed ones in a separate `<filename>_failed` file. Default `all`. |
| `--seed SEED` | Seed of the environment and the demonstrator. Default 0. |
| `--p_mistake P` | Probability that a subgoal is a deliberate mistake. Default: the demonstrator's own. |
| `--episode_length_s S` | The episode time limit. Default: the demonstrator's own, else 90. |
| `--camera_only` | Locate objects through the task's cameras instead of the simulator's state (demonstrators that support it, e.g. `robosuite_pick_place`). |
| `--show_viewer` | Open the Genesis viewer. |

## Examples

These read the `examples/` directory of a clone; a PyPI install has none. See [Examples](../examples/index.md).

### `eden examples list`

| Flag | Meaning |
|---|---|
| `--tier {project,policy,recipe}` | Only this tier. |
| `--capability TAG` | Only examples with this capability: a full `namespace.leaf` tag (`sensor.lidar`) or a namespace (`sensor`). |
| `--hardware {cpu,gpu,gpu-fp64,real-robot}` | Only examples with this hardware requirement. |
| `--search QUERY` | A substring search over key, purpose and capabilities. |
| `--json` | Print JSON records instead of a table. |

### `eden examples smoke`

```bash
eden examples smoke recipes/physics/mpm_tutorial
```

Runs one example headless for a few steps, with `EDEN_SMOKE=1`. The key is positional, as `eden examples list`
prints it. It refuses (exit 2, with the reason) examples that need real hardware, a teleoperation device, a missing
optional dependency, or a GPU the machine lacks.

| Flag | Meaning |
|---|---|
| `--timeout SECONDS` | Per-example timeout. Default 900. |

## Data

### `eden lerobot convert`

```bash
eden lerobot convert --hdf5 teleoperation.hdf5 --output-dir lerobot_output --repo-id local/teleop --force
```

Converts an Eden HDF5 recording into a LeRobot v3 dataset. It reads `state/qpos`, `state/dofs_vel`, `state/pos`,
`state/quat`, `action` and `timestamp` from each `demo_*` group under `/data`, the layout the state and action
recorders write. See [Datasets and recording](../guides/datasets-and-recording.md).

| Flag | Meaning |
|---|---|
| `--hdf5 PATH` | The input HDF5 file. |
| `--output-dir DIR` | Where the LeRobot dataset is created. |
| `--repo-id ID` | The LeRobot dataset `repo_id`. |
| `--robot-type TYPE` | The robot type recorded in the LeRobot metadata. |
| `--task TEXT` | The task string stored for every frame. |
| `--image-writer-threads N` | Threads LeRobot uses to encode videos. |
| `--force` | Overwrite `--output-dir` if it exists (it must be empty or a LeRobot dataset). |

### `eden pose gvhmr`

```bash
eden pose gvhmr --src video.mp4 --out motion.npz
```

Recovers world-grounded SMPL-X motion from monocular video with GVHMR and writes AMASS-style `.npz` files. Needs the
`pose` extra and GVHMR's checkpoints (`gvhmr-download-checkpoints`). See [Pose estimation](../guides/pose-estimation.md).

| Flag | Meaning |
|---|---|
| `--src SRC` | A video, or a directory searched recursively. Required. |
| `--out OUT` | The output directory (the source tree is mirrored), or an `.npz` file for a single video. Required. |
| `--pattern GLOB` | The file pattern for a directory `--src`. Default `*.mp4`. |
| `--checkpoint-root DIR` | GVHMR's checkpoint directory. Default: `$GVHMR_CHECKPOINT_ROOT`, else `./inputs/checkpoints`. |
| `--ckpt FILE` | The GVHMR checkpoint. Default `<root>/gvhmr/gvhmr_siga24_release.ckpt`. |
| `--static-cam` | Assume a static camera (skips visual odometry; much faster). |
| `--use-dpvo` | Use DPVO instead of SimpleVO for camera motion. |
| `--f-mm MM` | Full-frame focal length in mm. Default: estimated. |
| `--output-root DIR` | Where GVHMR writes and reuses its per-video intermediates. |
| `--skip-existing` | Skip a video whose output already exists. |
| `--verbose` | Save GVHMR's intermediate overlays. |

### `eden retarget`

Each retargeter is a subcommand. See [Retargeting](../guides/retargeting.md) and
[Hand retargeting](../guides/dex-retargeting.md).

#### `eden retarget omniretarget`

Robot-only OMOMO to G1 retargeting with OmniRetarget, ignoring the object.

| Flag | Meaning |
|---|---|
| `--src DIR` | The OMOMO dataset directory (the `.pt` sequence files). Required. |
| `--out DIR` | The output directory; joint positions are written as `<task-name>.npy`. Required. |
| `--task-name NAME` | The sequence to retarget, with or without `.pt`. Default `sub3_largebox_003`. |
| `--num-frames N` | Truncate the sequence to `N` frames. |
| `--skip-existing` | Skip when the output exists. |

#### `eden retarget otretarget`

Joint robot and object retargeting from surface interactions with OTRetarget. Needs the `ot-retargeter` extra.

| Flag | Meaning |
|---|---|
| `--source {omomo,seed}` | The demonstration format. Required. |
| `--src PATH` | omomo: a sequence directory or a folder of them; seed: a `.bvh` file or a directory of them. Required. |
| `--out DIR` | The output directory; one `<rig>_<clip>.npz` per clip. Required. |
| `--body-model-path PATH` | omomo: the folder with `SMPLX_{NEUTRAL,MALE,FEMALE}.npz`; seed: `SOMA_neutral.npz`. Required. |
| `--objects-dir DIR` | omomo: the release's `objects/` folder. |
| `--rig {g1_wuji,oli_wuji,g1,oli}` | The target embodiment. Default `g1_wuji`. |
| `--num-frames N` | Truncate each clip to `N` frames. |
| `--cpu` | Run Genesis on the CPU. |
| `--skip-existing` | Skip a clip whose output exists. |
| `--shard K/N` | Retarget only every `N`-th clip of the sorted list, starting at the `K`-th. Default `0/1`. |
| `--clip-list FILE` | A text file of clip names, one per line: retarget only those. |
| `--supports` | omomo: add the untracked tables the object rests on as fixed plates. |
| `--keep-going` | On a clip that raises, write `<rig>_<clip>.error.txt` and continue; exit non-zero at the end. |

#### `eden retarget soma`

Batch retargeting of AMASS-style SMPL, SMPL-H or SMPL-X motion to a robot with SomaRetargeter. Needs the
`soma-retargeter` extra (uv in a clone only).

| Flag | Meaning |
|---|---|
| `--src PATH` | An `.npz` clip, or a directory searched recursively. Required. |
| `--out PATH` | The output directory (source tree mirrored), or a file for a single clip. Required. |
| `--body-model {smpl,smplh,smplx}` | The source body model. Default `smplx`. |
| `--body-model-path PATH` | The body-model file (e.g. `SMPLX_NEUTRAL.npz`) or a per-gender directory. Required. |
| `--robot {unitree_g1,limx_oli}` | The target robot. Default `unitree_g1`. |
| `--gender {neutral,male,female}` | The default gender; a clip's own `gender` field overrides it. Default `neutral`. |
| `--target-fps FPS` | The frame rate every clip is resampled to. Default 30. |
| `--source-fps FPS` | Override the source frame rate when the clip's metadata is missing or wrong. |
| `--num-frames N` | Truncate each clip to `N` frames. |
| `--snap-to-ground` | Shift each clip vertically so its lowest collision point touches the ground. |
| `--skip-existing` | Skip a clip whose output exists. |
| `--cpu` | Use the CPU backend. |

#### `eden retarget dex`

Offline position retargeting of MANO keypoints to a dexterous hand with DexRetargeter. Needs the `dex-retargeter`
extra.

| Flag | Meaning |
|---|---|
| `--src PATH` | An `.npz` with keypoints `(T, 21, 3)` and `fps`, or a directory of them. Required. |
| `--out PATH` | An output `.pkl` file (single input) or directory. Required. |
| `--preset {xhand1_right_position}` | The hand preset. Default `xhand1_right_position`. |
| `--skip-existing` | Skip a clip whose output exists. |

#### `eden retarget flashdex import`

Imports a released hand-object dataset (MANO hands and object poses) into a FlashDexRetarget dataset:
`<dataset>/human_demo/<clip>.npz` and `<dataset>/objects/<id>/visual.obj`. Needs the `flashdexretarget` extra.

| Flag | Meaning |
|---|---|
| `--source {dexycb,hot3d,oakink2,taco}` | The release, read by its `eden.datasets` loader. |
| `--raw DIR` | The downloaded release's root directory. |
| `--dataset DIR` | The dataset directory to write. |
| `--mano-dir DIR` | The directory with `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`. |
| `--sides {both,right,left}` | The hands to import. Default `both`. |
| `--textures` | Export the objects with their materials and textures, for rendering. |
| `--config FILE` | A `.yaml` or `.json` file of option values (`eden.options.extensions.flashdexretarget`). |
| `--override YAML` | An inline YAML mapping merged over the options last; repeatable. |

#### `eden retarget flashdex kinematic`

Kinematic retargeting of an imported dataset onto robot hands: support disks and reject rules
(`<robot>/report.csv`), convex object parts, MuJoCo physics IK (`<robot>/retargeted/`), `<robot>/motion.pt` and
object SDF grids.

| Flag | Meaning |
|---|---|
| `--dataset DIR` | The dataset directory from the import. |
| `--robot {xhand,sharpa}` | The target robot. Default `xhand`. |
| `--workers N` | Processes. Default: half the CPU cores. |
| `--config FILE` | A `.yaml` or `.json` file of option values. |
| `--override YAML` | An inline YAML mapping merged over the options last; repeatable. Steps can be skipped, e.g. `--override "steps: {retarget: false, pack: false}"`. |

## Teleoperation and devices

See [Teleoperation](../guides/teleoperation.md) for the full pipelines.

### `eden gello calibrate`

Computes per-joint offsets and a gripper range for a GELLO leader arm held at a known pose, and prints a
ready-to-paste `GelloLeaderConfig(...)`. Needs the `gello` extra.

```bash
eden gello calibrate --port /dev/ttyUSB0 --start-dofs 0 0 0 0 0 0 --signs 1 1 -1 1 1 1 --gripper \
    --output configs/gello_ur5e.yaml
```

| Flag | Meaning |
|---|---|
| `--port PORT` | The Dynamixel serial port. |
| `--start-dofs RAD ...` | The arm's known joint positions (rad) at the calibration pose. |
| `--signs S ...` | The sign (`-1` or `1`) of each arm joint. |
| `--joint-ids ID ...` | Dynamixel IDs of the arm joints. Default `1..N`. |
| `--gripper` | A gripper is attached. |
| `--gripper-id ID` | The gripper's Dynamixel ID. Default `N+1`. |
| `--baudrate`, `--warmup-reads`, `--max-pi-multiple`, `--gripper-open-delta-deg`, `--gripper-close-delta-deg` | Low-level calibration settings. |
| `-o`, `--output PATH` | Save the `GelloLeaderConfig` (`.json`, `.yaml` or `.yml`). |

### `eden gello leader`

Reads calibrated joint positions from the leader arm and sends them to a `GelloZMQServer` over ZMQ.

| Flag | Meaning |
|---|---|
| `-c`, `--config PATH` | The `GelloLeaderConfig` file from `eden gello calibrate`. |
| `--port PORT` | The Dynamixel serial port. |
| `--zmq-host HOST` | The ZMQ host. |
| `--zmq-port PORT` | The ZMQ port. |

### `eden metaquest bridge`

Runs the televuer WebXR server that a Meta Quest connects to over Wi-Fi, and bridges its poses and the stereo view
to Eden over ZMQ. On start it prints the URL to open in the headset's browser. Needs the `metaquest` extra and its
git group.

| Flag | Meaning |
|---|---|
| `--width PX` | The full stereo frame width (both eyes). Default 1280. |
| `--height PX` | The frame height. Default 480. |
| `--fps FPS` | The headset display rate. Default 30. |
| `--display-mode {immersive,ego,pass-through}` | Default `immersive`. |
| `--tracking {hand,controller}` | Track hand joints (default) or motion controllers. |

!!! note
    `eden metaquest bridge --help` currently prints Vuer's own parameters instead of these flags. The table above is
    read from `eden/utils/devices/metaquest/xr_bridge.py`.

### `eden pico bridge`

Publishes PICO headset and controller tracking over ZMQ and streams Eden's rendered view back to the headset. Needs
the `pico` extra and the `xrobotoolkit_sdk` module built from source.

| Flag | Meaning |
|---|---|
| `--fps FPS` | The pose publish rate (frames are forwarded as they arrive). |
| `--control-address HOST:PORT` | Where to listen for the app's `OPEN_CAMERA` / `CLOSE_CAMERA` commands. |
| `--pose-port PORT` | The ZMQ PUB port for the tracking state. |
| `--frame-port PORT` | The ZMQ PULL port for Eden's rendered frames. |
| `--no-video` | Publish tracking only. |
| `--no-derived-body` | Publish only the lossless `body_raw`, without the derived per-joint body and trackers. |
| `--tracker-role SERIAL=ROLE` | Label a motion tracker by serial, e.g. `--tracker-role PA1234=waist` (repeatable). |

### `eden teleop sonic robot`

The robot side of SONIC whole-body teleoperation: a G1 body, the SONIC policy and its planner, listening for the
operator's commands. The body is the Genesis simulator by default. Needs the `sonic` extra and an `onnx` extra.

| Flag | Meaning |
|---|---|
| `--mujoco` / `--real` | Drive the body through the `eden deploy` MuJoCo backend, or the real Unitree G1 over DDS. |
| `-v`, `--vis` | Show the viewer (MuJoCo's own under `--mujoco`). |
| `-c`, `--cpu` | Use the CPU backend. |
| `-s`, `--steps N` | Stop after `N` control ticks. |
| `--operator-host HOST` | The host running the operator station. |
| `--cmd-port PORT`, `--debug-port PORT` | The command and debug ports. |
| `--encoder-path`, `--decoder-path`, `--planner-path` | Local overrides for the SONIC ONNX models. |
| `--max-close-ratio R` | The Dex3 close limit in [0.2, 1.0] (Genesis body only). |
| `--no-band` | Start with the elastic band (the gantry under `--mujoco`) released. |
| `--drop-band-after SECONDS` | Release the band this many seconds in. |
| `--auto-reset` | Reset to the launch state when the base falls with the band off. |
| `--deploy_kp_kd_scale`, `--target_lpf`, `--disable_joints`, `--debug_record`, `--debug_record_path` | As for [`eden deploy`](#eden-deploy), with `--mujoco` or `--real`. |
| `--joint-vel-limit RAD_S` | `--real`: stop when any body joint exceeds this speed (0 disables). |

### `eden teleop sonic operator`

The operator station: PICO body tracking, calibration, the mode state machine, and streaming to the robot process.

| Flag | Meaning |
|---|---|
| `--robot-host HOST` | The host running `eden teleop sonic robot`. |
| `--port PORT`, `--feedback-port PORT` | The command publish port and the robot's debug port. |
| `--target-fps FPS` | The pose stream rate. |
| `--num-frames N` | Frames per pose message. |
| `--record-dir DIR` | Record each sent pose message as `.npz` here. |
| `--no-service-start` | Do not spawn the XRoboToolkit PC service. |
| `--bridge-host HOST` | Connect to an `eden pico bridge` on this host instead of spawning one. |
| `--bridge-port PORT`, `--bridge-fps FPS` | The bridge's pose port and publish rate. |
| `--no-bridge-start` | Connect to a bridge already running locally. |
| `--sdk` | Read the XRoboToolkit SDK in-process instead of through the bridge. |
| `--vis-vr3pt` | Visualize the calibrated VR 3-point pose (needs the `sonic-viz` extra). |
| `--vis-smpl` | Also render the SMPL body (needs `--vis-vr3pt`). |
| `--waist-tracking`, `--no-g1` | Visualization only: the waist follows the neck; hide the G1 mesh. |

### `eden wuji scan`

Lists every Wuji Glove on the bus. Needs the `wuji` extra (Linux only). Takes no flags.

### `eden wuji monitor`

Shows a Wuji Glove's live hand pose and tactile sheet in the terminal, the first check after plugging one in.

| Flag | Meaning |
|---|---|
| `--sn SN` | The glove's serial number. Default: the only one found. |
| `--side {left,right}` | Pick between two connected gloves. |
| `--streams STREAM ...` | The streams to subscribe to, from `hand_skeleton`, `hand_joint_angles`, `tip_poses`, `emf_poses`, `tactile`, `tactile_zones`, `tactile_binary`, `tactile_residual`, `tactile_point_cloud`. Default `hand_skeleton hand_joint_angles tactile`. |
| `--rate-hz HZ` | Lower the device's push rate. Default 120 Hz. |
| `--seconds S` | Run for this long; 0 runs until Ctrl-C. |
| `--exclusive` | Take the glove for this process alone. |

## Viewing and utilities

### `eden view`

```bash
eden view Yam                     # an entity options class by name
eden view scan.ply                # a Gaussian-splat .ply
```

Opens an entity, by the name of its options class or a composite rig's name, or a Gaussian-splat `.ply` file, in a
viewer.

| Flag | Meaning |
|---|---|
| `-c`, `--collision` | Show collision geometry (entity mode). |
| `-m`, `--multiplier M` | The splat's radiance multiplier (splat mode). |
| `--env-brightness B` | A constant-color environment map's brightness in [0, 1] (splat mode). |

### `eden arrange`

```bash
eden arrange FrankaEmikaPanda YCB003CrackerBox -o scene_snippet.py
```

Spawns entities (by options class name) with ImGui sliders for each one's base pose and joints, and exports a
`SceneOptions` snippet with the arrangement. Needs the `vis` extra.

| Flag | Meaning |
|---|---|
| `-c`, `--collision` | Show collision geometry. |
| `-o`, `--output PATH` | Also write the snippet to this file. |
| `--spacing M` | Initial x-spacing between entities. Default 1.0 m. |

### `eden snap`

```bash
eden snap Yam -o yam.png
```

Renders one image of an entity.

| Flag | Meaning |
|---|---|
| `-o`, `--output PATH` | The output image. Default `snapshot.png`. |
| `-c`, `--collision` | Show collision geometry. |

### `eden gif`

```bash
eden gif video.mp4                # writes video.gif beside it
```

Converts an `.mp4` to a `.gif` at 10 fps and 640 px wide. Calls `ffmpeg`, which must be on your `PATH`.

### `eden clean`

Removes empty log directories, and run directories that hold only a `config.json` (runs that stopped before saving
anything else). Takes no flags.

!!! warning
    `eden clean` currently looks for a `logs/` directory inside the installed `eden` package
    (`eden/utils/logs`), not the `logs/` directory of your working directory, so it usually finds nothing.

## See also

- [Quick start](../getting-started/quickstart.md)
- [Environment variables](environment-variables.md)
- [Training runs](../guides/training-runs.md)
- [Deployment](../guides/deployment.md)
