# Examples

The `examples/` directory of the repository holds runnable scripts for most of Eden's features, from complete
trainable projects to one-concept recipes. This page explains how they are organized, how to find and run one,
and lists them all. The scripts are in the repository only, so you need a [clone](../getting-started/installation.md#install-from-a-clone);
a PyPI install does not include them.

## Three tiers

| Tier | Where | What it is |
|---|---|---|
| **Projects** | `examples/projects/<area>/` | End-to-end task templates that build an environment, train and evaluate. Fork one to start a real task. |
| **Policies** | `examples/policies/<name>/` | Run a published, pretrained policy in simulation; most can also run it on hardware. |
| **Recipes** | `examples/recipes/<capability>/` | Short scripts that each show one capability: a sensor, a solver, a teleop device, a term. |

A registered task trains with `eden train --task <name>`, so a project script only adds what `eden train` and
`eden inference` do not: an interactive viewer, keyboard control, recordings, data preparation. The published
policies also run without a script, through `eden inference --policy <name>` and `eden deploy --policy <name>`
([Policy zoo](../guides/policy-zoo.md)).

## Finding an example

Every runnable example carries a small metadata block in its module docstring: its tier, its capabilities, the
hardware it needs, any optional dependencies, and the command that runs it. `eden examples list` reads those blocks
without importing the scripts:

```bash
eden examples list                              # every example
eden examples list --tier recipe                # one tier
eden examples list --capability sensor          # a capability namespace...
eden examples list --capability sensor.lidar    # ...or one capability
eden examples list --hardware cpu               # what runs without a GPU
eden examples list --search splat               # substring search over key, purpose and capabilities
eden examples list --json                       # machine-readable records
```

The hardware values are `cpu`, `gpu`, `gpu-fp64` (a GPU and double precision, `en.init(precision="64")`) and
`real-robot`. Capabilities are `namespace.leaf` tags in a fixed set of namespaces: `solver`, `sensor`, `robot`,
`teleop`, `dataset`, `pose`, `retarget`, `term`, `render`, `planning`, `task`, `scene`, `deploy`, `collision`,
`terrain` and `rl`.

## Running an example

Run an example with the command in its metadata (`run:`), which `eden examples list --json` prints, from the
repository root:

```bash
uv run python examples/recipes/sensors/sensors_lidar_go2.py -v
uv run python examples/projects/reaching/reacher.py train
```

The examples share one set of short flags, and each letter means the same thing in every script:

| Short | Long | Meaning |
|---|---|---|
| `-v` | `--vis` | Open the viewer |
| `-c` | `--cpu` | Run on the CPU (examples default to the GPU) |
| `-b` | `--num-envs` | Number of parallel environments |
| `-s` | `--steps` | Number of steps or iterations |
| `-t` | `--seconds` | Simulated seconds |
| `-e` | `--exp-name` | Experiment name |
| `-r` | `--record` | Record a video |
| `-o` | `--output-dir` / `--output` | Where the script writes its output |
| `-d` | `--debug-vis` | Draw debug visualizations |

A script whose whole point is its viewer takes `--no-vis` instead of `-v`. Each script's `--help`, docstring or
README gives its other flags.

### Smoke-running

`eden examples smoke` runs one example headless for a few steps, with `EDEN_SMOKE=1`, which shortens the example's
horizon:

```bash
eden examples smoke recipes/physics/mpm_tutorial
eden examples smoke recipes/physics/mpm_tutorial --timeout 300
```

It refuses, with exit code 2 and the reason, an example that needs real hardware, a teleoperation device, a missing
optional dependency (the reason includes how to install it), or a GPU the machine lacks, and one that is not
instrumented for smoke mode or whose canonical command needs arguments. To smoke-test a registered task rather than
an example, use `eden smoke --task <name>`.

### Optional dependencies

Some examples need a package that is not part of Eden's core install. The metadata names them, and
`eden examples list --json` shows them as `optional_deps`; the most common are `onnx` (most published policies), `nyx`, `viser`, `gello`,
`metaquest`, `pico`, `wuji`, `sb3`, `skrl`, `graspgen` and `graspgenx`, which are Eden extras
([Installation](../getting-started/installation.md#optional-extras)), and `libuipc`, `curobo`, `ompl`,
`spacemouse` and `metamotivo`, which are installed separately. `eden examples smoke` prints the install command for
a missing one.

## Projects

| Example | What it does | Run |
|---|---|---|
| `projects/reaching/reacher` | Train and evaluate the `reacher` task on a Yam arm with rsl_rl | `uv run python examples/projects/reaching/reacher.py train` |
| `projects/locomotion/move_forward` | Train and evaluate `move_forward`, with the robot, controller and runner picked on the command line | `uv run python examples/projects/locomotion/move_forward.py --task move_forward` |
| `projects/locomotion/velocity_tracking/main` | Velocity-tracking training and interactive play for humanoids (G1 29-DOF, LimX Oli) | `uv run python examples/projects/locomotion/velocity_tracking/main.py train` |
| `projects/locomotion/mujoco_playground/getup/main` | Train and evaluate a Go1 get-up policy | `uv run python examples/projects/locomotion/mujoco_playground/getup/main.py train` |
| `projects/locomotion/mujoco_playground/handstand/main` | Train and evaluate a Go1 handstand policy | `uv run python examples/projects/locomotion/mujoco_playground/handstand/main.py train` |
| `projects/locomotion/extreme_parkour/main` | Extreme-parkour playback on a Go2 (teacher, or the distilled depth student) | `uv run python examples/projects/locomotion/extreme_parkour/main.py play` |
| `projects/locomotion/husky_skate/main` | Play a trained HUSKY skateboarding policy (G1 23-DOF and a skateboard) | `uv run python examples/projects/locomotion/husky_skate/main.py play --steps 50` |
| `projects/manipulation/main` | Train and run inference for the manipulation tasks | `uv run python examples/projects/manipulation/main.py train --task manipulation_reach_droid` |
| `projects/manipulation/piano/main` | Train two Wuji Hand 2 hands to play the piano, then watch and hear them | `uv run python examples/projects/manipulation/piano/main.py train --song twinkle_twinkle` |
| `projects/manipulation/polaris/main` | Keyboard-teleoperate a DROID arm across the PolaRiS Gaussian-splat scenes | `uv run python examples/projects/manipulation/polaris/main.py --task polaris_block_stack_kitchen` |
| `projects/manipulation/simpler/main` | Keyboard-teleoperate a WidowX-250 across the SIMPLER Bridge tasks | `uv run python examples/projects/manipulation/simpler/main.py --vis` |
| `projects/mimic/beyondmimic/main` | BeyondMimic motion imitation | `uv run python examples/projects/mimic/beyondmimic/main.py train` |
| `projects/mimic/mimic_beyondmimic/main` | BeyondMimic on `MimicRLEnvBase` | `uv run python examples/projects/mimic/mimic_beyondmimic/main.py train` |
| `projects/mimic/general_mimic/main` | General-mimic inference, with a ghost overlay of the reference | `uv run python examples/projects/mimic/general_mimic/main.py inference --model /path/to/general_mimic.pt` |
| `projects/mimic/assistmimic/main` | AssistMimic: a caregiver humanoid helps a weakened one up (SMPL humanoids) | `uv run python examples/projects/mimic/assistmimic/main.py train` |
| `projects/mimic/contrack/render` | Render a trained ConTrack checkpoint to a video | see its docstring |
| `projects/sysid/xhand1/collect_data`, `identify`, `verify` | System identification of an XHand1: collect real trajectories, identify DOF parameters, verify them | see `examples/projects/sysid/xhand1/` |

The BeyondMimic, AssistMimic and TWIST2 directories also hold data-preparation scripts under `scripts/` (CSV or
PKL to Eden's `.npz` motion format, replaying a converted clip). Several projects have a README next to the script.

## Policies

Each script builds an environment from the policy's own specification and rolls it out in simulation. See the
[policy zoo](../guides/policy-zoo.md) for what each policy is and needs.

| Example | Policy | Needs |
|---|---|---|
| `policies/amo/main` | AMO whole-body G1 control, with live keyboard commands | |
| `policies/beyondmimic_moves/main` | Any of the BeyondMimic G1-Moves motion-tracking policies (`--clip`) | |
| `policies/gmt/main` | GMT general motion tracking on a G1 | |
| `policies/holosoma/main` | Holosoma velocity-command locomotion on a G1 | `onnx` |
| `policies/homie/main`, `keyboard_teleop` | HOMIE walk/stand tracker on a G1, with keyboard teleoperation (simulation or real G1) | |
| `policies/humanoid_gpt/main` | Humanoid-GPT motion tracker on a kinematic reference; `analyze_record` reads an `eden deploy --debug_record` file | `onnx` |
| `policies/husky/main` | HUSKY skateboarding on a G1 | `onnx`, GPU |
| `policies/metamotivo/main` | MetaMotivo (FB-CPR) on Eden's SMPL humanoid | `metamotivo` |
| `policies/omnireset/main` | UWLab's OmniReset peg-insertion expert, measuring its success rate | GPU |
| `policies/omnixtreme/main` | OmniXtreme motion tracking on a G1 | `onnx` |
| `policies/policy_switch/main` | Hot-swap between Holosoma locomotion and SONIC tracking on one G1 | `onnx` |
| `policies/protomotionv3/main` | ProtoMotions v3 motion tracker on a kinematic reference | |
| `policies/simtoolreal/main` | Sim-to-sim check of the SimToolReal policy port | |
| `policies/sonic/main` | GEAR-SONIC universal-token policy on a kinematic reference | `onnx` |
| `policies/twist2/main` | TWIST2 motion tracking on an Eden `.npz` motion (`scripts/pkl_to_npz` converts its data) | `onnx` |

## Recipes

Recipes are grouped by the capability they show. `eden examples list --tier recipe` prints each one's purpose.

| Directory | Recipes | Shows |
|---|---|---|
| `recipes/physics/` | `rigid_*`, `mpm_*`, `sph_*`, `pbd_*`, `fem_*`, `coupler_*`, `ipc_*`, `sap_*` | The Genesis solvers and their coupling, ported from the Genesis physics catalogue ([below](#physics-recipes)) |
| `recipes/mochi/` | `rigid_bodies`, `soft_bodies`, `rods`, `cloth*`, `*_arm`, `finray_*`, `contacts_rj45_plug`, `newton_cradle`, `loop_closure`, `bolt_nut_self_screw`, `teleop/meta_cloth_folding`, ... | The Mochi implicit solver: rigid, soft, cloth and rod bodies in one system, some path traced with Nyx. See `examples/recipes/mochi/README.md`. |
| `recipes/sensors/` | `sensors_lidar_go2`, `sensors_imu_contact_go1`, `camera_as_sensor_with_nyx`, `entity_surface_points`, `engine_addon_contact_audio`, `engine_addon_piano`, `engine_addon_spatial_audio` | LiDAR, depth, IMU, contact and camera sensors, surface points, and audio sensors installed as engine add-ons |
| `recipes/rendering/` | `minimal_nyx`, `nyx_light_types`, `nyx_materials`, `nyx_multi_camera_multi_env`, `nyx_object_picking`, `camera_green_screen`, `follow_entity`, `moving_camera`, `render_wrist`, `pickup`, `speed_test`, `viser_visualization`, `viser_recording` | The Nyx path tracer, green-screen batch rendering, moving and following cameras, Viser in the browser, render throughput |
| `recipes/teleop/` | `keyboard/`, `gui/`, `ps4/`, `spacemouse/`, `gello/`, `metaquest/`, `pico/`, `wuji/` | Teleoperation, one directory per input device |
| `recipes/deployment/` | `real_g1_states`, `real_oli_states`, `real_franka_states`, `real_oli_teleoperation_slider`, `real_franka_teleoperation_keyboard`, `real_xhand`, `real_wuji_hand2` | Reading, comparing and commanding real hardware |
| `recipes/retargeting/` | `dex_retargeter/`, `omniretarget/`, `otretarget/`, `soma_retargeter/` | Human and hand motion to robots |
| `recipes/datasets/` | `amass`, `amass_ground_first_frame`, `arctic`, `dex_ycb`, `hhi_assist`, `lafan1`, `lafan1_smooth_start`, `omomo`, `phuma`, `seed` | Loading and replaying motion datasets, and dataset modifiers |
| `recipes/motion_optimization/` | `tcrs/adapt_clip`, `tcrs/adapt_dataset`, `tcrs/render_on_terrain` | Adapting flat-ground motion to terrain with TCRS |
| `recipes/pose_estimation/` | `gvhmr/video_to_motion` | SMPL-X motion from monocular video |
| `recipes/planning/` | `path_planning_ompl`, `path_planning_curobo`, `grasp_sampling_graspgen`, `grasp_sampling_graspgenx`, `grasp_sampling_graspgenx_hands` | Motion planning and grasp sampling |
| `recipes/rl_libraries/` | `cleanrl_ppo`, `sb3_ppo`, `skrl_ppo` | Training a task with CleanRL, Stable-Baselines3 or skrl ([Other RL libraries](../guides/rl-libraries.md)) |
| `recipes/terms/` | `termination`, `reacher_taskmod`, `rl_step_profile` | A custom term, building a task through its mods, profiling the step |
| `recipes/scenes/` | `asset_placement`, `lbm_eval` | Placing objects on support surfaces, a benchmark station |
| `recipes/heterogeneous/` | `heterogeneous_sim`, `heterogeneous_articulated` | Different object variants in different parallel environments |
| `recipes/collision/` | `drop_test`, `visualize_mask`, `robotiq_2f85_grasp` | Collision masks |
| `recipes/terrain/` | `terrain_gen` | Procedural terrain generation |
| `recipes/viewer/` | `mesh_point_selector` | Picking points on a mesh in the viewer |

### Physics recipes

`examples/recipes/physics/` ports the examples of the Genesis physics catalogue (all but *Stable Fluid: smoke*) to
Eden's options. Each script builds an `EnvBase` from `EnvOptions` and `SceneOptions` and steps it. The material
on an entity picks its solver, and coupling with rigid bodies is enabled per entity. Run any of them with `-v` to
open the viewer:

```bash
uv run python examples/recipes/physics/mpm_tutorial.py -v
```

| Genesis example | Eden script | Solvers |
|---|---|---|
| Rigid: franka cube | `rigid_franka_cube.py` | Rigid |
| Rigid: collision tower | `rigid_collision_tower.py` | Rigid |
| Rigid: contype | `rigid_contype.py` | Rigid |
| FEM: hard and soft constraint | `fem_hard_soft_constraint.py` | FEM |
| MPM: tutorial | `mpm_tutorial.py` | MPM |
| MPM: sand wheel | `mpm_sand_wheel.py` | MPM + rigid |
| SPH: rigid | `sph_rigid.py` | SPH + rigid |
| SPH + MPM | `sph_mpm.py` | SPH + MPM |
| PBD: liquid | `pbd_liquid.py` | PBD |
| PBD: cloth | `pbd_cloth.py` | PBD |
| IPC: robot cloth teleop | `ipc_robot_cloth_teleop.py` | IPC + FEM + rigid |
| Coupler: cloth on rigid | `coupler_cloth_on_rigid.py` | PBD + rigid |
| Coupler: rigid + MPM | `coupler_rigid_mpm.py` | MPM + rigid |
| Coupler: cut dragon | `coupler_cut_dragon.py` | MPM + rigid |
| Coupler: water wheel | `coupler_water_wheel.py` | SPH or MPM + rigid |
| Coupler: flush cubes | `coupler_flush_cubes.py` | MPM |
| SAP: franka grasp rigid cube | `sap_franka_grasp_cube.py` | SAP + rigid |

Beyond the catalogue: `ipc_objects_falling.py`, `ipc_robot_grasp_cube.py`, and `coupler_cut_dragon_teleop.py`, a
keyboard-teleoperated version of the cut dragon.

Most of these need the GPU backend; the metadata's `hardware` tag says which. The SAP grasp also needs double
precision (`en.init(precision="64")`), and the IPC scripts need the libuipc bindings. Genesis' particle emitters
have no Eden wrapper yet, so the sand, water-wheel and flush-cubes scripts call `env.scene.add_emitter(...)`
directly. `examples/recipes/physics/README.md` has per-script notes.

## Writing an example

`examples/AGENTS.md` in the repository is the contract for adding one: the metadata block and its closed
vocabularies, the flag convention, how to make a script smoke-runnable, and the tests that check all three
(`tests/utils/test_examples_catalog.py`, `tests/utils/test_example_flags.py`). Two rules matter most:

- Construct a registered task with `TASK_REGISTRY.build(name)`, never `TASK_REGISTRY.get(name)()` or the bare
  config class, which skip the task's mods.
- Shorten the horizon in smoke mode: `horizon = 5 if en.is_smoke() else 1000`.

## See also

- [Quick start](../getting-started/quickstart.md)
- [Your first task](../getting-started/first-task.md)
- [Command reference](../reference/cli.md#examples)
